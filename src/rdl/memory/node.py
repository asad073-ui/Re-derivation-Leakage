"""The memory node.

Read the `parent_ids` docstring before changing anything here. It is the whole argument.
"""

from __future__ import annotations

import uuid
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

__all__ = ["MemoryNode", "SourceKind", "new_node_id"]

SourceKind = Literal["ingest", "agent_answer", "summary", "tool"]


def new_node_id() -> str:
    return uuid.uuid4().hex


class MemoryNode(BaseModel):
    """One entry in the shared memory store.

    `parent_ids` — THE LOAD-BEARING FIELD
    -------------------------------------
    Derivation edges. Set **only** when this node was derived from a node that was
    actually present in the retrieval context at write time.

    A node written from an agent's *parametric* answer — the agent recalled it from
    weights, with no retrieved node in context — legitimately has ``parent_ids == []``.

    Do NOT "helpfully" synthesise an edge back to a semantically related node. That
    would fabricate exactly the provenance SBU is missing, and it would make the
    laundering result an artifact of our own instrumentation rather than a property of
    the mechanism under test. A laundered node has no parents *because there is nothing
    to point at* — that is the finding, not a bug.

    `refcount` / `outdated`
    -----------------------
    SBU's pruning model. `refcount` is **the number of live parents supporting this
    node**, with a floor of 1 for a node that is independently grounded (ingested from
    the corpus, or produced from a model's parametric memory with no retrieved context).
    It is NOT a count of citations from children — conflating the two makes deletion
    propagate in the wrong direction.

    When a node is deleted, every node in its dependency closure has its refcount
    decremented exactly once — once per *unique* node, so a diamond DAG does not
    double-decrement — and is then marked outdated if either

      - its refcount reached zero, or
      - every one of its parents is itself deleted or outdated.

    The second rule is what handles the diamond: the join node keeps a positive refcount
    (it was decremented only once) but both of its parents are gone, so its provenance
    is entirely dead and it must be marked.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    node_id: str = Field(default_factory=new_node_id)
    content: str
    embedding: np.ndarray | None = None
    source_agent: str = "system"
    source_kind: SourceKind = "ingest"
    parent_ids: list[str] = Field(default_factory=list)
    turn: int = 0
    refcount: int = 1

    # Pruning state. Not in the original SBU sketch as fields, but the invariant-2
    # check needs somewhere to observe "was this marked outdated".
    outdated: bool = False
    deleted: bool = False

    # Free-form provenance for the witness. Never used in metric computation.
    meta: dict[str, Any] = Field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        flags = "".join(("D" if self.deleted else "", "O" if self.outdated else ""))
        return (
            f"MemoryNode({self.node_id[:8]} t={self.turn} rc={self.refcount} {flags} "
            f"src={self.source_agent}/{self.source_kind} "
            f"parents={[p[:8] for p in self.parent_ids]} {self.content[:40]!r})"
        )

    @property
    def is_derived(self) -> bool:
        """True when this node cites at least one retrieved parent."""
        return bool(self.parent_ids)

    @property
    def is_parametric(self) -> bool:
        """True when this node came from model weights with no retrieved parent.

        This is the population the laundering metric is computed over.
        """
        return not self.parent_ids and self.source_kind == "agent_answer"

    @property
    def returnable(self) -> bool:
        """Whether retrieval may hand this node back, ignoring the blocklist.

        Deleted nodes stay INDEXED as tombstones and are suppressed here. They are not
        physically dropped from the index, because if they were, the blocklist would be
        redundant and invariant 1 would be vacuously true — we would be testing
        `del store[id]`, not SBU's mechanism. Keeping the tombstone is what lets the
        adversarial probe in `invariants.check_invariant_1` actually exercise the
        retrieval path, and what lets a `Retrieval` event record what was suppressed.

        Outdated nodes ARE returnable. SBU marks derived content outdated rather than
        removing it, and whether outdated content is still reachable is precisely what
        invariant 1 has to be checked against rather than assumed away.
        """
        return not self.deleted

    def to_record(self, *, with_embedding: bool = False) -> dict:
        """JSON-safe dict. Embeddings are omitted by default — they bloat witnesses."""
        d = self.model_dump(mode="json", exclude={"embedding"})
        if with_embedding and self.embedding is not None:
            d["embedding"] = [float(x) for x in self.embedding]
        return d
