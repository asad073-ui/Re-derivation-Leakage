"""Write-back policies — the reviewer-proofing.

The leak must come from a **defensible default**, not from a policy we designed to
leak. `FrameworkDefaultWritePolicy` therefore mirrors what production agent-memory
frameworks actually do out of the box, and says which ones:

  Letta / MemGPT      persists assistant turns into archival memory; recall/archival
                      entries are new records with their own ids. Provenance back to a
                      retrieved passage is not part of the stored record.
                      https://github.com/letta-ai/letta

  mem0                `add()` on conversation messages extracts and stores facts as new
                      memory entries keyed by fresh ids. Derivation from a retrieved
                      memory is not tracked as an edge on the new entry.
                      https://github.com/mem0ai/mem0

  LangGraph           the checkpointer persists the whole message state per thread;
  checkpointer        assistant messages are stored verbatim with no provenance field
                      at all. https://github.com/langchain-ai/langgraph

The behaviour common to all three, and the one we implement: **the assistant's answer
is written as a new node, and it links only to what was actually retrieved.** When
retrieval returned nothing — because the blocklist suppressed it — the new node has no
parents. That is not our design choice; it is theirs. Which is the whole argument.

Do not add a "semantic parent" inference here. See `MemoryNode.parent_ids`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..memory.blocklist import Blocklist, NoBlocklist
from ..memory.node import MemoryNode, SourceKind
from ..memory.store import MemoryStore
from .base import AgentReply

__all__ = [
    "DisabledWritePolicy",
    "FrameworkDefaultWritePolicy",
    "SanitizedWritePolicy",
    "WriteDecision",
    "WritePolicy",
    "build_write_policy",
]


@dataclass
class WriteDecision:
    write: bool
    reason: str
    policy: str
    parent_ids: list[str] = field(default_factory=list)
    node: MemoryNode | None = None
    blocked_score: float = 0.0

    def __bool__(self) -> bool:
        return self.write


@runtime_checkable
class WritePolicy(Protocol):
    name: str
    mode: str

    def maybe_write(
        self,
        store: MemoryStore,
        reply: AgentReply,
        *,
        question: str,
        retrieved_ids: Sequence[str],
        turn: int,
        blocklist: Blocklist | None = None,
    ) -> WriteDecision: ...


class DisabledWritePolicy:
    """No write-back at all. C1 — SBU's claimed behaviour and the leakage floor."""

    name = "disabled"
    mode = "disabled"

    def maybe_write(
        self,
        store: MemoryStore,
        reply: AgentReply,
        *,
        question: str,
        retrieved_ids: Sequence[str],
        turn: int,
        blocklist: Blocklist | None = None,
    ) -> WriteDecision:
        return WriteDecision(False, "write-back disabled", self.name)


class FrameworkDefaultWritePolicy:
    """Mirrors the default assistant-turn write of Letta/mem0/LangGraph. C2 and C3.

    `parent_ids` is set to the ids actually present in the retrieval context — and to
    nothing else. An answer produced from parametric memory with an empty context
    yields a node with no derivation edges.
    """

    name = "framework_default"
    mode = "framework_default"

    def __init__(
        self,
        *,
        source_kinds: Sequence[SourceKind] = ("agent_answer",),
        write_on_abstention: bool = False,
        mirrors_framework: str = "letta/mem0/langgraph default assistant-turn write",
    ) -> None:
        self.source_kinds = tuple(source_kinds)
        self.write_on_abstention = write_on_abstention
        self.mirrors_framework = mirrors_framework

    def maybe_write(
        self,
        store: MemoryStore,
        reply: AgentReply,
        *,
        question: str,
        retrieved_ids: Sequence[str],
        turn: int,
        blocklist: Blocklist | None = None,
    ) -> WriteDecision:
        if reply.abstained and not self.write_on_abstention:
            return WriteDecision(False, "answer was an abstention", self.name)
        if not reply.text.strip():
            return WriteDecision(False, "empty answer", self.name)

        # ONLY nodes that were actually in the retrieval context become parents.
        parents = [nid for nid in reply.context_node_ids if nid in store]

        node = store.add(
            reply.text,
            source_agent=reply.agent_id,
            source_kind="agent_answer",
            parent_ids=parents,
            turn=turn,
            meta={"question": question, "policy": self.name, "mirrors": self.mirrors_framework},
        )
        reason = (
            f"wrote assistant turn with {len(parents)} parent(s)"
            if parents
            else "wrote assistant turn with NO parents (parametric answer, empty context)"
        )
        return WriteDecision(True, reason, self.name, parents, node)


class SanitizedWritePolicy(FrameworkDefaultWritePolicy):
    """Phase 2: consult a content-level blocklist at WRITE time, not just retrieval time.

    This is the intervention the measurement motivates. An ID blocklist cannot help
    here — the node being written has a fresh id by construction — so this policy is
    only meaningful with a `SemanticBlocklist`. Passing an `IDBlocklist` is allowed but
    logged, because it degrades silently to `framework_default` and a silent
    degradation would look like a null result.
    """

    name = "sanitized"
    mode = "sanitized"

    def __init__(self, *, threshold: float = 0.75, **kwargs) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold

    def maybe_write(
        self,
        store: MemoryStore,
        reply: AgentReply,
        *,
        question: str,
        retrieved_ids: Sequence[str],
        turn: int,
        blocklist: Blocklist | None = None,
    ) -> WriteDecision:
        bl = blocklist if blocklist is not None else NoBlocklist()
        decision = bl.is_blocked_text(reply.text)
        if decision.blocked:
            return WriteDecision(
                False,
                f"write suppressed by content blocklist: {decision.reason}",
                self.name,
                blocked_score=decision.score,
            )
        out = super().maybe_write(
            store,
            reply,
            question=question,
            retrieved_ids=retrieved_ids,
            turn=turn,
            blocklist=bl,
        )
        out.policy = self.name
        out.blocked_score = decision.score
        return out


def build_write_policy(
    mode: str,
    *,
    source_kinds: Sequence[SourceKind] = ("agent_answer",),
    write_on_abstention: bool = False,
    mirrors_framework: str | None = None,
    threshold: float = 0.75,
) -> WritePolicy:
    if mode == "disabled":
        return DisabledWritePolicy()
    kwargs: dict[str, Any] = {
        "source_kinds": source_kinds,
        "write_on_abstention": write_on_abstention,
    }
    if mirrors_framework:
        kwargs["mirrors_framework"] = mirrors_framework
    if mode == "framework_default":
        return FrameworkDefaultWritePolicy(**kwargs)
    if mode == "sanitized":
        return SanitizedWritePolicy(threshold=threshold, **kwargs)
    raise ValueError(
        f"unknown write policy '{mode}' (expected disabled|framework_default|sanitized)"
    )
