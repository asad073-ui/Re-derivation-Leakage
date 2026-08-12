"""Staged persistent memory.

Writes are staged during an episode and committed after it. That is a scientific choice,
not an implementation detail: if a node written by agent B at depth 1 were immediately
retrievable by agent D at depth 2, agent-to-agent edge flow and memory-mediated flow
would be mixed in one number and no arm could attribute its leakage to either. The
``online_memory`` ablation flips ``visibility`` to ``immediate`` and is reported
separately.

Everything here delegates to ``rdl.memory.MemoryStore``. The staged layer adds only the
scope tag and the commit boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..memory.blocklist import Blocklist, NoBlocklist
from ..memory.node import MemoryNode
from ..memory.store import MemoryStore, StoreSnapshot
from .policy_node import PolicyRecord, node_forget_ids, tag_node
from .scope_index import ScopeIndex

__all__ = ["StagedMemory", "WriteCandidate"]


@dataclass
class WriteCandidate:
    """A write proposed by a node, before any guard has looked at it."""

    content: str
    source_node: str
    envelope_id: str
    parent_store_ids: tuple[str, ...] = ()
    forget_ids: tuple[str, ...] = ()
    score: float = 0.0
    detector_version: str = ""
    meta: dict = field(default_factory=dict)


class StagedMemory:
    """A ``MemoryStore`` with staged writes, scope tags and a commit boundary."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        blocklist: Blocklist | None = None,
        index: ScopeIndex | None = None,
        visibility: str = "after_episode",
    ) -> None:
        if visibility not in ("after_episode", "immediate"):
            raise ValueError(
                f"unknown memory visibility '{visibility}' (expected after_episode|immediate)"
            )
        self.store = store
        self.blocklist: Blocklist = blocklist if blocklist is not None else NoBlocklist()
        self.scopes = index if index is not None else ScopeIndex()
        self.visibility = visibility
        self._staged: list[WriteCandidate] = []
        self.committed_ids: list[str] = []
        self._baseline: StoreSnapshot | None = None
        # Every node already in the store gets its tag read back, so a restored snapshot
        # keeps its scopes even though the index is rebuilt.
        for node in store.all_nodes(include_deleted=True):
            ids = node_forget_ids(node)
            if ids:
                self.scopes.tag(node.node_id, ids)

    # ------------------------------------------------------------------ lifecycle --

    def mark_baseline(self) -> None:
        """Record the post-deletion state every arm must start from."""
        self._baseline = self.store.snapshot()

    def reset_to_baseline(self) -> None:
        if self._baseline is None:
            raise RuntimeError("no baseline snapshot; call mark_baseline() first")
        self.store.restore(self._baseline)
        self._staged.clear()
        self.committed_ids.clear()
        self.scopes = ScopeIndex()
        for node in self.store.all_nodes(include_deleted=True):
            ids = node_forget_ids(node)
            if ids:
                self.scopes.tag(node.node_id, ids)

    # ---------------------------------------------------------------------- write --

    def stage(self, candidate: WriteCandidate) -> str | None:
        """Queue a write. Under ``immediate`` visibility it is committed at once."""
        if self.visibility == "immediate":
            return self._commit_one(candidate)
        self._staged.append(candidate)
        return None

    def staged(self) -> tuple[WriteCandidate, ...]:
        return tuple(self._staged)

    def commit(self) -> list[str]:
        """Commit every staged write. Called once, after the episode."""
        written = [self._commit_one(c) for c in self._staged]
        self._staged.clear()
        return [nid for nid in written if nid]

    def _commit_one(self, candidate: WriteCandidate) -> str:
        parents = [pid for pid in candidate.parent_store_ids if pid in self.store]
        node = self.store.add(
            candidate.content,
            source_agent=candidate.source_node,
            source_kind="agent_answer",
            parent_ids=parents,
            meta=dict(candidate.meta),
        )
        # Inheritance from store parents as well as from the envelope: a write whose
        # explicit parent list is empty is not thereby clean, and a write derived from a
        # tagged node is tagged even when its own content scores below threshold.
        inherited = set(candidate.forget_ids)
        for parent in parents:
            inherited |= set(self.scopes.forget_ids(parent))
        ids = tuple(sorted(inherited))
        if ids:
            self.scopes.tag(node.node_id, ids)
        tag_node(
            node,
            PolicyRecord(
                forget_ids=ids,
                source_envelope_ids=(candidate.envelope_id,),
                source_nodes=(candidate.source_node,),
                detector_version=candidate.detector_version,
                score=candidate.score,
                release_status="blocked" if ids else "pass",
                reason="write-time scope inheritance",
            ),
        )
        self.committed_ids.append(node.node_id)
        return node.node_id

    # ------------------------------------------------------------------ retrieval --

    def candidates(self, query: str, k: int) -> tuple[tuple[str, str, tuple[str, ...]], ...]:
        """Top-k retrieval candidates as ``(node_id, content, tags)``.

        The store's own blocklist runs first — the SBU deletion is still in force — and
        the graph defence then decides over what survives it.
        """
        result = self.store.retrieve(query, k=k, blocklist=self.blocklist)
        return tuple(
            (node.node_id, node.content, self.scopes.forget_ids(node.node_id))
            for node in result.nodes
        )

    def texts_for(self, node_ids: Sequence[str]) -> tuple[str, ...]:
        out: list[str] = []
        for node_id in node_ids:
            node = self.store.get(node_id)
            if node is not None:
                out.append(node.content)
        return tuple(out)

    def nodes(self) -> list[MemoryNode]:
        return self.store.all_nodes()

    def stats(self) -> dict:
        return {
            "visibility": self.visibility,
            "n_staged": len(self._staged),
            "n_committed": len(self.committed_ids),
            "n_tagged": len(self.scopes),
            **self.store.stats(),
        }
