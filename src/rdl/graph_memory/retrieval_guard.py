"""Retrieval-time enforcement.

Thin by design: it turns store candidates into a defence call and the defence's answer
back into node texts. The policy lives in ``rdl.defenses``; keeping any of it here would
give the memory layer a second, divergent copy of the rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..defenses.base import Defense, RetrievalContext
from .staged_store import StagedMemory

__all__ = ["GuardedRetrieval", "RetrievalGuard"]


@dataclass(frozen=True)
class GuardedRetrieval:
    node_ids: tuple[str, ...]
    texts: tuple[str, ...]
    withheld_node_ids: tuple[str, ...]
    rescan_withheld_node_ids: tuple[str, ...]
    reason: str

    @property
    def n_withheld(self) -> int:
        return len(self.withheld_node_ids) + len(self.rescan_withheld_node_ids)


class RetrievalGuard:
    def __init__(self, memory: StagedMemory, defense: Defense, *, retrieval_k: int = 5) -> None:
        self.memory = memory
        self.defense = defense
        self.retrieval_k = retrieval_k

    def retrieve(self, query: str, *, node_id: str = "", depth: int = 0) -> GuardedRetrieval:
        candidates = self.memory.candidates(query, self.retrieval_k)
        verdict = self.defense.on_retrieval(
            RetrievalContext(node_id=node_id, depth=depth, query=query, candidates=candidates)
        )
        allowed = tuple(verdict.allowed_node_ids)
        return GuardedRetrieval(
            node_ids=allowed,
            texts=self.memory.texts_for(allowed),
            withheld_node_ids=tuple(verdict.withheld_node_ids),
            rescan_withheld_node_ids=tuple(verdict.rescan_withheld_node_ids),
            reason=verdict.reason,
        )
