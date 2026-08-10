"""Blocklists.

`IDBlocklist` is SBU's mechanism: a set of node ids, enforced at retrieval time.
`SemanticBlocklist` is the Phase-2 method: content-level blocking, which is what an
ID blocklist structurally cannot do.

Both satisfy one `Blocklist` protocol so that C1/C2/C3 and the Phase-2 method differ by
*config*, not by code path. If they ever need different call sites, the comparison
stops being clean.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from .index import Embedder, HashingEmbedder
from .node import MemoryNode

__all__ = [
    "BlockDecision",
    "Blocklist",
    "IDBlocklist",
    "NoBlocklist",
    "SemanticBlocklist",
    "build_blocklist",
]


@dataclass(frozen=True)
class BlockDecision:
    blocked: bool
    score: float = 0.0
    reason: str = ""
    matched_reference: str | None = None

    def __bool__(self) -> bool:
        return self.blocked


@runtime_checkable
class Blocklist(Protocol):
    kind: str

    def blocks(self, node: MemoryNode) -> BlockDecision:
        """Decide whether this node may be returned by retrieval."""
        ...

    def is_blocked_text(self, text: str) -> BlockDecision:
        """Decide on raw content, with no node identity. Used by the sanitized writer."""
        ...

    def blocked_ids(self) -> set[str]:
        """Ids known to be blocked. May be empty for a purely content-based blocklist."""
        ...


class NoBlocklist:
    """Nothing is blocked. The C0 / no-defence baseline."""

    kind = "none"

    def blocks(self, node: MemoryNode) -> BlockDecision:
        return BlockDecision(False, 0.0, "no blocklist configured")

    def is_blocked_text(self, text: str) -> BlockDecision:
        return BlockDecision(False, 0.0, "no blocklist configured")

    def blocked_ids(self) -> set[str]:
        return set()

    def __len__(self) -> int:
        return 0


class IDBlocklist:
    """SBU's mechanism: a set of node ids.

    Note what this can and cannot do. It can guarantee that node `m1` is never returned
    by retrieval. It cannot say anything at all about a *different* node `m2` whose
    content happens to restate `m1`, because `m2` has a different id. That gap is the
    paper's subject; do not paper over it by adding content matching here.
    """

    kind = "id"

    def __init__(self, ids: Iterable[str] | None = None) -> None:
        self._ids: set[str] = set(ids or ())

    def add(self, node_id: str) -> None:
        self._ids.add(node_id)

    def add_all(self, ids: Iterable[str]) -> None:
        self._ids.update(ids)

    def discard(self, node_id: str) -> None:
        self._ids.discard(node_id)

    def contains(self, node_id: str) -> bool:
        return node_id in self._ids

    def blocks(self, node: MemoryNode) -> BlockDecision:
        hit = node.node_id in self._ids
        return BlockDecision(
            hit, 1.0 if hit else 0.0, "id in blocklist" if hit else "id not listed"
        )

    def is_blocked_text(self, text: str) -> BlockDecision:
        # Structurally cannot decide on content. Saying so explicitly, rather than
        # returning a silent False, keeps the limitation visible at every call site.
        return BlockDecision(False, 0.0, "IDBlocklist cannot evaluate raw content")

    def blocked_ids(self) -> set[str]:
        return set(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __repr__(self) -> str:  # pragma: no cover
        return f"IDBlocklist({len(self._ids)} ids)"


class SemanticBlocklist:
    """Phase-2 method: block on content similarity, not identity.

    Holds embeddings of the blocked content. `is_blocked_text` returns
    ``(blocked, score, reason)`` where score is max cosine similarity against the
    blocked set. The optional NLI head is a placeholder hook — when `nli_fn` is
    supplied it is consulted for candidates that clear a lower similarity bar, which is
    how a real deployment would trade recall for cost.
    """

    kind = "semantic"

    def __init__(
        self,
        blocked_texts: Iterable[str] | None = None,
        *,
        embedder: Embedder | None = None,
        threshold: float = 0.75,
        nli_fn=None,
        nli_recall_floor: float = 0.45,
        ids: Iterable[str] | None = None,
    ) -> None:
        self.embedder = embedder if embedder is not None else HashingEmbedder(dim=64)
        self.threshold = threshold
        self.nli_fn = nli_fn
        self.nli_recall_floor = nli_recall_floor
        self._texts: list[str] = []
        self._mat: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._ids: set[str] = set(ids or ())
        for t in blocked_texts or ():
            self.add_text(t)

    def add_text(self, text: str) -> None:
        self._texts.append(text)
        v = self.embedder.encode([text])
        self._mat = np.vstack([self._mat, v]) if self._mat.size else v

    def add_id(self, node_id: str) -> None:
        self._ids.add(node_id)

    def _max_similarity(self, text: str) -> tuple[float, int]:
        if self._mat.shape[0] == 0:
            return 0.0, -1
        q = self.embedder.encode([text])[0]
        scores = self._mat @ q
        i = int(np.argmax(scores))
        return float(scores[i]), i

    def is_blocked_text(self, text: str) -> BlockDecision:
        score, i = self._max_similarity(text)
        if score >= self.threshold:
            return BlockDecision(
                True, score, f"cosine {score:.3f} >= {self.threshold} vs #{i}", self._texts[i]
            )
        if self.nli_fn is not None and score >= self.nli_recall_floor:
            entailed = bool(self.nli_fn(self._texts[i], text))
            if entailed:
                return BlockDecision(
                    True, score, f"NLI entailment vs #{i} (cosine {score:.3f})", self._texts[i]
                )
        return BlockDecision(False, score, f"cosine {score:.3f} < {self.threshold}")

    def blocks(self, node: MemoryNode) -> BlockDecision:
        if node.node_id in self._ids:
            return BlockDecision(True, 1.0, "id in blocklist")
        return self.is_blocked_text(node.content)

    def blocked_ids(self) -> set[str]:
        return set(self._ids)

    def __len__(self) -> int:
        return len(self._texts) + len(self._ids)

    def __repr__(self) -> str:  # pragma: no cover
        return f"SemanticBlocklist({len(self._texts)} texts, {len(self._ids)} ids, thr={self.threshold})"


def build_blocklist(
    kind: str,
    *,
    ids: Iterable[str] | None = None,
    texts: Iterable[str] | None = None,
    embedder: Embedder | None = None,
    threshold: float = 0.75,
) -> Blocklist:
    if kind == "none":
        return NoBlocklist()
    if kind == "id":
        return IDBlocklist(ids)
    if kind == "semantic":
        return SemanticBlocklist(texts, embedder=embedder, threshold=threshold, ids=ids)
    raise ValueError(f"unknown blocklist kind '{kind}' (expected none|id|semantic)")
