"""Which stored nodes are associated with which forgotten concepts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

__all__ = ["ScopeIndex"]


class ScopeIndex:
    """A two-way map between store node ids and Forget IDs.

    Deliberately not the authority on whether a node may be returned — that is the
    retrieval guard's job, and it consults semantics as well as this index. An index
    that were the sole authority would make tag removal a complete bypass.
    """

    def __init__(self) -> None:
        self._by_node: dict[str, set[str]] = {}
        self._by_concept: dict[str, set[str]] = {}

    def tag(self, node_id: str, forget_ids: Iterable[str]) -> tuple[str, ...]:
        ids = {str(x) for x in forget_ids}
        if not ids:
            self._by_node.setdefault(node_id, set())
            return ()
        self._by_node.setdefault(node_id, set()).update(ids)
        for fid in ids:
            self._by_concept.setdefault(fid, set()).add(node_id)
        return tuple(sorted(self._by_node[node_id]))

    def forget_ids(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._by_node.get(node_id, ())))

    def nodes_for(self, forget_id: str) -> tuple[str, ...]:
        return tuple(sorted(self._by_concept.get(forget_id, ())))

    def tagged_nodes(self) -> tuple[str, ...]:
        return tuple(sorted(nid for nid, ids in self._by_node.items() if ids))

    def untagged_nodes(self, candidates: Sequence[str]) -> tuple[str, ...]:
        return tuple(nid for nid in candidates if not self._by_node.get(nid))

    def copy(self) -> ScopeIndex:
        other = ScopeIndex()
        other._by_node = {k: set(v) for k, v in self._by_node.items()}
        other._by_concept = {k: set(v) for k, v in self._by_concept.items()}
        return other

    def to_dict(self) -> dict:
        return {nid: sorted(ids) for nid, ids in sorted(self._by_node.items()) if ids}

    def __len__(self) -> int:
        return len(self.tagged_nodes())

    def __repr__(self) -> str:  # pragma: no cover
        return f"ScopeIndex({len(self)} tagged nodes, {len(self._by_concept)} concepts)"
