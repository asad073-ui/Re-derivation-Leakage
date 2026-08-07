"""The shared memory store.

Dict-backed, with an exact vector index alongside. Deep-copyable and snapshot/restore
capable so that C1/C2/C3 can all start from a byte-identical state — otherwise the
between-condition comparison is confounded by store initialisation, which would be an
embarrassing way to lose a result.

Retrieval enforces the blocklist *at retrieval time* and reports what it suppressed.
The suppressed ids are recorded on the `Retrieval` event, not thrown away, because
"the blocklist worked and the item still surfaced elsewhere" is exactly the finding.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from .blocklist import Blocklist, NoBlocklist
from .derivation import DerivationDAG, PruneResult
from .index import Embedder, HashingEmbedder, VectorIndex, build_index
from .node import MemoryNode, SourceKind

__all__ = ["MemoryStore", "RetrievalResult", "StoreSnapshot"]


@dataclass
class RetrievalResult:
    """What a retrieval returned, and what it withheld."""

    nodes: list[MemoryNode] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    blocked_node_ids: list[str] = field(default_factory=list)
    query: str = ""

    @property
    def node_ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "returned_node_ids": self.node_ids,
            "scores": [round(s, 6) for s in self.scores],
            "blocked_node_ids": sorted(self.blocked_node_ids),
        }


@dataclass
class StoreSnapshot:
    nodes: dict[str, MemoryNode]
    dag: DerivationDAG
    turn: int


class MemoryStore:
    """Shared, cross-agent memory. One instance per episode-group."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        index: VectorIndex | None = None,
        index_backend: str = "numpy",
        embedding_dim: int = 64,
        dag: DerivationDAG | None = None,
        deletion_mode: str = "tombstone",
        refcount_semantics: str = "supporting_parents",
    ) -> None:
        # `is None`, never `or`: an EMPTY index, DAG, or blocklist is falsy — they all
        # define __len__ — so `x or default()` silently discards a caller's argument at
        # exactly the moment it is empty, which is construction time. That bug is
        # invisible until a run produces the wrong retrieval set.
        self.embedder: Embedder = (
            embedder if embedder is not None else HashingEmbedder(dim=embedding_dim)
        )
        self.index: VectorIndex = (
            index if index is not None else build_index(index_backend, dim=self.embedder.dim)
        )
        if self.index.dim != self.embedder.dim:
            raise ValueError(f"index dim {self.index.dim} != embedder dim {self.embedder.dim}")
        self.nodes: dict[str, MemoryNode] = {}
        self.dag: DerivationDAG = dag if dag is not None else DerivationDAG()
        self.turn: int = 0
        if deletion_mode not in ("tombstone", "hard"):
            raise ValueError(f"unknown deletion_mode '{deletion_mode}' (expected tombstone|hard)")
        if refcount_semantics not in ("supporting_parents", "dependent_children"):
            raise ValueError(
                f"unknown refcount_semantics '{refcount_semantics}' "
                "(expected supporting_parents|dependent_children)"
            )
        # See MemoryConfig.deletion_mode / .refcount_semantics and ADR-0014/0015.
        self.deletion_mode = deletion_mode
        self.refcount_semantics = refcount_semantics

    # ---------------------------------------------------------------------- CRUD --

    def add(
        self,
        content: str,
        *,
        source_agent: str = "system",
        source_kind: SourceKind = "ingest",
        parent_ids: Sequence[str] | None = None,
        turn: int | None = None,
        node_id: str | None = None,
        meta: dict | None = None,
    ) -> MemoryNode:
        """Write a node.

        `parent_ids` must be the ids that were ACTUALLY in the retrieval context. Never
        synthesise one — see `MemoryNode.parent_ids`.
        """
        parents = list(parent_ids or [])
        for p in parents:
            if p not in self.nodes:
                raise KeyError(f"parent node {p} not in store; cannot record a derivation edge")

        vec = self.embedder.encode([content])[0]
        kwargs = {
            "content": content,
            "embedding": vec,
            "source_agent": source_agent,
            "source_kind": source_kind,
            "parent_ids": parents,
            "turn": self.turn if turn is None else turn,
            # supporting_parents: live parents supporting this node, floored at 1 for a
            #   node that is independently grounded (ingested, or parametric).
            # dependent_children: nobody depends on a brand-new node yet, so it starts
            #   at 0 and its PARENTS each gain one dependent below.
            "refcount": (
                max(1, len(parents)) if self.refcount_semantics == "supporting_parents" else 0
            ),
            "meta": dict(meta or {}),
        }
        if node_id is not None:
            kwargs["node_id"] = node_id
        node = MemoryNode(**kwargs)

        if node.node_id in self.nodes:
            raise KeyError(f"node id already present: {node.node_id}")

        self.nodes[node.node_id] = node
        self.index.add(node.node_id, vec)
        self.dag.add_node_with_parents(node)
        self._acquire_references(node)
        return node

    def _acquire_references(self, node: MemoryNode) -> None:
        """Under `dependent_children`, a new node takes a reference on each parent."""
        if self.refcount_semantics != "dependent_children":
            return
        for p in node.parent_ids:
            parent = self.nodes.get(p)
            if parent is not None:
                parent.refcount += 1

    def add_node(self, node: MemoryNode) -> MemoryNode:
        """Insert a pre-built node (test fixtures, transcript replay)."""
        if node.node_id in self.nodes:
            raise KeyError(f"node id already present: {node.node_id}")
        if node.embedding is None:
            node.embedding = self.embedder.encode([node.content])[0]
        self.nodes[node.node_id] = node
        # Everything is indexed, including tombstones. See MemoryNode.returnable.
        self.index.add(node.node_id, node.embedding)
        self.dag.add_node_with_parents(node)
        self._acquire_references(node)
        return node

    def get(self, node_id: str) -> MemoryNode | None:
        return self.nodes.get(node_id)

    def __contains__(self, node_id: object) -> bool:
        return node_id in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    def all_nodes(self, include_deleted: bool = False) -> list[MemoryNode]:
        return [n for n in self.nodes.values() if include_deleted or not n.deleted]

    def indexed_ids(self) -> set[str]:
        """Every id the index holds, tombstones included."""
        return set(self.index.ids())

    def returnable_ids(self, blocklist: Blocklist | None = None) -> set[str]:
        """Ids retrieval may actually hand back. Invariant 1 is checked against this."""
        bl = blocklist if blocklist is not None else NoBlocklist()
        return {
            nid
            for nid in self.index.ids()
            if (n := self.nodes.get(nid)) is not None and n.returnable and not bl.blocks(n).blocked
        }

    # ----------------------------------------------------------------- retrieval --

    def retrieve(
        self,
        query: str,
        k: int = 5,
        blocklist: Blocklist | None = None,
        *,
        include_outdated: bool = True,
    ) -> RetrievalResult:
        """Exact top-k cosine retrieval with retrieval-time blocklist enforcement.

        Over-fetches so that suppressing blocked hits does not silently shrink k — a
        shrinking k would make the blocklist look more effective than it is.
        """
        bl = blocklist if blocklist is not None else NoBlocklist()
        result = RetrievalResult(query=query)
        if k <= 0 or len(self.index) == 0:
            return result

        qvec = self.embedder.encode([query])[0]
        # Fetch enough headroom to still return k after suppression.
        raw = self.index.search(qvec, min(len(self.index), max(k * 4, k + 10)))

        for node_id, score in raw:
            node = self.nodes.get(node_id)
            if node is None:
                continue
            # Blocklist FIRST, so that suppression is attributed to the blocklist rather
            # than to the tombstone. In the SBU setup a deleted node is also blocklisted,
            # and it is the blocklist we are measuring.
            if bl.blocks(node).blocked:
                result.blocked_node_ids.append(node_id)
                continue
            if not node.returnable:
                # Deleted with no blocklist configured (the C0 baseline). Suppressed, but
                # not attributable to a defence that is not there.
                continue
            if not include_outdated and node.outdated:
                continue
            result.nodes.append(node)
            result.scores.append(score)
            if len(result.nodes) >= k:
                break
        return result

    # ------------------------------------------------------------------ deletion --

    def delete(self, node_id: str, *, blocklist: Blocklist | None = None) -> PruneResult:
        """Delete a node and prune its derivation closure (the SBU memory pathway).

        `deletion_mode` decides what happens to the vector:

        ``tombstone`` (default)
            The node stays in the index; the blocklist is what suppresses it. Dropping
            it here would make invariant 1 vacuously true — we would be testing
            `del store[id]` rather than SBU's mechanism. See `MemoryNode.returnable`
            and ADR-0003.

        ``hard``
            The node AND its vector are removed, which is what the SBU paper describes.
            Invariant 1 then holds trivially, which is precisely why it is not the
            default here — but a repo that claims to test SBU has to be able to run
            SBU, and results produced under either mode must say which. ADR-0014.
        """
        node = self.nodes.get(node_id)
        if node is None:
            raise KeyError(f"no such node: {node_id}")
        res = self.dag.delete_with_closure(
            node_id, self.nodes, blocklist, refcount_semantics=self.refcount_semantics
        )
        if self.deletion_mode == "hard":
            self.index.remove(node_id)
            node.embedding = None
        return res

    def delete_many(
        self, node_ids: Iterable[str], *, blocklist: Blocklist | None = None
    ) -> PruneResult:
        res = PruneResult()
        for nid in node_ids:
            self.delete(nid, blocklist=blocklist)
            # delete() built its own result; re-derive the aggregate cheaply.
        # Recompute aggregate over the requested ids for a single reportable object.
        for nid in node_ids:
            res.deleted_ids.append(nid)
            res.closure_ids.extend(sorted(self.dag.dependency_closure(nid)))
        res.closure_ids = sorted(set(res.closure_ids))
        res.marked_outdated = sorted(
            n.node_id for n in self.nodes.values() if n.outdated and not n.deleted
        )
        return res

    # --------------------------------------------------------- snapshot / restore --

    def snapshot(self) -> StoreSnapshot:
        """Deep copy of the mutable state. C1/C2/C3 must start from the same one."""
        return StoreSnapshot(
            nodes=copy.deepcopy(self.nodes),
            dag=self.dag.copy(),
            turn=self.turn,
        )

    def restore(self, snap: StoreSnapshot) -> None:
        self.nodes = copy.deepcopy(snap.nodes)
        self.dag = snap.dag.copy()
        self.turn = snap.turn
        self.index = build_index(
            getattr(self.index, "backend_name", "numpy"), dim=self.embedder.dim
        )
        for node in self.nodes.values():
            # Under `hard` deletion the vector is gone on purpose. Re-encoding it here
            # would resurrect it on every snapshot round-trip and quietly undo the
            # deletion the run is supposed to be measuring.
            if node.deleted and self.deletion_mode == "hard":
                continue
            if node.embedding is None:
                node.embedding = self.embedder.encode([node.content])[0]
            self.index.add(node.node_id, node.embedding)

    def __deepcopy__(self, memo: dict) -> MemoryStore:
        other = MemoryStore(
            embedder=self.embedder,  # stateless and deterministic; sharing is safe
            index_backend=getattr(self.index, "backend_name", "numpy"),
            embedding_dim=self.embedder.dim,
            deletion_mode=self.deletion_mode,
            refcount_semantics=self.refcount_semantics,
        )
        other.restore(self.snapshot())
        return other

    # ------------------------------------------------------------------ reporting --

    def to_records(self, with_embedding: bool = False) -> list[dict]:
        return [n.to_record(with_embedding=with_embedding) for n in self.nodes.values()]

    def stats(self) -> dict:
        return {
            "deletion_mode": self.deletion_mode,
            "refcount_semantics": self.refcount_semantics,
            "n_nodes": len(self.nodes),
            "n_retrievable": len(self.index),
            "n_deleted": sum(1 for n in self.nodes.values() if n.deleted),
            "n_outdated": sum(1 for n in self.nodes.values() if n.outdated),
            "n_parametric": sum(1 for n in self.nodes.values() if n.is_parametric),
            "n_edges": len(self.dag.edges()),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"MemoryStore({self.stats()})"
