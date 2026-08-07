"""The derivation DAG and SBU's memory-pathway deletion.

Edge direction: ``parent -> child`` means *child was derived from parent*. So:

  dependency_closure(n)  = transitive DESCENDANTS of n = everything derived from n.
                           This is what deletion must reach.
  ancestors(n)           = transitive PARENTS of n = where n's content came from.
                           This is what `invariants.certify` walks to look for a path
                           back to blocked content.

The one thing to get right here is that a diamond must not double-decrement. If
``m1 -> m2 -> m4`` and ``m1 -> m3 -> m4``, deleting ``m1`` reaches ``m4`` by two paths
but must decrement its refcount exactly once. The closure is computed as a *set* and
the decrement is applied per unique node, which is why `test_derivation_closure.py`
constructs precisely that shape.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from .node import MemoryNode

__all__ = ["CycleError", "DerivationDAG", "PruneResult"]


class CycleError(ValueError):
    """Raised when an edge would make the derivation graph cyclic."""


@dataclass
class PruneResult:
    """What a deletion actually did. Feeds Inv2Report and the invariant witness."""

    deleted_ids: list[str] = field(default_factory=list)
    closure_ids: list[str] = field(default_factory=list)
    decremented: dict[str, int] = field(default_factory=dict)  # node_id -> new refcount
    marked_outdated: list[str] = field(default_factory=list)
    blocklisted: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "deleted_ids": sorted(self.deleted_ids),
            "closure_ids": sorted(self.closure_ids),
            "decremented": dict(sorted(self.decremented.items())),
            "marked_outdated": sorted(self.marked_outdated),
            "blocklisted": sorted(self.blocklisted),
        }


class DerivationDAG:
    """Directed acyclic graph over node ids."""

    def __init__(self) -> None:
        self._children: dict[str, set[str]] = {}
        self._parents: dict[str, set[str]] = {}

    # ------------------------------------------------------------------ mutation --

    def add_node(self, node_id: str) -> None:
        self._children.setdefault(node_id, set())
        self._parents.setdefault(node_id, set())

    def add_edge(self, parent: str, child: str) -> None:
        """Record that `child` was derived from `parent`."""
        if parent == child:
            raise CycleError(f"self-edge on {parent}")
        self.add_node(parent)
        self.add_node(child)
        # A new edge parent->child is a cycle iff parent is already a descendant of child.
        if parent in self.dependency_closure(child):
            raise CycleError(
                f"edge {parent[:8]}->{child[:8]} would create a cycle; "
                "derivation must be acyclic"
            )
        self._children[parent].add(child)
        self._parents[child].add(parent)

    def add_node_with_parents(self, node: MemoryNode) -> None:
        """Register a node and its `parent_ids`. A parametric node adds no edges."""
        self.add_node(node.node_id)
        for p in node.parent_ids:
            self.add_edge(p, node.node_id)

    def remove_node(self, node_id: str) -> None:
        for p in self._parents.pop(node_id, set()):
            self._children.get(p, set()).discard(node_id)
        for c in self._children.pop(node_id, set()):
            self._parents.get(c, set()).discard(node_id)

    # ------------------------------------------------------------------- queries --

    def nodes(self) -> set[str]:
        return set(self._children)

    def children(self, node_id: str) -> set[str]:
        return set(self._children.get(node_id, set()))

    def parents(self, node_id: str) -> set[str]:
        return set(self._parents.get(node_id, set()))

    def edges(self) -> list[tuple[str, str]]:
        return sorted((p, c) for p, cs in self._children.items() for c in cs)

    def dependency_closure(self, node_id: str, include_self: bool = False) -> set[str]:
        """Transitive descendants: everything derived, directly or indirectly, from `node_id`."""
        seen: set[str] = set()
        queue = deque(self._children.get(node_id, set()))
        while queue:
            cur = queue.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            queue.extend(self._children.get(cur, set()) - seen)
        if include_self:
            seen.add(node_id)
        return seen

    def ancestors(self, node_id: str, include_self: bool = False) -> set[str]:
        """Transitive parents: everything `node_id` was derived from."""
        seen: set[str] = set()
        queue = deque(self._parents.get(node_id, set()))
        while queue:
            cur = queue.popleft()
            if cur in seen:
                continue
            seen.add(cur)
            queue.extend(self._parents.get(cur, set()) - seen)
        if include_self:
            seen.add(node_id)
        return seen

    def path_to_ancestor(self, node_id: str, targets: Iterable[str]) -> list[str] | None:
        """Shortest ancestor path ``[node_id, ..., target]``, or None if unreachable.

        This is what makes an invariant certificate falsifiable: a laundered node must
        return None here, and the returned path is the counter-example when it does not.
        """
        target_set = set(targets)
        if not target_set:
            return None
        if node_id in target_set:
            return [node_id]
        prev: dict[str, str] = {}
        seen = {node_id}
        queue = deque([node_id])
        while queue:
            cur = queue.popleft()
            for p in sorted(self._parents.get(cur, set())):
                if p in seen:
                    continue
                seen.add(p)
                prev[p] = cur
                if p in target_set:
                    path = [p]
                    while path[-1] != node_id:
                        path.append(prev[path[-1]])
                    return list(reversed(path))
                queue.append(p)
        return None

    # ------------------------------------------------------------------- pruning --

    def prune(
        self,
        node_id: str,
        nodes: Mapping[str, MemoryNode],
        *,
        result: PruneResult | None = None,
        refcount_semantics: str = "supporting_parents",
    ) -> PruneResult:
        """Reference-count decrement + outdated-marking over `node_id`'s closure.

        Two refcount semantics, because the SBU paper and this implementation do not
        agree and pretending otherwise would be the kind of thing a reviewer catches:

        ``supporting_parents`` (default, ADR-0004)
            refcount = live parents supporting this node. Deletion walks DOWN the
            derivation edges and decrements each node in the closure exactly once —
            once per *unique* node, so a diamond does not double-decrement.

        ``dependent_children`` (SBU as written, ADR-0015)
            refcount = how many nodes depend on this one, i.e. classic reference
            counting for reclamation. Deleting a node releases its references, so
            deletion walks UP: each ancestor loses one dependent, and an ancestor whose
            count reaches zero has nothing left depending on it and is reclaimed. The
            downward closure is still marked outdated — that is invariant 2 and it is
            required in both modes — but it is marked by the provenance rule rather
            than by a decrement.

        Which one produced a number belongs in the results table. They differ on real
        shapes: upward reclamation frees shared context nodes that downward pruning
        leaves live, and downward pruning invalidates derived content that upward
        reclamation leaves retrievable.
        """
        res = result or PruneResult()
        closure = self.dependency_closure(node_id)
        res.closure_ids = sorted(set(res.closure_ids) | closure)

        if refcount_semantics == "dependent_children":
            # Release this node's references to everything it was derived from.
            for aid in sorted(self.ancestors(node_id)):
                anc = nodes.get(aid)
                if anc is None:
                    continue
                anc.refcount = max(0, anc.refcount - 1)
                res.decremented[aid] = anc.refcount
                if anc.refcount <= 0 and not anc.outdated:
                    anc.outdated = True
                    res.marked_outdated.append(aid)
        elif refcount_semantics == "supporting_parents":
            # Pass 1: decrement each unique node EXACTLY once. A diamond reaches its
            # join node by two paths and must still decrement it a single time.
            for cid in sorted(closure):
                node = nodes.get(cid)
                if node is None:
                    continue
                node.refcount = max(0, node.refcount - 1)
                res.decremented[cid] = node.refcount
        else:
            raise ValueError(
                f"unknown refcount_semantics '{refcount_semantics}' "
                "(expected supporting_parents|dependent_children)"
            )

        # Pass 2: mark outdated, to a fixpoint. A node is outdated when its refcount
        # reached zero, OR when every parent it has is itself deleted or outdated —
        # the second rule is what catches the diamond join, whose refcount survives the
        # single decrement but whose entire provenance is dead.
        changed = True
        while changed:
            changed = False
            for cid in sorted(closure):
                node = nodes.get(cid)
                if node is None or node.outdated:
                    continue
                parents = self._parents.get(cid, set())
                provenance_dead = bool(parents) and all(
                    (nodes[p].deleted or nodes[p].outdated) for p in parents if p in nodes
                )
                if node.refcount <= 0 or provenance_dead:
                    node.outdated = True
                    res.marked_outdated.append(cid)
                    changed = True
        return res

    def delete_with_closure(
        self,
        node_id: str,
        nodes: Mapping[str, MemoryNode],
        blocklist=None,
        *,
        result: PruneResult | None = None,
        refcount_semantics: str = "supporting_parents",
    ) -> PruneResult:
        """The SBU memory-pathway operation: delete a node, prune what derived from it.

        When a blocklist with an `add` method is supplied, the deleted id is added to
        it, which is what makes the deletion durable against re-retrieval.
        """
        res = result or PruneResult()
        node = nodes.get(node_id)
        if node is not None:
            node.deleted = True
            node.refcount = 0
            node.outdated = True
        res.deleted_ids.append(node_id)
        self.prune(node_id, nodes, result=res, refcount_semantics=refcount_semantics)
        if blocklist is not None and hasattr(blocklist, "add"):
            blocklist.add(node_id)
            res.blocklisted.append(node_id)
        return res

    # -------------------------------------------------------------- (de)serialise --

    def to_dict(self) -> dict:
        return {"nodes": sorted(self.nodes()), "edges": [list(e) for e in self.edges()]}

    @classmethod
    def from_dict(cls, d: Mapping) -> DerivationDAG:
        dag = cls()
        for n in d.get("nodes", []):
            dag.add_node(str(n))
        for parent, child in d.get("edges", []):
            dag.add_edge(str(parent), str(child))
        return dag

    def copy(self) -> DerivationDAG:
        other = DerivationDAG()
        other._children = {k: set(v) for k, v in self._children.items()}
        other._parents = {k: set(v) for k, v in self._parents.items()}
        return other

    def __len__(self) -> int:
        return len(self._children)

    def __repr__(self) -> str:  # pragma: no cover
        return f"DerivationDAG({len(self._children)} nodes, {len(self.edges())} edges)"
