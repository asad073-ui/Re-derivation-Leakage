"""Descendant tagging over the derivation DAG.

The two-agent work's deletion mechanism (SBU's: delete a node, prune its dependency
closure) is reused unchanged. What is added is the *scope* analogue: when a node turns
out to carry a forgotten concept, everything derived from it inherits the tag, because
a summary of tainted content is tainted content.

This is why the write guard alone is not enough. A node written before its concept was
recognised can be reached later; tagging the closure is what makes recognition
retroactive.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from ..memory.derivation import DerivationDAG
from ..memory.node import MemoryNode
from .policy_node import PolicyRecord, node_forget_ids, tag_node
from .scope_index import ScopeIndex

__all__ = ["tag_closure"]


def tag_closure(
    *,
    dag: DerivationDAG,
    nodes: Mapping[str, MemoryNode],
    index: ScopeIndex,
    node_id: str,
    forget_ids: Iterable[str],
    detector_version: str = "",
    reason: str = "inherited from a tagged ancestor",
) -> tuple[str, ...]:
    """Tag `node_id` and every node derived from it. Returns the tagged ids."""
    ids = tuple(sorted({str(x) for x in forget_ids}))
    if not ids:
        return ()
    targets = [node_id, *sorted(dag.dependency_closure(node_id))]
    tagged: list[str] = []
    for target in targets:
        node = nodes.get(target)
        if node is None:
            continue
        before = set(node_forget_ids(node))
        index.tag(target, ids)
        tag_node(
            node,
            PolicyRecord(
                forget_ids=ids,
                detector_version=detector_version,
                release_status="blocked",
                reason=reason if target != node_id else "detected on this node",
            ),
        )
        if set(ids) - before:
            tagged.append(target)
    return tuple(tagged)
