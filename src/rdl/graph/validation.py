"""Structural checks a graph spec must pass before anything is generated.

Every check here exists because its failure mode is *silent*: a duplicated node id, a
dangling edge, or an undeclared sink all produce a run that completes and reports a
number under a graph that is not the one the config describes.
"""

from __future__ import annotations

from .schema import GraphSpec

__all__ = ["GraphConfigError", "validate_graph"]


class GraphConfigError(ValueError):
    """Raised for any malformed topology or graph/profile mismatch."""


def validate_graph(spec: GraphSpec, *, expected_nodes: int | None = None) -> GraphSpec:
    """Validate `spec`, returning it unchanged so calls can be chained.

    `expected_nodes` is the profile's declared logical-agent count. Checking it here is
    what stops a runtime profile from quietly changing the science: a five-node topology
    run with `logical_count: 3` would execute a different experiment under the same
    study id.
    """
    ids = [n.node_id for n in spec.nodes]
    duplicates = sorted({nid for nid in ids if ids.count(nid) > 1})
    if duplicates:
        raise GraphConfigError(f"graph '{spec.name}': duplicate node ids {duplicates}")

    known = set(ids)
    for edge in spec.edges:
        if edge.src not in known or edge.dst not in known:
            raise GraphConfigError(
                f"graph '{spec.name}': edge {edge.src}->{edge.dst} references an undeclared node"
            )
        if edge.src == edge.dst:
            raise GraphConfigError(f"graph '{spec.name}': self-edge on {edge.src}")

    seen_edges = [e.as_tuple() for e in spec.edges]
    dup_edges = sorted({e for e in seen_edges if seen_edges.count(e) > 1})
    if dup_edges:
        raise GraphConfigError(f"graph '{spec.name}': duplicate edges {dup_edges}")

    try:
        spec.layers()
    except ValueError as exc:
        raise GraphConfigError(
            f"graph '{spec.name}' is not a DAG: {exc}. Cyclic topologies need an explicit "
            "round counter before provenance closure is well defined; they are out of "
            "scope for graph-unlearning-v1."
        ) from exc

    if spec.sink not in known:
        raise GraphConfigError(f"graph '{spec.name}': sink '{spec.sink}' is not a node")
    if spec.children(spec.sink):
        raise GraphConfigError(
            f"graph '{spec.name}': sink '{spec.sink}' has outgoing edges "
            f"{list(spec.children(spec.sink))}; the final answer must be produced by a "
            "node nothing consumes."
        )

    if len(spec.nodes) > 1:
        reachable = _reachable_from_roots(spec)
        orphans = sorted(known - reachable)
        if orphans:
            raise GraphConfigError(
                f"graph '{spec.name}': nodes {orphans} are unreachable from any root; "
                "an isolated agent contributes generations without participating in the "
                "measured communication."
            )

    if expected_nodes is not None and len(spec.nodes) != expected_nodes:
        raise GraphConfigError(
            f"graph '{spec.name}' has {len(spec.nodes)} nodes but the active profile "
            f"declares logical_count={expected_nodes}. A runtime profile must not change "
            "how many agents the experiment runs."
        )
    return spec


def _reachable_from_roots(spec: GraphSpec) -> set[str]:
    seen: set[str] = set()
    stack = list(spec.roots())
    while stack:
        nid = stack.pop()
        if nid in seen:
            continue
        seen.add(nid)
        stack.extend(spec.children(nid))
    return seen
