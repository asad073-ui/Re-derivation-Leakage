"""Agent nodes, directed edges, and the graph specification.

A ``GraphSpec`` is the *execution* graph: who talks to whom. It is an experimental
factor, never the defence. Two arms that differ in topology are not a controlled
comparison, so every arm in a study runs the identical spec and the enforcement layer
acts on edge *payloads* instead of on edge existence. See
``docs/graph_unlearning/PROTOCOL_v1.md`` §3.

Only DAGs are supported. A cyclic ("ring") topology needs an explicit round counter to
have a well-defined provenance closure at all, and inventing one silently would make
Forget-ID propagation depend on an undeclared iteration limit.
"""

from __future__ import annotations

from collections import deque
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["AgentNode", "AgentRole", "GraphBase", "GraphEdge", "GraphSpec"]

# Roles are fixed vocabulary rather than free text: the five arms must present
# byte-identical system prompts, and a typo'd role would silently give one arm a
# different prompt while every provenance hash still matched.
AgentRole = Literal["proposer", "analyst", "reviewer", "integrator", "finalizer"]


class GraphBase(BaseModel):
    """Frozen, strict, hashable. Every graph-study config model inherits this."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AgentNode(GraphBase):
    """One logical agent.

    ``model`` names a model profile, not a checkpoint copy. Five logical agents sharing
    one physical handle is the intended RTX-3090 configuration; the run manifest records
    the sharing explicitly so a report can never call them independently unlearned.
    """

    node_id: str
    role: AgentRole
    model: str = "primary"
    system_prompt: str | None = None
    # Whether this node's output is a persistent-memory write candidate. The write path
    # is a measured surface, so it is declared per node rather than assumed.
    writes_memory: bool = True


class GraphEdge(GraphBase):
    src: str
    dst: str

    def as_tuple(self) -> tuple[str, str]:
        return (self.src, self.dst)


class GraphSpec(GraphBase):
    """A validated DAG of agents.

    ``sink`` is the node whose output is the system's final answer. It is declared
    rather than inferred: a dense DAG can have several childless nodes, and letting the
    runner pick one by iteration order would make "final-output leakage" depend on dict
    ordering.
    """

    name: str
    description: str = ""
    nodes: tuple[AgentNode, ...] = Field(min_length=1)
    edges: tuple[GraphEdge, ...] = ()
    sink: str

    # ------------------------------------------------------------------- queries --

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.node_id for n in self.nodes)

    def node(self, node_id: str) -> AgentNode:
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        raise KeyError(f"no such node in graph '{self.name}': {node_id}")

    def parents(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(e.src for e in self.edges if e.dst == node_id))

    def children(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted(e.dst for e in self.edges if e.src == node_id))

    def descendants(self, node_id: str) -> tuple[str, ...]:
        """Every node reachable from `node_id`, excluding itself."""
        seen: set[str] = set()
        stack = list(self.children(node_id))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.children(current))
        return tuple(sorted(seen))

    def roots(self) -> tuple[str, ...]:
        return tuple(sorted(n.node_id for n in self.nodes if not self.parents(n.node_id)))

    def leaves(self) -> tuple[str, ...]:
        return tuple(sorted(n.node_id for n in self.nodes if not self.children(n.node_id)))

    def edge_tuples(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(e.as_tuple() for e in self.edges))

    def layers(self) -> tuple[tuple[str, ...], ...]:
        """Kahn levels: every node in layer *i* has all its parents in layers < *i*.

        This is the batching unit. Executing one whole trajectory at a time leaves the
        GPU idle between hops; executing a layer at a time lets every node at the same
        depth, across every item and sample in flight, go into one batch.

        Deterministic: nodes are sorted within a layer, so two runs of the same spec
        produce the same batch composition and therefore the same per-request seeds.
        """
        indegree = {n.node_id: len(self.parents(n.node_id)) for n in self.nodes}
        ready = deque(sorted(nid for nid, deg in indegree.items() if deg == 0))
        out: list[tuple[str, ...]] = []
        seen = 0
        while ready:
            layer = tuple(sorted(ready))
            out.append(layer)
            ready.clear()
            for nid in layer:
                seen += 1
                for child in self.children(nid):
                    indegree[child] -= 1
                    if indegree[child] == 0:
                        ready.append(child)
        if seen != len(self.nodes):
            # validation.validate_graph raises with a better message; this is the guard
            # for a spec constructed in code without going through it.
            raise ValueError(f"graph '{self.name}' is cyclic: only {seen}/{len(self.nodes)} ranked")
        return tuple(out)

    def depth_of(self, node_id: str) -> int:
        for depth, layer in enumerate(self.layers()):
            if node_id in layer:
                return depth
        raise KeyError(f"no such node in graph '{self.name}': {node_id}")

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")
