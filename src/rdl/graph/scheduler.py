"""Which (trajectory, node) pairs are ready at each graph depth.

Kept separate from the executor so the batching plan is testable without a backend:
"does a 4-item x 2-sample x 5-arm smoke really issue 168 generations" has to be
answerable before a GPU is rented, not after.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .schema import GraphSpec

__all__ = ["ReadyUnit", "plan_layers", "planned_generations"]


@dataclass(frozen=True)
class ReadyUnit:
    trajectory_id: str
    node_id: str
    depth: int


def plan_layers(
    spec: GraphSpec,
    trajectories: Sequence[tuple[str, frozenset[str]]],
) -> Iterator[tuple[int, tuple[ReadyUnit, ...]]]:
    """Yield ``(depth, units)`` for every graph layer.

    `trajectories` is ``[(trajectory_id, active_node_ids), ...]``. The single-agent arm
    passes a one-element node set; every multi-agent arm passes them all. Units within a
    layer are ordered by ``(trajectory_id, node_id)`` so batch composition — and
    therefore the per-request seeds and the cache keys — are reproducible.
    """
    for depth, layer in enumerate(spec.layers()):
        units = tuple(
            ReadyUnit(trajectory_id=tid, node_id=node_id, depth=depth)
            for tid, active in sorted(trajectories, key=lambda t: t[0])
            for node_id in layer
            if node_id in active
        )
        if units:
            yield depth, units


def planned_generations(
    spec: GraphSpec,
    *,
    n_items: int,
    n_samples: int,
    arms: Sequence[str],
    single_agent_arms: Sequence[str] = ("single_agent",),
    injected_nodes_per_trajectory: int = 0,
) -> dict[str, int]:
    """The dry-run cost model. Compared against the real count after a run.

    Deliberately ignores the response cache: a plan that assumed cache hits would fail
    the moment a defence rewrote a prompt, and the gate wants an upper bound.
    """
    per_multi = max(0, len(spec.nodes) - injected_nodes_per_trajectory)
    n_multi = sum(1 for arm in arms if arm not in set(single_agent_arms))
    n_single = len(arms) - n_multi
    per_traj_set = n_single * 1 + n_multi * per_multi
    return {
        "n_items": n_items,
        "n_samples": n_samples,
        "n_arms": len(arms),
        "nodes_per_graph": len(spec.nodes),
        "generations_per_item_sample": per_traj_set,
        "planned_generations": n_items * n_samples * per_traj_set,
        "planned_trajectories": n_items * n_samples * len(arms),
    }
