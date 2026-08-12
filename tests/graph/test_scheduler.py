"""Layer planning and the dry-run cost model.

The 168 in ``test_smoke_predicts_168_generations`` is the GPU-readiness gate's number.
If it changes, the protocol changed and the change has to be deliberate.
"""

from __future__ import annotations

from rdl.graph.scheduler import plan_layers, planned_generations

ARMS = (
    "single_agent",
    "multi_agent_control",
    "multi_agent_leak",
    "multi_agent_dragon",
    "multi_agent_graphforget",
)


def test_layers_group_by_depth_across_trajectories(diamond5):
    trajectories = [("t1", frozenset(diamond5.node_ids())), ("t0", frozenset(diamond5.node_ids()))]
    plan = list(plan_layers(diamond5, trajectories))
    assert [depth for depth, _units in plan] == [0, 1, 2, 3]
    depth1 = plan[1][1]
    # Both trajectories' B and C land in ONE batch, and the order is deterministic.
    assert [(u.trajectory_id, u.node_id) for u in depth1] == [
        ("t0", "B"),
        ("t0", "C"),
        ("t1", "B"),
        ("t1", "C"),
    ]


def test_single_agent_trajectory_contributes_one_unit(diamond5):
    plan = dict(plan_layers(diamond5, [("sa", frozenset({diamond5.sink}))]))
    units = [u for layer in plan.values() for u in layer]
    assert [u.node_id for u in units] == ["E"]


def test_smoke_predicts_168_generations(diamond5):
    plan = planned_generations(
        diamond5, n_items=4, n_samples=2, arms=ARMS, single_agent_arms=("single_agent",)
    )
    # 1 (SA) + 4 arms x 5 nodes = 21 per item/sample; 4 x 2 x 21 = 168.
    assert plan["generations_per_item_sample"] == 21
    assert plan["planned_generations"] == 168
    assert plan["planned_trajectories"] == 40


def test_small_discovery_predicts_3360(diamond5):
    plan = planned_generations(
        diamond5, n_items=20, n_samples=8, arms=ARMS, single_agent_arms=("single_agent",)
    )
    assert plan["planned_generations"] == 3360


def test_full_discovery_predicts_33600(diamond5):
    plan = planned_generations(
        diamond5, n_items=50, n_samples=32, arms=ARMS, single_agent_arms=("single_agent",)
    )
    assert plan["planned_generations"] == 33600


def test_injected_nodes_reduce_the_generation_count(diamond5):
    plain = planned_generations(diamond5, n_items=4, n_samples=2, arms=ARMS)
    injected = planned_generations(
        diamond5, n_items=4, n_samples=2, arms=ARMS, injected_nodes_per_trajectory=2
    )
    assert injected["planned_generations"] < plain["planned_generations"]
    # 1 + 4 x 3 = 13 per item/sample.
    assert injected["generations_per_item_sample"] == 13
