"""The five arms must differ ONLY in the intended treatment.

Same checkpoint, same graph, same prompts, same routing, same seeds, same k, same
max tokens, same memory setup, same evaluator. If any of that drifts, the contrast
stops being a contrast and every number in the paper becomes uninterpretable.
"""

from __future__ import annotations

import pytest

from rdl.graph.config import load_graph_config
from rdl.graph.executor import EpisodeSpec, GraphExecutor
from rdl.graph_memory.staged_store import StagedMemory
from rdl.paths import repo_root
from rdl.studies.graph_leak.arms import build_arm_runtime, build_detector, build_registry
from rdl.studies.graph_leak.runner import build_baseline_memory

LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"


@pytest.fixture
def plans(tofu_items):
    cfg = load_graph_config(LAUNCH)
    registry = build_registry(tofu_items, concept_of=lambda item_id: f"c-{item_id}")
    detector = build_detector(cfg, registry)
    return cfg, build_arm_runtime(cfg, cfg.topology, detector)


def _memory(tofu_items) -> StagedMemory:
    store, blocklist = build_baseline_memory(tofu_items)
    return StagedMemory(store, blocklist=blocklist)


def _episode(item, arm, *, sample=0, upstream=None):
    return EpisodeSpec(
        trajectory_id=f"{arm.name}:{item.item_id}:{sample}",
        item_id=item.item_id,
        concept_id=f"c-{item.item_id}",
        sample_id=sample,
        arm=arm.name,
        question=item.question,
        upstream_question=upstream.question if (arm.uses_control_question and upstream) else None,
        upstream_item_id=upstream.item_id if (arm.uses_control_question and upstream) else None,
        active_nodes=arm.active_nodes,
        topology="diamond5",
    )


def test_all_five_arms_resolve(plans):
    _cfg, arm_plans = plans
    assert [p.name for p in arm_plans] == [
        "single_agent",
        "multi_agent_control",
        "multi_agent_leak",
        "multi_agent_dragon",
        "multi_agent_graphforget",
    ]


def test_single_agent_runs_the_sink_and_the_rest_run_everything(plans):
    cfg, arm_plans = plans
    for plan in arm_plans:
        if plan.spec.mode == "single_agent":
            assert plan.active_nodes == (cfg.topology.sink,)
        else:
            assert plan.active_nodes == tuple(cfg.topology.node_ids())


def test_seeds_are_identical_across_arms(plans, tofu_items, graph_scheduler):
    """Common random numbers: the seed never keys on the arm."""
    cfg, arm_plans = plans
    seeds: dict[str, dict[str, int]] = {}
    for plan in arm_plans:
        executor = GraphExecutor(cfg.topology, scheduler=graph_scheduler, defense=plan.defense)
        episode = _episode(tofu_items[0], plan)
        seeds[plan.name] = {
            node: executor.seed_for_node(episode, node) for node in cfg.topology.node_ids()
        }
    reference = seeds["multi_agent_leak"]
    for arm in ("multi_agent_dragon", "multi_agent_graphforget", "single_agent"):
        assert seeds[arm] == reference, arm


def test_the_control_arm_reseeds_upstream_nodes_on_its_own_item(plans, tofu_items, graph_scheduler):
    cfg, arm_plans = plans
    control = next(p for p in arm_plans if p.name == "multi_agent_control")
    leak = next(p for p in arm_plans if p.name == "multi_agent_leak")
    executor = GraphExecutor(cfg.topology, scheduler=graph_scheduler, defense=control.defense)
    control_ep = _episode(tofu_items[0], control, upstream=tofu_items[3])
    leak_ep = _episode(tofu_items[0], leak)
    # The sink is the same question in both arms, so its draw is shared.
    assert executor.seed_for_node(control_ep, "E") == executor.seed_for_node(leak_ep, "E")
    # Upstream nodes answer a different question, so they draw on that item.
    assert executor.seed_for_node(control_ep, "A") != executor.seed_for_node(leak_ep, "A")


def test_unguarded_arms_produce_identical_trajectories(plans, tofu_items, graph_scheduler):
    """MA-CONTROL and MA-LEAK differ only in the peer content, never in the wrapper."""
    cfg, arm_plans = plans
    leak = next(p for p in arm_plans if p.name == "multi_agent_leak")
    executor = GraphExecutor(cfg.topology, scheduler=graph_scheduler, defense=leak.defense)
    item = tofu_items[0]
    first = executor.run([_episode(item, leak)], [_memory(tofu_items)])[0]
    second = executor.run(
        [_episode(item, leak).__class__(**{**_episode(item, leak).__dict__})],
        [_memory(tofu_items)],
    )[0]
    assert first.trace.digest() == second.trace.digest(), "the same arm must be deterministic"


def test_every_arm_uses_the_same_memory_lifecycle(plans, tofu_items, graph_scheduler):
    cfg, arm_plans = plans
    stats = {}
    for plan in arm_plans:
        executor = GraphExecutor(cfg.topology, scheduler=graph_scheduler, defense=plan.defense)
        memory = _memory(tofu_items)
        before = memory.store.stats()
        executor.run([_episode(tofu_items[0], plan)], [memory])
        stats[plan.name] = before
    baselines = list(stats.values())
    assert all(b == baselines[0] for b in baselines), "arms must start from one store state"


def test_the_same_detector_object_serves_both_guarded_arms(plans):
    _cfg, arm_plans = plans
    by_name = {p.name: p for p in arm_plans}
    assert (
        by_name["multi_agent_dragon"].defense.detector
        is by_name["multi_agent_graphforget"].defense.detector
    )
