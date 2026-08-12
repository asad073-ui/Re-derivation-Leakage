"""The two protocols measure different things and are never pooled.

The blocker this closes: the concept registry's prototypes include the forget questions
themselves, so a forget question has near-maximal similarity to its own prototype. Under
`end_to_end_safety` both guarded arms therefore fire at the root, produce a refusal
before the model is called, and come out clean — while propagation, edge enforcement and
write protection are never exercised at all. That comparison shows request filtering
works. It does not show the graph contribution.
"""

from __future__ import annotations

import pytest

from rdl.eval.graph_leak import leak_curves
from rdl.graph.config import load_graph_config
from rdl.graph.executor import EpisodeSpec, GraphExecutor
from rdl.graph_memory.staged_store import StagedMemory
from rdl.paths import repo_root
from rdl.studies.graph_leak.arms import build_arm_runtime, build_detector, build_registry
from rdl.studies.graph_leak.runner import build_baseline_memory

LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"


@pytest.fixture
def world(tofu_items, graph_scheduler):
    cfg = load_graph_config(LAUNCH)
    registry = build_registry(tofu_items, concept_of=lambda item_id: f"c-{item_id}")
    return cfg, build_detector(cfg, registry), graph_scheduler


def _run(cfg, plan, scheduler, item, tofu_items, *, injected=None):
    store, blocklist = build_baseline_memory(tofu_items)
    memory = StagedMemory(store, blocklist=blocklist)
    executor = GraphExecutor(cfg.topology, scheduler=scheduler, defense=plan.defense)
    episode = EpisodeSpec(
        trajectory_id=f"{plan.name}:{item.item_id}:0",
        item_id=item.item_id,
        concept_id=f"c-{item.item_id}",
        sample_id=0,
        arm=plan.name,
        question=item.question,
        active_nodes=plan.active_nodes,
        injected_outputs=injected or {},
        challenge="natural" if not injected else "direct",
        topology=cfg.topology.name,
    )
    return executor.run([episode], [memory])[0]


def _plans(cfg, detector, protocol):
    return {p.name: p for p in build_arm_runtime(cfg, cfg.topology, detector, protocol=protocol)}


def test_the_forget_question_is_its_own_detector_prototype(world, tofu_items):
    """The premise of the whole problem, asserted rather than assumed."""
    _cfg, detector, _scheduler = world
    result = detector.score(tofu_items[0].question)
    assert result.fired
    assert result.score >= 0.99


def test_end_to_end_safety_refuses_at_the_root(world, tofu_items):
    cfg, detector, scheduler = world
    plans = _plans(cfg, detector, "end_to_end_safety")
    traj = _run(cfg, plans["multi_agent_graphforget"], scheduler, tofu_items[0], tofu_items)
    executed = traj.trace.of_kind("node_executed")
    assert all(e.abstained for e in executed), "every node refuses on the query alone"
    assert traj.n_generations == 0, "the model is never called"


def test_graph_flow_lets_the_graph_actually_run(world, tofu_items):
    """Under graph_flow the guarded arm must still generate, or nothing is measured."""
    cfg, detector, scheduler = world
    plans = _plans(cfg, detector, "graph_flow")
    traj = _run(cfg, plans["multi_agent_graphforget"], scheduler, tofu_items[0], tofu_items)
    assert traj.n_generations > 0
    executed = traj.trace.of_kind("node_executed")
    assert not all(e.abstained for e in executed)


def test_graph_flow_holds_the_request_gate_constant_across_arms(world):
    """No arm inspects the query, so the gate cannot explain any arm's advantage."""
    cfg, detector, _scheduler = world
    plans = _plans(cfg, detector, "graph_flow")
    for name in ("multi_agent_leak", "multi_agent_dragon", "multi_agent_graphforget"):
        defense = plans[name].defense
        assert getattr(defense, "inspect_query", False) is False, name


def test_graph_flow_still_contains_what_the_graph_carries(world, tofu_items):
    """Holding the request gate constant is not turning the defence off."""
    cfg, detector, scheduler = world
    plans = _plans(cfg, detector, "graph_flow")
    injected = {"A": tofu_items[0].answer}
    leaky = _run(
        cfg, plans["multi_agent_leak"], scheduler, tofu_items[0], tofu_items, injected=injected
    )
    guarded = _run(
        cfg,
        plans["multi_agent_graphforget"],
        scheduler,
        tofu_items[0],
        tofu_items,
        injected=injected,
    )
    assert tofu_items[0].answer in " ".join(leaky.released_edge_texts())
    assert tofu_items[0].answer not in " ".join(guarded.released_edge_texts())
    assert not guarded.committed_write_ids


def test_the_two_protocols_produce_different_traces(world, tofu_items):
    cfg, detector, scheduler = world
    gated = _run(
        cfg,
        _plans(cfg, detector, "end_to_end_safety")["multi_agent_graphforget"],
        scheduler,
        tofu_items[0],
        tofu_items,
    )
    ungated = _run(
        cfg,
        _plans(cfg, detector, "graph_flow")["multi_agent_graphforget"],
        scheduler,
        tofu_items[0],
        tofu_items,
    )
    assert gated.trace.digest() != ungated.trace.digest()


def test_score_rows_are_filtered_by_protocol_never_pooled():
    rows = [
        {
            "arm": "a",
            "item_id": "i0",
            "concept_id": "c0",
            "sample_id": 0,
            "challenge": "natural",
            "protocol": protocol,
            "certified_persistent_leak": protocol == "graph_flow",
        }
        for protocol in ("end_to_end_safety", "graph_flow")
    ]
    safety = leak_curves(
        rows, k_values=[1], surfaces=["certified_persistent_leak"], protocol="end_to_end_safety"
    )
    flow = leak_curves(
        rows, k_values=[1], surfaces=["certified_persistent_leak"], protocol="graph_flow"
    )
    assert safety["certified_persistent_leak"].curve("a", [1])[1] == 0.0
    assert flow["certified_persistent_leak"].curve("a", [1])[1] == 1.0


def test_an_unknown_protocol_is_refused(tofu_items, graph_backend):
    from rdl.studies.graph_leak.cohort import load_cohort, resolve_cohort
    from rdl.studies.graph_leak.runner import GraphRunner

    cfg = load_graph_config(LAUNCH)
    cohort = load_cohort(repo_root() / "data/cohorts/graph_unlearning_v1/cpu_stub.json").limited(2)
    with pytest.raises(ValueError, match="not enabled by study"):
        GraphRunner(
            cfg=cfg,
            items=resolve_cohort(cohort, tofu_items),
            cohort=cohort,
            backend=graph_backend,
            output=repo_root() / "runs" / "graph" / "never-written",
            protocol="no_such_protocol",
        )
