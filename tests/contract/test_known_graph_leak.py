"""A known-leak scenario, end to end, showing what each arm does about it.

The forgotten answer is injected at the root, so leakage is guaranteed by construction
and the arms differ only in containment. This is the test that would fail if the
enforcement layer were ever silently disabled.
"""

from __future__ import annotations

import pytest

from rdl.eval.graph_leak import surface_flags
from rdl.eval.semantic import OfflineSemanticScorer
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
    detector = build_detector(cfg, registry)
    return cfg, build_arm_runtime(cfg, cfg.topology, detector), graph_scheduler


def _run(cfg, plan, scheduler, item, tofu_items):
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
        injected_outputs={"A": item.answer},
        challenge="direct",
        topology=cfg.topology.name,
    )
    return executor.run([episode], [memory])[0]


def _flags(trajectory, item):
    scorer = OfflineSemanticScorer()
    row = {
        "reference_answer": item.answer,
        "final_text": trajectory.final_text,
        "raw_outputs": {
            "agent_messages": trajectory.agent_output_texts(),
            "released_edge_payloads": trajectory.released_edge_texts(),
            "probe": {},
        },
        "memory_evidence": [
            {
                "content": node.content,
                "parent_ids": list(node.parent_ids),
                "is_parametric": node.is_parametric,
            }
            for node_id in trajectory.committed_write_ids
            if (node := trajectory.memory.store.get(node_id)) is not None
        ],
    }
    return surface_flags(row, lambda ref, cand: scorer.score(ref, cand).leaks)


def test_the_unguarded_arm_leaks_on_every_surface(world, tofu_items):
    cfg, plans, scheduler = world
    plan = next(p for p in plans if p.name == "multi_agent_leak")
    flags = _flags(_run(cfg, plan, scheduler, tofu_items[0], tofu_items), tofu_items[0])
    assert flags["raw_message_leak"]
    assert flags["edge_leak"]
    assert flags["store_leak"]
    assert flags["certified_persistent_leak"]


def test_the_node_local_baseline_leaves_persistent_memory_open(world, tofu_items):
    """The paper's central empirical point about DRAGON-style guarding."""
    cfg, plans, scheduler = world
    plan = next(p for p in plans if p.name == "multi_agent_dragon")
    flags = _flags(_run(cfg, plan, scheduler, tofu_items[0], tofu_items), tofu_items[0])
    assert not flags["sink_leak"], "its final-output check does fire"
    assert flags["edge_leak"], "but the content still crossed the graph"
    assert flags["store_leak"], "and it was still written to persistent memory"


def test_graphforget_contains_it_on_every_enforced_surface(world, tofu_items):
    cfg, plans, scheduler = world
    plan = next(p for p in plans if p.name == "multi_agent_graphforget")
    trajectory = _run(cfg, plan, scheduler, tofu_items[0], tofu_items)
    flags = _flags(trajectory, tofu_items[0])
    assert not flags["edge_leak"]
    assert not flags["sink_leak"]
    assert not flags["store_leak"]
    assert not flags["certified_persistent_leak"]
    # Honest about what it does NOT claim: the model still produced the content.
    assert flags["raw_message_leak"]


def test_containment_does_not_remove_any_edge(world, tofu_items):
    cfg, plans, scheduler = world
    plan = next(p for p in plans if p.name == "multi_agent_graphforget")
    trajectory = _run(cfg, plan, scheduler, tofu_items[0], tofu_items)
    decisions = trajectory.trace.of_kind("edge_decision")
    assert len(decisions) == len(cfg.topology.edges)
    assert all(d.edge_preserved for d in decisions)


def test_single_agent_never_receives_the_injection(world, tofu_items):
    cfg, plans, scheduler = world
    plan = next(p for p in plans if p.name == "single_agent")
    trajectory = _run(cfg, plan, scheduler, tofu_items[0], tofu_items)
    assert set(trajectory.node_outputs) == {cfg.topology.sink}
    assert not _flags(trajectory, tofu_items[0])["edge_leak"]
