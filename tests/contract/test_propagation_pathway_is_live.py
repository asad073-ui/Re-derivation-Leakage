"""The propagation contrast has to be able to MEASURE something (GU-0034).

This file exists because the previous design passed every structural check and measured
nothing. `taint_only` and `tag_source_quarantine` differ in exactly one flag; a test
asserted that, and it was true. But both arms quarantine the tagged note at retrieval
before any agent reads it, so no derivative is ever produced, nothing exists for a
forwarding arm to tag, and the contrast is zero at any cohort size. The traces said so:
both arms recorded `returned=0, withheld=40, edges_carrying_a_tag=0`.

Single-variable is necessary and not sufficient. The pathway the variable acts on must be
live, and "live" is not a property you can read off a config — it is a property of a run.
So this is an executor-level test, on the real defence objects, over a real
source -> derivative -> downstream-boundary path.

Every assertion below is a precondition for believing an M5 result, in either direction.
If they hold and M5 is null, that is a genuine negative finding about forward propagation.
If they do not hold, M5 is not evidence of anything.
"""

from __future__ import annotations

import pytest

from rdl.graph.config import GraphLaunchConfig, resolve
from rdl.graph.executor import EpisodeSpec, GraphExecutor
from rdl.graph_memory.staged_store import StagedMemory
from rdl.studies.graph_leak.arms import build_arm_runtime, build_detector, build_registry
from rdl.studies.graph_leak.runner import build_baseline_memory

M5_TREATMENT = "multi_agent_graphforget_taint_forward"
M5_BASELINE = "multi_agent_graphforget_no_forward"


@pytest.fixture
def mechanism_world(tofu_items, graph_scheduler):
    cfg = resolve(GraphLaunchConfig(study="graphforget_mechanism_v2", active_profile="local_cpu"))
    registry = build_registry(tofu_items, concept_of=lambda item_id: f"c-{item_id}")
    detector = build_detector(cfg, registry)
    plans = {
        p.name: p for p in build_arm_runtime(cfg, cfg.topology, detector, protocol="graph_flow")
    }
    return cfg, plans, graph_scheduler


def _run_arm(cfg, plan, scheduler, item, tofu_items):
    """One memory-reentry episode, with the seeded note tagged as the study seeds it."""
    store, blocklist = build_baseline_memory(tofu_items)
    memory = StagedMemory(
        store,
        blocklist=blocklist,
        visibility=cfg.study.memory.visibility,
        propagate_parent_scopes=bool(getattr(plan.defense, "propagates_scope", False)),
    )
    concept = f"c-{item.item_id}"
    seeded = memory.seed_tagged(
        f"Note from an earlier session: {item.answer}",
        forget_ids=(concept,),
        meta={"injected": True},
    )
    executor = GraphExecutor(cfg.topology, scheduler=scheduler, defense=plan.defense)
    episode = EpisodeSpec(
        trajectory_id=f"{plan.name}:{item.item_id}:0",
        item_id=item.item_id,
        concept_id=concept,
        sample_id=0,
        arm=plan.name,
        question=item.question,
        active_nodes=plan.active_nodes,
        challenge="memory_reentry",
        topology=cfg.topology.name,
    )
    traj = executor.run([episode], [memory])[0]
    return traj, memory, seeded


def test_the_m5_pair_is_matched_on_tagged_source_exposure(mechanism_world, tofu_items) -> None:
    """Both arms must READ the tagged note. If either quarantines it, the pair is vacuous.

    This is the assertion that would have failed on the previous design, on a laptop,
    before any GPU was rented.
    """
    cfg, plans, scheduler = mechanism_world
    item = tofu_items[0]
    exposure = {}
    for arm in (M5_BASELINE, M5_TREATMENT):
        traj, _memory, seeded = _run_arm(cfg, plans[arm], scheduler, item, tofu_items)
        events = traj.trace.to_dict()["events"]
        reads = [e for e in events if e.get("kind") == "memory_read"]
        exposure[arm] = {
            "allowed": sum(1 for e in reads if seeded in (e.get("returned_node_ids") or [])),
            "withheld": sum(1 for e in reads if seeded in (e.get("withheld_node_ids") or [])),
        }
    for arm, counts in exposure.items():
        assert counts["allowed"] > 0, (
            f"{arm} never let the tagged source reach an agent, so no derivative exists "
            "and the propagation contrast cannot measure forwarding"
        )
        assert counts["withheld"] == 0, f"{arm} quarantined the source it is meant to read"
    assert (
        exposure[M5_BASELINE]["allowed"] == exposure[M5_TREATMENT]["allowed"]
    ), "unequal tagged-source exposure: the arms differ in something other than forwarding"


def test_only_the_forwarding_arm_tags_the_derivative(mechanism_world, tofu_items) -> None:
    """The single variable, observed in the trace rather than read off the config."""
    cfg, plans, scheduler = mechanism_world
    item = tofu_items[0]
    carried = {}
    for arm in (M5_BASELINE, M5_TREATMENT):
        traj, _memory, _seeded = _run_arm(cfg, plans[arm], scheduler, item, tofu_items)
        events = traj.trace.to_dict()["events"]
        edges = [e for e in events if e.get("kind") == "edge_decision"]
        writes = [e for e in events if e.get("kind") == "memory_write"]
        carried[arm] = {
            "edges_with_scope": sum(1 for e in edges if e.get("forget_ids")),
            "edges_blocked": sum(1 for e in edges if e.get("release_status", "pass") != "pass"),
            "writes_tagged": sum(1 for e in writes if e.get("forget_ids")),
            "edges_total": len(edges),
        }
    assert carried[M5_BASELINE]["edges_total"] > 0, "no edge was exercised at all"
    assert (
        carried[M5_BASELINE]["edges_with_scope"] == 0
    ), "the no-forward arm attached a scope to a derivative; it is not a no-forward arm"
    assert carried[M5_TREATMENT]["edges_with_scope"] > 0, (
        "the taint-forward arm forwarded nothing: the pathway from a read tagged source "
        "to a tagged derivative is broken, and M5 would be zero for that reason alone"
    )
    assert (
        carried[M5_TREATMENT]["edges_blocked"] > 0
    ), "the forwarded scope was never enforced downstream, so forwarding changed nothing"


def test_the_forwarding_arm_records_inherited_only_interventions(
    mechanism_world, tofu_items
) -> None:
    """`inherited_only` is enforcement no node-local semantic guard could have produced.

    With the detector off in both arms it is the whole of the treatment's effect, so a
    zero here means the mechanism did nothing even if the leakage numbers happen to differ.
    """
    cfg, plans, scheduler = mechanism_world
    item = tofu_items[0]

    baseline_plan, treatment_plan = plans[M5_BASELINE], plans[M5_TREATMENT]
    _run_arm(cfg, baseline_plan, scheduler, item, tofu_items)
    _run_arm(cfg, treatment_plan, scheduler, item, tofu_items)

    baseline = baseline_plan.defense.attribution.to_dict()
    treatment = treatment_plan.defense.attribution.to_dict()
    assert (
        treatment["inherited_only_enforcements"] > 0
    ), "the taint-forward arm made no enforcement attributable to inherited provenance"
    assert baseline["inherited_only_enforcements"] == 0, (
        "the no-forward arm enforced on inherited provenance downstream, which is the "
        "mechanism it is the control for"
    )
    # And the scope genuinely arrived through MEMORY, not through a peer message.
    assert treatment_plan.defense.counters.memory_borne_scope_hits > 0
    assert baseline_plan.defense.counters.memory_borne_scope_hits > 0, (
        "matched exposure means the baseline read the tagged note too; it simply does "
        "not forward what it read"
    )


def test_the_quarantine_pair_is_the_vacuous_one_and_is_labelled_so(
    mechanism_world, tofu_items
) -> None:
    """The old M5. Kept as M8, asserted vacuous, so a null there is never read as M5.

    Both arms withhold the tagged note at retrieval, so neither can speak to forwarding.
    This test documents that as a measured property rather than a claim in a comment.
    """
    cfg, plans, scheduler = mechanism_world
    item = tofu_items[0]
    for arm in (
        "multi_agent_graphforget_tag_source_quarantine",
        "multi_agent_graphforget_taint_only",
    ):
        traj, _memory, seeded = _run_arm(cfg, plans[arm], scheduler, item, tofu_items)
        events = traj.trace.to_dict()["events"]
        reads = [e for e in events if e.get("kind") == "memory_read"]
        allowed = sum(1 for e in reads if seeded in (e.get("returned_node_ids") or []))
        withheld = sum(1 for e in reads if seeded in (e.get("withheld_node_ids") or []))
        assert withheld > 0 and allowed == 0, (
            f"{arm} is expected to quarantine the tagged source; if that changed, M8 is no "
            "longer the structurally-null control it is reported as"
        )


def test_the_contrast_table_marks_the_quarantine_pairs_as_positive_controls() -> None:
    """A reader must not be able to quote M4 or M8 as evidence about propagation."""
    from rdl.eval.defense_reduction import MECHANISM_CONTRASTS

    kinds = {row[0]: row[3] for row in MECHANISM_CONTRASTS}
    assert kinds["M5"] == "causal"
    assert kinds["M4"] == "positive_control"
    assert kinds["M8"] == "positive_control"
    assert kinds["M6"] == "combined"
    assert kinds["M7"] == "combined"
