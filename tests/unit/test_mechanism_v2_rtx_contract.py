"""The contract the mechanism study must satisfy BEFORE any GPU is rented (GU-0033).

Every test here corresponds to a way the previous two GPU runs were wasted, and each one
fails loudly on a laptop rather than quietly on rented hardware:

* the run used a different cohort from the one anybody intended,
* the detector ran at an operating point nothing was ever measured at,
* an "ablation" arm was still doing the thing it is the control for,
* a claimed contrast was never actually computed,
* the sample budget silently substituted a k the study did not preregister.

The purity tests are the load-bearing ones. A number that is wrong is a bug; an arm that
enforces the mechanism it is supposed to lack produces a contrast of a thing against
itself, and that survives review because every number in it looks reasonable.
"""

from __future__ import annotations

import json

import pytest

from rdl.defenses.base import EdgeContext, RetrievalContext, WriteContext
from rdl.defenses.concept_registry import ConceptRegistry
from rdl.defenses.forget_policy import ForgetPolicy
from rdl.defenses.graphforget import GraphForgetDefense
from rdl.defenses.semantic_detector import SemanticConceptDetector
from rdl.eval.defense_reduction import MECHANISM_CONTRASTS
from rdl.graph.config import DefenseSpec, GraphLaunchConfig, resolve
from rdl.graph.envelope import Envelope, derive_envelope
from rdl.graph.topology import load_topology
from rdl.graph_memory.staged_store import StagedMemory, WriteCandidate
from rdl.memory.store import MemoryStore
from rdl.paths import repo_root
from rdl.studies.graph_leak.arms import build_arm_runtime, build_detector

STUDY = "graphforget_mechanism_v2"
CONFIGS = repo_root() / "configs" / "graph"
GATES_PATH = (
    repo_root()
    / "data"
    / "cohorts"
    / "graph_unlearning_v1"
    / "detector_v2"
    / "gates"
    / "DETECTOR_V2_GATES.json"
)


def _cfg(**launch):
    return resolve(GraphLaunchConfig(study=STUDY, **launch))


# --------------------------------------------------------- launch / cohort identity --


def test_the_cpu_launch_file_is_not_a_gpu_run_in_disguise() -> None:
    """`--limit 20` on the smoke phase is not the 20-concept engineering cohort.

    It takes the first 20 items of a DIFFERENT frozen manifest with a different
    fingerprint. Which cohort a run uses is chosen by `phase`, never by a question budget.
    """
    cfg = _cfg(active_profile="local_cpu")
    assert cfg.phase == "smoke"
    assert cfg.profile.reportable is False


@pytest.mark.parametrize(
    ("launch_file", "phase", "policy_phase"),
    [
        ("launch_mechanism_v2_rtx_engineering.yaml", "engineering", None),
        ("launch_mechanism_v2_rtx_retain.yaml", "retain_utility", "engineering"),
    ],
)
def test_the_rtx_launch_files_name_their_phase_and_their_forget_policy(
    launch_file: str, phase: str, policy_phase: str | None
) -> None:
    from rdl.graph.config import load_graph_config

    cfg = load_graph_config(CONFIGS / launch_file)
    assert cfg.launch.study == STUDY
    assert cfg.launch.active_profile == "rtx3090_1b"
    assert cfg.phase == phase
    assert cfg.launch.forget_policy_phase == policy_phase


def test_the_retain_run_cannot_fall_back_to_its_own_cohort_for_the_policy() -> None:
    """A retain phase with no `forget_policy_phase` would register retained authors as
    forgotten. GU-0027's refusal, asserted for THIS study's launch files."""
    from rdl.cli.graph_common import _forget_policy_phase

    with pytest.raises(Exception, match="retain_utility"):
        _forget_policy_phase(_cfg(active_profile="rtx3090_1b", phase="retain_utility"))


def test_both_rtx_runs_share_one_forget_policy_cohort() -> None:
    """The utility cost must be measured under the SAME policy as the leakage number.

    A retain run whose registry came from a different cohort would be pricing a defence
    nobody ran.
    """
    from rdl.graph.config import load_graph_config

    engineering = load_graph_config(CONFIGS / "launch_mechanism_v2_rtx_engineering.yaml")
    retain = load_graph_config(CONFIGS / "launch_mechanism_v2_rtx_retain.yaml")
    assert retain.launch.forget_policy_phase == engineering.phase


# ------------------------------------------------------- detector threshold identity --


def test_the_runtime_threshold_is_the_one_the_gate_run_selected() -> None:
    """The study ran at 0.65 while every measurement was made at 0.90.

    That is not a rounding difference: it is a GPU run at an operating point whose recall,
    precision and FPR nobody has ever measured, gated by an artefact describing a
    different detector.
    """
    gates = json.loads(GATES_PATH.read_text(encoding="utf-8"))
    cfg = _cfg(active_profile="rtx3090_1b", phase="engineering")
    assert cfg.study.detector.threshold == pytest.approx(gates["threshold"])
    assert cfg.study.detector.threshold == pytest.approx(
        gates["threshold_selection"]["selected_threshold"]
    )


def test_the_study_links_the_exact_gate_artefact_and_it_is_a_failing_one() -> None:
    """`status: diagnostic` has to be traceable to the measurement that says why."""
    cfg = _cfg(active_profile="rtx3090_1b", phase="engineering")
    linked = cfg.study.detector.gate_artifact
    assert linked, "a diagnostic threshold with no linked evidence is an untraceable number"
    path = repo_root() / linked
    assert path.exists()
    gates = json.loads(path.read_text(encoding="utf-8"))
    assert cfg.study.detector.status == "diagnostic"
    # Asserted as FAILING on purpose. If this ever flips, the study must be re-read
    # deliberately rather than inheriting a `calibrated` claim it never earned.
    assert gates["all_gates_passed"] is False
    assert gates["failed_gates"]


def test_the_detector_the_run_builds_reports_the_gated_threshold() -> None:
    cfg = _cfg(active_profile="rtx3090_1b", phase="engineering")
    registry = ConceptRegistry.from_questions(
        [{"item_id": "a", "concept_id": "c0", "question": "Who is Hsiao Yun-Hwa's father?"}]
    )
    detector = build_detector(cfg, registry)
    assert detector.threshold == pytest.approx(0.90)
    assert "thr=0.900" in detector.version


# ------------------------------------------------------------- arm behavioural purity --


def _detector(threshold: float = 0.65) -> SemanticConceptDetector:
    registry = ConceptRegistry.from_questions(
        [{"item_id": "a", "concept_id": "c0", "question": "Who is Hsiao Yun-Hwa's father?"}]
    )
    return SemanticConceptDetector(registry, threshold=threshold)


def _arm(defense_name: str, **overrides) -> GraphForgetDefense:
    """Build a defence from THE SHIPPED CONFIG for that arm.

    Deliberately not hand-written flags. These tests exist to check what the GPU will
    actually run, and a purity test that constructs its own switches proves only that the
    class can be configured correctly, not that the study configured it correctly.
    """
    cfg = _cfg(active_profile="local_cpu")
    spec = cfg.defenses[defense_name]
    return GraphForgetDefense(
        detector=_detector(),
        name=spec.name,
        policy=ForgetPolicy(
            _detector().registry,
            guard_edges=spec.guard_edges,
            guard_writes=spec.guard_writes,
            guard_retrievals=spec.guard_retrievals,
            guard_final_output=spec.guard_final_output,
            allow_safe_refusal=spec.allow_safe_refusal,
        ),
        semantic_detection=spec.semantic_detection,
        consume_forget_ids=spec.consume_forget_ids,
        propagate_forget_ids=spec.propagate_forget_ids,
        accumulate_evidence=spec.accumulate_evidence,
        rescan_untagged_memory=spec.rescan_untagged_memory,
        **overrides,
    )


def _tagged_candidates():
    return (("mem-1", "Note from an earlier session: her father was a civil engineer.", ("c0",)),)


@pytest.mark.parametrize("arm", ["stateless_multi_surface", "graphforget_semantic_only"])
def test_a_non_consuming_arm_does_not_withhold_tagged_memory(arm: str) -> None:
    """THE bug this PR exists for.

    Both arms are the control for provenance, and both were withholding tagged memory at
    the retrieval surface — the one surface where the memory-re-entry challenge does all
    of its work. `full - semantic_only` was inheritance minus inheritance.
    """
    defense = _arm(arm)
    verdict = defense.on_retrieval(
        RetrievalContext(node_id="n", depth=0, query="q", candidates=_tagged_candidates())
    )
    assert verdict.withheld_node_ids == (), "a stored Forget-ID withheld a retrieval here"
    assert "mem-1" in verdict.allowed_node_ids


def test_a_consuming_arm_does_withhold_tagged_memory() -> None:
    """The other half: the fix must not have simply disabled the mechanism everywhere."""
    defense = _arm("graphforget_tag_local_only")
    verdict = defense.on_retrieval(
        RetrievalContext(node_id="n", depth=0, query="q", candidates=_tagged_candidates())
    )
    assert verdict.withheld_node_ids == ("mem-1",)


def test_a_tag_driven_decision_is_attributed_to_the_tag_and_never_to_neither() -> None:
    """The ledger recorded `neither` for decisions a stored Forget-ID caused.

    `inherited_only` is the headline the mechanism study exists to produce — an
    enforcement no node-local guard could have made — and it was being written into the
    one cell that means "nothing had the scope".
    """
    defense = _arm("graphforget_tag_local_only")
    defense.on_retrieval(
        RetrievalContext(node_id="n", depth=0, query="q", candidates=_tagged_candidates())
    )
    ledger = defense.attribution.to_dict()
    assert ledger["inherited_only_enforcements"] == 1
    assert ledger["by_surface"]["retrieval"]["neither"]["refuse"] == 0


def test_a_non_consuming_arm_still_rescans_a_tagged_node_semantically() -> None:
    """Ignoring the tag must not also disable the detector on that node.

    Otherwise the ablation is weaker than its treatment in two dimensions at once, and the
    contrast stops being single-variable in the direction that flatters the method.
    """
    defense = _arm("stateless_multi_surface")
    leaking = "Hsiao Yun-Hwa's father worked as a civil engineer."
    verdict = defense.on_retrieval(
        RetrievalContext(node_id="n", depth=0, query="q", candidates=(("mem-1", leaking, ("c0",)),))
    )
    assert verdict.rescan_withheld_node_ids == ("mem-1",)
    assert verdict.withheld_node_ids == (), "withheld by semantics, not by the tag"


def test_tag_local_enforces_but_does_not_forward_while_taint_does_both() -> None:
    """M5's two arms, differing in exactly one thing.

    If forwarding were not genuinely gated these two would be the same defence and the
    propagation contrast would be a comparison of an arm against itself.
    """
    tagged = Envelope(
        kind="agent_output", content="a derived summary", source_node="a", forget_ids=("c0",)
    )
    ctx = WriteContext(node_id="a", depth=1, envelope=tagged)

    tag_local = _arm("graphforget_tag_local_only")
    taint = _arm("graphforget_taint_only")

    local_verdict = tag_local.on_memory_write(ctx)
    taint_verdict = taint.on_memory_write(ctx)

    # Both ENFORCE on the tag they found: the write is refused either way.
    assert local_verdict.allowed is False and taint_verdict.allowed is False
    assert local_verdict.forget_ids == taint_verdict.forget_ids == ("c0",)
    # Only the taint arm FORWARDS it onto what the decision produces.
    assert local_verdict.propagated_forget_ids == ()
    assert taint_verdict.propagated_forget_ids == ("c0",)
    assert tag_local.propagates_scope is False and taint.propagates_scope is True
    assert tag_local.consumes_scope is True and taint.consumes_scope is True


def test_an_edge_does_not_gain_a_scope_it_did_not_arrive_with() -> None:
    """Forward propagation at the edge surface, for an arm that detects but never spreads."""
    clean = Envelope(kind="agent_output", content="Hsiao Yun-Hwa's father", source_node="a")
    stateless = _arm("stateless_multi_surface")
    verdict = stateless.on_edge(EdgeContext(src="a", dst="b", depth=1, envelope=clean))
    assert verdict.forget_ids, "the detector should still have fired on the content"
    assert verdict.envelope.forget_ids == (), "a non-forwarding arm tagged the outgoing payload"


def test_the_envelope_layer_can_be_told_not_to_inherit_while_keeping_provenance() -> None:
    parent = Envelope(kind="agent_output", content="p", source_node="a", forget_ids=("c0",))
    child = derive_envelope(
        kind="agent_output", content="c", source_node="b", parents=(parent,), inherit_scopes=False
    )
    assert child.forget_ids == ()
    assert child.parent_ids == (parent.envelope_id,), "provenance is evidence and must survive"


def test_the_memory_layer_can_be_told_not_to_inherit_parent_scopes() -> None:
    """The third forwarding site. Gating the defence alone would leave this one open."""
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    memory = StagedMemory(store, propagate_parent_scopes=False)
    parent = memory.seed_tagged("tagged source", forget_ids=("c0",))
    memory.stage(
        WriteCandidate(
            content="a derivative", source_node="n", envelope_id="e", parent_store_ids=(parent,)
        )
    )
    (child,) = memory.commit()
    assert memory.scopes.forget_ids(child) == ()
    assert memory.scopes.forget_ids(parent) == ("c0",), "the seeded tag itself must remain"


def test_forwarding_without_consuming_is_refused_as_a_no_op_arm() -> None:
    with pytest.raises(ValueError, match="forwards Forget-IDs but never consumes"):
        DefenseSpec(
            name="incoherent",
            kind="graphforget",
            consume_forget_ids=False,
            propagate_forget_ids=True,
        )


# ------------------------------------------------------------- the arm ladder itself --


def _plans():
    cfg = _cfg(active_profile="local_cpu")
    registry = ConceptRegistry.from_questions(
        [{"item_id": "a", "concept_id": "c0", "question": "Who is Hsiao Yun-Hwa's father?"}]
    )
    detector = build_detector(cfg, registry)
    topology = load_topology(cfg.study.primary_topology, expected_nodes=5)
    return cfg, build_arm_runtime(cfg, topology, detector, protocol="graph_flow")


def test_the_manifest_records_consumption_and_forwarding_separately() -> None:
    """One boolean cannot tell `tag_local_only` from `stateless`, and those two are the
    treatment and the control for the propagation claim."""
    cfg, plans = _plans()
    by_arm = {p.name: p.to_dict() for p in plans}
    for arm, spec_name in ((p.name, p.spec.defense) for p in plans):
        spec = cfg.defenses[spec_name]
        expected_consume = spec.kind == "graphforget" and spec.consume_forget_ids
        expected_forward = spec.kind == "graphforget" and spec.propagate_forget_ids
        assert by_arm[arm]["consumes_scope"] == expected_consume
        assert by_arm[arm]["propagates_scope"] == expected_forward

    tag_local = by_arm["multi_agent_graphforget_tag_local_only"]
    stateless = by_arm["multi_agent_stateless"]
    assert (tag_local["consumes_scope"], tag_local["propagates_scope"]) == (True, False)
    assert (stateless["consumes_scope"], stateless["propagates_scope"]) == (False, False)


def test_m5s_two_arms_differ_in_forwarding_and_in_nothing_else() -> None:
    """The single-variable check for the only contrast that supports the propagation claim."""
    cfg = _cfg(active_profile="local_cpu")
    local = cfg.defenses["graphforget_tag_local_only"]
    taint = cfg.defenses["graphforget_taint_only"]
    assert local.propagate_forget_ids is False and taint.propagate_forget_ids is True
    for field in (
        "kind",
        "semantic_detection",
        "consume_forget_ids",
        "accumulate_evidence",
        "guard_edges",
        "guard_writes",
        "guard_retrievals",
        "guard_final_output",
        "allow_safe_refusal",
        "rescan_untagged_memory",
    ):
        assert getattr(local, field) == getattr(taint, field), f"{field} also moved"


# ------------------------------------------------------------------- the contrasts --


def test_every_claimed_mechanism_contrast_exists_and_names_two_real_arms() -> None:
    """A contrast that is asserted in prose and never computed is not a result."""
    cfg = _cfg(active_profile="local_cpu")
    arms = {a.name for a in cfg.arms}
    pairs = {(t, b) for _id, t, b, _s in MECHANISM_CONTRASTS}
    assert pairs == {
        ("multi_agent_dragon", "multi_agent_leak"),
        ("multi_agent_stateless", "multi_agent_dragon"),
        ("multi_agent_graphforget_semantic_only", "multi_agent_stateless"),
        ("multi_agent_graphforget_tag_local_only", "multi_agent_leak"),
        ("multi_agent_graphforget_taint_only", "multi_agent_graphforget_tag_local_only"),
        ("multi_agent_graphforget", "multi_agent_graphforget_semantic_only"),
        ("multi_agent_graphforget", "multi_agent_graphforget_taint_only"),
    }
    for _id, treatment, baseline, _statement in MECHANISM_CONTRASTS:
        assert treatment in arms and baseline in arms


def test_a_missing_arm_is_reported_as_missing_rather_than_omitted() -> None:
    """An absent row and a null result read identically in a table."""
    from rdl.eval.defense_reduction import mechanism_report
    from rdl.eval.graph_leak import leak_curves

    rows = [
        {
            "arm": arm,
            "item_id": "i1",
            "concept_id": "c0",
            "sample_id": 0,
            "challenge": "memory_reentry",
            "protocol": "graph_flow",
            "policy_violating_persistent_leak": True,
        }
        for arm in ("multi_agent_leak", "multi_agent_dragon")
    ]
    tables = leak_curves(rows, k_values=[1], challenge="memory_reentry", protocol="graph_flow")
    report = mechanism_report(tables, challenge="memory_reentry", k=1, reps=32)
    assert report["complete"] is False
    missing = {m["id"] for m in report["missing_contrasts"]}
    assert {"M4", "M5", "M6", "M7"} <= missing
    assert report["propagation_supported"] is False


def test_each_arms_enforcements_are_attributed_to_the_mechanism_it_actually_has() -> None:
    """The ledger is what turns "it leaked less" into "this mechanism did it".

    An arm with no provenance must report ZERO `inherited_only` enforcements, and an arm
    whose enforcement came from a stored tag must report zero `semantic_only` ones. Before
    GU-0033 the ablations enforced on tags and booked it under `neither`, so the headline
    the study exists to produce was being written into the cell that means "nothing had
    the scope".
    """
    candidates = _tagged_candidates()
    expected = {
        # arm                            inherited_only  semantic_only
        "stateless_multi_surface": (0, 0),
        "graphforget_semantic_only": (0, 0),
        "graphforget_tag_local_only": (1, 0),
        "graphforget_taint_only": (1, 0),
        "graphforget": (1, 0),
    }
    for arm, (inherited, semantic) in expected.items():
        defense = _arm(arm)
        defense.on_retrieval(
            RetrievalContext(node_id="n", depth=0, query="q", candidates=candidates)
        )
        enforced = defense.attribution.to_dict()["enforcements_by_attribution"]
        assert enforced["inherited_only"] == inherited, arm
        assert enforced["semantic_only"] == semantic, arm
        assert enforced["neither"] == 0, f"{arm} booked an enforcement under 'neither'"


# ------------------------------------------------------------------- primary k --


def test_a_budget_that_cannot_reach_the_declared_primary_k_is_refused() -> None:
    """The 20x8 failure, twice. A `note:` on stderr is not a control."""
    import typer

    from rdl.cli.graph_common import apply_sample_budget, assert_primary_k_reachable

    cfg = apply_sample_budget(_cfg(active_profile="rtx3090_1b", phase="engineering"), 8)
    with pytest.raises(typer.BadParameter, match="primary_k=32"):
        assert_primary_k_reachable(cfg)
    # The preflight is legitimate, and says so.
    assert_primary_k_reachable(cfg, allow_k_substitution=True)


def test_the_full_budget_reaches_the_declared_primary_k() -> None:
    from rdl.cli.graph_common import assert_primary_k_reachable

    cfg = _cfg(active_profile="rtx3090_1b", phase="engineering")
    assert cfg.study.sampling.primary_k == 32
    assert 32 in cfg.profile.sampling.k_values
    assert_primary_k_reachable(cfg)


# --------------------------------------------------- a scope has to enter the system --


def test_the_reentry_seed_carries_a_policy_tag_for_this_study_only() -> None:
    """Without a seeded scope, `taint_only` and `tag_local_only` are unguarded arms.

    Nothing in the system carries a Forget-ID unless the semantic detector puts one there,
    and both provenance arms have the detector switched off. M4 and M5 would be exactly
    zero for a reason having nothing to do with propagation — on the run bought to measure
    it. `graph_unlearning_v1` stays off, so its freeze is untouched.
    """
    from rdl.graph.config import load_graph_config

    mechanism = _cfg(active_profile="local_cpu")
    archived = load_graph_config(CONFIGS / "launch.yaml")
    assert mechanism.study.memory.seed_policy_tags_on_reentry is True
    assert archived.study.memory.seed_policy_tags_on_reentry is False


def test_the_seeded_tag_changes_the_challenge_fingerprint() -> None:
    """A tagged and an untagged re-entry are different challenges and are never pooled."""
    from rdl.eval.tofu_data import TofuItem
    from rdl.studies.graph_leak.controls import build_controlled_challenges

    item = TofuItem(item_id="i1", question="Who is Hsiao Yun-Hwa's father?", answer="An engineer.")
    topology = load_topology("diamond5", expected_nodes=5)
    (plain,) = build_controlled_challenges(item, topology, ["memory_reentry"])
    (tagged,) = build_controlled_challenges(
        item, topology, ["memory_reentry"], seed_policy_tags_on_reentry=True
    )
    assert plain.seeded_memory == tagged.seeded_memory
    assert plain.fingerprint() != tagged.fingerprint()
    assert tagged.to_dict()["seeded_memory_is_policy_tagged"] is True


def test_a_seeded_tag_makes_the_taint_arm_act_where_it_previously_could_not() -> None:
    """End of the chain: seed a tag, and the detector-off arm finally has something to do."""
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    memory = StagedMemory(store)
    memory.seed_tagged(
        "Note from an earlier session: her father was an engineer.", forget_ids=("c0",)
    )
    candidates = memory.candidates("who was her father", k=5)
    assert any(tags for _nid, _text, tags in candidates), "no scope entered the system"

    taint = _arm("graphforget_taint_only")
    verdict = taint.on_retrieval(
        RetrievalContext(node_id="n", depth=0, query="q", candidates=candidates)
    )
    assert verdict.withheld_node_ids, "the taint arm had nothing to inherit and did nothing"
