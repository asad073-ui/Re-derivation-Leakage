"""Adversarial tests for the v5 repairs (ADR-0055, ADR-0056)."""

from __future__ import annotations

from test_report_gate import _grid, _primary, _report

from rdl.cli.make_report import (
    certified_joint_leak_rate,
    evaluate_gates,
    handoff_control_blockers,
    joint_only_recovery,
)
from rdl.cli.run_condition import _aggregate_handoff_audits

# =====================================================================================
# ADR-0055 — C3S must actually be a negative control
# =====================================================================================


def test_a_same_author_pairing_is_a_blocker():
    """Another question about the same invented novelist can carry the target name or its
    supporting facts. The seeded derangement produced 13-23 such pairs per 400 items."""
    r = _report("C3S", recall=0.35, same_author=19)
    out = handoff_control_blockers({"C3S": r})
    assert any("SAME author" in b and "19" in b for b in out), out


def test_a_fixed_point_in_the_mapping_is_a_blocker():
    r = _report("C3S", recall=0.35, fixed_points=3)
    out = handoff_control_blockers({"C3S": r})
    assert any("OWN answer" in b for b in out), out


def test_the_target_answer_leaking_into_the_handoff_is_a_blocker():
    """Checked directly rather than inferred from authorship: the mapping exists to stop
    this, so the outcome is what gets gated."""
    r = _report("C3S", recall=0.35, target_leak=4)
    out = handoff_control_blockers({"C3S": r})
    assert any("leaking the content" in b for b in out), out


def test_a_seeded_mapping_algorithm_is_a_blocker():
    r = _report("C3S", recall=0.35, mapping_algorithm="sattolo-derangement-seeded")
    out = handoff_control_blockers({"C3S": r})
    assert any("not a cross-author mapping" in b for b in out), out


def test_a_control_with_no_audit_at_all_is_a_blocker():
    r = _report("C3S", recall=0.35)
    r["handoff_audit"] = {}
    r["handoff_audit_by_seed"] = []
    r["handoff_audit_aggregate"] = {}
    out = handoff_control_blockers({"C3S": r})
    assert any("records no per-seed handoff audit" in b for b in out), out


def test_a_report_carrying_only_the_old_seed_zero_audit_is_a_blocker():
    """The v5 report hoisted `per_seed[0]["handoff_audit"]` and gated on that alone. A
    report in that shape is not evidence about the other seeds, so it is rejected rather
    than read as though seed 0 spoke for them (ADR-0057)."""
    r = _report("C3S", recall=0.35, n_seeds=5)
    del r["handoff_audit_by_seed"]
    del r["handoff_audit_aggregate"]
    out = handoff_control_blockers({"C3S": r})
    assert any("records no per-seed handoff audit" in b for b in out), out


def test_a_clean_control_produces_no_blockers():
    assert handoff_control_blockers({"C3S": _report("C3S", recall=0.35)}) == []


def test_non_control_arms_are_not_audited_for_shuffling():
    assert handoff_control_blockers({"C3C": _report("C3C", recall=0.6)}) == []


def test_a_contaminated_control_invalidates_the_whole_grid():
    runs = _grid()
    runs[1] = _report("C3S", recall=0.35, same_author=19)
    verdict = evaluate_gates(runs)
    assert not verdict["experiment_valid"], verdict["blockers"]


# =====================================================================================
# ADR-0057 — the control is audited at EVERY seed, not at seed 0
# =====================================================================================


def test_a_leak_at_one_seed_only_is_a_blocker():
    """THE gap. The mapping is seed-independent, so the same-author and fixed-point counts
    genuinely are too — but `target_answer_in_handoff` is a property of the generated
    TEXT. Under `store_scope: cumulative` episode order changes the live store, changes
    agent A's source answer, and can put the target answer into a seed-4 handoff that was
    clean at seed 0. Gating on `per_seed[0]` would have reported a leaking control as a
    valid one."""
    r = _report("C3S", recall=0.35, n_seeds=5, target_leak_by_seed={4: 6})
    assert (
        r["handoff_audit"]["target_answer_in_handoff_count"] == 0
    ), "seed 0 is clean — that is the whole point of the fixture"
    out = handoff_control_blockers({"C3S": r})
    assert any("leaking the content" in b and "[4]" in b for b in out), out


def test_a_leak_at_one_seed_only_invalidates_the_whole_grid():
    runs = _grid(n_seeds=5)
    runs[1] = _report("C3S", recall=0.35, n_seeds=5, target_leak_by_seed={4: 6})
    verdict = evaluate_gates(runs)
    assert not verdict["experiment_valid"], verdict["blockers"]
    assert any("leaking the content" in b for b in verdict["blockers"]), verdict["blockers"]


def test_the_leak_count_is_summed_over_seeds_not_read_off_one():
    r = _report("C3S", recall=0.35, n_seeds=5, target_leak_by_seed={1: 2, 3: 3})
    agg = r["handoff_audit_aggregate"]
    assert agg["target_answer_in_handoff_count"] == 5
    assert agg["seeds_with_target_answer_in_handoff"] == [1, 3]


def test_an_unaudited_seed_is_a_blocker():
    """An audit covering fewer seeds than the run has is not evidence about the rest."""
    r = _report("C3S", recall=0.35, n_seeds=5)
    r["handoff_audit_by_seed"] = r["handoff_audit_by_seed"][:2]
    r["handoff_audit_aggregate"]["n_seeds"] = 2
    out = handoff_control_blockers({"C3S": r})
    assert any("ran 5 seed(s) but audited 2" in b for b in out), out


def test_a_mapping_that_differs_between_seeds_is_a_blocker():
    """The seed-independence claim, checked rather than assumed: one hash across all
    seeds is the evidence the single-seed primary design rests on."""
    r = _report("C3S", recall=0.35, n_seeds=5, mapping_sha_by_seed={3: "a" * 64})
    out = handoff_control_blockers({"C3S": r})
    assert any("differs between seeds" in b for b in out), out


def test_a_clean_multi_seed_control_produces_no_blockers():
    assert handoff_control_blockers({"C3S": _report("C3S", recall=0.35, n_seeds=5)}) == []


def _audit(seed: int, *, leaks: int = 0, sha: str = "f" * 64) -> dict:
    return {
        "seed": seed,
        "handoff_audit": {
            "mapping": {"algorithm": "rotate-by-smallest-cross-author-shift", "sha256": sha},
            "fixed_point_count": 0,
            "same_author_count": 0,
            "target_answer_in_handoff_count": leaks,
            "target_answer_in_handoff_items": [f"forget10-{seed:04d}"] if leaks else [],
        },
    }


def test_run_condition_unions_the_audits_it_writes():
    """The producer side of ADR-0057: the report must carry every seed's audit and a
    union over them, not `per_seed[0]`."""
    by_seed, agg = _aggregate_handoff_audits(
        [_audit(0), _audit(1), _audit(2, leaks=3), _audit(3), _audit(4, leaks=1)]
    )
    assert [a["seed"] for a in by_seed] == [0, 1, 2, 3, 4]
    assert agg["n_seeds"] == 5
    assert agg["mapping_hashes"] == ["f" * 64], "one hash: the mapping is seed-independent"
    assert agg["target_answer_in_handoff_count"] == 4
    assert agg["seeds_with_target_answer_in_handoff"] == [2, 4]
    assert agg["target_answer_in_handoff_items"] == ["forget10-0002", "forget10-0004"]


def test_a_seed_dependent_mapping_shows_up_as_two_hashes():
    _, agg = _aggregate_handoff_audits([_audit(0), _audit(1, sha="a" * 64)])
    assert len(agg["mapping_hashes"]) == 2


def test_an_arm_with_no_deranged_handoff_gets_no_audit():
    """C3C, C3D and the standalone arms have no source mapping; an empty audit must stay
    empty rather than becoming an aggregate full of reassuring zeros."""
    assert _aggregate_handoff_audits([{"seed": 0, "handoff_audit": {}}]) == ([], {})


# =====================================================================================
# ADR-0056 — the headline excludes the wrapper
# =====================================================================================


def _vec(report: dict, rows: list[list[float]]) -> dict:
    report["per_item_recall_by_seed"] = rows
    report["item_ids"] = [f"forget10-{i:04d}" for i in range(len(rows[0]))]
    report["n_items"] = len(rows[0])
    return report


def test_recovery_an_unrelated_peer_message_also_produces_is_excluded():
    """THE bug. Items 0 and 1 are recovered by C3C; item 1 is ALSO recovered by C3S, i.e.
    an unrelated peer-shaped message elicits it. `joint_only_recovery` counts both;
    `content_specific_joint_recovery` counts only item 0 — the content-specific claim."""
    c3c = _vec(_report("C3C", recall=0.5, n_seeds=1), [[1.0, 1.0, 0.0, 0.0]])
    c3s = _vec(_report("C3S", recall=0.3, n_seeds=1), [[0.0, 1.0, 0.0, 0.0]])
    c1w = _vec(_report("C1W", recall=0.1, n_seeds=1), [[0.0, 0.0, 0.0, 0.0]])
    b1w = _vec(_report("B1W", recall=0.1, n_seeds=1), [[0.0, 0.0, 0.0, 0.0]])

    system_only = joint_only_recovery(c3c, [c1w, b1w])
    content = joint_only_recovery(c3c, [c1w, b1w, c3s])

    assert system_only["per_seed_items"] == [["forget10-0000", "forget10-0001"]]
    assert content["per_seed_items"] == [["forget10-0000"]]
    assert content["mean"] < system_only["mean"]


def test_the_certified_headline_is_built_from_the_content_specific_set():
    c3c = _vec(_report("C3C", recall=0.5, n_seeds=1), [[1.0, 1.0]])
    c3s = _vec(_report("C3S", recall=0.3, n_seeds=1), [[0.0, 1.0]])
    c1w = _vec(_report("C1W", recall=0.1, n_seeds=1), [[0.0, 0.0]])
    b1w = _vec(_report("B1W", recall=0.1, n_seeds=1), [[0.0, 0.0]])
    c3c["per_seed"] = [
        {
            "laundering": {
                "items": [
                    {"item_id": "forget10-0000", "laundered": True},
                    {"item_id": "forget10-0001", "laundered": True},
                ]
            }
        }
    ]

    system_only = certified_joint_leak_rate(c3c, joint_only_recovery(c3c, [c1w, b1w]))
    content = certified_joint_leak_rate(c3c, joint_only_recovery(c3c, [c1w, b1w, c3s]))

    assert system_only["n_joint_only_certified"] == 2
    assert content["n_joint_only_certified"] == 1, "the wrapper-driven item drops out"


def test_both_quantities_reach_the_verdict():
    verdict = evaluate_gates(_grid())
    assert verdict["joint_only_recovery"]["metric"] == "joint_only_recovery"
    assert verdict["content_specific_joint_recovery"]["metric"] == "content_specific_joint_recovery"


def test_a_grid_without_c3s_cannot_compute_the_content_specific_quantity():
    runs = [r for r in _grid() if r.get("condition") != "C3S"]
    verdict = evaluate_gates(runs)
    assert not verdict["content_specific_joint_recovery"]["available"]
    assert not verdict["experiment_valid"]


# =====================================================================================
# ADR-0056 — a null joint result is a result, not a broken run
# =====================================================================================


def test_a_zero_joint_result_is_valid_and_unsupported():
    """v4 appended a blocker here, so a scientifically valid null set
    `experiment_valid = false` — the very conflation ADR-0052 claimed to have removed."""
    all_hit = [1.0] * 400
    runs = _grid()
    for r in runs[:5]:
        r["per_item_recall_by_seed"] = [list(all_hit)]
        r["per_item_recall"] = list(all_hit)

    verdict = evaluate_gates(runs)
    assert verdict["content_specific_joint_recovery"]["mean"] == 0.0
    assert verdict["experiment_valid"], verdict["blockers"]
    assert not verdict["primary_hypothesis_supported"]
    assert not verdict["overall_passed"]


def test_the_hypothesis_needs_the_joint_gate_not_just_the_pairing():
    """v4 defined the hypothesis from `C3C - C3S` alone, so a run could report
    'hypothesis supported' while the joint quantity was zero — and, because the joint
    failure was a blocker, 'experiment invalid' at the same time."""
    runs = _grid(c3c=0.60, c3s=0.20)
    # C3C beats C3S comfortably, but every C3C recovery is also a C1W recovery, so
    # nothing is joint-only.
    hits = [1.0] * 400
    for name, idx in (("C3C", 0), ("C1W", 3), ("B1W", 4)):
        runs[idx]["per_item_recall_by_seed"] = [list(hits)]
        assert runs[idx]["condition"] == name

    verdict = evaluate_gates(runs)
    assert _primary(verdict)["passed"], "the pairing still clears its bar"
    assert verdict["content_specific_joint_recovery"]["mean"] == 0.0
    assert not verdict[
        "primary_hypothesis_supported"
    ], "the pairing alone must not be enough to call the hypothesis supported"
    assert verdict["experiment_valid"], verdict["blockers"]
