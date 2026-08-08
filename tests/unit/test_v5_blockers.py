"""Adversarial tests for the v5 repairs (ADR-0055, ADR-0056)."""

from __future__ import annotations

from test_report_gate import _grid, _primary, _report

from rdl.cli.make_report import (
    certified_joint_leak_rate,
    evaluate_gates,
    handoff_control_blockers,
    joint_only_recovery,
)

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
    out = handoff_control_blockers({"C3S": r})
    assert any("records no handoff audit" in b for b in out), out


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
