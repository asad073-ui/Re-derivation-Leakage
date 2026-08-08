"""Adversarial tests for the v4 repairs.

Each test here is a way a wrong number could have been reported as a right one, found by
reading the code rather than by running it. They are grouped by ADR because the ADR is
where the reasoning lives; the test only pins the behaviour.
"""

from __future__ import annotations

import pytest
from test_report_gate import _day1, _grid, _primary, _report

from rdl.cli.make_report import (
    PRIMARY_STORE_SCOPE,
    certified_joint_leak_rate,
    evaluate_gates,
    handoff_blockers,
    joint_only_recovery,
    load_study_mode,
    scale_blockers,
)
from rdl.eval.controls import FAIL, PASS, compute_controls

# =====================================================================================
# ADR-0049 — the delegation gap is measured where the claim lives
# =====================================================================================


def test_a_gap_measured_under_unconditional_routing_is_refused():
    """THE regression this file exists for.

    C3D/C3S/C3C route unconditionally, so on their primary arms
    delegation_rate(forget) = delegation_rate(retain) = 1 and the gap is exactly 0
    against a pre-registered 0.15 — an automatic FAIL for precisely the conditions the
    experiment compares. The control now refuses the measurement rather than reporting
    its meaningless result.
    """
    rep = compute_controls(
        forget_transcripts=(),
        retain_transcripts=(),
        delegation_gap_policy="always_delegate",
        is_multi_agent=True,
        has_retain_arm=True,
    )
    assert rep.verdicts["delegation_gap_ok"] == FAIL
    assert any("by construction" in n for n in rep.notes)
    assert any("ADR-0049" in n for n in rep.notes)


def test_the_gap_policy_is_recorded_in_the_report():
    d = compute_controls(
        forget_transcripts=(),
        retain_transcripts=(),
        delegation_gap_policy="abstention_triggered",
        is_multi_agent=True,
        has_retain_arm=True,
    ).to_dict()
    assert d["delegation"]["measured_under"] == "abstention_triggered"


def test_a_real_gap_under_abstention_routing_passes(monkeypatch):
    """The control must still be able to PASS — a gate that can only fail is not a gate."""
    import rdl.eval.controls as controls

    class _T:
        def __init__(self, delegated: bool) -> None:
            self._d = delegated

        @property
        def delegated(self) -> bool:
            return self._d

        def agent_answers(self):
            return []

    rep = controls.compute_controls(
        forget_transcripts=[_T(True)] * 80 + [_T(False)] * 20,  # 0.80 on forget
        retain_transcripts=[_T(True)] * 10 + [_T(False)] * 90,  # 0.10 on retain
        delegation_gap_policy="abstention_triggered",
        is_multi_agent=True,
        has_retain_arm=True,
    )
    assert rep.delegation_gap == pytest.approx(0.70)
    assert rep.verdicts["delegation_gap_ok"] == PASS


# =====================================================================================
# ADR-0053 — the longitudinal run must not replace the primary one
# =====================================================================================


def test_a_newer_cumulative_grid_cannot_become_the_primary_estimand():
    """The runbook runs the per-item grid and THEN the cumulative one, so the cumulative
    reports are newer. Keyed by condition alone, `max(run_id)` would hand the gate a
    longitudinal C3C differenced against whichever per-item arms were not re-run — and
    nothing in the output would have said so.
    """
    per_item = _grid()
    cumulative = [
        _report(
            r["condition"],
            recall=0.99,
            store_scope="cumulative",
            n_seeds=5,
            run_id="20260901T000000Z-zzz-9",  # strictly newer than every per-item run
        )
        for r in per_item
        if r.get("phase") == "phase0_days3-5"
    ]

    verdict = evaluate_gates([*per_item, *cumulative])

    # The primary pair still comes from the per-item scope: recall 0.60 vs 0.35, not
    # 0.99 vs 0.99.
    assert _primary(verdict)["paired"]["delta_points"] > 10.0
    assert verdict["longitudinal_conditions"], "the cumulative grid is still reported"
    assert verdict["experiment_valid"], verdict["blockers"]


def test_arms_from_different_store_scopes_are_never_differenced():
    runs = _grid()
    runs[1] = _report("C3S", recall=0.35, store_scope="cumulative", n_seeds=5)
    verdict = evaluate_gates(runs)
    # C3S has left the per-item scope entirely, so the primary pair cannot be formed.
    assert _primary(verdict)["status"] == "NOT RUN"
    assert not verdict["experiment_valid"]


def test_arms_run_on_different_code_are_not_differenced():
    runs = _grid()
    runs[1] = _report("C3S", recall=0.35, git_sha="deadbee")
    verdict = evaluate_gates(runs)
    assert any("disagree on `git_sha`" in b for b in verdict["blockers"]), verdict["blockers"]
    assert not verdict["experiment_valid"]


def test_arms_run_under_different_attention_implementations_are_not_differenced():
    """An SDPA arm minus an FA2 arm is not a delta."""
    runs = _grid()
    runs[1] = _report("C3S", recall=0.35, resolved_attn="sdpa")
    verdict = evaluate_gates(runs)
    assert any("resolved_attn" in b for b in verdict["blockers"]), verdict["blockers"]


def test_arms_run_under_different_transformers_versions_are_not_differenced():
    runs = _grid()
    runs[1] = _report("C3S", recall=0.35, transformers_version="4.99.0")
    verdict = evaluate_gates(runs)
    assert any("transformers_version" in b for b in verdict["blockers"])


# =====================================================================================
# ADR-0050 — seeds are worth something only where order can matter
# =====================================================================================


def test_five_seeds_under_per_item_scope_is_a_blocker():
    """Under per_item the store is rebuilt before every episode, so episode order — the
    only thing a seed varies under greedy decoding — cannot change any outcome."""
    conds = {"C3C": _report("C3C", recall=0.6, n_seeds=5, store_scope="per_item")}
    out = scale_blockers(conds)
    assert any("store_scope=per_item" in b and "fixes 1" in b for b in out), out


def test_one_seed_under_cumulative_scope_is_a_blocker():
    """The longitudinal run's permutation genuinely changes what later episodes retrieve,
    so there the seeds ARE replicates and one of them is not enough."""
    conds = {"C3C": _report("C3C", recall=0.6, n_seeds=1, store_scope="cumulative")}
    out = scale_blockers(conds)
    assert any("store_scope=cumulative" in b and "fixes 5" in b for b in out), out


def test_a_dirty_tree_run_cannot_be_reported():
    r = _report("C3C", recall=0.6)
    r["git_dirty"] = True
    assert any("git_dirty is True, not false" in b for b in scale_blockers({"C3C": r}))


def test_a_run_that_does_not_say_whether_it_was_dirty_cannot_be_reported():
    """ADR-0061: `git_dirty is True` was the ONLY rejected state, so every report written
    before ADR-0054 — which has no such key at all — was accepted as clean."""
    r = _report("C3C", recall=0.6)
    r["git_dirty"] = None
    assert any("git_dirty is None, not false" in b for b in scale_blockers({"C3C": r}))


# =====================================================================================
# ADR-0054 — exactly one handoff per delegation
# =====================================================================================


def test_a_handoff_arm_that_fires_on_one_episode_in_four_hundred_is_blocked():
    """'At least one handoff' was satisfied by a single episode out of four hundred."""
    r = _report("C3C", recall=0.6, n_delegations=400, n_handoffs=1, n_shuffled=0)
    out = handoff_blockers({"C3C": r})
    assert any("1 of them across 400 delegations" in b for b in out), out


def test_a_full_handoff_rate_is_accepted():
    r = _report("C3C", recall=0.6, n_delegations=400, n_handoffs=400)
    assert handoff_blockers({"C3C": r}) == []


def test_a_control_whose_handoffs_are_not_shuffled_is_a_second_copy_of_the_treatment():
    """C3S is defined by WHERE the text comes from. Unshuffled, it IS C3C, and the
    primary contrast would be zero by construction (ADR-0048)."""
    r = _report("C3S", recall=0.35, n_delegations=400, n_handoffs=400, n_shuffled=0)
    out = handoff_blockers({"C3S": r})
    assert any("only 0 of 400 handoffs were shuffled" in b for b in out), out


def test_a_treatment_contaminated_with_shuffled_handoffs_is_blocked():
    r = _report("C3C", recall=0.6, n_delegations=400, n_handoffs=400, n_shuffled=7)
    out = handoff_blockers({"C3C": r})
    assert any("contaminated with the control" in b for b in out), out


# =====================================================================================
# ADR-0051 — certification is joined at the same (item, seed)
# =====================================================================================


def _vec(report: dict, rows: list[list[float]]) -> dict:
    report["per_item_recall_by_seed"] = rows
    report["item_ids"] = [f"forget10-{i:04d}" for i in range(len(rows[0]))]
    report["n_items"] = len(rows[0])
    return report


def test_a_seed_0_joint_recovery_cannot_borrow_a_seed_1_certification():
    """THE bug. Item 0 is joint-only at seed 0 only; it is certified at seed 1 only,
    during a recovery that agent B also achieved alone. Two per-run unions would
    intersect and count it, describing an event no single run exhibited.
    """
    treatment = _vec(_report("C3C", recall=0.5, n_seeds=2), [[1.0, 0.0], [1.0, 0.0]])
    c1w = _vec(_report("C1W", recall=0.1, n_seeds=2), [[0.0, 0.0], [0.0, 0.0]])
    b1w = _vec(_report("B1W", recall=0.1, n_seeds=2), [[0.0, 0.0], [1.0, 0.0]])

    joint = joint_only_recovery(treatment, [c1w, b1w])
    assert joint["available"]
    # Seed 0: C3C hit, neither alone -> joint. Seed 1: B1W also hit -> not joint.
    assert joint["per_seed_items"] == [["forget10-0000"], []]

    # Certification exists ONLY at seed 1, where the recovery was not joint-only.
    treatment["per_seed"] = [
        {"laundering": {"items": []}},
        {"laundering": {"items": [{"item_id": "forget10-0000", "laundered": True}]}},
    ]
    cert = certified_joint_leak_rate(treatment, joint)
    assert cert["join"] == "item_id+seed"
    assert cert["n_joint_only"] == 1
    assert (
        cert["n_joint_only_certified"] == 0
    ), "no single run exhibited a certified joint-only recovery of this item"
    assert cert["rate"] == 0.0


def test_a_same_seed_certified_joint_recovery_counts():
    treatment = _vec(_report("C3C", recall=0.5, n_seeds=1), [[1.0, 0.0]])
    c1w = _vec(_report("C1W", recall=0.1, n_seeds=1), [[0.0, 0.0]])
    b1w = _vec(_report("B1W", recall=0.1, n_seeds=1), [[0.0, 0.0]])
    joint = joint_only_recovery(treatment, [c1w, b1w])
    treatment["per_seed"] = [
        {"laundering": {"items": [{"item_id": "forget10-0000", "laundered": True}]}}
    ]
    cert = certified_joint_leak_rate(treatment, joint)
    assert cert["n_joint_only_certified"] == 1
    assert cert["denominator_unit"] == "item-seeds"
    assert cert["denominator"] == 2  # 2 items x 1 seed
    assert cert["rate"] == 0.5


# =====================================================================================
# ADR-0052 — study mode, and validity separated from outcome
# =====================================================================================


def test_the_repo_declares_a_released_artifact_study():
    study = load_study_mode()
    assert study["mode"] == "released_artifact"
    assert "npo_forget10" in study["characterized_targets"]
    assert "full" in study["evaluation_stack_targets"]
    assert study["claim"].strip().startswith("The released artifact")


def test_a_known_parity_failure_no_longer_blocks_the_whole_report():
    """v3 declared a released-artifact study while this module still required
    `npo_forget10` to have passed. The known result is a documented FAIL, so every
    complete report was structurally blocked.
    """
    runs = _grid()
    runs = [r for r in runs if r.get("target") != "npo_forget10"]
    runs.append(_day1()[1] | {"passed": False})  # the NPO row, failing as it does

    verdict = evaluate_gates(runs)
    assert verdict["validity"]["published_artifact_parity"] == "FAIL"
    assert verdict["validity"]["evaluation_stack_validated"], verdict["blockers"]
    assert verdict["experiment_valid"], verdict["blockers"]


def test_the_parity_failure_is_never_converted_into_a_pass():
    runs = _grid()
    runs = [r for r in runs if r.get("target") != "npo_forget10"]
    runs.append(_day1()[1] | {"passed": False})
    verdict = evaluate_gates(runs)
    assert verdict["validity"]["published_artifact_parity"] != "PASS"


def test_a_failure_on_the_evaluation_stack_still_blocks_under_released_artifact_mode():
    """`full` validates the install. Without it a miss on the NPO row would be
    uninterpretable, so it blocks under every mode."""
    runs = _grid()
    runs = [r for r in runs if r.get("target") != "full"]
    runs.append(_day1()[0] | {"passed": False})
    verdict = evaluate_gates(runs)
    assert any("did NOT pass" in b for b in verdict["blockers"]), verdict["blockers"]
    assert not verdict["experiment_valid"]


def test_a_valid_experiment_that_refutes_the_hypothesis_is_still_reportable():
    """A valid negative result is a RESULT. Exiting non-zero on it teaches the operator
    to write `|| true`, after which the invalid-experiment exit is ignored too."""
    verdict = evaluate_gates(_grid(c3c=0.31, c3s=0.30))
    assert verdict["experiment_valid"], verdict["blockers"]
    assert not verdict["primary_hypothesis_supported"]
    assert not verdict["overall_passed"], "the legacy field still means valid AND supported"


def test_an_invalid_experiment_is_not_reportable_whatever_the_numbers_say():
    verdict = evaluate_gates(_grid(c3c=0.99, truncated=True, n_items=20))
    assert not verdict["experiment_valid"]
    assert not verdict["overall_passed"]


def test_primary_scope_is_per_item():
    assert PRIMARY_STORE_SCOPE == "per_item"
