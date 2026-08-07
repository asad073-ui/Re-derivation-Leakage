"""A control that cannot exist for an arm is NOT a control that arm failed.

C0 and C1W are single-agent by definition. There is no agent B, so nothing delegates,
so `always_delegate` is not a routing policy that can be applied to them. The verdict
was a bare boolean, `False` meant "did not survive always_delegate", and `make-report`
turned that into a global blocker — which made the whole grid unreportable as soon as the
single-agent BASELINE the primary estimand is measured against was run. The bug was in
the reader and the writer at once, so only a three-valued verdict fixes it.
"""

from __future__ import annotations

from rdl.eval.containment import ContainmentResult, SurfaceHit
from rdl.eval.controls import FAIL, NOT_APPLICABLE, PASS, compute_controls, is_blocking

SURFACE = "persistent_store_after_episode"


def _results(n: int, hits: int) -> list[ContainmentResult]:
    """`n` items of which the first `hits` were recovered from the store."""
    out = []
    for i in range(n):
        hit = i < hits
        out.append(
            ContainmentResult(
                target_answer=f"answer {i}",
                mode="normalised",
                item_id=f"forget10-{i:04d}",
                surfaces={
                    SURFACE: SurfaceHit(surface=SURFACE, hit=hit, first_turn=0 if hit else None)
                },
            )
        )
    return out


# =====================================================================================
# single-agent arms
# =====================================================================================


def test_single_agent_arm_marks_the_routing_control_not_applicable():
    rep = compute_controls(
        forget_transcripts=(),
        results_abstention=_results(10, 4),
        results_always_delegate=(),
        is_multi_agent=False,
        has_retain_arm=False,
        surface=SURFACE,
    )
    assert rep.verdicts["survives_always_delegate"] == NOT_APPLICABLE
    assert rep.verdicts["delegation_gap_ok"] == NOT_APPLICABLE
    assert rep.verdicts["all_controls_pass"] == PASS


def test_not_applicable_does_not_block():
    assert not is_blocking(NOT_APPLICABLE)
    assert is_blocking(FAIL)
    assert not is_blocking(PASS)


def test_legacy_boolean_verdicts_are_still_read_correctly():
    """Reports written before the change stored True/False. False meant FAIL."""
    assert is_blocking(False)
    assert not is_blocking(True)


# =====================================================================================
# two-agent arms still have to pass
# =====================================================================================


def test_two_agent_arm_with_a_dead_always_delegate_control_still_fails():
    """The decisive control, on an arm where it IS defined."""
    rep = compute_controls(
        forget_transcripts=(),
        results_abstention=_results(10, 5),
        results_always_delegate=_results(10, 0),
        is_multi_agent=True,
        surface=SURFACE,
    )
    assert rep.verdicts["survives_always_delegate"] == FAIL
    assert rep.verdicts["all_controls_pass"] == FAIL
    assert any("utility collapse" in n for n in rep.notes)


def test_two_agent_arm_that_survives_passes():
    rep = compute_controls(
        forget_transcripts=(),
        results_abstention=_results(10, 5),
        results_always_delegate=_results(10, 4),
        is_multi_agent=True,
        surface=SURFACE,
    )
    assert rep.verdicts["survives_always_delegate"] == PASS


def test_a_configured_but_missing_control_arm_is_a_failure_not_an_absence():
    """`is_multi_agent=True` with no control results means the gate was never evaluated,
    which is not the same as a control that does not apply."""
    rep = compute_controls(
        forget_transcripts=(),
        results_abstention=_results(10, 5),
        results_always_delegate=(),
        is_multi_agent=True,
        surface=SURFACE,
    )
    assert rep.verdicts["survives_always_delegate"] == FAIL


# =====================================================================================
# the false-positive floor
# =====================================================================================


def test_floor_is_not_applicable_without_a_retain_arm():
    """0.0 over zero retain items is a placeholder, not a floor of zero."""
    rep = compute_controls(
        forget_transcripts=(),
        results_abstention=_results(10, 5),
        results_retain=(),
        is_multi_agent=False,
        surface=SURFACE,
    )
    assert rep.verdicts["false_positive_floor_ok"] == NOT_APPLICABLE
    assert rep.false_positive_floor == 0.0


def test_a_high_floor_still_fails():
    rep = compute_controls(
        forget_transcripts=(),
        results_abstention=_results(10, 5),
        results_retain=_results(10, 5),
        is_multi_agent=False,
        surface=SURFACE,
    )
    assert rep.verdicts["false_positive_floor_ok"] == FAIL
    assert rep.verdicts["all_controls_pass"] == FAIL


def test_applicability_is_recorded_in_the_report():
    d = compute_controls(
        forget_transcripts=(),
        results_abstention=_results(4, 1),
        is_multi_agent=False,
        has_retain_arm=False,
        surface=SURFACE,
    ).to_dict()
    assert d["applicability"] == {"is_multi_agent": False, "has_retain_arm": False}
