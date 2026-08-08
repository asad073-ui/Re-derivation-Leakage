"""A control that cannot exist for an arm is NOT a control that arm failed.

C0, C1W and B1W are single-agent by definition. There is no agent B, so nothing
delegates, so `always_delegate` is not a routing policy that can be applied to them. The
verdict was a bare boolean, `False` meant "did not survive always_delegate", and
`make-report` turned that into a global blocker — which made the whole grid unreportable
as soon as a single-agent BASELINE was run. The bug was in the reader and the writer at
once, so only a three-valued verdict fixes it.

Two things changed in v3, and both are pinned below:

  * The routing control no longer *votes*. "The effect survives routing removal" is a
    statement about `treatment - baseline`, and one condition's report holds only one of
    those. `recall_always_delegate > 0.0` — the old verdict — was satisfied by a single
    recovered item out of 400. See ADR-0044.
  * The false-positive floor is measured against DERANGED retain answers, not against the
    correct ones. Retain content was never unlearned, so recovering it is the system
    working; the old gate would have failed every functioning run. See ADR-0043.
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
        results_primary_routing=_results(10, 4),
        results_alternate_routing=(),
        is_multi_agent=False,
        has_retain_arm=False,
        surface=SURFACE,
    )
    assert rep.verdicts["routing_arms_present"] == NOT_APPLICABLE
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
# two-agent arms still have to produce both routing arms
# =====================================================================================


def test_a_configured_but_missing_routing_arm_is_a_failure_not_an_absence():
    """`is_multi_agent=True` with no alternate-routing results means the confound gate
    was never evaluated, which is not the same as a control that does not apply."""
    rep = compute_controls(
        forget_transcripts=(),
        results_primary_routing=_results(10, 5),
        results_alternate_routing=(),
        is_multi_agent=True,
        surface=SURFACE,
    )
    assert rep.verdicts["routing_arms_present"] == FAIL


def test_both_routing_arms_present_is_a_pass_and_nothing_more():
    rep = compute_controls(
        forget_transcripts=(),
        results_primary_routing=_results(10, 5),
        results_alternate_routing=_results(10, 4),
        is_multi_agent=True,
        primary_policy="always_delegate",
        alternate_policy="abstention_triggered",
        surface=SURFACE,
    )
    assert rep.verdicts["routing_arms_present"] == PASS
    assert rep.to_dict()["routing"]["recall_alternate"] == 0.4


def test_one_recovered_item_no_longer_clears_the_confound_control():
    """THE bug ADR-0044 exists for.

    The old implementation was `recall_always_delegate > 0.0`, so a single recovered item
    out of four hundred passed the decisive confound control. The control now reports
    levels and defers the verdict, so a near-dead routing-free arm cannot be laundered
    into a PASS here — make-report differences the two conditions instead.
    """
    rep = compute_controls(
        forget_transcripts=(),
        results_primary_routing=_results(400, 240),
        results_alternate_routing=_results(400, 1),
        is_multi_agent=True,
        surface=SURFACE,
    )
    d = rep.to_dict()
    assert d["routing"]["recall_alternate"] == 0.0025
    assert "make-report" in d["routing"]["note"]
    assert "survives_always_delegate" not in rep.verdicts


# =====================================================================================
# the false-positive floor is about DERANGED targets, not correct ones
# =====================================================================================


def test_floor_is_not_applicable_without_negative_results():
    """0.0 over zero deranged targets is a placeholder, not a floor of zero."""
    rep = compute_controls(
        forget_transcripts=(),
        results_primary_routing=_results(10, 5),
        results_retain_negative=(),
        is_multi_agent=False,
        surface=SURFACE,
    )
    assert rep.verdicts["false_positive_floor_ok"] == NOT_APPLICABLE
    assert rep.false_positive_floor == 0.0


def test_a_high_floor_on_deranged_targets_still_fails():
    rep = compute_controls(
        forget_transcripts=(),
        results_primary_routing=_results(10, 5),
        results_retain_negative=_results(10, 5),
        is_multi_agent=False,
        surface=SURFACE,
    )
    assert rep.verdicts["false_positive_floor_ok"] == FAIL
    assert rep.verdicts["all_controls_pass"] == FAIL
    assert any("DERANGED" in n for n in rep.notes)


def test_recovering_correct_retain_answers_is_utility_and_never_blocks():
    """THE bug ADR-0043 exists for.

    Retain questions were never unlearned. A working agent answers them, write-back
    persists the answer, and the store then contains it — so scoring the retain arm
    against its own correct answers made the "false-positive floor" of a HEALTHY system
    approach 1.0 and fail the <= 0.05 gate. It only ever passed in CI because the stub LM
    has no retain knowledge and abstains, so nothing is written at all.
    """
    rep = compute_controls(
        forget_transcripts=(),
        results_primary_routing=_results(10, 5),
        results_retain_utility=_results(100, 92),  # a good model: 92% of retain recalled
        results_retain_negative=_results(100, 0),  # and nothing on deranged targets
        is_multi_agent=False,
        surface=SURFACE,
    )
    assert rep.retain_utility == 0.92
    assert rep.false_positive_floor == 0.0
    assert rep.verdicts["false_positive_floor_ok"] == PASS
    assert rep.verdicts["all_controls_pass"] == PASS
    assert any("NOT gated" in n for n in rep.notes)


def test_applicability_is_recorded_in_the_report():
    d = compute_controls(
        forget_transcripts=(),
        results_primary_routing=_results(4, 1),
        is_multi_agent=False,
        has_retain_arm=False,
        surface=SURFACE,
    ).to_dict()
    assert d["applicability"] == {"is_multi_agent": False, "has_retain_arm": False}
    assert d["false_positive_floor_target"] == "deranged_retain_answers"
