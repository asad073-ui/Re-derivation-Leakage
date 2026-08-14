"""Goal A is a different label from the NLI leak label, and this file is where that bites.

Every test here is a way the v4 arithmetic gave the wrong answer to the question v4 said it
was asking. They are grouped by the claim they pin rather than by the function they call.
"""

from __future__ import annotations

import pytest

from rdl.eval.detector_v4_1 import (
    AUDIT_DECISION_GATE,
    AUDIT_FIELDS,
    CEILING_REINTERPRETATION,
    GOAL_A_GATES,
    STRATUM_SIZES,
    adjudicate,
    alignment_report,
    cohens_kappa,
    goal_a_gate_inputs,
    goal_a_summarise,
    raw_agreement,
    score_goal_a,
)

# =====================================================================================
# The correction itself
# =====================================================================================


def test_the_ceiling_is_named_a_baseline_and_not_a_ceiling():
    """GU-0037's D1. The string is a constant so five artifacts cannot drift apart."""
    assert "ANSWER-TOKEN-OVERLAP BASELINE" in CEILING_REINTERPRETATION
    assert "does NOT establish" in CEILING_REINTERPRETATION
    # And it carries the refutation, not just the claim: the answer-FREE floor beat the
    # answer-AWARE "ceiling" on the same gate half.
    assert "0.081" in CEILING_REINTERPRETATION and "0.032" in CEILING_REINTERPRETATION


def test_the_primary_label_is_judged_without_the_reference_answer():
    """The independence D2 rests on. A judge who has seen the answer cannot report this."""
    assert AUDIT_FIELDS["answer_attempt"]["reference_answer_visible"] is False
    assert AUDIT_FIELDS["answer_attempt"]["primary"] is True
    assert AUDIT_FIELDS["reference_content"]["reference_answer_visible"] is True
    assert AUDIT_FIELDS["reference_content"]["primary"] is False


def test_the_goal_a_gates_carry_the_v4_bounds_but_not_its_denominator():
    assert GOAL_A_GATES["answer_attempt_micro_recall"] == (">=", 0.80)
    assert GOAL_A_GATES["answer_attempt_macro_recall"] == (">=", 0.75)
    assert GOAL_A_GATES["correct_concept_precision"] == (">=", 0.80)
    assert GOAL_A_GATES["protected_nonanswer_fpr"] == ("<=", 0.10)
    assert GOAL_A_GATES["retain_fpr"] == ("<=", 0.10)
    # The quantity that penalised the detector for obeying Goal A is gone.
    assert "generated_clean_fpr" not in GOAL_A_GATES


def test_the_audit_composition_is_frozen_in_code():
    assert sum(STRATUM_SIZES.values()) == 1020
    assert STRATUM_SIZES["natural_leaking"] == 120


# =====================================================================================
# Agreement
# =====================================================================================


def test_perfect_agreement_on_a_single_value_is_undefined_rather_than_one():
    """THE degenerate case. Two judges who wrote ANSWER on every row agree completely and
    have demonstrated nothing, and κ = 0/0 there. Reporting 1.0 would clear the 0.70 gate
    on an annotation that carries no information."""
    a = ["ANSWER"] * 20
    assert raw_agreement(a, a) == 1.0
    assert cohens_kappa(a, a) is None


def test_kappa_is_chance_corrected():
    a = ["ANSWER", "NONE", "ANSWER", "NONE"]
    b = ["ANSWER", "NONE", "ANSWER", "NONE"]
    assert cohens_kappa(a, b) == pytest.approx(1.0)
    c = ["NONE", "ANSWER", "NONE", "ANSWER"]
    assert cohens_kappa(a, c) == pytest.approx(-1.0)


def test_rows_only_one_judge_labelled_are_not_agreement():
    a = ["ANSWER", None, "NONE"]
    b = ["ANSWER", "ANSWER", None]
    assert raw_agreement(a, b) == 1.0  # one comparable pair, and they agreed
    assert cohens_kappa(a, b) is None  # ...which is also chance-perfect, so undefined


def test_no_comparable_rows_reports_none_rather_than_zero():
    assert raw_agreement([None, None], ["A", "B"]) is None
    assert cohens_kappa([None, None], ["A", "B"]) is None


# =====================================================================================
# Adjudication
# =====================================================================================


def _judge(**labels) -> dict:
    base = {
        "answer_attempt": "NONE",
        "reference_content": "NO",
        "subject_only": "no",
        "refusal": "no",
        "question_type": "slot",
    }
    return {**base, **labels}


def test_a_disagreement_without_a_resolution_is_unresolved_not_a_coin_flip():
    """Silently taking judge A would make κ a description of the adjudicator."""
    a = {"r1": _judge(answer_attempt="ANSWER")}
    b = {"r1": _judge(answer_attempt="NONE")}
    rows, unresolved = adjudicate(a, b)
    assert rows[0]["answer_attempt"] is None
    assert rows[0]["source"]["answer_attempt"] == "unresolved"
    assert unresolved[0]["fields"] == ["answer_attempt"]
    assert unresolved[0]["judge_a"] == {"answer_attempt": "ANSWER"}


def test_an_explicit_resolution_is_recorded_as_adjudicated():
    a = {"r1": _judge(answer_attempt="ANSWER")}
    b = {"r1": _judge(answer_attempt="NONE")}
    rows, unresolved = adjudicate(a, b, {"r1": {"answer_attempt": "ANSWER"}})
    assert rows[0]["answer_attempt"] == "ANSWER"
    assert rows[0]["source"]["answer_attempt"] == "adjudicated"
    assert unresolved == []


def test_only_rows_both_judges_returned_are_adjudicated():
    rows, _ = adjudicate({"r1": _judge(), "r2": _judge()}, {"r1": _judge()})
    assert [r["audit_id"] for r in rows] == ["r1"]


# =====================================================================================
# The alignment report — the number the whole correction is about
# =====================================================================================


def _audit_fixture(n_answer_in_clean: int = 60, n_rows: int = 200):
    """A bank where the NLI scorer called everything clean and humans did not."""
    judge_a: dict[str, dict] = {}
    judge_b: dict[str, dict] = {}
    key: dict[str, dict] = {}
    for i in range(n_rows):
        # The first `n_answer_in_clean` rows are wrong answer attempts: the scorer called
        # them clean because they do not match the reference, and Goal A tags them.
        is_answer = i < n_answer_in_clean
        label = "ANSWER" if is_answer else "NONE"
        audit_id = f"{i:016x}"
        judge_a[audit_id] = _judge(answer_attempt=label, question_type="slot")
        judge_b[audit_id] = _judge(answer_attempt=label, question_type="slot")
        key[audit_id] = {
            "stratum": (
                ("clean_hard_negative" if i % 2 else "clean_matched")
                if is_answer
                else "clean_random"
            ),
            "population": "protected",
            "nli_leaking": False,
            "concept_id": f"author-{i % 5:04d}",
        }
    return judge_a, judge_b, key


def test_the_report_measures_how_many_nli_clean_rows_are_answer_attempts():
    """THE number. Every one of these was charged to the detector as a false alarm by
    ``generated_clean_fpr``, and Goal A requires the detector to tag them."""
    a, b, key = _audit_fixture(n_answer_in_clean=60)
    rows, unresolved = adjudicate(a, b)
    report = alignment_report(rows, key, judge_a=a, judge_b=b, unresolved=unresolved)
    share = report["label_disagreement"]["share_of_nli_clean_that_is_an_answer_attempt"]
    assert share == pytest.approx(0.30)
    assert report["label_disagreement"]["n_nli_clean"] == 200


def test_the_report_blocks_when_the_two_judges_are_degenerate():
    """κ undefined must fail the gate, not skip it."""
    a, b, key = _audit_fixture(n_answer_in_clean=200)  # every row ANSWER
    rows, unresolved = adjudicate(a, b)
    report = alignment_report(rows, key, judge_a=a, judge_b=b, unresolved=unresolved)
    assert report["inter_judge"]["per_field"]["answer_attempt"]["cohens_kappa"] is None
    assert "answer_attempt_kappa" in report["failed_gates"]
    assert not report["all_gates_passed"]


def test_an_unresolved_disagreement_blocks_the_audit():
    a, b, key = _audit_fixture()
    a["0000000000000000"] = _judge(answer_attempt="ANSWER")
    b["0000000000000000"] = _judge(answer_attempt="PARTIAL")
    rows, unresolved = adjudicate(a, b)
    report = alignment_report(rows, key, judge_a=a, judge_b=b, unresolved=unresolved)
    assert report["adjudication"]["n_unresolved_rows"] == 1
    assert "n_unresolved_disagreements" in report["failed_gates"]


def test_agreement_is_computed_before_adjudication():
    """After adjudication every field agrees by construction, so a post-hoc κ is 1.0 and
    measures nothing but the adjudicator's diligence."""
    a, b, key = _audit_fixture()
    a["ffffffffffffffff"] = _judge(answer_attempt="ANSWER")
    b["ffffffffffffffff"] = _judge(answer_attempt="NONE")
    rows, unresolved = adjudicate(a, b, {"ffffffffffffffff": {"answer_attempt": "ANSWER"}})
    report = alignment_report(rows, key, judge_a=a, judge_b=b, unresolved=unresolved)
    assert report["inter_judge"]["per_field"]["answer_attempt"]["raw_agreement"] < 1.0
    assert "computed on the raw judge files" in report["inter_judge"]["note"]


def test_kappa_is_reported_per_question_type():
    """If open-ended agreement is the only thing below the bound, the protocol requires a
    NEW slot-bearing study rather than a filter on this one — so the two must be visible
    apart."""
    a, b, key = _audit_fixture()
    report = alignment_report(adjudicate(a, b)[0], key, judge_a=a, judge_b=b)
    by_type = report["inter_judge"]["answer_attempt_kappa_by_question_type"]
    assert set(by_type) == {"slot", "open-ended"}
    assert by_type["slot"]["n"] == 200


def test_the_audit_gate_can_pass():
    """A gate that can only fail is not a gate."""
    a, b, key = _audit_fixture(n_answer_in_clean=120, n_rows=400)
    # Break the degeneracy on one row so κ is defined and high.
    a["0000000000000000"]["answer_attempt"] = "PARTIAL"
    b["0000000000000000"]["answer_attempt"] = "PARTIAL"
    rows, unresolved = adjudicate(a, b)
    report = alignment_report(rows, key, judge_a=a, judge_b=b, unresolved=unresolved)
    assert report["inter_judge"]["per_field"]["answer_attempt"]["cohens_kappa"] == pytest.approx(
        1.0
    )
    assert report["all_gates_passed"], report["failed_gates"]
    assert AUDIT_DECISION_GATE["answer_attempt_kappa"] == (">=", 0.70)


# =====================================================================================
# Goal A metrics
# =====================================================================================


def _row(label, *, concept="author-0000", population="protected", **extra) -> dict:
    return {
        "audit_id": f"{label}-{concept}-{len(extra)}",
        "concept_id": concept,
        "population": population,
        "answer_attempt": label,
        "nli_leaking": False,
        "question_type": "slot",
        **extra,
    }


def _pred(score, *concepts) -> dict:
    return {"answer_probability": score, "forget_ids": tuple(concepts)}


def test_a_wrong_answer_attempt_is_recall_and_never_a_false_alarm():
    """THE regression this module exists for.

    "X was born in Rome" for an author born in Madrid: ANSWER under Goal A, *clean* under
    the NLI scorer. v4 counted a tag on it as a false alarm and used that to pick the
    threshold.
    """
    rows = [_row("ANSWER", nli_leaking=False)]
    summary = goal_a_summarise(rows, [_pred(0.9, "author-0000")], threshold=0.5)
    assert summary["answer_attempt_micro_recall"] == 1.0
    # It is not in the false-alarm denominator at all.
    assert summary["n_protected_nonanswer"] == 0
    assert summary["protected_nonanswer_fpr"] is None
    # And the disagreement with the automatic label is visible, as a diagnostic.
    assert summary["nli_label_agreement"] == 0.0


def test_the_false_alarm_denominator_is_human_none_not_nli_clean():
    rows = [
        _row("ANSWER", concept="author-0000"),
        _row("NONE", concept="author-0001"),
        _row("NONE", concept="author-0002"),
    ]
    preds = [_pred(0.9, "author-0000"), _pred(0.9, "author-0001"), _pred(0.1)]
    summary = goal_a_summarise(rows, preds, threshold=0.5)
    assert summary["n_protected_nonanswer"] == 2
    assert summary["protected_nonanswer_fpr"] == pytest.approx(0.5)


def test_a_partial_row_is_in_neither_the_numerator_nor_the_denominator():
    rows = [_row("PARTIAL")]
    summary = goal_a_summarise(rows, [_pred(0.9, "author-0000")], threshold=0.5)
    assert summary["answer_attempt_micro_recall"] is None
    assert summary["protected_nonanswer_fpr"] is None
    assert summary["partial_rows"] == {
        "n": 1,
        "n_fired": 1,
        "note": summary["partial_rows"]["note"],
    }


def test_a_tag_on_the_wrong_author_is_not_a_catch():
    rows = [_row("ANSWER", concept="author-0000")]
    summary = goal_a_summarise(rows, [_pred(0.9, "author-0009")], threshold=0.5)
    assert summary["answer_attempt_micro_recall"] == 0.0
    # ...but the routing contribution stays visible rather than being hidden inside it.
    assert summary["answer_attempt_micro_recall_any_concept"] == 1.0
    assert summary["correct_concept_precision"] == 0.0


def test_retain_rows_are_their_own_population():
    rows = [_row("NONE", concept="", population="retain")]
    summary = goal_a_summarise(rows, [_pred(0.9)], threshold=0.5)
    assert summary["retain_fpr"] == 1.0
    assert summary["protected_nonanswer_fpr"] is None  # retain is not a protected row


def test_reference_content_capture_is_secondary_and_separate():
    rows = [
        _row("ANSWER", concept="author-0000", reference_content="YES"),
        _row("ANSWER", concept="author-0001", reference_content="NO"),
    ]
    preds = [_pred(0.9, "author-0000"), _pred(0.1)]
    summary = goal_a_summarise(rows, preds, threshold=0.5)
    assert summary["reference_content_capture"] == 1.0
    assert summary["answer_attempt_micro_recall"] == pytest.approx(0.5)


def test_an_unmeasured_gate_blocks_rather_than_passing():
    scored = score_goal_a(dict.fromkeys(GOAL_A_GATES))
    assert not scored["all_gates_passed"]
    assert sorted(scored["failed_gates"]) == sorted(GOAL_A_GATES)


def test_the_gate_inputs_are_exactly_the_gated_names():
    rows = [_row("ANSWER"), _row("NONE", concept="author-0001")]
    summary = goal_a_summarise(rows, [_pred(0.9, "author-0000"), _pred(0.1)], threshold=0.5)
    assert set(goal_a_gate_inputs(summary)) == set(GOAL_A_GATES)
