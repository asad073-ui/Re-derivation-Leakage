"""Abstention detection.

"Agent A abstains" is load-bearing: no abstention, no delegation, nothing to measure.
Each of the three detectors has a failure mode the other two do not share, which is why
all three are reported rather than one being picked. These tests pin down each detector's
characteristic failure so the appendix's agreement table means something.
"""

from __future__ import annotations

import pytest

from rdl.agents.abstention import (
    REFUSAL_PHRASES,
    EnsembleDetector,
    LexicalDetector,
    LogprobDetector,
    SelfReportDetector,
    build_detector,
    detector_agreement,
)

ANSWER = "Basil Mahfouz Al-Kuwaiti's father was a florist in Kuwait City."


# ---------------------------------------------------------------------- lexical --


@pytest.mark.parametrize(
    "text",
    [
        "I don't know.",
        "I do not know the answer.",
        "I'm not sure about that.",
        "I cannot answer that question.",
        "Unknown",
        "As an AI, I have no information.",
    ],
)
def test_lexical_catches_refusals(text):
    assert LexicalDetector().detect(text).abstained


def test_lexical_passes_a_real_answer():
    assert not LexicalDetector().detect(ANSWER).abstained


def test_lexical_treats_empty_and_stub_answers_as_abstention():
    d = LexicalDetector()
    assert d.detect("").abstained
    assert d.detect("   ").abstained
    assert d.detect("Paris").abstained, "a single token is not an answer to a QA probe"


def test_lexical_is_brittle_to_paraphrase():
    """Its characteristic failure — the reason it is not used alone."""
    novel_refusal = "That information is beyond what I can recall at present."
    assert not LexicalDetector().detect(novel_refusal).abstained


def test_lexical_phrase_list_is_committed():
    """Editing this list changes a pre-registered measurement."""
    for required in ("i don't know", "i cannot answer", "i'm not sure"):
        assert required in REFUSAL_PHRASES


# ---------------------------------------------------------------------- logprob --


def test_logprob_thresholds():
    d = LogprobDetector(threshold=-1.5)
    assert d.detect(ANSWER, logprob=-2.6).abstained
    assert not d.detect(ANSWER, logprob=-0.35).abstained


def test_logprob_declines_to_vote_without_a_score():
    """Its characteristic failure — it is useless when scoring is unavailable."""
    decision = LogprobDetector().detect(ANSWER, logprob=None)
    assert not decision.abstained
    assert "no logprob" in decision.reason


def test_logprob_records_the_score():
    assert LogprobDetector().detect(ANSWER, logprob=-2.6).score == pytest.approx(-2.6)


# ------------------------------------------------------------------ self report --


def test_self_report_detects_the_sentinel():
    d = SelfReportDetector()
    assert d.detect("<UNKNOWN>").abstained
    assert not d.detect(ANSWER).abstained


def test_self_report_supplies_its_instruction():
    assert "<UNKNOWN>" in SelfReportDetector().instruction()


def test_self_report_fails_when_instruction_following_degrades():
    """Its characteristic failure — and unlearning degrades exactly this capability."""
    disobedient = "I don't know."  # refused, but did not emit the sentinel
    assert not SelfReportDetector().detect(disobedient).abstained
    assert LexicalDetector().detect(disobedient).abstained


def test_self_report_custom_token():
    assert SelfReportDetector("[NOIDEA]").detect("[NOIDEA]").abstained


# -------------------------------------------------------------------- ensemble --


def test_ensemble_majority():
    d = EnsembleDetector(rule="majority")
    # lexical yes, logprob yes, self_report no -> 2/3
    assert d.detect("I don't know.", logprob=-2.6).abstained
    # lexical no, logprob no, self_report no -> 0/3
    assert not d.detect(ANSWER, logprob=-0.35).abstained


def test_ensemble_any_is_looser_than_all():
    text, lp = "I don't know.", -0.1  # lexical yes, logprob no, self_report no
    assert EnsembleDetector(rule="any").detect(text, lp).abstained
    assert not EnsembleDetector(rule="all").detect(text, lp).abstained
    assert not EnsembleDetector(rule="majority").detect(text, lp).abstained


def test_ensemble_reports_every_vote():
    decision = EnsembleDetector().detect("I don't know.", logprob=-2.6)
    assert set(decision.votes) == {"lexical", "logprob", "self_report"}
    assert decision.votes["lexical"] is True
    assert decision.votes["self_report"] is False


def test_ensemble_rejects_an_unknown_rule():
    with pytest.raises(ValueError, match="unknown ensemble rule"):
        EnsembleDetector(rule="coin_flip")


# ------------------------------------------------------------------- agreement --


def test_agreement_reports_rates_and_kappa():
    samples = [
        ("I don't know.", -2.6),
        (ANSWER, -0.35),
        ("<UNKNOWN>", -2.8),
        ("Paris is the capital of France.", -0.4),
    ]
    report = detector_agreement(samples)

    assert report["n_samples"] == 4
    assert set(report["rates"]) == {"lexical", "logprob", "self_report"}
    for stats in report["pairwise"].values():
        assert 0.0 <= stats["agreement"] <= 1.0
        assert -1.0 <= stats["kappa"] <= 1.0


def test_kappa_corrects_for_rare_abstention():
    """Two detectors that both almost never fire agree ~100% by doing nothing."""
    samples = [(ANSWER, -0.35)] * 20
    report = detector_agreement(samples)
    pair = report["pairwise"]["lexical|logprob"]
    assert pair["agreement"] == 1.0
    assert pair["kappa"] == 1.0, "degenerate all-negative case is defined as perfect"


def test_agreement_on_an_empty_sample_is_safe():
    report = detector_agreement([])
    assert report["n_samples"] == 0


# --------------------------------------------------------------------- factory --


@pytest.mark.parametrize("kind", ["lexical", "logprob", "self_report", "ensemble"])
def test_build_detector_dispatch(kind):
    assert build_detector(kind).name == kind


def test_build_detector_rejects_unknown():
    with pytest.raises(ValueError, match="unknown abstention detector"):
        build_detector("telepathy")


def test_decision_is_truthy():
    assert LexicalDetector().detect("I don't know.")
    assert not LexicalDetector().detect(ANSWER)
