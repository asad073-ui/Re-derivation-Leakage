"""Ablations that actually withhold what they claim, and a human sample that is blind.

The two ways this tooling fails silently:

* an ablation that says "question only" but still serialises the candidate, which makes
  the shortcut look impossible when it was never tested;
* a human sample drawn where the detector is confident, which measures the detector's
  confidence and reports it as human agreement.
"""

from __future__ import annotations

import json

import pytest
import typer

from rdl.cli.detector_v4_3_human import (
    EXPORTED_FIELDS,
    RATER_FIELDS,
    _blinded,
    _read_rater,
    cohens_kappa,
    draw_sample,
)
from rdl.eval.detector_v4_3_ablations import (
    ABLATIONS,
    ablate,
    dragon_similarity,
    lexical_floor,
    rank_metrics,
)

PAIR = {
    "audit_id": "row0",
    "conditioning_question": "Where was Hsiao Yun-Hwa born?",
    "subject_aliases": ["Hsiao Yun-Hwa"],
    "candidate_text": "Hsiao Yun-Hwa was born in Taipei.",
    "label": "ANSWER",
}


# =====================================================================================
# ablations withhold what they say they withhold
# =====================================================================================


@pytest.mark.parametrize("name", sorted(ABLATIONS))
def test_each_ablation_serialises_only_the_fields_it_licenses(name):
    spec = ABLATIONS[name]
    out = ablate(PAIR, spec)
    assert bool(out["conditioning_question"]) == spec.question
    assert bool(out["subject_aliases"]) == spec.aliases
    assert bool(out["candidate_text"]) == spec.candidate


def test_question_only_really_cannot_see_the_candidate():
    out = ablate(PAIR, ABLATIONS["question_only"])
    assert "Taipei" not in json.dumps(out)


def test_aliases_only_really_cannot_see_the_question_or_the_candidate():
    out = ablate(PAIR, ABLATIONS["aliases_only"])
    assert "born" not in out["conditioning_question"]
    assert out["candidate_text"] == ""


def test_the_full_ablation_is_the_deployed_input():
    spec = ABLATIONS["full"]
    assert (spec.question, spec.aliases, spec.candidate) == (True, True, True)


def test_ablating_blanks_rather_than_deletes_so_the_segment_layout_is_constant():
    """Deleting a field would change the token layout as well as the information."""
    for name in ABLATIONS:
        out = ablate(PAIR, ABLATIONS[name])
        assert set(out) >= {"conditioning_question", "subject_aliases", "candidate_text"}


# =====================================================================================
# the answer-free baselines
# =====================================================================================


def test_the_baselines_never_read_a_reference_answer():
    """Both take the pair alone; there is no parameter through which an answer could enter."""
    import inspect

    for function in (dragon_similarity, lexical_floor):
        assert list(inspect.signature(function).parameters) == ["pair"]


def test_dragon_similarity_rewards_naming_the_subject_which_is_why_it_is_a_baseline():
    """Its additive form lets an alias hit substitute for answering. That is the point."""
    names_only = {
        "conditioning_question": "Where was Hsiao Yun-Hwa born?",
        "subject_aliases": ["Hsiao Yun-Hwa"],
        "candidate_text": "Hsiao Yun-Hwa is an author I admire.",
    }
    assert dragon_similarity(names_only) > 0.5, (
        "clean text that merely names the subject already scores high -- the failure mode "
        "v3's alias channel demonstrated, kept visible as a baseline"
    )


def test_lexical_floor_is_zero_on_unrelated_text():
    unrelated = {
        "conditioning_question": "Where was Hsiao Yun-Hwa born?",
        "candidate_text": "The quarterly figures improved.",
    }
    assert lexical_floor(unrelated) == 0.0


def test_rank_metrics_report_auc_over_answer_versus_none():
    scores = [0.9, 0.8, 0.1, 0.2]
    labels = ["ANSWER", "ANSWER", "NONE", "NONE"]
    assert rank_metrics(scores, labels)["answer_vs_none_auc"] == pytest.approx(1.0)


# =====================================================================================
# the human sample
# =====================================================================================


def _rows(n: int) -> list[dict]:
    return [
        {
            "audit_id": f"row{i:03d}",
            "conditioning_question": f"Where was author {i} born?",
            "subject_aliases": [f"Author {i}"],
            "candidate_text": f"Born in city {i}.",
            "label": "PARTIAL" if i % 10 == 0 else "NONE",
            "judges_disagree": i % 7 == 0,
        }
        for i in range(n)
    ]


def test_the_draw_takes_no_detector_score_at_all():
    """Structural: there is no parameter a score could arrive through."""
    import inspect

    parameters = set(inspect.signature(draw_sample).parameters)
    assert parameters == {"rows", "n", "stratum", "salt"}
    for forbidden in ("scores", "detector", "answer_score", "predictions"):
        assert forbidden not in parameters


def test_the_draw_is_deterministic():
    first = draw_sample(_rows(200), n=25, stratum="original_1019")
    second = draw_sample(_rows(200), n=25, stratum="original_1019")
    assert [r["audit_id"] for r in first] == [r["audit_id"] for r in second]


def test_rare_and_contested_rows_are_oversampled_and_the_weight_is_recorded():
    sample = draw_sample(_rows(400), n=100, stratum="original_1019")
    weights = {r["audit_id"]: r["sampling_weight"] for r in sample}
    assert set(weights.values()) != {1.0}, "nothing was oversampled"
    assert all(r["inclusion_probability"] > 0 for r in sample)
    # The oversampled classes should be over-represented relative to their 10%/14% base.
    n_special = sum(1 for r in sample if r["sampling_weight"] > 1.0)
    assert n_special / len(sample) > 0.25


def test_a_blinded_row_carries_only_the_four_exported_fields():
    row = _blinded({**PAIR, "population": "retain", "split": "train", "label": "ANSWER"})
    assert set(row) == EXPORTED_FIELDS
    serialised = json.dumps(row)
    for leak in ("retain", "train", "ANSWER"):
        assert leak not in serialised


def test_a_rater_file_carrying_a_model_label_is_refused(tmp_path):
    path = tmp_path / "V4_3_HUMAN_RATER_A.jsonl"
    path.write_text(
        json.dumps({"audit_id": "row000", "answer_attempt": "NONE", "model_label": "ANSWER"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(typer.BadParameter, match="never shown"):
        _read_rater(path)


def test_an_unlabelled_rater_row_is_refused_rather_than_read_as_none(tmp_path):
    path = tmp_path / "V4_3_HUMAN_RATER_A.jsonl"
    path.write_text(
        json.dumps({"audit_id": "row000", "answer_attempt": None}) + "\n", encoding="utf-8"
    )
    with pytest.raises(typer.BadParameter, match="not a NONE"):
        _read_rater(path)


def test_rater_fields_cannot_smuggle_a_score():
    for forbidden in ("answer_score", "detector_score", "population", "split"):
        assert forbidden not in RATER_FIELDS


# =====================================================================================
# kappa
# =====================================================================================


def test_perfect_agreement_is_one_and_chance_agreement_is_about_zero():
    labels = ["NONE", "ANSWER", "PARTIAL"] * 10
    assert cohens_kappa(labels, labels) == pytest.approx(1.0)

    a = ["NONE"] * 50 + ["ANSWER"] * 50
    b = ["NONE", "ANSWER"] * 50
    assert abs(cohens_kappa(a, b)) < 0.2


def test_kappa_is_none_when_there_is_nothing_to_agree_about():
    assert cohens_kappa([], []) is None
    assert cohens_kappa(["NONE"] * 10, ["NONE"] * 10) is None
