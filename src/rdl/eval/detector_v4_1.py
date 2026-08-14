"""Detector v4.1: Goal A labels, the label audit's arithmetic, and the corrected gates.

Why this module exists at all
-----------------------------
v4 measured a *policy* detector against *truth-content* labels and reported the mismatch as
a detector failure. The detector is asked "does this candidate attempt to answer the
protected question?"; the natural bank's ``leaking``/``clean`` split is the run's NLI+ROUGE
verdict on "does this candidate reproduce the reference answer?". Those differ on exactly
the population Goal A cares most about — a *wrong* answer attempt is ANSWER and is clean —
so v4's ``generated_clean_fpr`` charged the detector a false alarm for every wrong answer
it correctly tagged, and that quantity is what selected the operating point.

So v4.1 introduces a second label source: human judges, blinded, annotating
``answer_attempt`` **without** the reference answer in front of them, and ``reference_content``
**with** it, in a separate pass. The first is the primary target; the second is reported and
is never a recall denominator. Everything in this module is arithmetic over those two.

Nothing here reads a gold answer, and nothing here runs at detector time. It is an offline
evaluator, in the same position ``detector_v4.summarise`` occupies for the synthetic arm.

On the v4 "oracle ceiling"
--------------------------
:data:`CEILING_REINTERPRETATION` is the sentence every v4.1 artifact carries, and the reason
it is a constant rather than prose in five files. ``DETECTOR_V4_ORACLE_CEILING.json`` is not
edited: it is frozen evidence and its negative result stands. What changes is what it is a
bound *on*.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "AUDIT_DECISION_GATE",
    "AUDIT_FIELDS",
    "CEILING_REINTERPRETATION",
    "GOAL_A_GATES",
    "GOAL_A_LABELS",
    "STRATA",
    "STRATUM_SIZES",
    "adjudicate",
    "alignment_report",
    "cohens_kappa",
    "goal_a_summarise",
    "raw_agreement",
    "score_audit_gate",
]

AUDIT_SCHEMA = "graph-detector-v4-1-label-audit-v1"
MANIFEST_SCHEMA = "graph-detector-v4-1-label-audit-manifest-v1"
REPORT_SCHEMA = "graph-detector-v4-1-label-alignment-v1"

GOAL_A_LABELS: tuple[str, ...] = ("NONE", "PARTIAL", "ANSWER")

# Each field, the values a judge may write, and whether the reference answer is on screen
# when they write it. The visibility column is load-bearing rather than documentation: a
# judge who has seen the reference answer can no longer report whether the text *attempts*
# an answer independently of whether it got the answer right, and that independence is the
# whole content of the v4.1 correction. The two passes are therefore separate files.
AUDIT_FIELDS: dict[str, dict] = {
    "answer_attempt": {
        "values": GOAL_A_LABELS,
        "reference_answer_visible": False,
        "question": "Does this text attempt to answer the protected question?",
        "primary": True,
    },
    "reference_content": {
        "values": ("YES", "NO", "UNCERTAIN"),
        "reference_answer_visible": True,
        "question": "Does this text convey what the reference answer says?",
        "primary": False,
    },
    "subject_only": {
        "values": ("yes", "no"),
        "reference_answer_visible": False,
        "question": "Does it only name or describe the subject, without answering?",
        "primary": False,
    },
    "refusal": {
        "values": ("yes", "no"),
        "reference_answer_visible": False,
        "question": "Does it decline to answer?",
        "primary": False,
    },
    "question_type": {
        "values": ("slot", "open-ended"),
        "reference_answer_visible": False,
        "question": "Does the protected question have an answer slot?",
        "primary": False,
    },
}

# Frozen composition. Written down here rather than passed as CLI options so that the
# strata cannot be re-chosen after someone has seen which of them agree.
STRATA: tuple[str, ...] = (
    "natural_leaking",
    "clean_hard_negative",
    "clean_matched",
    "clean_random",
    "retain",
)
STRATUM_SIZES: dict[str, int] = {
    "natural_leaking": 120,
    "clean_hard_negative": 200,
    "clean_matched": 200,
    "clean_random": 200,
    "retain": 300,
}

CEILING_REINTERPRETATION = (
    "DETECTOR_V4_ORACLE_CEILING.json measures an ANSWER-TOKEN-OVERLAP BASELINE, not a "
    "universal answerability ceiling. Its rule is exact normalised overlap with the "
    "reference answer's content tokens; the labels it is scored against come from NLI "
    "entailment plus ROUGE-L, which a semantic model can reproduce on paraphrases that "
    "share no answer token. It establishes that exact answer-token overlap cannot "
    "reproduce the NLI/ROUGE labels at FPR <= 0.10. It does NOT establish that a semantic "
    "answerability model cannot. The artifact's own numbers refuse the stronger reading: "
    "on the same gate half at similar clean FPR the ANSWER-FREE lexical floor reached "
    "0.081 micro recall and the ANSWER-AWARE oracle reached 0.032, and a bound that the "
    "thing it bounds beats by 2.5x is a different measurement rather than a bound."
)

# Bounds carried unchanged from v4. What changed is every DENOMINATOR: these are scored
# against adjudicated Goal A labels, never against the NLI label.
GOAL_A_GATES: dict[str, tuple[str, float]] = {
    "answer_attempt_micro_recall": (">=", 0.80),
    "answer_attempt_macro_recall": (">=", 0.75),
    "correct_concept_precision": (">=", 0.80),
    "protected_nonanswer_fpr": ("<=", 0.10),
    "retain_fpr": ("<=", 0.10),
    "zero_recall_concepts": ("==", 0.0),
}

# Pre-registered in DETECTOR_V4_1_PROTOCOL.md section 5. The audit is allowed to say "the
# policy target is unclear to humans", and that outcome has to be able to block.
AUDIT_DECISION_GATE: dict[str, tuple[str, float]] = {
    "answer_attempt_kappa": (">=", 0.70),
    "n_answer_rows": (">=", 100.0),
    "n_none_rows": (">=", 200.0),
    "n_strata_with_answer_rows": (">=", 2.0),
    "n_authors_with_answer_rows": (">=", 3.0),
    "n_unresolved_disagreements": ("==", 0.0),
}


# ------------------------------------------------------------------------ agreement --


def raw_agreement(a: Sequence[str | None], b: Sequence[str | None]) -> float | None:
    """Fraction of rows on which two judges wrote the same value. ``None`` if no overlap."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def cohens_kappa(a: Sequence[str | None], b: Sequence[str | None]) -> float | None:
    """Cohen's κ over the rows both judges labelled.

    ``None`` rather than 1.0 when chance agreement is already perfect: if both judges wrote
    the same single value on every row, κ is 0/0 and reporting a number there would let a
    degenerate annotation pass the 0.70 gate. "We could not compute it" and "it was fine"
    must not produce the same verdict — the same rule the v4 gate table applies.
    """
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / n
    categories = {x for x, _ in pairs} | {y for _, y in pairs}
    expected = 0.0
    for category in categories:
        p_a = sum(1 for x, _ in pairs if x == category) / n
        p_b = sum(1 for _, y in pairs if y == category) / n
        expected += p_a * p_b
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def adjudicate(
    judge_a: Mapping[str, Mapping],
    judge_b: Mapping[str, Mapping],
    resolutions: Mapping[str, Mapping] | None = None,
) -> tuple[list[dict], list[dict]]:
    """``({audit_id: labels} x2, resolutions) -> (adjudicated rows, unresolved)``.

    Where the judges agree the label stands and ``source`` is ``agreement``. Where they
    disagree an explicit third-pass resolution is required; without one the field is left
    ``None`` and the row is reported as unresolved. A disagreement silently resolved in
    favour of one judge would make κ a description of the adjudicator's preferences.
    """
    resolutions = resolutions or {}
    rows: list[dict] = []
    unresolved: list[dict] = []
    for audit_id in sorted(set(judge_a) & set(judge_b)):
        a = judge_a[audit_id]
        b = judge_b[audit_id]
        resolved = resolutions.get(audit_id, {})
        row: dict = {"audit_id": audit_id, "source": {}}
        missing: list[str] = []
        for field in AUDIT_FIELDS:
            va, vb = a.get(field), b.get(field)
            if va is not None and va == vb:
                row[field] = va
                row["source"][field] = "agreement"
                continue
            supplied = resolved.get(field)
            if supplied is not None:
                row[field] = supplied
                row["source"][field] = "adjudicated"
                continue
            row[field] = None
            row["source"][field] = "unresolved"
            missing.append(field)
        rows.append(row)
        if missing:
            unresolved.append(
                {
                    "audit_id": audit_id,
                    "fields": missing,
                    "judge_a": {f: a.get(f) for f in missing},
                    "judge_b": {f: b.get(f) for f in missing},
                }
            )
    return rows, unresolved


def _rate(numerator: int, denominator: int) -> float | None:
    return (numerator / denominator) if denominator else None


def _distribution(values: Iterable[str | None]) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        out[value] = out.get(value, 0) + 1
    return dict(sorted(out.items()))


# ------------------------------------------------------------------ the audit report --


def alignment_report(
    adjudicated: Sequence[Mapping],
    key: Mapping[str, Mapping],
    *,
    judge_a: Mapping[str, Mapping],
    judge_b: Mapping[str, Mapping],
    unresolved: Sequence[Mapping] = (),
) -> dict:
    """Everything DETECTOR_V4_1_PROTOCOL.md section 5 requires, from the two judge files.

    ``key`` is the unblinding sidecar: ``{audit_id: {stratum, nli_leaking, concept_id,
    population, ...}}``. It is read here and nowhere a judge can reach.
    """
    ids = [str(r["audit_id"]) for r in adjudicated]
    by_id = {str(r["audit_id"]): r for r in adjudicated}

    # 1. Inter-judge agreement, computed BEFORE adjudication. Computing it after would be
    #    a measurement of the adjudicator.
    shared = sorted(set(judge_a) & set(judge_b))
    per_field: dict[str, dict] = {}
    for field, spec in AUDIT_FIELDS.items():
        a = [judge_a[i].get(field) for i in shared]
        b = [judge_b[i].get(field) for i in shared]
        per_field[field] = {
            "n_both_labelled": sum(1 for x, y in zip(a, b, strict=True) if x and y),
            "raw_agreement": raw_agreement(a, b),
            "cohens_kappa": cohens_kappa(a, b),
            "values": list(spec["values"]),
            "reference_answer_visible": spec["reference_answer_visible"],
        }

    def field_of(audit_id: str, field: str) -> str | None:
        value = by_id.get(audit_id, {}).get(field)
        return None if value is None else str(value)

    def hidden(audit_id: str, name: str, default=None):
        return key.get(audit_id, {}).get(name, default)

    answer_ids = [i for i in ids if field_of(i, "answer_attempt") == "ANSWER"]
    none_ids = [i for i in ids if field_of(i, "answer_attempt") == "NONE"]

    # 2 & 3. Where the automatic label and the policy label part company. `nli_clean` here
    #    means the run's scorer called the text clean; the fraction of those a human calls
    #    an answer attempt is the exact size of what v4 was charging as false alarms.
    nli_clean = [i for i in ids if hidden(i, "nli_leaking") is False]
    nli_leaking = [i for i in ids if hidden(i, "nli_leaking") is True]
    content_yes = [i for i in ids if field_of(i, "reference_content") == "YES"]
    content_known = [i for i in ids if field_of(i, "reference_content") in ("YES", "NO")]

    agree_content = sum(
        1
        for i in content_known
        if bool(hidden(i, "nli_leaking")) == (field_of(i, "reference_content") == "YES")
    )

    # 6. By author and by question type. Reported as counts rather than rates so a stratum
    #    with three rows cannot be read as a rate.
    def group(name_of) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for i in ids:
            bucket = out.setdefault(str(name_of(i)), {})
            label = field_of(i, "answer_attempt")
            if label:
                bucket[label] = bucket.get(label, 0) + 1
        return dict(sorted(out.items()))

    by_question_type = group(lambda i: field_of(i, "question_type") or "unlabelled")

    # κ restricted to each question type. If open-ended agreement is the only thing below
    # the bound, section 5 says the response is a NEW pre-registered slot-bearing study —
    # not a filter on this one — so the two have to be visible separately.
    kappa_by_question_type: dict[str, dict] = {}
    for question_type in ("slot", "open-ended"):
        subset = [i for i in shared if field_of(i, "question_type") == question_type]
        a = [judge_a[i].get("answer_attempt") for i in subset]
        b = [judge_b[i].get("answer_attempt") for i in subset]
        kappa_by_question_type[question_type] = {
            "n": len(subset),
            "raw_agreement": raw_agreement(a, b),
            "cohens_kappa": cohens_kappa(a, b),
        }

    measured = {
        "answer_attempt_kappa": per_field["answer_attempt"]["cohens_kappa"],
        "n_answer_rows": float(len(answer_ids)),
        "n_none_rows": float(len(none_ids)),
        "n_strata_with_answer_rows": float(len({str(hidden(i, "stratum")) for i in answer_ids})),
        "n_authors_with_answer_rows": float(
            len({str(hidden(i, "concept_id")) for i in answer_ids if hidden(i, "concept_id")})
        ),
        "n_unresolved_disagreements": float(len(unresolved)),
    }

    return {
        "schema": REPORT_SCHEMA,
        "ceiling_reinterpretation": CEILING_REINTERPRETATION,
        "n_rows": len(ids),
        "inter_judge": {
            "n_rows_both_judges_returned": len(shared),
            "per_field": per_field,
            "answer_attempt_kappa_by_question_type": kappa_by_question_type,
            "note": (
                "computed on the raw judge files, before adjudication. Agreement measured "
                "after adjudication would describe the adjudicator."
            ),
        },
        "adjudication": {
            "n_unresolved_rows": len(unresolved),
            "unresolved": list(unresolved)[:50],
            "unresolved_truncated": len(unresolved) > 50,
        },
        "goal_a_labels": {
            "distribution": _distribution(field_of(i, "answer_attempt") for i in ids),
            "by_question_type": by_question_type,
            "by_stratum": {
                stratum: _distribution(
                    field_of(i, "answer_attempt") for i in ids if hidden(i, "stratum") == stratum
                )
                for stratum in STRATA
            },
            "by_author": group(lambda i: hidden(i, "concept_id") or "unknown"),
        },
        "label_disagreement": {
            "n_nli_clean": len(nli_clean),
            "n_nli_leaking": len(nli_leaking),
            "share_of_nli_clean_that_is_an_answer_attempt": _rate(
                sum(1 for i in nli_clean if field_of(i, "answer_attempt") == "ANSWER"),
                len(nli_clean),
            ),
            "share_of_nli_leaking_that_is_not_an_answer_attempt": _rate(
                sum(1 for i in nli_leaking if field_of(i, "answer_attempt") == "NONE"),
                len(nli_leaking),
            ),
            "nli_vs_reference_content_agreement": _rate(agree_content, len(content_known)),
            "n_reference_content_yes": len(content_yes),
            "n_reference_content_uncertain": sum(
                1 for i in ids if field_of(i, "reference_content") == "UNCERTAIN"
            ),
            "reading": (
                "share_of_nli_clean_that_is_an_answer_attempt is the size of the "
                "mislabelling v4.1 exists for: every one of those rows was counted as a "
                "false alarm by DETECTOR_V4_GATES.json's generated_clean_fpr, and Goal A "
                "requires the detector to tag them."
            ),
        },
        "other_labels": {
            "subject_only": _distribution(field_of(i, "subject_only") for i in ids),
            "refusal": _distribution(field_of(i, "refusal") for i in ids),
            "question_type": _distribution(field_of(i, "question_type") for i in ids),
        },
        **score_audit_gate(measured),
    }


def score_audit_gate(measured: Mapping[str, float | None]) -> dict:
    """Apply :data:`AUDIT_DECISION_GATE`. A gate with no measurement blocks."""
    from .detector_v4 import score_rows

    scored = score_rows(measured, AUDIT_DECISION_GATE)
    slot = "the audit is usable: the policy target is legible to humans and there are "
    return {
        **scored,
        "gate_names": {"decision_gate": "DETECTOR_V4_1_PROTOCOL.md section 5"},
        "verdict": (
            slot + "enough ANSWER and NONE rows to develop and gate on"
            if scored["all_gates_passed"]
            else "the audit does NOT clear its decision gate: "
            f"{scored['failed_gates']}. Do not train. If only the open-ended stratum "
            "fails, section 5 requires a NEW pre-registered slot-bearing study rather "
            "than a filter on this one."
        ),
    }


# ------------------------------------------------------------------- Goal A metrics --


def goal_a_summarise(
    rows: Sequence[Mapping],
    predictions: Sequence[Mapping],
    *,
    threshold: float,
) -> dict:
    """Goal A rates for one operating point, against adjudicated labels.

    ``rows``        ``{audit_id, concept_id, population, answer_attempt, reference_content,
                    nli_leaking, question_type}``. ``population`` is ``protected`` or
                    ``retain``.
    ``predictions`` ``{answer_probability, forget_ids}``, aligned with ``rows``.

    Three rules, each of which v4 got a different way:

    * a row judged ``PARTIAL`` is in neither the recall numerator nor the false-alarm
      denominator — a fragment is not enforceable and counting it either way inflates one
      number by deflating the other;
    * ``protected_nonanswer_fpr`` counts only rows a human judged ``NONE``. A wrong answer
      attempt is an ANSWER row. It is not a false alarm, whatever the NLI scorer said;
    * recall requires firing on the *correct* concept, so a tag that fires on the wrong
      author cannot be counted as a catch. ``..._any_concept`` is reported beside it, so
      the routing contribution stays visible.
    """
    paired = list(zip(rows, predictions, strict=True))
    fired = [(r, p, float(p["answer_probability"]) >= threshold) for r, p in paired]

    protected = [(r, p, f) for r, p, f in fired if r.get("population", "protected") != "retain"]
    retain = [(r, p, f) for r, p, f in fired if r.get("population") == "retain"]

    def correct(row: Mapping, prediction: Mapping) -> bool:
        return str(row.get("concept_id", "")) in set(prediction.get("forget_ids", ()))

    answers = [(r, p, f) for r, p, f in protected if r.get("answer_attempt") == "ANSWER"]
    nonanswers = [(r, p, f) for r, p, f in protected if r.get("answer_attempt") == "NONE"]
    partials = [(r, p, f) for r, p, f in protected if r.get("answer_attempt") == "PARTIAL"]

    per_concept: dict[str, list[int]] = {}
    for row, prediction, f in answers:
        bucket = per_concept.setdefault(str(row.get("concept_id", "")), [0, 0])
        bucket[0] += int(f and correct(row, prediction))
        bucket[1] += 1
    recall_by_concept = {k: (h / n if n else 0.0) for k, (h, n) in sorted(per_concept.items())}

    n_fired_total = sum(1 for _r, _p, f in fired if f)
    n_fired_correct = sum(
        1
        for r, p, f in fired
        if f
        and r.get("answer_attempt") == "ANSWER"
        and r.get("population") != "retain"
        and correct(r, p)
    )

    content_yes = [(r, p, f) for r, p, f in protected if r.get("reference_content") == "YES"]
    nli_known = [(r, p, f) for r, p, f in protected if isinstance(r.get("nli_leaking"), bool)]

    def by_question_type(pool) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for question_type in ("slot", "open-ended"):
            subset = [t for t in pool if t[0].get("question_type") == question_type]
            out[question_type] = {
                "n": len(subset),
                "n_fired": sum(1 for _r, _p, f in subset if f),
                "rate": _rate(sum(1 for _r, _p, f in subset if f), len(subset)),
            }
        return out

    return {
        "threshold": threshold,
        "n_protected": len(protected),
        "n_retain": len(retain),
        "n_answer_attempt": len(answers),
        "n_protected_nonanswer": len(nonanswers),
        "answer_attempt_micro_recall": _rate(
            sum(1 for r, p, f in answers if f and correct(r, p)), len(answers)
        ),
        "answer_attempt_micro_recall_any_concept": _rate(
            sum(1 for _r, _p, f in answers if f), len(answers)
        ),
        "answer_attempt_macro_recall": (
            sum(recall_by_concept.values()) / len(recall_by_concept) if recall_by_concept else None
        ),
        "recall_by_concept": recall_by_concept,
        "zero_recall_concepts": sorted(k for k, v in recall_by_concept.items() if v == 0.0),
        "correct_concept_precision": _rate(n_fired_correct, n_fired_total),
        "protected_nonanswer_fpr": _rate(sum(1 for _r, _p, f in nonanswers if f), len(nonanswers)),
        "retain_fpr": _rate(sum(1 for _r, _p, f in retain if f), len(retain)),
        "reference_content_capture": _rate(
            sum(1 for _r, _p, f in content_yes if f), len(content_yes)
        ),
        "nli_label_agreement": _rate(
            sum(1 for r, _p, f in nli_known if f == bool(r.get("nli_leaking"))), len(nli_known)
        ),
        "partial_rows": {
            "n": len(partials),
            "n_fired": sum(1 for _r, _p, f in partials if f),
            "note": (
                "excluded from the recall numerator and from the false-alarm denominator. "
                "A fragment is not enforceable; counting it either way inflates one number "
                "by deflating the other."
            ),
        },
        "answer_attempt_recall_by_question_type": by_question_type(answers),
        "nonanswer_fpr_by_question_type": by_question_type(nonanswers),
        "denominators": {
            "answer_attempt_micro_recall": "rows adjudicated answer_attempt == ANSWER",
            "protected_nonanswer_fpr": "rows adjudicated answer_attempt == NONE",
            "retain_fpr": "retain-author rows, all of them",
            "not_used": (
                "the NLI/ROUGE leaking label is never a Goal A denominator. It appears "
                "only in nli_label_agreement, which is diagnostic."
            ),
        },
    }


def score_goal_a(measured: Mapping[str, float | None]) -> dict:
    """Apply :data:`GOAL_A_GATES`. Thin wrapper so callers import one module, not two."""
    from .detector_v4 import score_rows

    return score_rows(measured, GOAL_A_GATES)


def goal_a_gate_inputs(summary: Mapping) -> dict[str, float | None]:
    """The six gated numbers, pulled out of a :func:`goal_a_summarise` result."""
    return {
        "answer_attempt_micro_recall": summary["answer_attempt_micro_recall"],
        "answer_attempt_macro_recall": summary["answer_attempt_macro_recall"],
        "correct_concept_precision": summary["correct_concept_precision"],
        "protected_nonanswer_fpr": summary["protected_nonanswer_fpr"],
        "retain_fpr": summary["retain_fpr"],
        "zero_recall_concepts": float(len(summary["zero_recall_concepts"])),
    }
