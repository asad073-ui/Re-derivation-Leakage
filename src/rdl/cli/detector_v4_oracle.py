"""``rdl graph-detector-v4-oracle`` — is the problem separable at all?

Phase 5. Before any transformer is trained, hand an evaluator the protected answer and ask
whether it can separate leaking text from clean text. The logic is one-directional and is
the whole reason this runs first:

* the oracle FAILS  -> no answer-free detector can do better, and the next change is the
  task definition or the label source, not the model. Stop; do not rent an instance.
* the oracle PASSES and the answer-free detector fails -> the representation or the
  training is the problem, which is a tractable engineering result.

Detector v3 is the cautionary case. Its oracle reached 0.264 micro recall against a 0.80
bound, and that number was reported *after* a day of router work that could never have
mattered. Running the ceiling first is how that is not repeated.

Two arms, and the second is the one that counts
-----------------------------------------------
**Synthetic arm.** The oracle over ``DETECTOR_V4_DATASET.json``. This is a CONSTRUCTION
CHECK, not evidence: the rows were authored against the same class taxonomy the oracle
applies, so a high score here says the dataset is internally consistent and says nothing
about natural text. The artifact labels it that way and the gate verdict does not rest on
it.

**Natural arm.** The oracle over ``DETECTOR_V4_NATURAL_BANK.json`` — real unguarded-arm
messages from the natural graph_flow run, labelled by that run's own pinned NLI scorer.
The oracle sees the pinned TOFU answer for the item; the labels come from a different
function than the oracle's, so agreement is a measurement rather than an identity. This is
the arm the CPU checkpoint depends on.

The oracle's caveat is recorded in the artifact rather than left for a reader to notice:
the run's scorer combines NLI entailment with ROUGE-L, and the oracle is a content-token
overlap rule, so the two are correlated by construction to an unknown degree. A natural-arm
ceiling is therefore an upper bound on what an ANSWER-AWARE evaluator achieves against
THIS label source, which is exactly the quantity Phase 5 asks for, and is not a claim about
ground truth.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4 import ORACLE_GATES, oracle_verdicts, score_rows, summarise
from ..memory.index import normalise_text
from ..studies.graph_leak.cohort import load_cohort
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_data import (
    ANSWER_KEY_FILENAME,
    DATASET_FILENAME,
    DEFAULT_OUT,
    NATURAL_BANK_FILENAME,
)

__all__ = ["detector_v4_oracle"]

CEILING_FILENAME = "DETECTOR_V4_ORACLE_CEILING.json"
COVERAGE_GRID = tuple(round(0.05 * i, 4) for i in range(1, 21))  # 0.05 .. 1.00

# Words that carry no answer content. A gold answer restates the question — "Hsiao
# Yun-Hwa's father was a civil engineer" contains the whole question — so an overlap rule
# that counted them would fire on the question echo, which is the failure mode Goal A
# exists to exclude.
_STOP = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "that",
        "the",
        "their",
        "they",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _content_tokens(answer: str, question: str) -> frozenset[str]:
    """Answer tokens that the QUESTION did not already supply.

    This is the oracle's privileged information and its only privileged information: what
    the forgotten fact actually contains, minus everything the request handed over for
    free.
    """
    q = set(normalise_text(question).split())
    return frozenset(
        t for t in normalise_text(answer).split() if t not in q and t not in _STOP and len(t) > 2
    )


def _coverage(text: str, content: frozenset[str]) -> float:
    if not content:
        return 0.0
    tokens = set(normalise_text(text).split())
    return len(content & tokens) / len(content)


# Question frames whose answer is a discussion rather than a value. Goal A presumes a
# protected question has a SLOT — "where was X born" is answered by a place, and a
# candidate either supplies one or does not. "How would X advise aspiring authors" has no
# slot: any number of unrelated paragraphs answer it equally well, and "does this candidate
# attempt to supply the answer" stops being a well-formed question about the text.
#
# Recorded as a stratum rather than a filter. Whether TOFU forget10 is mostly slot-bearing
# is a fact about the cohort that a v4 report has to state, not one it may quietly select
# around.
_OPEN_ENDED_FRAMES = (
    "how would",
    "how did",
    "how has",
    "how have",
    "how does",
    "how do",
    "what inspired",
    "what motivated",
    "what influence",
    "what impact",
    "what challenges",
    "what advice",
    "what themes",
    "in what ways",
    "can you describe",
    "could you describe",
    "can you share",
    "describe ",
)


def _has_answer_slot(question: str) -> bool:
    lowered = (question or "").strip().lower()
    return not any(frame in lowered for frame in _OPEN_ENDED_FRAMES)


def _natural_rows(
    partition: Mapping,
    answers: Mapping[str, str],
    concept_of: Mapping[str, str],
) -> list[dict]:
    rows: list[dict] = []
    for label, key in ((True, "leaking"), (False, "clean")):
        for row in partition.get(key, ()):
            item_id = str(row.get("item_id", ""))
            answer = answers.get(item_id, "")
            rows.append(
                {
                    "text": str(row["text"]),
                    "request": str(row.get("request", "")),
                    "item_id": item_id,
                    "concept_id": concept_of.get(item_id, ""),
                    "leaking": label,
                    "content": _content_tokens(answer, str(row.get("request", ""))),
                    "has_answer": bool(answer),
                    "has_answer_slot": _has_answer_slot(str(row.get("request", ""))),
                }
            )
    return rows


def _failure_locus(dev_rows: Sequence[Mapping], gate_rows: Sequence[Mapping]) -> dict:
    """WHICH component failed, when the ceiling fails.

    A failing ceiling has two very different readings and they imply opposite next steps.
    Either the answerability framing is wrong for this data, or the LABEL SOURCE calls
    things leaks that no answer-aware evaluator can see. The second is measurable: if a
    text the run's scorer judged leaking shares no content token with the pinned answer,
    then handing an evaluator the answer buys it nothing on that row, and a detector's
    recall against that label is bounded by something other than the detector.
    """
    rows = [*dev_rows, *gate_rows]
    leaking = [r for r in rows if r["leaking"]]
    clean = [r for r in rows if not r["leaking"]]
    if not leaking:
        return {"measured": False, "reason": "no leaking rows in the bank"}

    covered = [(r, _coverage(str(r["text"]), r["content"])) for r in leaking]
    zero = [r for r, c in covered if c == 0.0]

    def stratum(pool: Sequence[Mapping], slot: bool) -> list[Mapping]:
        return [r for r in pool if r["has_answer_slot"] is slot]

    strata = {}
    for name, slot in (("slot_bearing", True), ("open_ended", False)):
        pos = stratum(leaking, slot)
        neg = stratum(clean, slot)
        pos_cov = [_coverage(str(r["text"]), r["content"]) for r in pos]
        strata[name] = {
            "n_leaking": len(pos),
            "n_clean": len(neg),
            "mean_answer_token_coverage_on_leaking": (
                sum(pos_cov) / len(pos_cov) if pos_cov else None
            ),
            "share_of_leaking_with_zero_coverage": (
                sum(1 for c in pos_cov if c == 0.0) / len(pos_cov) if pos_cov else None
            ),
        }

    return {
        "measured": True,
        "share_of_leaking_invisible_to_an_answer_aware_evaluator": len(zero) / len(leaking),
        "n_leaking": len(leaking),
        "n_leaking_with_zero_answer_token_coverage": len(zero),
        "share_of_leaking_from_open_ended_questions": (
            sum(1 for r in leaking if not r["has_answer_slot"]) / len(leaking)
        ),
        "by_question_type": strata,
        "reading": (
            "Goal A presumes the protected question has an answer SLOT. Where the leaking "
            "population is dominated by open-ended frames — 'how would X advise…', 'what "
            "inspired X' — there is no slot to fill, 'does this candidate attempt to supply "
            "the answer' is not a well-formed question about the text, and text the scorer "
            "judged leaking shares no content token with the pinned answer. An evaluator "
            "HOLDING the answer cannot separate those rows, so neither can any answer-free "
            "detector, and the bound is the leak definition rather than the model class."
        ),
        "implication": (
            "if this dominates, the next change is the protected-question population or the "
            "leak label — not a bigger encoder. Tuning a detector against a label an "
            "answer-aware oracle cannot reproduce is how v3 spent a day on a router that "
            "was never the bottleneck."
        ),
    }


def _measure(rows: Sequence[Mapping], coverage_threshold: float) -> dict:
    positives = [r for r in rows if r["leaking"]]
    negatives = [r for r in rows if not r["leaking"]]

    def fires(row: Mapping) -> bool:
        return _coverage(str(row["text"]), row["content"]) >= coverage_threshold

    per_concept: dict[str, list[int]] = {}
    for row in positives:
        bucket = per_concept.setdefault(str(row["concept_id"]), [0, 0])
        bucket[0] += int(fires(row))
        bucket[1] += 1
    recall_by_concept = {k: (h / n if n else 0.0) for k, (h, n) in sorted(per_concept.items())}
    macro = sum(recall_by_concept.values()) / len(recall_by_concept) if recall_by_concept else None
    return {
        "coverage_threshold": coverage_threshold,
        "n_leaking": len(positives),
        "n_clean": len(negatives),
        "micro_recall": (
            sum(1 for r in positives if fires(r)) / len(positives) if positives else None
        ),
        "macro_recall": macro,
        "generated_clean_fpr": (
            sum(1 for r in negatives if fires(r)) / len(negatives) if negatives else None
        ),
        "zero_recall_concepts": sorted(k for k, v in recall_by_concept.items() if v == 0.0),
        "recall_by_concept": recall_by_concept,
        "n_rows_without_a_pinned_answer": sum(1 for r in rows if not r["has_answer"]),
    }


def detector_v4_oracle(
    data_dir: Path = typer.Option(DEFAULT_OUT, "--data-dir"),
    policy_cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/discovery.json"), "--policy-cohort"
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Measure the answer-aware ceiling and write ``DETECTOR_V4_ORACLE_CEILING.json``."""
    dataset = json.loads((data_dir / DATASET_FILENAME).read_text(encoding="utf-8"))
    key = json.loads((data_dir / ANSWER_KEY_FILENAME).read_text(encoding="utf-8"))
    if key.get("dataset_content_sha256") != dataset.get("content_sha256"):
        raise typer.BadParameter("the answer key does not belong to this dataset")

    # ------------------------------------------------------------- synthetic arm --
    synthetic: dict[str, dict] = {}
    for split in ("development", "heldout"):
        rows = [r for r in dataset["rows"] if r["split"] == split]
        summary = summarise(rows, oracle_verdicts(rows, key))
        measured = {
            "micro_recall": summary["micro_recall"],
            "macro_recall": summary["macro_recall"],
            "generated_clean_fpr": summary["false_alarm_pools"]["generated_clean"]["fpr"],
            "retain_answer_fpr": summary["false_alarm_pools"]["retain_answer"]["fpr"],
            "zero_recall_subjects": float(len(summary["zero_recall_subjects"])),
        }
        synthetic[split] = {"summary": summary, **score_rows(measured, ORACLE_GATES)}

    # ---------------------------------------------------------------- natural arm --
    bank_path = data_dir / NATURAL_BANK_FILENAME
    natural: dict = {
        "measured": False,
        "reason": f"{bank_path} is absent; run `rdl graph-detector-v4-build-data --natural-run ...`",
    }
    if bank_path.exists():
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        answers = dict(key.get("natural_answers", {}))
        cohort = load_cohort(policy_cohort, exclusions_path=None)
        concept_of = {e.item_id: e.concept_id for e in cohort.items}

        dev_rows = _natural_rows(bank["partitions"]["development"], answers, concept_of)
        gate_rows = _natural_rows(bank["partitions"]["heldout"], answers, concept_of)

        # The coverage threshold is selected on DEVELOPMENT alone, under the same clean-FPR
        # ceiling the detector gate uses. Selecting it on the gate half would make the
        # ceiling a description of the gate half rather than a bound on it.
        grid = [_measure(dev_rows, t) for t in COVERAGE_GRID]
        eligible = [
            row
            for row in grid
            if row["generated_clean_fpr"] is not None
            and row["generated_clean_fpr"] <= ORACLE_GATES["generated_clean_fpr"][1]
            and row["micro_recall"] is not None
        ]
        best = max(
            eligible, key=lambda r: (r["micro_recall"], -r["coverage_threshold"]), default=None
        )
        selected = best["coverage_threshold"] if best else None
        held = _measure(gate_rows, selected) if selected is not None else None

        measured = (
            {
                "micro_recall": held["micro_recall"],
                "macro_recall": held["macro_recall"],
                "generated_clean_fpr": held["generated_clean_fpr"],
                # Not measured on this arm: the retain bank is scored by the gate command
                # under routing, and reporting it here from an unrouted oracle would
                # compare two different quantities.
                "retain_answer_fpr": None,
                "zero_recall_subjects": float(len(held["zero_recall_concepts"])),
            }
            if held
            else dict.fromkeys(ORACLE_GATES)
        )
        # The same ceiling restricted to questions that HAVE a slot. Reported separately
        # and never used for the verdict: it is measured on the same rows the selection
        # saw, and on this cohort it is a handful of examples. It exists so a reader can
        # see whether Goal A is ill-posed everywhere or only on the open-ended frames.
        slot_dev = [r for r in dev_rows if r["has_answer_slot"]]
        slot_gate = [r for r in gate_rows if r["has_answer_slot"]]
        slot_only = {
            "n_leaking_development": sum(1 for r in slot_dev if r["leaking"]),
            "n_leaking_heldout": sum(1 for r in slot_gate if r["leaking"]),
            "heldout_at_selected_threshold": (
                _measure(slot_gate, selected) if selected is not None else None
            ),
            "caveat": (
                "underpowered on this cohort and measured at a threshold selected on the "
                "pooled development half. Not a gate, and not evidence for one."
            ),
        }

        natural = {
            "measured": True,
            "bank_content_sha256": bank.get("content_sha256"),
            "failure_locus": _failure_locus(dev_rows, gate_rows),
            "slot_bearing_questions_only": slot_only,
            "label_source": bank.get("sources", {}),
            "selection": {
                "grid": grid,
                "selected_coverage_threshold": selected,
                "selected_on": "development half of the natural bank only",
                "constraint": f"generated_clean_fpr <= {ORACLE_GATES['generated_clean_fpr'][1]}",
            },
            "heldout": held,
            **score_rows(
                measured, {k: v for k, v in ORACLE_GATES.items() if k != "retain_answer_fpr"}
            ),
            "caveat": (
                "the run's label is NLI entailment plus ROUGE-L; the oracle is answer-token "
                "coverage. The two are correlated by construction to an unknown degree, so "
                "this is an upper bound on what an ANSWER-AWARE evaluator achieves against "
                "THIS label source — which is the quantity Phase 5 asks for — and is not a "
                "claim about ground truth."
            ),
        }

    verdict_arm = natural if natural.get("measured") else None
    report = {
        "schema": "graph-detector-v4-oracle-ceiling-v1",
        "phase": "Detector-v4 Phase 5 — offline answer-aware ceiling. No model is trained here.",
        "gate_policy": (
            "bounds carried unchanged from DETECTOR_V2_GATES.json and the v3 probe. Phase 5 "
            "is not entitled to move them."
        ),
        "gates": {k: {"comparison": c, "bound": b} for k, (c, b) in ORACLE_GATES.items()},
        "synthetic_arm": {
            "role": "CONSTRUCTION CHECK ONLY",
            "why_not_evidence": (
                "the rows were authored against the same Goal A class taxonomy the oracle "
                "applies. A high score here says the dataset is internally consistent and "
                "separable in principle; it says nothing about natural text, and the "
                "verdict below does not rest on it."
            ),
            "dataset_content_sha256": dataset.get("content_sha256"),
            **synthetic,
        },
        "natural_arm": natural,
        "verdict": (
            "the natural arm was not measured; the CPU checkpoint is not satisfied"
            if verdict_arm is None
            else (
                "an answer-aware evaluator clears the frozen bounds on natural text; the "
                "content primitive is separable and training an answer-free detector is "
                "justified"
                if verdict_arm.get("all_gates_passed")
                else "an answer-aware evaluator does NOT clear the frozen bounds on natural "
                "text; no answer-free detector can do better and Detector v4 should not "
                "proceed to training on this evidence. See natural_arm.failure_locus for "
                "WHICH component the bound belongs to."
            )
        ),
        "reads_gold_answers": True,
        "runtime_reads_gold_answers": False,
        "scope": {
            "model_trained": False,
            "graph_generation_run": False,
            "frozen_v1_v2_v3_artifacts_modified": False,
            "gpu_used": False,
        },
    }
    out = output or (data_dir / CEILING_FILENAME)
    atomic_json(out, report)
    typer.echo(f"wrote {out}")
    typer.echo(f"verdict: {report['verdict']}")
    if natural.get("measured"):
        for gate in natural["gates"]:
            mark = "PASS" if gate["passed"] else ("n/a " if gate["passed"] is None else "FAIL")
            typer.echo(
                f"  [{mark}] natural {gate['gate']}: {gate['measured']} "
                f"(need {gate['comparison']} {gate['bound']})"
            )
    raise typer.Exit(0 if (natural.get("measured") and natural.get("all_gates_passed")) else 1)
