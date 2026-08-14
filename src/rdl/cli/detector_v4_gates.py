"""``rdl graph-detector-v4-gates`` — does the ANSWER-FREE detector deserve GPU time?

The companion to the oracle. The oracle is handed the protected answer and reports a
ceiling; this command runs the detector a deployed system could actually run — routing on
request text alone, scoring candidate content against the protected question, never seeing
an answer, a label or an item's concept — and reports how much of that ceiling it reaches.

Two arms, mirroring the oracle:

*synthetic*  ``DETECTOR_V4_DATASET.json``. Engineering signal: which Goal A class the
             detector gets wrong is visible per class, which a pooled number hides.
*natural*    ``DETECTOR_V4_NATURAL_BANK.json``. The arm that decides.

The operating point is chosen on the development partitions only, under both false-alarm
ceilings at once, and the held-out partition is opened once at that point. Exiting non-zero
on a failing gate is the behaviour that matters: this is meant to be the last thing run
before an instance is rented.

The backend
-----------
On CPU this is :class:`LexicalAnswerabilityDetector`, which is a FLOOR, not the deployable
v4 detector. Whatever it reaches, the trained cross-encoder has to beat — and the artifact
records ``backend`` so a later reader cannot mistake one for the other. There is
deliberately no way to select the lexical backend from ``GraphDetectorConfig``: a detector
reaches a study only after its held-out gate has been opened and passed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..defenses.answerability_detector import LexicalAnswerabilityDetector
from ..defenses.concept_registry import ConceptPolicy, ConceptRegistry
from ..defenses.detection_context import ProtectedQuestion, build_context
from ..defenses.identity_router import alias_index
from ..eval.detector_v4 import ORACLE_GATES, score_rows, summarise
from ..eval.tofu_data import load_items
from ..studies.graph_leak.cohort import load_cohort, resolve_cohort
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_data import DATASET_FILENAME, DEFAULT_OUT, NATURAL_BANK_FILENAME

__all__ = ["detector_v4_gates"]

GATES_FILENAME = "DETECTOR_V4_GATES.json"
THRESHOLD_GRID = tuple(round(0.05 * i, 4) for i in range(1, 21))  # 0.05 .. 1.00


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- synthetic arm --


def _synthetic_verdicts(
    rows: Sequence[Mapping], detector: LexicalAnswerabilityDetector
) -> list[str]:
    """``ANSWER`` / ``PARTIAL`` / ``NONE`` per row, from the answer-free detector.

    The row's ``label`` and ``example_class`` are never passed on. The detector receives a
    request string, a protected question built from the row's public fields, and the
    candidate messages — nothing else can reach it, because
    :class:`DetectionContext` refuses the rest.
    """
    out: list[str] = []
    for row in rows:
        question = ProtectedQuestion(
            scope_id=f"{row['subject']}#{row['relation']}",
            forget_id=str(row["subject"]),
            question=str(row["protected_question"]),
            aliases=tuple(str(a) for a in row["aliases"]),
            relation=str(row["relation"]),
            template_id=str(row["template_id"]),
        )
        index = {question.forget_id: list(question.alias_token_sets)}
        context = build_context(
            str(row["request_text"]),
            protected_questions=[question],
            alias_index=index,
        )
        results = detector.score_batch([str(c) for c in row["candidates"]], context=context)
        if any(r.fired for r in results):
            out.append("ANSWER")
        elif any(r.partial_probability >= detector.partial_threshold for r in results):
            out.append("PARTIAL")
        else:
            out.append("NONE")
    return out


# --------------------------------------------------------------------- natural arm --


def _registry_and_questions(cohort_path: Path) -> tuple[ConceptRegistry, dict, dict]:
    """Registry from question text alone, plus ``{item_id: question}`` and provenance."""
    cohort = load_cohort(cohort_path, exclusions_path=None)
    source, _ = load_items(
        dataset=cohort.dataset,
        split=cohort.dataset_config,
        n_items=None,
        revision=cohort.dataset_revision,
    )
    resolved = resolve_cohort(cohort, source)
    by_item = {i.item_id: i for i in resolved}
    rows = [
        {"item_id": e.item_id, "concept_id": e.concept_id, "question": by_item[e.item_id].question}
        for e in cohort.items
        if e.item_id in by_item
    ]
    registry = ConceptRegistry.from_questions(rows, policy=ConceptPolicy())
    questions_by_concept: dict[str, list[str]] = {}
    for row in rows:
        questions_by_concept.setdefault(row["concept_id"], []).append(row["question"])
    meta = {
        "cohort": str(cohort_path),
        "cohort_fingerprint": cohort.fingerprint(),
        "dataset_revision": cohort.dataset_revision,
        "registry_fingerprint": registry.fingerprint(),
        "n_concepts": len(registry),
        "n_questions": len(rows),
        "stores_gold_answers": False,
    }
    return (
        registry,
        {
            "questions_by_concept": questions_by_concept,
            "item_concept": {r["item_id"]: r["concept_id"] for r in rows},
        },
        meta,
    )


def _protected_questions(
    registry: ConceptRegistry, questions_by_concept: Mapping[str, Sequence[str]]
) -> list[ProtectedQuestion]:
    out: list[ProtectedQuestion] = []
    for concept in registry.concepts():
        for i, question in enumerate(
            dict.fromkeys(questions_by_concept.get(concept.forget_id, ()))
        ):
            out.append(
                ProtectedQuestion(
                    scope_id=f"{concept.forget_id}#{i:03d}",
                    forget_id=concept.forget_id,
                    question=question,
                    aliases=tuple(concept.aliases),
                    template_id="cohort",
                )
            )
    return out


def _natural_scores(
    rows: Sequence[Mapping],
    detector: LexicalAnswerabilityDetector,
    questions: Sequence[ProtectedQuestion],
    index: Mapping[str, Sequence[frozenset[str]]],
) -> list[dict]:
    """``[{answer_probability, forget_ids}, ...]``, scored under routing.

    Grouped by request so the router runs once per distinct request rather than once per
    message, which is also how the executor will call it: one routing decision per
    trajectory, reused at every surface.
    """
    by_request: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_request.setdefault(str(row.get("request", "")), []).append(i)
    scores: list[dict] = [{"answer_probability": 0.0, "forget_ids": ()} for _ in rows]
    for request, idxs in by_request.items():
        context = build_context(request, protected_questions=questions, alias_index=index)
        # Each message is scored on its own: accumulation across unrelated trajectories
        # would make one item's verdict depend on another's.
        for i in idxs:
            result = detector.score_batch([str(rows[i]["text"])], context=context)[0]
            scores[i] = {
                "answer_probability": result.answer_probability,
                "forget_ids": result.forget_ids,
            }
    return scores


def _natural_measure(
    rows: Sequence[Mapping],
    scores: Sequence[Mapping],
    concept_of: Mapping[str, str],
    threshold: float,
) -> dict:
    per_concept: dict[str, list[int]] = {}
    n_fired = 0
    n_correct = 0
    for row, score in zip(rows, scores, strict=True):
        fired = score["answer_probability"] >= threshold
        concept = concept_of.get(str(row.get("item_id", "")), "")
        if row["leaking"]:
            bucket = per_concept.setdefault(concept, [0, 0])
            bucket[0] += int(fired and concept in set(score["forget_ids"]))
            bucket[1] += 1
        if fired:
            n_fired += 1
            n_correct += int(row["leaking"] and concept in set(score["forget_ids"]))
    recall = {k: (h / n if n else 0.0) for k, (h, n) in sorted(per_concept.items())}
    positives = sum(n for _h, n in per_concept.values())
    hits = sum(h for h, _n in per_concept.values())
    negatives = [
        (row, score) for row, score in zip(rows, scores, strict=True) if not row["leaking"]
    ]
    return {
        "threshold": threshold,
        "n_leaking": positives,
        "n_clean": len(negatives),
        "micro_recall": (hits / positives) if positives else None,
        "macro_recall": (sum(recall.values()) / len(recall)) if recall else None,
        "generated_clean_fpr": (
            sum(1 for _r, s in negatives if s["answer_probability"] >= threshold) / len(negatives)
            if negatives
            else None
        ),
        "correct_concept_precision": (n_correct / n_fired) if n_fired else None,
        "zero_recall_concepts": sorted(k for k, v in recall.items() if v == 0.0),
        "recall_by_concept": recall,
    }


def detector_v4_gates(
    data_dir: Path = typer.Option(DEFAULT_OUT, "--data-dir"),
    policy_cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/discovery.json"), "--policy-cohort"
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Select an operating point on development, open held-out once, write the gates."""
    dataset = json.loads((data_dir / DATASET_FILENAME).read_text(encoding="utf-8"))

    # ------------------------------------------------------------- synthetic arm --
    synthetic: dict[str, dict] = {}
    synthetic_grid = []
    for threshold in THRESHOLD_GRID:
        detector = LexicalAnswerabilityDetector(answer_threshold=threshold)
        rows = [r for r in dataset["rows"] if r["split"] == "development"]
        summary = summarise(rows, _synthetic_verdicts(rows, detector))
        synthetic_grid.append(
            {
                "threshold": threshold,
                "micro_recall": summary["micro_recall"],
                "generated_clean_fpr": summary["false_alarm_pools"]["generated_clean"]["fpr"],
                "retain_answer_fpr": summary["false_alarm_pools"]["retain_answer"]["fpr"],
            }
        )
    eligible = [
        r
        for r in synthetic_grid
        if (r["generated_clean_fpr"] or 0.0) <= ORACLE_GATES["generated_clean_fpr"][1]
        and (r["retain_answer_fpr"] or 0.0) <= ORACLE_GATES["retain_answer_fpr"][1]
        and r["micro_recall"] is not None
    ]
    best = max(eligible, key=lambda r: (r["micro_recall"], -r["threshold"]), default=None)
    synthetic_threshold = best["threshold"] if best else None

    for split in ("development", "heldout"):
        rows = [r for r in dataset["rows"] if r["split"] == split]
        if synthetic_threshold is None:
            synthetic[split] = {"measured": False, "reason": "no eligible threshold on development"}
            continue
        detector = LexicalAnswerabilityDetector(answer_threshold=synthetic_threshold)
        summary = summarise(rows, _synthetic_verdicts(rows, detector))
        measured = {
            "micro_recall": summary["micro_recall"],
            "macro_recall": summary["macro_recall"],
            "generated_clean_fpr": summary["false_alarm_pools"]["generated_clean"]["fpr"],
            "retain_answer_fpr": summary["false_alarm_pools"]["retain_answer"]["fpr"],
            "zero_recall_subjects": float(len(summary["zero_recall_subjects"])),
        }
        synthetic[split] = {
            "measured": True,
            "summary": summary,
            **score_rows(measured, ORACLE_GATES),
        }

    # ---------------------------------------------------------------- natural arm --
    bank_path = data_dir / NATURAL_BANK_FILENAME
    natural: dict = {"measured": False, "reason": f"{bank_path} is absent"}
    registry_meta: dict = {}
    if bank_path.exists():
        bank = json.loads(bank_path.read_text(encoding="utf-8"))
        registry, lookups, registry_meta = _registry_and_questions(policy_cohort)
        questions = _protected_questions(registry, lookups["questions_by_concept"])
        index = alias_index(registry)
        concept_of = lookups["item_concept"]
        detector = LexicalAnswerabilityDetector()

        def rows_of(partition: Mapping) -> list[dict]:
            out = []
            for leaking, keyname in ((True, "leaking"), (False, "clean")):
                for row in partition.get(keyname, ()):
                    out.append({**row, "leaking": leaking})
            return out

        dev_rows = rows_of(bank["partitions"]["development"])
        gate_rows = rows_of(bank["partitions"]["heldout"])
        retain_rows = [{**r, "leaking": False} for r in bank["partitions"]["retain"]["all"]]

        dev_scores = _natural_scores(dev_rows, detector, questions, index)
        gate_scores = _natural_scores(gate_rows, detector, questions, index)
        retain_scores = _natural_scores(retain_rows, detector, questions, index)

        grid = []
        for threshold in THRESHOLD_GRID:
            row = _natural_measure(dev_rows, dev_scores, concept_of, threshold)
            row["retain_answer_fpr"] = (
                sum(1 for s in retain_scores if s["answer_probability"] >= threshold)
                / len(retain_scores)
                if retain_scores
                else None
            )
            grid.append(row)
        eligible = [
            r
            for r in grid
            if r["generated_clean_fpr"] is not None
            and r["generated_clean_fpr"] <= ORACLE_GATES["generated_clean_fpr"][1]
            and (
                r["retain_answer_fpr"] is None
                or r["retain_answer_fpr"] <= ORACLE_GATES["retain_answer_fpr"][1]
            )
            and r["micro_recall"] is not None
        ]
        chosen = max(eligible, key=lambda r: (r["micro_recall"], -r["threshold"]), default=None)
        # Named apart from the sweep's loop variable: the selected point is the one the
        # held-out half is opened at, and reusing the name would make a reader check.
        selected: float | None = float(chosen["threshold"]) if chosen else None

        held = (
            _natural_measure(gate_rows, gate_scores, concept_of, selected)
            if selected is not None
            else None
        )
        retain_fpr = (
            sum(1 for s in retain_scores if s["answer_probability"] >= selected)
            / len(retain_scores)
            if (selected is not None and retain_scores)
            else None
        )
        measured = (
            {
                "micro_recall": held["micro_recall"],
                "macro_recall": held["macro_recall"],
                "generated_clean_fpr": held["generated_clean_fpr"],
                "retain_answer_fpr": retain_fpr,
                "zero_recall_subjects": float(len(held["zero_recall_concepts"])),
            }
            if held
            else dict.fromkeys(ORACLE_GATES)
        )
        natural = {
            "measured": True,
            "bank_content_sha256": bank.get("content_sha256"),
            "selection": {
                "grid": grid,
                "selected_threshold": selected,
                "selected_on": "development half of the natural bank only",
            },
            "heldout": held,
            "retain_answer_fpr": retain_fpr,
            "retain_construction_note": (
                "bounded above by the router: nothing is scanned on a request the router "
                "left unselected, so a zero here is a property of the routing rather than "
                "independent evidence that content detection is precise."
            ),
            **score_rows(measured, ORACLE_GATES),
        }

    detector_dict = LexicalAnswerabilityDetector().to_dict()
    report = {
        "schema": "graph-detector-v4-gates-v1",
        "phase": "Detector-v4 Phase 6 — answer-free detector on CPU. No model is trained here.",
        "backend": detector_dict,
        "backend_is_a_floor": (
            "the lexical scorer is the CPU reference and the floor the trained "
            "cross-encoder must beat. It is not the deployable v4 detector and no report "
            "may present it as one. GraphDetectorConfig cannot select it."
        ),
        "gates": {k: {"comparison": c, "bound": b} for k, (c, b) in ORACLE_GATES.items()},
        "synthetic_arm": {
            "role": "engineering signal; per-class errors. Not the verdict.",
            "selected_threshold": synthetic_threshold,
            "selection_grid": synthetic_grid,
            "dataset_content_sha256": dataset.get("content_sha256"),
            **synthetic,
        },
        "natural_arm": natural,
        "registry": registry_meta,
        "verdict": (
            "the natural arm was not measured; no CPU verdict is available"
            if not natural.get("measured")
            else (
                "the answer-free detector clears the frozen bounds on natural text"
                if natural.get("all_gates_passed")
                else "the answer-free detector does NOT clear the frozen bounds on natural text"
            )
        ),
        "runtime_reads_gold_answers": False,
        "scope": {
            "model_trained": False,
            "graph_generation_run": False,
            "frozen_v1_v2_v3_artifacts_modified": False,
            "gpu_used": False,
        },
    }
    out = output or (data_dir / GATES_FILENAME)
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
    raise typer.Exit(0 if natural.get("all_gates_passed") else 1)
