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

The backend, and why it is now a flag
-------------------------------------
``--backend lexical`` is :class:`LexicalAnswerabilityDetector`, a FLOOR rather than the
deployable v4 detector: whatever it reaches, the trained cross-encoder has to beat.
``--backend cross_encoder --model-artifact runs/…/DETECTOR_V4_MODEL.json`` runs the trained
checkpoint through the SAME gate arithmetic. That is the point of the flag — a gate
implementation that existed only for the floor would mean the two numbers were produced by
two pieces of code, and the comparison would be between those rather than between the
scorers. Neither backend is selectable from ``GraphDetectorConfig``: a detector reaches a
study only after its gate has been opened, on the fresh bank, and passed.

The Goal A arm (v4.1)
---------------------
The natural arm below scores against the run's NLI+ROUGE ``leaking`` label, which answers
"does this reproduce the reference answer?". Goal A asks "does this attempt to answer?", and
those disagree on every wrong answer attempt — which the old ``generated_clean_fpr``
therefore charged as a false alarm while Goal A requires the tag. So the NLI arm is now
DIAGNOSTIC, and the arm that decides is scored against the adjudicated human labels from
``LABEL_AUDIT_ADJUDICATED.jsonl``. Until those labels exist the Goal A arm reports
``measured: false`` and blocks, which is the correct state and not an absence.
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
from ..defenses.detector_protocol import ConceptDetector
from ..defenses.identity_router import alias_index
from ..eval.detector_v4 import ORACLE_GATES, score_rows, summarise
from ..eval.detector_v4_1 import (
    CEILING_REINTERPRETATION,
    GOAL_A_GATES,
    goal_a_gate_inputs,
    goal_a_summarise,
    score_goal_a,
)
from ..eval.tofu_data import load_items
from ..studies.graph_leak.cohort import load_cohort, resolve_cohort
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT
from .detector_v4_data import DATASET_FILENAME, DEFAULT_OUT, NATURAL_BANK_FILENAME
from .detector_v4_label_audit import ADJUDICATED_FILENAME, KEY_FILENAME

__all__ = ["build_backend", "detector_v4_gates"]

GATES_FILENAME = "DETECTOR_V4_GATES.json"
# v4.1 writes its own file. D6 of DETECTOR_V4_1_PROTOCOL.md: every correction is a new
# artifact, and DETECTOR_V4_GATES.json is frozen evidence that GU-0036 cites by name.
V4_1_GATES_FILENAME = "DETECTOR_V4_1_GATES.json"
# A coarse grid reports the recall of a threshold nobody chose. The sweep now runs over
# EXACT score breakpoints — every distinct score is an operating point, plus one above the
# maximum so "fire on nothing" is representable — with the 0.05 grid kept only as the
# fallback for a pool small enough that breakpoints and grid coincide.
COARSE_THRESHOLD_GRID = tuple(round(0.05 * i, 4) for i in range(1, 21))  # 0.05 .. 1.00
MAX_BREAKPOINTS = 512
BACKENDS = ("lexical", "cross_encoder")


def breakpoints(scores: Sequence[float]) -> tuple[float, ...]:
    """Exact operating points from observed scores, capped so the sweep stays bounded.

    Above :data:`MAX_BREAKPOINTS` distinct scores the list is thinned by rank rather than
    by value, so the retained points still sit ON observed scores. A uniform 0.01 grid
    would be finer than 0.05 and still describe thresholds no row realises.
    """
    distinct = sorted({float(s) for s in scores})
    if not distinct:
        return COARSE_THRESHOLD_GRID
    distinct.append(distinct[-1] + 1e-9)
    if len(distinct) <= MAX_BREAKPOINTS:
        return tuple(distinct)
    stride = len(distinct) / MAX_BREAKPOINTS
    thinned = {distinct[min(len(distinct) - 1, int(i * stride))] for i in range(MAX_BREAKPOINTS)}
    thinned.add(distinct[-1])
    return tuple(sorted(thinned))


def build_backend(
    backend: str,
    model_artifact: Path | None,
    *,
    answer_threshold: float = 0.5,
    device: str = "",
) -> ConceptDetector:
    """``--backend`` -> a :class:`ConceptDetector`. The only place the two differ.

    ``transformers`` is imported by ``CrossEncoderAnswerabilityDetector.from_artifact`` and
    nowhere else, so ``--backend lexical`` stays torch-free and the CPU gate stays offline.
    """
    if backend == "lexical":
        if model_artifact is not None:
            raise typer.BadParameter("--model-artifact is meaningless for the lexical backend")
        if device:
            raise typer.BadParameter("--device is meaningless for the lexical backend")
        return LexicalAnswerabilityDetector(answer_threshold=answer_threshold)
    if backend == "cross_encoder":
        if model_artifact is None:
            raise typer.BadParameter(
                "--backend cross_encoder needs --model-artifact pointing at the "
                "DETECTOR_V4_MODEL.json written next to the checkpoint. A checkpoint "
                "whose weights and tokenizer cannot be named is not eligible for a gate."
            )
        from ..defenses.cross_encoder_answerability import CrossEncoderAnswerabilityDetector

        return CrossEncoderAnswerabilityDetector.from_artifact(
            model_artifact, answer_threshold=answer_threshold, device=device
        )
    raise typer.BadParameter(f"--backend must be one of {BACKENDS}, got {backend!r}")


def _at_threshold(detector: ConceptDetector, threshold: float) -> ConceptDetector:
    """A copy of ``detector`` at a different operating point, without reloading weights."""
    with_thresholds = getattr(detector, "with_thresholds", None)
    if with_thresholds is None:  # pragma: no cover - both backends implement it
        raise typer.BadParameter(f"{type(detector).__name__} cannot be re-thresholded")
    return with_thresholds(answer_threshold=threshold)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------- synthetic arm --


def _synthetic_verdicts(rows: Sequence[Mapping], detector: ConceptDetector) -> list[str]:
    """``ANSWER`` / ``PARTIAL`` / ``NONE`` per row, from the answer-free detector.

    The row's ``label`` and ``example_class`` are never passed on. The detector receives a
    request string, a protected question built from the row's public fields, and the
    candidate messages — nothing else can reach it, because
    :class:`DetectionContext` refuses the rest.
    """
    partial_threshold = float(getattr(detector, "partial_threshold", 0.5))
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
        elif any(r.partial_probability >= partial_threshold for r in results):
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
    detector: ConceptDetector,
    questions: Sequence[ProtectedQuestion],
    index: Mapping[str, Sequence[frozenset[str]]],
) -> list[dict]:
    """``[{answer_probability, forget_ids}, ...]``, scored under routing.

    Grouped by request so the router runs once per distinct request rather than once per
    message, which is also how the executor will call it: one routing decision per
    trajectory, reused at every surface.

    Each message is scored on its own — accumulation across unrelated trajectories would
    make one item's verdict depend on another's — but "on its own" is a property of the
    SCORING, not of the batch size. v4.1 bought it with ``score_batch([one_text])`` per
    row, which meant the cross-encoder's configured batch size batched nothing and the
    natural bank ran as thousands of single-row forwards. A backend that offers
    :meth:`score_independent` gets the same independence in one batched pass; the lexical
    floor, which is pure Python and gains nothing from batching, takes the loop.
    """
    by_request: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_request.setdefault(str(row.get("request", "")), []).append(i)
    scores: list[dict] = [{"answer_probability": 0.0, "forget_ids": ()} for _ in rows]

    independent = getattr(detector, "score_independent", None)
    if callable(independent):
        order: list[int] = []
        requests: list[tuple] = []
        for request, idxs in by_request.items():
            context = build_context(request, protected_questions=questions, alias_index=index)
            for i in idxs:
                order.append(i)
                requests.append((context, str(rows[i]["text"])))
        for i, result in zip(order, independent(requests), strict=True):
            scores[i] = {
                "answer_probability": result.answer_probability,
                "forget_ids": result.forget_ids,
            }
        return scores

    for request, idxs in by_request.items():
        context = build_context(request, protected_questions=questions, alias_index=index)
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


def _goal_a_arm(
    v4_1_dir: Path,
    base: ConceptDetector,
    questions: Sequence[ProtectedQuestion],
    index: Mapping[str, Sequence[frozenset[str]]],
) -> dict:
    """Goal A rates against ADJUDICATED HUMAN labels. The arm the v4.1 verdict rests on.

    Absent labels report ``measured: false`` and block. "We did not measure it" and "it was
    fine" must not produce the same verdict — the rule the v4 gate table already applies to
    a missing number, applied here to a missing label source.
    """
    adjudicated_path = v4_1_dir / ADJUDICATED_FILENAME
    key_path = v4_1_dir / KEY_FILENAME
    judge_path = v4_1_dir / "LABEL_AUDIT_JUDGE_A.jsonl"
    missing = [str(p) for p in (adjudicated_path, key_path, judge_path) if not p.exists()]
    if missing:
        return {
            "measured": False,
            "reason": f"the blinded label audit has not produced {missing}",
            "next": (
                "rdl graph-detector-v4-label-audit, two judges, then "
                "rdl graph-detector-v4-label-report"
            ),
            "why_this_blocks": (
                "Goal A's target is human answer_attempt. Scoring the detector against the "
                "NLI+ROUGE leaking label instead is the v4 error: it charges a false alarm "
                "for every wrong answer attempt, which Goal A requires the detector to tag."
            ),
        }

    def load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    key = json.loads(key_path.read_text(encoding="utf-8")).get("rows", {})
    text_of = {r["audit_id"]: r for r in load(judge_path)}
    rows: list[dict] = []
    for row in load(adjudicated_path):
        audit_id = str(row["audit_id"])
        blind = text_of.get(audit_id)
        hidden = key.get(audit_id, {})
        if blind is None or row.get("answer_attempt") is None:
            continue
        rows.append(
            {
                "audit_id": audit_id,
                "request": blind["protected_question"],
                "text": blind["candidate_text"],
                "concept_id": hidden.get("concept_id", ""),
                "population": hidden.get("population", "protected"),
                "nli_leaking": hidden.get("nli_leaking"),
                "answer_attempt": row["answer_attempt"],
                "reference_content": row.get("reference_content"),
                "question_type": row.get("question_type"),
            }
        )
    if not rows:
        return {"measured": False, "reason": "no adjudicated rows carry an answer_attempt label"}

    scores = _natural_scores(rows, base, questions, index)
    # Content-addressed halving, as everywhere else in v4: select on one half, report on
    # the other, so the reported number is not a description of the selection.
    dev = [i for i, r in enumerate(rows) if int(r["audit_id"][:2], 16) % 2 == 0]
    rest = [i for i, r in enumerate(rows) if int(r["audit_id"][:2], 16) % 2 == 1]

    def measure(idxs: Sequence[int], threshold: float) -> dict:
        return goal_a_summarise(
            [rows[i] for i in idxs], [scores[i] for i in idxs], threshold=threshold
        )

    grid = [measure(dev, t) for t in breakpoints([s["answer_probability"] for s in scores])]
    eligible = [
        g
        for g in grid
        if g["answer_attempt_micro_recall"] is not None
        and (g["protected_nonanswer_fpr"] or 0.0) <= GOAL_A_GATES["protected_nonanswer_fpr"][1]
        and (g["retain_fpr"] or 0.0) <= GOAL_A_GATES["retain_fpr"][1]
    ]
    chosen = max(
        eligible, key=lambda g: (g["answer_attempt_micro_recall"], -g["threshold"]), default=None
    )
    selected = float(chosen["threshold"]) if chosen else None
    reported = measure(rest, selected) if selected is not None else None

    return {
        "measured": True,
        "is_one_shot_gate": False,
        "status": "ENGINEERING ONLY",
        "why_not_a_gate": (
            "these rows are drawn from DETECTOR_V4_NATURAL_BANK.json, which both the "
            "oracle and the lexical detector have already been run on. V4_1_DECISION.json "
            "marks it engineering-only. The one-shot gate opens the fresh bank "
            "pre-registered in FINAL_GATE_BANK_MANIFEST.json, which does not exist yet."
        ),
        "n_rows": len(rows),
        "n_selection_rows": len(dev),
        "n_reported_rows": len(rest),
        "selection": {
            "grid": grid,
            "selected_threshold": selected,
            "grid_rule": (
                "exact score breakpoints, not a fixed 0.05 grid. A coarse grid reports "
                "the recall of a threshold no row realises."
            ),
            "constraint": (
                f"protected_nonanswer_fpr <= {GOAL_A_GATES['protected_nonanswer_fpr'][1]} and "
                f"retain_fpr <= {GOAL_A_GATES['retain_fpr'][1]}"
            ),
        },
        "reported": reported,
        **score_goal_a(goal_a_gate_inputs(reported) if reported else dict.fromkeys(GOAL_A_GATES)),
    }


def detector_v4_gates(
    data_dir: Path = typer.Option(DEFAULT_OUT, "--data-dir"),
    policy_cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/discovery.json"), "--policy-cohort"
    ),
    backend: str = typer.Option("lexical", "--backend", help=f"one of {BACKENDS}"),
    model_artifact: Path | None = typer.Option(
        None, "--model-artifact", help="DETECTOR_V4_MODEL.json, for --backend cross_encoder"
    ),
    v4_1_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--v4-1-dir"),
    device: str = typer.Option(
        "", "--device", help="cuda | cuda:0 | cpu, for --backend cross_encoder"
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Select an operating point on development, open held-out once, write the gates."""
    # Checked BEFORE any work: a run that computes for a minute and then refuses to write
    # has spent the minute, and on a GPU box that minute is a model load.
    out = output or (v4_1_dir / V4_1_GATES_FILENAME)
    if out.name == GATES_FILENAME and out.parent == data_dir:
        raise typer.BadParameter(
            f"refusing to overwrite {out}. It is the frozen v4 artifact GU-0036 cites by "
            f"name, and this command now writes a v4.1 report against different "
            f"denominators. The default output is {v4_1_dir / V4_1_GATES_FILENAME}."
        )
    dataset = json.loads((data_dir / DATASET_FILENAME).read_text(encoding="utf-8"))
    # Built once. The sweep re-thresholds this object rather than constructing twenty, so
    # a cross-encoder's weights are loaded exactly once per invocation.
    base = build_backend(backend, model_artifact, device=device)

    # Checked BEFORE any scoring, like the output-path check above. A gate that asked for
    # CUDA and got CPU is not a slower gate: its latency, batching and numerics all belong
    # to a different run than the one the artifact would claim. Finding that out after the
    # sweep would mean discovering it an hour into rented time.
    parameter_device = str(base.to_dict().get("device_of_parameters") or "unknown")
    if device.startswith("cuda") and not parameter_device.startswith("cuda"):
        raise typer.BadParameter(
            f"--device {device} was requested but the model's parameters are on "
            f"{parameter_device!r}. Refusing to run: the artifact's gpu_used field would "
            "assert a GPU that never ran, which is the exact defect v4.1 shipped. Check "
            'that torch sees the card (`python -c "import torch; '
            'print(torch.cuda.is_available())"`) and re-run.'
        )

    # ------------------------------------------------------------- synthetic arm --
    synthetic: dict[str, dict] = {}
    synthetic_grid = []
    # The coarse grid stays HERE alone. This arm re-scores every row at every threshold
    # (the verdict, not a score, is what changes), so an exact-breakpoint sweep would be
    # hundreds of full passes for an arm whose own role field says it is not the verdict.
    for threshold in COARSE_THRESHOLD_GRID:
        detector = _at_threshold(base, threshold)
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
        detector = _at_threshold(base, synthetic_threshold)
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
        detector = base

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
        # Exact breakpoints: the scores are already computed, so every distinct one is a
        # free operating point and the 0.05 grid was only ever an approximation of them.
        for threshold in breakpoints(
            [s["answer_probability"] for s in (*dev_scores, *retain_scores)]
        ):
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
        goal_a = _goal_a_arm(v4_1_dir, base, questions, index)
    else:
        goal_a = {"measured": False, "reason": f"{bank_path} is absent"}

    detector_dict = base.to_dict()

    # -------------------------------------------------------------- what actually ran --
    # Measured after the sweep, so detector_stats reflects the work that was done.
    cuda_available = None
    if backend == "cross_encoder":
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
        except ImportError:  # pragma: no cover - the CPU gate has no torch
            cuda_available = False
    compute = {
        "device_requested": device or None,
        "device_of_parameters": parameter_device,
        "parameters_on_cuda": parameter_device.startswith("cuda"),
        "cuda_available": cuda_available,
        "gpu_name": None,
        "batched_independent_scoring": callable(getattr(base, "score_independent", None)),
        "detector_stats": base.stats() if hasattr(base, "stats") else {},
    }
    if compute["parameters_on_cuda"]:
        try:
            import torch

            compute["gpu_name"] = str(torch.cuda.get_device_name(0))
        except Exception:  # pragma: no cover - reporting, not control flow
            compute["gpu_name"] = "unknown"
    if device.startswith("cuda") and not compute["parameters_on_cuda"]:
        raise typer.BadParameter(
            f"--device {device} was requested but the model's parameters are on "
            f"{parameter_device!r}. Refusing to write a reportable gate: the artifact's "
            "gpu_used field would assert a GPU that never ran, which is the exact defect "
            'v4.1 shipped. Check that torch sees the card (`python -c "import torch; '
            'print(torch.cuda.is_available())"`) and re-run.'
        )

    report = {
        "schema": "graph-detector-v4-gates-v2",
        "phase": (
            "Detector-v4.1 Phase 6 — answerability detector on CPU. No model is trained here."
        ),
        "protocol": "docs/graph_unlearning/DETECTOR_V4_1_PROTOCOL.md",
        "backend_selector": backend,
        "backend": detector_dict,
        "model_artifact": str(model_artifact) if model_artifact else None,
        "backend_is_a_floor": (
            "the lexical scorer is the CPU reference and the floor the trained "
            "cross-encoder must beat. It is not the deployable v4 detector and no report "
            "may present it as one. GraphDetectorConfig cannot select either backend."
            if backend == "lexical"
            else "a trained checkpoint, scored through the same gate arithmetic as the "
            "lexical floor. GraphDetectorConfig cannot select it until its one-shot gate "
            "has been opened on the fresh bank and passed."
        ),
        "ceiling_reinterpretation": CEILING_REINTERPRETATION,
        "gates": {
            "goal_a": {k: {"comparison": c, "bound": b} for k, (c, b) in GOAL_A_GATES.items()},
            "nli_label_diagnostic": {
                k: {"comparison": c, "bound": b} for k, (c, b) in ORACLE_GATES.items()
            },
        },
        "synthetic_arm": {
            "role": "engineering signal; per-class errors. Not the verdict.",
            "selected_threshold": synthetic_threshold,
            "selection_grid": synthetic_grid,
            "dataset_content_sha256": dataset.get("content_sha256"),
            **synthetic,
        },
        "goal_a_arm": goal_a,
        "natural_arm": {
            "role": (
                "DIAGNOSTIC under v4.1. Scored against the run's NLI+ROUGE leaking label, "
                "which answers 'does this reproduce the reference answer'. Goal A asks "
                "'does this attempt to answer', and generated_clean_fpr charges a false "
                "alarm for every wrong answer attempt that Goal A requires to be tagged. "
                "Not the verdict."
            ),
            **natural,
        },
        "registry": registry_meta,
        "verdict": (
            "the Goal A arm was not measured; there is no v4.1 verdict and the label "
            "audit is the next step"
            if not goal_a.get("measured")
            else (
                "the detector clears the Goal A bounds on the adjudicated labels "
                "(ENGINEERING ONLY — the one-shot gate opens the fresh bank)"
                if goal_a.get("all_gates_passed")
                else "the detector does NOT clear the Goal A bounds on the adjudicated labels"
            )
        ),
        "runtime_reads_gold_answers": False,
        "compute": compute,
        "scope": {
            "model_trained": False,
            "graph_generation_run": False,
            "frozen_v1_v2_v3_artifacts_modified": False,
            # Measured, not inferred from the backend name. v4.1 wrote
            # `backend == "cross_encoder"` here while `from_artifact` never moved the
            # model off the CPU, so this field could assert a GPU that never ran.
            "gpu_used": bool(compute["parameters_on_cuda"]),
        },
    }
    atomic_json(out, report)
    typer.echo(f"wrote {out}")
    typer.echo(f"verdict: {report['verdict']}")
    for name, arm in (("goal_a", goal_a), ("nli-diagnostic", natural)):
        for gate in arm.get("gates", ()):
            mark = "PASS" if gate["passed"] else ("n/a " if gate["passed"] is None else "FAIL")
            typer.echo(
                f"  [{mark}] {name} {gate['gate']}: {gate['measured']} "
                f"(need {gate['comparison']} {gate['bound']})"
            )
    raise typer.Exit(0 if goal_a.get("all_gates_passed") else 1)
