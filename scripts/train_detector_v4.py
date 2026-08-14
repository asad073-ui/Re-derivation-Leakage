#!/usr/bin/env python
"""Detector-v4.2 **Phase 7**: fine-tune the answerability cross-encoder on an RTX 3090.

What changed from the v4.1 version of this file
-----------------------------------------------
v4.1 added a real loop, a candidate-preserving encoder and a pinned baseline. Reading it
against what the GPU would actually do turned up six defects, all of which would have
produced numbers rather than errors. They are fixed here and each is named in
``DECISIONS.md`` GU-0038.

**The final accumulation was discarded.** The optimizer stepped on
``step % gradient_accumulation_steps == 0``, so an epoch whose batch count is not a
multiple of the accumulation factor threw away its last partial accumulation — silently,
every epoch. ``steps_per_epoch`` also used floor division, so the LR schedule was built
for a different number of steps than the loop takes. Both use the tail-inclusive form now.

**The NLI entailment index was hard-coded.** ``BASELINE_ENTAILMENT_INDEX = 1``, with a
comment claiming it was confirmed on the box. A moved checkpoint, or a different NLI repo,
silently turns the baseline into a measurement of the contradiction head. It is resolved
from ``model.config.label2id`` / ``id2label``, raises when the entailment class is
ambiguous, and records what it resolved.

**Training and serving saw different inputs.** ``natural_examples`` set ``aliases: []``
while runtime inference receives routed identity aliases — training-serving skew in the
one field routing contributes. The natural rows are joined to the offline concept registry
and carry the same permitted aliases the runtime will pass.

**A checkpoint could not be verified.** The manifest recorded a path. It now records
SHA-256 of the weights, the tokenizer files, the config and the label map, so a copied or
released checkpoint can be checked against the one that produced the numbers.

**Selection was not the objective, and was not frozen.** The trainer picked the best epoch
on general macro F1 while the stated objective is ANSWER-recall under two false-alarm
ceilings. Three fixed seeds are trained, all are reported, and the checkpoint is chosen by
:data:`CHECKPOINT_SELECTION_RULE` — frozen here, before training, and recorded verbatim.

**There was no way to find a GPU-only error cheaply.** ``--smoke`` runs 64 rows for one
epoch, asserts the model and the batches are actually on CUDA, forces at least one
optimizer step, reloads the checkpoint and exercises batched inference. It writes an
artifact marked ``reportable: false`` and is the first thing to run on a rented box.

The readiness gate, and what authorises a run
---------------------------------------------
v4.1's gate is ``LABEL_ALIGNMENT_REPORT.json`` — the audit by two **human** judges. On a
solo project that is a human-time blocker, so v4.2 adds a second acceptable authority:
``DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json``, the same audit annotated by two
model judges. It authorises **engineering training only**, it is recorded in the manifest
by name, and it can never make a result publication-ready. There is still no override
flag: if the gate is wrong, change the gate in the protocol and say so in DECISIONS.md.

Order of operations on the box
------------------------------
1. ``--smoke`` — a non-reportable CUDA check. Minutes, not hours.
2. A passing label report — human (v4.1) or model-judge (v4.2).
3. ``--baseline-only`` — record the pinned zero-shot cross-encoder.
4. Fine-tune the three preregistered seeds.
5. Select the operating point on development (``rdl graph-detector-v4-gates --backend
   cross_encoder --device cuda``).
6. Generate the engineering bank and evaluate. The FINAL bank stays sealed.

Never tune on the v4 held-out data — it is engineering-only — and never on either bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

V4 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4"
V4_1 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_1"
V4_2 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_2"
LABELS = ("NONE", "PARTIAL", "ANSWER")
ANSWER_INDEX = LABELS.index("ANSWER")

# Three fixed seeds, frozen before training. All three are reported; one checkpoint is
# selected across all (seed, epoch) pairs by the rule below. Reporting one seed and
# calling it the result would make the number a draw from a distribution nobody showed.
PREREGISTERED_SEEDS = (20260814, 20260815, 20260816)

# Frozen before training. Recorded verbatim in the manifest so a reader can check that the
# checkpoint was chosen by this rule rather than by whichever number came out best.
CHECKPOINT_SELECTION_RULE = (
    "Across all (seed, epoch) pairs, select the checkpoint with the highest development "
    "ANSWER-recall subject to development NONE-false-alarm-rate <= 0.10, where recall and "
    "FPR are read at the best EXACT score breakpoint on the development pool. Ties break "
    "toward the lower threshold, then the earlier epoch, then the smaller seed. Macro F1 "
    "is recorded as a diagnostic and selects nothing. This is a clause-level proxy for "
    "Goal A and is NOT the operating point: the reportable threshold is chosen on ROUTED "
    "rows by `rdl graph-detector-v4-gates`, under both the protected-nonanswer and retain "
    "ceilings, which a clause pool cannot express."
)
DEV_FPR_CEILING = 0.10

# A pinned pretrained NLI cross-encoder. Its head is TRAINED, so a zero-shot number from it
# is a statement about what pretraining already knows; deberta-v3-base's fresh head is not.
DEFAULT_BASELINE_REPO = "cross-encoder/nli-deberta-v3-base"
# No hard-coded index. The entailment class is resolved from the checkpoint's own label
# map; see resolve_entailment_index.
ENTAILMENT_ALIASES = ("entailment", "entail", "label_0_entailment")


@dataclass(frozen=True)
class TrainingPins:
    """Everything that changes the checkpoint. All of it lands in DETECTOR_V4_MODEL.json."""

    model_repo_id: str = "microsoft/deberta-v3-base"
    # No default for either revision, deliberately. A revision is a claim about which
    # weights and which subword split were used; a guessed one is worse than an absent one.
    # Resolve on the box with `huggingface_hub.HfApi().model_info(repo).sha`.
    model_revision: str = ""
    tokenizer_revision: str = ""
    max_length: int = 256
    min_question_tokens: int = 8
    max_aliases: int = 4
    seeds: tuple[int, ...] = PREREGISTERED_SEEDS
    optimizer: str = "adamw_torch"
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    max_grad_norm: float = 1.0
    fp16: bool = True
    class_weighting: str = "inverse_frequency"
    checkpoint_selection: str = CHECKPOINT_SELECTION_RULE
    expected_device: str = "cuda"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["seeds"] = list(self.seeds)
        return out


def require_cuda(device: str) -> str:
    try:
        import torch
    except ImportError:
        raise SystemExit("torch is not installed; this script runs on the GPU box only") from None
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This is the RTX phase; the CPU phase is "
            "`rdl graph-detector-v4-2-llm-judge` and `rdl graph-detector-v4-2-label-report`."
        )
    return str(torch.cuda.get_device_name(0)) if device.startswith("cuda") else "cpu"


# ------------------------------------------------------------------- the label gate --

HUMAN_REPORT = "LABEL_ALIGNMENT_REPORT.json"
MODEL_REPORT = "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json"


def require_label_audit(v4_1_dir: Path, v4_2_dir: Path) -> dict:
    """Refuse to train until a label report says the target is legible. No override.

    Two authorities are acceptable and they are NOT equivalent:

    ``LABEL_ALIGNMENT_REPORT.json``
        v4.1. Two human judges. The only authority that can back a publication claim.

    ``DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json``
        v4.2. Two model judges. Authorises an ENGINEERING run. It carries
        ``human_grounded: false`` and ``publication_label_valid: false``, and this
        function refuses it if those flags have been edited to say otherwise — a report
        claiming human grounding from a model-judge schema is a tampered artifact, and
        loading it would put the claim into the training manifest.

    The human report wins when both exist. v4's ``--force-despite-failed-ceiling``
    existed to train despite a failing artifact; the correct response to that artifact was
    to fix its interpretation, and a flag that skips the fix is a way to keep the mistake.
    """
    human = v4_1_dir / HUMAN_REPORT
    model = v4_2_dir / MODEL_REPORT
    if human.exists():
        report = json.loads(human.read_text(encoding="utf-8"))
        authority = {
            "report": str(human),
            "judge_population": "two_human_judges",
            "human_grounded": True,
            "publication_label_valid": bool(report.get("all_gates_passed")),
            "authorises": "engineering and publication claims",
        }
    elif model.exists():
        report = json.loads(model.read_text(encoding="utf-8"))
        if report.get("human_grounded") is not False:
            raise SystemExit(
                f"{model} claims human_grounded={report.get('human_grounded')!r}. The "
                "v4.2 schema is model-judge output and that flag is false by "
                "construction; this file has been edited. Refusing to train on it."
            )
        if report.get("publication_label_valid") is not False:
            raise SystemExit(
                f"{model} claims publication_label_valid="
                f"{report.get('publication_label_valid')!r}. Two model judges agreeing is "
                "consistency evidence, not correctness. Refusing to train on it."
            )
        authority = {
            "report": str(model),
            "judge_population": str(report.get("judge_population")),
            "human_grounded": False,
            "publication_label_valid": False,
            "authorises": "ENGINEERING training and evaluation only",
            "warning": (
                "This run is authorised by MODEL judges. No result from it may be "
                "described as a Goal A result or as publication-ready. See "
                "DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md sections 2 and 10."
            ),
        }
    else:
        raise SystemExit(
            f"neither {human} nor {model} exists. Run a label audit first:\n"
            "  human (v4.1):  rdl graph-detector-v4-label-audit\n"
            "                 rdl graph-detector-v4-label-report --judge-a ... --judge-b ...\n"
            "  model (v4.2):  rdl graph-detector-v4-2-llm-judge --judge A --pass blind\n"
            "                 rdl graph-detector-v4-2-llm-judge --judge B --pass blind\n"
            "                 rdl graph-detector-v4-2-label-report\n"
            "Goal A's target is answer_attempt, and no GPU step precedes it."
        )

    if not report.get("all_gates_passed"):
        raise SystemExit(
            "the label audit does NOT clear its decision gate: "
            f"{report.get('failed_gates')}\n{report.get('verdict')}\n"
            "Fix the label protocol, not the model."
        )
    authority["all_gates_passed"] = True
    authority["answer_attempt_kappa"] = (
        report.get("inter_judge", {})
        .get("per_field", {})
        .get("answer_attempt", {})
        .get("cohens_kappa")
    )
    return authority


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # cudnn conv kernels; fp16 matmul is not exact


# ------------------------------------------------------------------------- data --


def synthetic_examples(dataset: dict, split: str) -> list[dict]:
    """``{question, aliases, candidate, label}`` per clause of every row in ``split``.

    One example per clause, not per row: the runtime scores clauses, and training on whole
    messages would fit a different unit than the one the threshold is chosen on.
    """
    from rdl.defenses.atomic_text import segment

    out: list[dict] = []
    for row in dataset["rows"]:
        if row["split"] != split:
            continue
        for candidate in row["candidates"]:
            for clause in segment(candidate) or ():
                out.append(
                    {
                        "question": row["protected_question"],
                        "aliases": list(row["aliases"]),
                        "candidate": clause.text,
                        "label": row["label"],
                        "split": split,
                        "source": "synthetic",
                        "group": row["subject"],
                    }
                )
    return out


def natural_alias_index(policy_cohort: Path, max_aliases: int) -> dict[str, list[str]]:
    """``{protected_question -> aliases}`` from the OFFLINE concept registry.

    The fix for the training-serving skew. v4.1 trained the natural rows with
    ``aliases: []`` while the runtime hands the model routed identity aliases in segment
    A — so the model was fitted on an input shape it is never served. This joins the
    adjudicated rows back to the same registry the gate builds, keyed by the question text
    the judge file carries, which is the only key the blinded row exposes.

    Built from committed files alone, with no dataset load and therefore no network, in
    the same way ``detector_v4_label_audit._registry_from_bank`` is. An unmatched question
    yields no aliases and is counted rather than silently becoming an empty list.
    """
    from rdl.defenses.concept_registry import ConceptPolicy, ConceptRegistry
    from rdl.studies.graph_leak.cohort import load_cohort

    bank_path = V4 / "DETECTOR_V4_NATURAL_BANK.json"
    if not (bank_path.exists() and policy_cohort.exists()):
        return {}
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    cohort = load_cohort(policy_cohort, exclusions_path=None)
    concept_of = {e.item_id: e.concept_id for e in cohort.items}

    seen: dict[tuple[str, str], str] = {}
    for partition in ("development", "heldout"):
        block = bank["partitions"].get(partition, {})
        for keyname in ("clean", "leaking"):
            for row in block.get(keyname, ()):
                item_id = str(row.get("item_id", ""))
                concept = concept_of.get(item_id)
                if concept and row.get("request"):
                    seen[(item_id, str(row["request"]))] = concept
    rows = [
        {"item_id": item_id, "concept_id": concept, "question": question}
        for (item_id, question), concept in sorted(seen.items())
    ]
    registry = ConceptRegistry.from_questions(rows, policy=ConceptPolicy())
    aliases_of_concept = {c.forget_id: list(c.aliases)[:max_aliases] for c in registry.concepts()}
    return {
        str(row["question"]): aliases_of_concept.get(str(row["concept_id"]), []) for row in rows
    }


def natural_examples(
    v4_1_dir: Path,
    v4_2_dir: Path,
    split: str,
    *,
    alias_index: dict[str, list[str]] | None = None,
) -> tuple[list[dict], dict]:
    """Adjudicated natural rows joined to their blinded text and to their routed aliases.

    Accepts either adjudication overlay — v4.1's human one or v4.2's model one. The
    blinded judge file supplies the question and the candidate; neither overlay carries a
    reference answer, which is why the join can be done here at all.
    """
    sources = (
        (v4_1_dir / "LABEL_AUDIT_ADJUDICATED.jsonl", v4_1_dir / "LABEL_AUDIT_JUDGE_A.jsonl"),
        (v4_2_dir / "V4_2_ADJUDICATED.jsonl", v4_1_dir / "LABEL_AUDIT_JUDGE_A.jsonl"),
    )
    adjudicated_path, judge_path = next(
        ((a, j) for a, j in sources if a.exists() and j.exists()), (None, None)
    )
    if adjudicated_path is None:
        return [], {"source": None, "n_rows": 0, "n_without_aliases": 0}

    def load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    alias_index = alias_index or {}
    text_of = {r["audit_id"]: r for r in load(judge_path)}
    out: list[dict] = []
    n_without = 0
    for row in load(adjudicated_path):
        label = row.get("answer_attempt")
        blind = text_of.get(row["audit_id"])
        if label not in LABELS or blind is None:
            continue
        # Content-addressed train/development halving, computed from the audit id alone so
        # it does not move when rows are added.
        half = "train" if int(row["audit_id"][:2], 16) % 2 == 0 else "development"
        if half != split:
            continue
        question = str(blind["protected_question"])
        aliases = list(alias_index.get(question, ()))
        n_without += int(not aliases)
        out.append(
            {
                "question": question,
                "aliases": aliases,
                "candidate": blind["candidate_text"],
                "label": label,
                "source": "natural_adjudicated",
                "group": row["audit_id"],
            }
        )
    return out, {
        "source": str(adjudicated_path),
        "n_rows": len(out),
        "n_without_aliases": n_without,
        "why_aliases": (
            "runtime inference receives routed identity aliases; v4.1 trained these rows "
            "with an empty alias list, which is training-serving skew in the one field "
            "the router contributes."
        ),
    }


def extra_examples(paths: list[Path]) -> list[dict]:
    """Public QA/answerability rows, as ``{question, candidate, label}`` JSONL."""
    out: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") not in LABELS:
                raise SystemExit(f"{path}: label must be one of {LABELS}, got {row.get('label')!r}")
            out.append(
                {
                    "question": str(row["question"]),
                    "aliases": list(row.get("aliases", ())),
                    "candidate": str(row["candidate"]),
                    "label": str(row["label"]),
                    "source": f"extra:{path.name}",
                    "group": str(row.get("group", path.name)),
                }
            )
    return out


class AnswerabilityDataset:
    """Encodes lazily through ``budget_encode`` and counts every candidate truncation."""

    def __init__(self, examples: list[dict], tokenizer, budget) -> None:
        from rdl.defenses.cross_encoder_answerability import budget_encode

        self._encode = budget_encode
        self.examples = examples
        self.tokenizer = tokenizer
        self.budget = budget
        self.n_candidate_truncations = 0
        self.n_question_truncations = 0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        row = self.examples[i]
        encoding, stats = self._encode(
            self.tokenizer,
            question=row["question"],
            aliases=row["aliases"],
            candidate=row["candidate"],
            budget=self.budget,
        )
        self.n_candidate_truncations += int(bool(stats["candidate_truncated"]))
        self.n_question_truncations += int(bool(stats["question_truncated"]))
        return {**encoding, "labels": LABELS.index(row["label"])}


def collate(features: list[dict], pad_id: int) -> dict:
    import torch

    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in features[0]]
    width = max(len(f["input_ids"]) for f in features)
    batch = {
        key: torch.tensor(
            [
                list(f[key]) + [pad_id if key == "input_ids" else 0] * (width - len(f[key]))
                for f in features
            ],
            dtype=torch.long,
        )
        for key in keys
    }
    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return batch


# -------------------------------------------------------------------- evaluation --


def evaluate(model, loader, device) -> dict:
    """Per-class metrics, macro F1, and the Goal A selection numbers on development.

    ``macro_f1`` is a diagnostic and selects nothing (see
    :data:`CHECKPOINT_SELECTION_RULE`). The number that selects is
    ``selection_answer_recall``: ANSWER-recall at the best EXACT score breakpoint whose
    NONE false-alarm rate clears :data:`DEV_FPR_CEILING`. Exact breakpoints rather than a
    0.05 grid, because a coarse grid reports the recall of a threshold nobody chose.
    """
    import torch

    model.eval()
    confusion = [[0] * len(LABELS) for _ in LABELS]
    answer_scores: list[float] = []
    golds: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float().cpu()
            probabilities = logits.softmax(-1)
            for gold, predicted, row in zip(
                labels.tolist(),
                logits.argmax(-1).tolist(),
                probabilities.tolist(),
                strict=True,
            ):
                confusion[gold][predicted] += 1
                answer_scores.append(float(row[ANSWER_INDEX]))
                golds.append(int(gold))

    per_class = {}
    f1s = []
    for i, name in enumerate(LABELS):
        tp = confusion[i][i]
        fp = sum(confusion[g][i] for g in range(len(LABELS))) - tp
        fn = sum(confusion[i]) - tp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall)
            else 0.0
        )
        per_class[name] = {
            "support": sum(confusion[i]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1s.append(f1)
    total = sum(sum(row) for row in confusion)
    return {
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "macro_f1_role": "diagnostic; selects nothing",
        "accuracy": (sum(confusion[i][i] for i in range(len(LABELS))) / total) if total else None,
        "per_class": per_class,
        "confusion": {
            LABELS[i]: dict(zip(LABELS, confusion[i], strict=True)) for i in range(len(LABELS))
        },
        **selection_metrics(answer_scores, golds),
    }


def selection_metrics(answer_scores: list[float], golds: list[int]) -> dict:
    """ANSWER-recall at the best exact breakpoint clearing the NONE-FPR ceiling."""
    positives = [s for s, g in zip(answer_scores, golds, strict=True) if g == ANSWER_INDEX]
    negatives = [s for s, g in zip(answer_scores, golds, strict=True) if g == LABELS.index("NONE")]
    if not positives or not negatives:
        return {
            "selection_answer_recall": None,
            "selection_threshold": None,
            "selection_none_fpr": None,
            "selection_note": "development pool lacks ANSWER or NONE rows",
        }
    # Exact breakpoints: every score is a candidate operating point, plus one above the
    # maximum so "fire on nothing" is representable.
    breakpoints = sorted({*answer_scores, max(answer_scores) + 1e-9})
    best: tuple[float, float, float] | None = None  # (recall, -threshold, fpr)
    for threshold in breakpoints:
        fpr = sum(1 for s in negatives if s >= threshold) / len(negatives)
        if fpr > DEV_FPR_CEILING:
            continue
        recall = sum(1 for s in positives if s >= threshold) / len(positives)
        candidate = (recall, -threshold, fpr)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return {
            "selection_answer_recall": 0.0,
            "selection_threshold": None,
            "selection_none_fpr": None,
            "selection_note": f"no threshold clears NONE FPR <= {DEV_FPR_CEILING}",
        }
    recall, negative_threshold, fpr = best
    return {
        "selection_answer_recall": recall,
        "selection_threshold": -negative_threshold,
        "selection_none_fpr": fpr,
        "selection_note": (
            "clause-level proxy on the development pool. NOT the reportable operating "
            "point, which `rdl graph-detector-v4-gates` chooses on routed rows under both "
            "the protected-nonanswer and retain ceilings."
        ),
    }


def resolve_entailment_index(config) -> tuple[int, dict]:
    """Find the entailment class in an NLI head's own label map. Ambiguity raises.

    v4.1 hard-coded ``1`` with a comment saying it was confirmed on the box. A moved
    checkpoint or a different NLI repo silently turns the baseline into a measurement of
    the contradiction head, and the number still looks like an AUC.
    """
    id2label = dict(getattr(config, "id2label", {}) or {})
    label2id = dict(getattr(config, "label2id", {}) or {})
    if not id2label and not label2id:
        raise SystemExit(
            "the baseline checkpoint exposes no label map, so the entailment class cannot "
            "be resolved. Pass --baseline-entailment-label with the exact label name."
        )
    if not id2label:
        id2label = {int(v): str(k) for k, v in label2id.items()}
    matches = [
        int(index)
        for index, name in id2label.items()
        if str(name).strip().lower().replace("-", "_") in ENTAILMENT_ALIASES
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"cannot resolve a unique entailment class from {id2label}. Found {matches}. "
            "Pass --baseline-entailment-label with the exact label name; a guessed index "
            "makes the baseline a measurement of a different head."
        )
    return matches[0], {"id2label": {int(k): str(v) for k, v in id2label.items()}}


def baseline_zero_shot(
    repo_id: str,
    revision: str,
    examples: list[dict],
    budget,
    device,
    *,
    entailment_label: str = "",
) -> dict:
    """Entailment mass from a pinned pretrained NLI cross-encoder, as an answerability proxy.

    Reported as a ranking baseline, not as a classifier: 'the candidate entails an answer to
    the question' and 'the candidate attempts to answer the question' are different
    predicates, and the number is here so the fine-tune's contribution is attributable
    rather than so it can be deployed.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from rdl.defenses.cross_encoder_answerability import budget_encode

    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, revision=revision)
    model.to(device).eval()

    if entailment_label:
        label2id = {str(k).lower(): int(v) for k, v in (model.config.label2id or {}).items()}
        if entailment_label.lower() not in label2id:
            raise SystemExit(
                f"--baseline-entailment-label {entailment_label!r} is not in " f"{sorted(label2id)}"
            )
        index = label2id[entailment_label.lower()]
        label_map = {"id2label": dict(model.config.id2label or {}), "override": entailment_label}
    else:
        index, label_map = resolve_entailment_index(model.config)

    scores: list[float] = []
    golds: list[str] = []
    for row in examples:
        encoding, _stats = budget_encode(
            tokenizer,
            question=row["question"],
            aliases=row["aliases"],
            candidate=row["candidate"],
            budget=budget,
        )
        batch = {k: torch.tensor([v], dtype=torch.long).to(device) for k, v in encoding.items()}
        with torch.inference_mode():
            probabilities = model(**batch).logits.float().softmax(-1)[0].cpu().tolist()
        scores.append(float(probabilities[index]))
        golds.append(row["label"])

    positives = [s for s, g in zip(scores, golds, strict=True) if g == "ANSWER"]
    negatives = [s for s, g in zip(scores, golds, strict=True) if g == "NONE"]
    pairs = len(positives) * len(negatives)
    auc = (
        (
            sum(1 for p in positives for n in negatives if p > n)
            + 0.5 * sum(1 for p in positives for n in negatives if p == n)
        )
        / pairs
        if pairs
        else None
    )
    return {
        "repo_id": repo_id,
        "revision": revision,
        "entailment_index": index,
        "entailment_index_resolved_from": label_map,
        "n_scored": len(scores),
        "n_answer": len(positives),
        "n_none": len(negatives),
        "answer_vs_none_auc": auc,
        "mean_entailment_on_answer": (sum(positives) / len(positives)) if positives else None,
        "mean_entailment_on_none": (sum(negatives) / len(negatives)) if negatives else None,
        "note": (
            "zero-shot ranking baseline from a pinned pretrained NLI head. "
            "microsoft/deberta-v3-base is NOT an answerability baseline: its "
            "classification head is randomly initialised, so its off-the-shelf number "
            "measures that random head."
        ),
    }


# ------------------------------------------------------------------ checkpoint hashes --

HASHED_FILE_GLOBS = (
    "*.safetensors",
    "pytorch_model.bin",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "spm.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hashes(path: Path) -> dict:
    """SHA-256 of everything that makes the checkpoint what it is.

    v4.1 recorded a local path. A path is not a checkpoint: a copied, re-exported or
    re-uploaded directory has the same path in a manifest and different bytes on disk, and
    nobody could tell which one produced the numbers. The label map is hashed separately
    because a permuted ``id2label`` changes every score at a fixed threshold while leaving
    the weights byte-identical.
    """
    files: dict[str, str] = {}
    for pattern in HASHED_FILE_GLOBS:
        for candidate in sorted(path.glob(pattern)):
            if candidate.is_file():
                files[candidate.name] = _sha256_file(candidate)
    label_map = None
    config_path = path / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        label_map = {
            "id2label": config.get("id2label"),
            "label2id": config.get("label2id"),
        }
    return {
        "files_sha256": files,
        "n_files_hashed": len(files),
        "label_map": label_map,
        "label_map_sha256": (
            hashlib.sha256(json.dumps(label_map, sort_keys=True).encode("utf-8")).hexdigest()
            if label_map
            else None
        ),
        "expected_label_order": list(LABELS),
        "why": (
            "a path is not a checkpoint. These hashes let a copied or released directory "
            "be checked against the one that produced the reported numbers."
        ),
    }


# ---------------------------------------------------------------------- one seed --


def train_one_seed(
    *,
    seed: int,
    pins: TrainingPins,
    train_rows: list[dict],
    dev_rows: list[dict],
    tokenizer,
    budget,
    output_dir: Path,
    device: str,
    epochs: int,
    min_optimizer_steps: int = 0,
) -> dict:
    """Train one seed. Returns its history and per-epoch development metrics."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        pins.model_repo_id, revision=pins.model_revision, num_labels=len(LABELS)
    ).to(device)
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {name: i for i, name in enumerate(LABELS)}

    train_set = AnswerabilityDataset(train_rows, tokenizer, budget)
    dev_set = AnswerabilityDataset(dev_rows, tokenizer, budget)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=pins.per_device_train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda f: collate(f, pad_id),
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=pins.per_device_train_batch_size * 2,
        shuffle=False,
        collate_fn=lambda f: collate(f, pad_id),
    )

    # Inverse-frequency class weights. The Goal A classes are not balanced — PARTIAL is
    # rare and is the class whose confusion with ANSWER makes the accumulation result
    # untestable — so an unweighted loss would learn to never predict it.
    counts = [max(1, sum(1 for r in train_rows if r["label"] == name)) for name in LABELS]
    weights = torch.tensor(
        [len(train_rows) / (len(LABELS) * c) for c in counts], dtype=torch.float
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=pins.learning_rate, weight_decay=pins.weight_decay
    )
    # Ceiling division, and the loop below steps on the tail. v4.1 used floor division
    # here and `step % k == 0` there, so the schedule was built for fewer steps than the
    # loop takes AND the last partial accumulation was discarded — two halves of the same
    # bug, each of which hides the other in the logs.
    steps_per_epoch = max(1, math.ceil(len(train_loader) / pins.gradient_accumulation_steps))
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * pins.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=pins.fp16 and device.startswith("cuda"))

    history: list[dict] = []
    n_optimizer_steps = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        n_batches = len(train_loader)
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=pins.fp16 and device.startswith("cuda")):
                logits = model(**batch).logits
                loss = loss_fn(logits.float(), labels) / pins.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running += float(loss.item()) * pins.gradient_accumulation_steps
            # The tail. An epoch whose batch count is not a multiple of the accumulation
            # factor ended with gradients in the buffer and no step; they were discarded
            # at the next zero_grad, every epoch, invisibly.
            should_step = (step % pins.gradient_accumulation_steps == 0) or (step == n_batches)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), pins.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                n_optimizer_steps += 1

        metrics = evaluate(model, dev_loader, device)
        path = output_dir / f"seed{seed}-checkpoint-epoch{epoch}"
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": running / max(1, n_batches),
                "development": metrics,
                "checkpoint": str(path),
                "checkpoint_hashes": checkpoint_hashes(path),
            }
        )
        print(
            f"seed {seed} epoch {epoch}: "
            f"selection_recall={metrics['selection_answer_recall']} "
            f"macro_f1={metrics['macro_f1']:.4f} loss={running / max(1, n_batches):.4f}"
        )

    if n_optimizer_steps < min_optimizer_steps:
        raise SystemExit(
            f"only {n_optimizer_steps} optimizer steps were taken; the smoke mode "
            f"requires at least {min_optimizer_steps}. The loop is not training."
        )
    return {
        "seed": seed,
        "history": history,
        "n_optimizer_steps": n_optimizer_steps,
        "steps_per_epoch": steps_per_epoch,
        "planned_total_optimizer_steps": total_steps,
        "wall_clock_seconds": round(time.time() - started, 1),
        "class_weights": dict(zip(LABELS, weights.detach().cpu().tolist(), strict=True)),
        "n_candidate_truncations": train_set.n_candidate_truncations,
        "n_question_truncations": train_set.n_question_truncations,
        "n_development_candidate_truncations": dev_set.n_candidate_truncations,
    }


def select_checkpoint(seed_results: list[dict]) -> dict:
    """Apply :data:`CHECKPOINT_SELECTION_RULE` across every (seed, epoch) pair."""
    candidates = [entry for result in seed_results for entry in result["history"]]
    eligible = [
        entry
        for entry in candidates
        if entry["development"].get("selection_answer_recall") is not None
        and (entry["development"].get("selection_none_fpr") is not None)
        and entry["development"]["selection_none_fpr"] <= DEV_FPR_CEILING
    ]
    if not eligible:
        return {
            "selected": None,
            "reason": (
                f"no (seed, epoch) pair reached a development threshold with NONE FPR "
                f"<= {DEV_FPR_CEILING}. The rule does not fall back to macro F1; a "
                "checkpoint selected by a different rule than the frozen one is not the "
                "checkpoint the protocol authorises."
            ),
        }
    best = max(
        eligible,
        key=lambda e: (
            e["development"]["selection_answer_recall"],
            -(e["development"]["selection_threshold"] or 0.0),
            -e["epoch"],
            -e["seed"],
        ),
    )
    return {
        "selected": best,
        "rule": CHECKPOINT_SELECTION_RULE,
        "n_candidates": len(candidates),
        "n_eligible": len(eligible),
    }


# ------------------------------------------------------------------------ smoke --


def run_smoke(args, pins: TrainingPins, budget, tokenizer, train_rows, dev_rows) -> int:
    """A small, explicitly NON-REPORTABLE CUDA check. Minutes, before the real run.

    Catches the class of error that only appears on a GPU — a tensor left on CPU, a
    checkpoint that will not reload, an inference path that was never exercised batched —
    on 64 rows instead of on the full corpus after an hour.
    """
    import torch

    from rdl.defenses.cross_encoder_answerability import CrossEncoderAnswerabilityDetector
    from rdl.defenses.detection_context import ProtectedQuestion, build_context

    device = args.device
    checks: dict[str, object] = {}
    checks["cuda_available"] = bool(torch.cuda.is_available())
    checks["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    train_rows = train_rows[: args.smoke_train_rows]
    dev_rows = dev_rows[: args.smoke_dev_rows]
    if not train_rows or not dev_rows:
        raise SystemExit("the smoke mode needs training and development rows; both pools are empty")

    result = train_one_seed(
        seed=pins.seeds[0],
        pins=pins,
        train_rows=train_rows,
        dev_rows=dev_rows,
        tokenizer=tokenizer,
        budget=budget,
        output_dir=args.output_dir,
        device=device,
        epochs=1,
        min_optimizer_steps=1,
    )
    checks["optimizer_steps"] = result["n_optimizer_steps"]
    checkpoint = Path(result["history"][-1]["checkpoint"])

    if device.startswith("cuda"):
        checks["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        if not checks["peak_memory_bytes"]:
            raise SystemExit("CUDA was requested and peak allocation is zero; nothing ran on it")

    # Reload, and confirm the reloaded model's parameters are actually on the device.
    from transformers import AutoModelForSequenceClassification

    reloaded = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    reloaded.eval()
    checks["reloaded_parameter_device"] = str(next(reloaded.parameters()).device)
    if device.startswith("cuda") and not checks["reloaded_parameter_device"].startswith("cuda"):
        raise SystemExit("the reloaded checkpoint is not on CUDA")

    # Batched independent inference, through the class the gate uses.
    detector = CrossEncoderAnswerabilityDetector(
        reloaded, tokenizer, model_repo_id=pins.model_repo_id, device=device, batch_size=8
    )
    question = ProtectedQuestion(
        scope_id="smoke#000",
        forget_id="smoke",
        question=dev_rows[0]["question"],
        aliases=tuple(dev_rows[0]["aliases"]),
        template_id="smoke",
    )
    context = build_context(
        dev_rows[0]["question"],
        protected_questions=[question],
        alias_index={"smoke": list(question.alias_token_sets)},
    )
    scores = detector.score_independent(
        [(context, r["candidate"]) for r in dev_rows[:16]], restrict_to=None
    )
    checks["independent_batch_scored"] = len(scores)
    checks["independent_batch_order_preserved"] = len(scores) == len(dev_rows[:16])
    checks["forward_pairs"] = detector.stats()["forward_pairs"]
    checks["detector_parameter_device"] = detector.parameter_device()

    artifact = {
        "schema": "graph-detector-v4-2-gpu-smoke-v1",
        "reportable": False,
        "why_not_reportable": (
            "64 training rows and one epoch. This measures that the GPU path RUNS, not "
            "how well it performs. No number here may appear in a result."
        ),
        "requested_device": device,
        "checks": checks,
        "checkpoint": str(checkpoint),
        "checkpoint_hashes": checkpoint_hashes(checkpoint),
        "versions": _versions(),
    }
    path = args.output_dir / "DETECTOR_V4_2_GPU_SMOKE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print("SMOKE PASSED (non-reportable)")
    return 0


# -------------------------------------------------------------------------- main --


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=V4)
    parser.add_argument("--v4-1-dir", type=Path, default=V4_1)
    parser.add_argument("--v4-2-dir", type=Path, default=V4_2)
    parser.add_argument(
        "--policy-cohort",
        type=Path,
        default=REPO / "data" / "cohorts" / "graph_unlearning_v1" / "discovery.json",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs" / "detector_v4")
    parser.add_argument("--model-repo-id", default=TrainingPins.model_repo_id)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--tokenizer-revision", default="")
    parser.add_argument("--baseline-repo-id", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-revision", default="")
    parser.add_argument("--baseline-entailment-label", default="")
    parser.add_argument("--extra-train", type=Path, nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=TrainingPins.num_train_epochs)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(PREREGISTERED_SEEDS),
        help="preregistered; all are trained and all are reported",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="non-reportable CUDA check: 64 rows, one epoch, batched inference verified",
    )
    parser.add_argument("--smoke-train-rows", type=int, default=128)
    parser.add_argument("--smoke-dev-rows", type=int, default=64)
    args = parser.parse_args()

    device_name = require_cuda(args.device)

    if not args.model_revision or not args.tokenizer_revision:
        raise SystemExit(
            "--model-revision and --tokenizer-revision are both required. A moved model "
            "tag changes the weights; a moved tokenizer tag changes the subword split, "
            "which changes every score at a fixed threshold. Record both even when they "
            "are equal."
        )
    pins = TrainingPins(
        model_repo_id=args.model_repo_id,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        seeds=tuple(args.seeds),
        num_train_epochs=args.epochs,
    )

    from transformers import AutoTokenizer

    from rdl.defenses.cross_encoder_answerability import EncodingBudget

    budget = EncodingBudget(
        max_length=pins.max_length,
        min_question_tokens=pins.min_question_tokens,
        max_aliases=pins.max_aliases,
    )
    tokenizer = AutoTokenizer.from_pretrained(pins.model_repo_id, revision=pins.tokenizer_revision)

    dataset = json.loads((args.data_dir / "DETECTOR_V4_DATASET.json").read_text(encoding="utf-8"))
    alias_index = natural_alias_index(args.policy_cohort, pins.max_aliases)
    natural_train, natural_train_meta = natural_examples(
        args.v4_1_dir, args.v4_2_dir, "train", alias_index=alias_index
    )
    natural_dev, natural_dev_meta = natural_examples(
        args.v4_1_dir, args.v4_2_dir, "development", alias_index=alias_index
    )
    train_rows = [
        *synthetic_examples(dataset, "train"),
        *natural_train,
        *extra_examples(list(args.extra_train)),
    ]
    dev_rows = [*synthetic_examples(dataset, "development"), *natural_dev]
    # The synthetic held-out split is opened by `rdl graph-detector-v4-gates` after the
    # threshold is frozen, and neither gate bank is a file this script names. Checked
    # rather than commented: a filter that silently stopped filtering would be invisible
    # in the manifest, which is where the guarantee is supposed to be readable.
    if any(r.get("split") == "heldout" for r in [*train_rows, *dev_rows]):
        raise SystemExit("a held-out row reached the training or development pool")
    if not train_rows:
        raise SystemExit("no training rows; check --data-dir")

    # ------------------------------------------------------------------- smoke --
    # Before the label gate, deliberately: the smoke mode is what you run on a freshly
    # rented box to find out whether CUDA works, and it must not require labels to exist.
    if args.smoke:
        return run_smoke(args, pins, budget, tokenizer, train_rows, dev_rows)

    authority = require_label_audit(args.v4_1_dir, args.v4_2_dir)

    baseline = {"skipped": True}
    if not args.skip_baseline:
        if not args.baseline_revision:
            raise SystemExit(
                "--baseline-revision is required (or pass --skip-baseline). An unpinned "
                "baseline is a number nobody can reproduce."
            )
        baseline = baseline_zero_shot(
            args.baseline_repo_id,
            args.baseline_revision,
            dev_rows,
            budget,
            args.device,
            entailment_label=args.baseline_entailment_label,
        )
        print(f"baseline AUC (ANSWER vs NONE): {baseline['answer_vs_none_auc']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "schema": "graph-detector-v4-model-v3",
        "phase": "Detector-v4.2 Phase 7 — cross-encoder fine-tune",
        "protocol": "docs/graph_unlearning/DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md",
        "device_requested": args.device,
        "device_name": device_name,
        "pins": pins.to_dict(),
        "encoding_budget": budget.to_dict(),
        "dataset_content_sha256": dataset["content_sha256"],
        "label_authority": authority,
        "human_grounded": bool(authority.get("human_grounded")),
        "publication_label_valid": bool(authority.get("publication_label_valid")),
        "n_train_rows": len(train_rows),
        "n_development_rows": len(dev_rows),
        "train_sources": _counts(train_rows, "source"),
        "development_sources": _counts(dev_rows, "source"),
        "train_label_distribution": _counts(train_rows, "label"),
        "natural_rows": {"train": natural_train_meta, "development": natural_dev_meta},
        "heldout_read_during_training": False,
        "gate_bank_read_during_training": False,
        "final_gate_bank_read_during_training": False,
        "labels": list(LABELS),
        "baseline_zero_shot": baseline,
        "baseline_only": bool(args.baseline_only),
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "development_fpr_ceiling": DEV_FPR_CEILING,
        "versions": _versions(),
    }

    if args.baseline_only:
        _write_manifest(args.output_dir, manifest)
        print("baseline recorded; no fine-tuning requested")
        return 0

    seed_results = [
        train_one_seed(
            seed=seed,
            pins=pins,
            train_rows=train_rows,
            dev_rows=dev_rows,
            tokenizer=tokenizer,
            budget=budget,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.epochs,
        )
        for seed in pins.seeds
    ]
    selection = select_checkpoint(seed_results)
    best = selection.get("selected")

    manifest.update(
        {
            "training": {"seeds": seed_results},
            "selection": selection,
            "selected_checkpoint": best["checkpoint"] if best else None,
            "selected_seed": best["seed"] if best else None,
            "selected_epoch": best["epoch"] if best else None,
            "selected_checkpoint_hashes": best["checkpoint_hashes"] if best else None,
            "development_selection_answer_recall": (
                best["development"]["selection_answer_recall"] if best else None
            ),
            "selected_on": CHECKPOINT_SELECTION_RULE,
            "truncation_why_counted": (
                "a candidate that did not fit the window is an evaluation error, not a "
                "NONE. v4 truncated the candidate by construction; this counts what the "
                "corrected budget still has to cut."
            ),
        }
    )
    _write_manifest(args.output_dir, manifest)
    if not best:
        print(f"NO CHECKPOINT SELECTED: {selection['reason']}")
        return 1
    print(
        f"selected {best['checkpoint']} "
        f"(seed {best['seed']}, epoch {best['epoch']}, "
        f"dev ANSWER recall {best['development']['selection_answer_recall']})"
    )
    print(
        "next: rdl graph-detector-v4-gates --backend cross_encoder --device "
        f"{args.device} --model-artifact {args.output_dir / 'DETECTOR_V4_MODEL.json'}"
    )
    return 0


def _counts(rows: list[dict], field: str) -> dict:
    out: dict[str, int] = {}
    for row in rows:
        out[str(row[field])] = out.get(str(row[field]), 0) + 1
    return dict(sorted(out.items()))


def _write_manifest(output_dir: Path, manifest: dict) -> None:
    path = output_dir / "DETECTOR_V4_MODEL.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _versions() -> dict:
    import importlib

    out: dict[str, str] = {}
    for name in ("torch", "transformers", "tokenizers", "numpy"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:  # pragma: no cover - reporting, not control flow
            out[name] = "absent"
    try:
        import torch

        out["cuda"] = torch.version.cuda or "none"
        out["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        out["cuda"] = "absent"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
