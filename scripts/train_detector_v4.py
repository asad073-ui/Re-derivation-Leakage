#!/usr/bin/env python
"""Detector-v4.1 **Phase 7**: fine-tune the answerability cross-encoder on an RTX 3090.

What changed from the v4 version of this file
---------------------------------------------
The v4 script loaded a model, tokenised, wrote a manifest and raised ``SystemExit``. It had
no optimizer, no loop, no development evaluation, no checkpoint selection, no checkpoint
saving and no inference path, so "the recipe is reviewable before an instance is rented"
described a document rather than a trainer. Four things are fixed here.

**A real loop.** Seeded, class-weighted, evaluated on development after every epoch, with
best-checkpoint selection on macro F1 and the checkpoint written to disk. The held-out
split is never loaded; the natural gate bank is a different file this script cannot name.

**The truncation bug.** v4 encoded ``tokenizer(question, identity + "[SEP]" + candidate,
truncation="only_second")``, which truncates the END of the second sequence — the
candidate, the only span the verdict is about. Encoding now goes through
:func:`rdl.defenses.cross_encoder_answerability.budget_encode`, which reserves the
candidate first and drops identity aliases, then question tokens, before it will cut one.
Every cut is counted into the manifest. Importing it from the runtime module rather than
reimplementing it is what makes train-time and serve-time segmentation the same object.

**Both revisions.** ``--model-revision`` and ``--tokenizer-revision`` are both required and
are recorded separately even when they are equal. A moved model tag changes the weights; a
moved tokenizer tag changes the subword split, which changes every score at a fixed
threshold. v4 left ``tokenizer_revision`` empty.

**A baseline that means something.** ``microsoft/deberta-v3-base`` has a randomly
initialised classification head: its "off-the-shelf" number is a measurement of that random
head, not of what a pretrained model already knows about answerability. So two baselines are
recorded — a pinned pretrained NLI cross-encoder, zero-shot, and the fine-tune.

The readiness gate
------------------
v4 refused to start unless ``DETECTOR_V4_ORACLE_CEILING.json`` passed, with
``--force-despite-failed-ceiling`` as an escape hatch. That artifact measures exact
answer-token overlap and is not a bound on a semantic model (GU-0037), so gating on it was
gating on the wrong thing and the escape hatch was a way to keep the mistake. Both are gone.

The gate is now the v4.1 blinded label audit: ``LABEL_ALIGNMENT_REPORT.json`` must exist and
must have cleared its own pre-registered decision gate. That is the artifact that answers
whether the target is legible to humans and whether there are enough ANSWER and NONE rows
to train and gate on — the question that actually bounds Goal A.

Order of operations on the box
------------------------------
1. ``rdl graph-detector-v4-label-report`` must pass.
2. ``--baseline-only`` first: record the pinned zero-shot cross-encoder.
3. Fine-tune on ``split == "train"`` plus the adjudicated natural training rows.
4. Select the operating point on development (``rdl graph-detector-v4-gates --backend
   cross_encoder``).
5. Generate the FRESH gate bank (``FINAL_GATE_BANK_MANIFEST.json``) and open it once.

Never tune on the v4 held-out data — it is engineering-only — and never on the fresh bank.
"""

from __future__ import annotations

import argparse
import json
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
LABELS = ("NONE", "PARTIAL", "ANSWER")


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
    seed: int = 20260814
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
    checkpoint_selection: str = "macro_f1 on development"
    expected_device: str = "cuda"

    def to_dict(self) -> dict:
        return asdict(self)


# A pinned pretrained NLI cross-encoder. Its head is TRAINED, so a zero-shot number from it
# is a statement about what pretraining already knows; deberta-v3-base's fresh head is not.
DEFAULT_BASELINE_REPO = "cross-encoder/nli-deberta-v3-base"
BASELINE_ENTAILMENT_INDEX = 1  # contradiction / entailment / neutral — confirmed on the box


def require_cuda() -> str:
    try:
        import torch
    except ImportError:
        raise SystemExit("torch is not installed; this script runs on the GPU box only") from None
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This is the RTX phase; the CPU phase is "
            "`rdl graph-detector-v4-label-audit` and `rdl graph-detector-v4-label-report`."
        )
    return str(torch.cuda.get_device_name(0))


def require_label_audit(v4_1_dir: Path) -> dict:
    """Refuse to train until the blinded label audit says the target is legible.

    There is no override. v4's ``--force-despite-failed-ceiling`` existed to train despite
    a failing artifact; the correct response to that artifact was to fix its
    interpretation, and a flag that skips the fix is a way to keep the mistake. If this
    gate is wrong, change the gate in the protocol and say so in DECISIONS.md.
    """
    path = v4_1_dir / "LABEL_ALIGNMENT_REPORT.json"
    if not path.exists():
        raise SystemExit(
            f"{path} is absent. Run the blinded label audit first:\n"
            "  rdl graph-detector-v4-label-audit\n"
            "  (two judges annotate)\n"
            "  rdl graph-detector-v4-label-report --judge-a ... --judge-b ...\n"
            "Goal A's target is human answer_attempt, and no GPU step precedes it."
        )
    report = json.loads(path.read_text(encoding="utf-8"))
    if not report.get("all_gates_passed"):
        raise SystemExit(
            "the v4.1 label audit does NOT clear its decision gate: "
            f"{report.get('failed_gates')}\n{report.get('verdict')}\n"
            "Fix the label protocol, not the model."
        )
    return report


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


def natural_examples(v4_1_dir: Path, split: str) -> list[dict]:
    """Adjudicated natural rows, joined to their blinded text. Empty when the audit is absent.

    The adjudicated overlay carries Goal A labels and ``audit_id``; the judge file carries
    the question and the candidate. Neither carries a reference answer, which is why the
    join can be done here at all.
    """
    adjudicated = v4_1_dir / "LABEL_AUDIT_ADJUDICATED.jsonl"
    judge = v4_1_dir / "LABEL_AUDIT_JUDGE_A.jsonl"
    if not (adjudicated.exists() and judge.exists()):
        return []

    def load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    text_of = {r["audit_id"]: r for r in load(judge)}
    out: list[dict] = []
    for row in load(adjudicated):
        label = row.get("answer_attempt")
        blind = text_of.get(row["audit_id"])
        if label not in LABELS or blind is None:
            continue
        # Content-addressed train/development halving, computed from the audit id alone so
        # it does not move when rows are added.
        half = "train" if int(row["audit_id"][:2], 16) % 2 == 0 else "development"
        if half != split:
            continue
        out.append(
            {
                "question": blind["protected_question"],
                "aliases": [],
                "candidate": blind["candidate_text"],
                "label": label,
                "source": "natural_adjudicated",
                "group": row["audit_id"],
            }
        )
    return out


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
                list(f[key]) + [(pad_id if key == "input_ids" else 0)] * (width - len(f[key]))
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
    """Per-class precision/recall/F1 and macro F1 on a loader. No threshold is chosen here."""
    import torch

    model.eval()
    confusion = [[0] * len(LABELS) for _ in LABELS]
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float().cpu()
            for gold, predicted in zip(labels.tolist(), logits.argmax(-1).tolist()):
                confusion[gold][predicted] += 1
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
        "accuracy": (sum(confusion[i][i] for i in range(len(LABELS))) / total) if total else None,
        "per_class": per_class,
        "confusion": {LABELS[i]: dict(zip(LABELS, confusion[i])) for i in range(len(LABELS))},
    }


def baseline_zero_shot(repo_id: str, revision: str, examples: list[dict], budget, device) -> dict:
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
        scores.append(float(probabilities[BASELINE_ENTAILMENT_INDEX]))
        golds.append(row["label"])

    positives = [s for s, g in zip(scores, golds) if g == "ANSWER"]
    negatives = [s for s, g in zip(scores, golds) if g == "NONE"]
    pairs = len(positives) * len(negatives)
    auc = (
        sum(1 for p in positives for n in negatives if p > n) + 0.5 * sum(
            1 for p in positives for n in negatives if p == n
        )
    ) / pairs if pairs else None
    return {
        "repo_id": repo_id,
        "revision": revision,
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


# -------------------------------------------------------------------------- main --


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=V4)
    parser.add_argument("--v4-1-dir", type=Path, default=V4_1)
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs" / "detector_v4")
    parser.add_argument("--model-repo-id", default=TrainingPins.model_repo_id)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--tokenizer-revision", default="")
    parser.add_argument("--baseline-repo-id", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-revision", default="")
    parser.add_argument("--extra-train", type=Path, nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=TrainingPins.num_train_epochs)
    parser.add_argument("--seed", type=int, default=TrainingPins.seed)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    device = require_cuda()
    audit = require_label_audit(args.v4_1_dir)

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
        seed=args.seed,
        num_train_epochs=args.epochs,
    )

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from rdl.defenses.cross_encoder_answerability import EncodingBudget

    set_seed(pins.seed)
    budget = EncodingBudget(
        max_length=pins.max_length,
        min_question_tokens=pins.min_question_tokens,
        max_aliases=pins.max_aliases,
    )

    dataset = json.loads((args.data_dir / "DETECTOR_V4_DATASET.json").read_text(encoding="utf-8"))
    train_rows = [
        *synthetic_examples(dataset, "train"),
        *natural_examples(args.v4_1_dir, "train"),
        *extra_examples(list(args.extra_train)),
    ]
    dev_rows = [
        *synthetic_examples(dataset, "development"),
        *natural_examples(args.v4_1_dir, "development"),
    ]
    # The synthetic held-out split is opened by `rdl graph-detector-v4-gates` after the
    # threshold is frozen, and the natural gate bank is a file this script never names.
    # Checked rather than commented: a filter that silently stopped filtering would be
    # invisible in the manifest, which is where the guarantee is supposed to be readable.
    if any(r.get("split") == "heldout" for r in [*train_rows, *dev_rows]):
        raise SystemExit("a held-out row reached the training or development pool")
    if not train_rows:
        raise SystemExit("no training rows; check --data-dir")

    tokenizer = AutoTokenizer.from_pretrained(
        pins.model_repo_id, revision=pins.tokenizer_revision
    )

    baseline = {"skipped": True}
    if not args.skip_baseline:
        if not args.baseline_revision:
            raise SystemExit(
                "--baseline-revision is required (or pass --skip-baseline). An unpinned "
                "baseline is a number nobody can reproduce."
            )
        baseline = baseline_zero_shot(
            args.baseline_repo_id, args.baseline_revision, dev_rows, budget, "cuda"
        )
        print(f"baseline AUC (ANSWER vs NONE): {baseline['answer_vs_none_auc']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict = {
        "schema": "graph-detector-v4-model-v2",
        "phase": "Detector-v4.1 Phase 7 — cross-encoder fine-tune",
        "protocol": "docs/graph_unlearning/DETECTOR_V4_1_PROTOCOL.md",
        "device": device,
        "pins": pins.to_dict(),
        "encoding_budget": budget.to_dict(),
        "dataset_content_sha256": dataset["content_sha256"],
        "label_audit": {
            "report": str(args.v4_1_dir / "LABEL_ALIGNMENT_REPORT.json"),
            "all_gates_passed": audit.get("all_gates_passed"),
            "answer_attempt_kappa": audit.get("inter_judge", {})
            .get("per_field", {})
            .get("answer_attempt", {})
            .get("cohens_kappa"),
        },
        "n_train_rows": len(train_rows),
        "n_development_rows": len(dev_rows),
        "train_sources": _counts(train_rows, "source"),
        "development_sources": _counts(dev_rows, "source"),
        "train_label_distribution": _counts(train_rows, "label"),
        "heldout_read_during_training": False,
        "gate_bank_read_during_training": False,
        "labels": list(LABELS),
        "baseline_zero_shot": baseline,
        "baseline_only": bool(args.baseline_only),
        "versions": _versions(),
    }

    if args.baseline_only:
        _write_manifest(args.output_dir, manifest)
        print("baseline recorded; no fine-tuning requested")
        return 0

    model = AutoModelForSequenceClassification.from_pretrained(
        pins.model_repo_id, revision=pins.model_revision, num_labels=len(LABELS)
    ).to("cuda")

    train_set = AnswerabilityDataset(train_rows, tokenizer, budget)
    dev_set = AnswerabilityDataset(dev_rows, tokenizer, budget)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    generator = torch.Generator().manual_seed(pins.seed)
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
    ).to("cuda")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=pins.learning_rate, weight_decay=pins.weight_decay
    )
    steps_per_epoch = max(1, len(train_loader) // pins.gradient_accumulation_steps)
    total_steps = steps_per_epoch * pins.num_train_epochs
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * pins.warmup_ratio), total_steps
    )
    scaler = torch.cuda.amp.GradScaler(enabled=pins.fp16)

    history: list[dict] = []
    best = {"macro_f1": -1.0, "epoch": None, "path": None}
    started = time.time()
    for epoch in range(1, pins.num_train_epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to("cuda")
            batch = {k: v.to("cuda") for k, v in batch.items()}
            with torch.autocast("cuda", enabled=pins.fp16):
                logits = model(**batch).logits
                loss = loss_fn(logits.float(), labels) / pins.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running += float(loss.item()) * pins.gradient_accumulation_steps
            if step % pins.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), pins.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        metrics = evaluate(model, dev_loader, "cuda")
        path = args.output_dir / f"checkpoint-epoch{epoch}"
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(1, len(train_loader)),
                "development": metrics,
                "checkpoint": str(path),
            }
        )
        print(f"epoch {epoch}: macro_f1={metrics['macro_f1']:.4f} loss={running / max(1, len(train_loader)):.4f}")
        if metrics["macro_f1"] > best["macro_f1"]:
            best = {"macro_f1": metrics["macro_f1"], "epoch": epoch, "path": str(path)}

    manifest.update(
        {
            "training": {
                "history": history,
                "wall_clock_seconds": round(time.time() - started, 1),
                "class_weights": {name: w for name, w in zip(LABELS, weights.cpu().tolist())},
                "steps_per_epoch": steps_per_epoch,
                "total_optimizer_steps": total_steps,
            },
            "selected_checkpoint": best["path"],
            "selected_on": "macro F1 on the development split. Held-out was never loaded.",
            "selected_epoch": best["epoch"],
            "development_macro_f1": best["macro_f1"],
            "truncation": {
                "train_candidate_truncations": train_set.n_candidate_truncations,
                "train_question_truncations": train_set.n_question_truncations,
                "development_candidate_truncations": dev_set.n_candidate_truncations,
                "why_counted": (
                    "a candidate that did not fit the window is an evaluation error, not "
                    "a NONE. v4 truncated the candidate by construction; this counts what "
                    "the corrected budget still has to cut."
                ),
            },
        }
    )
    _write_manifest(args.output_dir, manifest)
    print(f"selected {best['path']} (macro_f1={best['macro_f1']:.4f})")
    print("next: rdl graph-detector-v4-gates --backend cross_encoder --model-artifact "
          f"{args.output_dir / 'DETECTOR_V4_MODEL.json'}")
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
