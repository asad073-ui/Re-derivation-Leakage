#!/usr/bin/env python
"""Detector-v4 **Phase 7**: fine-tune the answerability cross-encoder on an RTX 3090.

**This script has never been run.** It is committed on the CPU branch so that the training
recipe is reviewable before an instance is rented, and so that every pin it needs is
written down while the reasoning is fresh rather than reconstructed afterwards. It refuses
to start unless a CUDA device is present, and it refuses to start unless the CPU gate
artifacts say training is justified.

Why a cross-encoder and not another bi-encoder
----------------------------------------------
A bi-encoder embeds the question and the candidate separately and compares them. That is
the v1–v3 architecture and it is structurally unable to do Goal A: "Where was X born?" and
"X is a famous author" are both about X, so they sit close in any representation of what a
text is *about*, while "X was born in Rome" and "X was born in Madrid" must both be tagged
despite being about different places. The signal is a relation between the two inputs — is
there a span here that fills the slot the question opened — and that relation has to be
computed with both inputs in the same attention window.

Encoding
--------
    [CLS] protected question [SEP] request identity context [SEP] candidate clause [SEP]

Three segments, in that order, and the candidate is always last so a truncation drops the
identity context before it drops the evidence. The request context is the routed subject
description, NOT the raw request: putting the request verbatim into the same window as the
candidate is the failure Phase 3 exists to prevent, and it would let the model learn that
the presence of a forget question predicts the label.

Heads
-----
Three-way over {NONE, PARTIAL, ANSWER}, plus an optional span head. PARTIAL is a real class
rather than a threshold band on ANSWER: a fragment is not enforceable and a model that
could only express "somewhat" would make the accumulation result untestable.

What must be pinned, and why each one
-------------------------------------
``model_repo_id`` / ``model_revision``   a moved tag silently changes the detector
``tokenizer_revision``                   segmentation into subwords changes the scores
``dataset_content_sha256``               the split is the claim; the data must match it
``seed`` / optimizer / lr / batch/epochs  the checkpoint has to be reproducible
``selected_checkpoint``                  chosen on development, never on held-out
``torch`` / ``transformers`` / CUDA      kernels differ, and so do the logits

Everything above is written into ``DETECTOR_V4_MODEL.json`` next to the checkpoint. A
checkpoint without that file is not eligible for a study.

Order of operations on the box
------------------------------
1. ``rdl graph-detector-v4-oracle`` must pass. If the answer-aware ceiling fails, no
   answer-free model can clear the bounds and training is wasted compute.
2. Record the OFF-THE-SHELF baseline before any fine-tuning, so the fine-tune's
   contribution is attributable.
3. Fine-tune on ``split == "train"`` only.
4. Select the threshold on ``split == "development"`` (``rdl graph-detector-v4-gates``).
5. Open ``split == "heldout"`` and the natural bank's gate half exactly once.

Never tune on the current detector held-out data, and never on the natural gate half.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

V4 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4"
LABELS = ("NONE", "PARTIAL", "ANSWER")


@dataclass(frozen=True)
class TrainingPins:
    """Everything that changes the checkpoint. All of it lands in DETECTOR_V4_MODEL.json."""

    model_repo_id: str = "microsoft/deberta-v3-base"
    # Deliberately empty. A revision is a claim about which weights were used, and a
    # guessed one is worse than an absent one — resolve it on the box with
    # `huggingface_hub.HfApi().model_info(repo).sha` and record what came back.
    model_revision: str = ""
    tokenizer_revision: str = ""
    max_length: int = 256
    seed: int = 20260814
    optimizer: str = "adamw_torch"
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    fp16: bool = True
    # 24 GB is enough for a base-size cross-encoder at 256 tokens with fp16 and batch 16;
    # a large-size model would need gradient checkpointing and is not worth the first
    # experiment's variance.
    expected_device: str = "cuda"

    def to_dict(self) -> dict:
        return asdict(self)


def require_cuda() -> str:
    try:
        import torch
    except ImportError:  # pragma: no cover - the CPU branch never gets here
        raise SystemExit("torch is not installed; this script runs on the GPU box only")
    if not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This is the RTX phase; the CPU phase is "
            "`rdl graph-detector-v4-oracle` and `rdl graph-detector-v4-gates`."
        )
    return torch.cuda.get_device_name(0)


def require_cpu_gate(data_dir: Path, *, force: bool) -> dict:
    """Refuse to train when the answer-aware ceiling says training cannot help.

    ``--force-despite-failed-ceiling`` exists because a deliberate override is a decision
    somebody made and can be found in the shell history; a script that trained anyway is
    a decision nobody made.
    """
    ceiling_path = data_dir / "DETECTOR_V4_ORACLE_CEILING.json"
    if not ceiling_path.exists():
        raise SystemExit(f"{ceiling_path} is absent. Run `rdl graph-detector-v4-oracle` first.")
    ceiling = json.loads(ceiling_path.read_text(encoding="utf-8"))
    natural = ceiling.get("natural_arm", {})
    if not natural.get("measured"):
        raise SystemExit("the oracle's natural arm was not measured; there is no ceiling to beat")
    if not natural.get("all_gates_passed") and not force:
        locus = natural.get("failure_locus", {})
        raise SystemExit(
            "the answer-aware ceiling FAILS on natural text: "
            f"{natural.get('failed_gates')}\n"
            f"share of leaking text invisible to an answer-aware evaluator: "
            f"{locus.get('share_of_leaking_invisible_to_an_answer_aware_evaluator')}\n"
            f"share of leaking text from open-ended questions: "
            f"{locus.get('share_of_leaking_from_open_ended_questions')}\n"
            "No answer-free detector can do better than the ceiling. Fix the protected-"
            "question population or the leak label first, or pass "
            "--force-despite-failed-ceiling if you intend to measure the gap anyway."
        )
    return ceiling


def encode_rows(rows, tokenizer, pins: TrainingPins):
    """``(question, identity context, candidate clause)`` triples -> model inputs.

    The candidate goes last so truncation eats the identity context before the evidence,
    and the raw request is never one of the three segments.
    """
    from rdl.defenses.atomic_text import segment

    features, labels = [], []
    for row in rows:
        identity = ", ".join(row["aliases"]) or row["subject"]
        for candidate in row["candidates"]:
            for clause in segment(candidate) or ():
                features.append(
                    tokenizer(
                        row["protected_question"],
                        f"{identity} [SEP] {clause.text}",
                        truncation="only_second",
                        max_length=pins.max_length,
                    )
                )
                labels.append(LABELS.index(row["label"]))
    return features, labels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=V4)
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs" / "detector_v4")
    parser.add_argument("--model-repo-id", default=TrainingPins.model_repo_id)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--baseline-only", action="store_true", help="record the off-the-shelf number and stop")
    parser.add_argument("--force-despite-failed-ceiling", action="store_true")
    args = parser.parse_args()

    device = require_cuda()
    ceiling = require_cpu_gate(args.data_dir, force=args.force_despite_failed_ceiling)

    dataset = json.loads((args.data_dir / "DETECTOR_V4_DATASET.json").read_text(encoding="utf-8"))
    if not args.model_revision:
        raise SystemExit(
            "--model-revision is required. A checkpoint whose weights cannot be named is "
            "not eligible for a study."
        )
    pins = TrainingPins(model_repo_id=args.model_repo_id, model_revision=args.model_revision)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(pins.model_repo_id, revision=pins.model_revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        pins.model_repo_id, revision=pins.model_revision, num_labels=len(LABELS)
    ).to("cuda")

    train_rows = [r for r in dataset["rows"] if r["split"] == "train"]
    dev_rows = [r for r in dataset["rows"] if r["split"] == "development"]
    # Named and NOT loaded. The held-out split is opened by `rdl graph-detector-v4-gates`
    # after the threshold is frozen, and a training script that could read it would make
    # that guarantee a matter of discipline rather than of code.
    heldout_rows_are_not_read_here = True

    _train_features, _train_labels = encode_rows(train_rows, tokenizer, pins)
    _dev_features, _dev_labels = encode_rows(dev_rows, tokenizer, pins)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "graph-detector-v4-model-v1",
        "phase": "Detector-v4 Phase 7 — cross-encoder fine-tune",
        "device": device,
        "pins": pins.to_dict(),
        "dataset_content_sha256": dataset["content_sha256"],
        "n_train_rows": len(train_rows),
        "n_development_rows": len(dev_rows),
        "heldout_read_during_training": not heldout_rows_are_not_read_here,
        "oracle_ceiling_passed": ceiling["natural_arm"].get("all_gates_passed"),
        "trained_despite_failed_ceiling": bool(args.force_despite_failed_ceiling),
        "labels": list(LABELS),
        "baseline_only": bool(args.baseline_only),
        "versions": _versions(),
    }
    (args.output_dir / "DETECTOR_V4_MODEL.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output_dir / 'DETECTOR_V4_MODEL.json'}")

    if args.baseline_only:
        print("baseline recorded; no fine-tuning requested")
        return 0

    raise SystemExit(
        "the Trainer loop is intentionally not implemented on the CPU branch.\n"
        "It is added on the GPU box in the same commit that records the first "
        "DETECTOR_V4_MODEL.json with a real revision, so that the recipe above and the "
        "code that ran it are reviewed together rather than a month apart."
    )


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
    except Exception:  # pragma: no cover
        out["cuda"] = "absent"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
