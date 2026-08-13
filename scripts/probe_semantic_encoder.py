"""Probe a pinned sentence encoder against the frozen detector corpus. NOT a backbone.

GU-0032's fallback clause: if alias-v2 fails the CPU gates, the next thing to try is a
stronger semantic encoder given to every guarded arm. This script is that experiment, run
before any of it is wired into the runtime — because the answer turned out to be no, and
adding a torch dependency to the detector for a channel that does not clear the gate would
have been the expensive way to learn it.

What it does: mean-pooled, L2-normalised embeddings from a PINNED model revision, cosine
against each concept's scope prototypes (questions and paraphrases, never gold answers),
swept over the same threshold grid the gates use, with the same retain90 negatives.

    python scripts/probe_semantic_encoder.py --run runs/graph/<a discovery run>

Requires torch and transformers, and downloads ~90 MB on first use. Deliberately outside
`src/rdl`: nothing in the runtime imports it, and the detector stays torch-free.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
GRID = [round(0.20 + 0.025 * i, 4) for i in range(29)]
CORPUS_DIR = Path("data/cohorts/graph_unlearning_v1/detector_v2")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="a discovery run directory")
    parser.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    from rdl.eval.tofu_data import load_items
    from rdl.studies.graph_leak.arms import build_registry
    from rdl.studies.graph_leak.cohort import load_cohort, resolve_cohort

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION).eval()

    @torch.no_grad()
    def encode(texts: list[str], batch: int = 64):
        out = []
        for start in range(0, len(texts), batch):
            enc = tokenizer(
                texts[start : start + batch],
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(pooled, dim=1))
        return torch.cat(out).numpy()

    def cohort_items(path: Path):
        cohort = load_cohort(path, exclusions_path=None)
        source, _ = load_items(
            dataset=cohort.dataset,
            split=cohort.dataset_config,
            n_items=None,
            revision=cohort.dataset_revision,
        )
        return cohort, resolve_cohort(cohort, source)

    policy, policy_items = cohort_items(args.run / "FORGET_POLICY_COHORT.json")
    concept_of = {item.item_id: item.concept_id for item in policy.items}
    registry = build_registry(policy_items, concept_of=lambda i: concept_of[i])
    prototypes = {c.forget_id: encode(list(c.scope_prototypes)) for c in registry.concepts()}
    concept_ids = sorted(prototypes)

    corpus = json.loads(
        (args.corpus_dir / "DETECTOR_GENERATED_CORPUS.json").read_text(encoding="utf-8")
    )
    split = json.loads(
        (args.corpus_dir / "DETECTOR_ENGINEERING_SPLIT.json").read_text(encoding="utf-8")
    )
    examples = corpus["examples"]
    example_vectors = encode([e["text"] for e in examples])

    _negatives, negative_items = cohort_items(
        Path("data/cohorts/graph_unlearning_v1/calibration_negatives.json")
    )
    negative_vectors = encode([item.question for item in negative_items])

    def scores(vectors):
        return [
            {cid: float((prototypes[cid] @ row).max()) for cid in concept_ids} for row in vectors
        ]

    example_scores = scores(example_vectors)
    negative_scores = scores(negative_vectors)

    dev = set(split["development_concepts"])
    held = set(split["heldout_concepts"])
    rows = []
    for threshold in GRID:
        per_concept: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for example, score in zip(examples, example_scores, strict=True):
            bucket = per_concept[example["concept_id"]]
            bucket[1] += 1
            bucket[0] += int(score[example["concept_id"]] >= threshold)

        def macro(group, counts=per_concept):
            seen = [c for c in group if counts[c][1]]
            return sum(counts[c][0] / counts[c][1] for c in seen) / max(1, len(seen))

        fpr = sum(1 for s in negative_scores if max(s.values()) >= threshold) / len(negative_scores)
        rows.append(
            {
                "threshold": threshold,
                "dev_macro_recall": macro(dev),
                "heldout_macro_recall": macro(held),
                "retain90_fpr": fpr,
                "fpr_within_ceiling": fpr <= 0.10,
            }
        )

    eligible = [r for r in rows if r["fpr_within_ceiling"]]
    best = max(eligible, key=lambda r: r["dev_macro_recall"]) if eligible else None
    report = {
        "schema": "graph-detector-encoder-probe-v1",
        "status": "PROBE ONLY — not wired into the runtime detector",
        "model": {"repo_id": MODEL_ID, "revision": MODEL_REVISION, "pooling": "mean, L2"},
        "corpus_sha256": corpus["content_sha256"],
        "split_sha256": split["content_sha256"],
        "grid": rows,
        "best_within_fpr_ceiling": best,
        "verdict": (
            "no operating point clears the FPR ceiling with usable recall"
            if best is None or best["dev_macro_recall"] < 0.75
            else "clears the development recall bar; wire it in and re-run the gates"
        ),
        "why": (
            "the negatives are retain90 QUESTIONS and the prototypes are forget QUESTIONS. "
            "In an encoder that represents text type, every TOFU author question is close "
            "to every other one, so recall only rises where the FPR is already near 1.0."
        ),
    }
    out = args.output or (args.corpus_dir / "gates" / "ENCODER_PROBE.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("best_within_fpr_ceiling", "verdict")}, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
