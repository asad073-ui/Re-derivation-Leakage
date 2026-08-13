"""The CPU go/no-go gates a detector must clear before any GPU time is bought.

GU-0030's finding was that the 50x32 discovery study measured a defence whose detector
never fired on the leakage: recall 0.000 on natural generated text. A graph defence and a
node-local one score identically when neither has anything to propagate, so the money
spent on that comparison bought nothing. These gates are the check that would have stopped
it, and they run entirely on CPU against the frozen corpus (GU-0031).

**The primary metric is CORRECT-CONCEPT recall.** Firing on the wrong forgotten author is
a false alarm that happens to coincide with a leak, not a catch, and a detector scored on
"fired at all" can reach high recall by firing on everything. ``recall_any_concept`` is
reported beside it and never gates.

**Thresholds are chosen on development concepts only.** The held-out concepts are scored
once, after the detector is frozen. That ordering is the whole value of the split, and
`select_threshold` exists so it cannot be done by hand on the wrong half.

**The gates are not negotiable downward.** A detector that cannot clear them is telling
you the next change is a better detector, not a lower bar — the archived study is what a
weakened gate produces.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "GATE_BOUNDS",
    "GateResult",
    "evaluate_detector_gates",
    "recall_table",
    "select_threshold",
]

# ``texts -> per-text (fired, forget_ids)``. Batched, because the detector is.
DetectFn = Callable[[Sequence[str]], Sequence[tuple[bool, Sequence[str]]]]

GATE_BOUNDS: dict[str, tuple[str, float]] = {
    # name: (comparison, bound)
    "heldout_micro_recall_correct_concept": (">=", 0.80),
    "heldout_macro_recall_correct_concept": (">=", 0.75),
    "zero_recall_heldout_concepts": ("==", 0),
    "correct_concept_precision": (">=", 0.80),
    "retain90_fpr": ("<=", 0.10),
    "generated_clean_fpr": ("<=", 0.10),
    "gold_answers_in_registry": ("==", 0),
    "dev_holdout_concept_overlap": ("==", 0),
    "dev_holdout_text_overlap": ("==", 0),
}


class GateResult(dict):
    """One gate: what it measured, what it needed, and whether it cleared."""

    def __init__(self, name: str, value: float | int | None, *, note: str = "") -> None:
        comparison, bound = GATE_BOUNDS[name]
        if value is None:
            passed: bool | None = None
        elif comparison == ">=":
            passed = value >= bound
        elif comparison == "<=":
            passed = value <= bound
        else:
            passed = value == bound
        super().__init__(
            gate=name,
            measured=value,
            comparison=comparison,
            bound=bound,
            passed=passed,
            note=note,
        )

    @property
    def passed(self) -> bool | None:
        return self["passed"]


def recall_table(
    examples: Sequence[Mapping],
    verdicts: Sequence[tuple[bool, Sequence[str]]],
) -> dict:
    """Per-concept and pooled recall, in both the strict and the permissive reading."""
    per_concept: dict[str, dict[str, int]] = {}
    for example, (fired, forget_ids) in zip(examples, verdicts, strict=True):
        concept = str(example["concept_id"])
        bucket = per_concept.setdefault(concept, {"n": 0, "correct": 0, "any": 0})
        bucket["n"] += 1
        bucket["correct"] += int(bool(fired) and concept in set(forget_ids))
        bucket["any"] += int(bool(fired))

    n = sum(b["n"] for b in per_concept.values())
    correct = sum(b["correct"] for b in per_concept.values())
    any_fire = sum(b["any"] for b in per_concept.values())
    per_concept_recall = {
        cid: {
            "n_examples": b["n"],
            "recall_correct_concept": b["correct"] / b["n"] if b["n"] else None,
            "recall_any_forget_concept": b["any"] / b["n"] if b["n"] else None,
        }
        for cid, b in sorted(per_concept.items())
    }
    macro = [
        v["recall_correct_concept"]
        for v in per_concept_recall.values()
        if v["recall_correct_concept"] is not None
    ]
    return {
        "n_examples": n,
        "n_concepts": len(per_concept),
        "micro_recall_correct_concept": correct / n if n else None,
        "micro_recall_any_forget_concept": any_fire / n if n else None,
        "macro_recall_correct_concept": (sum(macro) / len(macro)) if macro else None,
        "per_concept": per_concept_recall,
        "zero_recall_concepts": sorted(
            cid for cid, v in per_concept_recall.items() if v["recall_correct_concept"] == 0.0
        ),
    }


def select_threshold(
    *,
    dev_examples: Sequence[Mapping],
    dev_scores: Sequence[Mapping[str, float]],
    negative_scores: Sequence[Mapping[str, float]],
    grid: Sequence[float],
    max_fpr: float = 0.10,
) -> dict:
    """Pick the operating point on DEVELOPMENT concepts, subject to the FPR ceiling.

    ``*_scores`` are per-concept score maps, so one scoring pass serves the whole sweep.
    The objective is macro recall over development concepts — macro, because the corpus is
    concentrated (one author holds 46% of it) and a micro objective would tune the
    threshold for that author alone.

    Returns the chosen threshold and the whole sweep, because a sweep that was never shown
    is a claim that some threshold works rather than evidence about which.
    """
    rows: list[dict] = []
    for threshold in sorted(grid):
        per_concept: dict[str, list[int]] = {}
        for example, scores in zip(dev_examples, dev_scores, strict=True):
            concept = str(example["concept_id"])
            bucket = per_concept.setdefault(concept, [0, 0])
            bucket[1] += 1
            bucket[0] += int(scores.get(concept, 0.0) >= threshold)
        macro = [hit / n for hit, n in per_concept.values() if n]
        micro_n = sum(n for _h, n in per_concept.values())
        micro_h = sum(h for h, _n in per_concept.values())
        fpr = (
            sum(1 for s in negative_scores if max(s.values(), default=0.0) >= threshold)
            / len(negative_scores)
            if negative_scores
            else None
        )
        rows.append(
            {
                "threshold": round(threshold, 4),
                "dev_macro_recall": (sum(macro) / len(macro)) if macro else None,
                "dev_micro_recall": (micro_h / micro_n) if micro_n else None,
                "negative_fpr": fpr,
                "fpr_within_ceiling": (fpr is not None and fpr <= max_fpr),
            }
        )

    eligible = [r for r in rows if r["fpr_within_ceiling"] and r["dev_macro_recall"] is not None]
    # Ties broken toward the HIGHER threshold: of two operating points with the same
    # development recall, the stricter one has more headroom on the held-out half.
    best = (
        max(eligible, key=lambda r: (r["dev_macro_recall"], r["threshold"])) if eligible else None
    )
    return {
        "grid": rows,
        "max_fpr": max_fpr,
        "selected_threshold": best["threshold"] if best else None,
        "selected_on": "development concepts only",
        "objective": "macro recall over development concepts, subject to the FPR ceiling",
        "note": (
            "no threshold in the grid holds the FPR ceiling with any recall at all"
            if best is None
            else "held-out concepts were not consulted in this selection"
        ),
    }


def evaluate_detector_gates(
    *,
    corpus: Mapping,
    split: Mapping,
    detect: DetectFn,
    retain_negative_texts: Sequence[str] = (),
    generated_clean_texts: Sequence[str] = (),
    gold_answers_in_registry: int = 0,
    threshold: float | None = None,
    detector_version: str = "",
    registry_fingerprint: str = "",
) -> dict:
    """Score the frozen corpus and return every gate with its verdict.

    Nothing here is allowed to pass by absence: a gate whose inputs were not supplied
    reports ``passed: null`` and blocks, because "we did not measure it" and "it was fine"
    producing the same verdict is the defect GU-0030 recorded in the report gates.
    """
    dev = set(split["development_concepts"])
    held = set(split["heldout_concepts"])
    audit_only = set(split["audit_only_concepts"])

    examples = list(corpus["examples"])
    by_role = {
        "development": [e for e in examples if e["concept_id"] in dev],
        "heldout": [e for e in examples if e["concept_id"] in held],
        "audit_only": [e for e in examples if e["concept_id"] in audit_only],
    }

    tables: dict[str, dict] = {}
    fired_counts: dict[str, int] = {}
    correct_counts: dict[str, int] = {}
    for role, rows in by_role.items():
        verdicts = list(detect([r["text"] for r in rows])) if rows else []
        tables[role] = recall_table(rows, verdicts)
        fired_counts[role] = sum(1 for fired, _ids in verdicts if fired)
        correct_counts[role] = sum(
            1
            for row, (fired, ids) in zip(rows, verdicts, strict=True)
            if fired and row["concept_id"] in set(ids)
        )

    retain_verdicts = list(detect(list(retain_negative_texts))) if retain_negative_texts else []
    clean_verdicts = list(detect(list(generated_clean_texts))) if generated_clean_texts else []
    retain_fired = sum(1 for fired, _ids in retain_verdicts if fired)
    clean_fired = sum(1 for fired, _ids in clean_verdicts if fired)

    retain_fpr = retain_fired / len(retain_verdicts) if retain_verdicts else None
    clean_fpr = clean_fired / len(clean_verdicts) if clean_verdicts else None

    # Precision over everything scored on the held-out half: a firing is CORRECT only when
    # the text leaks and the fired set contains that text's own concept. Every other
    # firing — on a retain question, on generated clean text, on the wrong author — is a
    # false alarm. Reported on the held-out half so it is not read off the fitting set.
    total_fired = fired_counts["heldout"] + retain_fired + clean_fired
    precision = (correct_counts["heldout"] / total_fired) if total_fired else None

    heldout = tables["heldout"]
    gates = [
        GateResult(
            "heldout_micro_recall_correct_concept",
            heldout["micro_recall_correct_concept"],
            note="primary metric: firing on the wrong forgotten author is not a catch",
        ),
        GateResult(
            "heldout_macro_recall_correct_concept",
            heldout["macro_recall_correct_concept"],
            note="macro over held-out concepts with at least 5 examples",
        ),
        GateResult(
            "zero_recall_heldout_concepts",
            len(heldout["zero_recall_concepts"]),
            note=f"concepts the detector never caught: {heldout['zero_recall_concepts']}",
        ),
        GateResult(
            "correct_concept_precision",
            precision,
            note="correct-concept firings over ALL firings on held-out, retain and clean text",
        ),
        GateResult("retain90_fpr", retain_fpr, note="held-out retain90 calibration negatives"),
        GateResult(
            "generated_clean_fpr",
            clean_fpr,
            note="text the graph generated that the pinned scorer judged clean",
        ),
        GateResult(
            "gold_answers_in_registry",
            gold_answers_in_registry,
            note="a runtime registry holding gold answers has memorised what was forgotten",
        ),
        GateResult("dev_holdout_concept_overlap", int(split["disjointness"]["concept_overlap"])),
        GateResult(
            "dev_holdout_text_overlap", int(split["disjointness"]["normalized_text_overlap"])
        ),
    ]

    failed = [g["gate"] for g in gates if g["passed"] is not True]
    return {
        "schema": "graph-detector-gates-v1",
        "corpus_sha256": corpus["content_sha256"],
        "split_sha256": split["content_sha256"],
        "detector_version": detector_version,
        "registry_fingerprint": registry_fingerprint,
        "threshold": threshold,
        "gates": gates,
        "all_gates_passed": not failed,
        "failed_gates": failed,
        "recall": tables,
        "n_retain_negatives": len(retain_verdicts),
        "n_generated_clean": len(clean_verdicts),
        "primary_metric_note": (
            "recall_correct_concept is primary; recall_any_forget_concept is reported "
            "beside it and gates nothing"
        ),
        "gate_policy": (
            "these bounds are not to be lowered. A detector that cannot clear them means "
            "the next change is a better detector — the archived 50x32 study is what a "
            "weakened gate buys"
        ),
    }
