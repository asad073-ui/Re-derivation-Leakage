"""Choosing and freezing the detector threshold.

The threshold is selected on **development concepts only** and then frozen. Selecting it
on the same concepts the headline number is computed over would make the false-positive
rate an in-sample quantity and the whole comparison optimistic.

The sweep reports, per threshold:

  recall     fraction of positive probes (text that really is about a forgotten concept)
             the detector fires on
  fpr        fraction of retain/negative probes it fires on — the over-blocking cost
  youden     recall - fpr, used only to order candidates

The chosen threshold is the smallest one meeting a recall floor while staying under an
FPR ceiling; when none does, the artefact records the failure rather than picking the
best of a bad set.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

from ..defenses.semantic_detector import SemanticConceptDetector

__all__ = ["CalibrationResult", "calibrate_threshold", "sweep"]

DEFAULT_GRID: tuple[float, ...] = (
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
)


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float | None
    recall: float
    fpr: float
    recall_floor: float
    fpr_ceiling: float
    grid: tuple[float, ...]
    rows: tuple[dict, ...]
    n_positive: int
    n_negative: int
    ok: bool
    reason: str

    @property
    def calibration_id(self) -> str:
        payload = json.dumps(
            {
                "threshold": self.threshold,
                "recall_floor": self.recall_floor,
                "fpr_ceiling": self.fpr_ceiling,
                "n_positive": self.n_positive,
                "n_negative": self.n_negative,
                "rows": list(self.rows),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "calibration_id": self.calibration_id,
            "threshold": self.threshold,
            "recall": self.recall,
            "fpr": self.fpr,
            "recall_floor": self.recall_floor,
            "fpr_ceiling": self.fpr_ceiling,
            "grid": list(self.grid),
            "rows": list(self.rows),
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "ok": self.ok,
            "reason": self.reason,
            "selected_on": "development concepts only",
        }


def sweep(
    detector: SemanticConceptDetector,
    *,
    positives: Sequence[str],
    negatives: Sequence[str],
    grid: Sequence[float] = DEFAULT_GRID,
) -> list[dict]:
    """Score every probe once, then threshold. One detector pass, not one per grid point."""
    positive_scores = [r.score for r in detector.score_batch(list(positives))]
    negative_scores = [r.score for r in detector.score_batch(list(negatives))]
    rows: list[dict] = []
    for threshold in grid:
        recall = (
            sum(1 for s in positive_scores if s >= threshold) / len(positive_scores)
            if positive_scores
            else 0.0
        )
        fpr = (
            sum(1 for s in negative_scores if s >= threshold) / len(negative_scores)
            if negative_scores
            else 0.0
        )
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "recall": round(recall, 6),
                "fpr": round(fpr, 6),
                "youden": round(recall - fpr, 6),
            }
        )
    return rows


def calibrate_threshold(
    detector: SemanticConceptDetector,
    *,
    positives: Sequence[str],
    negatives: Sequence[str],
    grid: Sequence[float] = DEFAULT_GRID,
    recall_floor: float = 0.90,
    fpr_ceiling: float = 0.10,
) -> CalibrationResult:
    rows = sweep(detector, positives=positives, negatives=negatives, grid=grid)
    feasible = [r for r in rows if r["recall"] >= recall_floor and r["fpr"] <= fpr_ceiling]
    if feasible:
        # Highest threshold that still meets the recall floor: the least trigger-happy
        # detector consistent with the requirement, which minimises over-blocking.
        best = max(feasible, key=lambda r: (r["threshold"],))
        return CalibrationResult(
            threshold=float(best["threshold"]),
            recall=float(best["recall"]),
            fpr=float(best["fpr"]),
            recall_floor=recall_floor,
            fpr_ceiling=fpr_ceiling,
            grid=tuple(float(g) for g in grid),
            rows=tuple(rows),
            n_positive=len(positives),
            n_negative=len(negatives),
            ok=True,
            reason="highest threshold meeting the recall floor within the FPR ceiling",
        )
    return CalibrationResult(
        threshold=None,
        recall=0.0,
        fpr=0.0,
        recall_floor=recall_floor,
        fpr_ceiling=fpr_ceiling,
        grid=tuple(float(g) for g in grid),
        rows=tuple(rows),
        n_positive=len(positives),
        n_negative=len(negatives),
        ok=False,
        reason=(
            f"no threshold in the grid reaches recall >= {recall_floor} with "
            f"fpr <= {fpr_ceiling}. The detector backbone is not adequate for this "
            "cohort; report it as diagnostic rather than calibrated."
        ),
    )
