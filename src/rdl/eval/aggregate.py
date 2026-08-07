"""Aggregation across seeds and conditions.

The pre-registered gate is stated in terms of **non-overlapping 95% CIs across 5
seeds**, so the CI machinery lives here rather than being improvised at report time.

Two deliberate choices:

- The CI is a **bootstrap percentile** interval over seed-level means, not a normal
  approximation. With n=5 seeds a t-interval on a proportion is not credible, and a
  bootstrap at least states its assumptions honestly. Both are computed; the bootstrap
  is the one the gate uses.
- `non_overlapping` is strict: the intervals must not touch. Reviewers read
  "non-overlapping CIs" as a strong claim, and a boundary-touching pass would be a
  reporting sleight of hand.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "ConditionSummary",
    "GateResult",
    "SeedStats",
    "bootstrap_ci",
    "condition_delta_gate",
    "non_overlapping",
    "summarise_seeds",
]


@dataclass
class SeedStats:
    mean: float
    std: float
    n: int
    ci_low: float
    ci_high: float
    values: list[float] = field(default_factory=list)
    method: str = "bootstrap-percentile"

    def to_dict(self) -> dict:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "n": self.n,
            "ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "method": self.method,
            "values": [round(v, 4) for v in self.values],
        }


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI over the mean. Deterministic given `seed`."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def normal_ci(values: Sequence[float], *, z: float = 1.96) -> tuple[float, float]:
    """Normal-approximation CI. Reported for comparison; NOT what the gate uses."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        v = float(arr[0]) if arr.size else float("nan")
        return (v, v)
    m = float(arr.mean())
    se = float(arr.std(ddof=1)) / math.sqrt(arr.size)
    return (m - z * se, m + z * se)


def summarise_seeds(values: Sequence[float], *, seed: int = 42) -> SeedStats:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return SeedStats(float("nan"), float("nan"), 0, float("nan"), float("nan"), [])
    lo, hi = bootstrap_ci(arr, seed=seed)
    return SeedStats(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        n=int(arr.size),
        ci_low=lo,
        ci_high=hi,
        values=[float(v) for v in arr],
    )


def non_overlapping(a: SeedStats, b: SeedStats) -> bool:
    """Strict: the two 95% intervals must not touch."""
    if any(map(math.isnan, (a.ci_low, a.ci_high, b.ci_low, b.ci_high))):
        return False
    return a.ci_high < b.ci_low or b.ci_high < a.ci_low


@dataclass
class ConditionSummary:
    condition: str
    metrics: dict[str, SeedStats] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }


def summarise_condition(
    condition: str, per_seed: Mapping[str, Sequence[float]], *, seed: int = 42
) -> ConditionSummary:
    return ConditionSummary(
        condition=condition,
        metrics={k: summarise_seeds(v, seed=seed) for k, v in per_seed.items()},
    )


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "gate": self.name,
            "passed": self.passed,
            "reason": self.reason,
            "detail": self.detail,
        }


def condition_delta_gate(
    c_treatment: Sequence[float],
    c_baseline: Sequence[float],
    *,
    min_delta_points: float = 20.0,
    require_non_overlapping: bool = True,
    name: str = "C3 - C1",
    seed: int = 42,
) -> GateResult:
    """The pre-registered primary gate.

    `SysRecall@5(C3, persistent-store) - SysRecall@5(C1) >= 20 absolute points`, over
    5 seeds, with non-overlapping 95% CIs.
    """
    t = summarise_seeds(c_treatment, seed=seed)
    b = summarise_seeds(c_baseline, seed=seed)
    delta_points = 100.0 * (t.mean - b.mean)

    reasons: list[str] = []
    ok_delta = delta_points >= min_delta_points
    if not ok_delta:
        reasons.append(f"delta {delta_points:.1f} points < pre-registered {min_delta_points:.0f}")
    ok_ci = (not require_non_overlapping) or non_overlapping(t, b)
    if not ok_ci:
        reasons.append(
            f"95% CIs overlap: treatment [{t.ci_low:.3f}, {t.ci_high:.3f}] vs "
            f"baseline [{b.ci_low:.3f}, {b.ci_high:.3f}]"
        )

    # Pre-registered kill criterion, day 5.
    kill = delta_points < 10.0
    if kill:
        reasons.append(
            "KILL CRITERION (day 5): delta < 10 points. The finding would be "
            "'heterogeneous unlearning leaks', which is a configuration bug, not a "
            "mechanism. Re-scope to a workshop note or pivot this week."
        )

    return GateResult(
        name=name,
        passed=ok_delta and ok_ci,
        detail={
            "treatment": t.to_dict(),
            "baseline": b.to_dict(),
            "delta_points": round(delta_points, 2),
            "min_delta_points": min_delta_points,
            "non_overlapping_ci": non_overlapping(t, b),
            "kill_criterion_triggered": kill,
        },
        reason="; ".join(reasons) if reasons else "all pre-registered criteria met",
    )
