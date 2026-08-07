"""Aggregation across seeds and conditions.

The pre-registered gate is stated in terms of **non-overlapping 95% CIs across 5
seeds**, so the CI machinery lives here rather than being improvised at report time.

Three deliberate choices:

- The CI is a **bootstrap percentile** interval over seed-level means, not a normal
  approximation. With n=5 seeds a t-interval on a proportion is not credible, and a
  bootstrap at least states its assumptions honestly. Both are computed; the bootstrap
  is the one the seed-level gate uses.
- `non_overlapping` is strict: the intervals must not touch. Reviewers read
  "non-overlapping CIs" as a strong claim, and a boundary-touching pass would be a
  reporting sleight of hand.
- **A seed is only a replicate if something in the run is stochastic.** Greedy decoding
  with a fixed item order is a pure function of the checkpoint: running it five times
  produces five copies of one number, and bootstrapping five copies yields a
  zero-width interval that looks like enormous precision and means nothing. Anything
  that resamples seed-level values therefore goes through `is_degenerate`, which flags
  identical replicates, and the primary interval for a per-item proportion is the
  **paired item-level bootstrap** in `paired_delta_gate` — items are the sampling unit
  that actually varies. `run_condition` also permutes episode order per seed so that
  the seed genuinely enters the measurement; the two together are what make a CI
  reportable.
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
    "is_degenerate",
    "non_overlapping",
    "paired_bootstrap_delta",
    "paired_delta_gate",
    "summarise_seeds",
]


def is_degenerate(values: Sequence[float], *, tol: float = 1e-12) -> bool:
    """True when every replicate is the same number (to `tol`).

    Five identical values are one observation written down five times. Their bootstrap
    CI is a point, their std is 0, and reporting either as an uncertainty estimate
    overstates precision by an unbounded amount.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size < 2:
        return True
    return bool(np.nanmax(arr) - np.nanmin(arr) <= tol)


@dataclass
class SeedStats:
    mean: float
    std: float
    n: int
    ci_low: float
    ci_high: float
    values: list[float] = field(default_factory=list)
    method: str = "bootstrap-percentile"

    @property
    def degenerate(self) -> bool:
        return is_degenerate(self.values)

    def to_dict(self) -> dict:
        d = {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "n": self.n,
            "ci95": [round(self.ci_low, 4), round(self.ci_high, 4)],
            "method": self.method,
            "values": [round(v, 4) for v in self.values],
            "degenerate_replicates": self.degenerate,
        }
        if self.degenerate and self.n > 1:
            d["ci95_note"] = (
                f"all {self.n} replicates are identical, so this interval has zero "
                "width by construction and is NOT an uncertainty estimate. Use the "
                "paired item-level interval instead."
            )
        return d


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


def paired_bootstrap_delta(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    alpha: float = 0.05,
    n_boot: int = 10_000,
    seed: int = 42,
    clusters: Sequence[object] | None = None,
) -> dict:
    """Percentile bootstrap CI on the treatment-minus-baseline delta, **paired by item**.

    `treatment[i]` and `baseline[i]` must be the same forget-set item scored under the
    two conditions (0.0 / 1.0 for a recovery indicator). Items are resampled with
    replacement *together*, so the item-level correlation between the two arms — which
    is very high, they are the same questions — is preserved instead of being thrown
    away by treating the two arms as independent samples.

    `clusters` optionally groups items that are not independent of one another. TOFU
    is built from 200 synthetic authors with 20 questions each, so questions about one
    author share a great deal; passing the author id resamples whole authors and stops
    the interval from claiming n=400 independent observations when it has 200 at best.
    """
    t = np.asarray(list(treatment), dtype=float)
    b = np.asarray(list(baseline), dtype=float)
    if t.size != b.size:
        raise ValueError(f"paired arms must have equal length, got {t.size} and {b.size}")
    if t.size == 0:
        return {
            "delta": float("nan"),
            "ci95": [float("nan"), float("nan")],
            "n_pairs": 0,
            "method": "paired-percentile-bootstrap",
        }

    diff = t - b
    rng = np.random.default_rng(seed)

    if clusters is None:
        idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
        means = diff[idx].mean(axis=1)
        unit, n_units = "item", int(diff.size)
    else:
        keys = list(clusters)
        if len(keys) != diff.size:
            raise ValueError("clusters must be the same length as the paired arms")
        groups: dict[object, list[int]] = {}
        for i, key in enumerate(keys):
            groups.setdefault(key, []).append(i)
        blocks = [np.asarray(v, dtype=int) for v in groups.values()]
        n_units = len(blocks)
        pick = rng.integers(0, n_units, size=(n_boot, n_units))
        means = np.empty(n_boot, dtype=float)
        for r in range(n_boot):
            take = np.concatenate([blocks[j] for j in pick[r]])
            means[r] = diff[take].mean()
        unit = "cluster"

    return {
        "delta": float(diff.mean()),
        "ci95": [
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))),
        ],
        "n_pairs": int(diff.size),
        "resampling_unit": unit,
        "n_units": n_units,
        "method": "paired-percentile-bootstrap",
    }


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

    # A zero-width interval from identical replicates trivially satisfies
    # "non-overlapping" while carrying no information at all. Refuse to score it.
    degenerate = t.degenerate or b.degenerate
    ok_ci = (not require_non_overlapping) or non_overlapping(t, b)
    if require_non_overlapping and degenerate:
        ok_ci = False
        reasons.append(
            "seed-level replicates are identical, so the seed-level CIs have zero "
            "width and cannot support a non-overlap claim. Report the paired "
            "item-level interval (paired_delta_gate) as the primary uncertainty, and "
            "vary something real across seeds (episode order, sampling) before "
            "calling five reruns five seeds."
        )
    elif not ok_ci:
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
            "degenerate_replicates": degenerate,
            "kill_criterion_triggered": kill,
        },
        reason="; ".join(reasons) if reasons else "all pre-registered criteria met",
    )


def paired_delta_gate(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    min_delta_points: float = 20.0,
    clusters: Sequence[object] | None = None,
    name: str = "paired item-level delta",
    seed: int = 42,
) -> GateResult:
    """The primary gate for a per-item recovery proportion.

    Passes when the paired delta clears `min_delta_points` **and** the 95% interval
    excludes zero. Unlike the seed-level gate this cannot be satisfied by a degenerate
    replicate set: its sampling unit is the item (or the cluster), which varies whether
    or not decoding is stochastic.
    """
    res = paired_bootstrap_delta(treatment, baseline, clusters=clusters, seed=seed)
    delta_points = 100.0 * res["delta"]
    lo, hi = res["ci95"]

    reasons: list[str] = []
    ok_delta = delta_points >= min_delta_points
    if not ok_delta:
        reasons.append(f"paired delta {delta_points:.1f} points < required {min_delta_points:.0f}")
    excludes_zero = bool(lo > 0.0 or hi < 0.0)
    if not excludes_zero:
        reasons.append(f"95% paired interval [{lo:.3f}, {hi:.3f}] includes zero")

    return GateResult(
        name=name,
        passed=ok_delta and excludes_zero,
        detail={
            **res,
            "delta_points": round(delta_points, 2),
            "min_delta_points": min_delta_points,
            "interval_excludes_zero": excludes_zero,
        },
        reason="; ".join(reasons) if reasons else "paired criteria met",
    )
