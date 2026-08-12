"""Paired, concept-clustered intervals for the graph study.

TOFU is 200 synthetic authors x 20 questions, so questions are not independent
observations; the resampling unit is the concept. Arms share seeds by construction
(``GraphExecutor.seed_for_node`` never keys on the arm), so the bootstrap is paired at
the sample index — treating the two arms' draws as unrelated would throw away exactly
the common-random-number design that makes the contrast tight.

This wraps the two-agent work's ``hierarchical_bootstrap_delta`` rather than
reimplementing it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .leak_at_k import hierarchical_bootstrap_delta, leak_at_k

__all__ = ["paired_delta", "relative_reduction"]


def paired_delta(
    treatment: Mapping[str, Sequence[bool]],
    baseline: Mapping[str, Sequence[bool]],
    clusters: Mapping[str, str],
    *,
    k: int,
    reps: int = 2000,
    seed: int = 20260812,
) -> dict[str, float]:
    """``L_treatment(k) - L_baseline(k)`` with a concept-clustered 95% interval.

    Negative is better. Reported alongside the relative reduction, because an absolute
    drop of 0.02 means something very different at a baseline of 0.04 than at 0.80.
    """
    result = hierarchical_bootstrap_delta(treatment, baseline, clusters, k=k, reps=reps, seed=seed)
    items = sorted(set(treatment) & set(baseline))
    l_treatment = sum(leak_at_k(treatment[i], k) for i in items) / len(items) if items else 0.0
    l_baseline = sum(leak_at_k(baseline[i], k) for i in items) / len(items) if items else 0.0
    return {
        **result,
        "k": float(k),
        "leak_treatment": l_treatment,
        "leak_baseline": l_baseline,
        "absolute_reduction": l_treatment - l_baseline,
        "relative_reduction": relative_reduction(l_treatment, l_baseline),
        "n_items": float(len(items)),
        "n_clusters": float(len({clusters[i] for i in items})) if items else 0.0,
        "significant": result["ci_high"] < 0.0,
    }


def relative_reduction(treatment: float, baseline: float) -> float:
    """``1 - L_treatment / L_baseline``. Undefined at a zero baseline, reported as NaN.

    Returning NaN rather than 0.0 or 1.0 is deliberate: a baseline that never leaked
    supports no claim about relative reduction, and a number there would be read as one.
    """
    if baseline <= 0.0:
        return float("nan")
    return 1.0 - (treatment / baseline)
