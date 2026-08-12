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

import math
from collections.abc import Mapping, Sequence

from .leak_at_k import hierarchical_bootstrap_delta, leak_at_k

__all__ = ["finite_or_none", "paired_delta", "relative_reduction"]


def finite_or_none(value: float | None) -> float | None:
    """``None`` for anything JSON cannot represent.

    Applied on the way out of every statistic, so a NaN produced anywhere upstream — a
    degenerate bootstrap, a zero denominator — becomes an explicit "undefined" instead of
    a token that makes the whole report unparseable. See ``atomic_json``.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def paired_delta(
    treatment: Mapping[str, Sequence[bool]],
    baseline: Mapping[str, Sequence[bool]],
    clusters: Mapping[str, str],
    *,
    k: int,
    reps: int = 2000,
    seed: int = 20260812,
) -> dict[str, float | bool | None]:
    """``L_treatment(k) - L_baseline(k)`` with a concept-clustered 95% interval.

    Negative is better. Reported alongside the relative reduction, because an absolute
    drop of 0.02 means something very different at a baseline of 0.04 than at 0.80.
    """
    result = hierarchical_bootstrap_delta(treatment, baseline, clusters, k=k, reps=reps, seed=seed)
    items = sorted(set(treatment) & set(baseline))
    l_treatment = sum(leak_at_k(treatment[i], k) for i in items) / len(items) if items else 0.0
    l_baseline = sum(leak_at_k(baseline[i], k) for i in items) / len(items) if items else 0.0
    out: dict[str, float | bool | None] = {
        # Every float that reaches a report goes through `finite_or_none` first, so an
        # undefined bootstrap statistic is null rather than an unparseable token.
        **{key: finite_or_none(value) for key, value in result.items()},
        "k": float(k),
        "leak_treatment": l_treatment,
        "leak_baseline": l_baseline,
        "absolute_reduction": l_treatment - l_baseline,
        "relative_reduction": relative_reduction(l_treatment, l_baseline),
        "n_items": float(len(items)),
        "n_clusters": float(len({clusters[i] for i in items})) if items else 0.0,
    }
    ci_high = out.get("ci_high")
    out["significant"] = bool(isinstance(ci_high, float) and ci_high < 0.0)
    return out


def relative_reduction(treatment: float, baseline: float) -> float | None:
    """``1 - L_treatment / L_baseline``. Undefined at a zero baseline, reported as ``None``.

    Returning "undefined" rather than 0.0 or 1.0 is the science: a baseline that never
    leaked supports no claim about relative reduction, and a number there would be read
    as one. It used to be ``float('nan')``, which said the same thing to a human and made
    the enclosing JSON invalid for every strict parser — the three committed graph
    reports with literal ``NaN`` in them are what that cost. ``None`` serialises as
    ``null``, which is the JSON spelling of "undefined".
    """
    if baseline <= 0.0:
        return None
    return finite_or_none(1.0 - (treatment / baseline))
