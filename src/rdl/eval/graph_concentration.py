"""How many concepts a Leak@k actually rests on, and whether one of them carries it.

A Leak@32 of 0.16 over 50 items is 8 affected items. If those 8 items are 8 different
TOFU authors the effect is distributed; if they are two authors with four questions each
it is a property of two authors, and the concept-clustered interval is the only thing
standing between that and a headline. The interval already accounts for it — these
functions make it *visible*, which is what stops a discovery number being read as a
population estimate.

Three questions, all asked of the same per-item sample series the Leak@k tables hold:

``affected``            how many items and how many concepts leaked at least once, and
                        the affected-concept share of the cohort.
``concentration``       the most-affected concept's share of all affected items, plus the
                        Herfindahl index over concepts. A top share near 1.0 means the
                        result IS that concept.
``leave_one_out``       recompute Leak@k with each concept dropped in turn. The spread
                        between the smallest and largest refit says how much the number
                        moves when one author is removed; ``sign_stable`` says whether a
                        contrast between two arms keeps its direction under every drop.

Leave-one-out is reported rather than gated. A discovery result that flips sign when one
of twenty authors is dropped is not thereby wrong — it is thereby *underpowered*, and the
honest response is to say so next to the number rather than to suppress it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .graph_statistics import finite_or_none
from .leak_at_k import leak_at_k

__all__ = ["affected", "concentration", "concept_profile", "leave_one_out"]

Series = Mapping[str, Sequence[bool]]
Clusters = Mapping[str, str]


def _leaking_items(series: Series) -> list[str]:
    return sorted(item for item, flags in series.items() if any(flags))


def _mean_leak(series: Series, k: int, items: Sequence[str]) -> float:
    if not items:
        return 0.0
    return sum(leak_at_k(series[i], k) for i in items) / len(items)


def affected(series: Series, clusters: Clusters) -> dict:
    """Item and concept counts behind a curve.

    ``n_affected_items`` is the number the reader almost always wants and almost never
    gets: Leak@32 = 0.16 over 50 items means 8 items leaked in at least one of 32 draws,
    not that 16% of 1,600 trajectories leaked.
    """
    items = sorted(series)
    leaking = _leaking_items(series)
    concepts = {clusters[i] for i in items if i in clusters}
    leaking_concepts = {clusters[i] for i in leaking if i in clusters}
    return {
        "n_items": len(items),
        "n_affected_items": len(leaking),
        "n_concepts": len(concepts),
        "n_affected_concepts": len(leaking_concepts),
        "affected_item_share": (len(leaking) / len(items)) if items else 0.0,
        "affected_concept_share": (len(leaking_concepts) / len(concepts)) if concepts else 0.0,
        "affected_concepts": sorted(leaking_concepts),
    }


def concentration(series: Series, clusters: Clusters) -> dict:
    """Is the effect spread over concepts, or is it one concept?

    ``top_concept_share`` is over AFFECTED items, so it answers "of everything that
    leaked, how much came from the worst author". ``herfindahl`` is the sum of squared
    shares: 1.0 when a single concept supplies every affected item, 1/n when they are
    spread evenly over n concepts.
    """
    leaking = _leaking_items(series)
    by_concept: dict[str, int] = {}
    for item in leaking:
        concept = clusters.get(item, item)
        by_concept[concept] = by_concept.get(concept, 0) + 1
    total = sum(by_concept.values())
    if not total:
        return {
            "n_affected_items": 0,
            "top_concept": None,
            "top_concept_items": 0,
            "top_concept_share": None,
            "herfindahl": None,
            "by_concept": {},
        }
    top = max(sorted(by_concept), key=lambda c: by_concept[c])
    return {
        "n_affected_items": total,
        "top_concept": top,
        "top_concept_items": by_concept[top],
        "top_concept_share": by_concept[top] / total,
        "herfindahl": finite_or_none(sum((n / total) ** 2 for n in by_concept.values())),
        "by_concept": dict(sorted(by_concept.items(), key=lambda kv: (-kv[1], kv[0]))),
    }


def leave_one_out(
    series: Series,
    clusters: Clusters,
    *,
    k: int,
    baseline: Series | None = None,
) -> dict:
    """Recompute Leak@k with each concept dropped in turn.

    With ``baseline``, the quantity refitted is the CONTRAST ``L_series - L_baseline`` on
    the items the two share, and ``sign_stable`` reports whether every drop leaves the
    contrast pointing the same way as the full-cohort estimate. That is the check a
    twenty-author discovery cohort most needs and the one a bootstrap interval does not
    display directly.
    """
    items = sorted(series if baseline is None else set(series) & set(baseline))
    concepts = sorted({clusters.get(i, i) for i in items})

    def statistic(kept: Sequence[str]) -> float:
        value = _mean_leak(series, k, kept)
        if baseline is not None:
            value -= _mean_leak(baseline, k, kept)
        return value

    full = statistic(items)
    refits: dict[str, float] = {}
    for concept in concepts:
        kept = [i for i in items if clusters.get(i, i) != concept]
        if kept:
            refits[concept] = statistic(kept)

    values = sorted(refits.values())
    swing = (values[-1] - values[0]) if values else 0.0
    # Two ways stability is vacuous rather than true, both reported as None: nothing was
    # refitted, or the full estimate is exactly zero and so has no sign to preserve.
    # Either would otherwise emit a reassuring boolean that means nothing.
    sign_stable: bool | None
    if not values or full == 0.0:
        sign_stable = None
    else:
        sign_stable = all((v > 0) == (full > 0) and v != 0.0 for v in values)

    worst = max(refits, key=lambda c: abs(refits[c] - full)) if refits else None
    return {
        "k": k,
        "statistic": "contrast" if baseline is not None else "leak_at_k",
        "full": finite_or_none(full),
        "n_concepts": len(concepts),
        "min": finite_or_none(values[0]) if values else None,
        "max": finite_or_none(values[-1]) if values else None,
        "swing": finite_or_none(swing),
        "sign_stable": sign_stable,
        "most_influential_concept": worst,
        "most_influential_delta": finite_or_none(refits[worst] - full) if worst else None,
        "by_dropped_concept": {c: finite_or_none(v) for c, v in sorted(refits.items())},
    }


def concept_profile(
    series: Series,
    clusters: Clusters,
    *,
    k: int,
    baseline: Series | None = None,
) -> dict:
    """``affected`` + ``concentration`` + ``leave_one_out`` for one arm, in one object."""
    return {
        **affected(series, clusters),
        "concentration": concentration(series, clusters),
        "leave_one_concept_out": leave_one_out(series, clusters, k=k, baseline=baseline),
    }
