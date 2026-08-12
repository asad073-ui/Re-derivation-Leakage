"""The primary comparisons: ours vs unguarded, and ours vs the DRAGON-style baseline.

Both are evaluated **at the same k**, on arms produced by the same run. Comparing a new
Leak@32 against an older Leak@16, or against a number from a different protocol, is not
a comparison: Leak@k increases with k by construction, because more draws give more
chances to leak.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .graph_leak import GraphLeakTable
from .graph_statistics import paired_delta

__all__ = ["ComparisonResult", "compare_arms", "hypothesis_report"]


@dataclass(frozen=True)
class ComparisonResult:
    surface: str
    treatment: str
    baseline: str
    k: int
    stats: dict

    @property
    def supported(self) -> bool:
        """Lower point estimate AND an interval that excludes zero.

        An undefined interval (``None``, from a degenerate bootstrap) is not support.
        Treating a missing bound as a passing one is how a hypothesis gets declared
        supported by an absence of evidence.
        """
        absolute = self.stats.get("absolute_reduction")
        ci_high = self.stats.get("ci_high")
        if absolute is None or ci_high is None:
            return False
        return bool(absolute < 0 and ci_high < 0)

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "treatment": self.treatment,
            "baseline": self.baseline,
            "k": self.k,
            "supported": self.supported,
            **dict(self.stats),
        }


def compare_arms(
    table: GraphLeakTable,
    *,
    treatment: str,
    baseline: str,
    k: int,
    reps: int = 2000,
    seed: int = 20260812,
) -> ComparisonResult:
    treatment_series = table.series(treatment)
    baseline_series = table.series(baseline)
    shared = sorted(set(treatment_series) & set(baseline_series))
    if not shared:
        raise ValueError(
            f"arms '{treatment}' and '{baseline}' share no items on surface "
            f"'{table.surface}'; they cannot be compared"
        )
    stats = paired_delta(
        {i: treatment_series[i] for i in shared},
        {i: baseline_series[i] for i in shared},
        {i: table.concept_of[i] for i in shared},
        k=k,
        reps=reps,
        seed=seed,
    )
    return ComparisonResult(
        surface=table.surface, treatment=treatment, baseline=baseline, k=k, stats=stats
    )


def hypothesis_report(
    tables: Mapping[str, GraphLeakTable],
    *,
    treatment: str = "multi_agent_graphforget",
    baselines: Sequence[str] = ("multi_agent_leak", "multi_agent_dragon"),
    primary_surface: str = "certified_persistent_leak",
    k: int,
    reps: int = 2000,
    seed: int = 20260812,
) -> dict:
    """H1 and H2 at the primary k, plus every surface for the appendix.

    Note what is NOT asserted: that the DRAGON-style baseline must beat the unguarded
    system. Whether a node-local guard helps at all in a multi-agent graph is an
    empirical question and is reported, not required.
    """
    hypotheses: list[dict] = []
    for index, baseline in enumerate(baselines, start=1):
        table = tables[primary_surface]
        if treatment not in table.arms() or baseline not in table.arms():
            continue
        result = compare_arms(
            table, treatment=treatment, baseline=baseline, k=k, reps=reps, seed=seed
        )
        hypotheses.append(
            {
                "id": f"H{index}",
                "statement": f"L_{treatment}({k}) < L_{baseline}({k}) on {primary_surface}",
                **result.to_dict(),
            }
        )

    all_surfaces: dict[str, list[dict]] = {}
    for surface, table in tables.items():
        rows: list[dict] = []
        for baseline in baselines:
            if treatment not in table.arms() or baseline not in table.arms():
                continue
            rows.append(
                compare_arms(
                    table, treatment=treatment, baseline=baseline, k=k, reps=reps, seed=seed
                ).to_dict()
            )
        if rows:
            all_surfaces[surface] = rows

    return {
        "primary_k": k,
        "primary_surface": primary_surface,
        "treatment": treatment,
        "baselines": list(baselines),
        "hypotheses": hypotheses,
        "all_supported": bool(hypotheses) and all(h["supported"] for h in hypotheses),
        "by_surface": all_surfaces,
        "note": (
            "every comparison is between arms of the SAME run at the SAME k. Leak@k rises "
            "with k, so a cross-k or cross-protocol comparison is not a comparison."
        ),
    }
