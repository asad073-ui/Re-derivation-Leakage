"""The study's arm-to-arm comparisons, in both directions.

Two families, and conflating them is how a benchmark ends up with only half a result:

**Defence contrasts** — ours vs unguarded, ours vs the DRAGON-style template baseline.
The claim is a *reduction*, so support needs ``ci_high < 0``.

**Composition contrasts** (GU-0030) — MA-LEAK vs the single agent, MA-LEAK vs the
cross-concept multi-agent control, and the DRAGON-style baseline vs MA-LEAK. The claim
here is an *increase*: that composing agents reconstructs what neither isolated condition
released. Support needs ``ci_low > 0``. These are the contrasts the leakage phenomenon
itself rests on and they were never computed — the discovery report stated the phenomenon
from three point estimates (0.08 / 0.04 / 0.16) with no interval on any of the three
differences.

Both are evaluated **at the same k**, on arms produced by the same run. Comparing a new
Leak@32 against an older Leak@16, or against a number from a different protocol, is not
a comparison: Leak@k increases with k by construction, because more draws give more
chances to leak.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .graph_leak import GraphLeakTable, metric_applicability
from .graph_statistics import paired_delta

__all__ = [
    "COMPOSITION_CONTRASTS",
    "MECHANISM_CONTRASTS",
    "ComparisonResult",
    "compare_arms",
    "composition_report",
    "hypothesis_report",
    "mechanism_report",
]

# THE MECHANISM DECOMPOSITION (GU-0031/GU-0032/GU-0033).
#
# (id, treatment, baseline, what exactly this pair varies). Every one is a SINGLE-VARIABLE
# contrast between two arms of the same run at the same k, and every one is computed
# directly rather than inferred by subtracting two comparisons against the full defence.
#
# Why direct computation is the point: the report used to state only `full vs X`, and
# "stateless beat DRAGON" was then read off the difference of two intervals against a
# third arm. That is not a paired test — the pairing and the concept clustering are lost —
# and it is exactly the reasoning that produced GU-0030's over-claim. Reanalysing the raw
# rows afterwards is possible but is the same mistake in a new place: fixing the reporting
# after the generation has been paid for.
MECHANISM_CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "M1",
        "multi_agent_dragon",
        "multi_agent_leak",
        "does a node-local guard at every agent input help at all? Reported, never "
        "assumed: this is the same pair as composition contrast C3, stated here as the "
        "reduction claim the mechanism ladder starts from",
    ),
    (
        "M2",
        "multi_agent_stateless",
        "multi_agent_dragon",
        "BOUNDARY COVERAGE alone: the same detector, node-local at every input versus "
        "enforced at all five surfaces, with no accumulation and no provenance on either "
        "side",
    ),
    (
        "M3",
        "multi_agent_graphforget_semantic_only",
        "multi_agent_stateless",
        "SUBSET / EVIDENCE ACCUMULATION alone: five surfaces on both sides, neither "
        "consuming nor forwarding provenance",
    ),
    (
        "M4",
        "multi_agent_graphforget_tag_local_only",
        "multi_agent_leak",
        "ENFORCING EXISTING TAGS alone: no detector, no accumulation, no forwarding — "
        "only the refusal to release a source that already carries a Forget-ID",
    ),
    (
        "M5",
        "multi_agent_graphforget_taint_only",
        "multi_agent_graphforget_tag_local_only",
        "FORWARD PROPAGATION alone, and the only contrast that supports the propagation "
        "claim: both arms enforce existing tags with the detector off, and only the "
        "treatment attaches scopes to what it produces",
    ),
    (
        "M6",
        "multi_agent_graphforget",
        "multi_agent_graphforget_semantic_only",
        "PROVENANCE on top of full semantics: same detector, same accumulation, same five "
        "surfaces; the treatment additionally consumes and forwards Forget-IDs",
    ),
    (
        "M7",
        "multi_agent_graphforget",
        "multi_agent_graphforget_taint_only",
        "SEMANTICS on top of provenance: both arms carry and forward scopes; only the "
        "treatment can detect one that was never tagged",
    ),
)

# (id, treatment, baseline, what the contrast establishes). Stated as increases, because
# that is the direction the phenomenon claim runs in.
COMPOSITION_CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    (
        "C1",
        "multi_agent_leak",
        "single_agent",
        "composing agents leaks more than one agent asked the same question",
    ),
    (
        "C2",
        "multi_agent_leak",
        "multi_agent_control",
        "the excess is same-concept collaboration, not multi-agent chatter: the control "
        "runs the identical topology on a different concept",
    ),
    (
        "C3",
        "multi_agent_dragon",
        "multi_agent_leak",
        "whether the node-local template baseline helps at all. Reported, never assumed — "
        "an observed higher number is not a claim that a published system is worse",
    ),
)


@dataclass(frozen=True)
class ComparisonResult:
    surface: str
    treatment: str
    baseline: str
    k: int
    stats: dict
    direction: str = "decrease"

    @property
    def supported(self) -> bool:
        """Point estimate in the claimed direction AND an interval that excludes zero.

        An undefined interval (``None``, from a degenerate bootstrap) is not support.
        Treating a missing bound as a passing one is how a hypothesis gets declared
        supported by an absence of evidence.

        The bound that matters depends on which way the claim points. A reduction needs
        ``ci_high < 0``; an increase needs ``ci_low > 0``. Reading a reduction's bound for
        an increase claim would declare every contrast unsupported and quietly bury the
        phenomenon result.
        """
        absolute = self.stats.get("absolute_reduction")
        if absolute is None:
            return False
        if self.direction == "increase":
            ci_low = self.stats.get("ci_low")
            return bool(ci_low is not None and absolute > 0 and ci_low > 0)
        ci_high = self.stats.get("ci_high")
        return bool(ci_high is not None and absolute < 0 and ci_high < 0)

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "treatment": self.treatment,
            "baseline": self.baseline,
            "k": self.k,
            "direction": self.direction,
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
    direction: str = "decrease",
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
        surface=table.surface,
        treatment=treatment,
        baseline=baseline,
        k=k,
        stats=stats,
        direction=direction,
    )


def composition_report(
    tables: Mapping[str, GraphLeakTable],
    *,
    challenge: str,
    k: int,
    surfaces: Sequence[str] | None = None,
    reps: int = 2000,
    seed: int = 20260812,
) -> dict:
    """The three contrasts the leakage phenomenon rests on, with clustered intervals.

    Computed on the surfaces this CHALLENGE declares primary or secondary, and never on
    one it declares ``invalid`` — ranking arms on the rootless subtype under memory
    re-entry is exactly the mistake this function exists to stop being repeatable.

    ``supported`` here is an increase claim. C3 is stated in the same shape for symmetry
    but is genuinely two-sided in interpretation: whether a node-local guard helps, hurts
    or does nothing in a graph is an empirical question, and a higher observed number for
    a template baseline is not a finding about a published system.
    """
    roles = metric_applicability(challenge)
    if surfaces is None:
        surfaces = [s for s in tables if roles.get(s, {}).get("ranks_arms", True)]
    else:
        surfaces = [s for s in surfaces if s in tables]

    contrasts: list[dict] = []
    for contrast_id, treatment, baseline, statement in COMPOSITION_CONTRASTS:
        for surface in surfaces:
            table = tables[surface]
            if treatment not in table.arms() or baseline not in table.arms():
                continue
            role = roles.get(surface, {})
            result = compare_arms(
                table,
                treatment=treatment,
                baseline=baseline,
                k=k,
                reps=reps,
                seed=seed,
                direction="increase",
            )
            contrasts.append(
                {
                    "id": contrast_id,
                    "statement": statement,
                    "surface_role": role.get("role", "diagnostic"),
                    **result.to_dict(),
                }
            )

    primary = [c for c in contrasts if c["surface_role"] == "primary"]
    return {
        "challenge": challenge,
        "k": k,
        "contrasts": contrasts,
        "primary_surfaces": sorted({c["surface"] for c in primary}),
        # Only C1 and C2 are the phenomenon claim; C3 is reported, not required.
        "phenomenon_supported": bool(primary)
        and all(c["supported"] for c in primary if c["id"] in ("C1", "C2")),
        "excluded_surfaces": {
            s: roles[s]["reason"] for s in tables if not roles.get(s, {}).get("ranks_arms", True)
        },
        "note": (
            "increase claims: support requires ci_low > 0. Every contrast is between arms "
            "of the SAME run at the SAME k, paired at the shared sample index and "
            "resampled by concept."
        ),
    }


def mechanism_report(
    tables: Mapping[str, GraphLeakTable],
    *,
    challenge: str,
    k: int,
    surfaces: Sequence[str] | None = None,
    reps: int = 2000,
    seed: int = 20260812,
) -> dict:
    """Every single-variable mechanism contrast, computed directly and paired.

    Same machinery as the other two families — ``paired_delta`` at the shared sample
    index, resampled by concept — so a mechanism claim is stated on the same evidence
    standard as the headline defence claim rather than on a difference of point estimates.

    A contrast whose arms are not both present is reported as MISSING rather than
    silently omitted. An absent row and a null result read identically in a table, and
    only one of them is a measurement: a study that forgot to run `tag_local_only` would
    otherwise produce a mechanism section that looks complete and cannot support the
    propagation claim.
    """
    roles = metric_applicability(challenge)
    if surfaces is None:
        surfaces = [s for s in tables if roles.get(s, {}).get("ranks_arms", True)]
    else:
        surfaces = [s for s in surfaces if s in tables]

    contrasts: list[dict] = []
    missing: list[dict] = []
    for contrast_id, treatment, baseline, statement in MECHANISM_CONTRASTS:
        present_anywhere = False
        for surface in surfaces:
            table = tables[surface]
            if treatment not in table.arms() or baseline not in table.arms():
                continue
            present_anywhere = True
            result = compare_arms(
                table,
                treatment=treatment,
                baseline=baseline,
                k=k,
                reps=reps,
                seed=seed,
                direction="decrease",
            )
            contrasts.append(
                {
                    "id": contrast_id,
                    "statement": statement,
                    "surface_role": roles.get(surface, {}).get("role", "diagnostic"),
                    **result.to_dict(),
                }
            )
        if not present_anywhere:
            missing.append(
                {
                    "id": contrast_id,
                    "treatment": treatment,
                    "baseline": baseline,
                    "statement": statement,
                    "reason": "one or both arms were not run",
                }
            )

    primary = [c for c in contrasts if c["surface_role"] == "primary"]
    return {
        "challenge": challenge,
        "k": k,
        "contrasts": contrasts,
        "missing_contrasts": missing,
        "primary_surfaces": sorted({c["surface"] for c in primary}),
        "complete": not missing,
        "supported_ids": sorted({c["id"] for c in primary if c["supported"]}),
        # The propagation claim has exactly one supporting contrast. Named here so a
        # report cannot imply it from M6, which varies consumption and forwarding together.
        "propagation_contrast": "M5",
        "propagation_supported": bool(
            primary and any(c["id"] == "M5" and c["supported"] for c in primary)
        ),
        "note": (
            "reduction claims: support requires ci_high < 0. Every contrast is between two "
            "arms of the SAME run at the SAME k, paired at the shared sample index and "
            "resampled by concept. M5 is the only contrast that isolates forward "
            "propagation; M6 varies consumption and forwarding together and cannot stand "
            "in for it."
        ),
    }


def hypothesis_report(
    tables: Mapping[str, GraphLeakTable],
    *,
    treatment: str = "multi_agent_graphforget",
    baselines: Sequence[str] = ("multi_agent_leak", "multi_agent_dragon"),
    challenge: str = "natural",
    primary_surfaces: Sequence[str] | None = None,
    k: int,
    reps: int = 2000,
    seed: int = 20260812,
) -> dict:
    """The defence hypotheses at the primary k, plus every surface for the appendix.

    The primary surface is now a property of the CHALLENGE rather than a single global
    constant (GU-0030). It used to be ``certified_persistent_leak`` for every challenge,
    which under memory re-entry ranked the arms on a metric that is inverted there — the
    50x32 discovery run's headline said GraphForget was the only arm to leak (0.16 against
    0.00) on the very challenge where it cut total persistent leakage from 0.58 to 0.18.

    Note what is still NOT asserted: that the DRAGON-style baseline must beat the
    unguarded system. Whether a node-local guard helps at all in a multi-agent graph is an
    empirical question and is reported, not required.
    """
    from .graph_leak import primary_surfaces as declared_primary

    roles = metric_applicability(challenge)
    chosen = list(primary_surfaces) if primary_surfaces is not None else declared_primary(challenge)
    chosen = [s for s in chosen if s in tables]

    hypotheses: list[dict] = []
    for surface in chosen:
        table = tables[surface]
        for index, baseline in enumerate(baselines, start=1):
            if treatment not in table.arms() or baseline not in table.arms():
                continue
            result = compare_arms(
                table, treatment=treatment, baseline=baseline, k=k, reps=reps, seed=seed
            )
            hypotheses.append(
                {
                    "id": f"H{index}",
                    "statement": f"L_{treatment}({k}) < L_{baseline}({k}) on {surface}",
                    "surface_role": "primary",
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
                {
                    "surface_role": roles.get(surface, {}).get("role", "diagnostic"),
                    "ranks_arms": roles.get(surface, {}).get("ranks_arms", True),
                    **compare_arms(
                        table, treatment=treatment, baseline=baseline, k=k, reps=reps, seed=seed
                    ).to_dict(),
                }
            )
        if rows:
            all_surfaces[surface] = rows

    return {
        "primary_k": k,
        "challenge": challenge,
        "primary_surfaces": chosen,
        "treatment": treatment,
        "baselines": list(baselines),
        "hypotheses": hypotheses,
        "all_supported": bool(hypotheses) and all(h["supported"] for h in hypotheses),
        "by_surface": all_surfaces,
        "excluded_from_ranking": {
            s: roles[s]["reason"] for s in tables if not roles.get(s, {}).get("ranks_arms", True)
        },
        "note": (
            "every comparison is between arms of the SAME run at the SAME k. Leak@k rises "
            "with k, so a cross-k or cross-protocol comparison is not a comparison. "
            "Surfaces this challenge declares `invalid` appear in by_surface for the "
            "record and are never used to rank arms."
        ),
    }
