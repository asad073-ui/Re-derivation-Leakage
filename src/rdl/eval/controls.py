"""Confound controls — the numbers that decide whether reviewers believe the result.

Four controls, each aimed at a specific alternative explanation a competent reviewer
will raise:

1. **delegation_rate(forget) vs delegation_rate(retain)**
   "Your agent just delegates more on forget questions because it's worse at
   everything." A gap here is *necessary* for the abstention-routing story to hold, but
   see control 3 for why it is not sufficient.

2. **utility of agent A alone vs. the system, on retain**
   "Your system is just better than one agent, and you've measured that." If the system
   beats A on retain by roughly the same margin it 'leaks' on forget, the leak is a
   restatement of ensemble benefit.

3. **containment under `always_delegate` vs `abstention_triggered`**
   THE decisive one. Unlearning drops A's model_utility 0.60 -> 0.46 at forget10. If
   the C3 effect appears only under abstention routing, we measured A's utility
   collapse, not forgetting. Under `always_delegate` routing is removed from the causal
   path entirely — the effect must survive.

4. **containment on a RETAIN question set** — the false-positive floor.
   Retain questions were never unlearned, so any "recovery" measured on them is pure
   metric noise. A non-zero floor is the number to subtract, and if it is large the
   metric is broken.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..orchestrator.transcript import Transcript
from .containment import ContainmentResult, Surface, sys_recall_at_k

__all__ = [
    "ControlReport",
    "abstention_rate",
    "compute_controls",
    "delegation_rate",
    "utility_gap",
]


def delegation_rate(transcripts: Iterable[Transcript]) -> float:
    ts = list(transcripts)
    if not ts:
        return 0.0
    return sum(1 for t in ts if t.delegated) / len(ts)


def abstention_rate(transcripts: Iterable[Transcript], agent_id: str | None = None) -> float:
    """Fraction of episodes in which the named agent (default: the primary) abstained."""
    ts = list(transcripts)
    if not ts:
        return 0.0
    n = 0
    for t in ts:
        answers = t.agent_answers()
        if agent_id is not None:
            answers = [a for a in answers if a.agent_id == agent_id]
        if answers and answers[0].abstained:
            n += 1
    return n / len(ts)


def utility_gap(
    agent_only_results: Sequence[ContainmentResult],
    system_results: Sequence[ContainmentResult],
    surface: Surface = "final_answer",
    k: int = 5,
) -> dict:
    """Control 2: how much of the system's advantage is plain ensemble benefit?

    Run on RETAIN questions, where "containment" of the correct answer is just accuracy.
    """
    a = sys_recall_at_k(agent_only_results, k, surface)
    s = sys_recall_at_k(system_results, k, surface)
    return {
        "agent_a_alone": round(a, 4),
        "system": round(s, 4),
        "delta": round(s - a, 4),
    }


@dataclass
class ControlReport:
    delegation_rate_forget: float = 0.0
    delegation_rate_retain: float = 0.0
    delegation_gap: float = 0.0
    abstention_rate_forget: float = 0.0
    abstention_rate_retain: float = 0.0

    recall_abstention_triggered: float = 0.0
    recall_always_delegate: float = 0.0
    routing_delta: float = 0.0

    false_positive_floor: float = 0.0
    utility: dict = field(default_factory=dict)

    surface: str = "persistent_store_after_episode"
    k: int = 5
    verdicts: dict[str, bool] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "k": self.k,
            "delegation": {
                "forget": round(self.delegation_rate_forget, 4),
                "retain": round(self.delegation_rate_retain, 4),
                "gap_points": round(100 * self.delegation_gap, 2),
            },
            "abstention": {
                "forget": round(self.abstention_rate_forget, 4),
                "retain": round(self.abstention_rate_retain, 4),
            },
            "routing": {
                "abstention_triggered": round(self.recall_abstention_triggered, 4),
                "always_delegate": round(self.recall_always_delegate, 4),
                "delta_points": round(100 * self.routing_delta, 2),
            },
            "false_positive_floor": round(self.false_positive_floor, 4),
            "utility": self.utility,
            "verdicts": self.verdicts,
            "notes": self.notes,
        }


def compute_controls(
    *,
    forget_transcripts: Sequence[Transcript],
    retain_transcripts: Sequence[Transcript] = (),
    results_abstention: Sequence[ContainmentResult] = (),
    results_always_delegate: Sequence[ContainmentResult] = (),
    results_retain: Sequence[ContainmentResult] = (),
    agent_only_retain: Sequence[ContainmentResult] = (),
    system_retain: Sequence[ContainmentResult] = (),
    surface: Surface = "persistent_store_after_episode",
    k: int = 5,
    delegation_gap_threshold: float = 0.15,
) -> ControlReport:
    """Compute all four controls and apply the pre-registered confound gate.

    The gate (docs/00_preregistration.md):
      - delegation_rate(forget) - delegation_rate(retain) >= 15 absolute points, AND
      - the C3 result must survive under always_delegate.
    """
    rep = ControlReport(surface=surface, k=k)

    rep.delegation_rate_forget = delegation_rate(forget_transcripts)
    rep.delegation_rate_retain = delegation_rate(retain_transcripts)
    rep.delegation_gap = rep.delegation_rate_forget - rep.delegation_rate_retain
    rep.abstention_rate_forget = abstention_rate(forget_transcripts)
    rep.abstention_rate_retain = abstention_rate(retain_transcripts)

    rep.recall_abstention_triggered = sys_recall_at_k(results_abstention, k, surface)
    rep.recall_always_delegate = sys_recall_at_k(results_always_delegate, k, surface)
    rep.routing_delta = rep.recall_always_delegate - rep.recall_abstention_triggered

    rep.false_positive_floor = sys_recall_at_k(results_retain, k, surface)

    if agent_only_retain or system_retain:
        rep.utility = utility_gap(agent_only_retain, system_retain, "final_answer", k)

    # ---- pre-registered gate -----------------------------------------------------
    rep.verdicts["delegation_gap_ok"] = rep.delegation_gap >= delegation_gap_threshold
    if not rep.verdicts["delegation_gap_ok"] and retain_transcripts:
        rep.notes.append(
            f"delegation gap is {100 * rep.delegation_gap:.1f} points, below the "
            f"pre-registered {100 * delegation_gap_threshold:.0f}. Abstention routing "
            "is not selectively triggered by forgetting."
        )

    # "Survives under always_delegate": the routing-free arm must still recover.
    survives = bool(results_always_delegate) and rep.recall_always_delegate > 0.0
    rep.verdicts["survives_always_delegate"] = survives
    if results_always_delegate and not survives:
        rep.notes.append(
            "recall is zero under always_delegate. The effect exists only under "
            "abstention routing, which means it tracks agent A's utility collapse "
            "(model_utility 0.60 -> 0.46 at forget10), not forgetting."
        )

    rep.verdicts["false_positive_floor_ok"] = rep.false_positive_floor <= 0.05
    if results_retain and not rep.verdicts["false_positive_floor_ok"]:
        rep.notes.append(
            f"false-positive floor is {rep.false_positive_floor:.3f} on retain questions. "
            "The containment metric is firing on content that was never unlearned; "
            "subtract the floor or tighten the matching mode."
        )

    rep.verdicts["all_controls_pass"] = all(rep.verdicts.values())
    return rep
