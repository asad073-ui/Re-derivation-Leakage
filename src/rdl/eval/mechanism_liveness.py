"""Did the pathway the mechanism variable acts on actually get walked?

GU-0034 established that a contrast can vary exactly one flag and still measure nothing,
because the pathway the flag acts on is never exercised. `tag_source_quarantine` versus
`taint_only` differ only in forwarding, and both quarantine the tagged note at retrieval
before any agent reads it — so no derivative is produced, nothing exists to forward, and
the difference is zero at any cohort size.

GU-0036 closes the reporting half of that. The report's `mechanism_measurement_valid`
checked STATIC arm properties only: it read `propagates_scope` off the manifest, which is
a restatement of the config. Nothing in the artifact required that

* the tagged source actually reached a model in both arms,
* neither arm withheld it,
* the baseline produced no inherited-only enforcement,
* the treatment produced some,
* and the forwarding reached the edge and write surfaces rather than only the sink.

Those are DYNAMIC facts, and they live in PERFORMANCE.json's per-arm defence counters.
Until this module they were checked by an operator script that lived outside the
repository, which means the gate was only as good as whoever remembered to run it. The
run that produced the M5 result passed those checks; the artifact could not say so.

The conditions are expressed once, here, and consumed by both the report gate and
`rdl graph-audit-mechanism`, so the two cannot drift.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

__all__ = [
    "M5_BASELINE",
    "M5_TREATMENT",
    "M8_ARMS",
    "LivenessAudit",
    "audit_mechanism_liveness",
]

# The propagation contrast. Named here rather than passed in: this module encodes what
# M5 IS, and a caller that could rename the arms could quietly audit a different pair.
M5_BASELINE = "multi_agent_graphforget_no_forward"
M5_TREATMENT = "multi_agent_graphforget_taint_forward"

# The two arms whose vacuity is a PREDICTION of the design, not a defect. Both quarantine
# the tagged source at retrieval, so both must show zero tagged content reaching a model.
# If either ever shows a nonzero read, the source-quarantine control has stopped
# quarantining and M8's structural ~0 would no longer be structural.
M8_ARMS = ("multi_agent_graphforget_tag_source_quarantine", "multi_agent_graphforget_taint_only")


def _enforcements(stats: Mapping, surface: str) -> int:
    """Inherited-only interventions at one surface: everything that is not `allow`."""
    cell = ((stats.get("causal_attribution") or {}).get("by_surface") or {}).get(surface) or {}
    inherited = cell.get("inherited_only") or {}
    return sum(int(n) for action, n in inherited.items() if action != "allow")


def _inherited_only_total(stats: Mapping) -> int | None:
    attribution = stats.get("causal_attribution") or {}
    value = attribution.get("inherited_only_enforcements")
    return None if value is None else int(value)


@dataclass
class LivenessAudit:
    """The verdict, plus every condition that produced it.

    `conditions` is dense: a condition that passed is present with `True` rather than
    absent, because a missing key and a failed check read the same in an artifact and
    only one of them is a measurement. That is the same rule `AttributionLedger` follows.
    """

    applicable: bool
    blockers: list[str] = field(default_factory=list)
    conditions: dict[str, bool] = field(default_factory=dict)
    observed: dict[str, object] = field(default_factory=dict)
    m8_vacuous_as_designed: bool | None = None

    @property
    def live(self) -> bool:
        """True only when the audit ran AND every condition held."""
        return self.applicable and not self.blockers

    def to_dict(self) -> dict:
        return {
            "schema": "graph-mechanism-liveness-v1",
            "contrast": "M5",
            "treatment": M5_TREATMENT,
            "baseline": M5_BASELINE,
            "applicable": self.applicable,
            "pathway_live": self.live,
            "blockers": list(self.blockers),
            "conditions": dict(self.conditions),
            "observed": dict(self.observed),
            "m8_vacuous_as_designed": self.m8_vacuous_as_designed,
            "note": (
                "static arm purity says the two arms are CONFIGURED to differ only in "
                "forwarding. These conditions say the pathway that difference acts on was "
                "actually walked: matched tagged-source exposure, nothing withheld, no "
                "inherited-only enforcement in the baseline, and forwarding that reached "
                "the edge and write surfaces in the treatment."
            ),
        }


def audit_mechanism_liveness(
    defenses: Mapping[str, Mapping] | None,
    *,
    arms_present: Sequence[str] | None = None,
) -> LivenessAudit:
    """Audit the M5 pathway from PERFORMANCE.json's `defenses` block.

    ``applicable`` is False — with no blockers — when this run did not carry the M5 pair
    at all. A run of a different study is not a failing mechanism audit; it is a run the
    question was never asked of. Callers must not read `not live` as `failed`.
    """
    audit = LivenessAudit(applicable=False)
    stats_by_arm = dict(defenses or {})

    if arms_present is not None:
        pair_planned = {M5_BASELINE, M5_TREATMENT} <= set(arms_present)
    else:
        pair_planned = {M5_BASELINE, M5_TREATMENT} <= set(stats_by_arm)
    if not pair_planned:
        return audit

    audit.applicable = True

    base = stats_by_arm.get(M5_BASELINE)
    treat = stats_by_arm.get(M5_TREATMENT)
    if not base or not treat:
        audit.blockers.append(
            "PERFORMANCE.json carries no defence counters for one or both M5 arms, so the "
            "pathway cannot be shown to have been walked"
        )
        audit.conditions["counters_present"] = False
        return audit
    audit.conditions["counters_present"] = True

    # ---- the arms are the ablations they claim to be -----------------------------
    for label, stats, propagates in ((M5_BASELINE, base, False), (M5_TREATMENT, treat, True)):
        ok = stats.get("propagate_forget_ids") is propagates
        audit.conditions[f"{label}.propagate_forget_ids_is_{str(propagates).lower()}"] = ok
        if not ok:
            audit.blockers.append(
                f"{label} recorded propagate_forget_ids="
                f"{stats.get('propagate_forget_ids')!r}, so the pair does not isolate forwarding"
            )
        consumes = stats.get("consume_forget_ids") is True
        audit.conditions[f"{label}.consumes_scope"] = consumes
        if not consumes:
            audit.blockers.append(
                f"{label} does not consume Forget-IDs, so it cannot enforce on the tag it "
                "was handed and the contrast is not the one M5 describes"
            )
        detector_off = stats.get("semantic_detection") is False
        audit.conditions[f"{label}.semantic_detection_off"] = detector_off
        if not detector_off:
            audit.blockers.append(
                f"{label} ran with semantic detection ON, so an enforcement attributed to "
                "provenance could have been produced by the detector instead"
            )

    # ---- the tagged source reached a model in BOTH arms --------------------------
    for label, stats in ((M5_BASELINE, base), (M5_TREATMENT, treat)):
        hits = int(stats.get("memory_borne_scope_hits") or 0)
        audit.observed[f"{label}.memory_borne_scope_hits"] = hits
        ok = hits > 0
        audit.conditions[f"{label}.tagged_source_reached_a_model"] = ok
        if not ok:
            audit.blockers.append(
                f"{label} recorded memory_borne_scope_hits=0: no tagged content ever "
                "reached a model in this arm, so no derivative exists to forward and the "
                "contrast is vacuous"
            )

        withheld = int(stats.get("retrieval_withheld") or 0)
        audit.observed[f"{label}.retrieval_withheld"] = withheld
        ok = withheld == 0
        audit.conditions[f"{label}.source_not_withheld"] = ok
        if not ok:
            audit.blockers.append(
                f"{label} withheld {withheld} retrievals: an M5 arm that quarantines the "
                "tagged source is the vacuous design GU-0034 removed, not a matched baseline"
            )

    # ---- exposure is MATCHED, not merely nonzero in each --------------------------
    base_hits = int(base.get("memory_borne_scope_hits") or 0)
    treat_hits = int(treat.get("memory_borne_scope_hits") or 0)
    ok = base_hits == treat_hits
    audit.conditions["tagged_source_exposure_matched"] = ok
    if not ok:
        audit.blockers.append(
            f"tagged-source exposure is not matched across the M5 pair: {M5_BASELINE}="
            f"{base_hits} against {M5_TREATMENT}={treat_hits}. A reduction measured over "
            "unequal exposure prices exposure, not forwarding"
        )

    base_inputs = int(base.get("node_input_calls") or 0)
    treat_inputs = int(treat.get("node_input_calls") or 0)
    audit.observed[f"{M5_BASELINE}.node_input_calls"] = base_inputs
    audit.observed[f"{M5_TREATMENT}.node_input_calls"] = treat_inputs
    ok = base_inputs == treat_inputs
    audit.conditions["node_input_volume_matched"] = ok
    if not ok:
        audit.blockers.append(
            f"the M5 arms executed different numbers of node inputs ({base_inputs} against "
            f"{treat_inputs}), so they did not run the same graph"
        )

    # ---- only the treatment turned that exposure into enforcement ------------------
    base_inherited = _inherited_only_total(base)
    treat_inherited = _inherited_only_total(treat)
    audit.observed[f"{M5_BASELINE}.inherited_only_enforcements"] = base_inherited
    audit.observed[f"{M5_TREATMENT}.inherited_only_enforcements"] = treat_inherited

    if base_inherited is None or treat_inherited is None:
        audit.conditions["attribution_recorded"] = False
        audit.blockers.append(
            "causal_attribution is absent from one or both M5 arms, so no enforcement can "
            "be attributed to inheritance rather than to detection"
        )
        return audit
    audit.conditions["attribution_recorded"] = True

    ok = base_inherited == 0
    audit.conditions["baseline_produced_no_inherited_enforcement"] = ok
    if not ok:
        audit.blockers.append(
            f"{M5_BASELINE} produced {base_inherited} inherited-only enforcements, but an "
            "arm that forwards nothing has no downstream scope to enforce on. The baseline "
            "is not clean"
        )

    ok = treat_inherited > 0
    audit.conditions["treatment_produced_inherited_enforcement"] = ok
    if not ok:
        audit.blockers.append(
            f"{M5_TREATMENT} produced {treat_inherited} inherited-only enforcements: the "
            "forwarding pathway never reached a protected boundary, so a null M5 would be "
            "a wiring fact rather than a finding about propagation"
        )

    # ---- forwarding reached the DERIVATIVE surfaces, not only the sink -------------
    #
    # The final boundary alone is not enough. A sink that reads the tagged note directly
    # would be caught there whether or not anything was forwarded along an edge; the edge
    # and write surfaces are the ones that can only fire on a DERIVATIVE that inherited a
    # scope, and they are the laundering path the write guard exists to close.
    for surface in ("edge", "write"):
        count = _enforcements(treat, surface)
        audit.observed[f"{M5_TREATMENT}.inherited_only_{surface}_enforcements"] = count
        ok = count > 0
        audit.conditions[f"forwarding_reached_{surface}_surface"] = ok
        if not ok:
            audit.blockers.append(
                f"{M5_TREATMENT} produced no inherited-only enforcement at the {surface} "
                "surface, so forwarding never acted on a derivative in transit"
            )

    audit.observed[f"{M5_TREATMENT}.inherited_only_final_enforcements"] = _enforcements(
        treat, "final"
    )

    # ---- M8's vacuity is a prediction; record whether it held ----------------------
    #
    # Reported, never blocking. M8 being vacuous is the DESIGNED state, and a run that did
    # not carry those arms is not a failing run.
    m8_stats = [stats_by_arm.get(arm) for arm in M8_ARMS]
    if all(m8_stats):
        vacuous = True
        for arm, m8 in zip(M8_ARMS, m8_stats, strict=True):
            hits = int((m8 or {}).get("memory_borne_scope_hits") or 0)
            withheld = int((m8 or {}).get("retrieval_withheld") or 0)
            audit.observed[f"{arm}.memory_borne_scope_hits"] = hits
            audit.observed[f"{arm}.retrieval_withheld"] = withheld
            if hits != 0 or withheld <= 0:
                vacuous = False
        audit.m8_vacuous_as_designed = vacuous
        audit.conditions["m8_pathway_vacuous_as_designed"] = vacuous

    return audit
