"""The dynamic half of the mechanism verdict (GU-0036).

Every test here is a way the M5 contrast can be configured correctly and still measure
nothing. Static arm purity passes in all of them.
"""

from __future__ import annotations

import pytest

from rdl.eval.mechanism_liveness import (
    M5_BASELINE,
    M5_TREATMENT,
    M8_ARMS,
    audit_mechanism_liveness,
)


def _arm(
    *,
    propagate: bool,
    hits: int = 2880,
    withheld: int = 0,
    inherited: int = 0,
    edge: int = 0,
    write: int = 0,
    final: int = 0,
    node_inputs: int = 3200,
    semantic: bool = False,
    consume: bool = True,
) -> dict:
    def cell(n: int) -> dict:
        return {"inherited_only": {"allow": 0, "sanitize": 0, "quarantine": 0, "refuse": n}}

    return {
        "semantic_detection": semantic,
        "consume_forget_ids": consume,
        "propagate_forget_ids": propagate,
        "memory_borne_scope_hits": hits,
        "retrieval_withheld": withheld,
        "node_input_calls": node_inputs,
        "causal_attribution": {
            "inherited_only_enforcements": inherited,
            "by_surface": {"edge": cell(edge), "write": cell(write), "final": cell(final)},
        },
    }


def _live_pair(**overrides) -> dict:
    base = {
        M5_BASELINE: _arm(propagate=False),
        M5_TREATMENT: _arm(propagate=True, inherited=6336, edge=2880, write=2880, final=576),
    }
    base.update(overrides)
    return base


def test_the_measured_run_passes() -> None:
    """The exact counters PR #36 committed."""
    audit = audit_mechanism_liveness(_live_pair())
    assert audit.applicable is True
    assert audit.live is True
    assert audit.blockers == []
    assert audit.conditions["tagged_source_exposure_matched"] is True
    assert audit.conditions["forwarding_reached_edge_surface"] is True
    assert audit.conditions["forwarding_reached_write_surface"] is True


def test_a_study_without_the_m5_pair_is_not_applicable_rather_than_failing() -> None:
    """A run that never asked the question has not failed to answer it."""
    audit = audit_mechanism_liveness({"multi_agent_leak": {}})
    assert audit.applicable is False
    assert audit.blockers == []
    assert audit.live is False


def test_a_withheld_source_is_the_vacuous_design_and_is_refused() -> None:
    """GU-0034's defect: quarantine the note and no derivative is ever produced."""
    audit = audit_mechanism_liveness(
        _live_pair(**{M5_TREATMENT: _arm(propagate=True, hits=0, withheld=2880, inherited=2880)})
    )
    assert audit.live is False
    assert any("memory_borne_scope_hits=0" in b for b in audit.blockers)
    assert any("withheld" in b for b in audit.blockers)


def test_unmatched_exposure_prices_exposure_rather_than_forwarding() -> None:
    audit = audit_mechanism_liveness(_live_pair(**{M5_BASELINE: _arm(propagate=False, hits=1440)}))
    assert audit.live is False
    assert audit.conditions["tagged_source_exposure_matched"] is False
    assert any("not matched" in b for b in audit.blockers)


def test_a_baseline_that_enforces_on_inheritance_is_not_clean() -> None:
    audit = audit_mechanism_liveness(
        _live_pair(**{M5_BASELINE: _arm(propagate=False, inherited=17)})
    )
    assert audit.live is False
    assert audit.conditions["baseline_produced_no_inherited_enforcement"] is False


def test_a_treatment_that_never_enforced_makes_a_null_result_a_wiring_fact() -> None:
    audit = audit_mechanism_liveness(
        _live_pair(**{M5_TREATMENT: _arm(propagate=True, inherited=0)})
    )
    assert audit.live is False
    assert audit.conditions["treatment_produced_inherited_enforcement"] is False


def test_forwarding_that_reached_only_the_sink_does_not_count() -> None:
    """The final boundary alone would fire even if nothing crossed a derivation edge.

    A sink that read the tagged note directly is caught at `final` whether or not any
    scope was forwarded. Only the edge and write surfaces can fire on a DERIVATIVE, and
    the write surface is the laundering path the guard exists to close.
    """
    audit = audit_mechanism_liveness(
        _live_pair(
            **{M5_TREATMENT: _arm(propagate=True, inherited=576, edge=0, write=0, final=576)}
        )
    )
    assert audit.live is False
    assert audit.conditions["forwarding_reached_edge_surface"] is False
    assert audit.conditions["forwarding_reached_write_surface"] is False


def test_a_detector_left_on_makes_attribution_ambiguous() -> None:
    audit = audit_mechanism_liveness(
        _live_pair(
            **{
                M5_TREATMENT: _arm(
                    propagate=True, inherited=6336, edge=2880, write=2880, semantic=True
                )
            }
        )
    )
    assert audit.live is False
    assert any("semantic detection ON" in b for b in audit.blockers)


def test_arms_that_do_not_differ_in_forwarding_are_refused() -> None:
    audit = audit_mechanism_liveness(_live_pair(**{M5_BASELINE: _arm(propagate=True, inherited=0)}))
    assert audit.live is False
    assert any("does not isolate forwarding" in b for b in audit.blockers)


def test_missing_counters_block_rather_than_pass_silently() -> None:
    """A run that planned the pair but wrote no counters must not audit as live."""
    audit = audit_mechanism_liveness({}, arms_present=[M5_BASELINE, M5_TREATMENT])
    assert audit.applicable is True
    assert audit.live is False
    assert audit.conditions["counters_present"] is False


def test_m8_vacuity_is_reported_and_never_blocking() -> None:
    """M8's arms MUST be vacuous. That is the prediction, not a defect."""
    defenses = _live_pair()
    for arm in M8_ARMS:
        defenses[arm] = _arm(propagate=False, hits=0, withheld=1713, inherited=1713)
    audit = audit_mechanism_liveness(defenses)
    assert audit.live is True
    assert audit.m8_vacuous_as_designed is True


def test_an_m8_arm_that_stopped_quarantining_is_flagged() -> None:
    defenses = _live_pair()
    for arm in M8_ARMS:
        defenses[arm] = _arm(propagate=False, hits=2880, withheld=0)
    audit = audit_mechanism_liveness(defenses)
    # Still not blocking M5 — but the artifact records that the prediction failed.
    assert audit.live is True
    assert audit.m8_vacuous_as_designed is False


@pytest.mark.parametrize("surface", ["edge", "write"])
def test_each_derivative_surface_is_required_independently(surface: str) -> None:
    counts = {"edge": 2880, "write": 2880}
    counts[surface] = 0
    audit = audit_mechanism_liveness(
        _live_pair(
            **{
                M5_TREATMENT: _arm(
                    propagate=True, inherited=6336, final=576, **counts  # type: ignore[arg-type]
                )
            }
        )
    )
    assert audit.live is False
    assert audit.conditions[f"forwarding_reached_{surface}_surface"] is False
