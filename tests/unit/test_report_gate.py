"""`make-report` must apply the gate, not print a table and let a human squint at it.

Before, the report emitted one row per condition and stopped: no pairing, no
`condition_delta_gate`, no laundering criterion, no controls check, and exit code 0
whatever the numbers said. "Did the experiment pass?" was therefore answered after
seeing the results, which is what pre-registration exists to prevent.
"""

from __future__ import annotations

import numpy as np

from rdl.cli.make_report import GATE_PAIRINGS, evaluate_gates, gate_table


def _report(
    condition: str,
    *,
    recall: float,
    laundering: float = 0.9,
    n_items: int = 400,
    real: bool = True,
    controls: bool = True,
    survives: bool = True,
    fp_floor_ok: bool = True,
    seed_spread: float = 0.02,
    run_id: str = "20260807T000000Z-abc-0",
) -> dict:
    rng = np.random.default_rng(abs(hash(condition)) % 2**32)
    per_item = (rng.random(n_items) < recall).astype(float).tolist()
    item_ids = [f"forget10-{i:04d}" for i in range(n_items)]
    seeds = [
        {
            "recall_at_k": {
                "persistent_store_after_episode": recall + (s - 2) * seed_spread,
                "final_answer": 0.0,
                "any_agent_message": 0.0,
                "any_memory_write": 0.0,
            },
            "laundering": {"laundering_rate": laundering, "n_recovered": int(recall * n_items)},
            "delegation_rate": 0.5,
        }
        for s in range(5)
    ]
    return {
        "run_id": run_id,
        "phase": "phase0_days3-5",
        "condition": condition,
        "n_seeds": 5,
        "n_items": n_items,
        "controls_enabled": controls,
        "data_provenance": {
            "source": "locuslab/TOFU" if real else "fixture",
            "n_items": n_items,
            "is_real_data": real,
        },
        "recall_at_k": {"persistent_store_after_episode": {"mean": recall}},
        "laundering_rate": {"mean": laundering},
        "item_ids": item_ids,
        "clusters": [f"author-{i // 20:04d}" for i in range(n_items)],
        "cluster_by": "author",
        "per_item_recall": per_item,
        "per_seed": seeds,
        "control_reports": [
            {
                "verdicts": {
                    "survives_always_delegate": survives,
                    "false_positive_floor_ok": fp_floor_ok,
                    "delegation_gap_ok": True,
                }
            }
        ],
    }


def _primary(verdict: dict) -> dict:
    return next(g for g in verdict["gates"] if g["primary"])


def test_the_primary_pairing_is_c3d_minus_c1w():
    primary = [(t, b) for t, b, is_primary, _ in GATE_PAIRINGS if is_primary]
    assert primary == [("C3D", "C1W")], (
        "the estimand is multi-agent write-back minus SINGLE-agent write-back, not "
        "minus a condition where writing is disabled"
    )


def test_c3_minus_c1_is_still_reported_but_is_not_primary():
    legacy = [(t, b, p) for t, b, p, _ in GATE_PAIRINGS if (t, b) == ("C3", "C1")]
    assert legacy, "continuity with the frozen pre-registration is kept"
    assert legacy[0][2] is False


def test_a_real_effect_passes():
    runs = [_report("C3D", recall=0.60), _report("C1W", recall=0.15)]
    verdict = evaluate_gates(runs)
    assert _primary(verdict)["passed"], _primary(verdict)["paired"]["reason"]
    assert verdict["overall_passed"]


def test_a_small_effect_fails():
    runs = [_report("C3D", recall=0.22), _report("C1W", recall=0.20)]
    verdict = evaluate_gates(runs)
    assert not verdict["overall_passed"]
    assert "< required" in _primary(verdict)["paired"]["reason"]


def test_missing_conditions_are_not_a_pass():
    verdict = evaluate_gates([_report("C3D", recall=0.60)])
    assert _primary(verdict)["status"] == "NOT RUN"
    assert "C1W" in _primary(verdict)["missing"]
    assert not verdict["overall_passed"]


def test_low_laundering_rate_fails_even_with_a_large_delta():
    """Recovery through nodes that DO have a path to blocked content would mean the
    invariants are catching the leak — the opposite of the paper's claim."""
    runs = [_report("C3D", recall=0.60, laundering=0.2), _report("C1W", recall=0.10)]
    verdict = evaluate_gates(runs)
    assert not _primary(verdict)["laundering_ok"]
    assert not verdict["overall_passed"]


def test_fixture_data_is_a_blocker():
    runs = [_report("C3D", recall=0.60, real=False, n_items=8), _report("C1W", recall=0.10)]
    verdict = evaluate_gates(runs)
    assert any("not TOFU" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_no_controls_is_a_blocker():
    runs = [_report("C3D", recall=0.60, controls=False), _report("C1W", recall=0.10)]
    verdict = evaluate_gates(runs)
    assert any("--no-controls" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_effect_that_dies_under_always_delegate_is_a_blocker():
    """If it only exists under abstention routing, it tracks agent A's utility collapse
    (model_utility 0.60 -> 0.46), not forgetting."""
    runs = [_report("C3D", recall=0.60, survives=False), _report("C1W", recall=0.10)]
    verdict = evaluate_gates(runs)
    assert any("always_delegate" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_high_false_positive_floor_is_a_blocker():
    runs = [_report("C3D", recall=0.60, fp_floor_ok=False), _report("C1W", recall=0.10)]
    verdict = evaluate_gates(runs)
    assert any("false-positive floor" in b for b in verdict["blockers"])


def test_items_are_paired_by_id_not_by_position():
    """Episode order is permuted per seed, so a positional zip pairs unrelated
    questions and produces a delta that means nothing."""
    t = _report("C3D", recall=0.60)
    b = _report("C1W", recall=0.10)
    b["item_ids"] = list(reversed(b["item_ids"]))
    b["per_item_recall"] = list(reversed(b["per_item_recall"]))

    verdict = evaluate_gates([t, b])
    assert _primary(verdict)["paired"]["n_pairs"] == len(t["item_ids"])


def test_gate_table_renders_every_pairing():
    verdict = evaluate_gates([_report("C3D", recall=0.60), _report("C1W", recall=0.10)])
    table = gate_table(verdict)
    for treatment, baseline, _, _ in GATE_PAIRINGS:
        assert f"{treatment} - {baseline}" in table
    assert "**(primary)**" in table
