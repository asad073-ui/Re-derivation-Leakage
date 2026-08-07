"""`make-report` must apply the gate, not print a table and let a human squint at it.

Before, the report emitted one row per condition and stopped: no pairing, no
`condition_delta_gate`, no laundering criterion, no controls check, and exit code 0
whatever the numbers said. "Did the experiment pass?" was therefore answered after
seeing the results, which is what pre-registration exists to prevent.
"""

from __future__ import annotations

import numpy as np

from rdl.cli.make_report import (
    GATE_PAIRINGS,
    REQUIRED_REPRO_TARGETS,
    evaluate_gates,
    gate_table,
)
from rdl.eval.controls import FAIL, NOT_APPLICABLE, PASS

AGENT_A = (
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO"
    "_lr1e-05_beta0.1_alpha1_epoch10"
)
AGENT_A_REV = "94ed64eb73bc1872d52064833aaef364f4895c9c"
AGENT_B = (
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO"
    "_lr2e-05_beta0.5_alpha1_epoch10"
)
AGENT_B_REV = "eabf32c4883a5647c784c60c998b4b96cd48b798"
FULL = "open-unlearning/tofu_Llama-3.2-1B-Instruct_full"
FULL_REV = "88e31200b97e4c0c04ae0d2f0b591f427046d192"


def _repro(
    target: str,
    checkpoint: str,
    revision: str,
    *,
    passed: bool = True,
    batch_size: int = 32,
    seed: int = 0,
) -> dict:
    """A Days 1-2 report. Defaults are PUBLISHED PARITY — upstream's own eval settings."""
    return {
        "run_id": f"20260807T000000Z-{target}-b{batch_size}s{seed}",
        "phase": "phase0_days1-2_repro",
        "target": target,
        "checkpoint": checkpoint,
        "revision": revision,
        "passed": passed,
        "batch_size": batch_size,
        "seed": seed,
        "published_parity": batch_size == 32 and seed == 0,
        "comparisons": [],
    }


def _measure(checkpoint: str, revision: str, label: str = "agent_b_independent") -> dict:
    return {
        "run_id": f"20260807T000000Z-{label}-0",
        "phase": "phase0_days1-2_measure",
        "measure_only": True,
        "checkpoint": checkpoint,
        "revision": revision,
        "checkpoint_label": label,
        "metrics": {"model_utility": 0.44, "forget_truth_ratio": 0.66},
    }


def _day1(**kw) -> list[dict]:
    """The Days 1-2 prerequisites, all satisfied. `make-report` requires every one."""
    return [
        _repro("full", FULL, FULL_REV, **kw),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV, **kw),
        _measure(AGENT_B, AGENT_B_REV),
    ]


def _report(
    condition: str,
    *,
    recall: float,
    laundering: float = 0.9,
    n_items: int = 400,
    real: bool = True,
    controls: bool = True,
    survives: str = PASS,
    fp_floor_ok: str = PASS,
    seed_spread: float = 0.02,
    models: dict | None = None,
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
    if models is None:
        models = {
            "tofu_llama32_1b_npo_forget10": {
                "kind": "hf",
                "repo_id": AGENT_A,
                "revision": AGENT_A_REV,
            },
            "tofu_llama32_1b_npo_forget10_indep": {
                "kind": "hf",
                "repo_id": AGENT_B,
                "revision": AGENT_B_REV,
            },
        }
    return {
        "run_id": run_id,
        "phase": "phase0_days3-5",
        "condition": condition,
        "n_seeds": 5,
        "n_items": n_items,
        "controls_enabled": controls,
        "config": {
            "models": models,
            "data": {
                "dataset": "tofu",
                "forget_split": "forget10",
                "retain_split": "retain90",
                "cluster_by": "author",
            },
            "writepolicy": {"mode": "framework_default"},
        },
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
    runs = [_report("C3D", recall=0.60), _report("C1W", recall=0.15), *_day1()]
    verdict = evaluate_gates(runs)
    assert _primary(verdict)["passed"], _primary(verdict)["paired"]["reason"]
    assert verdict["overall_passed"], verdict["blockers"]


def test_a_small_effect_fails():
    runs = [_report("C3D", recall=0.22), _report("C1W", recall=0.20), *_day1()]
    verdict = evaluate_gates(runs)
    assert not verdict["overall_passed"]
    assert "< required" in _primary(verdict)["paired"]["reason"]


def test_missing_conditions_are_not_a_pass():
    verdict = evaluate_gates([_report("C3D", recall=0.60), *_day1()])
    assert _primary(verdict)["status"] == "NOT RUN"
    assert "C1W" in _primary(verdict)["missing"]
    assert not verdict["overall_passed"]


def test_low_laundering_rate_fails_even_with_a_large_delta():
    """Recovery through nodes that DO have a path to blocked content would mean the
    invariants are catching the leak — the opposite of the paper's claim."""
    runs = [
        _report("C3D", recall=0.60, laundering=0.2),
        _report("C1W", recall=0.10),
        *_day1(),
    ]
    verdict = evaluate_gates(runs)
    assert not _primary(verdict)["laundering_ok"]
    assert not verdict["overall_passed"]


def test_fixture_data_is_a_blocker():
    runs = [
        _report("C3D", recall=0.60, real=False, n_items=8),
        _report("C1W", recall=0.10),
        *_day1(),
    ]
    verdict = evaluate_gates(runs)
    assert any("not TOFU" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_no_controls_is_a_blocker():
    runs = [_report("C3D", recall=0.60, controls=False), _report("C1W", recall=0.10), *_day1()]
    verdict = evaluate_gates(runs)
    assert any("--no-controls" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_effect_that_dies_under_always_delegate_is_a_blocker():
    """If it only exists under abstention routing, it tracks agent A's utility collapse
    (model_utility 0.60 -> 0.46), not forgetting."""
    runs = [_report("C3D", recall=0.60, survives=FAIL), _report("C1W", recall=0.10), *_day1()]
    verdict = evaluate_gates(runs)
    assert any("always_delegate" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_single_agent_baseline_does_not_block_the_grid():
    """THE bug this pairing exists to avoid.

    C1W has no agent B, so `always_delegate` cannot be applied to it and the control
    reports NOT_APPLICABLE. Read as a boolean False, that made the baseline the primary
    estimand is measured against block every run — a grid that could never pass no matter
    what the numbers were.
    """
    runs = [
        _report("C3D", recall=0.60),
        _report("C1W", recall=0.15, survives=NOT_APPLICABLE, fp_floor_ok=NOT_APPLICABLE),
        *_day1(),
    ]
    verdict = evaluate_gates(runs)
    assert not any("always_delegate" in b for b in verdict["blockers"]), verdict["blockers"]
    assert verdict["overall_passed"], verdict["blockers"]


def test_high_false_positive_floor_is_a_blocker():
    runs = [_report("C3D", recall=0.60, fp_floor_ok=FAIL), _report("C1W", recall=0.10), *_day1()]
    verdict = evaluate_gates(runs)
    assert any("false-positive floor" in b for b in verdict["blockers"])


# =====================================================================================
# Days 1-2 are a PREREQUISITE, not a companion table
# =====================================================================================


def test_a_grid_without_any_reproduction_is_blocked():
    runs = [_report("C3D", recall=0.60), _report("C1W", recall=0.15)]
    verdict = evaluate_gates(runs)
    for target in REQUIRED_REPRO_TARGETS:
        assert any(f"--target {target}" in b for b in verdict["blockers"]), target
    assert not verdict["overall_passed"]


def test_a_failed_reproduction_blocks_the_grid():
    runs = [
        _report("C3D", recall=0.60),
        _report("C1W", recall=0.15),
        _repro("full", FULL, FULL_REV, passed=False),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV),
        _measure(AGENT_B, AGENT_B_REV),
    ]
    verdict = evaluate_gates(runs)
    assert any("did NOT pass" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_a_pass_only_at_batch_one_does_not_clear_the_trust_gate():
    """The published 0.46 / 0.70 were produced at upstream's batch 32 / seed 0.

    A pass at batch 1 / seed 42 is a fine second data point, but it cannot vouch for the
    install: if it had MISSED, the miss would have been ambiguous between a broken
    install and a batching difference — so its pass is equally ambiguous.
    """
    runs = [
        _report("C3D", recall=0.60),
        _report("C1W", recall=0.15),
        _repro("full", FULL, FULL_REV, batch_size=1, seed=42),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV, batch_size=1, seed=42),
        _measure(AGENT_B, AGENT_B_REV),
    ]
    verdict = evaluate_gates(runs)
    assert any("never at published parity" in b for b in verdict["blockers"]), verdict["blockers"]
    assert not verdict["overall_passed"]


def test_parity_pass_plus_batch_one_pass_is_the_intended_state():
    """Both runs present: the install is validated AND the protocol is characterised."""
    runs = [
        _report("C3D", recall=0.60),
        _report("C1W", recall=0.15),
        *_day1(),
        _repro("full", FULL, FULL_REV, batch_size=1, seed=42),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV, batch_size=1, seed=42),
    ]
    verdict = evaluate_gates(runs)
    assert verdict["overall_passed"], verdict["blockers"]


def test_a_legacy_report_without_the_parity_field_is_not_assumed_to_be_parity():
    """Reports written before the field existed carry no claim about their settings, and
    an absent claim is not a passing one."""
    legacy = _repro("full", FULL, FULL_REV)
    del legacy["published_parity"]
    runs = [
        _report("C3D", recall=0.60),
        _report("C1W", recall=0.15),
        legacy,
        _repro("npo_forget10", AGENT_A, AGENT_A_REV),
        _measure(AGENT_B, AGENT_B_REV),
    ]
    verdict = evaluate_gates(runs)
    assert any("never at published parity" in b for b in verdict["blockers"])


def test_agent_b_without_its_own_measurement_is_blocked():
    """B was unlearned at different hyperparameters and has no published row. Without a
    measure-only run, the C3D number is uninterpretable."""
    runs = [
        _report("C3D", recall=0.60),
        _report("C1W", recall=0.15),
        _repro("full", FULL, FULL_REV),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV),
    ]
    verdict = evaluate_gates(runs)
    assert any(AGENT_B in b and "never characterised" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_an_unpinned_checkpoint_is_blocked():
    unpinned = {
        "tofu_llama32_1b_npo_forget10": {"kind": "hf", "repo_id": AGENT_A, "revision": None},
    }
    runs = [
        _report("C3D", recall=0.60, models=unpinned),
        _report("C1W", recall=0.15, models=unpinned),
        *_day1(),
    ]
    verdict = evaluate_gates(runs)
    assert any("UNPINNED" in b for b in verdict["blockers"])


def test_a_grid_run_on_different_weights_than_the_reproduction_is_blocked():
    moved = {
        "tofu_llama32_1b_npo_forget10": {
            "kind": "hf",
            "repo_id": AGENT_A,
            "revision": "0" * 40,
        },
    }
    runs = [
        _report("C3D", recall=0.60, models=moved),
        _report("C1W", recall=0.15, models=moved),
        *_day1(),
    ]
    verdict = evaluate_gates(runs)
    assert any("does not vouch" in b for b in verdict["blockers"])


def test_conditions_measured_on_different_data_cannot_be_differenced():
    t = _report("C3D", recall=0.60)
    b = _report("C1W", recall=0.15)
    b["config"]["data"]["forget_split"] = "forget05"
    verdict = evaluate_gates([t, b, *_day1()])
    assert any("disagree on `forget_split`" in x for x in verdict["blockers"])


def test_conditions_run_at_different_seed_counts_are_blocked():
    t = _report("C3D", recall=0.60)
    b = _report("C1W", recall=0.15)
    b["n_seeds"] = 1
    verdict = evaluate_gates([t, b, *_day1()])
    assert any("disagree on `n_seeds`" in x for x in verdict["blockers"])


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
    verdict = evaluate_gates([_report("C3D", recall=0.60), _report("C1W", recall=0.10), *_day1()])
    table = gate_table(verdict)
    for treatment, baseline, _, _ in GATE_PAIRINGS:
        assert f"{treatment} - {baseline}" in table
    assert "**(primary)**" in table
