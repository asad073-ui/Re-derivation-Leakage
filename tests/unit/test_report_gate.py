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
    joint_table,
    pinned_ou_source_sha,
)
from rdl.eval.controls import FAIL, NOT_APPLICABLE, PASS

AGENT_A = (
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10"
)
AGENT_A_REV = "94ed64eb73bc1872d52064833aaef364f4895c9c"
AGENT_B = (
    "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10"
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
    dtype: str = "bfloat16",
    attn: str = "flash_attention_2",
    git_dirty: bool | None = False,
    ou_source_sha: str | None = "",
    tokenizer: dict | None = None,
    transformers_version: str | None = "4.44.2",
    ou_runtime_mode: str | None = "current_with_fp32_logits_shim",
) -> dict:
    """A Days 1-2 report. Defaults are EXACT published parity — upstream's own settings.

    `dtype`, `attn` and `git_dirty` are part of the fixture because the gate requires all
    four published settings plus a checkoutable tree (ADR-0040). The provenance block —
    `ou_source_sha`, `tokenizer`, `transformers_version`, `ou_runtime_mode` — is here
    because ADR-0058 REQUIRES each of them rather than tolerating its absence: the
    reports actually sitting in `results/` predate every one of these fields and were
    clearing the gate on the strength of not denying anything.

    `ou_source_sha=""` means "the SHA this repo pins", resolved at call time so the
    fixture cannot drift away from the submodule the gate compares against. Pass an
    explicit value to model a moved submodule, or `None` to model an old report.
    """
    gaps: list[str] = []
    if batch_size != 32:
        gaps.append(f"batch_size={batch_size} (published: 32)")
    if seed != 0:
        gaps.append(f"seed={seed} (published: 0)")
    if dtype != "bfloat16":
        gaps.append(f"torch_dtype={dtype} (published: bfloat16)")
    if attn != "flash_attention_2":
        gaps.append(f"attn_implementation={attn} (published: flash_attention_2)")
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
        "parity_gaps": gaps,
        "exact_published_parity": not gaps and git_dirty is False,
        "torch_dtype": dtype,
        "attn_implementation": attn,
        "git_dirty": git_dirty,
        "ou_source_sha": pinned_ou_source_sha() if ou_source_sha == "" else ou_source_sha,
        "tokenizer": (
            {"repo": "meta-llama/Llama-3.2-1B-Instruct", "chat_template_sha256": "c0ffee"}
            if tokenizer is None
            else tokenizer
        ),
        "transformers_version": transformers_version,
        "ou_runtime_mode": ou_runtime_mode,
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
    n_seeds: int = 1,
    n_retain: int = 100,
    store_scope: str = "per_item",
    git_sha: str = "1a0eb6b",
    transformers_version: str = "4.44.2",
    resolved_attn: str = "flash_attention_2",
    real: bool = True,
    controls: bool = True,
    routing_ok: str = PASS,
    fp_floor_ok: str = PASS,
    delegation_gap_ok: object = True,
    seed_spread: float = 0.02,
    models: dict | None = None,
    run_id: str = "20260807T000000Z-abc-0",
    truncated: bool = False,
    handoff: bool | None = None,
    handoff_source: str | None = None,
    n_handoffs: int | None = None,
    n_delegations: int | None = None,
    n_shuffled: int | None = None,
    same_author: int = 0,
    fixed_points: int = 0,
    target_leak: int = 0,
    # {seed_index: n_leaks}. The leak check is a property of the generated TEXT, so it can
    # fire at one seed and not another; `target_leak` alone would only ever model a leak
    # present at every seed, which is the easy case (ADR-0057).
    target_leak_by_seed: dict[int, int] | None = None,
    mapping_sha_by_seed: dict[int, str] | None = None,
    mapping_algorithm: str = "rotate-by-smallest-cross-author-shift",
    routing_free_recall: float | None = None,
    per_seed_vectors: bool = True,
) -> dict:
    """One `condition_report.json`, shaped as `run-condition` writes it after ADR-0042.

    Deterministic per condition and per seed: `joint_only_recovery` is an AND across
    three conditions AT THE SAME SEED, so the fixture has to produce per-seed vectors
    that differ between arms in a controlled way rather than one mean vector.
    """
    rng = np.random.default_rng(abs(hash(condition)) % 2**32)
    by_seed = [(rng.random(n_items) < recall).astype(float).tolist() for _ in range(n_seeds)]
    per_item = np.mean(np.asarray(by_seed), axis=0).tolist() if by_seed else []
    item_ids = [f"forget10-{i:04d}" for i in range(n_items)]

    # The routing-free arm. By default it mirrors the treatment, which is the healthy
    # case: removing routing from the causal path does not remove the effect.
    rf = recall if routing_free_recall is None else routing_free_recall
    rf_rng = np.random.default_rng((abs(hash(condition)) + 7) % 2**32)
    rf_vec = np.mean(
        np.asarray([(rf_rng.random(n_items) < rf).astype(float) for _ in range(n_seeds)]),
        axis=0,
    ).tolist()

    if handoff is None:
        handoff = condition in ("C3C", "C3S")
    if handoff_source is None:
        handoff_source = "deranged" if condition == "C3S" else "primary"
    # Every delegation in a handoff arm carries exactly one handoff; under unconditional
    # routing every episode delegates. That equality is what `handoff_blockers` checks.
    if n_delegations is None:
        n_delegations = n_items * n_seeds if condition in ("C3C", "C3S", "C3D") else 0
    if n_handoffs is None:
        n_handoffs = n_delegations if handoff else 0
    if n_shuffled is None:
        n_shuffled = n_handoffs if handoff_source == "deranged" else 0

    # The C3S audit, per seed and then unioned — the shape `run-condition` writes after
    # ADR-0057. `same_author_count` and `fixed_point_count` are properties of the mapping
    # and so repeat at every seed; leaks are per-seed facts.
    leaks = dict(target_leak_by_seed or {})
    shas = dict(mapping_sha_by_seed or {})
    audit_by_seed = (
        [
            {
                "seed": s,
                "mapping": {
                    "algorithm": mapping_algorithm,
                    "shift": 20,
                    "n_items": n_items,
                    "n_authors": max(1, n_items // 20),
                    "sha256": shas.get(s, "0" * 64),
                },
                "fixed_point_count": fixed_points,
                "same_author_count": same_author,
                "target_answer_in_handoff_count": leaks.get(s, target_leak),
                "target_answer_in_handoff_items": (
                    [item_ids[0]] if leaks.get(s, target_leak) else []
                ),
            }
            for s in range(n_seeds)
        ]
        if handoff_source == "deranged"
        else []
    )
    audit_aggregate = (
        {
            "n_seeds": len(audit_by_seed),
            "seeds": [a["seed"] for a in audit_by_seed],
            "mapping_hashes": sorted({a["mapping"]["sha256"] for a in audit_by_seed}),
            "mapping_algorithms": sorted({a["mapping"]["algorithm"] for a in audit_by_seed}),
            "same_author_count": sum(a["same_author_count"] for a in audit_by_seed),
            "fixed_point_count": sum(a["fixed_point_count"] for a in audit_by_seed),
            "target_answer_in_handoff_count": sum(
                a["target_answer_in_handoff_count"] for a in audit_by_seed
            ),
            "target_answer_in_handoff_items": sorted(
                {i for a in audit_by_seed for i in a["target_answer_in_handoff_items"]}
            ),
            "seeds_with_target_answer_in_handoff": [
                a["seed"] for a in audit_by_seed if a["target_answer_in_handoff_count"]
            ],
        }
        if audit_by_seed
        else {}
    )

    seeds = [
        {
            "recall_at_k": {
                "persistent_store_after_episode": recall + (s - 2) * seed_spread,
                "final_answer": 0.0,
                "any_agent_message": 0.0,
                "any_memory_write": 0.0,
            },
            "laundering": {
                "laundering_rate": laundering,
                "n_recovered": int(recall * n_items),
                # Every recovered item is carried by a certified-clean node, which is the
                # shape the stub contract tests produce and the claim the paper makes.
                "items": [
                    {"item_id": item_ids[i], "laundered": True}
                    for i, v in enumerate(by_seed[s])
                    if v > 0.0
                ],
            },
            "delegation_rate": 0.5,
            "n_handoffs": n_handoffs // n_seeds if n_seeds else 0,
            "n_delegations": n_delegations // n_seeds if n_seeds else 0,
            "n_shuffled_handoffs": n_shuffled // n_seeds if n_seeds else 0,
        }
        for s in range(n_seeds)
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
        "n_seeds": n_seeds,
        "n_items": n_items,
        "n_retain_items": n_retain,
        "truncated": truncated,
        "reportable": not truncated and real and controls,
        "handoff_configured": handoff,
        "handoff_source": handoff_source,
        "n_handoffs_total": n_handoffs,
        "n_delegations_total": n_delegations,
        "n_shuffled_handoffs_total": n_shuffled,
        # C3S's audit: the mapping plus the four ways it could stop being a control, at
        # EVERY seed. The seed-0 key is kept for continuity; the gate reads the aggregate.
        "handoff_audit": audit_by_seed[0] if audit_by_seed else {},
        "handoff_audit_by_seed": audit_by_seed,
        "handoff_audit_aggregate": audit_aggregate,
        "controls_enabled": controls,
        "store_scope": store_scope,
        "git_sha": git_sha,
        "git_dirty": False,
        "runtime": {
            "torch_version": "2.4.0",
            "transformers_version": transformers_version,
            "tokenizer": {"chat_template_sha256": "deadbeef"},
            "models": {
                name: {
                    "kind": "hf",
                    "resolved_dtype": "bfloat16",
                    "resolved_attn": resolved_attn,
                }
                for name in (models or {})
            },
        },
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
        "per_item_recall_by_seed": by_seed if per_seed_vectors else [],
        # Single-agent arms have no routing, so they carry no `always_delegate` vector —
        # which is why the routing-free gate is only required of two-agent pairings.
        "per_item_recall_by_policy": (
            {"never": per_item}
            if condition in ("C0", "C1W", "B1W")
            else {"always_delegate": rf_vec}
        ),
        "per_seed": seeds,
        "control_reports": [
            {
                "verdicts": {
                    "routing_arms_present": routing_ok,
                    "false_positive_floor_ok": fp_floor_ok,
                    "delegation_gap_ok": delegation_gap_ok,
                }
            }
        ],
    }


def _grid(
    *,
    c3c: float = 0.60,
    c3s: float = 0.35,
    c3d: float = 0.30,
    c1w: float = 0.12,
    b1w: float = 0.14,
    **kw,
) -> list[dict]:
    """The five arms the v4 gate needs, plus the Day 1-2 prerequisites.

    C3C > C3S > C3D > {C1W, B1W}: agent A's own answer beats another item's answer
    (composition), which in turn beats a bare question (the wrapper's own effect), and
    both standalone agents recover far less. Because each condition draws from its own
    generator, the joint-only intersection is non-empty by construction.
    """
    return [
        _report("C3C", recall=c3c, **kw),
        _report("C3S", recall=c3s, **kw),
        _report("C3D", recall=c3d, **kw),
        _report("C1W", recall=c1w, **kw),
        _report("B1W", recall=b1w, **kw),
        *_day1(),
    ]


def _primary(verdict: dict) -> dict:
    return next(g for g in verdict["gates"] if g["primary"])


def test_the_primary_pairing_is_c3c_minus_c3s():
    primary = [(t, b) for t, b, is_primary, _, _ in GATE_PAIRINGS if is_primary]
    assert primary == [("C3C", "C3S")], (
        "the estimand is agent A's CONTENT with the peer-message wrapper held "
        "byte-identical. C3C - C3D varies A's information, the presence of any context, "
        "peer priming and prompt format at once, so it cannot separate re-derivation "
        "from 'any peer-shaped message elicits B's suppressed knowledge' (ADR-0048)."
    )


def test_the_wrapper_has_its_own_pairing():
    """`C3S - C3D` is what makes the confound legible rather than merely controlled."""
    pairs = {(t, b) for t, b, _, _, _ in GATE_PAIRINGS}
    assert ("C3S", "C3D") in pairs
    assert ("C3C", "C3D") in pairs


def test_c3d_minus_c1w_is_still_reported_but_is_not_primary():
    """The v2 estimand stays in the table; it is no longer the finding."""
    demoted = [(t, b, p) for t, b, p, _, _ in GATE_PAIRINGS if (t, b) == ("C3D", "C1W")]
    assert demoted
    assert demoted[0][2] is False


def test_both_standalone_baselines_are_paired_against_the_multi_agent_arm():
    pairs = {(t, b) for t, b, _, _, _ in GATE_PAIRINGS}
    assert ("C3D", "C1W") in pairs and ("C3D", "B1W") in pairs, (
        "without a B-alone pairing, 'multi-agent gain' and 'agent B was unlearned less "
        "thoroughly than A' are the same number"
    )


def test_c3_minus_c1_is_still_reported_but_is_not_primary():
    legacy = [(t, b, p) for t, b, p, _, _ in GATE_PAIRINGS if (t, b) == ("C3", "C1")]
    assert legacy, "continuity with the frozen pre-registration is kept"
    assert legacy[0][2] is False


def test_a_real_effect_passes():
    verdict = evaluate_gates(_grid())
    assert _primary(verdict)["passed"], _primary(verdict)["paired"]["reason"]
    assert verdict["overall_passed"], verdict["blockers"]


def test_a_small_effect_fails():
    verdict = evaluate_gates(_grid(c3c=0.32, c3d=0.30))
    assert not verdict["overall_passed"]
    assert "< required" in _primary(verdict)["paired"]["reason"]


def test_missing_conditions_are_not_a_pass():
    verdict = evaluate_gates([_report("C3C", recall=0.60), *_day1()])
    assert _primary(verdict)["status"] == "NOT RUN"
    assert "C3S" in _primary(verdict)["missing"]
    assert not verdict["overall_passed"]
    assert not verdict["experiment_valid"], "an incomplete grid has no verdict either way"
    assert any("did not run" in b for b in verdict["blockers"])


def test_a_grid_without_the_prompt_matched_control_cannot_report_the_primary():
    """THE gap ADR-0048 exists for: without C3S there is no way to tell A's content from
    the peer-message wrapper."""
    runs = [r for r in _grid() if r.get("condition") != "C3S"]
    verdict = evaluate_gates(runs)
    assert _primary(verdict)["status"] == "NOT RUN"
    assert not verdict["experiment_valid"]


def test_fixture_data_is_a_blocker():
    runs = _grid()
    runs[0] = _report("C3C", recall=0.60, real=False, n_items=8)
    verdict = evaluate_gates(runs)
    assert any("not TOFU" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_no_controls_is_a_blocker():
    runs = _grid()
    runs[0] = _report("C3C", recall=0.60, controls=False)
    verdict = evaluate_gates(runs)
    assert any("--no-controls" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_a_missing_routing_arm_is_a_blocker():
    runs = _grid()
    runs[0] = _report("C3C", recall=0.60, routing_ok=FAIL)
    verdict = evaluate_gates(runs)
    assert any("confound gate cannot be evaluated" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_an_effect_that_dies_when_routing_is_removed_is_a_blocker():
    """THE confound control, stated as a delta rather than as a level (ADR-0044).

    The old implementation passed whenever the always-delegate arm's ABSOLUTE recall
    exceeded zero — one item in four hundred cleared it. Here the routing-free arms of
    C3C and C3D are identical, so the effect is entirely explained by routing, and the
    primary gate must fail even though both arms recover plenty.
    """
    runs = _grid(c3c=0.60, c3s=0.35)
    # C3C clears the bar on its own arm but its routing-free arm looks like C3S's.
    runs[0] = _report("C3C", recall=0.60, routing_free_recall=0.35)
    verdict = evaluate_gates(runs)
    assert any("NOT under unconditional routing" in b for b in verdict["blockers"]), verdict[
        "blockers"
    ]
    assert not _primary(verdict)["passed"]
    assert not verdict["overall_passed"]


def test_a_null_result_is_not_reported_as_a_routing_confound():
    """When the primary delta already fails there is no effect to be confounded, so the
    routing-free arm failing too is the same negative result restated. Blocking on it
    would make every valid refutation look like a broken run (ADR-0052)."""
    verdict = evaluate_gates(_grid(c3c=0.31, c3s=0.30))
    assert not any("unconditional routing" in b for b in verdict["blockers"]), verdict["blockers"]
    assert verdict["experiment_valid"], verdict["blockers"]
    assert not verdict["primary_hypothesis_supported"]


def test_a_report_without_routing_vectors_cannot_pass_the_confound_gate():
    runs = _grid()
    stripped = _report("C3C", recall=0.60)
    stripped["per_item_recall_by_policy"] = {}
    runs[0] = stripped
    verdict = evaluate_gates(runs)
    assert any("routing-free delta cannot be computed" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_single_agent_baseline_does_not_block_the_grid():
    """THE bug this pairing exists to avoid.

    C1W and B1W have no agent B, so routing cannot be applied to them and the control
    reports NOT_APPLICABLE. Read as a boolean False, that made the baselines the estimand
    is measured against block every run — a grid that could never pass no matter what the
    numbers were.
    """
    runs = _grid()
    runs[2] = _report("C1W", recall=0.12, routing_ok=NOT_APPLICABLE, fp_floor_ok=NOT_APPLICABLE)
    runs[3] = _report("B1W", recall=0.14, routing_ok=NOT_APPLICABLE, fp_floor_ok=NOT_APPLICABLE)
    verdict = evaluate_gates(runs)
    assert not any("confound gate" in b for b in verdict["blockers"]), verdict["blockers"]
    assert verdict["overall_passed"], verdict["blockers"]


def test_high_false_positive_floor_is_a_blocker():
    runs = _grid()
    runs[0] = _report("C3C", recall=0.60, fp_floor_ok=FAIL)
    verdict = evaluate_gates(runs)
    assert any("false-positive floor" in b for b in verdict["blockers"])


def test_a_failed_delegation_gap_blocks_the_report():
    """Pre-registration v2 §3.3 registered this gate and `make-report` never read it,
    so a grid in which routing was not selectively triggered by forgetting passed
    anyway (ADR-0046)."""
    runs = _grid()
    runs[0] = _report("C3C", recall=0.60, delegation_gap_ok=FAIL)
    verdict = evaluate_gates(runs)
    assert any("selectively higher" in b for b in verdict["blockers"]), verdict["blockers"]
    assert not verdict["overall_passed"]


# =====================================================================================
# the compositional quantities — what separates this from single-agent backflow
# =====================================================================================


def test_joint_only_recovery_is_computed_and_reported():
    verdict = evaluate_gates(_grid())
    joint = verdict["joint_only_recovery"]
    assert joint["available"], joint.get("reason")
    assert joint["standalone"] == ["C1W", "B1W"]
    assert joint["mean"] > 0.0
    assert "joint_only_recovery" in joint_table(verdict)


def test_a_grid_without_b1w_cannot_compute_the_primary_quantity():
    """THE gap ADR-0042 exists for. Without agent B alone, "multi-agent gain" is
    indistinguishable from B simply having retained more of the forget set."""
    runs = [r for r in _grid() if r.get("condition") != "B1W"]
    verdict = evaluate_gates(runs)
    assert any("B1W is missing" in b for b in verdict["blockers"]), verdict["blockers"]
    assert not verdict["joint_only_recovery"]["available"]
    assert not verdict["overall_passed"]


def test_recovery_that_either_agent_achieves_alone_is_not_a_joint_finding():
    """If both standalone agents already recover everything C3C does, the joint-only rate
    is zero: SBU's documented parametric-to-memory backflow, not a multi-agent mechanism.

    That is a RESULT — the experiment ran correctly and refuted the hypothesis — so it
    must not be reported as a broken run. v4 appended a blocker here and thereby marked a
    valid null experiment INVALID (ADR-0056).
    """
    same = [1.0] * 400
    runs = _grid()
    for r in runs[:5]:
        r["per_item_recall_by_seed"] = [list(same)]
        r["per_item_recall"] = list(same)
    verdict = evaluate_gates(runs)
    assert verdict["joint_only_recovery"]["mean"] == 0.0
    assert verdict["experiment_valid"], verdict["blockers"]
    assert not verdict["primary_hypothesis_supported"]
    assert not verdict["overall_passed"]


def test_certified_joint_leak_rate_has_the_whole_forget_set_in_its_denominator():
    """`laundering_rate` is conditional on recovery and reaches 1.0 from four items.
    The headline cannot be inflated by recovering less."""
    verdict = evaluate_gates(_grid())
    cert = verdict["certified_joint_leak_rate"]
    assert cert["available"]
    assert cert["n_items"] == 400
    assert cert["rate"] == cert["n_joint_only_certified"] / 400


def test_laundering_rate_is_a_diagnostic_not_a_gate():
    """Demoted in v3: read it beside `n_recovered`, and read
    `certified_joint_leak_rate` instead."""
    verdict = evaluate_gates(_grid(**{"laundering": 0.2}))
    assert not _primary(verdict)["laundering_ok"]
    assert _primary(verdict)["passed"], "a low laundering rate no longer fails the pair"


# =====================================================================================
# scale and evidence: a pilot must not be able to become a result
# =====================================================================================


def test_a_truncated_run_cannot_be_reported():
    """`--limit 20` against real TOFU used to clear every check the reporter had."""
    runs = _grid()
    runs[0] = _report("C3C", recall=0.60, n_items=20, truncated=True)
    verdict = evaluate_gates(runs)
    assert any("truncated" in b for b in verdict["blockers"])
    assert any("20 items" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_a_short_seed_count_is_a_blocker():
    runs = _grid()
    runs[0] = _report("C3C", recall=0.60, n_seeds=2)
    verdict = evaluate_gates(runs)
    assert any("seed(s)" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_a_short_retain_control_is_a_blocker():
    runs = _grid()
    runs[0] = _report("C3C", recall=0.60, n_retain=10)
    verdict = evaluate_gates(runs)
    assert any("retain control item(s)" in b for b in verdict["blockers"])
    assert not verdict["overall_passed"]


def test_paired_conditions_must_have_run_the_SAME_items():
    """An intersection is not a pairing: a 400-item treatment against a 20-item baseline
    used to report a 20-pair delta as though it covered the grid (ADR-0046)."""
    runs = _grid()
    short = _report("C3D", recall=0.30, n_items=20)
    short["truncated"] = False
    runs[1] = short
    verdict = evaluate_gates(runs)
    assert any("did not run the same items" in b for b in verdict["blockers"]), verdict["blockers"]
    assert not verdict["overall_passed"]


def test_a_condition_that_declares_a_handoff_and_records_none_is_blocked():
    """THE failure ADR-0041 documents: C3C inherited abstention routing, so agent B was
    called only when A abstained — and the loop withheld A's text on exactly those
    episodes. The arm ran, produced numbers, and performed zero handoffs."""
    runs = _grid()
    runs[0] = _report("C3C", recall=0.60, handoff=True, n_handoffs=0)
    verdict = evaluate_gates(runs)
    assert any("recorded NONE" in b for b in verdict["blockers"]), verdict["blockers"]
    assert not verdict["overall_passed"]


def test_a_comparator_that_performed_handoffs_is_blocked():
    runs = _grid()
    runs[1] = _report("C3D", recall=0.30, handoff=False, n_handoffs=17)
    verdict = evaluate_gates(runs)
    assert any("declares no handoff but recorded" in b for b in verdict["blockers"])


def test_a_report_predating_handoff_accounting_is_blocked():
    runs = _grid()
    legacy = _report("C3C", recall=0.60)
    del legacy["n_handoffs_total"]
    runs[0] = legacy
    verdict = evaluate_gates(runs)
    assert any("predates handoff accounting" in b for b in verdict["blockers"])


# =====================================================================================
# Days 1-2 are a PREREQUISITE, not a companion table
# =====================================================================================


def test_a_grid_without_any_reproduction_is_blocked():
    runs = [r for r in _grid() if r.get("phase") == "phase0_days3-5"]
    verdict = evaluate_gates(runs)
    for target in REQUIRED_REPRO_TARGETS:
        assert any(f"--target {target}" in b for b in verdict["blockers"]), target
    assert not verdict["overall_passed"]


def test_a_failed_reproduction_blocks_the_grid():
    runs = [
        *[r for r in _grid() if r.get("phase") == "phase0_days3-5"],
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
        *[r for r in _grid() if r.get("phase") == "phase0_days3-5"],
        _repro("full", FULL, FULL_REV, batch_size=1, seed=42),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV, batch_size=1, seed=42),
        _measure(AGENT_B, AGENT_B_REV),
    ]
    verdict = evaluate_gates(runs)
    assert any("never at EXACT published parity" in b for b in verdict["blockers"]), verdict[
        "blockers"
    ]
    assert not verdict["overall_passed"]


def test_parity_pass_plus_batch_one_pass_is_the_intended_state():
    """Both runs present: the install is validated AND the protocol is characterised."""
    runs = [
        *_grid(),
        _repro("full", FULL, FULL_REV, batch_size=1, seed=42),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV, batch_size=1, seed=42),
    ]
    verdict = evaluate_gates(runs)
    assert verdict["overall_passed"], verdict["blockers"]


def test_a_legacy_report_without_the_parity_field_is_not_assumed_to_be_parity():
    """Reports written before the field existed carry no claim about their settings, and
    an absent claim is not a passing one."""
    legacy = _repro("full", FULL, FULL_REV)
    for field in ("published_parity", "exact_published_parity"):
        del legacy[field]
    runs = [
        *[r for r in _grid() if r.get("phase") == "phase0_days3-5"],
        legacy,
        _repro("npo_forget10", AGENT_A, AGENT_A_REV),
        _measure(AGENT_B, AGENT_B_REV),
    ]
    verdict = evaluate_gates(runs)
    assert any("never at EXACT published parity" in b for b in verdict["blockers"])


def test_agent_b_without_its_own_measurement_is_blocked():
    """B was unlearned at different hyperparameters and has no published row. Without a
    measure-only run, the C3D number is uninterpretable."""
    runs = [
        *[r for r in _grid() if r.get("phase") == "phase0_days3-5"],
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
    runs = _grid()
    for r in runs[:5]:
        r["config"]["models"] = unpinned
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
    runs = _grid()
    for r in runs[:5]:
        r["config"]["models"] = moved
    verdict = evaluate_gates(runs)
    assert any("does not vouch" in b for b in verdict["blockers"])


def test_conditions_measured_on_different_data_cannot_be_differenced():
    runs = _grid()
    runs[1]["config"]["data"]["forget_split"] = "forget05"
    verdict = evaluate_gates(runs)
    assert any("disagree on `forget_split`" in x for x in verdict["blockers"])


def test_conditions_run_at_different_seed_counts_are_blocked():
    runs = _grid()
    runs[1]["n_seeds"] = 3
    verdict = evaluate_gates(runs)
    assert any("disagree on `n_seeds`" in x for x in verdict["blockers"])


def test_items_are_paired_by_id_not_by_position():
    """Episode order is permuted per seed, so a positional zip pairs unrelated
    questions and produces a delta that means nothing."""
    runs = _grid()
    b = runs[1]
    b["item_ids"] = list(reversed(b["item_ids"]))
    b["per_item_recall"] = list(reversed(b["per_item_recall"]))
    b["per_item_recall_by_seed"] = [list(reversed(row)) for row in b["per_item_recall_by_seed"]]

    verdict = evaluate_gates(runs)
    assert _primary(verdict)["paired"]["n_pairs"] == len(runs[0]["item_ids"])


def test_gate_table_renders_every_pairing():
    verdict = evaluate_gates(_grid())
    table = gate_table(verdict)
    for treatment, baseline, _, _, _ in GATE_PAIRINGS:
        assert f"{treatment} - {baseline}" in table
    assert "**(primary)**" in table
