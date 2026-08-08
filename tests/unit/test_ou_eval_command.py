"""The exact open-unlearning command line.

Every assertion here corresponds to a way the command was wrong in a manner that does
not crash locally, only on the GPU box forty minutes in — or worse, does not crash at
all and produces a number under settings the report does not describe.

Verified against the pinned submodule `4ad738a`:
  configs/experiment/eval/tofu/default.yaml   forget_split / holdout_split / retain_logs_path
  configs/eval/tofu.yaml                      batch_size: 32, overwrite: false
  configs/eval.yaml                           seed: 0
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from rdl.eval.openunlearning_bridge import (
    TOFU_SPLITS,
    UPSTREAM_EVAL_BATCH_SIZE,
    UPSTREAM_EVAL_SEED,
    EvalSpec,
    build_eval_command,
    compare_to_published,
    find_summary,
    is_published_parity,
    parity_gaps,
)
from rdl.hardware import HardwareProfile


def _t4() -> HardwareProfile:
    return HardwareProfile(
        device="cuda",
        name="Tesla T4",
        compute_capability=(7, 5),
        supports_bf16=False,
        supports_flash_attn2=False,
        total_vram_gb=15.8,
        recommended_dtype="float16",
        recommended_train_dtype="float32",
        recommended_attn="sdpa",
        torch_version="2.4.1",
        cuda_version="12.1",
        python_version="3.11.0",
        platform="Linux",
        free_disk_gb=60.0,
        cpu_count=8,
    )


def _cmd(**kw) -> list[str]:
    spec = EvalSpec(model_path="org/ckpt", task_name="t", **kw)
    return build_eval_command(spec, _t4())


# =====================================================================================
# The override that aborts the run
# =====================================================================================


def test_never_emits_retain_split():
    """`retain_split` does not exist in the eval config tree.

    It is defined only under the *training* configs. Hydra refuses an override for an
    unknown key — "Could not override 'retain_split'" — so a command carrying it never
    reaches the model load. This was in every generated command and in the notebook.
    """
    assert not any(c.startswith("retain_split=") for c in _cmd())


def test_emits_holdout_split_derived_from_forget_split():
    for forget, (holdout, _) in TOFU_SPLITS.items():
        assert f"holdout_split={holdout}" in _cmd(forget_split=forget)


def test_holdout_and_retain_logs_stay_paired_with_the_forget_split():
    """forget10 with holdout05 does not crash — it silently scores against the wrong
    reference distribution, so the pairing is derived rather than passed twice."""
    spec = EvalSpec(model_path="m", task_name="t", forget_split="forget05")
    assert spec.holdout_split == "holdout05"
    assert "retain95" in (spec.retain_logs_path or "")


def test_unknown_forget_split_is_rejected():
    with pytest.raises(ValueError, match="unknown forget_split"):
        EvalSpec(model_path="m", task_name="t", forget_split="forget42")


# =====================================================================================
# Values the spec carried but never sent
# =====================================================================================


def test_seed_is_actually_passed():
    """Upstream defaults `seed: 0`. A spec with seed=42 that does not emit it produces
    a report header saying 42 and a run that used 0."""
    assert "seed=42" in _cmd(seed=42)
    assert "seed=7" in _cmd(seed=7)


def test_batch_size_is_actually_passed():
    """Upstream defaults `eval.tofu.batch_size: 32`. The pre-registration commits to 1
    for every number in the paper; unsent, the commitment was decorative."""
    assert f"eval.tofu.batch_size={1}" in _cmd(batch_size=1)
    assert "eval.tofu.batch_size=8" in _cmd(batch_size=8)


def test_overwrite_is_forced_on():
    """`overwrite: false` upstream means a rerun into the same output_dir skips metrics
    whose logs exist — a "new" run replaying old numbers."""
    assert "eval.tofu.overwrite=true" in _cmd()
    assert "eval.tofu.overwrite=false" in _cmd(overwrite=False)


def test_t4_overrides_are_unconditional():
    cmd = _cmd()
    assert "model.model_args.torch_dtype=float16" in cmd
    assert "model.model_args.attn_implementation=sdpa" in cmd
    assert "bfloat16" not in " ".join(cmd)
    assert "flash_attention_2" not in " ".join(cmd)


# =====================================================================================
# Ampere without the flash_attn wheel
# =====================================================================================


def _ampere(*, flash_attn_installed: bool) -> HardwareProfile:
    """An RTX 3090. FA2-capable silicon; the wheel may or may not be there."""
    return HardwareProfile(
        device="cuda",
        name="NVIDIA GeForce RTX 3090",
        compute_capability=(8, 6),
        supports_bf16=True,
        supports_flash_attn2=True,
        flash_attn_installed=flash_attn_installed,
        total_vram_gb=25.4,
        recommended_dtype="bfloat16",
        recommended_train_dtype="bfloat16",
        recommended_attn="flash_attention_2" if flash_attn_installed else "sdpa",
        torch_version="2.4.1",
        cuda_version="12.1",
        python_version="3.11.9",
        platform="Linux",
        free_disk_gb=100.0,
        cpu_count=16,
    )


def test_ampere_without_the_wheel_gets_sdpa_not_a_crash():
    """Upstream's configs/model/Llama-3.2-1B-Instruct.yaml pins
    `attn_implementation: flash_attention_2`. On a fresh 3090 image with no nvcc there is
    no wheel to import, so this override is the only thing between that default and an
    ImportError inside from_pretrained — after the checkpoint has downloaded."""
    cmd = build_eval_command(
        EvalSpec(model_path="org/ckpt", task_name="t"), _ampere(flash_attn_installed=False)
    )
    assert "model.model_args.attn_implementation=sdpa" in cmd
    assert "model.model_args.torch_dtype=bfloat16" in cmd


def test_ampere_with_the_wheel_uses_fa2():
    cmd = build_eval_command(
        EvalSpec(model_path="org/ckpt", task_name="t"), _ampere(flash_attn_installed=True)
    )
    assert "model.model_args.attn_implementation=flash_attention_2" in cmd


# =====================================================================================
# Revision pinning
# =====================================================================================


def test_revision_is_emitted_with_the_hydra_append_prefix():
    """`revision` is not a key in upstream's model_args, so a plain override aborts the
    run the same way `retain_split=` does. The `+` adds it; upstream splats model_args
    into `from_pretrained`, which takes `revision`."""
    cmd = _cmd()
    assert not any(c.startswith("+model.model_args.revision=") for c in cmd)

    spec = EvalSpec(model_path="org/ckpt", task_name="t", revision="a" * 40)
    pinned = build_eval_command(spec, _t4())
    assert f"+model.model_args.revision={'a' * 40}" in pinned
    assert "model.model_args.revision=" + "a" * 40 not in [
        c for c in pinned if not c.startswith("+")
    ]


# =====================================================================================
# Stale results
# =====================================================================================


def test_find_summary_ignores_files_older_than_the_run(tmp_path: Path):
    """A crashed run must not inherit the previous run's summary.

    `paths.output_dir` is reused across invocations with the same task_name, and the
    old glob took the last path in sort order regardless of age — so a failed eval
    still "found" a summary and the gate scored last week's number as today's.
    """
    d = tmp_path / "saves" / "eval" / "task"
    d.mkdir(parents=True)
    old = d / "TOFU_SUMMARY.json"
    old.write_text(json.dumps({"model_utility": 0.46}), encoding="utf-8")

    barrier = time.time() + 1.0
    assert find_summary(d) == old, "the file is there"
    assert find_summary(d, newer_than=barrier) is None, "but it predates this run"


def test_find_summary_returns_the_newest_by_mtime(tmp_path: Path):
    """Upstream filenames sort by metric, not by recency, so name order is not age."""
    d = tmp_path / "out"
    d.mkdir()
    first = d / "ZZZ_SUMMARY.json"
    first.write_text("{}", encoding="utf-8")
    time.sleep(0.01)
    second = d / "AAA_SUMMARY.json"
    second.write_text("{}", encoding="utf-8")
    import os

    os.utime(second, (time.time() + 5, time.time() + 5))
    assert find_summary(d) == second


# =====================================================================================
# Reporting honesty
# =====================================================================================


def test_batch_size_divergence_from_upstream_is_recorded():
    rep = compare_to_published(
        {"model_utility": 0.46, "forget_truth_ratio": 0.70}, "npo_forget10", batch_size=1
    )
    assert rep.passed
    assert rep.meta["upstream_eval_batch_size"] == UPSTREAM_EVAL_BATCH_SIZE
    assert "batch_size_note" in rep.meta, (
        "a batch-1 number compared against a batch-32 published reference must say so"
    )


def test_no_note_when_batching_matches_upstream():
    rep = compare_to_published(
        {"model_utility": 0.46, "forget_truth_ratio": 0.70},
        "npo_forget10",
        batch_size=UPSTREAM_EVAL_BATCH_SIZE,
        seed=UPSTREAM_EVAL_SEED,
    )
    assert "batch_size_note" not in rep.meta


# =====================================================================================
# Published parity
# =====================================================================================


def test_upstream_defaults_match_the_pinned_submodule():
    """Read off configs/eval/tofu.yaml (`batch_size: 32`) and configs/eval.yaml
    (`seed: 0`) at 4ad738a. If a SHA bump moves either, the parity gate is measuring
    against the wrong reference and this is the test that says so."""
    assert UPSTREAM_EVAL_BATCH_SIZE == 32
    assert UPSTREAM_EVAL_SEED == 0


def test_parity_requires_both_batch_and_seed():
    assert is_published_parity(batch_size=32, seed=0)
    assert not is_published_parity(batch_size=1, seed=0)
    assert not is_published_parity(batch_size=32, seed=42)


def test_parity_gaps_name_every_difference():
    gaps = parity_gaps(batch_size=1, seed=42, dtype="float16", attn="sdpa")
    joined = " ".join(gaps)
    assert "batch_size=1" in joined
    assert "seed=42" in joined
    assert "torch_dtype=float16" in joined
    assert "attn_implementation=sdpa" in joined


def test_a_parity_run_on_a_t4_still_reports_its_hardware_gaps():
    """A T4 cannot run bf16 or FA2 at all, so batch/seed parity is the most it can
    offer. The gap is reported rather than silently folded into a pass."""
    assert is_published_parity(batch_size=32, seed=0)
    gaps = parity_gaps(batch_size=32, seed=0, dtype="float16", attn="sdpa")
    assert len(gaps) == 2 and all("batch_size" not in g and "seed=" not in g for g in gaps)


def test_report_records_parity_and_the_reason_it_is_not():
    off = compare_to_published(
        {"model_utility": 0.46, "forget_truth_ratio": 0.70},
        "npo_forget10",
        batch_size=1,
        seed=42,
        dtype="bfloat16",
        attn="sdpa",
    )
    assert off.meta["published_parity"] is False
    assert any("seed=42" in g for g in off.meta["parity_gaps"])
    assert "Run the parity gate" in off.meta["batch_size_note"]

    on = compare_to_published(
        {"model_utility": 0.46, "forget_truth_ratio": 0.70},
        "npo_forget10",
        batch_size=32,
        seed=0,
        dtype="bfloat16",
        attn="flash_attention_2",
    )
    assert on.meta["published_parity"] is True
    assert on.meta["parity_gaps"] == []
