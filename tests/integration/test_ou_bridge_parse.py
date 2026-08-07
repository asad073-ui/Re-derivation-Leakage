"""The open-unlearning bridge, exercised against a checked-in SUMMARY.json fixture.

This is the cheapest high-value test in the repo: it catches upstream schema drift
**without a GPU, without a download, and without a model**. When open-unlearning renames
a metric key, this goes red on a laptop instead of forty minutes into a Colab session.

No network required despite living in tests/integration — it is grouped here because it
is about the third-party boundary, not because it needs the internet.
"""

from __future__ import annotations

import json

import pytest

from rdl.eval.openunlearning_bridge import (
    DEFAULT_TOLERANCES,
    PUBLISHED_TARGETS,
    EvalSpec,
    build_eval_command,
    command_string,
    compare_to_published,
    find_summary,
    parse_summary,
)
from rdl.hardware import HardwareProfile


def _t4() -> HardwareProfile:
    """A Tesla T4, constructed rather than detected, so the test runs anywhere."""
    return HardwareProfile(
        device="cuda",
        name="Tesla T4",
        compute_capability=(7, 5),
        supports_bf16=False,
        supports_flash_attn2=False,
        total_vram_gb=16.0,
        recommended_dtype="float16",
        recommended_train_dtype="float32",
        recommended_attn="sdpa",
        torch_version="2.4.0",
        cuda_version="12.1",
        python_version="3.11.0",
        platform="Linux",
        free_disk_gb=100.0,
        cpu_count=8,
    )


def _h100() -> HardwareProfile:
    return HardwareProfile(
        device="cuda",
        name="NVIDIA H100",
        compute_capability=(9, 0),
        supports_bf16=True,
        supports_flash_attn2=True,
        total_vram_gb=80.0,
        recommended_dtype="bfloat16",
        recommended_train_dtype="bfloat16",
        recommended_attn="flash_attention_2",
        torch_version="2.4.0",
        cuda_version="12.1",
        python_version="3.11.0",
        platform="Linux",
        free_disk_gb=1000.0,
        cpu_count=64,
    )


# ------------------------------------------------------------------- parsing --


def test_parses_the_published_metrics(ou_summary_path):
    metrics = parse_summary(ou_summary_path)
    assert metrics["model_utility"] == pytest.approx(0.46)
    assert metrics["forget_truth_ratio"] == pytest.approx(0.70)
    assert metrics["forget_quality"] == pytest.approx(0.02)


def test_unwraps_the_nested_agg_value_form(ou_summary_path):
    """Upstream nests some metrics under `agg_value`; a bare float must work too."""
    raw = json.loads(ou_summary_path.read_text(encoding="utf-8"))
    assert isinstance(raw["forget_truth_ratio"], dict), "fixture must exercise the nested form"
    assert isinstance(raw["model_utility"], float), "...and the flat form"

    metrics = parse_summary(ou_summary_path)
    assert metrics["forget_truth_ratio"] == pytest.approx(0.70)


def test_unaliased_metrics_are_passed_through_not_dropped(ou_summary_path):
    """A new upstream metric must become visible, not vanish."""
    metrics = parse_summary(ou_summary_path)
    assert "retain_Q_A_ROUGE" in metrics
    assert metrics["retain_Q_A_ROUGE"] == pytest.approx(0.8912)


def test_key_aliases_absorb_a_rename(tmp_path):
    renamed = tmp_path / "SUMMARY.json"
    renamed.write_text(
        json.dumps({"tofu_model_utility": 0.46, "forget_Truth_Ratio": {"agg_value": 0.70}})
    )
    metrics = parse_summary(renamed)
    assert metrics["model_utility"] == pytest.approx(0.46)
    assert metrics["forget_truth_ratio"] == pytest.approx(0.70)


def test_rejects_a_non_object_summary(tmp_path):
    bad = tmp_path / "SUMMARY.json"
    bad.write_text("[1, 2, 3]")
    with pytest.raises(ValueError, match="JSON object"):
        parse_summary(bad)


def test_find_summary(tmp_path):
    assert find_summary(tmp_path) is None
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "TOFU_SUMMARY.json").write_text("{}")
    assert find_summary(tmp_path).name == "TOFU_SUMMARY.json"


# --------------------------------------------------------------------- gate --


def test_gate_passes_on_the_published_numbers(ou_summary_path):
    report = compare_to_published(parse_summary(ou_summary_path), "npo_forget10")
    assert report.passed
    gated = {c.metric for c in report.comparisons if c.gated}
    assert gated == set(DEFAULT_TOLERANCES)


def test_forget_quality_is_reported_but_never_gated(ou_summary_path):
    """It is a KS p-value spanning ~200 orders of magnitude. Gating it is meaningless."""
    report = compare_to_published(parse_summary(ou_summary_path), "npo_forget10")
    fq = next(c for c in report.comparisons if c.metric == "forget_quality")

    assert not fq.gated
    assert fq.ours == pytest.approx(0.02)
    assert "log10" in fq.note

    # Even a wildly wrong forget_quality must not flip the gate.
    metrics = parse_summary(ou_summary_path)
    metrics["forget_quality"] = 1.06e-239
    assert compare_to_published(metrics, "npo_forget10").passed


def test_gate_fails_outside_tolerance(ou_summary_path):
    metrics = parse_summary(ou_summary_path)
    metrics["model_utility"] = 0.50  # 0.04 off, tolerance is 0.01
    report = compare_to_published(metrics, "npo_forget10")

    assert not report.passed
    mu = next(c for c in report.comparisons if c.metric == "model_utility")
    assert not mu.passed and "0.0400" in mu.note


def test_gate_tolerance_boundary(ou_summary_path):
    metrics = parse_summary(ou_summary_path)
    metrics["model_utility"] = 0.46 + 0.01
    assert compare_to_published(metrics, "npo_forget10").passed
    metrics["model_utility"] = 0.46 + 0.0101
    assert not compare_to_published(metrics, "npo_forget10").passed


def test_missing_key_fails_loudly():
    report = compare_to_published({"forget_truth_ratio": 0.70}, "npo_forget10")
    assert not report.passed
    mu = next(c for c in report.comparisons if c.metric == "model_utility")
    assert mu.note == "MISSING KEY"


def test_unknown_target_raises():
    with pytest.raises(KeyError, match="unknown published target"):
        compare_to_published({}, "not_a_real_checkpoint")


def test_published_targets_match_the_spec():
    """Transcribed from open-unlearning/docs/repro.md. Do not edit without a citation."""
    assert PUBLISHED_TARGETS["full"]["model_utility"] == 0.60
    assert PUBLISHED_TARGETS["full"]["forget_truth_ratio"] == 0.48
    assert PUBLISHED_TARGETS["retain90"]["model_utility"] == 0.59
    assert PUBLISHED_TARGETS["npo_forget10"]["model_utility"] == 0.46
    assert PUBLISHED_TARGETS["npo_forget10"]["forget_truth_ratio"] == 0.70


# ------------------------------------------------------------ command building --


def test_t4_command_forces_fp16_and_sdpa():
    """The two overrides a T4 session must never be without."""
    cmd = command_string(build_eval_command(EvalSpec(model_path="org/ckpt", task_name="t"), _t4()))
    assert "model.model_args.torch_dtype=float16" in cmd
    assert "model.model_args.attn_implementation=sdpa" in cmd
    assert "bfloat16" not in cmd
    assert "flash_attention_2" not in cmd


def test_ampere_command_uses_bf16_and_flash_attention():
    cmd = command_string(
        build_eval_command(EvalSpec(model_path="org/ckpt", task_name="t"), _h100())
    )
    assert "model.model_args.torch_dtype=bfloat16" in cmd
    assert "model.model_args.attn_implementation=flash_attention_2" in cmd


def test_command_carries_the_retain_logs_path():
    """forget_quality cannot be computed without the retain model's eval log."""
    cmd = command_string(build_eval_command(EvalSpec(model_path="org/ckpt", task_name="t"), _t4()))
    assert "retain_logs_path=" in cmd
    assert "tofu_Llama-3.2-1B-Instruct_retain90" in cmd


def test_command_is_pure_and_runs_nothing():
    spec = EvalSpec(model_path="org/ckpt", task_name="t")
    assert build_eval_command(spec, _t4()) == build_eval_command(spec, _t4())


def test_dry_run_never_touches_the_filesystem():
    from rdl.eval.openunlearning_bridge import run_eval

    result = run_eval(EvalSpec(model_path="org/ckpt", task_name="t"), _t4(), dry_run=True)
    assert result["dry_run"] is True
    assert result["summary_path"] is None
    assert "src/eval.py" in result["command"]
