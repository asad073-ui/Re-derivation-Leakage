"""The bridge to `third_party/open-unlearning`.

Design invariant 1, restated as code: **we never fork, vendor, or patch
open-unlearning.** This module builds their command line, runs it as a subprocess, and
parses the `*_SUMMARY.json` it produces. It does not import from `third_party` in any
way that would require patching it. Every metric number we report must be producible by
their code on their configs, so a reviewer can re-run it.

The two T4 overrides that must always be applied, and why:

  torch_dtype=float16          T4 (SM 7.5) has no bf16 datapath
  attn_implementation=sdpa     FlashAttention-2 is SM80+

`build_eval_command` applies them from the `HardwareProfile`, so a T4 session cannot
accidentally inherit a bf16 default out of an upstream config.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..hardware import HardwareProfile
from ..logging_utils import get_logger
from ..paths import open_unlearning_dir

__all__ = [
    "DEFAULT_TOLERANCES",
    "PUBLISHED_TARGETS",
    "EvalSpec",
    "MetricComparison",
    "ReproReport",
    "build_eval_command",
    "compare_to_published",
    "find_summary",
    "parse_summary",
    "run_eval",
]

log = get_logger(__name__)

# ---------------------------------------------------------------------------------
# Published targets, transcribed from open-unlearning/docs/repro.md
# TOFU, Llama-3.2-1B-Instruct, forget10.
# ---------------------------------------------------------------------------------
PUBLISHED_TARGETS: dict[str, dict[str, float]] = {
    "full": {"model_utility": 0.60, "forget_truth_ratio": 0.48, "forget_quality": 1.66e-21},
    "retain90": {"model_utility": 0.59, "forget_truth_ratio": 0.63, "forget_quality": 1.0},
    "npo_forget10": {"model_utility": 0.46, "forget_truth_ratio": 0.70, "forget_quality": 0.02},
    "grad_ascent_forget10": {
        "model_utility": 0.00,
        "forget_truth_ratio": 2.25e-18,
        "forget_quality": 1.06e-239,
    },
    "grad_diff_forget10": {
        "model_utility": 0.49,
        "forget_truth_ratio": 3.53e-27,
        "forget_quality": 1.06e-239,
    },
}

# forget_quality is DELIBERATELY absent from the gated set. It is the p-value of a KS
# test between the forget and retain truth-ratio distributions: it ranges over 200
# orders of magnitude across methods and is unstable to single-sample changes. Gating a
# reproduction on "0.02 +/- tolerance" is meaningless. Report it; check only that
# log10(forget_quality) lands in the same order of magnitude.
DEFAULT_TOLERANCES: dict[str, float] = {
    "model_utility": 0.01,
    "forget_truth_ratio": 0.01,
}

FORGET_QUALITY_LOG10_RANGE: tuple[float, float] = (-2.0, 0.0)

# open-unlearning's SUMMARY.json keys have moved between releases. Map to stable names
# here so a schema change is one edit in one place, caught by
# tests/integration/test_ou_bridge_parse.py without a GPU.
_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "model_utility": ("model_utility", "utility", "tofu_model_utility"),
    "forget_quality": ("forget_quality", "tofu_forget_quality"),
    "forget_truth_ratio": (
        "forget_truth_ratio",
        "forget_Truth_Ratio",
        "forget_truth_ratio_agg",
    ),
    "forget_Q_A_ROUGE": ("forget_Q_A_ROUGE", "forget_qa_rouge"),
    "forget_Q_A_Prob": ("forget_Q_A_Prob", "forget_qa_prob"),
}


@dataclass
class EvalSpec:
    """One open-unlearning eval invocation."""

    model_path: str
    task_name: str
    experiment: str = "eval/tofu/default"
    model_config: str = "Llama-3.2-1B-Instruct"
    forget_split: str = "forget10"
    retain_split: str = "retain90"
    retain_logs_path: str = "saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
    output_dir: str | None = None
    batch_size: int = 1
    seed: int = 42
    extra_overrides: list[str] = field(default_factory=list)

    def resolved_output_dir(self) -> str:
        return self.output_dir or f"saves/eval/{self.task_name}"


def build_eval_command(
    spec: EvalSpec,
    hw: HardwareProfile,
    *,
    python: str = "python",
) -> list[str]:
    """Build the exact command. Pure — safe to call in `--dry-run`."""
    dtype = hw.recommended_dtype if hw.is_cuda else "float32"
    attn = hw.recommended_attn

    cmd = [
        python,
        "src/eval.py",
        "--config-name=eval.yaml",
        f"experiment={spec.experiment}",
        f"model={spec.model_config}",
        f"model.model_args.pretrained_model_name_or_path={spec.model_path}",
        # The two T4 overrides. Applied unconditionally from the hardware profile so a
        # bf16 default in an upstream config can never reach a Turing card.
        f"model.model_args.attn_implementation={attn}",
        f"model.model_args.torch_dtype={dtype}",
        f"forget_split={spec.forget_split}",
        f"retain_split={spec.retain_split}",
        f"retain_logs_path={spec.retain_logs_path}",
        f"task_name={spec.task_name}",
        f"paths.output_dir={spec.resolved_output_dir()}",
    ]
    cmd.extend(spec.extra_overrides)
    return cmd


def command_string(cmd: Sequence[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def run_eval(
    spec: EvalSpec,
    hw: HardwareProfile,
    *,
    cwd: Path | None = None,
    python: str = "python",
    timeout: int = 7200,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Shell out to open-unlearning. Captures stdout, return code, and the summary path."""
    ou = cwd or open_unlearning_dir()
    cmd = build_eval_command(spec, hw, python=python)

    if dry_run:
        return {
            "dry_run": True,
            "cwd": str(ou),
            "command": command_string(cmd),
            "argv": cmd,
            "summary_path": None,
        }

    if not ou.exists():
        raise FileNotFoundError(
            f"open-unlearning submodule not found at {ou}. Run: "
            "git submodule update --init --recursive"
        )

    log.info("running open-unlearning eval: %s", command_string(cmd))
    proc = subprocess.run(cmd, cwd=str(ou), capture_output=True, text=True, timeout=timeout)

    summary = find_summary(ou / spec.resolved_output_dir())
    return {
        "dry_run": False,
        "cwd": str(ou),
        "command": command_string(cmd),
        "argv": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
        "summary_path": str(summary) if summary else None,
    }


def find_summary(output_dir: Path) -> Path | None:
    """Locate the produced `*SUMMARY.json`, newest last."""
    if not output_dir.exists():
        return None
    hits = sorted(output_dir.rglob("*SUMMARY.json"))
    return hits[-1] if hits else None


def _coerce(value: Any) -> float | None:
    """open-unlearning sometimes nests a metric under `agg_value`."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("agg_value", "value", "score"):
            if key in value:
                return _coerce(value[key])
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_summary(path: str | Path) -> dict[str, float]:
    """Parse a SUMMARY.json into stable metric names.

    Tolerates upstream key renames via `_KEY_ALIASES` and one level of nesting. Anything
    numeric that is not aliased is passed through under its own name, so a new upstream
    metric is visible rather than silently dropped.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict):
        raise ValueError(f"{p}: expected a JSON object at the top level")

    # Flatten one level so {"forget10": {...}} style summaries still resolve.
    flat: dict[str, Any] = {}
    for k, v in raw.items():
        flat.setdefault(k, v)
        if isinstance(v, Mapping) and not any(key in v for key in ("agg_value", "value", "score")):
            for k2, v2 in v.items():
                flat.setdefault(k2, v2)

    out: dict[str, float] = {}
    for stable, aliases in _KEY_ALIASES.items():
        for alias in aliases:
            if alias in flat:
                val = _coerce(flat[alias])
                if val is not None:
                    out[stable] = val
                    break

    claimed = {a for aliases in _KEY_ALIASES.values() for a in aliases}
    for k, v in flat.items():
        if k in claimed or k in out:
            continue
        val = _coerce(v)
        if val is not None:
            out[k] = val
    return out


@dataclass
class MetricComparison:
    metric: str
    ours: float | None
    published: float | None
    tolerance: float | None
    passed: bool
    gated: bool
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "ours": self.ours,
            "published": self.published,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "gated": self.gated,
            "note": self.note,
        }


@dataclass
class ReproReport:
    checkpoint_key: str
    comparisons: list[MetricComparison] = field(default_factory=list)
    passed: bool = False
    metrics: dict[str, float] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "checkpoint_key": self.checkpoint_key,
            "passed": self.passed,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "metrics": self.metrics,
            "meta": self.meta,
        }

    def table(self) -> str:  # pragma: no cover - human-facing
        rows = [f"{'metric':<24}{'ours':>14}{'published':>14}  verdict"]
        for c in self.comparisons:
            verdict = ("PASS" if c.passed else "FAIL") if c.gated else "reported, NOT gated"
            if c.note:
                verdict = f"{verdict} ({c.note})"
            rows.append(f"{c.metric:<24}{c.ours!s:>14}{c.published!s:>14}  {verdict}")
        rows.append(f"\nGATE: {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(rows)


def compare_to_published(
    ours: Mapping[str, float],
    published: Mapping[str, float] | str = "npo_forget10",
    tolerances: Mapping[str, float] | None = None,
    *,
    checkpoint_key: str = "",
) -> ReproReport:
    """Apply the Days 1-2 gate.

    Gated:      model_utility and forget_truth_ratio, each within +/- 0.01.
    Not gated:  forget_quality — reported, with an order-of-magnitude sanity check.
    """
    import math

    if isinstance(published, str):
        checkpoint_key = checkpoint_key or published
        if published not in PUBLISHED_TARGETS:
            raise KeyError(
                f"unknown published target '{published}'. Known: {sorted(PUBLISHED_TARGETS)}"
            )
        target = PUBLISHED_TARGETS[published]
    else:
        target = dict(published)

    tol = dict(tolerances or DEFAULT_TOLERANCES)
    report = ReproReport(checkpoint_key=checkpoint_key or "custom", metrics=dict(ours))
    all_pass = True

    for metric, tolerance in tol.items():
        got = ours.get(metric)
        want = target.get(metric)
        if got is None:
            all_pass = False
            report.comparisons.append(
                MetricComparison(metric, None, want, tolerance, False, True, "MISSING KEY")
            )
            continue
        if want is None:
            report.comparisons.append(
                MetricComparison(metric, got, None, tolerance, True, False, "no published target")
            )
            continue
        delta = abs(got - want)
        # Epsilon so that a value sitting EXACTLY on the tolerance passes. Without it,
        # 0.46 + 0.01 == 0.47000000000000003 fails a `<= 0.01` check by 6e-17, and you
        # spend an hour on Colab bisecting a chat template that was never wrong.
        ok = delta <= tolerance + 1e-9
        all_pass &= ok
        report.comparisons.append(
            MetricComparison(metric, got, want, tolerance, ok, True, f"|d|={delta:.4f}")
        )

    # forget_quality: reported, never gated.
    fq = ours.get("forget_quality")
    fq_target = target.get("forget_quality")
    note = "KS p-value; unstable over 200 orders of magnitude — reported only"
    if fq is not None and fq > 0:
        log10 = math.log10(fq)
        lo, hi = FORGET_QUALITY_LOG10_RANGE
        in_range = lo <= log10 <= hi
        note = (
            f"log10={log10:.2f}, expected in [{lo}, {hi}]: {'ok' if in_range else 'OUT OF RANGE'}"
        )
    report.comparisons.append(
        MetricComparison("forget_quality", fq, fq_target, None, True, False, note)
    )

    report.passed = all_pass
    return report
