"""The bridge to `third_party/open-unlearning`.

Design invariant 1, restated as code: **we never fork, vendor, or patch
open-unlearning.** This module builds their command line, runs it as a subprocess, and
parses the `*_SUMMARY.json` it produces. It does not import from `third_party` in any
way that would require patching it. Every metric number we report must be producible by
their code on their configs, so a reviewer can re-run it.

The two model-arg overrides that must always be applied, and why:

  torch_dtype                  upstream's configs/model/Llama-3.2-1B-Instruct.yaml pins
                               `bfloat16`, which a T4 (SM 7.5) has no datapath for
  attn_implementation          the same file pins `flash_attention_2`, which needs
                               SM80+ **and** an installed `flash_attn` wheel

`build_eval_command` applies both from the `HardwareProfile`, so neither a Turing card
nor an Ampere card without the wheel can inherit those upstream defaults. On a bare
RTX 3090 image the second one is the difference between an eval that runs under SDPA and
an `ImportError` inside `from_pretrained` after the checkpoint has downloaded.

**The override names are upstream's, not ours.** Verified against the pinned submodule
at `4ad738a`:

  configs/experiment/eval/tofu/default.yaml  defines `forget_split`, `holdout_split`,
                                             `retain_logs_path`. There is NO
                                             `retain_split` key in the eval tree —
                                             passing one makes Hydra abort with
                                             "Could not override 'retain_split'".
  configs/eval/tofu.yaml                     `batch_size: 32`, `overwrite: false`
  configs/eval.yaml                          `seed: 0`

Everything in that list is something the caller must set explicitly or silently inherit
a value that does not match what the run claims to have used. `build_eval_command`
therefore emits `seed`, `eval.tofu.batch_size`, and `eval.tofu.overwrite` on every
invocation, and never emits `retain_split`.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import time
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
    "TOFU_SPLITS",
    "UPSTREAM_EVAL_BATCH_SIZE",
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

# TOFU's splits travel in triples. Upstream's own scripts/tofu_unlearn.sh pairs them as
# below; the eval only ever needs `forget_split` + `holdout_split`, with `retain_split`
# appearing exclusively in the *training* config tree.
TOFU_SPLITS: dict[str, tuple[str, str]] = {
    # forget_split: (holdout_split, retain_split-for-the-retain-logs-path)
    "forget01": ("holdout01", "retain99"),
    "forget05": ("holdout05", "retain95"),
    "forget10": ("holdout10", "retain90"),
}

# `configs/eval/tofu.yaml` ships `batch_size: 32`. The published numbers in
# docs/repro.md were produced at that default. We evaluate at batch_size=1 for
# determinism (see docs/00_preregistration.md §7), which means a batch-1 result is NOT
# bit-identical to the published reference. `compare_to_published` annotates the report
# when the two differ so the distinction cannot be lost in a table.
UPSTREAM_EVAL_BATCH_SIZE = 32

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
    """One open-unlearning eval invocation.

    `holdout_split` and `retain_logs_path` default to `None` and are derived from
    `forget_split` via `TOFU_SPLITS`, so the three can never drift apart — pairing
    forget10 with holdout05 is a silent scoring error, not a crash.
    """

    model_path: str
    task_name: str
    experiment: str = "eval/tofu/default"
    model_config: str = "Llama-3.2-1B-Instruct"
    # Exact Hub commit for `model_path`. Emitted as `+model.model_args.revision=<sha>`:
    # upstream splats `model_args` into `from_pretrained`, and the `+` is required
    # because the key does not exist in their config (Hydra struct mode rejects a plain
    # override for an absent key, exactly as it does for `retain_split`).
    revision: str | None = None
    forget_split: str = "forget10"
    holdout_split: str | None = None
    retain_logs_path: str | None = None
    output_dir: str | None = None
    batch_size: int = 1
    seed: int = 42
    overwrite: bool = True
    extra_overrides: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.forget_split not in TOFU_SPLITS and (
            self.holdout_split is None or self.retain_logs_path is None
        ):
            raise ValueError(
                f"unknown forget_split '{self.forget_split}'. Known: {sorted(TOFU_SPLITS)}. "
                "Pass holdout_split and retain_logs_path explicitly to use another split."
            )
        holdout, retain = TOFU_SPLITS.get(self.forget_split, ("", ""))
        if self.holdout_split is None:
            self.holdout_split = holdout
        if self.retain_logs_path is None:
            self.retain_logs_path = f"saves/eval/tofu_{self.model_config}_{retain}/TOFU_EVAL.json"

    def resolved_output_dir(self) -> str:
        return self.output_dir or f"saves/eval/{self.task_name}"


def build_eval_command(
    spec: EvalSpec,
    hw: HardwareProfile,
    *,
    python: str = "python",
) -> list[str]:
    """Build the exact command. Pure — safe to call in `--dry-run`.

    Every value the run *claims* to have used appears here. In particular `seed` and
    `eval.tofu.batch_size`: upstream defaults them to 0 and 32, so a spec that carried
    them without emitting them produced a report whose header disagreed with the run.
    """
    dtype = hw.recommended_dtype if hw.is_cuda else "float32"
    attn = hw.recommended_attn

    cmd = [
        python,
        "src/eval.py",
        "--config-name=eval.yaml",
        f"experiment={spec.experiment}",
        f"model={spec.model_config}",
        f"model.model_args.pretrained_model_name_or_path={spec.model_path}",
        # Applied unconditionally from the hardware profile, so neither the bf16 default
        # nor the flash_attention_2 default in upstream's model config can reach a device
        # that cannot run it. `hw.recommended_attn` already accounts for whether the
        # flash_attn wheel is installed, not just for the SM version.
        f"model.model_args.attn_implementation={attn}",
        f"model.model_args.torch_dtype={dtype}",
        f"forget_split={spec.forget_split}",
        # NOT `retain_split`: the eval config tree has no such key. See the module
        # docstring — Hydra aborts on an override for a key that does not exist.
        f"holdout_split={spec.holdout_split}",
        f"retain_logs_path={spec.retain_logs_path}",
        f"seed={spec.seed}",
        f"eval.tofu.batch_size={spec.batch_size}",
        # Without this, upstream skips metrics whose logs already exist under
        # output_dir, and a "new" run can silently be a replay of an old one.
        f"eval.tofu.overwrite={str(spec.overwrite).lower()}",
        f"task_name={spec.task_name}",
        f"paths.output_dir={spec.resolved_output_dir()}",
    ]
    if spec.revision:
        # `+` because `revision` is not a key in upstream's model_args. Without the pin,
        # `main` can move between the reproduction and the grid and the two runs would
        # silently be evaluating different weights.
        cmd.append(f"+model.model_args.revision={spec.revision}")
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
    # Recorded BEFORE the subprocess starts, and with a one-second slack for
    # coarse-grained filesystem timestamps. Anything older than this is a leftover from
    # a previous run and must never be reported as this run's result.
    started_at = time.time() - 1.0
    proc = subprocess.run(cmd, cwd=str(ou), capture_output=True, text=True, timeout=timeout)

    out_dir = ou / spec.resolved_output_dir()
    summary = find_summary(out_dir, newer_than=started_at)
    stale = find_summary(out_dir) if summary is None else None

    return {
        "dry_run": False,
        "cwd": str(ou),
        "command": command_string(cmd),
        "argv": cmd,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-8000:],
        "stderr_tail": proc.stderr[-8000:],
        "started_at": started_at,
        "summary_path": str(summary) if summary else None,
        # Surfaced, never used. A stale file under the output dir is the signature of a
        # failed run in a reused task_name; reporting its path is how the operator
        # finds out instead of quietly accepting last week's number.
        "stale_summary_ignored": str(stale) if stale else None,
    }


def find_summary(output_dir: Path, *, newer_than: float | None = None) -> Path | None:
    """Locate the `*SUMMARY.json` this run produced.

    `newer_than` is a POSIX timestamp: files modified before it are ignored. Without
    it, a run whose eval crashed still "finds" the summary a previous run left in the
    same `paths.output_dir`, and the gate passes on a number nothing in this session
    computed. Candidates are ordered by mtime, not by name — upstream's filenames sort
    by metric, not by recency.
    """
    if not output_dir.exists():
        return None
    hits = list(output_dir.rglob("*SUMMARY.json"))
    if newer_than is not None:
        hits = [p for p in hits if p.stat().st_mtime >= newer_than]
    if not hits:
        return None
    return max(hits, key=lambda p: (p.stat().st_mtime, str(p)))


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
    batch_size: int | None = None,
) -> ReproReport:
    """Apply the Days 1-2 gate.

    Gated:      model_utility and forget_truth_ratio, each within +/- 0.01.
    Not gated:  forget_quality — reported, with an order-of-magnitude sanity check.

    `batch_size` is recorded on the report. When it differs from upstream's evaluation
    default (32), the report says so: our number and the published number were not
    produced under the same batching, so "reproduced" means "within tolerance of", not
    "identical to".
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

    if batch_size is not None:
        report.meta["batch_size"] = batch_size
        report.meta["upstream_eval_batch_size"] = UPSTREAM_EVAL_BATCH_SIZE
        if batch_size != UPSTREAM_EVAL_BATCH_SIZE:
            report.meta["batch_size_note"] = (
                f"evaluated at batch_size={batch_size}; the published reference was "
                f"produced at upstream's default of {UPSTREAM_EVAL_BATCH_SIZE}. Agreement "
                "within tolerance is not bit-identity, and the paper must say which "
                "batching produced which number."
            )

    report.passed = all_pass
    return report
