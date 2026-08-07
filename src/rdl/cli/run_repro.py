"""`rdl run-repro` — Days 1-2.

What "reproduce" means here, and what it must NOT mean:

  DO NOT reproduce the TRAINING. Upstream's numbers come from 2x L40s under DeepSpeed
  ZeRO-3 in bf16; they say themselves that the numbers shift when the distributed setup
  changes, including on a single GPU. A gradient-ascent-family objective in fp16 is
  numerically unsound on top of that. Attempting it burns the week and produces a
  mismatch you cannot interpret.

  DO reproduce the EVALUATION, on a published checkpoint. That is deterministic enough
  to be a real gate, costs no training, and tests exactly what Phase 0 depends on: that
  OUR installation of open-unlearning computes THEIR metrics correctly.

Run order: `--target full` FIRST. It uses a checkpoint that definitely exists, so it
isolates "is my install correct" from "does the unlearned checkpoint exist".
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..eval.openunlearning_bridge import (
    PUBLISHED_TARGETS,
    EvalSpec,
    compare_to_published,
    parse_summary,
    run_eval,
)
from ..hardware import detect
from ..logging_utils import get_logger
from ..models.registry import KNOWN_MODELS
from ..paths import append_manifest, git_sha, make_run_id, open_unlearning_dir, run_dir
from ..seeding import set_all_seeds

__all__ = ["run_repro"]

log = get_logger(__name__)


def _utc_tag() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


_TARGET_TO_REPO = {
    "full": KNOWN_MODELS["tofu_llama32_1b_full"].repo_id,
    "retain90": KNOWN_MODELS["tofu_llama32_1b_retain90"].repo_id,
    "npo_forget10": KNOWN_MODELS["tofu_llama32_1b_npo_forget10"].repo_id,
}


def run_repro(
    target: str = typer.Option(
        "full",
        "--target",
        help="which published checkpoint to gate on: full | retain90 | npo_forget10. "
        "RUN `full` FIRST.",
    ),
    model_path: str | None = typer.Option(
        None, "--model-path", help="override the HF repo id (e.g. a newly discovered NPO ckpt)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print the exact open-unlearning command and exit"
    ),
    seed: int = typer.Option(42, "--seed"),
    forget_split: str = typer.Option("forget10", "--forget-split"),
    batch_size: int = typer.Option(
        1,
        "--batch-size",
        help="1 for any number that goes in the paper. Upstream's eval default is 32, "
        "which is what produced the published reference — the report records both.",
    ),
    timeout: int = typer.Option(7200, "--timeout", help="seconds"),
    condition: Path | None = typer.Option(
        None, "--condition", help="condition YAML (recorded in the report; not required)"
    ),
) -> None:
    """Run the open-unlearning eval and gate against the published numbers."""
    if target not in PUBLISHED_TARGETS and model_path is None:
        typer.secho(
            f"unknown target '{target}'. Known: {sorted(_TARGET_TO_REPO)}", fg=typer.colors.RED
        )
        raise typer.Exit(code=2)

    repo = model_path or _TARGET_TO_REPO[target]
    hw = detect()
    seeds = set_all_seeds(seed)

    typer.echo(f"hardware   {hw.summary()}")
    typer.echo(f"checkpoint {repo}")
    typer.echo(f"target     {target}  {PUBLISHED_TARGETS.get(target, {})}")
    if hw.is_cuda and not hw.supports_bf16:
        typer.secho(
            "T4-class device: forcing float16 + sdpa (no bf16, no FlashAttention-2).",
            fg=typer.colors.YELLOW,
        )

    # A unique task_name per invocation. Upstream writes into
    # `saves/eval/<task_name>/`, and reusing a name means a crashed run leaves the
    # previous run's SUMMARY.json sitting exactly where this run's is looked for.
    task_name = f"rdl_repro_{target}_{forget_split}_s{seed}_b{batch_size}_{_utc_tag()}"

    spec = EvalSpec(
        model_path=repo,
        task_name=task_name,
        forget_split=forget_split,
        batch_size=batch_size,
        seed=seed,
        overwrite=True,
    )
    typer.echo(f"task_name  {task_name}")
    typer.echo(f"splits     forget={spec.forget_split} holdout={spec.holdout_split}")

    result = run_eval(spec, hw, timeout=timeout, dry_run=dry_run)

    if dry_run:
        typer.echo("\n--- exact command (nothing was run) ---")
        typer.echo(f"cd {result['cwd']}")
        typer.echo(result["command"])
        return

    if result.get("returncode") != 0:
        typer.secho(f"\neval FAILED (rc={result.get('returncode')})", fg=typer.colors.RED)
        typer.echo(result.get("stderr_tail", "")[-3000:])
        raise typer.Exit(code=1)

    summary_path = result.get("summary_path")
    if not summary_path:
        typer.secho(
            f"no SUMMARY.json written by THIS run under "
            f"{open_unlearning_dir() / spec.resolved_output_dir()}",
            fg=typer.colors.RED,
        )
        if result.get("stale_summary_ignored"):
            typer.secho(
                f"  a pre-existing summary was found and IGNORED: "
                f"{result['stale_summary_ignored']}\n"
                "  it predates this run and is not this run's result.",
                fg=typer.colors.YELLOW,
            )
        raise typer.Exit(code=1)

    metrics = parse_summary(summary_path)
    report = compare_to_published(metrics, target, checkpoint_key=target, batch_size=batch_size)

    typer.echo("\n" + report.table())
    if not report.passed:
        typer.secho(
            "\nBisect in THIS order, one change at a time:\n"
            "  1. chat template   2. padding_side   3. batch_size   4. dtype   "
            "5. transformers version\n"
            "Log every attempt in docs/04_decisions.md.",
            fg=typer.colors.YELLOW,
        )

    # ---- persist, append-only ----------------------------------------------------
    cfg_hash = report.checkpoint_key + "-" + str(seed)
    import hashlib

    cfg_hash = hashlib.sha256(cfg_hash.encode()).hexdigest()
    run_id = make_run_id(cfg_hash)
    out = run_dir(run_id)

    payload = {
        "run_id": run_id,
        "phase": "phase0_days1-2_repro",
        "target": target,
        "checkpoint": repo,
        "hardware": hw.to_dict(),
        "seeding": seeds.to_dict(),
        "open_unlearning_command": result["command"],
        "task_name": task_name,
        "forget_split": spec.forget_split,
        "holdout_split": spec.holdout_split,
        "batch_size": batch_size,
        "seed": seed,
        "summary_path": summary_path,
        "summary_written_after": result.get("started_at"),
        "condition_file": str(condition) if condition else None,
        "git_sha": git_sha(),
        **report.to_dict(),
    }
    (out / "repro_report.json").write_text(json.dumps(payload, indent=2, default=str))
    Path(out / "SUMMARY.json").write_text(Path(summary_path).read_text(encoding="utf-8"))

    append_manifest(
        {
            "run_id": run_id,
            "phase": "phase0_repro",
            "target": target,
            "checkpoint": repo,
            "passed": report.passed,
            "device": hw.name,
            "git_sha": git_sha(),
        }
    )
    typer.echo(f"\nwrote {out}")

    if not report.passed:
        raise typer.Exit(code=1)
