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

_TARGET_TO_REPO = {
    "full": "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
    "retain90": "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
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
    retain_split: str = typer.Option("retain90", "--retain-split"),
    batch_size: int = typer.Option(
        1, "--batch-size", help="1 for any number that goes in the paper"
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

    spec = EvalSpec(
        model_path=repo,
        task_name=f"rdl_repro_{target}",
        forget_split=forget_split,
        retain_split=retain_split,
        batch_size=batch_size,
        seed=seed,
    )

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
            f"no SUMMARY.json under {open_unlearning_dir() / spec.resolved_output_dir()}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    metrics = parse_summary(summary_path)
    report = compare_to_published(metrics, target, checkpoint_key=target)

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
        "summary_path": summary_path,
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
