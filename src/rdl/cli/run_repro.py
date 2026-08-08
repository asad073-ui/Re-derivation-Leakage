"""`rdl run-repro` — Days 1-2.

What "reproduce" means here, and what it must NOT mean:

  DO NOT reproduce the TRAINING. Upstream's numbers come from 2x L40s under DeepSpeed
  ZeRO-3 in bf16; they say themselves that the numbers shift when the distributed setup
  changes, including on a single GPU. A gradient-ascent-family objective in fp16 is
  numerically unsound on top of that. Attempting it burns the week and produces a
  mismatch you cannot interpret. **This does not change on an RTX 3090.** 24 GB of bf16
  makes the objective *sound*; it does not make a single card equal to two L40S under
  ZeRO-3. Training here produces a NEW number, not a reproduction.

  DO reproduce the EVALUATION, on a published checkpoint. That is deterministic enough
  to be a real gate, costs no training, and tests exactly what Phase 0 depends on: that
  OUR installation of open-unlearning computes THEIR metrics correctly.

Run order: `--target full` FIRST. It uses a checkpoint that definitely exists, so it
isolates "is my install correct" from "does the unlearned checkpoint exist".

**`--measure-only` is not a weaker gate; it is a different question.** Agent B's
checkpoint (C3D/C3C) was unlearned at lr2e-05 / beta0.5, not at the lr1e-05 / beta0.1
that produced the published 0.46 / 0.70 row. Comparing it against those numbers would
manufacture a pass or a fail out of a hyperparameter difference. `--measure-only`
records what B actually is, which is what makes the C3D result interpretable
(docs/00b_preregistration_v2.md, acceptance item 8), and `make-report` requires it.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..config import ConfigError, load_env
from ..eval.openunlearning_bridge import (
    PUBLISHED_TARGETS,
    UPSTREAM_EVAL_BATCH_SIZE,
    UPSTREAM_EVAL_SEED,
    EvalSpec,
    compare_to_published,
    is_exact_published_parity,
    is_published_parity,
    parity_gaps,
    parse_summary,
    run_eval,
)
from ..hardware import EnvHardwareMismatch, assert_env_matches_hardware, detect
from ..logging_utils import get_logger
from ..models.registry import KNOWN_MODELS, entry_for
from ..paths import (
    append_manifest,
    git_diff_sha256,
    git_dirty,
    git_sha,
    make_run_id,
    open_unlearning_dir,
    run_dir,
)
from ..provenance import pkg_version, tokenizer_provenance
from ..seeding import set_all_seeds

__all__ = ["run_repro"]

log = get_logger(__name__)


def _utc_tag() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


_TARGET_TO_ALIAS = {
    "full": "tofu_llama32_1b_full",
    "retain90": "tofu_llama32_1b_retain90",
    "npo_forget10": "tofu_llama32_1b_npo_forget10",
}
_TARGET_TO_REPO = {t: KNOWN_MODELS[a].repo_id for t, a in _TARGET_TO_ALIAS.items()}


def _pinned_revision(repo: str) -> str | None:
    """The Hub commit this repo is pinned to, if the registry knows the checkpoint."""
    entry = entry_for(repo)
    return entry.revision if entry else None


# Shared with `run-condition`, which needs the same facts about what actually ran.
# See rdl/provenance.py and ADR-0054.
_pkg_version = pkg_version


def _ou_source_sha() -> str | None:
    """HEAD of the open-unlearning submodule that actually ran.

    The superproject SHA does not identify the evaluator: the submodule can be moved
    without the parent noticing, and every metric comes out of the submodule's code.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-C", str(open_unlearning_dir()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


_tokenizer_provenance = tokenizer_provenance


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
    revision: str | None = typer.Option(
        None,
        "--revision",
        help="exact Hub commit. Defaults to the pin in models/registry.py when the "
        "checkpoint is a known one.",
    ),
    measure_only: bool = typer.Option(
        False,
        "--measure-only",
        help="record the metrics WITHOUT comparing them to a published target. Required "
        "for agent B's checkpoint, whose hyperparameters differ from the published row.",
    ),
    checkpoint_label: str | None = typer.Option(
        None,
        "--checkpoint-label",
        help="name this measurement carries in the report, e.g. agent_b_independent",
    ),
    environment: str | None = typer.Option(
        None,
        "--env",
        help="execution environment profile from configs/env, e.g. vast_rtx3090. "
        "Applies HF_HOME and refuses to start on hardware that does not match it.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="print the exact open-unlearning command and exit"
    ),
    seed: int = typer.Option(
        UPSTREAM_EVAL_SEED,
        "--seed",
        help="defaults to upstream's 0, which is what produced the published reference. "
        "Use --seed 42 --batch-size 1 for the pre-registered deterministic protocol.",
    ),
    forget_split: str = typer.Option("forget10", "--forget-split"),
    batch_size: int = typer.Option(
        UPSTREAM_EVAL_BATCH_SIZE,
        "--batch-size",
        help="defaults to upstream's 32, which is what produced the published reference. "
        "Pass 1 for the pre-registered deterministic protocol — that is a SECOND run, "
        "not a substitute for the parity gate.",
    ),
    timeout: int = typer.Option(7200, "--timeout", help="seconds"),
    condition: Path | None = typer.Option(
        None, "--condition", help="condition YAML (recorded in the report; not required)"
    ),
    allow_dirty: bool = typer.Option(
        False,
        "--allow-dirty",
        help="permit a run from a working tree with uncommitted tracked changes. The "
        "report is then marked git_dirty=true and can NEVER clear the Day-1 gate — it "
        "is a diagnostic run, because the recorded git_sha does not describe the code "
        "that produced the number.",
    ),
) -> None:
    """Run the open-unlearning eval and gate against the published numbers."""
    if target not in PUBLISHED_TARGETS and model_path is None:
        typer.secho(
            f"unknown target '{target}'. Known: {sorted(_TARGET_TO_REPO)}", fg=typer.colors.RED
        )
        raise typer.Exit(code=2)

    # ---- provenance, BEFORE anything downloads --------------------------------------
    # A report that records `git_sha: X` while executing code X does not contain is not
    # reproducible from that SHA, and nothing downstream could tell. That is exactly how
    # the first Day-1 GPU runs were recorded against `1ea12bf` while running a
    # compatibility shim added afterwards.
    # `--dry-run` is exempt: it writes no report, so there is no provenance to get
    # wrong, and inspecting the command you are ABOUT to commit to is exactly what you
    # do while the tree is still dirty.
    dirty = git_dirty()
    if dirty and not allow_dirty and not dry_run:
        typer.secho(
            f"REFUSING TO RUN: the working tree has uncommitted tracked changes, but "
            f"this report would record git_sha={git_sha()}.\n"
            "  A reviewer checking out that commit could not reproduce the recorded "
            "command.\n"
            "  -> commit the changes, then re-run; or pass --allow-dirty to take a "
            "DIAGNOSTIC measurement that can never clear the Day-1 gate.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    if dirty:
        typer.secho(
            "--allow-dirty: this is a DIAGNOSTIC run. git_dirty=true is recorded and "
            "the report is disqualified from exact published parity.",
            fg=typer.colors.YELLOW,
        )

    repo = model_path or _TARGET_TO_REPO[target]
    rev = revision or _pinned_revision(repo)
    hw = detect()
    seeds = set_all_seeds(seed)

    # ---- environment profile -------------------------------------------------------
    # Optional here (the eval only needs `detect()`), but on a rented box it is what
    # turns "wrong GPU" into a four-second refusal and keeps the HF cache on the
    # persistent volume instead of the container overlay.
    env_cfg = None
    if environment:
        import os

        try:
            env_cfg = load_env(environment)
        except ConfigError as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=2) from exc
        if env_cfg.hf_home:
            os.environ.setdefault("HF_HOME", env_cfg.hf_home)
        try:
            assert_env_matches_hardware(env_cfg, hw)
        except EnvHardwareMismatch as exc:
            typer.secho(str(exc), fg=typer.colors.RED)
            raise typer.Exit(code=2) from exc
        typer.echo(f"env        {env_cfg.name}  HF_HOME={os.environ.get('HF_HOME')}")

    run_dtype = hw.recommended_dtype if hw.is_cuda else "float32"
    parity = is_published_parity(batch_size=batch_size, seed=seed)
    gaps = parity_gaps(batch_size=batch_size, seed=seed, dtype=run_dtype, attn=hw.recommended_attn)

    typer.echo(f"hardware   {hw.summary()}")
    typer.echo(f"checkpoint {repo}")
    typer.echo(f"revision   {rev or 'UNPINNED (main can move under you)'}")
    typer.echo(
        f"settings   batch_size={batch_size} seed={seed} dtype={run_dtype} "
        f"attn={hw.recommended_attn}"
    )
    if parity and not gaps:
        typer.secho("           PUBLISHED PARITY — this is the Day 1-2 trust gate.", fg="green")
    elif parity:
        typer.secho(
            "           batch/seed match the published reference, but: "
            + "; ".join(gaps)
            + ".\n           Hardware-driven, so this is still the parity gate — the gap "
            "is recorded in the report.",
            fg=typer.colors.YELLOW,
        )
    elif not measure_only:
        typer.secho(
            "           OFF-PARITY GATED RUN: " + "; ".join(gaps) + ".\n"
            "           A miss here cannot separate a broken install from a settings "
            "difference.\n           Run the parity gate first: --batch-size 32 --seed 0",
            fg=typer.colors.YELLOW,
        )
    if measure_only:
        typer.secho(
            "measure-only: metrics are RECORDED, not compared to any published target.",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.echo(f"target     {target}  {PUBLISHED_TARGETS.get(target, {})}")
    if hw.is_cuda and not hw.supports_bf16:
        typer.secho(
            "T4-class device: forcing float16 + sdpa (no bf16, no FlashAttention-2).",
            fg=typer.colors.YELLOW,
        )
    if hw.is_cuda and hw.supports_flash_attn2 and not hw.flash_attn_installed:
        typer.secho(
            f"{hw.name} supports FlashAttention-2 but the `flash_attn` package is not "
            "installed: this run uses SDPA, and that is recorded in the report. Do not "
            "compare an SDPA run against an FA2 run inside one table.",
            fg=typer.colors.YELLOW,
        )

    # A unique task_name per invocation. Upstream writes into
    # `saves/eval/<task_name>/`, and reusing a name means a crashed run leaves the
    # previous run's SUMMARY.json sitting exactly where this run's is looked for.
    entry = entry_for(repo)
    label = checkpoint_label or (entry.alias if entry else "custom")
    kind = "measure" if measure_only else "repro"
    task_name = f"rdl_{kind}_{label}_{forget_split}_s{seed}_b{batch_size}_{_utc_tag()}"

    spec = EvalSpec(
        model_path=repo,
        task_name=task_name,
        revision=rev,
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

    report = None
    if measure_only:
        typer.echo("\nmeasured (NOT gated — no published target applies to this checkpoint):")
        for k in ("model_utility", "forget_truth_ratio", "forget_quality"):
            if k in metrics:
                typer.echo(f"  {k:<24}{metrics[k]}")
    else:
        report = compare_to_published(
            metrics,
            target,
            checkpoint_key=target,
            batch_size=batch_size,
            seed=seed,
            dtype=run_dtype,
            attn=hw.recommended_attn,
        )
        typer.echo("\n" + report.table())
        if not report.passed:
            typer.secho(
                "\nBisect in THIS order, one change at a time:\n"
                "  1. chat template   2. padding_side   3. batch_size   4. dtype   "
                "5. attn implementation (FA2 vs SDPA)   6. transformers version\n"
                "Log every attempt in docs/04_decisions.md.",
                fg=typer.colors.YELLOW,
            )

    # ---- persist, append-only ----------------------------------------------------
    import hashlib

    cfg_hash = hashlib.sha256(f"{label}-{kind}-{seed}-{batch_size}".encode()).hexdigest()
    run_id = make_run_id(cfg_hash)
    out = run_dir(run_id)

    payload = {
        "run_id": run_id,
        # Two distinct phases on purpose: `make-report` requires a *repro* for the
        # gated checkpoints and a *measurement* for agent B, and conflating them would
        # let a measurement stand in for the gate it was never compared against.
        "phase": "phase0_days1-2_measure" if measure_only else "phase0_days1-2_repro",
        "measure_only": measure_only,
        "target": None if measure_only else target,
        "checkpoint": repo,
        "revision": rev,
        "checkpoint_label": label,
        "hardware": hw.to_dict(),
        "attn_implementation": hw.recommended_attn,
        "torch_dtype": run_dtype,
        # Which upstream compatibility shims this number was produced under. Recorded
        # because the shim changes the numerical environment (fp32 logits, as
        # transformers <= 4.45.1 produced them) even though it changes neither the
        # weights' dtype nor any published-parity setting. A report that does not say
        # this is a report that cannot be checked. See rdl.compat.fp32_logits.
        "ou_compat_shims": ["fp32_logits"],
        # Which evaluator produced this number. `current_with_fp32_logits_shim` is NOT
        # the same statement as `historical_exact` (open-unlearning at fd825ea under
        # transformers 4.45.1, no shim), and conflating a reconstruction with the
        # original is exactly the claim this field exists to prevent. See ADR-0039.
        "ou_runtime_mode": "current_with_fp32_logits_shim",
        "ou_source_sha": _ou_source_sha(),
        "transformers_version": _pkg_version("transformers"),
        # Provenance of the CODE, not merely of the settings. See paths.git_dirty.
        "git_dirty": bool(dirty) if dirty is not None else None,
        "git_diff_sha256": git_diff_sha256() if dirty else None,
        # The ONLY parity predicate the Day-1 gate may use: all four published settings
        # and a tree a reviewer can check out. `published_parity` below is batch+seed
        # only and is kept for backwards comparison.
        "exact_published_parity": is_exact_published_parity(
            batch_size=batch_size,
            seed=seed,
            dtype=run_dtype,
            attn=hw.recommended_attn,
            git_dirty=dirty,
        ),
        # Fix E: the tokenizer is a separate artifact from the weights and was never
        # pinned. Its chat template decides how every prompt is rendered, so an upstream
        # edit to it moves every metric with nothing in the report to show for it.
        "tokenizer": _tokenizer_provenance(),
        # Which of the two Day 1-2 runs this is. `make-report` requires the parity one:
        # a miss at batch 1 / seed 42 cannot distinguish a broken install from a
        # settings difference, so only the parity run can clear the trust gate.
        "published_parity": parity,
        "parity_gaps": gaps,
        "env_profile": env_cfg.name if env_cfg else None,
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
        "metrics": metrics,
        **(report.to_dict() if report else {}),
    }
    name = "measure_report.json" if measure_only else "repro_report.json"
    (out / name).write_text(json.dumps(payload, indent=2, default=str))
    Path(out / "SUMMARY.json").write_text(Path(summary_path).read_text(encoding="utf-8"))

    append_manifest(
        {
            "run_id": run_id,
            "phase": "phase0_measure" if measure_only else "phase0_repro",
            "target": None if measure_only else target,
            "checkpoint": repo,
            "revision": rev,
            "checkpoint_label": label,
            "passed": None if measure_only else report.passed if report else None,
            "published_parity": parity,
            "batch_size": batch_size,
            "seed": seed,
            "device": hw.name,
            "git_sha": git_sha(),
        }
    )
    typer.echo(f"\nwrote {out}")

    if report is not None and not report.passed:
        raise typer.Exit(code=1)
