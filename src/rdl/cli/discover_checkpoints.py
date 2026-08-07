"""`rdl discover-checkpoints` — RUN THIS FIRST ON DAY 1.

The proposal's zero-training cost model depends on the **unlearned** NPO/forget10
checkpoint being published, not just `full` and `retain90`. Confirm that before
planning around it.

If it is absent, REPO_SPEC 7.4 fallback 1 applies: gate on `full` only, and validate
the metric code by recomputing metrics from the published eval logs (the HF dataset
`open-unlearning/eval`, which `python setup_data.py --eval_logs` downloads — the relevant
files are `tofu*/evals*/*_SUMMARY.json`). Recomputing their metrics from their logs
validates our metric code without any model at all.

NEEDS NETWORK.
"""

from __future__ import annotations

import datetime as _dt
import os
import re
from pathlib import Path

import typer

from ..paths import docs_dir

__all__ = ["classify", "discover_checkpoints", "list_org_models"]

ORG = "open-unlearning"
_TARGET_RE = re.compile(r"tofu_Llama-3\.2-1B-Instruct", re.IGNORECASE)
_NPO_RE = re.compile(r"_NPO_", re.IGNORECASE)
_FORGET10_RE = re.compile(r"_forget10_", re.IGNORECASE)
# The hyperparameters docs/repro.md was generated under: lr 1e-5, alpha 1, beta 0.1,
# 10 epochs. Any other NPO repo is a different run whose metrics are NOT the published
# row, so gating it against 0.46 / 0.70 would fail for a reason that is not our install.
_REPRO_HP_RE = re.compile(r"_lr1e-05_beta0\.1_alpha1_epoch10$", re.IGNORECASE)


def list_org_models(author: str = ORG, token: str | None = None) -> list[str]:
    from huggingface_hub import HfApi

    return sorted(m.id for m in HfApi(token=token).list_models(author=author))


def classify(model_ids: list[str]) -> dict[str, list[str]]:
    """Group the org's models.

    The old matcher looked for the substring "npo" anywhere and then accepted "10"
    anywhere, which matched `epoch10` on every forget01/forget05 repo and reported them
    as forget10 candidates. It also expected the pattern
    `tofu_<model>_<METHOD>_<split>`, which is how the FINETUNED and RETAIN checkpoints
    are named but not the unlearned ones — those are published one-per-hyperparameter
    setting as `unlearn_tofu_<model>_<forget_split>_<METHOD>_lr..._beta..._alpha..._epoch...`.
    """
    target = [m for m in model_ids if _TARGET_RE.search(m)]
    npo = [m for m in target if _NPO_RE.search(m)]
    npo10 = [m for m in npo if _FORGET10_RE.search(m)]
    return {
        "all_llama32_1b": target,
        "npo": npo,
        "npo_forget10_candidates": npo10,
        # The one whose published metrics are the Days 1-2 gate.
        "npo_forget10_repro_match": [m for m in npo10 if _REPRO_HP_RE.search(m)],
        "full": [m for m in target if m.lower().endswith("_full")],
        "retain90": [m for m in target if "retain90" in m.lower()],
    }


def discover_checkpoints(
    author: str = typer.Option(ORG, "--author", help="HF org to enumerate"),
    out: Path | None = typer.Option(
        None, "--out", help="markdown output (default docs/checkpoints_discovered.md)"
    ),
    token: str | None = typer.Option(None, "--token", help="HF token (default $HF_TOKEN)"),
) -> None:
    """List published checkpoints and confirm whether NPO/forget10 exists. NEEDS NETWORK."""
    token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        models = list_org_models(author, token)
    except Exception as exc:
        typer.secho(f"Hub query failed: {type(exc).__name__}: {exc}", fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc

    groups = classify(models)
    typer.echo(f"{len(models)} models under {author}\n")
    for m in groups["all_llama32_1b"]:
        typer.echo(f"  {m}")

    npo10 = groups["npo_forget10_candidates"]
    repro = groups["npo_forget10_repro_match"]
    typer.echo("")
    if npo10:
        typer.secho(f"NPO forget10 candidates: {npo10}", fg=typer.colors.GREEN)
        verdict = "FOUND"
        if repro:
            typer.secho(
                f"\nUse THIS one for the Days 1-2 gate: {repro[0]}\n"
                "  It is the lr1e-05 / beta0.1 / alpha1 / epoch10 run, which is the "
                "setting docs/repro.md was generated under. The other NPO repos are "
                "different hyperparameters and their metrics are not the published row.",
                fg=typer.colors.GREEN,
            )
        else:
            typer.secho(
                "\nWARNING: none of the candidates matches the published repro "
                "hyperparameters (lr1e-05, beta0.1, alpha1, epoch10). Gating any of "
                "them against model_utility 0.46 / forget_truth_ratio 0.70 will fail "
                "for a reason that is not your install.",
                fg=typer.colors.YELLOW,
            )
    else:
        typer.secho(
            "NPO forget10: NONE FOUND.\n"
            "  -> REPO_SPEC 7.4 fallback 1 applies. Gate on `full` only, and validate\n"
            "     the metric code against the published eval logs "
            "(`python setup_data.py --eval_logs`).\n"
            "  -> Do NOT train NPO on a T4: fp16 + a gradient-ascent objective NaNs "
            "silently.",
            fg=typer.colors.YELLOW,
        )
        verdict = "ABSENT — fallback 1 applies"

    path = out or (docs_dir() / "checkpoints_discovered.md")
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Discovered checkpoints",
        "",
        f"Generated by `rdl discover-checkpoints` at {stamp}. Regenerate; do not hand-edit.",
        "",
        f"- org: `{author}`",
        f"- total models: {len(models)}",
        f"- **NPO forget10: {verdict}**",
        "",
        "## Llama-3.2-1B-Instruct TOFU checkpoints",
        "",
        *([f"- `{m}`" for m in groups["all_llama32_1b"]] or ["_none_"]),
        "",
        "## Grouped",
        "",
        f"- `_full`: {groups['full'] or 'none'}",
        f"- `retain90`: {groups['retain90'] or 'none'}",
        f"- NPO (any split): {groups['npo'] or 'none'}",
        f"- NPO forget10 candidates: {npo10 or 'none'}",
        f"- **matches the published repro hyperparameters**: {repro or 'none'}",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    typer.echo(f"\nwrote {path}")
