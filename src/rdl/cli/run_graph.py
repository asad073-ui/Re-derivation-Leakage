"""``rdl graph-run`` — the generation phase. No scoring, ever."""

from __future__ import annotations

from pathlib import Path

import typer

from ..paths import make_run_id, repo_root
from ..studies.graph_leak.runner import GraphRunner
from .graph_common import (
    DEFAULT_LAUNCH,
    apply_sample_budget,
    build_backend,
    load_config_or_fail,
    resolve_cohort_items,
)

__all__ = ["run_graph"]


def run_graph(
    launch: Path = typer.Option(Path(DEFAULT_LAUNCH), "--launch"),
    profile: str | None = typer.Option(None, "--profile", help="override active_profile"),
    topology: str | None = typer.Option(None, "--topology"),
    fixture: Path | None = typer.Option(None, "--fixture", help="offline fixture; CPU only"),
    cohort: Path | None = typer.Option(None, "--cohort", help="explicit cohort manifest"),
    limit: int | None = typer.Option(None, "--limit", min=1),
    n_samples: int | None = typer.Option(
        None, "--n-samples", min=1, help="reduce (never exceed) the profile's sample budget"
    ),
    challenges: str = typer.Option("natural", "--challenges", help="comma-separated"),
    protocol: str = typer.Option(
        "end_to_end_safety",
        "--protocol",
        help="end_to_end_safety (does the deployed system release it) or graph_flow "
        "(can it propagate once produced)",
    ),
    output: Path | None = typer.Option(None, "--output"),
    resume: bool = typer.Option(False, "--resume"),
    token: str | None = typer.Option(None, "--hf-token"),
) -> None:
    """Run every arm of a graph study and write immutable evidence shards."""
    overrides = [f"active_profile={profile}"] if profile else None
    cfg = load_config_or_fail(launch, overrides=overrides, topology=topology)
    # The SAME function `graph-plan` uses, so the plan describes this run.
    cfg = apply_sample_budget(cfg, n_samples)
    if cfg.study.sampling.primary_k not in cfg.profile.sampling.k_values:
        typer.echo(
            f"note: primary_k={cfg.study.sampling.primary_k} is outside this run's sample "
            "budget; the run is a wiring check and its reports will be diagnostic.",
            err=True,
        )

    cohort_obj, items = resolve_cohort_items(
        cfg, fixture=fixture, cohort_path=cohort, token=token, limit=limit
    )
    backend, tokenizer_revision = build_backend(cfg, items, token=token)
    out = output or (repo_root() / "runs" / "graph" / make_run_id(cfg.resolved_run_hash()))
    runner = GraphRunner(
        cfg=cfg,
        items=items,
        cohort=cohort_obj,
        backend=backend,
        output=out,
        challenges=tuple(c.strip() for c in challenges.split(",") if c.strip()),
        protocol=protocol,
        tokenizer_revision=tokenizer_revision,
        resume=resume,
    )
    try:
        manifest = runner.run()
    finally:
        backend.close()
    typer.echo(
        f"wrote {out} — {manifest['completed_trajectories']} trajectories, "
        f"{manifest['actual_graph_generations']} model calls dispatched"
    )
