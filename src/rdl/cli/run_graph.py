"""``rdl graph-run`` — the generation phase. No scoring, ever."""

from __future__ import annotations

from pathlib import Path

import typer

from ..paths import make_run_id, repo_root
from ..studies.graph_leak.runner import GraphRunner
from .graph_common import (
    DEFAULT_LAUNCH,
    apply_sample_budget,
    assert_control_arm_has_enough_items,
    build_backend,
    load_config_or_fail,
    resolve_run_cohorts,
    stub_source_items,
)

__all__ = ["run_graph"]


def run_graph(
    launch: Path = typer.Option(Path(DEFAULT_LAUNCH), "--launch"),
    profile: str | None = typer.Option(None, "--profile", help="override active_profile"),
    topology: str | None = typer.Option(None, "--topology"),
    fixture: Path | None = typer.Option(None, "--fixture", help="offline fixture; CPU only"),
    cohort: Path | None = typer.Option(None, "--cohort", help="explicit cohort manifest"),
    policy_cohort: Path | None = typer.Option(
        None,
        "--forget-policy-cohort",
        help="explicit FORGET cohort for the concept registry and the deleted baseline "
        "memory; defaults to the launch file's forget_policy_phase",
    ),
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="evaluation questions only; never narrows the forget policy"
    ),
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

    cohorts = resolve_run_cohorts(
        cfg,
        fixture=fixture,
        cohort_path=cohort,
        policy_cohort_path=policy_cohort,
        token=token,
        limit=limit,
    )
    assert_control_arm_has_enough_items(cfg, cohorts.evaluation_items)
    backend, tokenizer_revision = build_backend(cfg, stub_source_items(cohorts), token=token)
    out = output or (repo_root() / "runs" / "graph" / make_run_id(cfg.resolved_run_hash()))
    runner = GraphRunner(
        cfg=cfg,
        items=cohorts.evaluation_items,
        cohort=cohorts.evaluation,
        policy_cohort=cohorts.policy,
        policy_items=cohorts.policy_items,
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
    dispatch = manifest["actual_generations"]
    typer.echo(
        f"wrote {out} — {manifest['completed_trajectories']} trajectories, "
        f"{dispatch['graph']['dispatched']} graph and {dispatch['probe']['dispatched']} "
        f"probe generations dispatched"
    )
    typer.echo(
        f"forget policy: {manifest['forget_policy_split']} "
        f"({manifest['forget_policy_cohort']['n_concepts']} concepts) — "
        f"questions: {manifest['cohort_split']} "
        f"({manifest['evaluation_cohort']['n_concepts']} concepts)"
    )
