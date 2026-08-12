"""``rdl graph-plan`` — resolve the config, print the cost model, generate nothing.

This is the GPU-readiness gate's arithmetic. If the predicted generation count is not
what the protocol says (168 for the 4x2 smoke), the wiring is wrong and no GPU time
should be spent finding that out.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..studies.graph_leak.evidence import atomic_json
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

__all__ = ["plan_graph_run"]


def plan_graph_run(
    launch: Path = typer.Option(Path(DEFAULT_LAUNCH), "--launch", help="launch yaml"),
    profile: str | None = typer.Option(None, "--profile", help="override active_profile"),
    topology: str | None = typer.Option(None, "--topology"),
    fixture: Path | None = typer.Option(None, "--fixture", help="offline fixture; CPU only"),
    cohort: Path | None = typer.Option(None, "--cohort", help="explicit cohort manifest"),
    policy_cohort: Path | None = typer.Option(
        None, "--forget-policy-cohort", help="explicit FORGET cohort for the concept registry"
    ),
    limit: int | None = typer.Option(
        None, "--limit", min=1, help="evaluation questions only; never narrows the forget policy"
    ),
    n_samples: int | None = typer.Option(
        None, "--n-samples", min=1, help="reduce (never exceed) the profile's sample budget"
    ),
    challenges: str = typer.Option("natural", "--challenges", help="comma-separated"),
    protocol: str = typer.Option(
        "end_to_end_safety", "--protocol", help="end_to_end_safety or graph_flow"
    ),
    token: str | None = typer.Option(None, "--hf-token"),
    output: Path | None = typer.Option(None, "--output", help="write PLAN.json here"),
) -> None:
    """Dry-run cost estimation for a graph study configuration.

    Takes the SAME `--n-samples` as `graph-run` and applies it through the same
    function, so the printed cost is the cost of the run that follows.
    """
    overrides = [f"active_profile={profile}"] if profile else None
    cfg = load_config_or_fail(launch, overrides=overrides, topology=topology)
    cfg = apply_sample_budget(cfg, n_samples)
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
    modes = tuple(c.strip() for c in challenges.split(",") if c.strip())
    runner = GraphRunner(
        cfg=cfg,
        items=cohorts.evaluation_items,
        cohort=cohorts.evaluation,
        policy_cohort=cohorts.policy,
        policy_items=cohorts.policy_items,
        backend=backend,
        output=output or Path("."),
        challenges=modes,
        protocol=protocol,
        tokenizer_revision=tokenizer_revision,
    )
    plan = runner.plan()
    cohort_obj = cohorts.evaluation
    payload = {
        "study_id": cfg.study.study_id,
        "phase": cfg.phase,
        "profile": cfg.profile.name,
        "backend": cfg.profile.runtime.backend,
        "model": cfg.model.name,
        "cohort_split": cohort_obj.split,
        "cohort_fingerprint": cohort_obj.fingerprint(),
        "cohort_dataset_revision": cohort_obj.dataset_revision,
        "n_concepts": len(cohort_obj.concept_ids),
        **cohorts.to_dict(),
        "protocol": protocol,
        **cfg.hashes(),
        **plan.to_dict(),
        "logical_agents": cfg.profile.agents.logical_count,
        "shared_model_handles": len(set(getattr(backend, "shared_handle_ids", dict)().values())),
        "k_values": list(cfg.profile.sampling.k_values),
        "primary_k": cfg.study.sampling.primary_k,
        "primary_k_reachable": cfg.study.sampling.primary_k in cfg.profile.sampling.k_values,
        "profile_reportable": cfg.profile.reportable,
        "detector_backend": cfg.study.detector.backend,
        "detector_status": cfg.study.detector.status,
        "model_repo_id": cfg.model.repo_id,
        "model_revision": cfg.model.revision,
        "tokenizer_revision": cfg.model.tokenizer_revision,
        "model_pinned": cfg.model.pinned,
    }
    backend.close()
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    if output:
        output.mkdir(parents=True, exist_ok=True)
        # Through `atomic_json`, so the plan is written strictly and atomically by the
        # same code path as every other artefact.
        atomic_json(output / "PLAN.json", payload)
