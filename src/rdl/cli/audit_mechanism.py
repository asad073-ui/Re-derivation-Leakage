"""`rdl graph-audit-mechanism` — the runtime half of the mechanism verdict.

The report's static purity check reads `propagates_scope` off the manifest, which is a
restatement of the config file. This command reads the COUNTERS the run actually
produced and writes `MECHANISM_RUNTIME_AUDIT.json` beside the report.

It exits nonzero when the pathway was not live, so it can be used as a gate in a runbook
rather than as a thing an operator reads and interprets.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..eval.mechanism_liveness import audit_mechanism_liveness
from ..studies.graph_leak.evidence import atomic_json

__all__ = ["audit_mechanism"]


def audit_mechanism(
    run: Path = typer.Option(..., "--run", help="a runs/graph/<run-id> directory"),
    require_applicable: bool = typer.Option(
        True,
        "--require-applicable/--allow-not-applicable",
        help=(
            "treat a run that never carried the M5 pair as a failure. Off, a study "
            "without those arms audits as 'not applicable' and exits 0"
        ),
    ),
    output: Path | None = typer.Option(
        None, "--output", help="defaults to <run>/MECHANISM_RUNTIME_AUDIT.json"
    ),
) -> None:
    """Audit whether the M5 forwarding pathway was actually exercised at runtime."""
    performance_path = run / "PERFORMANCE.json"
    manifest_path = run / "RUN_MANIFEST.json"
    if not performance_path.exists():
        raise typer.BadParameter(
            f"{performance_path} does not exist. The runtime audit reads per-arm defence "
            "counters, which only a completed run writes."
        )

    performance = json.loads(performance_path.read_text(encoding="utf-8"))
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    arms_present = [str(a.get("arm")) for a in (manifest.get("arms") or []) if a.get("arm")] or None

    audit = audit_mechanism_liveness(performance.get("defenses"), arms_present=arms_present)
    payload = {
        **audit.to_dict(),
        "run": run.name,
        "study_id": manifest.get("study_id"),
        "protocol": manifest.get("protocol"),
        "git_sha": manifest.get("git_sha"),
    }

    destination = output or (run / "MECHANISM_RUNTIME_AUDIT.json")
    atomic_json(destination, payload)

    typer.echo(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))

    if not audit.applicable:
        typer.echo(
            "M5's arms were not both run here, so there is no propagation pathway to audit.",
            err=True,
        )
        raise typer.Exit(1 if require_applicable else 0)

    if not audit.live:
        typer.echo(
            f"MECHANISM PATHWAY NOT LIVE — {len(audit.blockers)} blocker(s). The M5 numbers "
            "in this run must not be read as evidence about forward propagation.",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"mechanism pathway live -> {destination}")
