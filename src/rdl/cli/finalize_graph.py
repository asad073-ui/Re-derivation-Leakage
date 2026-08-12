"""``rdl graph-finalize`` — verify the evidence, then stamp the run.

Recomputes every shard hash against the manifest. A shard edited after the run fails
here, which is the point: the report reads the shards, and nothing else would notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..paths import git_dirty, git_sha
from ..studies.graph_leak.evidence import atomic_json, read_shards, verify_shards

__all__ = ["finalize_graph"]


def finalize_graph(
    run: Path = typer.Option(..., "--run"),
    raw_evidence_uri: str | None = typer.Option(
        None, "--raw-evidence-uri", help="immutable external archive URI for the shards"
    ),
) -> None:
    manifest_path = run / "RUN_MANIFEST.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"no RUN_MANIFEST.json under {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    generations = verify_shards(run / "generations", manifest.get("evidence_shards", []))
    traces = verify_shards(run / "traces", manifest.get("trace_shards", []), prefix="graph-events")
    rows = list(read_shards(run / "generations"))
    plan = manifest.get("plan", {})
    planned = int(plan.get("planned_trajectories", 0))

    finalization = {
        "schema": "graph-finalization-v1",
        "run": run.name,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "generations": generations,
        "traces": traces,
        "n_trajectories": len(rows),
        "planned_trajectories": planned,
        "complete": manifest.get("complete", False) and len(rows) == planned,
        "raw_evidence_uri": raw_evidence_uri,
        "report_present": (run / "GRAPH_LEAK_REPORT.json").exists(),
        "ok": generations["ok"] and traces["ok"],
    }
    atomic_json(run / "FINALIZATION.json", finalization)
    typer.echo(json.dumps(finalization, indent=2, sort_keys=True))
    if not finalization["ok"]:
        raise typer.Exit(code=1)
