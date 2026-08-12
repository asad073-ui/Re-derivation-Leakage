"""``rdl graph-finalize`` — verify the evidence, then stamp the run.

Recomputes every shard hash against the manifest and against the per-shard ledger. A
shard edited after the run fails here, which is the point: the report reads the shards,
and nothing else would notice.

``ok`` is the whole gate, not just the hashes (GU-0024). It used to be
``generations.ok and traces.ok``, so a run that stopped a third of the way through, or
that had never been reported, exited 0 and printed a green object. That is the exact
moment the finalizer exists for — it is what a GPU session runs immediately before
archiving evidence and destroying the instance — and an exit code that says "verified"
over a partial run is worse than no check at all. ``ok`` now requires:

    hashes_ok        every shard matches the manifest AND the ledger
    complete         the run finished and the row count matches the plan
    report_present   GRAPH_LEAK_REPORT.json exists

``raw_evidence_uri`` is deliberately NOT part of ``ok``: it is reported and warned about,
because a diagnostic run that is never archived is a legitimate thing to finalize, while
a partial or unreported one is not.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..paths import git_dirty, git_sha
from ..studies.graph_leak.evidence import atomic_json, read_shards, verify_ledger, verify_shards

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
    generations_ledger = verify_ledger(run / "generations")
    traces_ledger = verify_ledger(run / "traces", prefix="graph-events")
    rows = list(read_shards(run / "generations"))
    plan = manifest.get("plan", {})
    planned = int(plan.get("planned_trajectories", 0))

    hashes_ok = (
        generations["ok"] and traces["ok"] and generations_ledger["ok"] and traces_ledger["ok"]
    )
    complete = bool(manifest.get("complete", False)) and len(rows) == planned and planned > 0
    report_present = (run / "GRAPH_LEAK_REPORT.json").exists()

    blocking: list[str] = []
    if not hashes_ok:
        blocking.append("shard hashes do not verify")
    if not complete:
        blocking.append(
            f"run is not complete: manifest.complete={manifest.get('complete')}, "
            f"{len(rows)} trajectories on disk against {planned} planned"
        )
    if not report_present:
        blocking.append("GRAPH_LEAK_REPORT.json is absent; run `rdl graph-report` first")

    warnings: list[str] = []
    if not raw_evidence_uri:
        warnings.append(
            "no --raw-evidence-uri: the shards exist only on this machine. Archive them "
            "before the instance is destroyed."
        )
    if manifest.get("git_dirty"):
        warnings.append("the run recorded git_dirty=true; it is diagnostic, not reportable")

    finalization = {
        "schema": "graph-finalization-v2",
        "run": run.name,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "generations": generations,
        "traces": traces,
        "generations_ledger": generations_ledger,
        "traces_ledger": traces_ledger,
        "n_trajectories": len(rows),
        "planned_trajectories": planned,
        "hashes_ok": hashes_ok,
        "complete": complete,
        "raw_evidence_uri": raw_evidence_uri,
        "report_present": report_present,
        "blocking": blocking,
        "warnings": warnings,
        # The whole gate. `hashes_ok` alone let a partial, unreported run exit 0.
        "ok": not blocking,
    }
    atomic_json(run / "FINALIZATION.json", finalization)
    typer.echo(json.dumps(finalization, indent=2, sort_keys=True))
    for warning in warnings:
        typer.echo(f"WARNING: {warning}", err=True)
    if not finalization["ok"]:
        for reason in blocking:
            typer.echo(f"BLOCKING: {reason}", err=True)
        raise typer.Exit(code=1)
