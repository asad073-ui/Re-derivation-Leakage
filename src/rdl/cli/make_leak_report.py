"""Report sampled Leak@k records with compatibility gates."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..eval.leak_at_k import hierarchical_bootstrap_delta, leak_at_k, validate_complete_samples

__all__ = ["make_leak_report"]


def make_leak_report(
    records: Path = typer.Option(..., "--records"),
    manifest: Path = typer.Option(..., "--manifest"),
    out: Path | None = typer.Option(None, "--out"),
    bootstrap_reps: int = typer.Option(2_000, "--bootstrap-reps", min=1),
) -> None:
    """Compute item-level curves and the pre-registered k=32 composition delta."""
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines() if line]
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    validate_complete_samples(
        rows, expected_samples=int(meta["n_samples"]), required_arms=meta["arms"]
    )
    k_values = [k for k in (1, 2, 4, 8, 16, 32, 64, 128) if k <= meta["n_samples"]]
    curves: dict[str, dict[str, dict[str, float]]] = {}
    for arm in meta["arms"]:
        arm_rows = [r for r in rows if r["arm"] == arm]
        by_item: dict[str, list[dict]] = {}
        for row in arm_rows:
            by_item.setdefault(row["item_id"], []).append(row)
        curves[arm] = {}
        for surface in (
            "direct_leak",
            "agent_message_leak",
            "final_leak",
            "store_leak",
            "certified_store_leak",
            "readback_leak",
        ):
            curves[arm][surface] = {
                str(k): sum(
                    leak_at_k([r[surface] for r in samples], k) for samples in by_item.values()
                )
                / len(by_item)
                for k in k_values
            }
    primary = "32" if "32" in k_values else str(k_values[-1])
    composition = (
        curves["C3C"]["certified_store_leak"][primary]
        - curves["C3S"]["certified_store_leak"][primary]
    )
    by_arm_item: dict[str, dict[str, list[bool]]] = {}
    clusters: dict[str, str] = {}
    for arm in ("C3C", "C3S", "C3C-guard"):
        by_arm_item[arm] = {}
        for row in rows:
            if row["arm"] == arm:
                by_arm_item[arm].setdefault(row["item_id"], []).append(
                    bool(row["certified_store_leak"])
                )
                clusters[row["item_id"]] = row["author_id"]
    composition_interval = hierarchical_bootstrap_delta(
        by_arm_item["C3C"], by_arm_item["C3S"], clusters, k=int(primary), reps=bootstrap_reps
    )
    guard_interval = hierarchical_bootstrap_delta(
        by_arm_item["C3C"],
        by_arm_item["C3C-guard"],
        clusters,
        k=int(primary),
        reps=bootstrap_reps,
    )
    body = {
        "reportable": not any(r.get("guard_diagnostic") for r in rows if r["arm"] != "C3C-guard"),
        "primary_k": int(primary),
        "composition_delta_certified_store_leak": composition,
        "composition_interval": composition_interval,
        "guard_delta_certified_store_leak": curves["C3C"]["certified_store_leak"][primary]
        - curves["C3C-guard"]["certified_store_leak"][primary],
        "guard_interval": guard_interval,
        "curves": curves,
        "warnings": [
            "OfflineSemanticScorer is a CPU regression scorer, not an LLM/NLI judge.",
            "Intervals resample authors and stochastic trajectories; secondary k curves need simultaneous correction before a paper claim.",
        ],
    }
    target = out or records.with_name("LEAK_REPORT.json")
    target.write_text(json.dumps(body, indent=2), encoding="utf-8")
    typer.echo(f"wrote {target}")
