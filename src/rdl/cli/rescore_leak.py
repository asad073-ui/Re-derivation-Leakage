"""Offline scoring for saved Leak@k generations.

Generation and semantic judgement are intentionally separate: a judge revision change
must create a new scored evidence file, never trigger an expensive re-generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from ..eval.semantic import LeakKOfficialScorer

__all__ = ["rescore_leak"]


def rescore_leak(
    records: Path = typer.Option(..., "--records"),
    manifest: Path = typer.Option(..., "--manifest"),
    out: Path = typer.Option(..., "--out"),
    cache: Path = typer.Option(..., "--cache"),
    device: int = typer.Option(-1, "--device", help="Transformers pipeline device; -1 is CPU"),
) -> None:
    """Apply the pinned released Leak-k NLI+ROUGE scorer to retained raw outputs."""
    source = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines() if line]
    scorer = LeakKOfficialScorer(cache, device=device)
    scored: list[dict] = []
    for row in source:
        reference = row.get("reference_answer")
        if not isinstance(reference, str):
            raise typer.BadParameter("records lack reference_answer; cannot rescore safely")
        raw = row.get("raw_outputs") or {}
        direct = str(raw.get("direct") or "")
        final = str(raw.get("final") or direct)
        messages = "\n".join(str(x) for x in (raw.get("agent_messages") or []))
        writes = "\n".join(str(x) for x in (raw.get("memory_writes") or []))
        probe = raw.get("probe") or {}
        verdicts = {
            "direct": scorer.score(reference, direct),
            "final": scorer.score(reference, final),
            "agent_messages": scorer.score(reference, messages),
            "memory_writes": scorer.score(reference, writes),
            "probe_with_store": scorer.score(reference, str(probe.get("with_store_text") or "")),
            "probe_without_store": scorer.score(
                reference, str(probe.get("without_store_text") or "")
            ),
        }
        updated = dict(row)
        updated["scorer_version"] = scorer.version
        updated["semantic_scores"] = {
            key: {"label": value.label, "score": value.score, "version": value.version}
            for key, value in verdicts.items()
        }
        updated["direct_leak"] = (
            verdicts["direct"].leaks if str(row["arm"]).startswith("D-") else False
        )
        updated["agent_message_leak"] = verdicts["agent_messages"].leaks
        updated["final_leak"] = verdicts["final"].leaks
        updated["store_leak"] = verdicts["memory_writes"].leaks
        updated["post_episode_probe_leak"] = verdicts["probe_with_store"].leaks
        scored.append(updated)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in scored), encoding="utf-8"
    )
    out_manifest = out.with_name("leak_scored_manifest.json")
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    meta.update(
        {
            "semantic_scorer": scorer.version,
            # Pinning reproduces the paper scorer; calibration of a separate primary
            # evaluator is still a prerequisite for a new-method superiority claim.
            "semantic_scorer_reportable": False,
            "records_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
            "source_records_sha256": hashlib.sha256(records.read_bytes()).hexdigest(),
        }
    )
    out_manifest.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    typer.echo(f"wrote {out} and {out_manifest}")
