"""``rdl graph-score`` — the scoring phase, run after generation has finished.

Separate process, separate GPU residency: the generator is gone by the time the NLI
evaluator loads. The output is one score row per trajectory, carrying the scorer version
so a later rescore under a different evaluator is a new file rather than an overwrite.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..eval.graph_leak import score_row
from ..eval.semantic import LeakKOfficialScorer, OfflineSemanticScorer
from ..studies.graph_leak.evidence import atomic_json, read_shards

__all__ = ["score_graph"]


def score_graph(
    run: Path = typer.Option(..., "--run", help="a runs/graph/<run-id> directory"),
    scorer: str = typer.Option(
        "offline", "--scorer", help="offline (CPU, diagnostic) or leakk (pinned NLI+ROUGE)"
    ),
    device: int = typer.Option(-1, "--device", help="NLI device index; -1 is CPU"),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Score raw generations into ``scores/semantic-scores.jsonl``."""
    generations = run / "generations"
    if not generations.is_dir():
        raise typer.BadParameter(f"no generations directory under {run}")

    engine: OfflineSemanticScorer | LeakKOfficialScorer
    if scorer == "offline":
        engine = OfflineSemanticScorer()
        version = engine.version
        reportable = False
    elif scorer == "leakk":
        engine = LeakKOfficialScorer(cache_path=run / "scores" / "nli-cache.jsonl", device=device)
        version = engine.version
        reportable = True
    else:
        raise typer.BadParameter("--scorer must be offline or leakk")

    def leaks(reference: str, candidate: str) -> bool:
        if not candidate or not candidate.strip():
            return False
        return engine.score(reference, candidate).leaks

    scores_dir = output or (run / "scores")
    scores_dir.mkdir(parents=True, exist_ok=True)
    path = scores_dir / "semantic-scores.jsonl"
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in read_shards(generations):
            fh.write(
                json.dumps(score_row(row, leaks, scorer_version=version), sort_keys=True) + "\n"
            )
            n += 1
    atomic_json(
        scores_dir / "SCORING.json",
        {
            "scorer": scorer,
            "scorer_version": version,
            "reportable": reportable,
            "n_rows": n,
            "note": (
                "the offline token-F1 scorer is deterministic and auditable but NOT the "
                "released semantic gate; a reportable semantic claim needs --scorer leakk"
            ),
        },
    )
    typer.echo(f"scored {n} trajectories -> {path}")
