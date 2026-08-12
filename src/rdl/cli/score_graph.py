"""``rdl graph-score`` — the scoring phase, run after generation has finished.

Separate process, separate GPU residency: the generator is gone by the time the NLI
evaluator loads. The output is one score row per trajectory, carrying the scorer version
so a later rescore under a different evaluator is a new file rather than an overwrite.

**Two passes over the shards** (GU-0024). The first pass collects every
``(reference, candidate)`` question the six leak surfaces will ask; those are
deduplicated, ROUGE-gated in bulk and pushed through the NLI model in batches. The second
pass writes the score rows, answering from the resulting table.

The previous single pass called the scorer once per surface per string per row, so the
classifier ran at batch size one — tens of thousands of separate forward passes for a
50x32 run, each with its own tokenisation and host/device round trip, on a model that
processes 64 pairs in barely more time than one. Two passes over the shards cost a second
read of a few hundred MB of JSONL; they save hours.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..eval.graph_leak import candidate_pairs, score_row
from ..eval.semantic import LeakKOfficialScorer, OfflineSemanticScorer, Pair, SemanticVerdict
from ..studies.graph_leak.evidence import atomic_json, read_shards

__all__ = ["score_graph"]


def score_graph(
    run: Path = typer.Option(..., "--run", help="a runs/graph/<run-id> directory"),
    scorer: str = typer.Option(
        "offline", "--scorer", help="offline (CPU, diagnostic) or leakk (pinned NLI+ROUGE)"
    ),
    device: int = typer.Option(-1, "--device", help="NLI device index; -1 is CPU"),
    batch_size: int = typer.Option(
        64, "--batch-size", help="NLI pairs per forward pass; 64 is sane for a 24 GB card"
    ),
    checkpoint_every: int = typer.Option(
        512, "--checkpoint-every", help="verdicts buffered before the NLI cache is appended to"
    ),
    output: Path | None = typer.Option(None, "--output"),
) -> None:
    """Score raw generations into ``scores/semantic-scores.jsonl``."""
    generations = run / "generations"
    if not generations.is_dir():
        raise typer.BadParameter(f"no generations directory under {run}")
    if batch_size < 1:
        raise typer.BadParameter("--batch-size must be >= 1")

    engine: OfflineSemanticScorer | LeakKOfficialScorer
    if scorer == "offline":
        engine = OfflineSemanticScorer()
        reportable = False
    elif scorer == "leakk":
        engine = LeakKOfficialScorer(
            cache_path=run / "scores" / "nli-cache.jsonl",
            device=device,
            batch_size=batch_size,
            checkpoint_every=checkpoint_every,
        )
        reportable = True
    else:
        raise typer.BadParameter("--scorer must be offline or leakk")
    version = engine.version

    # Pass 1: what has to be judged at all.
    pairs: dict[Pair, None] = {}
    n_questions = 0
    for row in read_shards(generations):
        for pair in candidate_pairs(row):
            n_questions += 1
            pairs.setdefault(pair, None)
    typer.echo(
        f"{n_questions} surface questions over {len(pairs)} distinct (reference, candidate) "
        f"pairs; judging with {scorer}"
    )
    verdicts: dict[Pair, SemanticVerdict] = engine.score_batch(pairs)

    # Pass 2: the score rows. Any pair the table somehow misses falls back to a direct
    # call rather than being silently reported as clean.
    misses = 0

    def leaks(reference: str, candidate: str) -> bool:
        nonlocal misses
        if not candidate or not candidate.strip():
            return False
        verdict = verdicts.get((reference, candidate))
        if verdict is None:
            misses += 1
            verdict = engine.score(reference, candidate)
        return verdict.leaks

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
    if isinstance(engine, LeakKOfficialScorer):
        engine.flush_cache()

    atomic_json(
        scores_dir / "SCORING.json",
        {
            "scorer": scorer,
            "scorer_version": version,
            "reportable": reportable,
            "n_rows": n,
            "n_surface_questions": n_questions,
            "n_distinct_pairs": len(pairs),
            # Nonzero means `candidate_pairs` and `surface_flags` have drifted apart:
            # the run is still correct, it just fell back to unbatched scoring for those.
            "n_batch_misses": misses,
            **engine.stats(),
            "note": (
                "the offline token-F1 scorer is deterministic and auditable but NOT the "
                "released semantic gate; a reportable semantic claim needs --scorer leakk"
            ),
        },
    )
    typer.echo(f"scored {n} trajectories -> {path}")
