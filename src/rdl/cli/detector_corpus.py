"""``rdl graph-detector-corpus`` — freeze the detector-development corpus and its split.

A reanalysis phase, like ``graph-detector-recall``: it regenerates nothing and rescores
nothing. It reads committed shards and the run's own scoring cache, applies the
eligibility rules in ``eval.detector_corpus``, and writes two artefacts:

    DETECTOR_GENERATED_CORPUS.json   the examples, their provenance and the audits
    DETECTOR_ENGINEERING_SPLIT.json  which concepts fit and which are held out

**Why it is frozen before Detector v2 exists.** The split has to be immutable before
anyone looks at held-out performance, or "held out" means "held out until it was
inconvenient". ``--verify`` re-derives both artefacts from the same runs and compares the
content hashes, so a reviewer can check that the committed freeze is what this code
produces from that evidence rather than taking it on trust.

The raw shards are large and stay local (see `.gitignore`); the frozen artefacts are
small and are committed. That is why every source run is recorded with its shard-ledger
hash: the corpus can be checked against the evidence even on a box that does not hold it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from ..eval.detector_corpus import (
    ALLOWED_CHALLENGES,
    CollectionAudit,
    SourceRun,
    build_corpus,
    build_split,
    collect_examples,
    content_hash,
)
from ..studies.graph_leak.evidence import (
    atomic_json,
    read_nli_cache,
    read_shards,
    scorer_label_fn,
)
from .graph_common import option_value

__all__ = ["detector_corpus"]

CORPUS_FILENAME = "DETECTOR_GENERATED_CORPUS.json"
SPLIT_FILENAME = "DETECTOR_ENGINEERING_SPLIT.json"


def _ledger(run: Path) -> tuple[str | None, int, int]:
    path = run / "generations" / "SHARDS.json"
    if not path.exists():
        return None, 0, 0
    ledger = json.loads(path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest, int(ledger.get("n_shards", 0)), int(ledger.get("n_rows", 0))


def _source_for(run: Path) -> tuple[SourceRun, str]:
    manifest_path = run / "RUN_MANIFEST.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"no RUN_MANIFEST.json under {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not (run / "generations").is_dir():
        raise typer.BadParameter(
            f"no generations under {run}. The corpus is built from raw shards; restore the "
            "archived evidence from the run's raw_evidence_uri first."
        )
    scoring_path = run / "scores" / "SCORING.json"
    if not scoring_path.exists():
        raise typer.BadParameter(f"no {scoring_path}; run `rdl graph-score` first")
    scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
    version = str(scoring.get("scorer_version", ""))

    challenges = manifest.get("challenges") or []
    challenge = str(challenges[0] if isinstance(challenges, list) and challenges else challenges)
    if challenge not in ALLOWED_CHALLENGES:
        raise typer.BadParameter(
            f"{run.name} ran challenge '{challenge}', which is not eligible for the "
            f"detector corpus {list(ALLOWED_CHALLENGES)}. `split_clues` and `tool_reentry` "
            "are refusal-confounded and record the harness's own gold-derived text."
        )
    if manifest.get("retain_evaluation"):
        raise typer.BadParameter(
            f"{run.name} is a RETAIN run: a 'leak' there is a correct answer, and fitting "
            "a forget detector on correct answers is how over-blocking is trained in."
        )
    digest, n_shards, n_rows = _ledger(run)
    return (
        SourceRun(
            run_id=run.name,
            challenge=challenge,
            protocol=str(manifest.get("protocol", "")),
            scorer_version=version,
            release=str(manifest.get("raw_evidence_uri") or "") or None,
            shard_ledger_sha256=digest,
            n_shards=n_shards,
            n_rows=n_rows,
            forget_policy_fingerprint=manifest.get("forget_policy_fingerprint"),
            detector_version=(manifest.get("detector_calibration") or {}).get("calibration_id"),
        ),
        version,
    )


def detector_corpus(
    runs: list[Path] = typer.Option(
        ..., "--run", help="a runs/graph/<run-id> directory; repeat for each source run"
    ),
    output: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/detector_v2"),
        "--output",
        help="directory the two frozen artefacts are written to",
    ),
    verify: bool = typer.Option(
        False,
        "--verify",
        help="re-derive and compare against what is already frozen; write nothing",
    ),
) -> None:
    """Build (or verify) the frozen detector corpus and its engineering split."""
    runs = list(option_value(runs, []))
    output = Path(option_value(output, Path("data/cohorts/graph_unlearning_v1/detector_v2")))
    verify = bool(option_value(verify, False))
    if not runs:
        raise typer.BadParameter("at least one --run is required")

    audit = CollectionAudit()
    per_run = []
    policy_fingerprints: set[str] = set()
    for run in sorted(runs, key=lambda p: p.name):
        source, version = _source_for(run)
        cache = read_nli_cache(run / "scores" / "nli-cache.jsonl", version)
        if not cache:
            raise typer.BadParameter(
                f"no usable scoring cache at {run / 'scores' / 'nli-cache.jsonl'} for scorer "
                f"'{version}'. The corpus labels must be the run's OWN verdicts."
            )
        examples, audit = collect_examples(
            read_shards(run / "generations"),
            label=scorer_label_fn(cache, version),
            run_id=source.run_id,
            challenge=source.challenge,
            audit=audit,
        )
        if source.forget_policy_fingerprint:
            policy_fingerprints.add(str(source.forget_policy_fingerprint))
        per_run.append((source, examples))

    if len(policy_fingerprints) > 1:
        raise typer.BadParameter(
            f"the source runs disagree on the forget policy {sorted(policy_fingerprints)}. "
            "Pooling runs that forgot different things gives concepts whose examples come "
            "from different deployments, and a split over them is not a split."
        )

    corpus = build_corpus(
        per_run,
        audit=audit,
        registry_fingerprint=next(iter(policy_fingerprints), None),
    )
    split = build_split(corpus)

    corpus_path = output / CORPUS_FILENAME
    split_path = output / SPLIT_FILENAME
    if verify:
        problems: list[str] = []
        for path, rebuilt in ((corpus_path, corpus), (split_path, split)):
            if not path.exists():
                problems.append(f"{path} does not exist")
                continue
            frozen = json.loads(path.read_text(encoding="utf-8"))
            if frozen.get("content_sha256") != content_hash(frozen):
                problems.append(f"{path.name}: content_sha256 does not match its own content")
            if frozen.get("content_sha256") != rebuilt["content_sha256"]:
                problems.append(
                    f"{path.name}: frozen {frozen.get('content_sha256')} != rebuilt "
                    f"{rebuilt['content_sha256']}"
                )
        if problems:
            typer.echo(json.dumps({"verified": False, "problems": problems}, indent=2))
            raise typer.Exit(code=1)
        typer.echo(
            json.dumps(
                {
                    "verified": True,
                    "corpus_sha256": corpus["content_sha256"],
                    "split_sha256": split["content_sha256"],
                },
                indent=2,
            )
        )
        return

    atomic_json(corpus_path, corpus)
    atomic_json(split_path, split)
    typer.echo(
        json.dumps(
            {
                "n_examples": corpus["n_examples"],
                "n_concepts": corpus["n_concepts"],
                "n_gateable_concepts": corpus["n_gateable_concepts"],
                "development_concepts": split["development_concepts"],
                "heldout_concepts": split["heldout_concepts"],
                "audit_only_concepts": split["audit_only_concepts"],
                "corpus_sha256": corpus["content_sha256"],
                "split_sha256": split["content_sha256"],
            },
            indent=2,
        )
    )
    typer.echo(f"wrote {corpus_path} and {split_path}")
