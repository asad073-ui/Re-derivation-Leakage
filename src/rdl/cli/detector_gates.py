"""``rdl graph-detector-gates`` — does this detector deserve GPU time?

Reads the frozen corpus and split (GU-0031), builds the run's registry and detector,
selects a threshold on the DEVELOPMENT concepts, then scores the held-out concepts once
and writes ``DETECTOR_V2_GATES.json``.

Everything is CPU and offline except the TOFU questions, which come from the pinned
dataset revision the cohorts name. Nothing is generated and nothing is rescored.

The command exits non-zero when a gate fails. That is the point: it is meant to be the
last thing run before an instance is rented, and a failing gate means the comparison the
instance would buy cannot distinguish a propagation failure from a detection failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..defenses.concept_registry import normalise_scope_text
from ..defenses.semantic_detector import SemanticConceptDetector
from ..eval.detector_corpus import ALLOWED_ARM, normalise_corpus_text
from ..eval.detector_gates import evaluate_detector_gates, select_threshold
from ..eval.tofu_data import load_items
from ..studies.graph_leak.arms import build_registry
from ..studies.graph_leak.cohort import CohortError, load_cohort, resolve_cohort
from ..studies.graph_leak.evidence import (
    atomic_json,
    read_nli_cache,
    read_shards,
    scorer_label_fn,
)
from .detector_corpus import CORPUS_FILENAME, SPLIT_FILENAME
from .graph_common import option_value

__all__ = ["detector_gates"]

GATES_FILENAME = "DETECTOR_V2_GATES.json"
DEFAULT_GRID = tuple(round(0.20 + 0.025 * i, 4) for i in range(29))  # 0.200 .. 0.900


def _policy_items(cohort_path: Path):
    cohort = load_cohort(cohort_path, exclusions_path=None)
    source, _ = load_items(
        dataset=cohort.dataset,
        split=cohort.dataset_config,
        n_items=None,
        revision=cohort.dataset_revision,
    )
    return cohort, resolve_cohort(cohort, source)


def _generated_clean_texts(runs: list[Path], limit: int) -> tuple[list[str], list[str]]:
    """Unguarded-arm text the run's own scorer judged CLEAN, for the in-situ FPR.

    The same eligibility rule as the corpus: the unguarded arm only. A guarded arm's clean
    text is clean partly because the guard removed the rest of it.
    """
    texts: list[str] = []
    used: list[str] = []
    for run in runs:
        scoring_path = run / "scores" / "SCORING.json"
        if not scoring_path.exists():
            continue
        version = str(
            json.loads(scoring_path.read_text(encoding="utf-8")).get("scorer_version", "")
        )
        cache = read_nli_cache(run / "scores" / "nli-cache.jsonl", version)
        if not cache:
            continue
        label = scorer_label_fn(cache, version)
        used.append(run.name)
        for row in read_shards(run / "generations"):
            if str(row.get("arm", "")) != ALLOWED_ARM:
                continue
            reference = str(row.get("reference_answer", ""))
            for text in [
                str(t) for t in (row.get("raw_outputs", {}) or {}).get("agent_messages", []) or []
            ]:
                if not text.strip():
                    continue
                if label(reference, text) is False:
                    texts.append(text)
                if len(texts) >= limit:
                    return texts, used
    return texts, used


def detector_gates(
    corpus_dir: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/detector_v2"),
        "--corpus-dir",
        help="directory holding the frozen corpus and split",
    ),
    policy_cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/discovery.json"),
        "--policy-cohort",
        help="the forget-policy cohort the registry is built from",
    ),
    negatives_cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/calibration_negatives.json"),
        "--negatives-cohort",
        help="held-out retain90 questions the detector must stay silent on",
    ),
    runs: list[Path] = typer.Option(
        [], "--run", help="source run(s) supplying generated CLEAN text for the in-situ FPR"
    ),
    max_clean: int = typer.Option(2000, "--max-clean", help="cap on generated clean texts"),
    threshold: float | None = typer.Option(
        None, "--threshold", help="skip selection and score at this operating point"
    ),
    output: Path | None = typer.Option(None, "--output", help="defaults to <corpus-dir>/gates"),
) -> None:
    """Select a threshold on development concepts and score the held-out half once."""
    corpus_dir = Path(
        option_value(corpus_dir, Path("data/cohorts/graph_unlearning_v1/detector_v2"))
    )
    policy_cohort = Path(
        option_value(policy_cohort, Path("data/cohorts/graph_unlearning_v1/discovery.json"))
    )
    negatives_cohort = Path(
        option_value(
            negatives_cohort, Path("data/cohorts/graph_unlearning_v1/calibration_negatives.json")
        )
    )
    runs = list(option_value(runs, []))
    max_clean = int(option_value(max_clean, 2000))
    threshold = option_value(threshold, None)
    output = option_value(output, None)

    corpus_path = corpus_dir / CORPUS_FILENAME
    split_path = corpus_dir / SPLIT_FILENAME
    for path in (corpus_path, split_path):
        if not path.exists():
            raise typer.BadParameter(
                f"{path} is missing. Freeze the corpus first: `rdl graph-detector-corpus`."
            )
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split["corpus_sha256"] != corpus["content_sha256"]:
        raise typer.BadParameter(
            "the split was cut from a different corpus than the one on disk. Re-freeze "
            "both together rather than scoring a split against examples it never saw."
        )

    try:
        policy, policy_items = _policy_items(policy_cohort)
        _negatives, negative_items = _policy_items(negatives_cohort)
    except (CohortError, OSError, ValueError) as exc:
        raise typer.BadParameter(f"could not load a cohort's questions: {exc}") from exc

    concept_of = {item.item_id: item.concept_id for item in policy.items}
    registry = build_registry(policy_items, concept_of=lambda i: concept_of[i])

    # The gold answers are loaded HERE, in an offline audit, and never reach the registry.
    # The check is that no scope prototype contains one — which is what GU-0005 promises
    # and what nothing until now actually verified against real answers.
    gold = [normalise_corpus_text(item.answer) for item in policy_items if item.answer]
    prototypes = [normalise_corpus_text(p) for c in registry.concepts() for p in c.scope_prototypes]
    gold_hits = sum(1 for g in gold if g and any(g in p for p in prototypes))

    detector = SemanticConceptDetector(registry, threshold=0.65)
    dev = set(split["development_concepts"])
    dev_rows = [e for e in corpus["examples"] if e["concept_id"] in dev]
    negative_texts = [item.question for item in negative_items]

    if threshold is None:
        dev_scores = [r.per_concept for r in detector.score_batch([r["text"] for r in dev_rows])]
        negative_scores = [r.per_concept for r in detector.score_batch(negative_texts)]
        sweep = select_threshold(
            dev_examples=dev_rows,
            dev_scores=dev_scores,
            negative_scores=negative_scores,
            grid=DEFAULT_GRID,
            max_fpr=0.10,
        )
        chosen = sweep["selected_threshold"]
    else:
        sweep = {"selected_threshold": float(threshold), "note": "supplied on the command line"}
        chosen = float(threshold)

    if chosen is None:
        # No operating point holds the FPR ceiling with any recall. Score at the ceiling's
        # strictest end anyway so the artefact carries numbers rather than a null.
        chosen = 0.65
        sweep["fallback_threshold"] = chosen

    scored = detector.with_threshold(float(chosen))

    def detect(texts):
        return [(bool(r.fired), tuple(r.forget_ids)) for r in scored.score_batch(list(texts))]

    clean_texts, clean_runs = _generated_clean_texts(runs, max_clean)
    report = evaluate_detector_gates(
        corpus=corpus,
        split=split,
        detect=detect,
        retain_negative_texts=negative_texts,
        generated_clean_texts=clean_texts,
        gold_answers_in_registry=gold_hits,
        threshold=float(chosen),
        detector_version=scored.version,
        registry_fingerprint=registry.fingerprint(),
    )
    report["threshold_selection"] = sweep
    report["registry_version"] = registry.version
    report["rejected_ambiguous_aliases"] = {
        k: list(v) for k, v in sorted(registry.rejected_aliases.items())
    }
    report["clean_text_runs"] = clean_runs
    report["policy_cohort"] = {"split": policy.split, "fingerprint": policy.fingerprint()}
    # The ceiling any lexical channel can reach on this corpus: the share of leaking
    # examples that contain ANY token of their own concept's aliases. If the primary gate
    # is above this number, no alias work can clear it and the next move is a different
    # detection channel, not more aliases.
    alias_tokens = {
        c.forget_id: {t for a in c.aliases for t in normalise_scope_text(a).split()}
        for c in registry.concepts()
    }
    lexical = [
        bool(alias_tokens.get(e["concept_id"], set()) & set(e["normalized_text"].split()))
        for e in corpus["examples"]
    ]
    per_concept: dict[str, list[int]] = {}
    for example, hit in zip(corpus["examples"], lexical, strict=True):
        bucket = per_concept.setdefault(example["concept_id"], [0, 0])
        bucket[1] += 1
        bucket[0] += int(hit)
    report["lexical_ceiling"] = {
        "micro": sum(lexical) / len(lexical) if lexical else None,
        "macro": (
            sum(h / n for h, n in per_concept.values()) / len(per_concept) if per_concept else None
        ),
        "definition": (
            "share of leaking examples containing at least one token of their own "
            "concept's aliases — the highest recall ANY alias channel could reach here"
        ),
    }

    out_dir = output or (corpus_dir / "gates")
    atomic_json(out_dir / GATES_FILENAME, report)
    typer.echo(
        json.dumps(
            {
                "threshold": report["threshold"],
                "detector_version": report["detector_version"],
                "all_gates_passed": report["all_gates_passed"],
                "failed_gates": report["failed_gates"],
                "heldout_micro_recall": report["recall"]["heldout"]["micro_recall_correct_concept"],
                "heldout_macro_recall": report["recall"]["heldout"]["macro_recall_correct_concept"],
                "lexical_ceiling_micro": report["lexical_ceiling"]["micro"],
            },
            indent=2,
        )
    )
    typer.echo(f"wrote {out_dir / GATES_FILENAME}")
    if not report["all_gates_passed"]:
        typer.echo(
            "GATES FAILED — this detector does not yet justify GPU time. A failing recall "
            "gate means a graph-versus-node-local comparison cannot distinguish a "
            "propagation failure from a detection one."
        )
        raise typer.Exit(code=1)
