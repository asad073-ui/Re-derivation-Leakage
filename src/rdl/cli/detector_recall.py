"""``rdl graph-detector-recall`` — how much of the leakage the detector could even see.

A pure reanalysis phase: it regenerates nothing and rescores nothing. It joins two things
the run already committed to evidence —

    the raw generations   every message, edge payload, stored node and final answer
    the scoring cache     the pinned NLI verdict for every (reference, candidate) pair

— and re-runs the run's own detector, at the run's own frozen threshold, over each of
those texts. The result is recall against exactly the leaks the leak surfaces counted.

**Why this is worth its own command.** ``DETECTOR_CALIBRATION.json`` reports recall 0.95
on held-out forget QUESTIONS. That bounds how well a request guard recognises "tell me
about author X". It says nothing about a paraphrase produced by the third agent in a
chain, and a graph defence's whole job is downstream of the question. If recall on
generated text is low, then a propagation defence and a node-local one will score
identically no matter how good the propagation is — there is nothing for either to
propagate — and the honest next move is to fix detection, not the graph.

The detector is deterministic (a 64-dimensional hashing backbone), so this runs on CPU
and produces the same numbers as the GPU box would.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..defenses.semantic_detector import SemanticConceptDetector
from ..eval.concept_recall import detector_recall_on_generated_leakage
from ..eval.tofu_data import load_items
from ..studies.graph_leak.arms import build_registry
from ..studies.graph_leak.cohort import CohortError, load_cohort, resolve_cohort
from ..studies.graph_leak.evidence import (
    atomic_json,
    read_nli_cache,
    read_shards,
    scorer_label_fn,
)
from .graph_common import option_value

__all__ = ["detector_recall"]


def detector_recall(
    run: Path = typer.Option(..., "--run", help="a runs/graph/<run-id> directory"),
    primary_arm: str = typer.Option("multi_agent_leak", "--primary-arm"),
    unrestricted: bool = typer.Option(
        False,
        "--unrestricted",
        help="count a fire on ANY concept as a catch, not only on the row's own concept",
    ),
    max_misses: int = typer.Option(50, "--max-misses", help="missed strings to record"),
) -> None:
    """Write DETECTOR_RECALL.json for a completed run."""
    primary_arm = str(option_value(primary_arm, "multi_agent_leak"))
    unrestricted = bool(option_value(unrestricted, False))
    max_misses = int(option_value(max_misses, 50))

    manifest_path = run / "RUN_MANIFEST.json"
    if not manifest_path.exists():
        raise typer.BadParameter(f"no RUN_MANIFEST.json under {run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Checked FIRST, before cohorts, shards or the scoring cache are touched: it is the
    # cheapest check and the one whose failure invalidates everything downstream.
    recorded = manifest.get("detector") or {}
    if "threshold" not in recorded:
        raise typer.BadParameter(
            f"{manifest_path} has no `detector` block, so the threshold this run actually "
            "generated at is unknown. Recall measured at a guessed threshold describes a "
            "detector that never ran. Re-run `rdl graph-run` on this revision, or measure "
            "recall on a run whose manifest records its detector."
        )
    if not (run / "generations").is_dir():
        raise typer.BadParameter(
            f"no generations under {run}. This command reads the raw shards; restore the "
            "archived evidence from the run's raw_evidence_uri first."
        )

    scoring_path = run / "scores" / "SCORING.json"
    if not scoring_path.exists():
        raise typer.BadParameter(f"no {scoring_path}; run `rdl graph-score` first")
    scoring = json.loads(scoring_path.read_text(encoding="utf-8"))
    version = str(scoring.get("scorer_version", ""))
    cache = read_nli_cache(run / "scores" / "nli-cache.jsonl", version)
    if not cache:
        raise typer.BadParameter(
            f"no usable scoring cache at {run / 'scores' / 'nli-cache.jsonl'} for scorer "
            f"'{version}'. Recall must be measured against the run's OWN leak labels; "
            "rescoring here would compare a detector against a different evaluator."
        )
    label = scorer_label_fn(cache, version)

    # The registry is the FORGET POLICY's, never the evaluation cohort's (GU-0027): the
    # detector's scope is what the deployment must withhold, not what it was asked.
    policy_path = run / "FORGET_POLICY_COHORT.json"
    if not policy_path.exists():
        raise typer.BadParameter(f"no FORGET_POLICY_COHORT.json under {run}")
    policy_cohort = load_cohort(policy_path, exclusions_path=None)
    try:
        source, _ = load_items(
            dataset=policy_cohort.dataset,
            split=policy_cohort.dataset_config,
            n_items=None,
            revision=policy_cohort.dataset_revision,
        )
        policy_items = resolve_cohort(policy_cohort, source)
    except (CohortError, OSError, ValueError) as exc:
        raise typer.BadParameter(
            f"could not load the forget-policy questions for {policy_cohort.split}: {exc}. "
            "The detector's scope is defined by those questions, so recall cannot be "
            "measured without them."
        ) from exc
    concept_of = {item.item_id: item.concept_id for item in policy_cohort.items}
    registry = build_registry(policy_items, concept_of=lambda i: concept_of[i])

    # THE RUN'S OWN DETECTOR, reconstructed from what the run recorded (GU-0035).
    #
    # This used to read `detector_calibration` and fall back to a literal 0.65 when it was
    # absent. A DIAGNOSTIC study has no calibration artefact by definition, so on exactly
    # the runs this command exists for, recall was measured at 0.65 against generations
    # produced at 0.90 — the operating-point inconsistency the threshold fix was supposed
    # to eliminate, reappearing one phase later and after the GPU had been paid for.
    #
    # `manifest["detector"]` is written before the first model call and is the authority.
    # A run too old to carry it is refused rather than defaulted (checked above): a recall
    # number at an unknown threshold is not a measurement.
    calibration = manifest.get("detector_calibration") or {}
    threshold = float(recorded["threshold"])
    detector = SemanticConceptDetector(
        registry,
        threshold=threshold,
        alias_weight=float(recorded.get("alias_weight", 1.0)),
        calibrated=bool(recorded.get("calibrated", False)),
        calibration_id=recorded.get("calibration_id"),
    )
    # The registry is rebuilt here from the forget-policy cohort, so it must come out
    # byte-identical to the one that ran. If it does not, the detector being measured is
    # not the detector that generated, and every recall number below is about a different
    # system — which is precisely the class of error this command is meant to expose.
    rebuilt = registry.fingerprint()
    expected = recorded.get("registry_fingerprint")
    if expected and rebuilt != expected:
        raise typer.BadParameter(
            f"rebuilt concept registry {rebuilt[:12]} does not match the one this run "
            f"used ({str(expected)[:12]}). Recall would be measured against a different "
            "scope than the run enforced. The usual cause is a changed forget-policy "
            "cohort or a detector/registry version bump since the run."
        )
    if detector.version != recorded.get("version", detector.version):
        raise typer.BadParameter(
            f"reconstructed detector '{detector.version}' does not match the run's "
            f"'{recorded.get('version')}'"
        )

    def detect(texts):
        results = detector.score_batch(list(texts))
        return [(bool(r.fired), tuple(r.forget_ids)) for r in results]

    report = detector_recall_on_generated_leakage(
        read_shards(run / "generations"),
        label=label,
        detect=detect,
        primary_arm=primary_arm,
        restrict_to_row_concept=not unrestricted,
        max_misses=max_misses,
    )
    # On a RETAIN cohort every "leaking" text is a CORRECT answer to a question the
    # system is supposed to answer, and the forget-policy detector is supposed to stay
    # silent. Recall 0.0 there is the desired result, not a failure, and saying so in the
    # artefact is cheaper than having a reader draw the opposite conclusion.
    retain = bool(manifest.get("retain_evaluation", False))
    report.update(
        {
            "run": run.name,
            "applicable": True,
            "retain_evaluation": retain,
            "reads_as": (
                "OVER-BLOCKING CHECK: these are retain questions, so a 'leaking' text is a "
                "correct answer and low recall / zero false-alarm rate is the DESIRED "
                "result, not a detection failure"
                if retain
                else "DETECTION CHECK: low recall here means a graph defence had nothing "
                "to propagate, and the fix is detection rather than the graph"
            ),
            "challenge": manifest.get("challenges"),
            "protocol": manifest.get("protocol"),
            "detector_version": detector.version,
            "threshold": threshold,
            # Carried so a reader — and the run gate — can check without reopening the
            # manifest that recall was measured on the detector that actually generated.
            "registry_fingerprint": rebuilt,
            "manifest_detector_version": recorded.get("version"),
            "manifest_threshold": recorded.get("threshold"),
            "operating_point_note": (
                "threshold, detector version and registry fingerprint are reconstructed "
                "from RUN_MANIFEST.json['detector'], not from a calibration artefact. A "
                "diagnostic run has no artefact, and defaulting there measured recall at "
                "an operating point the run never used (GU-0035)."
            ),
            "scorer_version": version,
            "n_cached_verdicts": len(cache),
            "calibration_recall_on_questions": calibration.get("recall"),
            "calibration_contrast_note": (
                "calibration recall is measured on held-out forget QUESTIONS; the recall "
                "above is measured on text the graph actually generated. A large gap "
                "means a defence's failure is a detection failure, not a propagation one."
            ),
        }
    )
    atomic_json(run / "DETECTOR_RECALL.json", report)
    typer.echo(
        f"detector recall on generated leakage ({primary_arm}): "
        f"{report['recall']} over {report['n_leaking']} leaking texts "
        f"({report['n_missed']} missed) -> {run / 'DETECTOR_RECALL.json'}"
    )
