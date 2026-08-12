"""``rdl graph-calibrate`` — choose the detector threshold once, then freeze it.

The threshold is the single knob that trades leakage against over-blocking, and until
now it was a number typed into a study file. A run could therefore claim
``detector.status: calibrated`` with nothing behind it, and the false-positive rate the
report gate is supposed to enforce had never been measured at all.

This command produces the artefact that makes the claim checkable:

    threshold                   the chosen operating point
    fpr / fnr                   measured on the FROZEN calibration cohorts
    recall                      1 - fnr, kept explicitly because both names are used
    calibration_cohort_sha256   fingerprints of the two cohorts, positives and negatives
    artifact_sha256             the artefact's own content hash, over everything above

**The split discipline.** Negatives are retain90 authors congruent to 1 mod 4, which no
other cohort uses; the retain-utility cohort draws from authors 0 mod 4. So the FPR is
held out from the retain questions the utility gate later scores — selecting a threshold
on the same authors it is then judged against would make the FPR an in-sample number and
the over-blocking cost would look smaller than it is.

Positives are held-out *questions* about the forget-policy authors. They cannot be
held-out authors: the detector's prototypes ARE those authors, so a positive about a
different author would not be a positive. The artefact records that limitation in
``positives_held_out_at`` rather than implying a stronger guarantee than it has.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from ..defenses.concept_registry import ConceptPolicy, ConceptRegistry
from ..defenses.semantic_detector import SemanticConceptDetector
from ..eval.detector_calibration import calibrate_threshold
from ..eval.tofu_data import TofuItem, load_items
from ..paths import repo_root
from ..studies.graph_leak.cohort import Cohort, CohortError, load_cohort, resolve_cohort
from ..studies.graph_leak.evidence import atomic_json
from .graph_common import DEFAULT_LAUNCH, load_config_or_fail, resolve_run_cohorts

__all__ = ["ARTIFACT_SCHEMA", "artifact_sha256", "calibrate_detector", "load_calibration"]

ARTIFACT_SCHEMA = "graph-detector-calibration-v1"


def artifact_sha256(payload: dict) -> str:
    """Content hash over everything except the hash field itself."""
    body = {k: v for k, v in payload.items() if k != "artifact_sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def load_calibration(path: Path) -> dict:
    """Read and verify a calibration artefact. Raises if it is absent or edited."""
    if not path.exists():
        raise CohortError(
            f"detector calibration artefact not found: {path}\n"
            "A run may not declare `detector.status: calibrated` without it. Produce one "
            "with `rdl graph-calibrate --write`."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != ARTIFACT_SCHEMA:
        raise CohortError(f"{path}: unknown calibration schema {payload.get('schema')!r}")
    declared = payload.get("artifact_sha256")
    actual = artifact_sha256(payload)
    if declared != actual:
        raise CohortError(
            f"{path}: artifact_sha256 does not match its contents "
            f"(declared {declared}, actual {actual}). The threshold, the FPR or the "
            "cohort fingerprints were edited after calibration."
        )
    if not payload.get("ok"):
        raise CohortError(
            f"{path}: calibration did not succeed — {payload.get('reason')}. "
            "The detector must be reported as diagnostic."
        )
    return payload


def _probe_texts(cohort: Cohort, items: list[TofuItem]) -> list[str]:
    """One probe per calibration item: the QUESTION only.

    Never the gold answer. The runtime registry is built from questions and the registry
    constructor refuses answers outright; calibrating on a signal the detector will never
    see at run time would pick a threshold for a different problem.
    """
    del cohort
    return [item.question for item in items]


def calibrate_detector(
    launch: Path = typer.Option(Path(DEFAULT_LAUNCH), "--launch"),
    profile: str | None = typer.Option(None, "--profile", help="override active_profile"),
    recall_floor: float = typer.Option(0.90, "--recall-floor", min=0.0, max=1.0),
    fpr_ceiling: float | None = typer.Option(
        None, "--fpr-ceiling", help="defaults to the study's evaluation.max_detector_fpr"
    ),
    output: Path | None = typer.Option(
        None, "--output", help="defaults to <cohort_dir>/DETECTOR_CALIBRATION.json"
    ),
    token: str | None = typer.Option(None, "--hf-token"),
    write: bool = typer.Option(False, "--write", help="write the artefact; otherwise print it"),
) -> None:
    """Select and freeze the detector threshold on the fixed calibration cohorts."""
    cfg = load_config_or_fail(launch, overrides=[f"active_profile={profile}"] if profile else None)
    root = repo_root()
    base = root / cfg.study.data.cohort_dir
    ceiling = cfg.study.evaluation.max_detector_fpr if fpr_ceiling is None else fpr_ceiling

    # The registry the detector will actually run with: the run's FORGET POLICY, not the
    # calibration positives. Calibrating a different registry from the one deployed would
    # measure a detector nobody runs.
    cohorts = resolve_run_cohorts(cfg, token=token)
    registry = ConceptRegistry.from_questions(
        [
            {
                "item_id": item.item_id,
                "concept_id": next(
                    e.concept_id for e in cohorts.policy.items if e.item_id == item.item_id
                ),
                "question": item.question,
            }
            for item in cohorts.policy_items
        ],
        policy=ConceptPolicy(
            allow_refusal=True,
            allow_persistent_write=False,
            allow_edge_release=False,
            allow_retrieval=False,
        ),
    )
    detector = SemanticConceptDetector(
        registry,
        threshold=cfg.study.detector.threshold,
        alias_weight=cfg.study.detector.alias_weight,
    )

    probes: dict[str, tuple[Cohort, list[TofuItem]]] = {}
    for role, manifest in (
        ("positives", cfg.study.data.calibration_positives_manifest),
        ("negatives", cfg.study.data.calibration_negatives_manifest),
    ):
        try:
            cohort = load_cohort(
                base / manifest,
                exclusions_path=None,
                require_frozen=cfg.study.data.require_frozen_hashes,
            )
            source, _ = load_items(
                dataset="tofu",
                split=cohort.dataset_config,
                n_items=None,
                token=token,
                revision=cohort.dataset_revision,
            )
            probes[role] = (cohort, resolve_cohort(cohort, source))
        except CohortError as exc:
            raise typer.BadParameter(str(exc)) from exc

    positive_cohort, positive_items = probes["positives"]
    negative_cohort, negative_items = probes["negatives"]

    # The two hard separation checks. Both are refusals, not warnings: a calibration set
    # that overlaps what it is later judged on produces an optimistic threshold and no
    # downstream check can detect it.
    overlap_items = sorted(set(negative_cohort.item_ids) & set(cohorts.evaluation.item_ids))
    if overlap_items:
        raise typer.BadParameter(
            f"calibration negatives share items {overlap_items[:5]} with the evaluation "
            "cohort. The false-positive rate would be measured in sample."
        )
    overlap_authors = sorted(set(negative_cohort.concept_ids) & set(cohorts.evaluation.concept_ids))
    if overlap_authors:
        raise typer.BadParameter(
            f"calibration negatives share authors {overlap_authors[:5]} with the "
            "evaluation cohort. Keep calibration authors separate from the engineering, "
            "retain and validation authors."
        )

    result = calibrate_threshold(
        detector,
        positives=_probe_texts(positive_cohort, positive_items),
        negatives=_probe_texts(negative_cohort, negative_items),
        recall_floor=recall_floor,
        fpr_ceiling=ceiling,
    )

    payload = {
        "schema": ARTIFACT_SCHEMA,
        "study_id": cfg.study.study_id,
        "detector_backend": cfg.study.detector.backend,
        "detector_alias_weight": cfg.study.detector.alias_weight,
        "calibration_id": result.calibration_id,
        "threshold": result.threshold,
        "ok": result.ok,
        "reason": result.reason,
        "recall": result.recall,
        # FNR is recorded explicitly rather than left to the reader: a missed forgotten
        # concept and a blocked retained one are different failures with different costs.
        "fnr": round(1.0 - result.recall, 6) if result.ok else None,
        "fpr": result.fpr if result.ok else None,
        "recall_floor": recall_floor,
        "fpr_ceiling": ceiling,
        "grid": list(result.grid),
        "sweep": list(result.rows),
        "n_positive": result.n_positive,
        "n_negative": result.n_negative,
        "registry_from": {
            "split": cohorts.policy.split,
            "fingerprint": cohorts.policy.fingerprint(),
            "n_concepts": len(cohorts.policy.concept_ids),
        },
        "calibration_cohort_sha256": {
            "positives": positive_cohort.fingerprint(),
            "negatives": negative_cohort.fingerprint(),
        },
        "calibration_cohorts": {
            "positives": {
                "split": positive_cohort.split,
                "dataset_config": positive_cohort.dataset_config,
                "dataset_revision": positive_cohort.dataset_revision,
                "n_items": len(positive_cohort.items),
                "n_concepts": len(positive_cohort.concept_ids),
            },
            "negatives": {
                "split": negative_cohort.split,
                "dataset_config": negative_cohort.dataset_config,
                "dataset_revision": negative_cohort.dataset_revision,
                "n_items": len(negative_cohort.items),
                "n_concepts": len(negative_cohort.concept_ids),
            },
        },
        # What is and is not held out, stated rather than implied.
        "positives_held_out_at": "question",
        "negatives_held_out_at": "author",
        "held_out_from_evaluation": True,
        "note": (
            "FPR is measured on retain90 authors that appear in no evaluation cohort, so "
            "it is held out at the author level. Recall is measured on held-out QUESTIONS "
            "about the forget-policy authors and cannot be held out at the author level: "
            "the detector's prototypes are those authors. Author-level held-out positives "
            "need a new unlearning checkpoint over a fresh concept split."
        ),
    }
    payload["artifact_sha256"] = artifact_sha256(payload)

    destination = output or (base / "DETECTOR_CALIBRATION.json")
    if write:
        atomic_json(destination, payload)
        typer.echo(f"wrote {destination}")
    else:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))

    if not result.ok:
        typer.echo(f"BLOCKING: {result.reason}", err=True)
        raise typer.Exit(code=1)
    typer.echo(
        f"threshold={result.threshold} recall={result.recall:.4f} "
        f"fpr={result.fpr:.4f} (ceiling {ceiling}) calibration_id={result.calibration_id}"
    )
