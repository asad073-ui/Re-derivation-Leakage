"""`detector.status: calibrated` must be backed by an artefact nobody can fake. GU-0027.

Before this, `status` was a string in a yaml file. The report gate refused to call a run
reportable unless it said `calibrated`, so the only thing standing between a diagnostic
detector and a "reportable" result was somebody typing a word — and the false-positive
rate the study preregisters a 10% ceiling for had never been measured at all.

Everything here runs offline against the committed frozen cohorts and the committed
artefact. Producing the artefact needs the dataset; checking it does not.
"""

from __future__ import annotations

import json

import pytest

from rdl.cli.calibrate_detector import ARTIFACT_SCHEMA, artifact_sha256, load_calibration
from rdl.eval.graph_utility import detector_fpr_gate
from rdl.graph.config import GraphDetectorConfig, load_graph_config
from rdl.paths import repo_root
from rdl.studies.graph_leak.cohort import CohortError, load_cohort, resolve_cohort
from rdl.studies.graph_leak.runner import GraphRunner

COHORTS = repo_root() / "data" / "cohorts" / "graph_unlearning_v1"
ARTIFACT = COHORTS / "DETECTOR_CALIBRATION.json"
LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"


@pytest.fixture
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


# ------------------------------------------------------------ the fixed dataset --


def test_the_calibration_cohorts_are_frozen_against_a_pinned_dataset_revision():
    for name in ("calibration_positives", "calibration_negatives"):
        cohort = load_cohort(COHORTS / f"{name}.json")
        assert cohort.frozen, name
        assert cohort.dataset_revision, name


def test_calibration_negatives_are_authors_no_evaluation_cohort_uses():
    """The FPR must be held out from the retain questions the utility gate scores.

    Selecting a threshold on the same retain authors it is later judged against would
    make the false-positive rate an in-sample quantity, and the over-blocking cost would
    read smaller than it is.
    """
    negatives = load_cohort(COHORTS / "calibration_negatives.json")
    for other in ("retain_utility", "engineering", "discovery", "smoke"):
        cohort = load_cohort(COHORTS / f"{other}.json")
        assert not set(negatives.concept_ids) & set(cohort.concept_ids), other
        assert not set(negatives.item_ids) & set(cohort.item_ids), other


def test_calibration_positives_are_held_out_questions_not_held_out_authors(artifact):
    """The limitation is recorded rather than implied.

    Positives cannot be author-disjoint from the forget policy: the detector's prototypes
    ARE those authors, so a positive about a different author would not be a positive.
    What CAN be held out is the question, and the artefact says exactly that instead of
    claiming a stronger guarantee.
    """
    positives = load_cohort(COHORTS / "calibration_positives.json")
    engineering = load_cohort(COHORTS / "engineering.json")
    assert not set(positives.item_ids) & set(engineering.item_ids)
    assert set(positives.concept_ids) == set(engineering.concept_ids)
    assert artifact["positives_held_out_at"] == "question"
    assert artifact["negatives_held_out_at"] == "author"


# ---------------------------------------------------------------- the artefact --


def test_the_artifact_records_everything_the_gate_needs(artifact):
    assert artifact["schema"] == ARTIFACT_SCHEMA
    assert artifact["ok"] is True
    for key in (
        "threshold",
        "fpr",
        "fnr",
        "recall",
        "calibration_id",
        "calibration_cohort_sha256",
        "artifact_sha256",
    ):
        assert artifact.get(key) is not None, key
    assert artifact["fnr"] == pytest.approx(1.0 - artifact["recall"], abs=1e-6)
    assert set(artifact["calibration_cohort_sha256"]) == {"positives", "negatives"}


def test_the_measured_fpr_is_under_the_studys_ceiling(artifact):
    ceiling = load_graph_config(LAUNCH).study.evaluation.max_detector_fpr
    assert artifact["fpr"] <= ceiling
    gate = detector_fpr_gate(artifact, ceiling=ceiling)
    assert gate["applicable"] and gate["within_ceiling"] and not gate["blocking"]


def test_the_cohort_fingerprints_in_the_artifact_are_the_committed_ones(artifact):
    assert artifact["calibration_cohort_sha256"]["positives"] == (
        load_cohort(COHORTS / "calibration_positives.json").fingerprint()
    )
    assert artifact["calibration_cohort_sha256"]["negatives"] == (
        load_cohort(COHORTS / "calibration_negatives.json").fingerprint()
    )


def test_an_edited_artifact_is_refused(tmp_path, artifact):
    """A threshold or an FPR changed after the fact must not load."""
    tampered = dict(artifact)
    tampered["fpr"] = 0.0
    path = tmp_path / "DETECTOR_CALIBRATION.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(CohortError, match="does not match its contents"):
        load_calibration(path)


def test_a_failed_calibration_is_refused(tmp_path, artifact):
    failed = {**artifact, "ok": False, "reason": "no threshold met the recall floor"}
    failed["artifact_sha256"] = artifact_sha256(failed)
    path = tmp_path / "DETECTOR_CALIBRATION.json"
    path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(CohortError, match="calibration did not succeed"):
        load_calibration(path)


def test_a_missing_artifact_is_refused(tmp_path):
    with pytest.raises(CohortError, match="not found"):
        load_calibration(tmp_path / "absent.json")


# ------------------------------------------------------------------ the config --


def test_calibrated_without_an_artifact_is_a_config_error():
    with pytest.raises(ValueError, match="calibration_artifact"):
        GraphDetectorConfig(status="calibrated")


def test_calibrated_without_a_calibration_id_is_a_config_error():
    with pytest.raises(ValueError, match="calibration_id"):
        GraphDetectorConfig(status="calibrated", calibration_artifact="somewhere.json")


def test_the_committed_study_id_matches_the_committed_artifact(artifact):
    cfg = load_graph_config(LAUNCH)
    assert cfg.study.detector.calibration_id == artifact["calibration_id"]
    assert (repo_root() / cfg.study.detector.calibration_artifact).resolve() == ARTIFACT.resolve()


# -------------------------------------------------------------------- the run --


def test_a_run_whose_forget_policy_the_artifact_does_not_cover_reports_diagnostic(
    tofu_items, graph_backend, tmp_path, artifact
):
    """The study says `calibrated`; this run's policy is the four-concept stub cohort.

    A threshold is calibrated FOR A REGISTRY. Inheriting the word `calibrated` here would
    attach a false-positive rate measured on twenty forget10 authors to a run guarding
    four invented ones. The run downgrades itself and records why, which is the honest
    outcome and the one the report gate then acts on.
    """
    cfg = load_graph_config(LAUNCH)
    assert cfg.study.detector.status == "calibrated"

    cohort = load_cohort(COHORTS / "cpu_stub.json").limited(4)
    manifest = GraphRunner(
        cfg=cfg,
        items=resolve_cohort(cohort, tofu_items),
        cohort=cohort,
        backend=graph_backend,
        output=tmp_path / "run",
        tokenizer_revision="stub",
    ).run()

    assert manifest["detector_status_declared"] == "calibrated"
    assert manifest["detector_status"] == "diagnostic"
    assert manifest["detector_calibration_covers_forget_policy"] is False
    assert "does not cover" in manifest["detector_calibration_note"]
    assert manifest["detector_calibration"] is None
    # And the fallback threshold is the study's, not the artefact's.
    assert cfg.study.detector.threshold != artifact["threshold"]
