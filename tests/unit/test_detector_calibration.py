"""Threshold selection is a development-set decision, and it is allowed to fail."""

from __future__ import annotations

from itertools import pairwise

from rdl.defenses.concept_registry import ConceptRegistry
from rdl.defenses.semantic_detector import SemanticConceptDetector
from rdl.eval.detector_calibration import calibrate_threshold, sweep


def _detector() -> SemanticConceptDetector:
    registry = ConceptRegistry.from_questions(
        [
            {
                "item_id": "x-0",
                "concept_id": "author-17",
                "question": "Who was Amara Rossi and what did she write?",
            }
        ]
    )
    return SemanticConceptDetector(registry, threshold=0.55)


POSITIVES = [
    "Who was Amara Rossi and what did she write?",
    "Tell me about Amara Rossi.",
    "Amara Rossi wrote several novels.",
    "Give details on Amara Rossi.",
]
NEGATIVES = [
    "What is the boiling point of water?",
    "Explain how photosynthesis works.",
    "Who painted the ceiling of the Sistine Chapel?",
    "When did the Second World War end?",
]


def test_sweep_reports_recall_and_fpr_per_threshold():
    rows = sweep(_detector(), positives=POSITIVES, negatives=NEGATIVES)
    assert {"threshold", "recall", "fpr", "youden"} == set(rows[0])
    # Recall can only fall as the threshold rises.
    recalls = [r["recall"] for r in rows]
    assert all(b <= a + 1e-9 for a, b in pairwise(recalls))


def test_calibration_picks_the_least_trigger_happy_feasible_threshold():
    result = calibrate_threshold(
        _detector(), positives=POSITIVES, negatives=NEGATIVES, recall_floor=0.75, fpr_ceiling=0.25
    )
    assert result.ok
    assert result.threshold is not None
    assert result.recall >= 0.75
    assert result.fpr <= 0.25
    feasible = [r["threshold"] for r in result.rows if r["recall"] >= 0.75 and r["fpr"] <= 0.25]
    assert result.threshold == max(feasible)


def test_an_impossible_requirement_is_reported_not_papered_over():
    result = calibrate_threshold(
        _detector(), positives=POSITIVES, negatives=POSITIVES, recall_floor=0.99, fpr_ceiling=0.0
    )
    assert not result.ok
    assert result.threshold is None
    assert "diagnostic rather than calibrated" in result.reason


def test_the_calibration_id_is_content_addressed():
    a = calibrate_threshold(_detector(), positives=POSITIVES, negatives=NEGATIVES)
    b = calibrate_threshold(_detector(), positives=POSITIVES, negatives=NEGATIVES)
    assert a.calibration_id == b.calibration_id
    c = calibrate_threshold(_detector(), positives=POSITIVES[:2], negatives=NEGATIVES)
    assert c.calibration_id != a.calibration_id


def test_a_calibrated_detector_says_so_in_its_version():
    detector = _detector()
    assert ":diagnostic" in detector.version
    calibrated = detector.with_threshold(0.6, calibration_id="abc123")
    assert ":calibrated:abc123" in calibrated.version
    assert calibrated.threshold == 0.6
    assert detector.threshold == 0.55, "the original must not be mutated"


def test_the_calibration_record_states_where_it_was_selected():
    result = calibrate_threshold(_detector(), positives=POSITIVES, negatives=NEGATIVES)
    assert result.to_dict()["selected_on"] == "development concepts only"
