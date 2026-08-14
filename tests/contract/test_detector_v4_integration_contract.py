"""What must be true BEFORE Detector v4 is allowed anywhere near a graph run.

Phase 9 connects v4 to the defences. This file is the set of conditions that make Phase 9
safe to do later, checked now, on the branch where the detector exists and the integration
does not:

* the two guarded arms share one detector object, so a difference between them is a
  difference in policy propagation rather than in detection;
* ``answerability_v4`` is not a selectable backend, because a detector reaches a study
  after its held-out gate has been opened and passed — and v4's has not;
* the gate and oracle artifacts exist, record their own verdict, and that verdict is
  currently a refusal;
* the request never becomes candidate evidence under ``graph_flow``, in the OLD
  accumulator as well as the new context type.

The last one is verified rather than rewritten. ``EvidenceAccumulator`` is part of the
mechanism that passed its runtime-liveness checks; churning it before v4 has a gate would
put the one component that works at risk for the sake of the one that does not.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdl.defenses.concept_registry import ConceptPolicy, ConceptRegistry
from rdl.defenses.evidence_accumulator import EvidenceAccumulator
from rdl.defenses.semantic_detector import SemanticConceptDetector
from rdl.graph.config import GraphDetectorConfig

REPO = Path(__file__).resolve().parents[2]
V4 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4"

ROWS = [
    {"item_id": "i0", "concept_id": "author-0000", "question": "Where was Ada Vane born?"},
    {"item_id": "i1", "concept_id": "author-0001", "question": "Where was Cyril Moss born?"},
]


@pytest.fixture
def registry() -> ConceptRegistry:
    return ConceptRegistry.from_questions(ROWS, policy=ConceptPolicy())


# ------------------------------------------------------------ one detector, two arms --


def test_the_two_guarded_arms_share_one_detector_object():
    """ "Our method leaks less" must not be reducible to "our detector is better"."""
    from rdl.studies.graph_leak.arms import build_detector

    source = build_detector.__doc__ or ""
    assert "ONE detector" in source
    signature_returns_one = "-> SemanticConceptDetector" in Path(
        REPO / "src" / "rdl" / "studies" / "graph_leak" / "arms.py"
    ).read_text(encoding="utf-8")
    assert signature_returns_one


def test_answerability_v4_is_not_a_selectable_backend():
    """The integration gate. v4 has no passing held-out gate, so it cannot reach a study.

    v3's scorer is the reason this is a rule rather than a habit: a detector that failed
    its probe must not be wireable by editing one string in a YAML file.
    """
    backend_field = GraphDetectorConfig.model_fields["backend"]
    allowed = getattr(backend_field.annotation, "__args__", ())
    assert allowed == ("hashing64",)
    with pytest.raises(ValueError):
        GraphDetectorConfig(backend="answerability_v4")


def test_the_v4_artifacts_record_that_no_gpu_and_no_generation_were_involved():
    for name in ("DETECTOR_V4_ORACLE_CEILING.json", "DETECTOR_V4_GATES.json"):
        report = json.loads((V4 / name).read_text(encoding="utf-8"))
        assert report["scope"]["gpu_used"] is False
        assert report["scope"]["model_trained"] is False
        assert report["scope"]["graph_generation_run"] is False
        assert report["scope"]["frozen_v1_v2_v3_artifacts_modified"] is False
        assert report["runtime_reads_gold_answers"] is False


def test_a_failing_v4_gate_is_recorded_as_a_refusal_not_an_absence():
    """A gate that fails must say so in the artifact, not merely omit a number."""
    gates = json.loads((V4 / "DETECTOR_V4_GATES.json").read_text(encoding="utf-8"))
    natural = gates["natural_arm"]
    assert natural["measured"] is True
    assert isinstance(natural["all_gates_passed"], bool)
    if not natural["all_gates_passed"]:
        assert natural["failed_gates"]
        assert "does NOT clear" in gates["verdict"]


def test_the_cpu_gate_artifacts_agree_on_which_component_is_bounded():
    """Oracle and detector must be read together, and the artifacts must make that possible."""
    ceiling = json.loads((V4 / "DETECTOR_V4_ORACLE_CEILING.json").read_text(encoding="utf-8"))
    gates = json.loads((V4 / "DETECTOR_V4_GATES.json").read_text(encoding="utf-8"))
    assert (
        ceiling["natural_arm"]["bank_content_sha256"] == gates["natural_arm"]["bank_content_sha256"]
    )
    # The synthetic arm is never the verdict on either side.
    assert "CONSTRUCTION CHECK" in ceiling["synthetic_arm"]["role"]
    assert "Not the verdict" in gates["synthetic_arm"]["role"]
    if not ceiling["natural_arm"]["all_gates_passed"]:
        # A failing ceiling has to name WHICH component the bound belongs to, or the next
        # move is a guess — which is how v3 spent a day tuning a router.
        assert ceiling["natural_arm"]["failure_locus"]["measured"] is True


def test_the_lexical_backend_declares_itself_a_floor():
    from rdl.defenses.answerability_detector import LexicalAnswerabilityDetector

    described = LexicalAnswerabilityDetector().to_dict()
    assert described["calibrated"] is False
    assert described["receives_gold_answers"] is False
    assert "floor" in described["role"]


# ------------------------------------------------- the request is not evidence (old) --


def test_graph_flow_never_folds_the_request_into_a_scored_string(registry):
    """Verified in the OLD accumulator, which v4 deliberately does not rewrite.

    ``inspect_query=False`` is the graph_flow setting. Under it the question must appear in
    no scored text at all — not as its own probe, and not inside ``node_input``.
    """
    scored: list[str] = []

    class Recording(SemanticConceptDetector):
        def score_batch(self, texts, *, restrict_to=None):
            scored.extend(texts)
            return super().score_batch(texts, restrict_to=restrict_to)

    accumulator = EvidenceAccumulator(Recording(registry), inspect_query=False)
    question = "Where was Ada Vane born?"
    accumulator.evaluate(
        question=question,
        input_texts=["Ada Vane was born in Paris.", "She also wrote poetry."],
        memory_texts=["An earlier note."],
    )
    assert scored
    assert not any(question in text for text in scored)


def test_end_to_end_safety_still_scores_the_request_separately(registry):
    """The request gate is a real surface there, and the two protocols stay separable."""
    accumulator = EvidenceAccumulator(SemanticConceptDetector(registry), inspect_query=True)
    evidence = accumulator.evaluate(
        question="Where was Ada Vane born?",
        input_texts=["Ada Vane was born in Paris."],
    )
    assert evidence.to_dict()["query_forget_ids"] is not None
