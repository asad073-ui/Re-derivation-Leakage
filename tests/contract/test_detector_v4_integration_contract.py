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

v4.1 (GU-0037) adds the conditions that make an RTX rental defensible, and they are here
for the same reason: each one is a claim about the repository that a document could assert
and only a test can hold.

* the v4 "oracle ceiling" is renamed a lexical baseline, beside an unedited original;
* the v4 held-out data is marked engineering-only, against the bank's own hash;
* the fresh final gate bank is pre-registered under seeds that are not the study's;
* the Goal A gate table no longer contains the quantity that charged a wrong answer
  attempt as a false alarm;
* the cross-encoder backend exists, and is still not selectable;
* one gate implementation scores both backends, and blocks until Goal A labels exist.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rdl.defenses.concept_registry import ConceptPolicy, ConceptRegistry
from rdl.defenses.evidence_accumulator import EvidenceAccumulator
from rdl.defenses.semantic_detector import SemanticConceptDetector
from rdl.eval.detector_v4_1 import GOAL_A_GATES
from rdl.graph.config import GraphDetectorConfig

REPO = Path(__file__).resolve().parents[2]
V4 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4"
V4_1 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_1"

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


# ------------------------------------------------------------- the v4.1 correction --


def test_the_v4_ceiling_artifact_is_not_edited_and_is_corrected_beside_it():
    """GU-0037 D1/D6. The negative result stands; what changes is what it bounds, and the
    correction carries the original's hash so the two cannot drift apart."""
    correction = json.loads(
        (V4_1 / "DETECTOR_V4_1_CEILING_CORRECTION.json").read_text(encoding="utf-8")
    )
    original = (V4 / "DETECTOR_V4_ORACLE_CEILING.json").read_bytes()
    assert correction["corrects_file_sha256"] == hashlib.sha256(original).hexdigest()
    assert correction["original_is_edited"] is False
    assert correction["renamed_to"] == "answer-token-overlap baseline"
    assert correction["must_not_be_called"] == "answerability ceiling"
    # And the refutation is the artifact's own numbers, not an opinion: the answer-FREE
    # detector beat the answer-AWARE "ceiling" on the same gate half.
    contradiction = correction["the_internal_contradiction"]
    assert (
        contradiction["answer_free_lexical_micro_recall"]
        > contradiction["answer_aware_oracle_micro_recall"]
    )


def test_the_v4_heldout_data_is_marked_engineering_only():
    """A held-out set is a one-shot instrument, and this one has been read twice."""
    decision = json.loads((V4_1 / "V4_1_DECISION.json").read_text(encoding="utf-8"))
    engineering = decision["engineering_only"]
    assert "heldout" in engineering["partitions_marked"]
    assert "the final detector gate" in engineering["not_usable_for"]
    bank = json.loads((V4 / "DETECTOR_V4_NATURAL_BANK.json").read_text(encoding="utf-8"))
    assert engineering["bank_content_sha256"] == bank["content_sha256"]
    assert decision["scope"]["gpu_used"] is False


def test_the_final_gate_bank_is_preregistered_with_new_seeds():
    """A 'fresh' bank drawn under the study's own base_seed would redraw the trajectories
    the engineering-only bank already contains."""
    manifest = json.loads((V4_1 / "FINAL_GATE_BANK_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "PRE-REGISTERED, NOT YET GENERATED"
    assert manifest["seeds"]["base_seed"] != 1729
    assert manifest["arms"]["natural"] == "multi_agent_leak"
    assert manifest["opening_rule"]["opened"] == "exactly once, at the frozen threshold"
    assert set(manifest["gates"]) == set(GOAL_A_GATES)


def test_the_goal_a_gates_do_not_charge_a_wrong_answer_attempt_as_a_false_alarm():
    """The correction, as a property of the gate table rather than of a document."""
    assert "generated_clean_fpr" not in GOAL_A_GATES
    assert GOAL_A_GATES["protected_nonanswer_fpr"] == ("<=", 0.10)


def test_the_cross_encoder_backend_exists_and_is_still_not_selectable():
    """Condition 9 of the twelve, and condition 12's other half: the backend is real, and
    a real backend is exactly the thing that must not be wireable by editing a YAML string
    before its gate has been opened."""
    from rdl.defenses.cross_encoder_answerability import CrossEncoderAnswerabilityDetector
    from rdl.defenses.detector_protocol import ConceptDetector

    assert issubclass(CrossEncoderAnswerabilityDetector, object)
    assert hasattr(CrossEncoderAnswerabilityDetector, "score_batch")
    assert "cross_encoder" in CrossEncoderAnswerabilityDetector.backend
    # The protocol is structural, so conformance is checked on an instance in
    # tests/unit/test_cross_encoder_answerability.py; here we only need the class to be
    # importable without torch or a network, and the config to still refuse it.
    assert ConceptDetector is not None
    allowed = getattr(GraphDetectorConfig.model_fields["backend"].annotation, "__args__", ())
    assert allowed == ("hashing64",)


def test_the_gate_command_can_evaluate_either_backend():
    """Condition 10. One gate implementation scores both, or the comparison between the
    floor and the checkpoint is a comparison between two pieces of code."""
    from rdl.cli.detector_v4_gates import BACKENDS, build_backend

    assert BACKENDS == ("lexical", "cross_encoder")
    assert build_backend("lexical", None).backend == "answerability_v4_lexical"
    with pytest.raises(Exception, match="model-artifact"):
        build_backend("cross_encoder", None)


def test_the_gate_command_cannot_overwrite_the_frozen_v4_artifact(tmp_path):
    """D6. GU-0036 cites DETECTOR_V4_GATES.json by name, and this command now reports
    against different denominators; a default output that clobbered it would silently
    rewrite the evidence a dated decision rests on."""
    from typer.testing import CliRunner

    from rdl.cli.__main__ import app
    from rdl.cli.detector_v4_gates import GATES_FILENAME, V4_1_GATES_FILENAME

    assert V4_1_GATES_FILENAME != GATES_FILENAME
    result = CliRunner().invoke(
        app,
        [
            "graph-detector-v4-gates",
            "--data-dir",
            str(V4),
            "--output",
            str(V4 / GATES_FILENAME),
            "--v4-1-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "refusing to overwrite" in result.output


def test_the_gate_command_blocks_until_goal_a_labels_exist(tmp_path):
    """ "We did not measure it" and "it was fine" must not exit the same way.

    The authority is resolved first and comes back empty here — neither the v4.1 human
    adjudication nor the v4.2 model one exists in an empty directory — and an arm with no
    label source blocks rather than falling back to the NLI label.
    """
    from rdl.cli.detector_v4_gates import _goal_a_arm, build_backend, resolve_label_authority

    authority = resolve_label_authority(tmp_path, tmp_path, "auto")
    assert authority["label_source"] is None
    arm = _goal_a_arm(tmp_path, build_backend("lexical", None), [], {}, authority=authority)
    assert arm["measured"] is False
    assert "no adjudicated label file exists" in arm["reason"]
    assert "wrong answer attempt" in arm["why_this_blocks"]


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
