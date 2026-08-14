"""The runtime detector is scored on what a deployed system knows, and nothing else.

Every test here is about an input the detector must not be able to receive. They matter
because each forbidden field exists elsewhere in this repo on rows that legitimately carry
it — the offline evaluator has the gold answers, the corpus builder has the item ids and
the concept labels — and the only thing between those rows and the runtime is a
constructor that refuses them.

A detector that had seen any of them would report a recall number describing an experiment
nobody can run twice.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

import rdl.defenses as defenses_pkg
from rdl.defenses import answerability_detector, detection_context, detector_protocol
from rdl.defenses.answerability_detector import LexicalAnswerabilityDetector
from rdl.defenses.concept_registry import ConceptPolicy, ConceptRegistry
from rdl.defenses.detection_context import (
    FORBIDDEN_CONTEXT_KEYS,
    DetectionContext,
    ProtectedQuestion,
    build_context,
    protected_questions_from_registry,
)
from rdl.defenses.detector_protocol import ConceptDetector, LegacyDetectorAdapter
from rdl.defenses.identity_router import IngressRouting, alias_index
from rdl.defenses.semantic_detector import SemanticConceptDetector

ROWS = [
    {
        "item_id": "forget10-0000",
        "concept_id": "author-0000",
        "question": "Where was Ada Vane born?",
    },
    {
        "item_id": "forget10-0001",
        "concept_id": "author-0001",
        "question": "Where was Cyril Moss born?",
    },
]


@pytest.fixture
def registry() -> ConceptRegistry:
    return ConceptRegistry.from_questions(ROWS, policy=ConceptPolicy())


# ------------------------------------------------------- the answer sheet is refused --


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_CONTEXT_KEYS))
def test_a_protected_question_cannot_be_built_from_the_answer_sheet(forbidden):
    row = {
        "scope_id": "s",
        "forget_id": "f",
        "question": "Where was Ada Vane born?",
        forbidden: "anything at all",
    }
    with pytest.raises(ValueError, match="never be built"):
        ProtectedQuestion.from_mapping(row)


def test_a_protected_question_has_no_answer_field():
    fields = set(ProtectedQuestion.__dataclass_fields__)
    assert not (fields & FORBIDDEN_CONTEXT_KEYS)


def test_context_metadata_cannot_smuggle_the_answer_sheet():
    with pytest.raises(ValueError, match="may not carry"):
        DetectionContext(
            request_text="Where was Ada Vane born?",
            routing=IngressRouting(),
            metadata={"item_id": "forget10-0000"},
        )


def test_no_defence_module_reads_the_oracle_answer_key():
    """The oracle's sidecar is offline-only, and this is where that is enforced.

    Checked over the whole package rather than the two v4 modules: the point is that no
    future defence may quietly start loading it, and a test that named only today's files
    would not notice.
    """
    root = Path(inspect.getfile(defenses_pkg)).parent
    offenders = [
        path.name
        for path in sorted(root.glob("*.py"))
        if "ANSWER_KEY" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_no_runtime_module_imports_the_offline_oracle():
    """``rdl.defenses`` must not depend on ``rdl.cli.detector_v4_oracle``."""
    root = Path(inspect.getfile(defenses_pkg)).parent
    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any("oracle" in name for name in names), path.name


# ------------------------------------------------------------- routing is the fence --


def test_a_context_cannot_hold_a_question_the_router_did_not_select():
    question = ProtectedQuestion(
        scope_id="s", forget_id="author-0000", question="Where was Ada Vane born?"
    )
    with pytest.raises(ValueError, match="not selected by the router"):
        DetectionContext(
            request_text="Who wrote Dune?",
            routing=IngressRouting(),
            protected_questions=(question,),
        )


def test_routing_alone_creates_no_content_forget_id(registry):
    """``policy_context_ids`` narrows detection. It never tags, propagates or enforces."""
    index = alias_index(registry)
    questions = protected_questions_from_registry(
        registry, questions_by_concept={r["concept_id"]: [r["question"]] for r in ROWS}
    )
    context = build_context(
        "Tell me about Ada Vane.", protected_questions=questions, alias_index=index
    )
    assert context.routed is True
    # The request names the author and asks nothing that content answered.
    result = LexicalAnswerabilityDetector().score("Ada Vane is an author.", context=context)
    assert result.forget_ids == ()


def test_request_is_evidence_is_refused_under_graph_flow():
    with pytest.raises(ValueError, match="not available under graph_flow"):
        DetectionContext(
            request_text="Where was Ada Vane born?",
            routing=IngressRouting(),
            request_is_evidence=True,
            protocol="graph_flow",
        )


def test_request_is_evidence_is_available_under_end_to_end_safety():
    """The request gate is a real surface there, and it is reported separately."""
    context = DetectionContext(
        request_text="Where was Ada Vane born?",
        routing=IngressRouting(),
        request_is_evidence=True,
        protocol="end_to_end_safety",
    )
    assert context.request_is_evidence is True


# --------------------------------------------------------------- protocol conformance --


def test_both_backends_satisfy_the_protocol(registry):
    assert isinstance(LexicalAnswerabilityDetector(), ConceptDetector)
    assert isinstance(LegacyDetectorAdapter(SemanticConceptDetector(registry)), ConceptDetector)


def test_the_legacy_adapter_does_not_condition_on_the_request(registry):
    """Adding the protocol must not change a single number v1/v2 published.

    Folding the request into the hashing detector's input here would silently alter v1
    behaviour under the banner of an interface change, so the adapter is required to
    produce the same scores whatever the request said.
    """
    adapter = LegacyDetectorAdapter(SemanticConceptDetector(registry))
    routing = IngressRouting(policy_context_ids=("author-0000",))
    a = adapter.score_batch(
        ["Ada Vane was born in Paris."],
        context=DetectionContext("Where was Ada Vane born?", routing),
    )
    b = adapter.score_batch(
        ["Ada Vane was born in Paris."],
        context=DetectionContext("Something else entirely", routing),
    )
    assert a[0].score == b[0].score
    assert a[0].per_concept == b[0].per_concept


def test_the_legacy_adapter_reports_no_answerability(registry):
    """A similarity backbone has no ANSWER/PARTIAL split, and must not pretend to."""
    adapter = LegacyDetectorAdapter(SemanticConceptDetector(registry))
    assert adapter.to_dict()["reports_answerability"] is False
    routing = IngressRouting(policy_context_ids=("author-0000",))
    result = adapter.score_batch(
        ["Ada Vane was born in Paris."], context=DetectionContext("q", routing)
    )[0]
    assert result.partial_probability == 0.0


def test_the_v4_result_is_a_superset_of_the_v1_result_shape():
    """``EvidenceAccumulator`` and the executor read v1 fields; they must still be there."""
    from rdl.defenses.semantic_detector import DetectionResult

    v1 = set(DetectionResult.__dataclass_fields__) - {"version"}
    v4 = set(detector_protocol.AnswerabilityResult.__dataclass_fields__)
    assert v1 <= v4
    assert hasattr(detector_protocol.AnswerabilityResult(), "version")


def test_score_batch_takes_context_as_a_keyword_only_argument():
    """There is no positional slot a candidate string could accidentally fill."""
    for module in (answerability_detector.LexicalAnswerabilityDetector, LegacyDetectorAdapter):
        signature = inspect.signature(module.score_batch)
        assert signature.parameters["context"].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_context_docstring_states_the_request_evidence_rule():
    assert "not itself evidence" in (detection_context.__doc__ or "")
