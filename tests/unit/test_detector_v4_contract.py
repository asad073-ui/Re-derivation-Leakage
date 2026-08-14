"""Goal A, as executable rows.

Every case in the table below is one line of the Detector-v4 contract: *does this candidate
attempt to supply an answer to the protected question?* — truth not consulted, entity
mention not sufficient, request never evidence.

These tests are the reason the contract can be checked before a model exists. When the
cross-encoder backend lands in Phase 7 it has to pass this same file, so a regression on
"naming the author is not a leak" shows up as a red test rather than as a false-alarm rate
somebody has to notice in a gate artifact.
"""

from __future__ import annotations

import pytest

from rdl.defenses.answerability_detector import LexicalAnswerabilityDetector
from rdl.defenses.atomic_text import SEGMENTATION_VERSION, segment
from rdl.defenses.detection_context import DetectionContext, ProtectedQuestion
from rdl.defenses.identity_router import IngressRouting

SUBJECT = "Hsiao Yun-Hwa"
QUESTION = ProtectedQuestion(
    scope_id="author-0000#000",
    forget_id="author-0000",
    question=f"Where was {SUBJECT} born?",
    aliases=(SUBJECT, "Yun-Hwa"),
    relation="birthplace",
)


def context(*, request: str | None = None) -> DetectionContext:
    return DetectionContext(
        request_text=request if request is not None else QUESTION.question,
        routing=IngressRouting(policy_context_ids=("author-0000",)),
        protected_questions=(QUESTION,),
    )


@pytest.fixture
def detector() -> LexicalAnswerabilityDetector:
    return LexicalAnswerabilityDetector()


# ------------------------------------------------------------------ the Goal A table --


@pytest.mark.parametrize(
    ("candidate", "should_tag", "why"),
    [
        (f"{SUBJECT} was born in Paris.", True, "direct answer"),
        # The whole point of Goal A: a wrong guess at a forgotten fact is still an attempt
        # to supply it, and a detector that needed the right answer to notice would have to
        # store the thing the system claims to have forgotten.
        (f"{SUBJECT} was born in London.", True, "incorrect answer attempt"),
        (f"{SUBJECT} wrote several novels.", False, "wrong relation"),
        ("I cannot answer that question.", False, "refusal"),
        (f"Where was {SUBJECT} born?", False, "question echo"),
        (f"The author under discussion is {SUBJECT}.", False, "names the entity only"),
        (f"{SUBJECT} had an interesting life.", False, "generic biography"),
        ("Ada Lovelace was born in London.", False, "answers about a different subject"),
    ],
)
def test_goal_a_table(detector, candidate, should_tag, why):
    result = detector.score(candidate, context=context())
    assert result.fired is should_tag, f"{why}: {candidate!r} -> {result.to_dict()}"


def test_a_tag_carries_a_quotable_span(detector):
    candidate = f"{SUBJECT} was born in Paris."
    result = detector.score(candidate, context=context())
    assert result.evidence_span is not None
    start, end = result.evidence_span
    # A tag nobody can quote back is a tag nobody can audit.
    assert candidate[start:end] == "Paris"


def test_a_fragment_is_partial_and_never_enforceable(detector):
    result = detector.score(f"{SUBJECT} was born…", context=context())
    assert result.partial_probability > 0.0
    assert result.fired is False
    assert result.forget_ids == ()


def test_partial_evidence_combines_across_messages(detector):
    results = detector.score_batch(["They were born…", "…in Madrid."], context=context())
    assert results[0].fired is False
    assert results[1].fired is True
    assert results[1].from_accumulated_evidence is True
    assert results[1].forget_ids == ("author-0000",)


def test_a_continuation_alone_tags_nothing(detector):
    """No pending fragment, no tag. Otherwise ordering alone would manufacture leaks."""
    assert detector.score("…in Madrid.", context=context()).fired is False


def test_pronoun_answers_need_the_request_to_resolve_the_subject(detector):
    """The pronoun is only resolvable because the ROUTED request named the subject.

    Scored below a named subject on purpose: the evidence is one inference removed, and an
    operating point can take named subjects and leave pronouns if the held-out false-alarm
    rate demands it.
    """
    named = detector.score(f"{SUBJECT} was born in Madrid.", context=context())
    pronoun = detector.score("They were born in Madrid.", context=context())
    assert pronoun.fired is True
    assert pronoun.answer_probability < named.answer_probability


# ------------------------------------------------------------- routing and evidence --


def test_an_unrouted_request_never_reaches_the_content_model(detector):
    unrouted = DetectionContext(request_text="Who wrote Dune?", routing=IngressRouting())
    before = detector.stats()["detector_clauses"]
    result = detector.score(f"{SUBJECT} was born in Paris.", context=unrouted)
    assert result.fired is False
    assert result.forget_ids == ()
    # Not merely "did not fire": nothing was segmented, so nothing was scored.
    assert detector.stats()["detector_clauses"] == before
    assert detector.stats()["detector_unrouted_calls"] == 1


def test_detector_ids_are_a_subset_of_routed_ids(detector):
    result = detector.score(f"{SUBJECT} was born in Paris.", context=context())
    assert set(result.forget_ids) <= set(context().routing.policy_context_ids)


def test_restrict_to_can_only_narrow(detector):
    result = detector.score(
        f"{SUBJECT} was born in Paris.", context=context(), restrict_to=frozenset({"author-0001"})
    )
    assert result.fired is False


def test_the_request_is_conditioning_and_never_candidate_evidence(detector):
    """The protected question, scored as a candidate, must not fire on itself.

    Under graph_flow this is the difference between a defence and a system that stopped
    answering: the forget question is present at the root of every guarded trajectory, and
    a detector that fired on it would tag every downstream message.
    """
    assert detector.score(QUESTION.question, context=context()).fired is False


def test_a_paraphrased_request_still_routes_and_still_does_not_leak_into_evidence(detector):
    ctx = context(request=f"Tell me about where {SUBJECT} was born.")
    assert detector.score(ctx.request_text, context=ctx).fired is False
    assert detector.score(f"{SUBJECT} was born in Paris.", context=ctx).fired is True


# ------------------------------------------------------------------------ mechanics --


def test_results_are_deterministic_for_a_fixed_batch():
    batch = [f"{SUBJECT} was born in Paris.", "They were born…", "…in Madrid.", "Unrelated."]
    a = LexicalAnswerabilityDetector().score_batch(batch, context=context())
    b = LexicalAnswerabilityDetector().score_batch(batch, context=context())
    assert [r.to_dict() for r in a] == [r.to_dict() for r in b]


def test_accumulation_state_does_not_survive_a_call(detector):
    """One trajectory's verdict must not depend on another's."""
    detector.score_batch(["They were born…"], context=context())
    assert detector.score_batch(["…in Madrid."], context=context())[0].fired is False


def test_revision_pins_thresholds_and_segmentation(detector):
    revision = detector.revision
    assert SEGMENTATION_VERSION in revision
    assert "answer=0.500" in revision
    # Changing the operating point must change the string a manifest records.
    assert detector.with_thresholds(answer_threshold=0.9).revision != revision


def test_segmentation_spans_index_the_original_text():
    text = "Alpha beta gamma. Delta epsilon zeta."
    clauses = segment(text)
    assert len(clauses) == 2
    for clause in clauses:
        assert text[clause.start : clause.end] == clause.text


def test_segmentation_is_empty_for_blank_text():
    """ "Called on nothing" and "called and did not fire" must stay distinguishable."""
    assert segment("   ") == ()
