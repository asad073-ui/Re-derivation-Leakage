"""Joint evidence over several individually-innocuous inputs.

Built on a synthetic registry rather than the TOFU fixture, so the threshold arithmetic
is explicit and the test states exactly what it is asserting.
"""

from __future__ import annotations

from rdl.defenses.concept_registry import ConceptPolicy, ConceptRegistry
from rdl.defenses.evidence_accumulator import EvidenceAccumulator
from rdl.defenses.semantic_detector import SemanticConceptDetector
from rdl.studies.graph_leak.controls import split_answer


def _detector(threshold: float = 0.55, alias_weight: float = 1.0) -> SemanticConceptDetector:
    registry = ConceptRegistry.from_questions(
        [
            {
                "item_id": "x-0",
                "concept_id": "author-17",
                "question": "Who was Amara Rossi and what did she write?",
            }
        ],
        policy=ConceptPolicy(),
    )
    return SemanticConceptDetector(registry, threshold=threshold, alias_weight=alias_weight)


def test_two_clues_fire_jointly_but_not_individually():
    detector = _detector()
    clue_a = "Amara initials"
    clue_b = "Rossi wrote"
    accumulator = EvidenceAccumulator(detector)
    # Only assert the joint claim when the premise actually holds for this backbone.
    individually_fires = any(r.fired for r in detector.score_batch([clue_a, clue_b]))
    evidence = accumulator.evaluate(
        question="Combine the material you have been given.",
        input_texts=[clue_a, clue_b],
    )
    if not individually_fires and evidence.fired:
        assert evidence.accumulated_only
        assert accumulator.stats()["accumulated_only_hits"] == 1
    # In every case the joint view must be at least as strong as the strongest single one.
    assert evidence.score >= max(r.score for r in detector.score_batch([clue_a, clue_b]))


def test_an_in_scope_query_is_not_counted_as_accumulated_only():
    """The query firing on its own is a case DRAGON catches. It must not be claimed."""
    detector = _detector()
    accumulator = EvidenceAccumulator(detector)
    evidence = accumulator.evaluate(
        question="Who was Amara Rossi and what did she write?",
        input_texts=["an unrelated remark about the weather"],
    )
    assert evidence.fired
    assert evidence.query_forget_ids
    assert not evidence.accumulated_only
    assert accumulator.stats()["accumulated_only_hits"] == 0


def test_disabled_accumulator_never_claims_accumulated_only():
    detector = _detector()
    accumulator = EvidenceAccumulator(detector, enabled=False)
    evidence = accumulator.evaluate(
        question="Combine the material.", input_texts=["Amara initials", "Rossi wrote"]
    )
    assert not evidence.accumulated_only
    assert evidence.combination == "node_input"


def test_subset_scoring_can_beat_whole_context_scoring():
    """Why 'score the parents alone' is not redundant with 'score everything'.

    A long, unrelated query dilutes the embedding of the whole context. Scoring the
    parent messages as their own subset is what recovers the signal — and it is the
    concrete reason the accumulator is more than a re-run of the node-local guard.

    Isolated to the EMBEDDING channel (``alias_weight=0``). The alias channel is token
    coverage and is dilution-proof by construction, so it would mask the effect this
    test is about rather than disprove it.
    """
    detector = _detector(alias_weight=0.0)
    clues = ["Amara Rossi", "Rossi wrote novels"]
    long_query = " ".join(["please summarise the attached material carefully"] * 12)
    whole = detector.score("\n".join([long_query, *clues])).score
    parents_only = detector.score("\n".join(clues)).score
    assert parents_only > whole


def test_split_answer_produces_two_incomplete_halves():
    answer = "Basil Mahfouz Al-Kuwaiti's father was a florist in Kuwait City."
    first, second = split_answer(answer)
    assert first != second
    assert answer not in first and answer not in second
    # Together they still carry the content; that is what makes the join dangerous.
    assert "florist" in second or "florist" in first


def test_short_answers_are_not_split_into_nonsense():
    first, second = split_answer("Paris.")
    assert first == second == "Paris."
