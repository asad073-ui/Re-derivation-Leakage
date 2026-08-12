"""The registry never holds a forgotten answer, and the detector is shared."""

from __future__ import annotations

import pytest

from rdl.defenses.concept_registry import (
    ConceptRegistry,
    extract_aliases,
    paraphrase_question,
)
from rdl.graph.config import load_graph_config
from rdl.paths import repo_root
from rdl.studies.graph_leak.arms import build_arm_runtime, build_detector

LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"


def test_registry_refuses_gold_answers():
    with pytest.raises(ValueError, match="never be constructed from gold answers"):
        ConceptRegistry.from_questions(
            [{"item_id": "x", "concept_id": "c", "question": "Q?", "answer": "A."}]
        )


def test_registry_records_that_it_stores_no_answers(registry, tofu_items):
    payload = registry.to_dict()
    assert payload["stores_gold_answers"] is False
    serialised = str(payload)
    for item in tofu_items:
        assert item.answer not in serialised


def test_prototypes_come_from_questions_and_paraphrases(concept_rows):
    registry = ConceptRegistry.from_questions(concept_rows)
    concept = registry.concepts()[0]
    assert concept.scope_prototypes
    assert any(p.startswith(("tell me about", "give details on")) for p in concept.scope_prototypes)


def test_alias_extraction_finds_multi_word_names():
    assert "Basil Mahfouz Al-Kuwaiti" in extract_aliases(
        "Who was Basil Mahfouz Al-Kuwaiti's father?"
    ) or any("Basil" in a for a in extract_aliases("Who was Basil Mahfouz Al-Kuwaiti's father?"))


def test_paraphrase_is_deterministic():
    assert paraphrase_question("Who is X?") == paraphrase_question("Who is X?")
    assert paraphrase_question("Who is X?") != "Who is X?"


def test_registry_fingerprint_is_stable(concept_rows):
    a = ConceptRegistry.from_questions(concept_rows)
    b = ConceptRegistry.from_questions(list(reversed(concept_rows)))
    assert a.fingerprint() == b.fingerprint()


def test_dragon_and_graphforget_share_one_detector_object():
    """The single most likely reviewer objection, closed by construction."""
    cfg = load_graph_config(LAUNCH)
    registry = ConceptRegistry.from_questions(
        [{"item_id": "x-0", "concept_id": "c-0", "question": "Who was Amara Rossi?"}]
    )
    detector = build_detector(cfg, registry)
    plans = {p.name: p for p in build_arm_runtime(cfg, cfg.topology, detector)}
    dragon = plans["multi_agent_dragon"].defense
    ours = plans["multi_agent_graphforget"].defense
    assert dragon.detector is ours.detector is detector
    assert dragon.detector.threshold == ours.detector.threshold


def test_only_our_arm_propagates_scope():
    cfg = load_graph_config(LAUNCH)
    registry = ConceptRegistry.from_questions(
        [{"item_id": "x-0", "concept_id": "c-0", "question": "Who was Amara Rossi?"}]
    )
    plans = {p.name: p for p in build_arm_runtime(cfg, cfg.topology, build_detector(cfg, registry))}
    assert plans["multi_agent_graphforget"].defense.propagates_scope
    assert not plans["multi_agent_dragon"].defense.propagates_scope
    assert not plans["multi_agent_leak"].defense.propagates_scope
