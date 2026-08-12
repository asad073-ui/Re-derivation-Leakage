"""MA-CONTROL carries messages about a DIFFERENT concept, in the same wrapper."""

from __future__ import annotations

import pytest

from rdl.eval.negatives import NoCrossAuthorMapping
from rdl.studies.graph_leak.controls import (
    build_controlled_challenges,
    concept_control_mapping,
    paraphrase_answer,
)


def test_the_mapping_never_pairs_an_item_with_its_own_concept(tofu_items):
    concepts = [f"c-{i.index // 20:04d}" for i in tofu_items]
    mapping = concept_control_mapping(concepts)
    assert mapping.same_author_count == 0
    assert mapping.fixed_point_count == 0
    for i, j in enumerate(mapping.permutation):
        assert concepts[i] != concepts[j]


def test_the_mapping_is_seed_independent(tofu_items):
    concepts = [f"c-{i.index // 20:04d}" for i in tofu_items]
    assert (
        concept_control_mapping(concepts).permutation
        == concept_control_mapping(concepts).permutation
    )


def test_a_single_concept_cohort_cannot_support_the_control():
    with pytest.raises(NoCrossAuthorMapping):
        concept_control_mapping(["c-0"] * 6)


def test_the_mapping_is_auditable(tofu_items):
    concepts = [f"c-{i.index // 20:04d}" for i in tofu_items]
    payload = concept_control_mapping(concepts).to_dict()
    assert payload["algorithm"] == "rotate-by-smallest-cross-author-shift"
    assert payload["sha256"]
    assert payload["same_author_count"] == 0


def test_controlled_injections_are_built_per_item(tofu_items, diamond5):
    """So a cross-concept arm can be given the CONTROL item's injection."""
    a = build_controlled_challenges(tofu_items[0], diamond5, ["direct"])[0]
    b = build_controlled_challenges(tofu_items[3], diamond5, ["direct"])[0]
    assert a.injected_outputs != b.injected_outputs
    assert a.fingerprint() != b.fingerprint()


def test_split_clues_needs_a_join_node(tofu_items, diamond5, chain5):
    assert build_controlled_challenges(tofu_items[0], diamond5, ["split_clues"])
    assert build_controlled_challenges(tofu_items[0], chain5, ["split_clues"]) == []


def test_split_clues_delegates_the_join_nodes_query(tofu_items, diamond5):
    """Otherwise the query alone is in scope and a node-local guard catches it free."""
    challenge = build_controlled_challenges(tofu_items[0], diamond5, ["split_clues"])[0]
    assert set(challenge.injected_outputs) == {"B", "C"}
    assert set(challenge.node_questions) == {"D", "E"}
    assert tofu_items[0].question not in set(challenge.node_questions.values())


def test_every_controlled_challenge_declares_gold_answer_use(tofu_items, diamond5):
    for challenge in (
        "direct",
        "paraphrase",
        "partial_clue",
        "split_clues",
        "memory_reentry",
        "tool_reentry",
    ):
        for built in build_controlled_challenges(tofu_items[0], diamond5, [challenge]):
            assert built.uses_gold_answer
            assert built.to_dict()["uses_gold_answer"]


def test_paraphrase_breaks_exact_match_but_keeps_content():
    answer = "Basil Mahfouz Al-Kuwaiti's father was a florist in Kuwait City."
    para = paraphrase_answer(answer)
    assert para != answer
    assert "florist" in para
