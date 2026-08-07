"""Blocklists.

The assertion worth stating outright: `IDBlocklist` **cannot** decide on content, and
that limitation is the paper's subject rather than a bug to be patched. If someone ever
"fixes" it by adding content matching, the C1/C3 comparison stops measuring SBU.
"""

from __future__ import annotations

import pytest

from rdl.memory.blocklist import (
    IDBlocklist,
    NoBlocklist,
    SemanticBlocklist,
    build_blocklist,
)
from rdl.memory.node import MemoryNode

FACT = "Basil Mahfouz Al-Kuwaiti's father was a florist in Kuwait City."


def _node(content: str, node_id: str = "n1") -> MemoryNode:
    return MemoryNode(node_id=node_id, content=content)


# ------------------------------------------------------------------------ no-op --


def test_no_blocklist_blocks_nothing():
    bl = NoBlocklist()
    assert not bl.blocks(_node(FACT))
    assert not bl.is_blocked_text(FACT)
    assert bl.blocked_ids() == set()
    assert len(bl) == 0


# ----------------------------------------------------------------------- by id --


def test_id_blocklist_matches_on_id():
    bl = IDBlocklist(["n1"])
    assert bl.blocks(_node("anything", "n1")).blocked
    assert not bl.blocks(_node("anything", "n2")).blocked
    assert bl.contains("n1")
    assert bl.blocked_ids() == {"n1"}


def test_id_blocklist_cannot_decide_on_content():
    """THE structural gap the paper is about. Do not 'fix' this."""
    bl = IDBlocklist(["m1"])
    decision = bl.is_blocked_text(FACT)
    assert not decision.blocked
    assert "cannot evaluate raw content" in decision.reason


def test_id_blocklist_misses_a_restatement_under_a_fresh_id():
    """m2 restates m1 verbatim. Same content, different id. The blocklist is blind."""
    bl = IDBlocklist(["m1"])
    m2 = _node(FACT, "m2")  # identical content, fresh id
    assert not bl.blocks(m2).blocked


def test_id_blocklist_add_and_discard():
    bl = IDBlocklist()
    bl.add("a")
    bl.add_all(["b", "c"])
    assert bl.blocked_ids() == {"a", "b", "c"}
    bl.discard("b")
    assert bl.blocked_ids() == {"a", "c"}
    assert len(bl) == 2


# ------------------------------------------------------------------- semantic --


def test_semantic_blocklist_catches_the_exact_restatement():
    bl = SemanticBlocklist([FACT], threshold=0.75)
    decision = bl.is_blocked_text(FACT)
    assert decision.blocked
    assert decision.score >= 0.75


def test_semantic_blocklist_catches_a_near_paraphrase():
    bl = SemanticBlocklist([FACT], threshold=0.6)
    near = "Basil Mahfouz Al-Kuwaiti's father worked as a florist in Kuwait City."
    assert bl.is_blocked_text(near).blocked


def test_semantic_blocklist_passes_unrelated_content():
    bl = SemanticBlocklist([FACT], threshold=0.6)
    decision = bl.is_blocked_text("Quantum chromodynamics describes the strong interaction.")
    assert not decision.blocked
    assert decision.score < 0.6


def test_semantic_blocklist_succeeds_where_the_id_blocklist_fails():
    """The Phase-2 argument, in two lines."""
    m2 = _node(FACT, "m2")
    assert not IDBlocklist(["m1"]).blocks(m2).blocked
    assert SemanticBlocklist([FACT], threshold=0.75).blocks(m2).blocked


def test_semantic_blocklist_still_honours_ids():
    bl = SemanticBlocklist([], ids=["m1"])
    assert bl.blocks(_node("totally unrelated text", "m1")).blocked


def test_semantic_blocklist_consults_the_nli_head_below_threshold():
    calls = []

    def nli(premise, hypothesis):
        calls.append((premise, hypothesis))
        return True

    bl = SemanticBlocklist([FACT], threshold=0.99, nli_fn=nli, nli_recall_floor=0.1)
    decision = bl.is_blocked_text("His father sold flowers in Kuwait City.")
    assert decision.blocked
    assert "NLI entailment" in decision.reason
    assert calls


def test_semantic_blocklist_empty_scores_zero():
    bl = SemanticBlocklist([], threshold=0.5)
    decision = bl.is_blocked_text("anything")
    assert not decision.blocked
    assert decision.score == 0.0


# ------------------------------------------------------------------ one protocol --


def test_all_blocklists_share_one_protocol():
    """C1/C3 and the Phase-2 method must differ by config, not by code path."""
    node = _node(FACT, "m1")
    for bl in (NoBlocklist(), IDBlocklist(["m1"]), SemanticBlocklist([FACT])):
        decision = bl.blocks(node)
        assert isinstance(decision.blocked, bool)
        assert isinstance(decision.score, float)
        assert isinstance(decision.reason, str)
        assert isinstance(bl.blocked_ids(), set)
        assert isinstance(bl.is_blocked_text(FACT).blocked, bool)


def test_block_decision_is_truthy():
    bl = IDBlocklist(["m1"])
    assert bl.blocks(_node("x", "m1"))
    assert not bl.blocks(_node("x", "m2"))


@pytest.mark.parametrize("kind", ["none", "id", "semantic"])
def test_build_blocklist_dispatch(kind):
    bl = build_blocklist(kind, ids=["m1"], texts=[FACT])
    assert bl.kind == kind


def test_build_blocklist_rejects_unknown():
    with pytest.raises(ValueError, match="unknown blocklist kind"):
        build_blocklist("regex")
