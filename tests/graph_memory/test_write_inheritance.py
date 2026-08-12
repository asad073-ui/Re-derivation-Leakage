"""A new memory node cannot become clean by being new.

This is the laundering path the two-agent work found: an agent's parametric answer is
written with an empty parent list, so nothing in the store points back at the deleted
content and the ID blocklist has no id to block. Scope inheritance is what closes it.
"""

from __future__ import annotations

from rdl.defenses.base import WriteContext
from rdl.defenses.graphforget import GraphForgetDefense
from rdl.graph.envelope import derive_envelope
from rdl.graph_memory.policy_node import PolicyRecord, node_forget_ids, tag_node
from rdl.graph_memory.staged_store import WriteCandidate


def test_staged_writes_are_invisible_until_commit(staged_memory, tofu_items):
    query = tofu_items[0].question
    before = staged_memory.candidates(query, 5)
    staged_memory.stage(
        WriteCandidate(content=tofu_items[0].answer, source_node="E", envelope_id="e1")
    )
    assert staged_memory.candidates(query, 5) == before, "a staged write must not be retrievable"
    staged_memory.commit()
    after = staged_memory.candidates(query, 5)
    assert len(after) > len(before)


def test_immediate_visibility_is_a_separate_mode(tofu_items):
    from rdl.graph_memory.staged_store import StagedMemory
    from rdl.studies.graph_leak.runner import build_baseline_memory

    store, blocklist = build_baseline_memory(tofu_items)
    memory = StagedMemory(store, blocklist=blocklist, visibility="immediate")
    node_id = memory.stage(
        WriteCandidate(content=tofu_items[0].answer, source_node="E", envelope_id="e1")
    )
    assert node_id is not None
    assert node_id in memory.committed_ids


def test_a_write_inherits_scopes_from_its_envelope(staged_memory, tofu_items):
    staged_memory.stage(
        WriteCandidate(
            content="a bland restatement",
            source_node="E",
            envelope_id="e1",
            forget_ids=("stub-concept-0000",),
        )
    )
    (node_id,) = staged_memory.commit()
    assert staged_memory.scopes.forget_ids(node_id) == ("stub-concept-0000",)
    node = staged_memory.store.get(node_id)
    assert node_forget_ids(node) == ("stub-concept-0000",)


def test_a_write_inherits_scopes_from_its_store_parents(staged_memory):
    parent = staged_memory.store.add("tagged parent content", source_agent="ingest")
    staged_memory.scopes.tag(parent.node_id, ["stub-concept-0000"])
    staged_memory.stage(
        WriteCandidate(
            content="a summary with no scope of its own",
            source_node="E",
            envelope_id="e2",
            parent_store_ids=(parent.node_id,),
        )
    )
    (node_id,) = staged_memory.commit()
    assert "stub-concept-0000" in staged_memory.scopes.forget_ids(node_id)


def test_an_empty_parent_list_does_not_make_a_write_clean(detector, tofu_items):
    """The parametric-answer case. No parents to point at; the guard still blocks it."""
    guard = GraphForgetDefense(detector=detector)
    envelope = derive_envelope(kind="agent_output", content=tofu_items[0].answer, source_node="E")
    assert envelope.parent_ids == ()
    verdict = guard.on_memory_write(
        WriteContext(node_id="E", depth=3, envelope=envelope, retrieved_node_ids=())
    )
    assert not verdict.allowed
    assert verdict.forget_ids


def test_tags_only_ever_grow(staged_memory):
    node = staged_memory.store.add("content", source_agent="ingest")
    tag_node(node, PolicyRecord(forget_ids=("a",), score=0.6))
    tag_node(node, PolicyRecord(forget_ids=("b",), score=0.4))
    assert node_forget_ids(node) == ("a", "b")
    assert node.meta["policy"]["score"] == 0.6


def test_baseline_reset_restores_the_post_deletion_state(staged_memory, tofu_items):
    staged_memory.mark_baseline()
    staged_memory.stage(
        WriteCandidate(content=tofu_items[0].answer, source_node="E", envelope_id="e1")
    )
    staged_memory.commit()
    assert staged_memory.committed_ids
    staged_memory.reset_to_baseline()
    assert staged_memory.committed_ids == []
    assert not staged_memory.staged()
