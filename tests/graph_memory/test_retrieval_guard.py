"""Retrieval protection, including the rescan that survives tag removal."""

from __future__ import annotations

from rdl.defenses.dragon_style import DragonStyleDefense
from rdl.defenses.graphforget import GraphForgetDefense
from rdl.defenses.none import NoDefense
from rdl.graph_memory.retrieval_guard import RetrievalGuard
from rdl.graph_memory.staged_store import WriteCandidate


def _plant(memory, text, forget_ids=()):
    memory.stage(
        WriteCandidate(content=text, source_node="E", envelope_id="e", forget_ids=tuple(forget_ids))
    )
    return memory.commit()[0]


def test_a_tagged_node_is_withheld(staged_memory, detector, tofu_items):
    node_id = _plant(staged_memory, tofu_items[0].answer, ["stub-concept-0000"])
    guard = RetrievalGuard(staged_memory, GraphForgetDefense(detector=detector), retrieval_k=5)
    result = guard.retrieve(tofu_items[0].question, node_id="E")
    assert node_id not in result.node_ids
    assert node_id in result.withheld_node_ids


def test_removing_the_tag_does_not_defeat_the_guard(staged_memory, detector, tofu_items):
    """Semantics, not bookkeeping, is what withholds the node."""
    node_id = _plant(staged_memory, tofu_items[0].answer, ["stub-concept-0000"])
    # Simulate a dropped tag: rebuild the index with nothing in it.
    from rdl.graph_memory.scope_index import ScopeIndex

    staged_memory.scopes = ScopeIndex()
    assert staged_memory.scopes.forget_ids(node_id) == ()

    guard = RetrievalGuard(staged_memory, GraphForgetDefense(detector=detector), retrieval_k=5)
    result = guard.retrieve(tofu_items[0].question, node_id="E")
    assert node_id not in result.node_ids
    assert node_id in result.rescan_withheld_node_ids


def test_disabling_the_rescan_is_an_ablation(staged_memory, detector, tofu_items):
    node_id = _plant(staged_memory, tofu_items[0].answer, [])
    guard = RetrievalGuard(
        staged_memory,
        GraphForgetDefense(detector=detector, rescan_untagged_memory=False),
        retrieval_k=5,
    )
    result = guard.retrieve(tofu_items[0].question, node_id="E")
    assert node_id in result.node_ids


def test_clean_memory_is_still_returned(staged_memory, detector):
    node_id = _plant(staged_memory, "Water boils at 100 degrees Celsius.", [])
    guard = RetrievalGuard(staged_memory, GraphForgetDefense(detector=detector), retrieval_k=5)
    result = guard.retrieve("What temperature does water boil at?", node_id="E")
    assert node_id in result.node_ids


def test_the_node_local_baseline_does_not_guard_retrieval(staged_memory, detector, tofu_items):
    node_id = _plant(staged_memory, tofu_items[0].answer, ["stub-concept-0000"])
    guard = RetrievalGuard(staged_memory, DragonStyleDefense(detector=detector), retrieval_k=5)
    result = guard.retrieve(tofu_items[0].question, node_id="E")
    assert node_id in result.node_ids
    assert result.n_withheld == 0


def test_the_unguarded_arm_returns_everything(staged_memory, tofu_items):
    node_id = _plant(staged_memory, tofu_items[0].answer, ["stub-concept-0000"])
    guard = RetrievalGuard(staged_memory, NoDefense(), retrieval_k=5)
    assert node_id in guard.retrieve(tofu_items[0].question, node_id="E").node_ids


def test_the_sbu_blocklist_still_runs_first(staged_memory, tofu_items):
    """The ingested forget content was deleted and blocklisted before any arm started."""
    result = staged_memory.candidates(tofu_items[0].question, 5)
    assert all(tofu_items[0].answer not in text for _nid, text, _tags in result)
