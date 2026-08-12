"""Recognising a concept late must reach everything already derived from it."""

from __future__ import annotations

from rdl.graph_memory.derivation_closure import tag_closure
from rdl.graph_memory.policy_node import node_forget_ids
from rdl.graph_memory.scope_index import ScopeIndex
from rdl.memory.store import MemoryStore


def _chain(store):
    a = store.add("root content", source_agent="ingest")
    b = store.add("a summary of the root", source_agent="B", parent_ids=[a.node_id])
    c = store.add("a summary of the summary", source_agent="C", parent_ids=[b.node_id])
    return a, b, c


def test_tagging_reaches_every_descendant():
    store = MemoryStore(embedding_dim=64)
    index = ScopeIndex()
    a, b, c = _chain(store)
    tagged = tag_closure(
        dag=store.dag, nodes=store.nodes, index=index, node_id=a.node_id, forget_ids=["a17"]
    )
    assert set(tagged) == {a.node_id, b.node_id, c.node_id}
    for node in (a, b, c):
        assert index.forget_ids(node.node_id) == ("a17",)
        assert node_forget_ids(node) == ("a17",)


def test_a_diamond_join_is_tagged_once():
    store = MemoryStore(embedding_dim=64)
    index = ScopeIndex()
    root = store.add("root", source_agent="ingest")
    left = store.add("left", source_agent="B", parent_ids=[root.node_id])
    right = store.add("right", source_agent="C", parent_ids=[root.node_id])
    join = store.add("join", source_agent="D", parent_ids=[left.node_id, right.node_id])
    tag_closure(
        dag=store.dag, nodes=store.nodes, index=index, node_id=root.node_id, forget_ids=["a17"]
    )
    assert index.forget_ids(join.node_id) == ("a17",)


def test_ancestors_are_not_tagged():
    """Deriving FROM tainted content taints you; being derived from does not."""
    store = MemoryStore(embedding_dim=64)
    index = ScopeIndex()
    a, b, _c = _chain(store)
    tag_closure(
        dag=store.dag, nodes=store.nodes, index=index, node_id=b.node_id, forget_ids=["a17"]
    )
    assert index.forget_ids(a.node_id) == ()


def test_empty_scope_is_a_no_op():
    store = MemoryStore(embedding_dim=64)
    index = ScopeIndex()
    a, _b, _c = _chain(store)
    assert (
        tag_closure(dag=store.dag, nodes=store.nodes, index=index, node_id=a.node_id, forget_ids=[])
        == ()
    )
    assert len(index) == 0


def test_scope_index_is_two_way():
    index = ScopeIndex()
    index.tag("n1", ["a17"])
    index.tag("n2", ["a17", "a03"])
    assert index.nodes_for("a17") == ("n1", "n2")
    assert index.nodes_for("a03") == ("n2",)
    assert index.untagged_nodes(["n1", "n3"]) == ("n3",)
