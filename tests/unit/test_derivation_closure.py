"""Derivation closure, refcounts, and the diamond.

The diamond is the case worth spelling out: when ``m1 -> m2 -> m4`` and
``m1 -> m3 -> m4``, deleting ``m1`` reaches ``m4`` along two paths but must decrement it
exactly once. Double-decrementing would make the store look more thoroughly pruned than
it is, which would understate the leak.
"""

from __future__ import annotations

import pytest

from rdl.memory.blocklist import IDBlocklist
from rdl.memory.derivation import CycleError, DerivationDAG
from rdl.memory.store import MemoryStore


def _chain(store: MemoryStore, n: int):
    nodes = [store.add("root", source_kind="ingest")]
    for i in range(1, n):
        nodes.append(store.add(f"derived {i}", parent_ids=[nodes[-1].node_id]))
    return nodes


# ------------------------------------------------------------------- closure shape --


def test_closure_is_transitive(store):
    a, b, c, d = _chain(store, 4)
    closure = store.dag.dependency_closure(a.node_id)
    assert closure == {b.node_id, c.node_id, d.node_id}
    assert a.node_id not in closure


def test_closure_can_include_self():
    dag = DerivationDAG()
    dag.add_edge("a", "b")
    assert dag.dependency_closure("a", include_self=True) == {"a", "b"}


def test_ancestors_is_the_reverse_of_closure(store):
    a, b, c = _chain(store, 3)
    assert store.dag.ancestors(c.node_id) == {a.node_id, b.node_id}
    assert store.dag.dependency_closure(a.node_id) == {b.node_id, c.node_id}


def test_closure_of_a_leaf_is_empty(store):
    _, leaf = _chain(store, 2)
    assert store.dag.dependency_closure(leaf.node_id) == set()


def test_cycles_are_rejected():
    dag = DerivationDAG()
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")
    with pytest.raises(CycleError):
        dag.add_edge("c", "a")
    with pytest.raises(CycleError):
        dag.add_edge("a", "a")


# ------------------------------------------------------------------------ refcounts --


def test_refcount_is_the_number_of_supporting_parents(store):
    root = store.add("root", source_kind="ingest")
    other = store.add("other root", source_kind="ingest")
    both = store.add("derived from two", parent_ids=[root.node_id, other.node_id])

    assert root.refcount == 1, "an independently grounded node is floored at 1"
    assert both.refcount == 2, "one reference per supporting parent"


def test_chain_deletion_decrements_and_marks_outdated(store):
    a, b, c = _chain(store, 3)
    store.delete(a.node_id)

    assert store.get(a.node_id).deleted
    assert store.get(b.node_id).refcount == 0
    assert store.get(b.node_id).outdated
    assert store.get(c.node_id).outdated


def test_diamond_does_not_double_decrement(store):
    """m1 -> {m2, m3} -> m4. Deleting m1 reaches m4 twice; it must decrement once."""
    m1 = store.add("root", source_kind="ingest")
    m2 = store.add("left", parent_ids=[m1.node_id])
    m3 = store.add("right", parent_ids=[m1.node_id])
    m4 = store.add("join", parent_ids=[m2.node_id, m3.node_id])

    assert m4.refcount == 2

    result = store.delete(m1.node_id)

    assert set(result.closure_ids) == {m2.node_id, m3.node_id, m4.node_id}
    # THE ASSERTION THIS TEST EXISTS FOR: exactly one decrement, 2 -> 1, not 2 -> 0.
    assert result.decremented[m4.node_id] == 1
    assert store.get(m4.node_id).refcount == 1

    # ...and yet m4 is still marked outdated, because BOTH of its parents are dead.
    assert store.get(m2.node_id).outdated
    assert store.get(m3.node_id).outdated
    assert store.get(m4.node_id).outdated, (
        "the join node's refcount survives the single decrement, but its entire "
        "provenance is gone, so the second marking rule must catch it"
    )


def test_node_with_a_surviving_parent_stays_live(store):
    a = store.add("doomed root", source_kind="ingest")
    survivor = store.add("live root", source_kind="ingest")
    child = store.add("derived from both", parent_ids=[a.node_id, survivor.node_id])

    store.delete(a.node_id)

    node = store.get(child.node_id)
    assert node.refcount == 1
    assert not node.outdated, "one parent survived, so provenance is not entirely dead"


def test_deleted_node_leaves_the_returnable_set(store):
    a = store.add("some content to be forgotten", source_kind="ingest")
    assert a.node_id in store.returnable_ids()
    store.delete(a.node_id)
    assert a.node_id not in store.returnable_ids()
    assert a.node_id in store.indexed_ids(), "it stays as a tombstone"


def test_outdated_nodes_remain_returnable(store):
    """Outdated is not deleted. Whether outdated content is still reachable is the
    question invariant 1 has to be checked against, not assumed away."""
    a = store.add("root", source_kind="ingest")
    b = store.add("derived", parent_ids=[a.node_id])
    store.delete(a.node_id)

    assert store.get(b.node_id).outdated
    assert b.node_id in store.returnable_ids()


def test_delete_with_closure_adds_to_the_blocklist(store):
    blocklist = IDBlocklist()
    a = store.add("root", source_kind="ingest")
    result = store.delete(a.node_id, blocklist=blocklist)

    assert blocklist.contains(a.node_id)
    assert a.node_id in result.blocklisted


def test_delete_unknown_node_raises(store):
    with pytest.raises(KeyError):
        store.delete("nope")


def test_add_with_unknown_parent_raises(store):
    with pytest.raises(KeyError):
        store.add("orphan", parent_ids=["not-a-real-node"])


# ------------------------------------------------------------------- path finding --


def test_path_to_ancestor_finds_the_shortest_route():
    dag = DerivationDAG()
    dag.add_edge("m1", "m2")
    dag.add_edge("m2", "m3")
    dag.add_edge("m1", "m3")  # a shorter route from m3 to m1

    assert dag.path_to_ancestor("m3", {"m1"}) == ["m3", "m1"]


def test_path_to_ancestor_returns_none_when_unreachable():
    dag = DerivationDAG()
    dag.add_edge("a", "b")
    dag.add_node("island")
    assert dag.path_to_ancestor("island", {"a"}) is None
    assert dag.path_to_ancestor("b", set()) is None


def test_path_to_ancestor_handles_self():
    dag = DerivationDAG()
    dag.add_node("x")
    assert dag.path_to_ancestor("x", {"x"}) == ["x"]


# --------------------------------------------------------------------- round trip --


def test_dag_round_trips_through_dict():
    dag = DerivationDAG()
    dag.add_edge("a", "b")
    dag.add_edge("b", "c")

    back = DerivationDAG.from_dict(dag.to_dict())
    assert back.edges() == dag.edges()
    assert back.nodes() == dag.nodes()


def test_dag_copy_is_independent():
    dag = DerivationDAG()
    dag.add_edge("a", "b")
    other = dag.copy()
    other.add_edge("b", "c")
    assert len(dag.edges()) == 1
    assert len(other.edges()) == 2
