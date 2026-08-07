"""Both readings of SBU's memory pathway, side by side.

Our default (`tombstone` + `supporting_parents`) diverges from the paper on two points.
The divergence is defensible — it makes the defence do strictly more work — but calling
it "SBU implemented faithfully" was an overclaim, so the paper's version is implemented
too and these tests pin both. ADR-0014 and ADR-0015.
"""

from __future__ import annotations

import pytest

from rdl.memory.blocklist import IDBlocklist
from rdl.memory.invariants import check_invariant_1
from rdl.memory.store import MemoryStore

FACT = "The mentor of Basil Mahfouz Al-Kuwaiti was a florist in Kuwait City."


def _store(**kw) -> MemoryStore:
    return MemoryStore(index_backend="numpy", embedding_dim=64, **kw)


# =====================================================================================
# deletion_mode
# =====================================================================================


def test_tombstone_keeps_the_vector_and_makes_the_blocklist_do_the_work():
    store = _store(deletion_mode="tombstone")
    n = store.add(FACT, source_kind="ingest")
    bl = IDBlocklist()
    store.delete(n.node_id, blocklist=bl)

    assert n.node_id in store.indexed_ids(), "the tombstone stays indexed by design"
    assert store.retrieve(FACT, k=5, blocklist=bl).nodes == [], "retrieval refuses it"
    # I1 is a real behavioural check here: the retrieval path had to enforce something.
    assert check_invariant_1(store, bl).satisfied


def test_hard_delete_removes_the_vector_as_the_paper_describes():
    store = _store(deletion_mode="hard")
    n = store.add(FACT, source_kind="ingest")
    bl = IDBlocklist()
    store.delete(n.node_id, blocklist=bl)

    assert n.node_id not in store.indexed_ids(), "target AND vector are gone"
    assert store.get(n.node_id).embedding is None
    # I1 now holds trivially — that is a property of the paper's design, and it is why
    # the laundering result matters: the leak never touches retrieval.
    assert check_invariant_1(store, bl).satisfied


def test_hard_deleted_nodes_are_not_resurrected_by_a_snapshot_round_trip():
    """`restore` re-encodes any node whose embedding is None. Without a guard that
    silently undoes every hard delete on the first snapshot, and the store quietly goes
    back to being retrievable."""
    store = _store(deletion_mode="hard")
    n = store.add(FACT, source_kind="ingest")
    store.delete(n.node_id, blocklist=IDBlocklist())

    store.restore(store.snapshot())
    assert n.node_id not in store.indexed_ids()


def test_unknown_modes_are_rejected_at_construction():
    with pytest.raises(ValueError, match="deletion_mode"):
        _store(deletion_mode="soft")
    with pytest.raises(ValueError, match="refcount_semantics"):
        _store(refcount_semantics="citations")


# =====================================================================================
# refcount_semantics
# =====================================================================================


def test_supporting_parents_counts_live_parents():
    store = _store(refcount_semantics="supporting_parents")
    root = store.add("root", source_kind="ingest")
    child = store.add("child", source_kind="summary", parent_ids=[root.node_id])

    assert root.refcount == 1, "independently grounded, floored at 1"
    assert child.refcount == 1, "one supporting parent"


def test_dependent_children_counts_dependants_as_the_paper_does():
    store = _store(refcount_semantics="dependent_children")
    root = store.add("root", source_kind="ingest")
    assert root.refcount == 0, "nothing depends on it yet"

    store.add("child-a", source_kind="summary", parent_ids=[root.node_id])
    assert root.refcount == 1
    store.add("child-b", source_kind="summary", parent_ids=[root.node_id])
    assert root.refcount == 2, "two nodes now depend on the root"


def test_dependent_children_reclaims_upward_when_the_last_dependant_goes():
    """Classic reference counting: deleting a node releases its references, and a
    support node with nothing left depending on it is reclaimed."""
    store = _store(refcount_semantics="dependent_children")
    support = store.add("shared context", source_kind="ingest")
    a = store.add("derived a", source_kind="summary", parent_ids=[support.node_id])
    b = store.add("derived b", source_kind="summary", parent_ids=[support.node_id])
    assert support.refcount == 2

    store.delete(a.node_id, blocklist=IDBlocklist())
    assert support.refcount == 1
    assert not support.outdated, "b still depends on it"

    store.delete(b.node_id, blocklist=IDBlocklist())
    assert support.refcount == 0
    assert support.outdated, "nothing depends on it any more — reclaimed"


def test_downward_invalidation_holds_under_both_semantics():
    """Invariant 2 is required in both readings: content derived from a deleted node
    must not stay live with dead provenance."""
    for semantics in ("supporting_parents", "dependent_children"):
        store = _store(refcount_semantics=semantics)
        root = store.add("root", source_kind="ingest")
        child = store.add("derived", source_kind="summary", parent_ids=[root.node_id])
        store.delete(root.node_id, blocklist=IDBlocklist())
        assert child.outdated, f"{semantics}: derived content was left live"


def test_a_parametric_node_is_untouched_by_either_semantics():
    """The whole finding in one assertion: a node with no parents is outside the scope
    of both readings of the refcount, no matter what its content says."""
    for semantics in ("supporting_parents", "dependent_children"):
        store = _store(refcount_semantics=semantics)
        ingested = store.add(FACT, source_kind="ingest")
        store.delete(ingested.node_id, blocklist=IDBlocklist())

        parametric = store.add(FACT, source_agent="B", source_kind="agent_answer")
        assert parametric.parent_ids == []
        assert not parametric.outdated and not parametric.deleted, semantics
        assert parametric in store.all_nodes()
