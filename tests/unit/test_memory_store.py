"""MemoryStore: writes, retrieval, blocklist enforcement, snapshot/restore.

Snapshot/restore carries real weight: C1, C2, and C3 must all start from a
byte-identical store. If they do not, the between-condition comparison is confounded by
store initialisation, which would be an embarrassing way to lose a result.
"""

from __future__ import annotations

import copy

import pytest

from rdl.memory.blocklist import IDBlocklist, NoBlocklist
from rdl.memory.node import MemoryNode
from rdl.memory.store import MemoryStore


def test_add_returns_a_node_with_an_embedding(store):
    node = store.add("hello world", source_agent="ingest", source_kind="ingest")
    assert node.node_id
    assert node.embedding is not None
    assert node.embedding.shape == (64,)
    assert store.get(node.node_id) is node
    assert node.node_id in store


def test_duplicate_node_id_is_rejected(store):
    store.add("a", node_id="fixed")
    with pytest.raises(KeyError):
        store.add("b", node_id="fixed")


def test_retrieval_ranks_by_similarity(store):
    store.add("the capital of France is Paris")
    target = store.add("Basil Mahfouz Al-Kuwaiti's father was a florist")
    store.add("quantum chromodynamics and the strong force")

    res = store.retrieve("Who was Basil Mahfouz Al-Kuwaiti's father?", k=1)
    assert res.node_ids == [target.node_id]
    assert res.scores[0] > 0


def test_retrieval_enforces_the_blocklist_and_reports_it(store):
    blocked = store.add("Basil Mahfouz Al-Kuwaiti's father was a florist")
    store.add("an unrelated fact about geography")
    blocklist = IDBlocklist([blocked.node_id])

    res = store.retrieve("Basil Mahfouz Al-Kuwaiti father", k=5, blocklist=blocklist)

    assert blocked.node_id not in res.node_ids
    # Suppressed ids are RECORDED, not discarded: "the blocklist worked and the item
    # surfaced anyway" needs both halves to be a finding.
    assert blocked.node_id in res.blocked_node_ids


def test_retrieval_overfetches_so_k_is_not_silently_shrunk(store):
    """Suppressing blocked hits must not reduce how many live nodes come back."""
    blocked = [store.add(f"blocked secret number {i}") for i in range(3)]
    live = [store.add(f"live document number {i}") for i in range(5)]
    blocklist = IDBlocklist([n.node_id for n in blocked])

    res = store.retrieve("document number", k=5, blocklist=blocklist)

    assert len(res.nodes) == 5
    assert set(res.node_ids) <= {n.node_id for n in live}


def test_retrieval_skips_deleted_nodes(store):
    a = store.add("content to delete")
    store.add("content to keep")
    store.delete(a.node_id)

    res = store.retrieve("content", k=5)
    assert a.node_id not in res.node_ids


def test_deleted_nodes_stay_indexed_as_tombstones(store):
    """If deletion dropped nodes from the index, the blocklist would be redundant and
    invariant 1 would be vacuously true. See MemoryNode.returnable."""
    a = store.add("content to delete")
    store.delete(a.node_id)

    assert a.node_id in store.indexed_ids()
    assert a.node_id not in store.returnable_ids()


def test_suppression_is_attributed_to_the_blocklist_not_the_tombstone(store):
    """A blocklisted deletion reports the id; a bare deletion does not."""
    from rdl.memory.blocklist import IDBlocklist as _ID

    a = store.add("Basil Mahfouz Al-Kuwaiti's father was a florist")
    bl = _ID()
    store.delete(a.node_id, blocklist=bl)
    assert (
        a.node_id
        in store.retrieve("Basil Mahfouz father florist", k=5, blocklist=bl).blocked_node_ids
    )

    b = store.add("Nikolai Abilov wrote African American literature")
    store.delete(b.node_id)  # no blocklist: the C0 baseline
    res = store.retrieve("Nikolai Abilov African American literature", k=5)
    assert b.node_id not in res.node_ids
    assert res.blocked_node_ids == [], "no defence configured, so nothing to attribute"


def test_retrieval_can_exclude_outdated(store):
    a = store.add("root", source_kind="ingest")
    b = store.add("derived content", parent_ids=[a.node_id])
    store.delete(a.node_id)
    assert store.get(b.node_id).outdated

    assert b.node_id in store.retrieve("derived content", k=5).node_ids
    assert b.node_id not in store.retrieve("derived content", k=5, include_outdated=False).node_ids


def test_retrieval_on_an_empty_store_is_empty(store):
    res = store.retrieve("anything", k=5)
    assert res.nodes == [] and res.scores == []


def test_retrieval_with_k_zero_is_empty(store):
    store.add("something")
    assert store.retrieve("something", k=0).nodes == []


def test_no_blocklist_blocks_nothing(store):
    n = store.add("anything at all")
    res = store.retrieve("anything at all", k=5, blocklist=NoBlocklist())
    assert n.node_id in res.node_ids
    assert res.blocked_node_ids == []


# ------------------------------------------------------------- snapshot / restore --


def test_snapshot_restore_round_trips(seeded_store):
    store, _blocklist, ids = seeded_store
    snap = store.snapshot()
    before = store.stats()

    store.add("a new node written after the snapshot", source_kind="agent_answer")
    assert store.stats()["n_nodes"] == before["n_nodes"] + 1

    store.restore(snap)
    assert store.stats() == before
    for nid in ids:
        assert store.get(nid).deleted


def test_snapshot_is_a_deep_copy(store):
    node = store.add("mutable content")
    snap = store.snapshot()
    node.content = "mutated"
    assert snap.nodes[node.node_id].content == "mutable content"


def test_restore_rebuilds_the_index(store):
    a = store.add("retrievable content")
    snap = store.snapshot()
    store.delete(a.node_id)
    assert a.node_id not in store.returnable_ids()

    store.restore(snap)
    assert a.node_id in store.returnable_ids()
    assert a.node_id in store.retrieve("retrievable content", k=5).node_ids


def test_conditions_start_from_identical_state(seeded_store):
    """The property C1/C2/C3 depend on."""
    store, _, _ = seeded_store
    snap = store.snapshot()

    c1 = copy.deepcopy(store)
    c1.restore(snap)
    c3 = copy.deepcopy(store)
    c3.restore(snap)

    assert c1.stats() == c3.stats()
    assert sorted(c1.nodes) == sorted(c3.nodes)
    for nid in c1.nodes:
        assert c1.get(nid).content == c3.get(nid).content
        assert c1.get(nid).deleted == c3.get(nid).deleted


def test_deepcopy_is_independent(seeded_store):
    store, _, _ = seeded_store
    other = copy.deepcopy(store)
    other.add("only in the copy", source_kind="agent_answer")
    assert len(other) == len(store) + 1


# ------------------------------------------------------------------------- misc --


def test_add_node_accepts_a_prebuilt_node(store):
    node = MemoryNode(content="prebuilt", source_kind="ingest")
    store.add_node(node)
    assert store.get(node.node_id) is node
    assert node.embedding is not None


def test_stats_counts_parametric_nodes(store):
    store.add("ingested", source_kind="ingest")
    store.add("parametric answer", source_agent="B", source_kind="agent_answer")
    root = store.add("root", source_kind="ingest")
    store.add(
        "derived answer", source_agent="B", source_kind="agent_answer", parent_ids=[root.node_id]
    )

    assert store.stats()["n_parametric"] == 1, "only the parentless agent_answer counts"


def test_index_dim_must_match_embedder():
    from rdl.memory.index import NumpyBruteForce

    with pytest.raises(ValueError, match="dim"):
        MemoryStore(index=NumpyBruteForce(dim=32), embedding_dim=64)
