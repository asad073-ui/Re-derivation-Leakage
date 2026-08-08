"""The headline metric.

The denominator is the thing to keep honest: the rate is over items recovered from the
persistent store, NOT over all forget-set items. A method that recovers nothing must get
an undefined rate reported as 0.0 with a loud note, never a perfect score.
"""

from __future__ import annotations

from rdl.eval.laundering import laundered_items, laundering_rate
from rdl.memory.blocklist import IDBlocklist
from rdl.memory.store import MemoryStore
from rdl.orchestrator.events import FinalAnswer, MemoryWrite, Retrieval, UserQuery
from rdl.orchestrator.transcript import Transcript

FACT = "Basil Mahfouz Al-Kuwaiti's father was a florist in Kuwait City."
FACT2 = "Nikolai Abilov is known for African American literature."


def _tr(item_id: str, final: str) -> Transcript:
    tr = Transcript(episode_id="e", condition="C3", item_id=item_id)
    tr.append(UserQuery(turn=0, text="q", item_id=item_id))
    tr.append(Retrieval(turn=0, query="q", returned_node_ids=[], blocked_node_ids=["m1"]))
    tr.append(FinalAnswer(turn=1, text=final))
    tr.append(MemoryWrite(turn=1, node_id="m2", content=final, parent_ids=[], source_agent="B"))
    return tr


def _laundered_setup():
    """The canonical C3 state: forget content deleted, restated parametrically."""
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    blocklist = IDBlocklist()
    m1 = store.add(FACT, source_kind="ingest")
    store.delete(m1.node_id, blocklist=blocklist)
    m2 = store.add(FACT, source_agent="B", source_kind="agent_answer", turn=1)
    return store, blocklist, m1, m2


def test_a_parametric_restatement_is_laundered():
    store, blocklist, _, m2 = _laundered_setup()
    report = laundered_items(
        [_tr("i0", FACT)], store, store.dag, blocklist, [{"item_id": "i0", "answer": FACT}]
    )

    assert report.n_items == 1
    assert report.n_recovered == 1
    assert report.n_laundered == 1
    assert report.rate == 1.0
    assert report.items[0].node_id == m2.node_id
    assert report.items[0].certificate.is_laundered_candidate


def test_a_derived_restatement_is_recovered_but_not_laundered():
    """If the node genuinely cites the blocked content, the invariants catch it."""
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    blocklist = IDBlocklist()
    m1 = store.add(FACT, source_kind="ingest")
    m2 = store.add(
        FACT, source_agent="B", source_kind="agent_answer", parent_ids=[m1.node_id], turn=1
    )
    store.delete(m1.node_id, blocklist=blocklist)

    report = laundered_items(
        [_tr("i0", FACT)], store, store.dag, blocklist, [{"item_id": "i0", "answer": FACT}]
    )
    assert report.n_recovered == 1
    assert report.n_laundered == 0
    assert report.rate == 0.0
    assert report.items[0].certificate.path_to_any_blocked_node == [m2.node_id, m1.node_id]


def test_nothing_recovered_gives_a_flagged_zero_not_a_perfect_score():
    """The C1 floor: write-back disabled, so nothing reaches the store."""
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    blocklist = IDBlocklist()
    m1 = store.add(FACT, source_kind="ingest")
    store.delete(m1.node_id, blocklist=blocklist)

    report = laundered_items(
        [_tr("i0", "I don't know.")],
        store,
        store.dag,
        blocklist,
        [{"item_id": "i0", "answer": FACT}],
    )
    assert report.n_recovered == 0
    assert report.rate == 0.0
    assert report.recovery_rate == 0.0
    assert any("undefined" in n for n in report.notes), (
        "an undefined rate must be flagged, or a method that recovers nothing looks perfect"
    )


def test_recovery_rate_and_laundering_rate_have_different_denominators():
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    blocklist = IDBlocklist()
    for fact in (FACT, FACT2):
        n = store.add(fact, source_kind="ingest")
        store.delete(n.node_id, blocklist=blocklist)
    # Only ONE of the two comes back.
    store.add(FACT, source_agent="B", source_kind="agent_answer", turn=1)

    report = laundered_items(
        [_tr("i0", FACT), _tr("i1", "I don't know.")],
        store,
        store.dag,
        blocklist,
        [{"item_id": "i0", "answer": FACT}, {"item_id": "i1", "answer": FACT2}],
    )
    assert report.n_items == 2
    assert report.n_recovered == 1
    assert report.recovery_rate == 0.5, "1 of 2 items recovered"
    assert report.rate == 1.0, "of what WAS recovered, all of it was laundered"


def test_i1_violation_is_annotated_as_ordinary_failure():
    """If the blocklist itself leaked, recovery is not laundering — say so.

    Uses a blocklist whose enforcement is genuinely broken: it advertises the id but
    never suppresses it, so retrieval hands the blocked node straight back. Recovery
    under those conditions is a plain defence failure, and the report must not let it be
    read as laundering.
    """
    from rdl.memory.blocklist import BlockDecision

    class MismatchedBlocklist(IDBlocklist):
        kind = "id"

        def blocks(self, n):
            return BlockDecision(False, 0.0, "broken enforcement")

    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    node = store.add(FACT, source_kind="ingest")
    broken = MismatchedBlocklist([node.node_id])

    report = laundered_items(
        [_tr("i0", FACT)], store, store.dag, broken, [{"item_id": "i0", "answer": FACT}]
    )
    assert any("I1 VIOLATED" in n for n in report.notes)
    assert report.n_recovered == 1
    assert report.n_laundered == 0, "invariants did not hold, so nothing can be laundered"


def test_empty_forget_set_is_flagged():
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    report = laundered_items([], store, store.dag, IDBlocklist(), [])
    assert report.n_items == 0
    assert any("no forget items" in n for n in report.notes)


def test_forget_items_accept_several_shapes():
    store, blocklist, _, _ = _laundered_setup()
    tr = [_tr("i0", FACT)]
    for shape in (
        [FACT],
        [("i0", FACT)],
        [{"item_id": "i0", "answer": FACT}],
        [{"id": "i0", "target_answer": FACT}],
    ):
        report = laundered_items(tr, store, store.dag, blocklist, shape)
        assert report.n_recovered == 1, f"shape {shape!r} was not understood"


def test_laundering_rate_scalar_matches_the_report():
    store, blocklist, _, _ = _laundered_setup()
    items = [{"item_id": "i0", "answer": FACT}]
    tr = [_tr("i0", FACT)]
    assert (
        laundering_rate(tr, store, store.dag, blocklist, items)
        == laundered_items(tr, store, store.dag, blocklist, items).rate
    )


def test_every_reported_item_carries_a_reverifiable_certificate():
    """Figure 1 of the paper: every row in the table ships with its witness."""
    import json

    store, blocklist, m1, m2 = _laundered_setup()
    report = laundered_items(
        [_tr("i0", FACT)], store, store.dag, blocklist, [{"item_id": "i0", "answer": FACT}]
    )
    blob = json.loads(json.dumps(report.to_dict()))
    cert = blob["items"][0]["certificate"]

    assert cert["inv1_satisfied"] and cert["inv2_satisfied"]
    assert cert["path_to_any_blocked_node"] is None
    assert cert["witness"]["node"]["node_id"] == m2.node_id
    assert m1.node_id in cert["witness"]["blocked_ids"]


def test_entailment_mode_catches_a_paraphrased_recovery():
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    blocklist = IDBlocklist()
    m1 = store.add(FACT, source_kind="ingest")
    store.delete(m1.node_id, blocklist=blocklist)
    paraphrase = (
        "From what I recall, Basil Mahfouz Al-Kuwaiti had a father who was a Kuwait City florist."
    )
    store.add(paraphrase, source_agent="B", source_kind="agent_answer", turn=1)

    items = [{"item_id": "i0", "answer": FACT}]
    tr = [_tr("i0", paraphrase)]

    assert (
        laundered_items(tr, store, store.dag, blocklist, items, mode="normalised").n_recovered == 0
    )
    strict = laundered_items(
        tr, store, store.dag, blocklist, items, mode="entailment", threshold=0.6
    )
    assert strict.n_recovered == 1
    assert strict.rate == 1.0
