"""C1 — the leakage floor, runnable end to end before a GPU exists.

C1 is SBU's claimed behaviour: two unlearned agents, the id blocklist enforced, and no
write-back. It defines the baseline the pre-registered gate subtracts, so the property
that matters is that **nothing durable accumulates**. If C1's floor is not near zero,
`C3 - C1` measures nothing.
"""

from __future__ import annotations

from rdl.agents.delegation import build_delegation_policy
from rdl.agents.writer import build_write_policy
from rdl.eval.containment import containment, recall_table, sys_recall_at_k
from rdl.eval.laundering import laundered_items
from rdl.memory.invariants import check_invariant_1, check_invariant_2
from rdl.orchestrator.loop import EpisodePolicies, run_episodes


def _run_c1(seeded_store, agent_a, agent_b, items):
    store, blocklist, _ = seeded_store
    policies = EpisodePolicies(
        delegation=build_delegation_policy("abstention_triggered", 1),
        write=build_write_policy("disabled"),  # THE defining property of C1
        blocklist=blocklist,
        retrieval_k=5,
    )
    transcripts = run_episodes(
        [(it.question, it.item_id) for it in items],
        [agent_a, agent_b],
        store,
        policies,
        condition="C1",
    )
    return store, blocklist, transcripts


def test_c1_writes_nothing_to_the_store(seeded_store, agent_a, agent_b_unlearned, tofu_items):
    store, _, transcripts = _run_c1(seeded_store, agent_a, agent_b_unlearned, tofu_items)

    assert all(t.memory_writes() == [] for t in transcripts)
    assert store.stats()["n_nodes"] == len(tofu_items), "only the ingested (deleted) nodes"
    assert store.stats()["n_deleted"] == len(tofu_items)


def test_c1_recall_floor_is_zero(seeded_store, agent_a, agent_b_unlearned, tofu_items):
    """The number the pre-registered gate subtracts."""
    store, _, transcripts = _run_c1(seeded_store, agent_a, agent_b_unlearned, tofu_items)

    results = [
        containment(t, it.answer, "normalised", store=store)
        for t, it in zip(transcripts, tofu_items, strict=True)
    ]
    table = recall_table(results, k=5)

    assert table["persistent_store_after_episode"] == 0.0
    assert table["final_answer"] == 0.0
    assert sys_recall_at_k(results, k=5) == 0.0


def test_c1_both_invariants_hold_throughout(seeded_store, agent_a, agent_b_unlearned, tofu_items):
    store, blocklist, _ = _run_c1(seeded_store, agent_a, agent_b_unlearned, tofu_items)

    inv1 = check_invariant_1(store, blocklist)
    inv2 = check_invariant_2(store, store.dag)
    assert inv1.satisfied, inv1.violations
    assert inv2.satisfied, inv2.violations


def test_c1_laundering_rate_is_a_flagged_zero(seeded_store, agent_a, agent_b_unlearned, tofu_items):
    """Nothing recovered means an UNDEFINED rate, flagged — not a clean pass."""
    store, blocklist, transcripts = _run_c1(seeded_store, agent_a, agent_b_unlearned, tofu_items)

    report = laundered_items(
        transcripts,
        store,
        store.dag,
        blocklist,
        [{"item_id": it.item_id, "answer": it.answer} for it in tofu_items],
    )
    assert report.n_recovered == 0
    assert report.rate == 0.0
    assert any("undefined" in n for n in report.notes)


def test_c1_retrieval_is_blocked_every_time(seeded_store, agent_a, agent_b_unlearned, tofu_items):
    _, _, transcripts = _run_c1(seeded_store, agent_a, agent_b_unlearned, tofu_items)

    for t in transcripts:
        for r in t.retrievals():
            assert r.returned_node_ids == []
            assert r.blocked_node_ids, "I1 fired and the transcript recorded it"


def test_c1_with_a_knowing_delegate_still_persists_nothing(
    seeded_store, agent_a, agent_b_full, tofu_items
):
    """Isolates the write policy from B's knowledge: even a fully knowledgeable B
    leaves no durable trace when write-back is off."""
    store, _, transcripts = _run_c1(seeded_store, agent_a, agent_b_full, tofu_items)

    results = [
        containment(t, it.answer, "normalised", store=store)
        for t, it in zip(transcripts, tofu_items, strict=True)
    ]
    table = recall_table(results, k=5)

    assert table["final_answer"] == 1.0, "B answered every question out loud"
    assert table["persistent_store_after_episode"] == 0.0, "...but nothing was persisted"
