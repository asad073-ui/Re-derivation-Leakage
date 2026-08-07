"""The orchestrator loop, end to end, on StubLM.

`loop.py` is pure orchestration: no metric logic, no model-specific code. These tests
are what prove it — if any of them ever needs a real model, something leaked into the
loop that does not belong there.
"""

from __future__ import annotations

from rdl.agents.delegation import AlwaysDelegate, NeverDelegate, build_delegation_policy
from rdl.agents.writer import build_write_policy
from rdl.memory.blocklist import IDBlocklist, NoBlocklist
from rdl.orchestrator.loop import EpisodePolicies, run_episode, run_episodes

QUESTION = "Who was Basil Mahfouz Al-Kuwaiti's father?"
FACT = "Basil Mahfouz Al-Kuwaiti's father was a florist in Kuwait City."


def test_single_agent_episode_shape(store, stub_knowing):
    from rdl.agents.llm_agent import LLMAgent

    agent = LLMAgent("A", stub_knowing)
    tr = run_episode(QUESTION, [agent], store, EpisodePolicies(delegation=NeverDelegate()))

    assert [e.kind for e in tr] == ["user_query", "retrieval", "agent_answer", "final_answer"]
    assert tr.final_text == FACT
    assert not tr.delegated


def test_abstention_triggers_delegation(store, agent_a, agent_b_full):
    policies = EpisodePolicies(
        delegation=build_delegation_policy("abstention_triggered", 1),
        write=build_write_policy("disabled"),
    )
    tr = run_episode(QUESTION, [agent_a, agent_b_full], store, policies)

    assert tr.primary_abstained
    assert tr.delegated
    deleg = tr.delegations()[0]
    assert (deleg.from_id, deleg.to_id) == ("A", "B")
    assert tr.final_text == FACT, "B's non-abstaining answer becomes the final answer"


def test_no_delegation_when_the_primary_answers(store, agent_b_full):
    from rdl.agents.llm_agent import LLMAgent

    confident = LLMAgent("A", agent_b_full.lm)
    policies = EpisodePolicies(delegation=build_delegation_policy("abstention_triggered", 1))
    tr = run_episode(QUESTION, [confident, agent_b_full], store, policies)
    assert not tr.delegated


def test_always_delegate_fires_regardless_of_the_primary(store, agent_b_full):
    """The control: routing is removed from the causal path."""
    from rdl.agents.llm_agent import LLMAgent

    confident = LLMAgent("A", agent_b_full.lm)
    tr = run_episode(
        QUESTION, [confident, agent_b_full], store, EpisodePolicies(delegation=AlwaysDelegate(1))
    )
    assert not tr.primary_abstained
    assert tr.delegated


def test_delegation_budget_is_respected(store, agent_a, agent_b_unlearned):
    """Both agents abstain; the loop must not spin."""
    policies = EpisodePolicies(delegation=build_delegation_policy("abstention_triggered", 1))
    tr = run_episode(QUESTION, [agent_a, agent_b_unlearned], store, policies)
    assert len(tr.delegations()) == 1
    assert tr.final_text.strip() == "I don't know."


def test_retrieval_is_recorded_for_both_agents(seeded_store, agent_a, agent_b_full):
    store, blocklist, _ = seeded_store
    policies = EpisodePolicies(
        delegation=build_delegation_policy("abstention_triggered", 1), blocklist=blocklist
    )
    tr = run_episode(QUESTION, [agent_a, agent_b_full], store, policies)

    retrievals = tr.retrievals()
    assert len(retrievals) == 2, "the delegate retrieves under the SAME blocklist"
    assert {r.agent_id for r in retrievals} == {"A", "B"}
    for r in retrievals:
        assert r.returned_node_ids == []
        assert r.blocked_node_ids, "the blocklist suppressed something and said so"


def test_write_back_produces_a_memory_write_event(store, agent_a, agent_b_full):
    policies = EpisodePolicies(
        delegation=build_delegation_policy("abstention_triggered", 1),
        write=build_write_policy("framework_default"),
    )
    tr = run_episode(QUESTION, [agent_a, agent_b_full], store, policies)

    writes = tr.memory_writes()
    assert len(writes) == 1
    assert writes[0].content == FACT
    assert writes[0].parent_ids == [], "parametric answer, empty context, no edges"
    assert writes[0].node_id in store


def test_disabled_write_policy_writes_nothing(store, agent_a, agent_b_full):
    policies = EpisodePolicies(
        delegation=build_delegation_policy("abstention_triggered", 1),
        write=build_write_policy("disabled"),
    )
    tr = run_episode(QUESTION, [agent_a, agent_b_full], store, policies)
    assert tr.memory_writes() == []
    assert len(store) == 0


def test_abstentions_are_not_written_back(store, agent_a, agent_b_unlearned):
    policies = EpisodePolicies(
        delegation=build_delegation_policy("abstention_triggered", 1),
        write=build_write_policy("framework_default"),
    )
    tr = run_episode(QUESTION, [agent_a, agent_b_unlearned], store, policies)
    assert tr.memory_writes() == [], "a refusal carries no content worth persisting"


def test_cited_context_becomes_parent_ids(store, stub_knowing):
    """The other half of the write-path contract: a real citation IS recorded."""
    from rdl.agents.llm_agent import LLMAgent

    source = store.add(FACT, source_kind="ingest")
    agent = LLMAgent("A", stub_knowing)
    policies = EpisodePolicies(
        delegation=NeverDelegate(), write=build_write_policy("framework_default"), retrieval_k=5
    )
    tr = run_episode(QUESTION, [agent], store, policies)

    write = tr.memory_writes()[0]
    assert source.node_id in write.parent_ids, (
        "when a node genuinely was in context, the edge must be recorded — the empty "
        "parent_ids case is meaningful only because this case is not empty"
    )


def test_transcript_metadata_records_the_policies(store, agent_a, agent_b_full):
    policies = EpisodePolicies(
        delegation=build_delegation_policy("always_delegate", 1),
        write=build_write_policy("framework_default"),
        blocklist=IDBlocklist(),
    )
    tr = run_episode(QUESTION, [agent_a, agent_b_full], store, policies, condition="C3", seed=3)

    assert tr.condition == "C3" and tr.seed == 3
    assert tr.meta["delegation_policy"] == "always_delegate"
    assert tr.meta["write_policy"] == "framework_default"
    assert tr.meta["blocklist_kind"] == "id"


def test_every_event_carries_the_episode_id(store, agent_a, agent_b_full):
    tr = run_episode(QUESTION, [agent_a, agent_b_full], store, EpisodePolicies())
    assert tr.episode_id
    assert all(e.episode_id == tr.episode_id for e in tr)


def test_run_episodes_shares_one_persistent_store(store, agent_a, agent_b_full, tofu_items):
    """Cross-episode persistence is the mechanism under test; the store is NOT reset."""
    policies = EpisodePolicies(
        delegation=build_delegation_policy("abstention_triggered", 1),
        write=build_write_policy("framework_default"),
        blocklist=NoBlocklist(),
    )
    queries = [(it.question, it.item_id) for it in tofu_items[:3]]
    transcripts = run_episodes(queries, [agent_a, agent_b_full], store, policies)

    assert len(transcripts) == 3
    assert len(store) == 3, "each episode's answer accumulates in the shared store"
    assert [t.item_id for t in transcripts] == [it.item_id for it in tofu_items[:3]]


def test_loop_is_deterministic(store, agent_a, agent_b_full):
    a = run_episode(QUESTION, [agent_a, agent_b_full], store, EpisodePolicies())
    b = run_episode(QUESTION, [agent_a, agent_b_full], store, EpisodePolicies())
    assert [e.kind for e in a] == [e.kind for e in b]
    assert a.final_text == b.final_text
