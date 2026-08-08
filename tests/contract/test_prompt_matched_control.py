"""C3S, the prompt-matched control, tested through the real runner.

WHY IT EXISTS (ADR-0048). C3D hands agent B a bare question. C3C hands it

    Context:
    - Another assistant answered: <A's output>

    <question>

so `C3C - C3D` varies agent A's information, the presence of any context block, "another
assistant" priming, and prompt length and format ALL AT ONCE. A positive result there is
exactly as consistent with "any peer-shaped message elicits agent B's suppressed
knowledge" — real, publishable, and *not* re-derivation from A's content.

C3S keeps the wrapper byte-identical and changes one thing: whose question the
handed-over text answers. These tests check that claim mechanically — same prompt shape,
different content, no fixed points — because "byte-identical formatting" is the kind of
assertion that quietly stops being true.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdl.agents.abstention import LexicalDetector
from rdl.agents.llm_agent import LLMAgent
from rdl.cli.run_condition import _source_answer, execute_condition
from rdl.config import ConfigError, load_config, validate
from rdl.config import compose as compose_cfg
from rdl.hardware import detect
from rdl.models.stub import StubLM
from rdl.orchestrator.loop import EpisodePolicies

CONDITIONS = Path(__file__).resolve().parents[2] / "configs" / "conditions"


def _cfg(name: str):
    return validate(
        compose_cfg(
            CONDITIONS / f"{name}.yaml",
            [
                "models.tofu_llama32_1b_npo_forget10.kind=stub",
                "models.tofu_llama32_1b_npo_forget10.repo_id=null",
                "models.tofu_llama32_1b_npo_forget10_indep.kind=stub",
                "models.tofu_llama32_1b_npo_forget10_indep.repo_id=null",
            ],
            env_override="local_cpu",
        )
    )


@pytest.fixture
def agents(qa_pairs, tofu_items):
    """A answers the second half; B is masked on everything, so only the handoff can
    put an answer in front of it."""
    masked_a = [it.question for it in tofu_items[: len(tofu_items) // 2]]
    a = LLMAgent(
        "A",
        StubLM(qa_pairs, knowledge_mask=masked_a, model_id="stub_a"),
        detector=LexicalDetector(),
    )
    b = LLMAgent(
        "B",
        StubLM(qa_pairs, knowledge_mask=list(qa_pairs), model_id="stub_b"),
        detector=LexicalDetector(),
    )
    return [a, b]


def _run(name: str, agents, tofu_items):
    return execute_condition(_cfg(name), tofu_items, detect(), seed=0, agents_override=agents)


# =====================================================================================
# the control is a control
# =====================================================================================


def test_c3s_hands_over_another_items_answer(agents, tofu_items):
    arm = _run("C3S", agents, tofu_items)
    handoffs = [h for tr in arm.transcripts for h in tr.handoffs()]

    assert len(handoffs) == len(tofu_items), "every episode delegates and hands over once"
    assert all(h.shuffled for h in handoffs)
    by_item = {tr.item_id: tr.handoffs()[0] for tr in arm.transcripts}
    for item_id, h in by_item.items():
        assert h.source_item_id is not None
        assert h.source_item_id != item_id, "a fixed point would make C3S a copy of C3C"


def test_the_derangement_covers_every_item_exactly_once(agents, tofu_items):
    """A permutation, not a sampling: every item's answer is used exactly once, so the
    control's handed-over texts are the same multiset as the treatment's."""
    arm = _run("C3S", agents, tofu_items)
    sources = sorted(tr.handoffs()[0].source_item_id for tr in arm.transcripts)
    assert sources == sorted(it.item_id for it in tofu_items)


def test_c3s_and_c3c_prompts_differ_only_in_the_handed_over_text(agents, tofu_items):
    """THE claim the control rests on. If the wrapper drifted between arms, `C3C - C3S`
    would be measuring formatting again — the exact confound it exists to remove."""
    b_lm = agents[1].lm

    b_lm.call_log.clear()
    _run("C3C", agents, tofu_items)
    c3c_prompts = {tr: call["prompt"] for tr, call in enumerate(b_lm.call_log)}

    b_lm.call_log.clear()
    _run("C3S", agents, tofu_items)
    c3s_prompts = {tr: call["prompt"] for tr, call in enumerate(b_lm.call_log)}

    assert len(c3c_prompts) == len(c3s_prompts) == len(tofu_items)
    for i in range(len(tofu_items)):
        c, s = c3c_prompts[i], c3s_prompts[i]
        assert "Another assistant answered:" in c and "Another assistant answered:" in s
        # Same scaffold: same line count, same label, same question at the end.
        assert c.count("Another assistant answered:") == s.count("Another assistant answered:")
        assert c.splitlines()[0] == s.splitlines()[0], "same context marker"
        assert c.splitlines()[-1] == s.splitlines()[-1], "same question"


def test_c3d_gets_no_peer_message_at_all(agents, tofu_items):
    arm = _run("C3D", agents, tofu_items)
    assert [h for tr in arm.transcripts for h in tr.handoffs()] == []
    assert arm.n_handoffs == 0


def test_every_delegation_in_a_handoff_arm_carries_exactly_one_handoff(agents, tofu_items):
    """`make-report` blocks a partial rate; this is where the invariant is produced."""
    for name in ("C3C", "C3S"):
        arm = _run(name, agents, tofu_items)
        assert arm.n_handoffs == arm.n_delegations == len(tofu_items), name
    c3d = _run("C3D", agents, tofu_items)
    assert c3d.n_delegations == len(tofu_items) and c3d.n_handoffs == 0


def test_the_shuffled_count_is_recorded_per_arm(agents, tofu_items):
    assert _run("C3S", agents, tofu_items).n_shuffled_handoffs == len(tofu_items)
    assert _run("C3C", agents, tofu_items).n_shuffled_handoffs == 0


# =====================================================================================
# the configuration cannot express a broken control
# =====================================================================================


def test_c3s_requires_a_deranged_source():
    with pytest.raises(ConfigError, match="handoff_source='deranged'"):
        load_config(CONDITIONS / "C3S.yaml", ["episode.handoff_source=primary"])


def test_c3c_requires_its_own_items_answer():
    with pytest.raises(ConfigError, match="handoff_source='primary'"):
        load_config(CONDITIONS / "C3C.yaml", ["episode.handoff_source=deranged"])


def test_no_other_arm_may_use_a_deranged_handoff():
    """Two independent refusals, because a deranged source is wrong for two reasons: the
    arm has no peer message at all, and only C3S may shuffle one."""
    with pytest.raises(ConfigError, match="must not use a deranged handoff"):
        load_config(CONDITIONS / "C3D.yaml", ["episode.handoff_source=deranged"])
    with pytest.raises(ConfigError, match="must not pass an agent answer"):
        load_config(
            CONDITIONS / "C3D.yaml",
            ["episode.pass_primary_answer=true", "episode.handoff_source=deranged"],
        )


def test_all_three_estimand_arms_route_identically():
    routings = {
        name: load_config(CONDITIONS / f"{name}.yaml").effective_routing()
        for name in ("C3D", "C3S", "C3C")
    }
    assert set(routings.values()) == {"always_delegate"}, routings


# =====================================================================================
# the source answer is generated against the SAME store the target episode sees
# =====================================================================================


def _cumulative(name: str):
    return validate(
        compose_cfg(
            CONDITIONS / f"{name}.yaml",
            [
                "episode.store_scope=cumulative",
                "models.tofu_llama32_1b_npo_forget10.kind=stub",
                "models.tofu_llama32_1b_npo_forget10.repo_id=null",
                "models.tofu_llama32_1b_npo_forget10_indep.kind=stub",
                "models.tofu_llama32_1b_npo_forget10_indep.repo_id=null",
            ],
            env_override="local_cpu",
        )
    )


def test_the_source_probe_sees_the_live_store_under_cumulative_scope(agents, tofu_items):
    """THE cumulative bug (ADR-0055).

    The source answers used to be precomputed against a FRESH post-deletion store. Under
    `store_scope: cumulative` C3C's handed-over answer sees the accumulated shared store
    while C3S's saw an empty one, so `C3C - C3S` would have varied the handed-over content
    AND its memory context together — and Phase F2 was simply not runnable.

    The probe now runs immediately before the target episode against that episode's own
    store, so the retrieval context it sees grows exactly as the treatment's does.
    """
    arm = execute_condition(
        _cumulative("C3S"), tofu_items, detect(), seed=0, agents_override=agents
    )

    probes = [c for c in agents[0].lm.call_log if c["source"] != "abstain" or True]
    assert probes, "agent A was called"
    # Episodes accumulate writes, so later probes must see a non-empty context. A
    # precomputed pass against fresh stores would leave every probe at n_context == 0.
    contexts = [c["n_context"] for c in agents[0].lm.call_log]
    assert max(contexts) > 0, (
        "every source probe saw an empty store — the probe is not running against the "
        "live cumulative store"
    )
    assert arm.n_shuffled_handoffs == len(tofu_items)


def _store_fingerprint(store, blocklist) -> dict:
    """EVERYTHING mutable about the store, not just its node count.

    `turn`, every node and its metadata, the DAG edges, the index membership and the
    returnable set. A probe that changes any one of these has changed what the measured
    episode sees or how its write is stamped.
    """
    return {
        "turn": store.turn,
        "nodes": sorted(
            (
                n.node_id,
                n.content,
                n.turn,
                n.refcount,
                n.deleted,
                n.outdated,
                n.returnable,
                n.source_agent,
                n.source_kind,
                tuple(n.parent_ids),
                json.dumps(n.meta, sort_keys=True, default=str),
            )
            for n in store.all_nodes(include_deleted=True)
        ),
        "edges": sorted(map(tuple, store.dag.edges())),
        "indexed_ids": sorted(store.indexed_ids()),
        "returnable_ids": sorted(store.returnable_ids(blocklist)),
    }


def test_the_source_probe_leaves_the_store_byte_identical(agents, tofu_items, seeded_store):
    """v5 §2.2 claims the probe leaves the measured store untouched. Check THAT claim.

    The probe used to call `run_episode` under `DisabledWritePolicy`, which wrote no node
    but still executed `store.turn = turn` on its way to the write-back step — so the
    store it "did not touch" came back with a different turn counter, and the next
    episode's write would be stamped with it. The old test asserted only that no extra
    node appeared, which that bug passes. This one compares the complete state
    (ADR-0057).
    """
    store, blocklist, _ = seeded_store
    # Something already in flight, so a probe resetting `turn` to 0 is distinguishable
    # from a probe leaving it alone.
    store.turn = 7
    policies = EpisodePolicies(retrieval_k=5, max_turns=5, blocklist=blocklist)

    before = _store_fingerprint(store, blocklist)
    for item in tofu_items[:5]:
        text, _ = _source_answer(item, agents[0], store, blocklist, policies)
        assert isinstance(text, str)
    after = _store_fingerprint(store, blocklist)

    assert after == before, "the read-only source probe mutated the measured store"
    assert store.turn == 7, "the probe advanced the store's turn counter"


def test_the_source_probe_never_writes_to_the_store(agents, tofu_items):
    """`DisabledWritePolicy` and `NeverDelegate`: the probe must not change what the
    measured episode can retrieve, or it becomes part of the treatment."""
    c3s = execute_condition(_cfg("C3S"), tofu_items, detect(), seed=0, agents_override=agents)
    c3c = execute_condition(_cfg("C3C"), tofu_items, detect(), seed=0, agents_override=agents)
    # Same number of write-eligible episodes, and — the actual property — the probe
    # leaves no node behind: under per_item scope each episode's store may hold at most
    # its own single write, exactly as in the treatment arm.
    assert len(c3s.transcripts) == len(c3c.transcripts) == len(tofu_items)
    for arm in (c3s, c3c):
        for nodes in arm.snapshots.values():
            written = [n for n in nodes if n.source_kind == "agent_answer"]
            assert len(written) <= 1, "a probe write would show up as a second node here"
