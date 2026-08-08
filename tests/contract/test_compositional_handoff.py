"""The C3C handoff, tested where it actually happens: in the loop.

THE BUG THESE TESTS EXIST FOR (ADR-0041). Two conditions in `run_episode` were mutually
exclusive:

    abstention_triggered routing  ->  agent B is called IFF agent A abstained
    the handoff                   ->  A's text is passed IFF agent A did NOT abstain

C3C shipped with `abstention_triggered` inherited from `configs/agents/A_unlearned.yaml`,
so `peer_answers` was empty on every episode B ever ran. C3C was byte-identical to C3D
and `C3C - C3D` was structurally zero — the single-variable contrast the paper's
"re-derivation" claim rests on could not be non-zero.

Nothing caught it. `test_experiment_design_guards.py` checks that the config FLAG cannot
be unset; `test_prompt_style.py` calls `LLMAgent.answer(peer_answers=...)` directly. No
test ran a condition through the loop and asked whether the handoff happened. These do.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from rdl.agents.abstention import LexicalDetector
from rdl.agents.delegation import build_delegation_policy
from rdl.agents.llm_agent import LLMAgent
from rdl.agents.writer import build_write_policy
from rdl.config import ConfigError, load_config
from rdl.models.stub import StubLM
from rdl.orchestrator.loop import EpisodePolicies, run_episode

CONDITIONS = Path(__file__).resolve().parents[2] / "configs" / "conditions"


def _policies(*, handoff: bool, routing: str, blocklist):
    return EpisodePolicies(
        delegation=build_delegation_policy(routing, 1),
        write=build_write_policy("framework_default"),
        blocklist=blocklist,
        pass_primary_answer_to_secondary=handoff,
    )


@pytest.fixture
def agents(qa_pairs, tofu_items):
    """A masked on everything (so it abstains), B holding the second half."""
    a = LLMAgent(
        "A",
        StubLM(qa_pairs, knowledge_mask=list(qa_pairs), model_id="stub_a"),
        detector=LexicalDetector(),
    )
    masked = [it.question for it in tofu_items[: len(tofu_items) // 2]]
    b = LLMAgent(
        "B", StubLM(qa_pairs, knowledge_mask=masked, model_id="stub_b"), detector=LexicalDetector()
    )
    return [a, b]


# =====================================================================================
# the handoff happens, and leaves a witness
# =====================================================================================


def test_the_handoff_fires_when_the_primary_abstains(agents, tofu_items, seeded_store):
    """THE regression. Agent A abstains on every item here — which under the old code was
    exactly the case where A's text was withheld, so B never saw anything."""
    store, blocklist, _ = seeded_store
    it = tofu_items[0]

    tr = run_episode(
        it.question,
        agents,
        store,
        _policies(handoff=True, routing="always_delegate", blocklist=blocklist),
        item_id=it.item_id,
        condition="C3C",
    )

    assert tr.agent_answers()[0].abstained, "the primary abstained (the old suppression case)"
    assert tr.n_handoffs == 1, "the handoff must fire anyway: an abstention is information"
    handoff = tr.handoffs()[0]
    assert handoff.included_abstention is True


def test_the_handoff_carries_the_primarys_exact_output(agents, tofu_items, seeded_store):
    """Not a summary, not a flag: the exact string, with a hash so a transcript can be
    checked against agent A's own AgentAnswer without trusting either copy."""
    store, blocklist, _ = seeded_store
    # Second half: agent B answers, agent A still abstains.
    it = tofu_items[-1]

    tr = run_episode(
        it.question,
        agents,
        store,
        _policies(handoff=True, routing="always_delegate", blocklist=blocklist),
        item_id=it.item_id,
        condition="C3C",
    )

    a_answer = tr.agent_answers()[0]
    handoff = tr.handoffs()[0]
    assert handoff.from_id == "A" and handoff.to_id == "B"
    assert handoff.text == a_answer.text
    assert handoff.text_sha256 == hashlib.sha256(a_answer.text.encode("utf-8")).hexdigest()


def test_no_handoff_means_no_handoff_event(agents, tofu_items, seeded_store):
    """C3D is the comparator. If the event appeared here the contrast would be gone."""
    store, blocklist, _ = seeded_store
    tr = run_episode(
        tofu_items[-1].question,
        agents,
        store,
        _policies(handoff=False, routing="always_delegate", blocklist=blocklist),
        item_id=tofu_items[-1].item_id,
        condition="C3D",
    )
    assert tr.handoffs() == []
    assert tr.meta["n_handoffs"] == 0


def test_the_handoff_reaches_the_delegates_prompt(agents, tofu_items, seeded_store):
    """The event is evidence of intent; this is evidence of effect.

    The delegate's stub records every prompt it was given, so "B received A's text" is
    checked against what the model actually saw rather than against our own bookkeeping.
    """
    store, blocklist, _ = seeded_store
    it = tofu_items[-1]
    b_lm = agents[1].lm

    tr = run_episode(
        it.question,
        agents,
        store,
        _policies(handoff=True, routing="always_delegate", blocklist=blocklist),
        item_id=it.item_id,
        condition="C3C",
    )

    a_text = tr.agent_answers()[0].text
    assert b_lm.call_log, "the delegate was called"
    assert any(a_text.strip() in call["prompt"] for call in b_lm.call_log), (
        "agent B's prompt must contain agent A's answer — otherwise C3C is C3D"
    )


def test_transcript_meta_counts_the_handoff(agents, tofu_items, seeded_store):
    """`make-report` blocks a condition that declares a handoff and recorded none, so the
    count has to survive into the run report."""
    store, blocklist, _ = seeded_store
    tr = run_episode(
        tofu_items[0].question,
        agents,
        store,
        _policies(handoff=True, routing="always_delegate", blocklist=blocklist),
        item_id=tofu_items[0].item_id,
        condition="C3C",
    )
    assert tr.meta["handoff"] is True
    assert tr.meta["n_handoffs"] == 1
    assert tr.meta["peer_answer_count"] == 1


# =====================================================================================
# the configuration can no longer express the broken combination
# =====================================================================================


def test_c3c_and_c3d_both_route_unconditionally():
    c3c = load_config(CONDITIONS / "C3C.yaml")
    c3d = load_config(CONDITIONS / "C3D.yaml")
    assert c3c.effective_routing() == "always_delegate"
    assert c3d.effective_routing() == "always_delegate"
    assert c3c.effective_routing() == c3d.effective_routing(), (
        "the handoff must be the ONLY variable between the estimand arms"
    )


def test_abstention_routed_c3c_is_refused():
    """The exact configuration that shipped. It cannot be composed any more."""
    with pytest.raises(ConfigError, match="must route unconditionally"):
        load_config(CONDITIONS / "C3C.yaml", ["episode.routing=abstention_triggered"])
    with pytest.raises(ConfigError, match="must route unconditionally"):
        load_config(CONDITIONS / "C3D.yaml", ["episode.routing=abstention_triggered"])


def test_a_single_agent_arm_cannot_declare_routing():
    with pytest.raises(ConfigError, match="nowhere to route to"):
        load_config(CONDITIONS / "C1W.yaml", ["episode.routing=always_delegate"])


# =====================================================================================
# the B-alone baseline
# =====================================================================================


def test_b1w_is_agent_b_alone_with_writeback():
    cfg = load_config(CONDITIONS / "B1W.yaml")
    assert cfg.agent_b is None
    assert cfg.agent_a.agent_id == "B"
    assert cfg.writepolicy.mode == "framework_default"
    assert cfg.agent_a.model == load_config(CONDITIONS / "C3D.yaml").agent_b.model, (
        "B1W must load the SAME checkpoint agent B uses inside C3D/C3C, or the "
        "joint-only subtraction is against a different model"
    )


def test_b1w_cannot_quietly_become_a_second_a1w():
    """`joint_only_recovery` subtracts C1W and B1W. If B1W loaded agent A, the
    subtraction would remove A twice and inflate the joint-only count by everything
    agent B alone can already produce."""
    with pytest.raises(ConfigError, match="B-alone baseline"):
        load_config(CONDITIONS / "B1W.yaml", ["agent_a.agent_id=A"])
