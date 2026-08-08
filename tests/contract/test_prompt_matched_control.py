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

from pathlib import Path

import pytest

from rdl.agents.abstention import LexicalDetector
from rdl.agents.llm_agent import LLMAgent
from rdl.cli.run_condition import execute_condition
from rdl.config import ConfigError, load_config, validate
from rdl.config import compose as compose_cfg
from rdl.hardware import detect
from rdl.models.stub import StubLM

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
