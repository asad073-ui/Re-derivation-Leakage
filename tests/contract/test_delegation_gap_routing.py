"""The delegation gap can only be measured where abstention routing is in play.

THE REGRESSION (ADR-0049). ADR-0041 moved C3D/C3S/C3C to unconditional routing so the
handoff would be the only variable between them. The retain control arm inherited the
treatment's routing, so for those three conditions

    delegation_rate(forget) = 1     delegation_rate(retain) = 1     gap = 0

against a pre-registered 0.15 — and ADR-0046 had just made `delegation_gap_ok` blocking.
The two fixes composed into a guaranteed FAIL for precisely the conditions the experiment
exists to compare, and the abstention-routed forget arm that WAS being run was not the
arm the gap was computed from.

These tests demonstrate the underlying fact rather than asserting the plumbing: under
unconditional routing the gap is 0 whatever the agents do, and under abstention routing it
tracks what agent A actually knows.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rdl.agents.abstention import LexicalDetector
from rdl.agents.llm_agent import LLMAgent
from rdl.cli.run_condition import ECOLOGICAL_POLICY, execute_condition
from rdl.config import compose as compose_cfg
from rdl.config import validate
from rdl.eval.controls import FAIL, PASS, compute_controls, delegation_rate
from rdl.hardware import detect
from rdl.models.stub import StubLM

CONDITIONS = Path(__file__).resolve().parents[2] / "configs" / "conditions"


@pytest.fixture
def cfg():
    return validate(
        compose_cfg(
            CONDITIONS / "C3D.yaml",
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
    """Agent A knows the SECOND half and abstains on the first — a stand-in for a
    checkpoint that forgot some items and not others."""
    masked = [it.question for it in tofu_items[: len(tofu_items) // 2]]
    a = LLMAgent(
        "A", StubLM(qa_pairs, knowledge_mask=masked, model_id="stub_a"), detector=LexicalDetector()
    )
    b = LLMAgent(
        "B",
        StubLM(qa_pairs, knowledge_mask=list(qa_pairs), model_id="stub_b"),
        detector=LexicalDetector(),
    )
    return [a, b]


def _arm(cfg, items, agents, policy):
    return execute_condition(
        cfg, items, detect(), seed=0, delegation_override=policy, agents_override=agents
    )


def test_under_unconditional_routing_every_episode_delegates(cfg, agents, tofu_items):
    """Which is the point of the arm — and exactly why the gap cannot be read off it."""
    arm = _arm(cfg, tofu_items, agents, "always_delegate")
    assert delegation_rate(arm.transcripts) == 1.0


def test_the_gap_is_structurally_zero_under_the_treatment_routing(cfg, agents, tofu_items):
    """Both sides are 1, so the gap is 0 whether or not routing tracks forgetting.

    This is the measurement the shipped code was making for C3D/C3S/C3C. The number is not
    merely wrong; it carries no information in either direction.
    """
    known = tofu_items[len(tofu_items) // 2 :]  # agent A answers these
    forget_arm = _arm(cfg, tofu_items, agents, "always_delegate")
    retain_like = _arm(cfg, known, agents, "always_delegate")

    rep = compute_controls(
        forget_transcripts=forget_arm.transcripts,
        retain_transcripts=retain_like.transcripts,
        delegation_gap_policy="always_delegate",
        is_multi_agent=True,
        has_retain_arm=True,
    )
    assert rep.delegation_rate_forget == rep.delegation_rate_retain == 1.0
    assert rep.delegation_gap == 0.0
    assert rep.verdicts["delegation_gap_ok"] == FAIL
    assert any("by construction" in n for n in rep.notes)


def test_under_abstention_routing_the_gap_tracks_what_agent_a_knows(cfg, agents, tofu_items):
    """The same agents, the same items, the routing the claim is about — and now the gap
    is a real measurement: A abstains on the masked half and answers the rest."""
    masked = tofu_items[: len(tofu_items) // 2]
    known = tofu_items[len(tofu_items) // 2 :]

    forget_arm = _arm(cfg, masked, agents, ECOLOGICAL_POLICY)
    retain_arm = _arm(cfg, known, agents, ECOLOGICAL_POLICY)

    rep = compute_controls(
        forget_transcripts=forget_arm.transcripts,
        retain_transcripts=retain_arm.transcripts,
        delegation_gap_policy=ECOLOGICAL_POLICY,
        is_multi_agent=True,
        has_retain_arm=True,
    )
    assert rep.delegation_rate_forget == 1.0, "A abstained on everything it forgot"
    assert rep.delegation_rate_retain == 0.0, "and answered everything it retained"
    assert rep.delegation_gap == 1.0
    assert rep.verdicts["delegation_gap_ok"] == PASS
    assert rep.to_dict()["delegation"]["measured_under"] == ECOLOGICAL_POLICY


def test_the_ecological_policy_is_abstention_triggered():
    """Named once, so the runner, the control and the report cannot drift apart."""
    assert ECOLOGICAL_POLICY == "abstention_triggered"
