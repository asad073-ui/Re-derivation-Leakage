"""The prompt the agent loop sends must be the prompt the reproduction gate sends.

Upstream open-unlearning evaluates TOFU by putting the **bare question** through the
model's chat template, with `system_prompt: "You are a helpful assistant."` (see
`third_party/open-unlearning/configs/model/Llama-3.2-1B-Instruct.yaml`). The agent loop
used to wrap the question in `Question:` / `Answer:` scaffolding, inline the system
message into the user turn, and pass `system_prompt: null`. Agent-loop numbers were
therefore produced under a different prompt from the Days 1-2 gate and could not be put
in the same table as it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rdl.agents.abstention import LexicalDetector
from rdl.agents.llm_agent import LLMAgent
from rdl.models.stub import CONTEXT_MARKER, StubLM, format_prompt, parse_prompt

AGENTS = Path(__file__).resolve().parents[2] / "configs" / "agents"
UPSTREAM_SYSTEM_PROMPT = "You are a helpful assistant."


def test_bare_question_is_the_user_turn():
    assert format_prompt("Who was X's mentor?") == "Who was X's mentor?"


def test_system_prompt_is_not_inlined_into_the_user_turn():
    """It has to reach the tokenizer as a separate `system` role, or the rendered
    template differs from upstream's by an entire message."""
    out = format_prompt("Who?", system=UPSTREAM_SYSTEM_PROMPT)
    assert UPSTREAM_SYSTEM_PROMPT not in out


def test_qa_scaffold_is_still_available_and_still_different():
    out = format_prompt("Who?", style="qa_scaffold")
    assert "Question: Who?" in out and out.endswith("Answer:")


def test_unknown_style_is_rejected():
    with pytest.raises(ValueError, match="unknown prompt style"):
        format_prompt("Who?", style="freeform")


@pytest.mark.parametrize("style", ["openunlearning", "qa_scaffold"])
def test_round_trip_recovers_question_and_context(style):
    ctx = ["Alpha beta gamma.", "Delta epsilon."]
    parsed = parse_prompt(format_prompt("Who was X's mentor?", ctx, style=style))
    assert parsed.question == "Who was X's mentor?"
    assert parsed.context == ctx


def test_context_block_does_not_swallow_the_question():
    """Without the blank-line separator the stub answers the last context line."""
    prompt = format_prompt("Who?", ["some retrieved fact"])
    assert CONTEXT_MARKER in prompt
    assert parse_prompt(prompt).question == "Who?"


def test_agent_configs_carry_the_upstream_system_prompt():
    for p in sorted(AGENTS.glob("*.yaml")):
        with p.open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        assert cfg.get("system_prompt") == UPSTREAM_SYSTEM_PROMPT, p.name


def test_agent_passes_the_system_prompt_to_the_handle_not_the_prompt_string():
    seen = {}

    class Recording(StubLM):
        def generate(self, prompt, max_new_tokens=128, *, system=None):
            seen["prompt"] = prompt
            seen["system"] = system
            return "an answer"

    agent = LLMAgent(
        "A",
        Recording({}, model_id="rec"),
        detector=LexicalDetector(),
        system_prompt=UPSTREAM_SYSTEM_PROMPT,
    )
    agent.answer("Who was X's mentor?")

    assert seen["prompt"] == "Who was X's mentor?"
    assert seen["system"] == UPSTREAM_SYSTEM_PROMPT


# =====================================================================================
# The compositional handoff
# =====================================================================================


def test_peer_answers_reach_the_prompt():
    seen = {}

    class Recording(StubLM):
        def generate(self, prompt, max_new_tokens=128, *, system=None):
            seen["prompt"] = prompt
            return "b's answer"

    agent = LLMAgent("B", Recording({}, model_id="rec"), detector=LexicalDetector())
    agent.answer("Who?", peer_answers=["A said the mentor was a florist."])

    assert "A said the mentor was a florist." in seen["prompt"]


def test_peer_answers_never_become_derivation_parents():
    """A peer's utterance is not a memory node. Recording it as one would fabricate
    exactly the provenance edge whose absence is the finding."""
    agent = LLMAgent("B", StubLM({}, model_id="s"), detector=LexicalDetector())
    reply = agent.answer("Who?", peer_answers=["A's answer"])
    assert reply.context_node_ids == []
    assert reply.meta["n_peer_answers"] == 1
