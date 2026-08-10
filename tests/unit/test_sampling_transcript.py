from __future__ import annotations

from rdl.agents.abstention import LexicalDetector
from rdl.agents.llm_agent import LLMAgent
from rdl.agents.writer import build_write_policy
from rdl.memory.blocklist import SemanticBlocklist
from rdl.memory.store import MemoryStore
from rdl.models.stub import GenerationRequest, StubLM
from rdl.orchestrator.loop import EpisodePolicies, run_episode


def test_trajectory_provenance_and_blocked_write_attempt_are_typed() -> None:
    question, answer = "Who is the person?", "Ada Lovelace"
    agent = LLMAgent("A", StubLM({question: answer}), detector=LexicalDetector())
    blocklist = SemanticBlocklist([answer], threshold=0.4)
    tr = run_episode(
        question,
        [agent],
        MemoryStore(),
        EpisodePolicies(write=build_write_policy("sanitized", threshold=0.4), blocklist=blocklist),
        item_id="item-1",
        sample_id="3",
        trajectory_id="W-A:item-1:3",
        prompt_sha256="prompt",
        decoding_sha256="decode",
        generation_requests={"A": GenerationRequest(do_sample=True, seed=123)},
    )
    attempts = tr.write_attempts()
    assert len(attempts) == 1
    assert not attempts[0].allowed
    assert attempts[0].matched_reference == answer
    assert attempts[0].trajectory_id == "W-A:item-1:3"
    assert tr.memory_writes() == []
