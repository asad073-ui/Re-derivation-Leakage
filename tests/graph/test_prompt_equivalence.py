"""The prompt wrapper is identical across arms; only the content differs.

If this ever fails, MA-CONTROL stops being a prompt-matched control and every
"the peer message caused it" claim in the paper becomes unsupported.
"""

from __future__ import annotations

from rdl.graph.envelope import derive_envelope
from rdl.graph.prompt_builder import (
    PEER_PREFIX,
    build_agent_prompt,
    prompt_sha256,
    role_system_prompt,
)
from rdl.models.stub import parse_prompt


def _peer(text: str, src="B", dst="D"):
    return derive_envelope(kind="agent_output", content=text, source_node=src, dest_node=dst)


def test_control_and_leak_wrappers_are_byte_identical_in_shape():
    leak = build_agent_prompt(question="Who was X?", peer_envelopes=[_peer("X was a florist.")])
    control = build_agent_prompt(question="Who was X?", peer_envelopes=[_peer("Y was a poet.")])
    # Same number of lines, same prefixes, same question line. Only the payload differs.
    assert leak.count(PEER_PREFIX) == control.count(PEER_PREFIX) == 1
    assert len(leak.splitlines()) == len(control.splitlines())
    assert parse_prompt(leak).question == parse_prompt(control).question == "Who was X?"


def test_prompt_round_trips_through_the_stub_parser():
    prompt = build_agent_prompt(
        question="In which city was Raven Marais born?",
        memory_texts=["an earlier note"],
        peer_envelopes=[_peer("a peer said something")],
    )
    parsed = parse_prompt(prompt)
    assert parsed.question == "In which city was Raven Marais born?"
    assert len(parsed.context) == 2


def test_empty_payloads_are_dropped_not_rendered_as_blank_blocks():
    with_empty = build_agent_prompt(question="Q?", peer_envelopes=[_peer("   ")])
    bare = build_agent_prompt(question="Q?")
    assert with_empty == bare


def test_prompt_hash_covers_the_system_message():
    a = prompt_sha256("same user turn", "system one")
    b = prompt_sha256("same user turn", "system two")
    assert a != b, "a cache keyed on this would serve the proposer's answer to the finalizer"


def test_roles_have_distinct_stable_system_prompts():
    roles = ("proposer", "analyst", "reviewer", "integrator", "finalizer")
    prompts = [role_system_prompt(r) for r in roles]
    assert len(set(prompts)) == len(roles)
    assert role_system_prompt("analyst", "override") == "override"


def test_tool_payloads_are_labelled_differently_from_peers():
    tool = derive_envelope(kind="tool_response", content="a search result", source_node="tool")
    prompt = build_agent_prompt(question="Q?", peer_envelopes=[tool])
    assert "A tool returned:" in prompt
    assert PEER_PREFIX not in prompt
