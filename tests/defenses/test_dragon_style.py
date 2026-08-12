"""The DRAGON-style baseline: what it does, and what it deliberately does not."""

from __future__ import annotations

import pytest

from rdl.defenses.base import (
    EdgeContext,
    FinalContext,
    NodeInputContext,
    RetrievalContext,
    WriteContext,
)
from rdl.defenses.dragon_style import GUARD_INSTRUCTION, DragonStyleDefense
from rdl.defenses.sanitizer import SAFE_REFUSAL
from rdl.graph.envelope import derive_envelope


def _ctx(question, inputs=(), node="D", depth=2):
    return NodeInputContext(
        node_id=node,
        role="integrator",
        depth=depth,
        question=question,
        inputs=tuple(inputs),
    )


def test_fires_on_an_in_scope_query(detector, tofu_items):
    guard = DragonStyleDefense(detector=detector)
    verdict = guard.on_node_input(_ctx(tofu_items[0].question))
    assert verdict.fired
    assert verdict.forced_output == SAFE_REFUSAL


def test_passes_an_unrelated_query(detector):
    guard = DragonStyleDefense(detector=detector)
    verdict = guard.on_node_input(_ctx("What is the boiling point of water?"))
    assert not verdict.fired
    assert verdict.forced_output is None


def test_guard_prompt_mode_appends_an_instruction_instead_of_refusing(detector, tofu_items):
    guard = DragonStyleDefense(detector=detector, guard_action="guard_prompt")
    verdict = guard.on_node_input(_ctx(tofu_items[0].question))
    assert verdict.fired
    assert verdict.forced_output is None
    assert verdict.guard_system_suffix == GUARD_INSTRUCTION


def test_external_prompt_only_skips_internal_nodes(detector, tofu_items):
    guard = DragonStyleDefense(detector=detector, apply_at="external_prompt_only")
    assert guard.on_node_input(_ctx(tofu_items[0].question, depth=0)).fired
    assert not guard.on_node_input(_ctx(tofu_items[0].question, depth=2)).fired


def test_does_not_enforce_on_edges(detector, tofu_items):
    guard = DragonStyleDefense(detector=detector)
    envelope = derive_envelope(
        kind="agent_output", content=tofu_items[0].answer, source_node="A", dest_node="B"
    )
    verdict = guard.on_edge(EdgeContext(src="A", dst="B", depth=0, envelope=envelope))
    assert verdict.status == "pass"
    assert verdict.envelope.content == tofu_items[0].answer


def test_does_not_guard_persistent_writes(detector, tofu_items):
    guard = DragonStyleDefense(detector=detector)
    envelope = derive_envelope(kind="agent_output", content=tofu_items[0].answer, source_node="A")
    assert guard.on_memory_write(WriteContext(node_id="A", depth=0, envelope=envelope)).allowed


def test_does_not_guard_retrieval(detector, tofu_items):
    guard = DragonStyleDefense(detector=detector)
    verdict = guard.on_retrieval(
        RetrievalContext(
            node_id="A",
            depth=0,
            query=tofu_items[0].question,
            candidates=(("n1", tofu_items[0].answer, ("stub-concept-0000",)),),
        )
    )
    assert verdict.allowed_node_ids == ("n1",)
    assert verdict.withheld_node_ids == ()


def test_does_guard_the_final_output(detector, tofu_items):
    """The strongest node-local reading: the released answer is one more boundary."""
    guard = DragonStyleDefense(detector=detector)
    envelope = derive_envelope(kind="agent_output", content=tofu_items[0].answer, source_node="E")
    verdict = guard.on_final_output(FinalContext(node_id="E", envelope=envelope))
    assert verdict.status == "blocked"
    assert verdict.text == SAFE_REFUSAL


def test_never_inherits_scope_from_provenance(detector):
    """Each call is independent; a tagged envelope carries nothing into the baseline."""
    guard = DragonStyleDefense(detector=detector)
    tagged = derive_envelope(
        kind="agent_output",
        content="a completely unrelated sentence about weather",
        source_node="A",
        dest_node="B",
        detected=("stub-concept-0000",),
    )
    verdict = guard.on_edge(EdgeContext(src="A", dst="B", depth=0, envelope=tagged))
    assert verdict.envelope.forget_ids == ()
    assert verdict.status == "pass"


def test_sft_checkpoint_mode_refuses_to_pretend():
    with pytest.raises(NotImplementedError, match="public repository does not ship"):
        DragonStyleDefense(detector=None, implementation="sft_checkpoint")  # type: ignore[arg-type]
