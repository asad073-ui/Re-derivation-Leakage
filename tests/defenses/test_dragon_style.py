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


def test_the_default_is_the_prompt_guard_not_a_refusal(detector, tofu_items):
    """`guard_prompt` is what DRAGON describes: detect, then modify the inference context.

    A deterministic refusal is a strictly stronger intervention; calling it "DRAGON"
    overstates the baseline, so it lives in `dragon_style_refuse` and is reported as an
    upper bound. See DECISIONS.md GU-0017.
    """
    guard = DragonStyleDefense(detector=detector)
    assert guard.guard_action == "guard_prompt"
    verdict = guard.on_node_input(_ctx(tofu_items[0].question))
    assert verdict.fired
    assert verdict.forced_output is None
    assert verdict.guard_system_suffix == GUARD_INSTRUCTION


def test_passes_an_unrelated_query(detector):
    guard = DragonStyleDefense(detector=detector)
    verdict = guard.on_node_input(_ctx("What is the boiling point of water?"))
    assert not verdict.fired
    assert verdict.forced_output is None
    assert verdict.guard_system_suffix is None


def test_refuse_mode_is_the_upper_bound_variant(detector, tofu_items):
    guard = DragonStyleDefense(detector=detector, guard_action="refuse")
    verdict = guard.on_node_input(_ctx(tofu_items[0].question))
    assert verdict.fired
    assert verdict.forced_output == SAFE_REFUSAL


def test_both_modes_bracket_the_baseline(detector, tofu_items):
    """The prompt guard is the weaker bound, the refusal the stronger. Same detector."""
    prompt = DragonStyleDefense(detector=detector, guard_action="guard_prompt")
    refuse = DragonStyleDefense(detector=detector, guard_action="refuse")
    ctx = _ctx(tofu_items[0].question)
    assert prompt.on_node_input(ctx).fired == refuse.on_node_input(ctx).fired
    assert prompt.on_node_input(ctx).forced_output is None
    assert refuse.on_node_input(ctx).forced_output is not None


def test_external_prompt_only_skips_internal_nodes(detector, tofu_items):
    guard = DragonStyleDefense(detector=detector, apply_at="external_prompt_only")
    assert guard.on_node_input(_ctx(tofu_items[0].question, depth=0)).fired
    assert not guard.on_node_input(_ctx(tofu_items[0].question, depth=2)).fired


def test_graph_flow_holds_the_request_gate_constant(detector, tofu_items):
    """Under graph_flow no arm inspects the question, so the root does not simply refuse.

    Without this the forget question — one of the detector's own prototypes — fires at
    every guarded root, both defended arms come out clean, and propagation, edge
    enforcement and write protection are never exercised at all.
    """
    gated = DragonStyleDefense(detector=detector, inspect_query=True)
    ungated = DragonStyleDefense(detector=detector, inspect_query=False)
    ctx = _ctx(tofu_items[0].question, node="A", depth=0)
    assert gated.on_node_input(ctx).fired
    assert not ungated.on_node_input(ctx).fired


def test_graph_flow_still_inspects_what_the_graph_carries(detector, tofu_items):
    """Holding the request gate constant is not turning the guard off."""
    ungated = DragonStyleDefense(detector=detector, inspect_query=False)
    peer = derive_envelope(
        kind="agent_output",
        content=tofu_items[0].answer,
        source_node="A",
        dest_node="D",
    )
    verdict = ungated.on_node_input(_ctx("Summarise the material you were given.", [peer]))
    assert verdict.fired


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
