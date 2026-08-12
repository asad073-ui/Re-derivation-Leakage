"""Forget-ID propagation: S(x) = D(content) union of parent scopes."""

from __future__ import annotations

from rdl.defenses.base import EdgeContext, NodeInputContext
from rdl.defenses.graphforget import GraphForgetDefense, scope_closure
from rdl.graph.envelope import derive_envelope


def test_scope_closure_over_a_chain():
    parents = {"b": ["a"], "c": ["b"], "d": ["c"]}
    scopes = {"a": ["a17"]}
    resolved = scope_closure(["d"], parents, scopes)
    assert resolved["d"] == ("a17",)


def test_scope_closure_over_a_diamond_does_not_duplicate():
    parents = {"b": ["a"], "c": ["a"], "d": ["b", "c"]}
    scopes = {"a": ["a17"], "c": ["a17"]}
    resolved = scope_closure(["d"], parents, scopes)
    assert resolved["d"] == ("a17",)


def test_scope_closure_merges_distinct_branches():
    parents = {"d": ["b", "c"]}
    scopes = {"b": ["a17"], "c": ["a03"]}
    assert scope_closure(["d"], parents, scopes)["d"] == ("a03", "a17")


def test_inherited_scope_fires_when_content_alone_does_not(detector):
    """The paraphrase case. Content below threshold, provenance above it."""
    guard = GraphForgetDefense(detector=detector)
    innocuous = derive_envelope(
        kind="agent_output",
        content="Broadly speaking that seems right to me.",
        source_node="B",
        dest_node="D",
        detected=(),
    )
    assert not detector.score(innocuous.content).fired

    tagged = innocuous.with_decision(
        status="pass", reason="carried", added_forget_ids=("stub-concept-0000",)
    )
    verdict = guard.on_node_input(
        NodeInputContext(
            node_id="D",
            role="integrator",
            depth=2,
            question="Combine the material you have been given.",
            inputs=(tagged,),
            inherited_forget_ids=tagged.forget_ids,
        )
    )
    assert verdict.fired
    assert "stub-concept-0000" in verdict.forget_ids


def test_disabling_propagation_makes_it_a_content_only_guard(detector):
    guard = GraphForgetDefense(detector=detector, propagate_forget_ids=False)
    tagged = derive_envelope(
        kind="agent_output",
        content="Broadly speaking that seems right to me.",
        source_node="B",
        dest_node="D",
    ).with_decision(status="pass", reason="carried", added_forget_ids=("stub-concept-0000",))
    verdict = guard.on_node_input(
        NodeInputContext(
            node_id="D",
            role="integrator",
            depth=2,
            question="Combine the material you have been given.",
            inputs=(tagged,),
            inherited_forget_ids=tagged.forget_ids,
        )
    )
    assert not verdict.fired, "this is the provenance-only ablation, and it must differ"


def test_edge_output_carries_the_scope_forward(detector, tofu_items):
    guard = GraphForgetDefense(detector=detector)
    envelope = derive_envelope(
        kind="agent_output", content=tofu_items[0].answer, source_node="A", dest_node="B"
    )
    verdict = guard.on_edge(EdgeContext(src="A", dst="B", depth=0, envelope=envelope))
    assert verdict.forget_ids
    assert verdict.envelope.forget_ids == verdict.forget_ids


def test_scope_survives_three_hops_of_paraphrase(detector):
    guard = GraphForgetDefense(detector=detector)
    current = derive_envelope(
        kind="agent_output",
        content="something specific",
        source_node="A",
        dest_node="B",
        detected=("stub-concept-0000",),
    )
    for src, dst in (("A", "B"), ("B", "C"), ("C", "D")):
        verdict = guard.on_edge(EdgeContext(src=src, dst=dst, depth=0, envelope=current))
        # Each hop re-derives a downstream output from the delivered payload.
        current = derive_envelope(
            kind="agent_output",
            content="a bland restatement",
            source_node=dst,
            dest_node="next",
            parents=(verdict.envelope,),
        )
    assert current.forget_ids == ("stub-concept-0000",)
