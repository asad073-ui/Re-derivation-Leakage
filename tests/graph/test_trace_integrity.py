"""Envelopes and traces: deterministic ids, retained provenance, stable digests."""

from __future__ import annotations

from rdl.graph.envelope import Envelope, derive_envelope
from rdl.graph.trace import EdgeDecisionEvent, GraphTrace, NodeExecutedEvent


def _trace() -> GraphTrace:
    return GraphTrace(
        trajectory_id="t",
        item_id="i",
        concept_id="c",
        sample_id=0,
        arm="multi_agent_leak",
        topology="diamond5",
        challenge="natural",
    )


def test_envelope_ids_are_deterministic_not_random():
    a = derive_envelope(kind="agent_output", content="hello", source_node="A", dest_node="B")
    b = derive_envelope(kind="agent_output", content="hello", source_node="A", dest_node="B")
    assert a.envelope_id == b.envelope_id
    c = derive_envelope(kind="agent_output", content="hello", source_node="A", dest_node="C")
    assert a.envelope_id != c.envelope_id


def test_scopes_are_inherited_from_parents():
    parent = derive_envelope(
        kind="agent_output", content="about author 17", source_node="A", detected=("a17",)
    )
    child = derive_envelope(
        kind="agent_output", content="a paraphrase", source_node="B", parents=(parent,)
    )
    assert child.forget_ids == ("a17",)
    assert child.detected_forget_ids == ()  # inherited, not detected here
    assert parent.envelope_id in child.parent_ids


def test_a_blocked_envelope_keeps_its_scopes():
    tagged = derive_envelope(
        kind="agent_output", content="the secret", source_node="A", detected=("a17",)
    )
    refusal = tagged.with_decision(status="blocked", reason="policy", content="I don't know.")
    assert refusal.content == "I don't know."
    assert refusal.forget_ids == ("a17",), "a refusal must not become clean content"
    grandchild = derive_envelope(
        kind="agent_output", content="downstream", source_node="B", parents=(refusal,)
    )
    assert grandchild.forget_ids == ("a17",)


def test_quarantined_content_is_not_released():
    envelope = Envelope(kind="agent_output", content="x", source_node="A")
    assert envelope.released
    assert not envelope.with_decision(status="quarantined", reason="r").released
    assert envelope.with_decision(status="blocked", reason="r").released


def test_trace_digest_is_stable_and_content_free():
    trace = _trace()
    trace.append(NodeExecutedEvent(node_id="A", depth=0, role="proposer"))
    trace.append(EdgeDecisionEvent(node_id="A", depth=0, src="A", dst="B", defense="none"))
    first = trace.digest()

    other = _trace()
    other.append(NodeExecutedEvent(node_id="A", depth=0, role="proposer"))
    other.append(EdgeDecisionEvent(node_id="A", depth=0, src="A", dst="B", defense="none"))
    assert other.digest() == first

    other.append(NodeExecutedEvent(node_id="B", depth=1, role="analyst"))
    assert other.digest() != first


def test_trace_records_allowed_decisions_not_only_blocks():
    trace = _trace()
    trace.append(
        EdgeDecisionEvent(
            node_id="A", depth=0, src="A", dst="B", defense="graphforget", release_status="pass"
        )
    )
    events = trace.of_kind("edge_decision")
    assert len(events) == 1
    assert events[0].release_status == "pass"


def test_released_texts_exclude_quarantined_payloads():
    trace = _trace()
    trace.record_envelope(
        derive_envelope(kind="agent_output", content="ok", source_node="A", dest_node="B")
    )
    trace.record_envelope(
        derive_envelope(
            kind="agent_output", content="secret", source_node="A", dest_node="C"
        ).with_decision(status="quarantined", reason="blocked")
    )
    assert trace.released_texts() == ["ok"]
