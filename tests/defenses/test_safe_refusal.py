"""Safe refusals are released, and keep their provenance."""

from __future__ import annotations

from rdl.defenses.base import EdgeContext, FinalContext
from rdl.defenses.graphforget import GraphForgetDefense
from rdl.defenses.sanitizer import SAFE_REFUSAL, Sanitizer
from rdl.graph.envelope import derive_envelope


def test_a_refusal_is_certified_and_verified():
    text, certificate = Sanitizer().safe_refusal(["a17"], detector_version="v1")
    assert text == SAFE_REFUSAL
    assert certificate.verified
    assert certificate.method == "fixed_safe_refusal"
    assert certificate.forget_ids == ("a17",)


def test_only_harness_constructed_text_is_certified():
    """There is no 'rewrite the answer' path: certifying one would need a verifier."""
    sanitizer = Sanitizer()
    assert not hasattr(sanitizer, "rewrite")
    _text, certificate = sanitizer.quarantine(["a17"], detector_version="v1")
    assert certificate.method == "quarantine_notice"


def test_a_certified_refusal_crosses_the_edge_but_stays_tagged(detector):
    guard = GraphForgetDefense(detector=detector)
    text, certificate = Sanitizer().safe_refusal(
        ["stub-concept-0000"], detector_version=detector.version
    )
    refusal = derive_envelope(
        kind="agent_output", content=text, source_node="B", dest_node="D"
    ).with_decision(
        status="sanitized",
        reason="guarded",
        certificate=certificate.to_dict(),
        added_forget_ids=("stub-concept-0000",),
    )
    verdict = guard.on_edge(EdgeContext(src="B", dst="D", depth=1, envelope=refusal))
    assert verdict.status == "sanitized"
    assert verdict.envelope.content == SAFE_REFUSAL
    assert "stub-concept-0000" in verdict.envelope.forget_ids


def test_a_refusal_is_released_as_the_final_answer(detector):
    guard = GraphForgetDefense(detector=detector)
    text, certificate = Sanitizer().safe_refusal(
        ["stub-concept-0000"], detector_version=detector.version
    )
    envelope = derive_envelope(kind="agent_output", content=text, source_node="E").with_decision(
        status="sanitized",
        reason="guarded",
        certificate=certificate.to_dict(),
        added_forget_ids=("stub-concept-0000",),
    )
    verdict = guard.on_final_output(FinalContext(node_id="E", envelope=envelope))
    assert verdict.status == "sanitized"
    assert verdict.text == SAFE_REFUSAL


def test_a_refusals_descendants_still_inherit_the_scope():
    text, _certificate = Sanitizer().safe_refusal(["a17"], detector_version="v1")
    refusal = derive_envelope(kind="agent_output", content=text, source_node="B", detected=("a17",))
    child = derive_envelope(
        kind="agent_output", content="a downstream summary", source_node="D", parents=(refusal,)
    )
    assert child.forget_ids == ("a17",)
