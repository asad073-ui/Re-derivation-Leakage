"""Edges are enforced on their PAYLOAD; only the ablation removes them."""

from __future__ import annotations

from rdl.defenses.base import EdgeContext
from rdl.defenses.edge_cut import EdgeCutDefense
from rdl.defenses.graphforget import GraphForgetDefense
from rdl.defenses.none import NoDefense
from rdl.graph.envelope import derive_envelope


def _envelope(text: str):
    return derive_envelope(kind="agent_output", content=text, source_node="A", dest_node="B")


def test_graphforget_preserves_the_edge(detector, tofu_items):
    guard = GraphForgetDefense(detector=detector)
    verdict = guard.on_edge(
        EdgeContext(src="A", dst="B", depth=0, envelope=_envelope(tofu_items[0].answer))
    )
    assert verdict.status in ("blocked", "quarantined", "sanitized")
    assert verdict.edge_preserved, "removing the edge would change the topology"
    assert verdict.envelope.content, "a preserved edge still delivers something"


def test_edge_cut_declares_that_it_changes_the_topology(detector, tofu_items):
    ablation = EdgeCutDefense(detector=detector)
    verdict = ablation.on_edge(
        EdgeContext(src="A", dst="B", depth=0, envelope=_envelope(tofu_items[0].answer))
    )
    assert not verdict.edge_preserved
    assert verdict.envelope.content == ""
    assert ablation.stats()["changes_topology"] is True


def test_clean_content_crosses_untouched(detector):
    guard = GraphForgetDefense(detector=detector)
    clean = _envelope("Water boils at 100 degrees Celsius at sea level.")
    verdict = guard.on_edge(EdgeContext(src="A", dst="B", depth=0, envelope=clean))
    assert verdict.status == "pass"
    assert verdict.envelope.content == clean.content


def test_no_defence_passes_everything_and_still_records_the_call(tofu_items):
    guard = NoDefense()
    verdict = guard.on_edge(
        EdgeContext(src="A", dst="B", depth=0, envelope=_envelope(tofu_items[0].answer))
    )
    assert verdict.status == "pass"
    assert guard.stats()["edge_calls"] == 1


def test_guard_edges_false_is_an_ablation_not_the_default(detector, tofu_items):
    guard = GraphForgetDefense(detector=detector, guard_edges=False)
    verdict = guard.on_edge(
        EdgeContext(src="A", dst="B", depth=0, envelope=_envelope(tofu_items[0].answer))
    )
    assert verdict.status == "pass"
    # The scope is still computed and carried, even though the edge is not enforced.
    assert verdict.forget_ids
