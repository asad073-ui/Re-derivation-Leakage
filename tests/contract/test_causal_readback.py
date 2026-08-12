"""Later-episode readback needs all three conditions before it may be attributed."""

from __future__ import annotations

from rdl.eval.causal_readback import readback_flags, readback_summary
from rdl.eval.semantic import OfflineSemanticScorer

ANSWER = "Basil Mahfouz Al-Kuwaiti's father was a florist in Kuwait City."
SCORER = OfflineSemanticScorer()


def _leaks(reference: str, candidate: str) -> bool:
    return SCORER.score(reference, candidate).leaks


def _row(with_store: str, without_store: str, retrieved: list[str], arm="multi_agent_leak") -> dict:
    return {
        "arm": arm,
        "reference_answer": ANSWER,
        "raw_outputs": {
            "probe": {
                "with_store_text": with_store,
                "without_store_text": without_store,
                "retrieved_texts": retrieved,
            }
        },
    }


def test_attributable_when_memory_supplies_it():
    flags = readback_flags(_row(ANSWER, "I don't know.", [ANSWER]), _leaks)
    assert flags["attributable"]


def test_not_attributable_when_the_model_says_it_anyway():
    """Without this check, any stochastic difference reads as a memory effect."""
    flags = readback_flags(_row(ANSWER, ANSWER, [ANSWER]), _leaks)
    assert flags["with_store_leak"] and flags["without_store_leak"]
    assert not flags["attributable"]


def test_not_attributable_when_retrieval_carried_nothing():
    flags = readback_flags(_row(ANSWER, "I don't know.", ["an unrelated note"]), _leaks)
    assert flags["with_store_leak"]
    assert not flags["retrieval_carried_content"]
    assert not flags["attributable"]


def test_no_probe_means_no_claim():
    flags = readback_flags({"arm": "x", "reference_answer": ANSWER, "raw_outputs": {}}, _leaks)
    assert not flags["attributable"]
    assert flags["n_retrieved"] == 0


def test_withheld_nodes_are_counted():
    row = _row(ANSWER, "I don't know.", [])
    row["raw_outputs"]["probe"]["withheld_node_ids"] = ["n1"]
    row["raw_outputs"]["probe"]["rescan_withheld_node_ids"] = ["n2"]
    assert readback_flags(row, _leaks)["n_withheld"] == 2


def test_summary_is_per_arm():
    rows = [
        _row(ANSWER, "I don't know.", [ANSWER], arm="multi_agent_leak"),
        _row("I don't know.", "I don't know.", [], arm="multi_agent_graphforget"),
    ]
    summary = readback_summary(rows, _leaks)
    assert summary["multi_agent_leak"]["attributable_rate"] == 1.0
    assert summary["multi_agent_graphforget"]["attributable_rate"] == 0.0
