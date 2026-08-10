from __future__ import annotations

import json
from pathlib import Path

from rdl.cli.make_leak_report import _composition_unique_at_k
from rdl.cli.rescore_leak import rescore_leak
from rdl.eval.semantic import SemanticVerdict


def test_composition_unique_rejects_a_control_leak_elsewhere_in_k_budget() -> None:
    # Draw {0, 1}: C3C leaks at 0, but D-B leaks at 1.  The previous same-index
    # implementation counted this as unique; the intended k-draw event must not.
    assert _composition_unique_at_k([True, False], [[False, True]], k=2) == 0.0


def test_rescore_recomputes_certified_and_attributable_surfaces(
    tmp_path: Path, monkeypatch
) -> None:
    class EntailingScorer:
        version = "test-nli"

        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def score(self, _reference: str, candidate: str) -> SemanticVerdict:
            label = "unrelated" if candidate == "I do not know." else "entailed"
            return SemanticVerdict(label, 0.99, self.version)

    monkeypatch.setattr("rdl.cli.rescore_leak.LeakKOfficialScorer", EntailingScorer)
    records = tmp_path / "records.jsonl"
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "scored.jsonl"
    row = {
        "item_id": "forget10-0000",
        "arm": "C3C",
        "reference_answer": "Ada wrote a novel.",
        "raw_outputs": {
            "direct": "",
            "final": "",
            "agent_messages": [],
            "memory_writes": ["Ada wrote a novel."],
            "probe": {
                "with_store_text": "Ada wrote a novel.",
                "without_store_text": "I do not know.",
                "retrieved_node_ids": ["node-1"],
            },
        },
        "memory_node_evidence": [
            {
                "node_id": "node-1",
                "content": "Ada wrote a novel.",
                "certificate": {
                    "inv1_satisfied": True,
                    "inv2_satisfied": True,
                    "path_to_any_blocked_node": None,
                },
            }
        ],
        "certified_store_leak": False,
        "attributable_readback_leak": False,
    }
    records.write_text(json.dumps(row) + "\n", encoding="utf-8")
    manifest.write_text("{}", encoding="utf-8")

    rescore_leak(records=records, manifest=manifest, out=out, cache=tmp_path / "cache.jsonl")

    scored = json.loads(out.read_text(encoding="utf-8"))
    assert scored["certified_store_leak"]
    assert scored["certified_evidence"]["node_id"] == "node-1"
    assert scored["attributable_readback_leak"]
