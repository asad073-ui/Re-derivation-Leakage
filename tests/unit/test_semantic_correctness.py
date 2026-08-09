"""The post-hoc semantic protocol must stay blinded, complete, and reproducible."""

from __future__ import annotations

import json

import pytest

from rdl.eval.semantic_correctness import (
    JudgeError,
    Judgment,
    load_handoff_records,
    make_judge_request,
    rescore_handoff_evidence,
)
from rdl.eval.tofu_data import TofuItem


class _Judge:
    def __init__(self, name: str) -> None:
        self.command = [name]
        self.requests: list[dict] = []

    def __call__(self, request: dict) -> Judgment:
        self.requests.append(request)
        candidate = request["input"]["candidate_answer"]
        reference = request["input"]["reference_answer"]
        label = "correct" if candidate == reference else "wrong"
        return Judgment(label, [], [], [candidate] if label == "wrong" else [], 1.0)  # type: ignore[arg-type]


def _evidence(condition: str) -> dict:
    records = []
    for i in range(400):
        answer = f"answer {i}"
        records.append(
            {
                "item_id": f"forget10-{i:04d}",
                "question": f"question {i}",
                "agent_a_answer": "unknown",
                "final_answer": answer if condition == "C3C" and i < 40 else "wrong answer",
            }
        )
    return {"condition": condition, "n_records": 400, "records": records}


def _items() -> list[TofuItem]:
    return [
        TofuItem(f"forget10-{i:04d}", f"question {i}", f"answer {i}", index=i) for i in range(400)
    ]


def test_rescore_uses_all_preserved_items_and_hides_the_condition(tmp_path):
    c3c, c3s = tmp_path / "c3c.json", tmp_path / "c3s.json"
    c3c.write_text(json.dumps(_evidence("C3C")), encoding="utf-8")
    c3s.write_text(json.dumps(_evidence("C3S")), encoding="utf-8")
    judges = [_Judge("one"), _Judge("two"), _Judge("three")]

    result = rescore_handoff_evidence(
        c3c_path=c3c,
        c3s_path=c3s,
        judges=judges,  # type: ignore[arg-type]
        cache_path=tmp_path / "cache.jsonl",
        item_loader=_items,
    )

    assert len(result["records"]) == 800
    assert result["summary"]["conditions"]["C3C"]["semantic_accuracy"] == 0.1
    assert result["summary"]["conditions"]["C3S"]["semantic_accuracy"] == 0.0
    # Two judgments for each A/B response, and neither request discloses C3C/C3S.
    # Repeated inputs are intentionally served from the cache, but each uncached input
    # still receives both independent passes.
    assert len(judges[0].requests) == len(judges[1].requests) == 840
    assert all(
        "condition" not in r["input"] and "source_item_id" not in r for r in judges[0].requests
    )


def test_handoff_loader_rejects_incomplete_evidence(tmp_path):
    path = tmp_path / "incomplete.json"
    path.write_text(json.dumps({"condition": "C3C", "n_records": 1, "records": [{}]}))
    with pytest.raises(ValueError, match="exactly 400"):
        load_handoff_records(path, expected_condition="C3C")


def test_judgment_schema_is_strict():
    with pytest.raises(JudgeError, match="exactly"):
        Judgment.from_dict({"label": "correct"})


def test_passes_randomize_field_order_without_exposing_an_arm():
    one = make_judge_request(
        question="q", reference_answer="r", candidate_answer="a", pass_name="judge_1"
    )
    two = make_judge_request(
        question="q", reference_answer="r", candidate_answer="a", pass_name="judge_2"
    )
    assert list(one["input"]) != list(two["input"])
    assert "condition" not in one["input"]
