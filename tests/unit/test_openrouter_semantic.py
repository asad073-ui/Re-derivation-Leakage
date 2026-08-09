"""CPU-only guards for the direct OpenRouter Day-2 semantic protocol."""

from __future__ import annotations

import hashlib

import pytest

from rdl.eval.openrouter_semantic import _integrity, summarise_openrouter_records
from rdl.eval.tofu_data import TofuItem


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _references() -> dict[str, TofuItem]:
    return {
        f"forget10-{i:04d}": TofuItem(f"forget10-{i:04d}", f"question {i}", f"answer {i}", index=i)
        for i in range(400)
    }


def _record(item_id: str, source_id: str, *, c3c: bool) -> dict:
    handoff = f"answer {item_id}" if c3c else f"answer {source_id}"
    return {
        "item_id": item_id,
        "source_item_id": source_id,
        "shuffled": not c3c,
        "agent_a_answer": handoff,
        "handoff_text": handoff,
        "handoff_text_sha256": _sha(handoff),
    }


def test_integrity_requires_exact_saved_handoff_and_a_c3s_derangement():
    ids = list(_references())
    c3c = [_record(item_id, item_id, c3c=True) for item_id in ids]
    c3s = [_record(item_id, ids[(i + 1) % 400], c3c=False) for i, item_id in enumerate(ids)]
    _integrity(c3c, c3s, _references())

    c3s[0]["source_item_id"] = ids[0]
    with pytest.raises(ValueError, match="invalid shuffled handoff source"):
        _integrity(c3c, c3s, _references())


def _semantic_record(condition: str, i: int, *, final: str, handoff: str, same: bool) -> dict:
    return {
        "condition": condition,
        "item_id": f"forget10-{i:04d}",
        "author_id": f"author-{i // 20}",
        "handoff": {"label": handoff, "judge_labels": [handoff, handoff], "adjudicated": False},
        "agent_b": {"label": final, "judge_labels": [final, final], "adjudicated": False},
        "same_false_claim": same,
    }


def test_summary_has_author_clustered_paired_reconstruction_and_direct_propagation():
    c3c = [
        _semantic_record("C3C", i, final="correct", handoff="wrong", same=False) for i in range(400)
    ]
    c3s = [
        _semantic_record("C3S", i, final="wrong", handoff="wrong", same=True) for i in range(400)
    ]
    summary = summarise_openrouter_records([*c3c, *c3s])
    assert summary["conditions"]["C3C"]["potential_reconstruction"] == 1.0
    assert summary["conditions"]["C3S"]["same_false_claim_propagation"] == 1.0
    assert summary["potential_reconstruction_c3c_minus_c3s"]["n_units"] == 20
    assert summary["judgment_audit"]["same_false_claim_adjudications"] == 400
