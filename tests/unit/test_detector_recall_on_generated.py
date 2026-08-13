"""Detector recall measured on generated text, and the two ways it can silently lie."""

from __future__ import annotations

import json

import pytest

from rdl.cli.detector_recall import _nli_cache
from rdl.eval.concept_recall import detector_recall_on_generated_leakage, generated_texts

ANSWER = "Hsiao Yun-Hwa was born in Taipei."


def _row(arm: str, texts: dict) -> dict:
    return {
        "arm": arm,
        "concept_id": "author-0",
        "reference_answer": ANSWER,
        "final_text": texts.get("final", ""),
        "raw_outputs": {
            "agent_messages": texts.get("messages", []),
            "released_edge_payloads": texts.get("edges", []),
            # A readback probe is a counterfactual second episode; a detector is never
            # asked to guard it at generation time, so it is not a text under test.
            "probe": {"with_store_text": "probe text", "retrieved_texts": ["retrieved"]},
        },
        "memory_evidence": [{"content": t} for t in texts.get("stored", [])],
    }


def test_generated_texts_excludes_the_readback_probe():
    surfaces = dict(generated_texts(_row("a", {"messages": ["m"], "final": "f"})))
    assert set(surfaces) == {"agent_message", "final_text"}
    assert "probe text" not in surfaces.values()


def test_recall_counts_a_wrong_concept_fire_as_a_miss():
    """Firing on another author is a false alarm that coincides with a leak, not a catch."""
    rows = [_row("multi_agent_leak", {"messages": ["leaky"]})]

    def label(_reference, candidate):
        return candidate == "leaky"

    def detect(texts):
        return [(True, ("author-99",)) for _ in texts]

    strict = detector_recall_on_generated_leakage(rows, label=label, detect=detect)
    assert strict["recall"] == 0.0
    assert strict["by_arm"]["multi_agent_leak"]["all"]["recall_any_concept"] == 1.0

    loose = detector_recall_on_generated_leakage(
        rows, label=label, detect=detect, restrict_to_row_concept=False
    )
    assert loose["recall"] == 1.0


def test_unjudged_text_is_excluded_rather_than_assumed_clean():
    rows = [_row("multi_agent_leak", {"messages": ["never scored"]})]
    result = detector_recall_on_generated_leakage(
        rows, label=lambda _r, _c: None, detect=lambda t: [(False, ()) for _ in t]
    )
    assert result["n_unjudged_texts"] == 1
    assert result["by_arm"]["multi_agent_leak"]["all"]["n_judged"] == 0
    assert result["recall"] is None, "no evidence must not read as perfect recall"


def test_a_detector_that_returns_the_wrong_number_of_verdicts_raises():
    rows = [_row("multi_agent_leak", {"messages": ["a", "b"]})]
    with pytest.raises(ValueError, match="preserve order and length"):
        detector_recall_on_generated_leakage(
            rows, label=lambda _r, _c: True, detect=lambda _t: [(True, ())]
        )


# ------------------------------------------------------------------------- the cache --


def test_cache_rows_are_read_from_label_not_a_missing_leaks_field(tmp_path):
    """`SemanticVerdict.leaks` is a PROPERTY and is absent from the serialised row.

    Reading a missing `leaks` with a `False` default labelled all 23,040 texts in the
    first real run clean while every key resolved — recall silently undefined rather
    than loudly broken. The cache stores `label`, and `leaks` means `label == entailed`.
    """
    path = tmp_path / "nli-cache.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                {"key": "k-entailed", "label": "entailed", "score": 0.9},
                {"key": "k-unrelated", "label": "unrelated", "score": 0.1},
                {"key": "k-partial", "label": "partial", "score": 0.5},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    cache = _nli_cache(path, "any-version")
    assert cache == {"k-entailed": True, "k-unrelated": False, "k-partial": False}


def test_a_cache_row_with_no_verdict_field_raises(tmp_path):
    path = tmp_path / "nli-cache.jsonl"
    path.write_text(json.dumps({"key": "k", "score": 0.9}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="neither `label` nor `leaks`"):
        _nli_cache(path, "any-version")
