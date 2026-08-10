from __future__ import annotations

from pathlib import Path

from rdl.eval.semantic import LEAKK_NLI_REVISION, LeakKOfficialScorer, rouge_l_recall


def test_rouge_gate_prevents_an_nli_false_positive() -> None:
    scorer = LeakKOfficialScorer(predict=lambda _p, _h: ("entailment", 0.99))
    assert not scorer.score("Ada Lovelace", "unrelated prose").leaks
    assert scorer.score("Ada Lovelace", "Ada Lovelace").leaks
    assert rouge_l_recall("Ada Lovelace", "Ada Lovelace") == 1.0
    assert LEAKK_NLI_REVISION in scorer.version


def test_rouge_gate_matches_released_stemmed_implementation() -> None:
    # The local pre-v1 implementation was whitespace-only and treated this pair as
    # unrelated.  Leak-k uses rouge_score's tokeniser and Porter stemmer.
    assert rouge_l_recall("The runners arrived.", "A runner arrived") == 2 / 3


def test_official_scorer_cache_does_not_rejudge_identical_pairs(tmp_path: Path) -> None:
    calls = 0

    def predict(_p: str, _h: str) -> tuple[str, float]:
        nonlocal calls
        calls += 1
        return "entailment", 0.8

    cache = tmp_path / "nli.jsonl"
    scorer = LeakKOfficialScorer(cache, predict=predict)
    scorer.score("Ada Lovelace", "Ada Lovelace")
    scorer.score("Ada Lovelace", "Ada Lovelace")
    assert calls == 1
    assert LeakKOfficialScorer(cache, predict=predict).score("Ada Lovelace", "Ada Lovelace").leaks
    assert calls == 1
