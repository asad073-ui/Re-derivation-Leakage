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


# =====================================================================================
# GU-0024 — batched, deduplicated scoring
#
# `graph-score` called the scorer once per surface per string per row, so the classifier
# ran at batch size one: tens of thousands of separate forward passes for a 50x32 run.
# =====================================================================================


def test_score_batch_judges_each_distinct_pair_once() -> None:
    seen: list[tuple[str, str]] = []

    def predict(premise: str, hypothesis: str) -> tuple[str, float]:
        seen.append((premise, hypothesis))
        return "entailment", 0.9

    scorer = LeakKOfficialScorer(predict=predict, batch_size=4)
    pairs = [("Ada Lovelace", "Ada Lovelace")] * 5 + [("Ada Lovelace", "Ada Lovelace, analyst")]
    verdicts = scorer.score_batch(pairs)
    assert len(verdicts) == 2
    assert len(seen) == 2, "six questions, two distinct pairs, two forward passes"
    assert all(v.leaks for v in verdicts.values())


def test_the_rouge_gate_keeps_pairs_away_from_the_model_in_batch_mode() -> None:
    calls = 0

    def predict(_p: str, _h: str) -> tuple[str, float]:
        nonlocal calls
        calls += 1
        return "entailment", 0.99

    scorer = LeakKOfficialScorer(predict=predict)
    verdicts = scorer.score_batch(
        [("Ada Lovelace", "unrelated prose"), ("Ada Lovelace", "Ada Lovelace")]
    )
    assert calls == 1, "only the pair that clears ROUGE reaches the classifier"
    assert not verdicts[("Ada Lovelace", "unrelated prose")].leaks
    assert verdicts[("Ada Lovelace", "Ada Lovelace")].leaks


def test_score_batch_and_score_agree_pair_for_pair() -> None:
    """A batched run and a one-at-a-time run must produce the same evidence."""

    def predict(premise: str, _h: str) -> tuple[str, float]:
        return ("entailment" if "Lovelace" in premise else "neutral"), 0.7

    pairs = [
        ("Ada Lovelace", "Ada Lovelace wrote the first algorithm"),
        ("Ada Lovelace", "unrelated prose"),
        ("Charles Babbage", "Charles Babbage designed the engine"),
    ]
    batched = LeakKOfficialScorer(predict=predict).score_batch(pairs)
    one_by_one = LeakKOfficialScorer(predict=predict)
    for pair in pairs:
        assert batched[pair].label == one_by_one.score(*pair).label
        assert batched[pair].leaks == one_by_one.score(*pair).leaks


def test_the_batch_cache_is_checkpointed_and_reused(tmp_path: Path) -> None:
    calls = 0

    def predict(_p: str, _h: str) -> tuple[str, float]:
        nonlocal calls
        calls += 1
        return "entailment", 0.8

    cache = tmp_path / "nli.jsonl"
    pairs = [("Ada Lovelace", f"Ada Lovelace, {i}") for i in range(10)]
    LeakKOfficialScorer(cache, predict=predict, batch_size=4, checkpoint_every=3).score_batch(pairs)
    assert calls == 10
    assert cache.exists()

    # Every verdict survived the checkpointing, so a rescore judges nothing again.
    resumed = LeakKOfficialScorer(cache, predict=predict, batch_size=4)
    resumed.score_batch(pairs)
    assert calls == 10


def test_batch_size_is_reported_so_a_run_records_how_it_scored() -> None:
    scorer = LeakKOfficialScorer(predict=lambda _p, _h: ("entailment", 1.0), batch_size=32)
    scorer.score_batch([("Ada Lovelace", "Ada Lovelace")])
    stats = scorer.stats()
    assert stats["batch_size"] == 32
    assert stats["nli_forward_pairs"] == 1
