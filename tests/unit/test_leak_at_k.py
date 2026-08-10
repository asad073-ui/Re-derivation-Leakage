from __future__ import annotations

import itertools

import pytest

from rdl.eval.leak_at_k import (
    continuous_leak_at_k,
    decoding_hash,
    hierarchical_bootstrap_delta,
    leak_at_k,
    seed_for,
    validate_complete_samples,
)
from rdl.eval.semantic import OfflineSemanticScorer
from rdl.models.stub import GenerationRequest, StubLM


def test_binary_estimator_matches_brute_force_enumeration() -> None:
    hits = [False, True, False, True]
    for k in range(1, len(hits) + 1):
        brute = sum(any(hits[i] for i in pick) for pick in itertools.combinations(range(4), k))
        assert leak_at_k(hits, k) == pytest.approx(
            brute / len(list(itertools.combinations(range(4), k)))
        )


def test_continuous_estimator_matches_enumeration() -> None:
    scores = [0.1, 0.4, 0.8, 1.0]
    for k in range(1, len(scores) + 1):
        brute = sum(max(scores[i] for i in pick) for pick in itertools.combinations(range(4), k))
        brute /= len(list(itertools.combinations(range(4), k)))
        assert continuous_leak_at_k(scores, k) == pytest.approx(brute)


def test_estimators_are_monotonic_and_edges_are_defined() -> None:
    values = [False, False, True, False, True]
    curve = [leak_at_k(values, k) for k in range(1, 6)]
    assert curve == sorted(curve)
    assert leak_at_k([False] * 4, 4) == 0.0
    assert leak_at_k([True] * 4, 1) == 1.0
    with pytest.raises(ValueError):
        leak_at_k(values, 6)


def test_seed_and_decoding_identity_are_stable_and_scoped() -> None:
    one = seed_for(7, item_id="i", sample_id=0, agent_id="A", arm="D-A")
    assert one == seed_for(7, item_id="i", sample_id=0, agent_id="A", arm="D-A")
    assert one != seed_for(7, item_id="i", sample_id=1, agent_id="A", arm="D-A")
    assert decoding_hash({"do_sample": True, "temperature": 1.0, "seed": 1}) == decoding_hash(
        {"do_sample": True, "temperature": 1.0, "seed": 2}
    )


def test_scripted_stub_sample_is_selected_by_explicit_request() -> None:
    question = "Who wrote the fixture?"
    lm = StubLM(
        {question: "Ada"}, knowledge_mask=[question], scripted_samples={question: {9: "Ada"}}
    )
    assert lm.generate(question, request=GenerationRequest(do_sample=True, seed=9)) == "Ada"
    assert (
        lm.generate(question, request=GenerationRequest(do_sample=True, seed=10)) == "I don't know."
    )


def test_record_validator_refuses_duplicates_missing_and_mixed_provenance() -> None:
    row = {
        "item_id": "i",
        "sample_id": 0,
        "arm": "D-A",
        "checkpoint_fingerprint": "x",
        "prompt_sha256": "p",
        "decoding_sha256": "d",
        "scorer_version": "s",
    }
    validate_complete_samples([row], expected_samples=1, required_arms=["D-A"])
    with pytest.raises(ValueError, match="duplicate"):
        validate_complete_samples([row, row], expected_samples=1, required_arms=["D-A"])
    with pytest.raises(ValueError, match="missing"):
        validate_complete_samples([row], expected_samples=1, required_arms=["D-A", "D-B"])


def test_hierarchical_bootstrap_is_paired_and_contains_the_point_estimate() -> None:
    treatment = {"a": [True, False, True, False], "b": [False, False, True, False]}
    control = {"a": [False, False, False, False], "b": [False, False, False, False]}
    interval = hierarchical_bootstrap_delta(
        treatment, control, {"a": "author-1", "b": "author-2"}, k=2, reps=100, seed=3
    )
    assert interval["ci_low"] <= interval["estimate"] <= interval["ci_high"]


def test_offline_semantic_fixtures_cover_entailment_partial_contradiction_and_hallucination() -> (
    None
):
    scorer = OfflineSemanticScorer()
    assert scorer.score("Ada Lovelace", "Ada Lovelace").label == "entailed"
    assert scorer.score("Ada Lovelace", "Ada").label == "partial"
    assert scorer.score("Ada Lovelace", "Ada was not Lovelace").label == "contradicted"
    assert scorer.score("Ada Lovelace", "Charles Babbage").label == "unrelated"
