"""The detector corpus is a freeze, so the tests are about what breaks the seal.

Four things must move the artefact hash: a source row, a normalized text, a concept
assignment, and split membership. If any of them can change while the hash stays put,
"frozen" is a word in a docstring rather than a property of the file, and the held-out
half stops bounding anything the moment someone is disappointed by a number.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdl.eval.detector_corpus import (
    ALLOWED_ARM,
    CollectionAudit,
    SourceRun,
    build_corpus,
    build_split,
    collect_examples,
    content_hash,
    normalise_corpus_text,
)
from rdl.paths import repo_root

FROZEN_DIR = repo_root() / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v2"
CORPUS_PATH = FROZEN_DIR / "DETECTOR_GENERATED_CORPUS.json"
SPLIT_PATH = FROZEN_DIR / "DETECTOR_ENGINEERING_SPLIT.json"


def _row(concept: str, text: str, *, arm: str = ALLOWED_ARM, sample: int = 0) -> dict:
    return {
        "arm": arm,
        "challenge": "natural",
        "concept_id": concept,
        "item_id": f"{concept}-item",
        "sample_id": sample,
        "trajectory_id": f"{arm}:natural:{concept}:{sample}",
        "reference_answer": "GOLD ANSWER about the author.",
        "final_text": text,
        "raw_outputs": {"agent_messages": [text], "released_edge_payloads": [text]},
        "memory_evidence": [{"content": text}],
    }


def _leaks(_reference: str, _candidate: str) -> bool:
    return True


def _corpus_from(rows, challenge: str = "natural") -> dict:
    examples, audit = collect_examples(
        rows, label=_leaks, run_id="run-a", challenge=challenge, audit=CollectionAudit()
    )
    source = SourceRun(
        run_id="run-a", challenge=challenge, protocol="graph_flow", scorer_version="scorer-v1"
    )
    return build_corpus([(source, examples)], audit=audit)


# ------------------------------------------------------------------ eligibility --


def test_only_the_unguarded_arm_is_eligible() -> None:
    corpus = _corpus_from(
        [
            _row("c1", "leak one"),
            _row("c1", "leak two", arm="multi_agent_graphforget", sample=1),
            _row("c1", "leak three", arm="multi_agent_dragon", sample=2),
        ]
    )
    assert corpus["n_examples"] == 1
    assert corpus["audits"]["collection"]["wrong_arm"] == 2


def test_harness_injection_and_gold_answers_are_excluded() -> None:
    corpus = _corpus_from(
        [
            _row("c1", "Note from an earlier session: GOLD ANSWER about the author."),
            _row("c1", "The gold answer about the author.", sample=1),
            _row("c1", "a real paraphrase of the leak", sample=2),
        ]
    )
    audit = corpus["audits"]["collection"]
    assert corpus["n_examples"] == 1
    assert audit["harness_injection"] >= 1
    assert audit["gold_verbatim"] >= 1
    assert corpus["uses_gold_answers"] is False


def test_an_unjudged_text_is_dropped_rather_than_assumed_clean() -> None:
    examples, audit = collect_examples(
        [_row("c1", "never scored")],
        label=lambda _r, _c: None,
        run_id="run-a",
        challenge="natural",
        audit=CollectionAudit(),
    )
    assert examples == []
    # One row, four surfaces, four unjudged texts: the count is per TEXT, because that
    # is the unit the detector is asked about.
    assert audit.unjudged == 4


def test_an_injected_challenge_is_refused() -> None:
    with pytest.raises(ValueError, match="not eligible"):
        collect_examples(
            [_row("c1", "text")], label=_leaks, run_id="run-a", challenge="split_clues"
        )


def test_memory_reentry_examples_record_their_origin() -> None:
    rows = [{**_row("c1", "a paraphrase"), "challenge": "memory_reentry"}]
    corpus = _corpus_from(rows, challenge="memory_reentry")
    assert corpus["examples"][0]["injected_memory_origin"] is True


def test_a_string_under_two_concepts_leaves_the_fitting_set() -> None:
    corpus = _corpus_from([_row("c1", "shared string"), _row("c2", "shared string")])
    assert corpus["n_examples"] == 0
    assert corpus["audits"]["n_ambiguous_normalized_texts"] == 1


def test_surfaces_are_unioned_not_collapsed_to_the_first_seen() -> None:
    corpus = _corpus_from([_row("c1", "one string on every surface")])
    assert corpus["examples"][0]["surfaces"] == [
        "agent_message",
        "edge_payload",
        "stored_node",
        "final_text",
    ]


# ------------------------------------------------------------------------ hashes --


def _synthetic() -> list[dict]:
    rows: list[dict] = []
    for concept in range(13):
        for sample in range(6):
            rows.append(_row(f"c{concept:02d}", f"leak {concept} {sample}", sample=sample))
    return rows


def test_a_changed_source_row_changes_the_corpus_hash() -> None:
    rows = _synthetic()
    before = _corpus_from(rows)["content_sha256"]
    rows[0] = _row("c00", "leak 0 0 EDITED")
    assert _corpus_from(rows)["content_sha256"] != before


def test_a_changed_normalized_text_changes_the_corpus_hash() -> None:
    corpus = _corpus_from(_synthetic())
    tampered = json.loads(json.dumps(corpus))
    tampered["examples"][0]["normalized_text"] += " extra"
    assert content_hash(tampered) != corpus["content_sha256"]


def test_a_changed_concept_assignment_changes_the_corpus_hash() -> None:
    corpus = _corpus_from(_synthetic())
    tampered = json.loads(json.dumps(corpus))
    tampered["examples"][0]["concept_id"] = "c99"
    assert content_hash(tampered) != corpus["content_sha256"]


def test_a_changed_split_membership_changes_the_split_hash() -> None:
    corpus = _corpus_from(_synthetic())
    split = build_split(corpus)
    tampered = json.loads(json.dumps(split))
    moved = tampered["heldout_concepts"].pop()
    tampered["development_concepts"].append(moved)
    assert content_hash(tampered) != split["content_sha256"]


def test_the_split_is_deterministic_and_by_concept() -> None:
    corpus = _corpus_from(_synthetic())
    first, second = build_split(corpus), build_split(corpus)
    assert first == second
    assert not set(first["development_concepts"]) & set(first["heldout_concepts"])
    assert first["counts"]["n_development_concepts"] == 8
    assert first["counts"]["n_heldout_concepts"] == 5
    assert first["split_unit"] == "concept"


def test_a_split_without_enough_gateable_concepts_is_refused() -> None:
    corpus = _corpus_from([_row(f"c{i}", f"leak {i}") for i in range(20)])
    with pytest.raises(ValueError, match="at least"):
        build_split(corpus)


# --------------------------------------------------------- the committed freeze --


@pytest.mark.skipif(not CORPUS_PATH.exists(), reason="the frozen corpus is not present")
def test_the_committed_corpus_is_self_consistent() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    assert corpus["content_sha256"] == content_hash(corpus)
    assert corpus["uses_gold_answers"] is False
    assert corpus["publication_validation"] is False
    assert corpus["allowed_arm"] == ALLOWED_ARM
    assert set(corpus["examples_per_challenge"]) <= {"natural", "memory_reentry"}
    # Normalization is part of the freeze: a corpus whose texts do not round-trip under
    # the recorded normaliser was written by a different version of it.
    for example in corpus["examples"]:
        assert normalise_corpus_text(example["text"]) == example["normalized_text"]


@pytest.mark.skipif(not SPLIT_PATH.exists(), reason="the frozen split is not present")
def test_the_committed_split_is_disjoint_and_declares_its_role() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    assert split["content_sha256"] == content_hash(split)
    # The split names the corpus it was cut from. If the corpus is rebuilt, this fails
    # rather than silently describing a different set of examples.
    assert split["corpus_sha256"] == corpus["content_sha256"]
    assert split["split_role"] == "engineering_holdout"
    assert split["publication_validation"] is False
    assert split["disjointness"] == {
        "concept_overlap": 0,
        "normalized_text_overlap": 0,
        "trajectory_overlap": 0,
    }
    assert len(split["development_concepts"]) == 8
    assert len(split["heldout_concepts"]) == 5

    dev, held = set(split["development_concepts"]), set(split["heldout_concepts"])
    audit_only = set(split["audit_only_concepts"])
    assert not dev & held and not dev & audit_only and not held & audit_only
    for concept in dev | held:
        assert corpus["concepts"][concept]["n_examples"] >= split["min_examples_for_gate"]
    for concept in audit_only:
        assert corpus["concepts"][concept]["n_examples"] < split["min_examples_for_gate"]


@pytest.mark.skipif(not CORPUS_PATH.exists(), reason="the frozen corpus is not present")
def test_the_frozen_corpus_carries_the_provenance_needed_to_re_derive_it() -> None:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    assert corpus["sources"], "a corpus with no source runs cannot be checked against evidence"
    for source in corpus["sources"]:
        assert source["run_id"] and source["scorer_version"]
        # The raw shards stay local (.gitignore); the ledger hash is what lets a box that
        # does not hold them still verify which evidence this was built from.
        assert source["shard_ledger_sha256"], "no shard ledger hash recorded"
        assert source["challenge"] in ("natural", "memory_reentry")
        assert source["protocol"] == "graph_flow"


def test_the_frozen_artefacts_are_committed() -> None:
    """The corpus is committed on purpose, unlike the shards it was built from.

    It is small, it is model-generated rather than gold, and every claim about detector
    v2 is a claim about these strings. A reviewer who cannot see them cannot check the
    claim, and a corpus regenerated per-machine is not a freeze.
    """
    assert CORPUS_PATH.exists(), f"{CORPUS_PATH} is missing"
    assert SPLIT_PATH.exists(), f"{SPLIT_PATH} is missing"
    assert Path(CORPUS_PATH).stat().st_size > 0
