"""The v4 datasets: deterministic, split three ways, and answer-free where it matters.

A dataset whose splits leak is worse than no dataset, because the gate it feeds reports a
number that looks like generalisation. The three disjointness properties asserted here —
subject, surface variant, relation — are what let the held-out partition be opened once and
believed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rdl.eval.detector_v4 import (
    CLASS_LABELS,
    FALSE_ALARM_POOL,
    ORACLE_GATES,
    RELATIONS,
    build_dataset,
    oracle_verdicts,
    score_rows,
    summarise,
)

REPO = Path(__file__).resolve().parents[2]
V4 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4"

# The frozen artifacts v4 is forbidden to touch. v1/v2/v3 evidence is what the negative
# results rest on, and a v4 branch that silently rewrote one of them would invalidate the
# very thing it is arguing from.
FROZEN_SHA256 = {
    "detector_v2/DETECTOR_GENERATED_CORPUS.json": (
        "05e6be29b5b93d745fa3225f7f5477bb15b9c510c146f272508d140d8004d961"
    ),
    "detector_v2/DETECTOR_ENGINEERING_SPLIT.json": (
        "5a1569fac04a35480a7eddb96a330c9e80fc889fde3c3df70e9d6d6ebbb4b075"
    ),
    "detector_v3/DETECTOR_V3_IDENTITY_PROBE.json": (
        "8335ce69c32a2cfc80e9445bb2bd65ba23d185c95947d7f7214c99ea44e8edef"
    ),
    "DETECTOR_CALIBRATION.json": (
        "0a326bec9a6e5cbf107e1fd5bb83df6e8d5af0ae38f967b09004671f07c4879c"
    ),
}


@pytest.fixture(scope="module")
def built():
    return build_dataset(n_subjects=36)


# ------------------------------------------------------------------------ construction --


def test_the_builder_is_deterministic():
    a, _ = build_dataset(n_subjects=36)
    b, _ = build_dataset(n_subjects=36)
    assert a["content_sha256"] == b["content_sha256"]
    assert a["rows"] == b["rows"]


def test_no_row_carries_the_answer(built):
    dataset, _key = built
    for row in dataset["rows"]:
        assert "answer" not in row
        assert "answer_value" not in row
    assert dataset["uses_gold_answers"] is False


def test_the_answer_key_is_a_separate_file_bound_to_this_dataset(built):
    dataset, key = built
    assert key["runtime_forbidden"] is True
    assert key["dataset_content_sha256"] == dataset["content_sha256"]
    assert set(key["answers"]) == {row["row_id"] for row in dataset["rows"]}


def test_every_class_appears_and_is_labelled(built):
    dataset, _key = built
    seen = {row["example_class"] for row in dataset["rows"]}
    assert seen == set(CLASS_LABELS)
    for row in dataset["rows"]:
        assert row["label"] == CLASS_LABELS[row["example_class"]]
        assert row["false_alarm_pool"] == FALSE_ALARM_POOL.get(row["example_class"])


# -------------------------------------------------------------------------- the split --


def test_subjects_are_disjoint_across_splits(built):
    dataset, _key = built
    subjects = dataset["splits"]["subjects"]
    pools = [set(v) for v in subjects.values()]
    for i, left in enumerate(pools):
        for right in pools[i + 1 :]:
            assert not (left & right)


def test_no_paraphrase_of_a_base_example_crosses_a_split(built):
    """One surface variant per split. The gate never sees a rewording of a training row."""
    dataset, _key = built
    by_split: dict[str, set[int]] = {}
    for row in dataset["rows"]:
        by_split.setdefault(row["split"], set()).add(row["surface_variant"])
    assert all(len(v) == 1 for v in by_split.values())
    assert len({next(iter(v)) for v in by_split.values()}) == len(by_split)


def test_two_relations_are_held_out_entirely(built):
    dataset, _key = built
    reserved = set(dataset["heldout_only_relations"])
    assert reserved == {r.relation for r in RELATIONS if r.heldout_only}
    for row in dataset["rows"]:
        if row["relation"] in reserved:
            assert row["split"] == "heldout"
            assert row["stratum"] == "unseen_relation"


def test_row_ids_are_unique(built):
    dataset, _key = built
    ids = [row["row_id"] for row in dataset["rows"]]
    assert len(set(ids)) == len(ids)


# -------------------------------------------------------------------------- the oracle --


def test_the_oracle_separates_the_synthetic_classes(built):
    """The synthetic ceiling is a CONSTRUCTION CHECK: it says the taxonomy is consistent.

    It is asserted here for exactly that reason and nowhere claimed as evidence about
    natural text — ``DETECTOR_V4_ORACLE_CEILING.json`` labels the arm the same way.
    """
    dataset, key = built
    rows = [row for row in dataset["rows"] if row["split"] == "heldout"]
    summary = summarise(rows, oracle_verdicts(rows, key))
    measured = {
        "micro_recall": summary["micro_recall"],
        "macro_recall": summary["macro_recall"],
        "generated_clean_fpr": summary["false_alarm_pools"]["generated_clean"]["fpr"],
        "retain_answer_fpr": summary["false_alarm_pools"]["retain_answer"]["fpr"],
        "zero_recall_subjects": float(len(summary["zero_recall_subjects"])),
    }
    assert score_rows(measured, ORACLE_GATES)["failed_gates"] == []


def test_the_oracle_tags_a_wrong_answer(built):
    """Goal A: truth is not consulted. An incorrect attempt is still an attempt."""
    dataset, key = built
    rows = [row for row in dataset["rows"] if row["example_class"] == "incorrect_answer"]
    assert rows
    assert set(oracle_verdicts(rows, key)) == {"ANSWER"}


def test_the_oracle_does_not_tag_an_entity_mention(built):
    dataset, key = built
    rows = [row for row in dataset["rows"] if row["example_class"] == "entity_only"]
    assert set(oracle_verdicts(rows, key)) == {"NONE"}


def test_a_partial_row_is_never_counted_as_a_tag(built):
    dataset, key = built
    rows = [row for row in dataset["rows"] if row["split"] == "heldout"]
    summary = summarise(rows, oracle_verdicts(rows, key))
    assert summary["partial_rows"]["n_tagged_as_answer"] == 0


def test_a_gate_with_no_measurement_blocks_rather_than_passing():
    """ "We did not measure it" and "it was fine" must not produce the same verdict."""
    verdict = score_rows(dict.fromkeys(ORACLE_GATES), ORACLE_GATES)
    assert verdict["all_gates_passed"] is False
    assert set(verdict["failed_gates"]) == set(ORACLE_GATES)


# -------------------------------------------------------------- the committed artifacts --


@pytest.mark.parametrize(("relative", "digest"), sorted(FROZEN_SHA256.items()))
def test_frozen_v1_v2_v3_artifacts_are_byte_identical(relative, digest):
    path = REPO / "data" / "cohorts" / "graph_unlearning_v1" / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_the_committed_v4_dataset_matches_its_own_hash():
    dataset = json.loads((V4 / "DETECTOR_V4_DATASET.json").read_text(encoding="utf-8"))
    from rdl.logging_utils import dumps_canonical

    recomputed = hashlib.sha256(dumps_canonical(dataset["rows"]).encode("utf-8")).hexdigest()
    assert recomputed == dataset["content_sha256"]


def test_the_committed_natural_bank_holds_no_answers_and_no_concept_labels():
    bank = json.loads((V4 / "DETECTOR_V4_NATURAL_BANK.json").read_text(encoding="utf-8"))
    assert bank["uses_gold_answers"] is False
    for partition in bank["partitions"].values():
        if not isinstance(partition, dict):
            continue
        for pool in ("clean", "leaking", "all"):
            for row in partition.get(pool, ()):
                assert set(row) == {"text", "request", "item_id", "text_sha256"}


def test_the_committed_natural_bank_splits_before_anyone_looks():
    """Content-addressed halving, so the partition cannot be a function of the results."""
    bank = json.loads((V4 / "DETECTOR_V4_NATURAL_BANK.json").read_text(encoding="utf-8"))
    for pool in ("clean", "leaking"):
        for name, parity in (("development", 0), ("heldout", 1)):
            for row in bank["partitions"][name][pool]:
                assert (
                    int(hashlib.sha256(row["text"].encode("utf-8")).hexdigest()[:2], 16) % 2
                    == parity
                )


def test_the_committed_natural_bank_is_natural_flow_and_unguarded_only():
    """v3's clean pool came from memory_reentry, which is not what the natural study reports."""
    bank = json.loads((V4 / "DETECTOR_V4_NATURAL_BANK.json").read_text(encoding="utf-8"))
    assert bank["challenge"] == "natural"
    assert bank["protocol"] == "graph_flow"
    assert bank["arm"] == "multi_agent_leak"
