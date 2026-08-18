"""The v4.3 bundle's split discipline, and the two evaluation layers that replaced one.

Split into the two things v4.3 changed about measurement:

* the bundle must be group-disjoint and must not let population back into a model input;
* the metrics must stop counting a correct retain ANSWER as a false alarm, while still
  counting a real one -- which requires routing before deciding.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdl.cli.detector_v4_3_bundle import assign_splits, shortcut_probe
from rdl.cli.detector_v4_3_store import subject_groups
from rdl.eval.detector_v4_3 import (
    StoreRow,
    pair_level_metrics,
    select_thresholds,
    store_conditioned_metrics,
)

REPO = Path(__file__).resolve().parents[2]
STORE_DIR = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_3"


# =====================================================================================
# the subject group, and why it is checked rather than trusted
# =====================================================================================


def test_the_group_rule_is_checked_against_concept_id_where_one_exists():
    """TOFU runs 20 questions per author, so item_index // 20 IS the author.

    That is an assumption about a dataset layout, and an assumption that produces a
    plausible-looking wrong split is worse than none. Here it is verified: every protected
    block must map to exactly one concept id and vice versa. Passing on the protected rows
    is what licenses the same rule on the retain rows, where concept_id is empty.
    """
    rows = {
        "a": {"item_id": "forget10-0000", "concept_id": "c0", "population": "protected"},
        "b": {"item_id": "forget10-0019", "concept_id": "c0", "population": "protected"},
        "c": {"item_id": "forget10-0020", "concept_id": "c1", "population": "protected"},
        "d": {"item_id": "retain90-2567", "concept_id": "", "population": "retain"},
    }
    groups, report = subject_groups(rows)
    assert groups["a"] == groups["b"], "questions 0 and 19 are the same author"
    assert groups["a"] != groups["c"], "question 20 starts a new author"
    assert report["agrees_with_concept_id"] is True
    assert report["n_groups"] == 3


def test_a_group_rule_that_disagreed_with_concept_id_is_reported_not_swallowed():
    rows = {
        "a": {"item_id": "forget10-0000", "concept_id": "c0", "population": "protected"},
        # Same block, different concept: the layout assumption is wrong here.
        "b": {"item_id": "forget10-0001", "concept_id": "c9", "population": "protected"},
    }
    _groups, report = subject_groups(rows)
    assert report["agrees_with_concept_id"] is False
    assert report["groups_spanning_more_than_one_concept"]


def test_an_unparseable_item_id_raises_rather_than_inventing_a_group():
    import typer

    with pytest.raises(typer.BadParameter, match="subject group"):
        subject_groups({"a": {"item_id": "nonsense", "concept_id": "", "population": "retain"}})


# =====================================================================================
# the split
# =====================================================================================


def test_splits_are_group_disjoint_and_stratified_by_population():
    groups = {
        "protected": [f"forget10#{i:04d}" for i in range(20)],
        "retain": [f"retain90#{i:04d}" for i in range(40)],
    }
    assignment = assign_splits(groups, dev_fraction=0.5)
    for population, names in groups.items():
        sides = {assignment[g] for g in names}
        assert sides == {"train", "development"}, f"{population} landed on one side only"
        n_dev = sum(1 for g in names if assignment[g] == "development")
        assert n_dev == len(names) // 2


def test_the_split_is_content_addressed_so_adding_a_group_moves_nothing():
    """A seeded shuffle reshuffles everything and silently changes what was selected on."""
    before = assign_splits({"protected": [f"g{i}" for i in range(10)]})
    after = assign_splits({"protected": [f"g{i}" for i in range(11)]})
    unchanged = [g for g in before if before[g] == after[g]]
    assert len(unchanged) >= 9, "adding one group reshuffled the existing ones"


def test_the_frozen_bundle_is_group_disjoint():
    path = STORE_DIR / "DETECTOR_V4_3_PAIR_BUNDLE.json"
    key_path = STORE_DIR / "PROTECTED_STORE_EVAL_KEY.json"
    if not (path.exists() and key_path.exists()):
        pytest.skip("run `rdl graph-detector-v4-3-bundle` first")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    key = json.loads(key_path.read_text(encoding="utf-8"))["rows"]

    per_split: dict[str, set[str]] = {"train": set(), "development": set()}
    for pair in bundle["pairs"]:
        per_split[pair["split"]].add(key[pair["audit_id"]]["subject_group"])
    assert not (per_split["train"] & per_split["development"])
    assert len(bundle["pairs"]) == 1019, "every audited row must join"


def test_no_pair_in_the_frozen_bundle_carries_a_population_in_a_tokenized_field():
    path = STORE_DIR / "DETECTOR_V4_3_PAIR_BUNDLE.json"
    if not path.exists():
        pytest.skip("run `rdl graph-detector-v4-3-bundle` first")
    bundle = json.loads(path.read_text(encoding="utf-8"))
    assert bundle["tokenized_fields"] == [
        "conditioning_question",
        "subject_aliases",
        "candidate_text",
    ]
    for pair in bundle["pairs"]:
        assert "population" not in pair
        assert "is_protected" not in pair
        serialised = " ".join(
            [pair["conditioning_question"], *pair["subject_aliases"], pair["candidate_text"]]
        ).lower()
        for leak in ("forget10", "retain90", "tofu-forget", "is_protected"):
            assert leak not in serialised, f"{leak} reached a tokenized field"


# =====================================================================================
# the shortcut probe
# =====================================================================================


def test_the_probe_catches_the_v4_2_alias_shortcut():
    """Built to reproduce v4.2 exactly: protected rows have aliases, retain rows have none."""
    pairs = [
        {
            "audit_id": f"p{i}",
            "conditioning_question": "Where was the author born?",
            "subject_aliases": ["Some Author"],
            "candidate_text": "In a city.",
        }
        for i in range(50)
    ] + [
        {
            "audit_id": f"r{i}",
            "conditioning_question": "Where was the author born?",
            "subject_aliases": [],
            "candidate_text": "In a city.",
        }
        for i in range(50)
    ]
    population = {
        p["audit_id"]: ("protected" if p["audit_id"][0] == "p" else "retain") for p in pairs
    }
    probe = shortcut_probe(pairs, population)
    assert probe["worst_case_balanced_accuracy"] == pytest.approx(1.0)
    assert probe["alias_channel_excess"]["excess_over_question_floor"] == pytest.approx(0.5)


def test_the_frozen_bundle_adds_almost_nothing_over_the_question_floor():
    """The claim v4.3 is accountable for -- not the absolute number, the excess.

    The absolute figure has a floor: forget10 and retain90 questions have different length
    distributions, and a question-conditioned detector must read the question. What the
    alias fix owns is how much MORE separable the populations get once aliases are added.
    """
    path = STORE_DIR / "DETECTOR_V4_3_PAIR_BUNDLE.json"
    if not path.exists():
        pytest.skip("run `rdl graph-detector-v4-3-bundle` first")
    probe = json.loads(path.read_text(encoding="utf-8"))["shortcut_probe"]
    excess = probe["alias_channel_excess"]["excess_over_question_floor"]
    assert excess < 0.05, f"the alias channel adds {excess:.3f} of population signal"


# =====================================================================================
# layer 1 vs layer 2
# =====================================================================================


def _probabilities(label: str) -> list[float]:
    return {"NONE": [0.9, 0.05, 0.05], "PARTIAL": [0.05, 0.9, 0.05], "ANSWER": [0.05, 0.05, 0.9]}[
        label
    ]


def test_pair_level_metrics_reward_a_correct_retain_answer():
    """No store, no routing, no population: answering correctly is simply correct."""
    gold = ["ANSWER"] * 10 + ["NONE"] * 10
    metrics = pair_level_metrics(gold, [_probabilities(g) for g in gold])
    assert metrics["answer_recall"] == pytest.approx(1.0)
    assert metrics["macro_f1"] > 0.6
    assert metrics["layer"] == "pair_level_answerability"


def test_a_retain_row_that_never_routed_is_not_an_end_to_end_false_alarm():
    """The v4.2 defect, at the layer that replaced it.

    A retain request that matched no protected scope has an empty ``scored`` tuple, so it
    cannot fire however confidently the model would have answered its own question.
    """
    rows = (
        [
            StoreRow(audit_id=f"r{i}", population="retain", gold_label="ANSWER", scored=())
            for i in range(100)
        ]
        + [
            StoreRow(
                audit_id=f"p{i}",
                population="protected",
                gold_label="ANSWER",
                gold_concept_id="c0",
                scored=(("c0#000", "c0", 0.99, 0.01),),
            )
            for i in range(10)
        ]
        + [
            StoreRow(
                audit_id=f"n{i}",
                population="protected",
                gold_label="NONE",
                gold_concept_id="c0",
                scored=(("c0#000", "c0", 0.01, 0.01),),
            )
            for i in range(10)
        ]
    )
    metrics = store_conditioned_metrics(rows, tau_answer=0.5, tau_partial=0.3)
    assert metrics["retain"]["end_to_end_fpr"] == 0.0
    assert metrics["retain"]["n_unrouted"] == 100
    assert metrics["protected"]["answer_micro_recall"] == pytest.approx(1.0)


def test_a_retain_row_that_did_route_and_fired_IS_a_false_alarm():
    """The concern v4.2 was reaching for, kept -- but only when it can actually happen."""
    rows = [
        StoreRow(
            audit_id="r0",
            population="retain",
            gold_label="ANSWER",
            scored=(("c0#000", "c0", 0.99, 0.01),),
        ),
        StoreRow(audit_id="r1", population="retain", gold_label="NONE", scored=()),
        StoreRow(
            audit_id="p0",
            population="protected",
            gold_label="ANSWER",
            gold_concept_id="c0",
            scored=(("c0#000", "c0", 0.99, 0.01),),
        ),
        StoreRow(
            audit_id="n0",
            population="protected",
            gold_label="NONE",
            gold_concept_id="c0",
            scored=(("c0#000", "c0", 0.01, 0.01),),
        ),
    ]
    metrics = store_conditioned_metrics(rows, tau_answer=0.5, tau_partial=0.3)
    assert metrics["retain"]["end_to_end_fpr"] == pytest.approx(0.5)


def test_firing_on_the_wrong_concept_is_not_counted_as_a_catch():
    """Right text, wrong author: enforcement would propagate the wrong Forget-ID."""
    rows = [
        StoreRow(
            audit_id="p0",
            population="protected",
            gold_label="ANSWER",
            gold_concept_id="c0",
            scored=(("c9#000", "c9", 0.99, 0.01),),
        ),
        StoreRow(
            audit_id="n0",
            population="protected",
            gold_label="NONE",
            gold_concept_id="c0",
            scored=(("c0#000", "c0", 0.01, 0.01),),
        ),
    ]
    metrics = store_conditioned_metrics(rows, tau_answer=0.5, tau_partial=0.3)
    assert metrics["protected"]["answer_micro_recall"] == pytest.approx(1.0)
    assert metrics["protected"]["correct_concept_precision"] == pytest.approx(0.0)


def test_zero_recall_concepts_are_named_not_averaged_away():
    rows = [
        StoreRow(
            audit_id=f"c0-{i}",
            population="protected",
            gold_label="ANSWER",
            gold_concept_id="c0",
            scored=(("c0#000", "c0", 0.99, 0.01),),
        )
        for i in range(9)
    ] + [
        StoreRow(
            audit_id="c1-0",
            population="protected",
            gold_label="ANSWER",
            gold_concept_id="c1",
            scored=(("c1#000", "c1", 0.01, 0.01),),
        ),
        StoreRow(
            audit_id="n0",
            population="protected",
            gold_label="NONE",
            gold_concept_id="c0",
            scored=(("c0#000", "c0", 0.01, 0.01),),
        ),
    ]
    metrics = store_conditioned_metrics(rows, tau_answer=0.5, tau_partial=0.3)
    assert metrics["protected"]["answer_micro_recall"] == pytest.approx(0.9)
    assert metrics["protected"]["answer_macro_recall"] == pytest.approx(0.5)
    assert metrics["protected"]["zero_recall_concepts"] == ["c1"]


def test_thresholds_are_selected_separately_and_never_blended():
    rows = (
        [
            StoreRow(
                audit_id=f"p{i}",
                population="protected",
                gold_label="ANSWER",
                gold_concept_id="c0",
                scored=(("c0#000", "c0", 0.8, 0.1),),
            )
            for i in range(20)
        ]
        + [
            StoreRow(
                audit_id=f"n{i}",
                population="protected",
                gold_label="NONE",
                gold_concept_id="c0",
                scored=(("c0#000", "c0", 0.1, 0.1),),
            )
            for i in range(20)
        ]
        + [
            StoreRow(
                audit_id=f"pa{i}",
                population="protected",
                gold_label="PARTIAL",
                gold_concept_id="c0",
                scored=(("c0#000", "c0", 0.2, 0.7),),
            )
            for i in range(10)
        ]
    )
    chosen = select_thresholds(rows)
    assert chosen["tau_answer"] is not None
    assert chosen["tau_partial"] is not None
    assert chosen["selected_answer_micro_recall"] == pytest.approx(1.0)
    assert chosen["selected_protected_clean_fpr"] <= 0.10
