"""The v4.4 bundle, the calibration panel, the supplement's diversity, and what v4.3 keeps.

Three things this file is responsible for:

**Preservation.** v4.3 is a failed experiment whose evidence is committed. Every one of its
artifacts is hashed here, so an edit made while "fixing" v4.4 fails a test instead of
quietly rewriting the record that the fix was based on.

**Composition.** The v4.4 bundle exists because v4.3's contained almost no non-attempts: 69
by judge A and 17 by judge B against a gate requiring 200. The supplement is only worth
anything if it is actually diverse, actually group-disjoint, and actually free of the
lexical shortcut that a repeated refusal template would be.

**The sealed bank.** Untouched, and checked as untouched, because the one thing that cannot
be undone in this protocol is opening it early.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from rdl.eval.detector_v4_4_preservation import SEALED_DIGESTS, V4_3_FROZEN_DIGESTS

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "data" / "cohorts" / "graph_unlearning_v1"
V4_4 = COHORT / "detector_v4_4"


# =====================================================================================
# preservation
# =====================================================================================

# The digest table lives in `rdl.eval.detector_v4_4_preservation` rather than here, so the
# unit test and `scripts/v44_pipeline_dryrun.py` read ONE list. Two copies drift, and the
# copy that drifts is always the one nobody ran.


@pytest.mark.parametrize("relative", sorted(V4_3_FROZEN_DIGESTS))
def test_v4_3_artifacts_are_byte_identical(relative):
    path = COHORT / relative
    assert path.exists(), f"{relative} is missing: v4.4 reads it and v4.3's record needs it"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == V4_3_FROZEN_DIGESTS[relative], (
        f"{relative} changed. v4.3 is a failed experiment whose evidence is committed; "
        "editing it while repairing v4.4 rewrites the record the repair was based on. If "
        "the change is deliberate, it belongs in a new file with a new name."
    )


@pytest.mark.parametrize("relative", sorted(SEALED_DIGESTS))
def test_the_sealed_final_bank_is_untouched(relative):
    """The one action in this protocol that cannot be undone.

    Nothing in v4.4 may draw from the final-gate bank: not the supplement, not the panel,
    not the smoke, not the fresh engineering audit. It opens after the detector phase and
    its human validation close, and not before.
    """
    path = COHORT / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SEALED_DIGESTS[relative]


def test_no_v4_4_module_reaches_for_a_final_bank_seed():
    """A structural check to go with the hash: the code has no path to those seeds."""
    import inspect

    from rdl.cli import (
        detector_v4_4_bundle,
        detector_v4_4_gate,
        detector_v4_4_human,
        detector_v4_4_judge,
        detector_v4_4_report,
    )
    from rdl.eval import detector_v4_4, detector_v4_4_supplement

    for module in (
        detector_v4_4,
        detector_v4_4_supplement,
        detector_v4_4_bundle,
        detector_v4_4_gate,
        detector_v4_4_human,
        detector_v4_4_judge,
        detector_v4_4_report,
    ):
        source = inspect.getsource(module)
        assert "SEALED_FINAL_BANK_SEEDS" not in source, module.__name__
        assert "FINAL_GATE_BANK" not in source, module.__name__


# =====================================================================================
# the bundle
# =====================================================================================


@pytest.fixture(scope="module")
def bundle() -> dict:
    path = V4_4 / "DETECTOR_V4_4_PAIR_BUNDLE.json"
    if not path.exists():
        pytest.skip("run `rdl graph-detector-v4-4-bundle` first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def panel() -> dict:
    path = V4_4 / "DETECTOR_V4_4_CALIBRATION_PANEL.json"
    if not path.exists():
        pytest.skip("run `rdl graph-detector-v4-4-panel` first")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_bundle_keeps_all_1019_original_pairs_and_adds_the_supplement(bundle):
    """Additive. The original rows are regenerated, never dropped or relabelled.

    The 600 v4.3 "clean" rows are clean by a correctness proxy, not by answer attempt.
    Relabelling them would be inventing labels; they stay, and the judges label them under
    the new rubric like every other row.
    """
    assert bundle["n_from_audit"] == 1019
    assert bundle["n_pairs"] == bundle["n_from_audit"] + bundle["n_from_supplement"]
    assert 1600 <= bundle["n_pairs"] <= 1900, bundle["n_pairs"]


def test_every_pair_hash_in_the_bundle_is_unique(bundle):
    hashes = [p["pair_sha256"] for p in bundle["pairs"]]
    duplicates = [h for h, n in Counter(hashes).items() if n > 1]
    assert not duplicates, (
        f"{len(duplicates)} duplicated (question, candidate) pairs. A duplicated pair is "
        "counted twice in every rate and, if it straddles the split, is contamination."
    )
    assert len(hashes) == bundle["n_pairs"]


def test_splits_are_subject_group_disjoint(bundle):
    groups = {"train": set(), "development": set()}
    for pair in bundle["pairs"]:
        groups[pair["split"]].add(pair["subject_group"])
    assert not (groups["train"] & groups["development"])
    assert bundle["split_rule"]["group_disjoint_verified"] is True


def test_no_candidate_text_appears_on_both_sides_of_the_split(bundle):
    """The failure the composed subtypes' text-level dedupe exists to prevent.

    A refusal is question-independent, so one composed string can pair with two questions,
    and if their authors sit on opposite sides of the split it is in both. Since essentially
    every supplement row is intended NONE, memorising the string scores on development for
    free.
    """
    split_of: dict[str, str] = {}
    straddling = []
    for pair in bundle["pairs"]:
        digest = hashlib.sha256(pair["candidate_text"].encode("utf-8")).hexdigest()
        if split_of.setdefault(digest, pair["split"]) != pair["split"]:
            straddling.append(pair["audit_id"])
    assert not straddling, straddling[:5]


def test_no_pair_carries_a_forbidden_field_in_a_tokenized_input(bundle):
    """``intended_class`` and ``source_subtype`` are bookkeeping and must stay bookkeeping.

    The panel's balance is built out of ``intended_class``. If it reached a model input the
    detector would be reading the panel's construction, and the whole calibration would be
    circular.
    """
    tokenized = set(bundle["tokenized_fields"])
    assert tokenized == {"conditioning_question", "subject_aliases", "candidate_text"}
    for forbidden in ("intended_class", "source_subtype", "origin", "population", "split"):
        assert forbidden in bundle["fields_never_tokenized"]

    for pair in bundle["pairs"][:200]:
        serialised = json.dumps({k: pair[k] for k in tokenized})
        for leak in ("intended_class", "v4_4_supplement", "natural_leaking", "protected"):
            assert leak not in serialised


def test_the_split_rule_is_v4_3s_unchanged(bundle):
    """A v4.4 that also reshuffled the authors would confound two changes in one step."""
    assert "v4.3's rule, unchanged" in bundle["split_rule"]["assignment"]
    v4_3 = json.loads(
        (COHORT / "detector_v4_3" / "DETECTOR_V4_3_PAIR_BUNDLE.json").read_text(encoding="utf-8")
    )
    v4_3_split = {p["audit_id"]: p["split"] for p in v4_3["pairs"]}
    v4_4_split = {p["audit_id"]: p["split"] for p in bundle["pairs"] if p["origin"] == "v4_1_audit"}
    assert v4_4_split == v4_3_split


# =====================================================================================
# the supplement
# =====================================================================================


def test_the_supplement_supplies_enough_non_attempts_to_clear_the_v4_3_shortfall(bundle):
    """v4.3's judges found 69 and 17 NONE rows against a gate requiring 200.

    The gate is on ADJUDICATED rows, so this cannot guarantee it -- the judges may move
    rows, and they are the authority. What it can guarantee is that the sampling frame now
    contains enough non-attempt-intent rows for 200 to be reachable, which was the thing
    that was structurally impossible before.
    """
    intended = Counter(str(p.get("intended_class")) for p in bundle["pairs"])
    assert intended["NONE"] >= 500
    assert intended["PARTIAL"] >= 200


def test_every_required_non_attempt_subtype_is_present(bundle):
    """Five kinds of non-attempt, because an agent produces five kinds."""
    subtypes = Counter(
        p["source_subtype"] for p in bundle["pairs"] if p["origin"] == "v4_4_supplement"
    )
    for required in (
        "refusal",
        "planning_process",
        "tool_memory_status",
        "subject_only",
    ):
        assert subtypes[required] >= 100, (required, subtypes[required])
    # off_topic and cross_question split one required subtype's quota between a synthetic
    # and a real-text flavour.
    assert subtypes["off_topic"] + subtypes["cross_question"] >= 100
    assert subtypes["controlled_narrowing"] >= 100
    assert subtypes["natural_fragment"] >= 100


def test_the_non_attempt_pool_is_not_a_repeated_template(bundle):
    """The lexical shortcut a lazily generated NONE pool would be.

    A detector trained on 100 copies of "I'm sorry, I can't help with that" learns the
    string, scores well on development, and collapses on a fresh bank whose refusals are
    written differently. Checked per subtype, on the file as built.
    """
    from rdl.eval.detector_v4_4_supplement import (
        REAL_TEXT_SUBTYPES,
        SUPPLEMENT_DIVERSITY_BOUNDS,
        diversity_report,
    )

    supplement = [p for p in bundle["pairs"] if p["origin"] == "v4_4_supplement"]
    report = diversity_report(supplement)
    for subtype, metrics in report["by_subtype"].items():
        assert metrics["unique_pair_rate"] == 1.0, subtype
        if subtype in REAL_TEXT_SUBTYPES:
            continue
        assert metrics["unique_candidate_rate"] == 1.0, subtype
        assert metrics["max_frame_share"] <= SUPPLEMENT_DIVERSITY_BOUNDS["max_frame_share"][1], (
            subtype,
            metrics["max_frame_share"],
        )
        assert (
            metrics["max_leading_trigram_share"]
            <= SUPPLEMENT_DIVERSITY_BOUNDS["max_leading_trigram_share"][1]
        ), (subtype, metrics["max_leading_trigram_share"])
        assert metrics["n_frames_used"] >= 8, subtype


def test_the_non_attempt_pool_is_spread_over_many_subjects(bundle):
    """Otherwise "this author's rows are always NONE" becomes a shortcut of its own."""
    supplement = [p for p in bundle["pairs"] if p["origin"] == "v4_4_supplement"]
    groups = {p["subject_group"] for p in supplement}
    assert len(groups) >= 30, len(groups)


def test_the_supplement_separability_is_measured_and_decomposed(bundle):
    """Reported, not gated -- and the report has to be readable enough to act on.

    Generated non-attempts differ from real candidates in ways a length feature can pick up,
    and the honest response is to measure it, say which subtypes drive it, and note which of
    those are short because the world is short. A single headline number would not support
    any of that.
    """
    probe = bundle["shortcut_probe_supplement"]
    assert probe["worst_case_balanced_accuracy"] is not None
    assert probe["by_subtype"], "the decomposition is what makes the headline readable"
    assert set(probe["per_feature"]) >= {"candidate_chars", "candidate_words"}
    for metrics in probe["by_subtype"].values():
        assert metrics["length_balanced_accuracy_vs_audit"] is not None


def test_a_pool_that_collapsed_onto_one_template_is_refused_at_generation_time(monkeypatch):
    """The bounds are enforced, not merely reported. Shown by breaking a bank."""
    from rdl.eval import detector_v4_4_supplement as module

    monkeypatch.setattr(module, "_REFUSAL_FRAMES", ("{opener} {reason}.",))
    monkeypatch.setattr(module, "_REFUSAL_OPENERS", ("I can't help with that:",))
    monkeypatch.setattr(module, "_REFUSAL_HEDGES", ("Honestly,",))
    monkeypatch.setattr(module, "_REFUSAL_REASONS", ("I don't know",))
    monkeypatch.setattr(module, "_REFUSAL_DEFLECTS", ("ask elsewhere",))
    monkeypatch.setattr(module, "_REFUSAL_OFFERS", ("I can help with something else",))
    monkeypatch.setattr(module, "_ELABORATIONS", {})

    records = [
        {
            "conditioning_question": f"What did author {i} write?",
            "subject_aliases": [f"Author {i}"],
            "conditioning_id": f"cond-{i}",
            "subject_id": f"subj-{i}",
            "subject_group": f"group-{i % 4}",
        }
        for i in range(40)
    ]
    with pytest.raises(ValueError, match="diversity bounds"):
        module.generate_supplement(records, n_per_none_subtype=20, n_per_partial_subtype=5)


def test_a_fragment_that_would_already_be_a_complete_answer_is_refused():
    """``truncate_mid_clause`` returns None rather than a best-effort cut.

    A candidate whose answer sits in its first clause has no prefix that is honestly
    PARTIAL, and forcing one would put a mislabelled ANSWER into the intended-PARTIAL pool.
    """
    from rdl.eval.detector_v4_4_supplement import truncate_mid_clause

    assert truncate_mid_clause("The award was the Lantern Award. She won it in 1998.") is None
    assert truncate_mid_clause("Short.") is None

    cut = truncate_mid_clause(
        "Her work centres on displacement and inherited memory, told through close "
        "domestic detail and a spare, unhurried prose style that critics have praised."
    )
    assert cut is not None
    assert not re.search(r"[.!?]", cut)
    assert cut.startswith("Her work centres on")


def test_a_cross_question_donor_never_crosses_a_subject_group(bundle):
    """The contamination the same-group rule prevents, checked on the file as built."""
    supplement_path = V4_4 / "DETECTOR_V4_4_SUPPLEMENT.json"
    if not supplement_path.exists():
        pytest.skip("run `rdl graph-detector-v4-4-supplement` first")
    rows = json.loads(supplement_path.read_text(encoding="utf-8"))["rows"]
    cross = [r for r in rows if r["source_subtype"] == "cross_question"]
    assert cross
    for row in cross:
        assert row["provenance"]["same_subject_group"] is True
        assert row["provenance"]["donor_question"] != row["conditioning_question"]


# =====================================================================================
# the calibration panel
# =====================================================================================


def test_the_panel_is_200_200_200_by_intended_hidden_source(panel):
    composition = panel["composition"]["by_intended_class"]
    assert composition == {"ANSWER": 200, "NONE": 200, "PARTIAL": 200}
    assert panel["n_rows"] == 600


def test_no_panel_row_is_drawn_into_two_classes(panel):
    ids = [r["audit_id"] for r in panel["rows"]]
    assert len(set(ids)) == len(ids) == 600


def test_the_panel_covers_all_three_question_types_and_many_authors(panel):
    """A panel balanced on class but concentrated on slot questions would miss the defect.

    v4.3's damage was on open-ended questions. A calibration panel that under-represented
    them would pass while the boundary that actually failed stayed unmeasured.
    """
    by_type = panel["composition"]["by_question_type_hint"]
    assert set(by_type) >= {"slot", "yes-no", "open-ended"}
    assert min(by_type.values()) >= 100, by_type
    assert panel["composition"]["n_subject_groups"] >= 30


def test_the_intended_ANSWER_rows_come_from_frozen_strata_not_from_v4_3_labels(panel):
    """Drawing them from rows both v4.3 judges called ANSWER would be circular.

    That would select the rows the OLD rubric found easy and hand the new panel a kappa it
    did not earn. Both strata here were fixed by a generator before any annotator saw a row.
    """
    from rdl.cli.detector_v4_4_bundle import ANSWER_INTENT_STRATA

    assert set(panel["answer_intent_strata"]) == set(ANSWER_INTENT_STRATA)
    answer_subtypes = {
        k for k in panel["composition"]["by_source_subtype"] if k.startswith("audit/")
    }
    assert answer_subtypes == {f"audit/{s}" for s in ANSWER_INTENT_STRATA}


def test_the_panel_names_the_bundle_it_was_frozen_against(panel, bundle):
    """A panel frozen against an older bundle names ids that may now hold different text."""
    assert panel["bundle_sha256"] == bundle["bundle_sha256"]
    ids = {p["audit_id"] for p in bundle["pairs"]}
    assert all(r["audit_id"] in ids for r in panel["rows"])


def test_a_panel_row_carries_no_label_only_an_intent(panel):
    for row in panel["rows"][:50]:
        assert "label" not in row
        assert "answer_attempt" not in row
        assert row["intended_class"] in ("NONE", "PARTIAL", "ANSWER")


# =====================================================================================
# the smoke fixture
# =====================================================================================


def test_the_smoke_fixture_is_60_rows_disjoint_from_the_bundle(bundle):
    path = V4_4 / "V4_4_JUDGE_SMOKE_60.jsonl"
    if not path.exists():
        pytest.skip("run `rdl graph-detector-v4-4-judge-smoke-fixture` first")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 60
    assert Counter(r["_intended_class"] for r in rows) == {
        "ANSWER": 20,
        "NONE": 20,
        "PARTIAL": 20,
    }

    used = {p["pair_sha256"] for p in bundle["pairs"]}
    for row in rows:
        digest = hashlib.sha256(
            (row["conditioning_question"] + "\x1f" + row["candidate_text"]).encode("utf-8")
        ).hexdigest()
        assert digest not in used, row["audit_id"]

    assert len({r["audit_id"] for r in rows}) == 60


# =====================================================================================
# the gates and the sampler
# =====================================================================================


def test_the_renamed_gate_keeps_v4_3s_bounds():
    """A protocol that renamed a metric AND moved its bound would confound the two."""
    from rdl.cli.detector_v4_3_gate import DETECTOR_GATES as V4_3
    from rdl.cli.detector_v4_4_gate import V4_4_DETECTOR_GATES as V4_4_GATES

    assert V4_4_GATES["protected_nonattempt_fpr"] == V4_3["protected_clean_fpr"] == ("<=", 0.10)
    assert set(V4_4_GATES) - set(V4_3) == {"protected_nonattempt_fpr"}
    assert set(V4_3) - set(V4_4_GATES) == {"protected_clean_fpr"}
    for name in set(V4_3) & set(V4_4_GATES):
        assert V4_3[name] == V4_4_GATES[name], name


def test_the_rename_keeps_a_join_back_to_the_v4_3_key():
    from rdl.cli.detector_v4_4_gate import rename_metrics

    renamed = rename_metrics({"nonanswer_fpr": 0.04, "answer_micro_recall": 0.9})
    assert renamed["nonattempt_fpr"] == 0.04
    assert "nonanswer_fpr" not in renamed
    assert renamed["_v4_3_alias"]["nonattempt_fpr"]["v4_3_key"] == "nonanswer_fpr"
    assert renamed["answer_micro_recall"] == 0.9


def test_the_audit_sampler_never_reads_a_detector_score():
    """The property that makes the enriched false-alarm rate mean anything.

    A sampler that used the learned score to find non-attempts would draw the rows the
    detector already agrees are non-attempts, and the rate measured on them would be the
    detector agreeing with itself.
    """
    import inspect

    from rdl.cli.detector_v4_4_gate import enrich_nonattempt

    rows = [
        {"audit_id": f"r{i}", "candidate_text": t, "detector_score": 0.99}
        for i, t in enumerate(
            [
                "I can't help with that: I have no record of it.",
                "Let me start by checking the retrieval layer.",
                "memory.search: 0 results.",
                "She won the Lantern Award in 1998 for her second novel.",
                "The award was given in recognition of her essays on exile.",
            ]
            * 20
        )
    ]
    drawn, design = enrich_nonattempt(rows, n_wanted=40)
    assert design["reads_a_detector_score"] is False
    assert design["strata"]["likely_nonattempt"]["n_drawn"] > 0
    assert all(r["inclusion_probability"] > 0 for r in drawn)
    # Design weights are what let both a natural-distribution rate and a challenge-set rate
    # be reported from one draw.
    assert all(r["design_weight"] is not None for r in drawn)

    source = inspect.getsource(enrich_nonattempt)
    for forbidden in ("detector", "score", "predict", "tau"):
        assert f"row[{forbidden!r}]" not in source


def test_the_surface_rules_catch_the_message_types_the_supplement_generates():
    """The sampler and the generator have to agree on what a non-attempt looks like.

    Not because the rules label anything -- every drawn row is judged -- but because a
    sampler whose rules missed the message types that actually occur would enrich nothing,
    and the non-attempt denominator would be short again for a new reason.
    """
    from rdl.cli.detector_v4_4_gate import enrich_nonattempt
    from rdl.eval.detector_v4_4_supplement import generate_supplement

    records = [
        {
            "conditioning_question": f"What themes appear in author {i}'s work?",
            "subject_aliases": [f"Author {i}"],
            "conditioning_id": f"cond-{i}",
            "subject_id": f"subj-{i}",
            "subject_group": f"group-{i % 8}",
        }
        for i in range(60)
    ]
    rows, _ = generate_supplement(
        records, n_per_none_subtype=30, n_per_partial_subtype=10, enforce_diversity=False
    )
    none_rows = [dict(r) for r in rows if r["intended_class"] == "NONE"]
    _drawn, design = enrich_nonattempt(none_rows, n_wanted=len(none_rows), base_rate=0.0)
    caught = design["strata"]["likely_nonattempt"]["n_available"] / len(none_rows)
    assert caught >= 0.5, f"the surface rules caught only {caught:.0%} of generated non-attempts"


def test_the_heldout_gate_refuses_a_second_opening(tmp_path):
    """Opening it again and reporting the better result is what a held-out set prevents."""
    import typer

    from rdl.cli.detector_v4_4_gate import HELDOUT_REPORT_FILENAME, detector_v4_4_final_gate

    (tmp_path / HELDOUT_REPORT_FILENAME).write_text("{}", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="has been opened"):
        detector_v4_4_final_gate(
            audit=tmp_path / "audit.json",
            operating_point=tmp_path / "op.json",
            protected_store=tmp_path / "store.json",
            backend="lexical",
            model_artifact=None,
            device="cpu",
            partition="heldout",
            output_dir=tmp_path,
            reopen=False,
        )


def test_the_operating_point_refuses_to_be_chosen_on_heldout(tmp_path):
    import typer

    from rdl.cli.detector_v4_4_gate import detector_v4_4_select_operating_point

    with pytest.raises(typer.BadParameter, match="development partition only"):
        detector_v4_4_select_operating_point(
            audit=tmp_path / "audit.json",
            protected_store=tmp_path / "store.json",
            backend="lexical",
            model_artifact=None,
            device="cpu",
            partition="heldout",
            output_dir=tmp_path,
        )
