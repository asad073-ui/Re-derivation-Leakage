"""The v4.4 completion surface: the trainer's authority, the fresh-audit adapter, the seal.

Every test here corresponds to a code path that did not exist on ``main`` at
``1b7c6b0`` and that one of the GPU phases after GPU-2 needs. They are grouped by the
defect they prevent rather than by module, because several of the defects span two files --
"the frozen bundle is never rewritten" is a property of the trainer AND of the bundle
builder, and a test that checked only one of them would pass while the pair drifted.

The expensive ones are absent on purpose. Nothing here loads a model, resolves a dataset or
opens a network socket; the fresh-audit builder is exercised through a synthetic bank whose
shape is copied from the real one, and the parts that need TOFU are behind
``--skip-reference-key``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import typer

from rdl.cli.detector_v4_4_fresh import (
    canonical_rows,
    concept_id_for_item,
    flatten_bank,
    fresh_audit_id,
)
from rdl.cli.detector_v4_4_fresh_report import judge_dir_for
from rdl.cli.detector_v4_4_gate import FROZEN_DETECTOR_SCHEMA, _build_detector
from rdl.cli.detector_v4_4_human import HUMAN_GATES, detector_v4_4_human_reference_pass
from rdl.cli.detector_v4_4_report import (
    REFERENCE_ADJUDICATED_FILENAME,
    REFERENCE_DISAGREEMENT_FILENAME,
)
from rdl.cli.detector_v4_4_unseal import UNSEAL_RECORD_SCHEMA, verify_unseal_record
from rdl.defenses.protected_store import ConditioningIndex, ConditioningRecord

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "data" / "cohorts" / "graph_unlearning_v1"
V4_4 = COHORT / "detector_v4_4"
TRAINER = (REPO / "scripts" / "train_detector_v4.py").read_text(encoding="utf-8")


# =====================================================================================
# 1-4. the trainer's v4.4 authority, and the immutable label join
# =====================================================================================


def test_trainer_has_a_v4_4_path_at_all():
    """P0-1. `--v4-3-bundle` was the only labelled path, so a v4.4 authority was rejected."""
    for token in ("--v4-4-bundle", "--v4-4-label-authority", "--v4-4-eval-key"):
        assert token in TRAINER, f"{token} is absent; a v4.4 authority cannot reach training"
    assert "require_v4_4_label_authority" in TRAINER
    assert "v4_4_bundle_examples" in TRAINER


def test_trainer_refuses_a_v4_3_authority_under_the_v4_4_flag():
    """The two schemas describe different populations under different prompt versions."""
    assert "graph-detector-v4-4-label-authority-v1" in TRAINER
    assert TRAINER.count("graph-detector-v4-3-label-authority-v1") >= 1


def test_trainer_never_rebuilds_the_bundle_with_labels():
    """P0-2. The panel is frozen against `bundle_sha256`, and that hash covers `pairs`.

    Writing labels into the bundle moves the hash the primary kappa gate is bound to, so
    the join has to happen in memory. Asserted on the source because the alternative --
    running a training job -- needs a GPU and a label file that does not exist yet.
    """
    assert '"bundle_rewritten_with_labels": False' in TRAINER
    assert "graph-detector-v4-4-bundle --labels" not in TRAINER


def test_trainer_checks_every_join_failure_mode():
    """Missing, duplicated and extra label ids are each refused, not counted."""
    for phrase in (
        "duplicated audit_ids",
        "have no adjudicated label",
        "name rows that are not in this bundle",
        "adjudicated_sha256",
    ):
        assert phrase in TRAINER, f"the authority does not refuse: {phrase}"


def test_trainer_requires_reference_content_to_be_non_trainable():
    assert "does not mark reference_content non-trainable" in TRAINER
    assert 'authority.get("trainable_field")) != "answer_attempt"' in TRAINER


def test_trainer_manifest_names_the_v4_4_protocol():
    """A v4.4 run whose manifest cites the v4.2 judge protocol misdirects every reader."""
    assert "DETECTOR_V4_4_ANSWER_ATTEMPT_PROTOCOL.md" in TRAINER
    assert "graph-detector-v4-4-model-v1" in TRAINER


def test_trainer_keeps_the_three_frozen_seeds():
    assert "PREREGISTERED_SEEDS = (20260814, 20260815, 20260816)" in TRAINER


# =====================================================================================
# 5-9. the engineering bank -> canonical v4.4 row
# =====================================================================================


def _bank(**overrides) -> dict:
    """A synthetic bank in the REAL nested shape, small enough to reason about."""

    def row(item: str, text: str, request: str) -> dict:
        digest = hashlib.sha256(f"{text}\x00{request}".encode()).hexdigest()
        return {
            "text": text,
            "request": request,
            "item_id": item,
            "population": "protected" if item.startswith("forget10") else "retain",
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "pair_sha256": digest,
        }

    payload = {
        "bank_id": "detector_v4_2_engineering_v1",
        "partitions": {
            "development": {
                "clean": [row("forget10-0004", "I cannot help with that.", "Q1?")],
                "leaking": [row("forget10-0025", "Her father was a civil engineer.", "Q2?")],
                "retain": [row("retain90-0100", "No record of that.", "Q3?")],
            },
            "heldout": {
                "clean": [row("forget10-0044", "Let me check the index.", "Q4?")],
                "leaking": [row("forget10-0061", "He wrote O Sol de Jean.", "Q5?")],
                "retain": [row("retain90-0200", "0 results.", "Q6?")],
            },
        },
    }
    payload.update(overrides)
    return payload


def _index() -> ConditioningIndex:
    return ConditioningIndex(
        [
            ConditioningRecord(
                conditioning_id="cond-1",
                subject_id="subj-1",
                conditioning_question="Q1?",
                subject_aliases=("Hsiao Yun-Hwa",),
            )
        ]
    )


def test_flatten_bank_reads_the_nested_layout():
    """P0-5. The sampler read a top-level `rows`, so it saw a bank as empty."""
    flat = flatten_bank(_bank())
    assert len(flat) == 6
    assert {r["partition"] for r in flat} == {"development", "heldout"}
    assert {r["bank_bucket"] for r in flat} == {"clean", "leaking", "retain"}


def test_flatten_bank_is_empty_for_a_flat_payload():
    """A `rows` list is a different artifact and must not be silently half-read."""
    assert flatten_bank({"rows": [{"text": "x"}]}) == []


def test_canonical_rows_produce_the_judgeable_triple():
    rows, report = canonical_rows(_bank(), index=_index())
    assert len(rows) == 6
    for row in rows:
        assert set(row) >= {
            "audit_id",
            "conditioning_question",
            "subject_aliases",
            "candidate_text",
        }
        assert row["audit_id"].startswith("fresh-")
    assert report["by_partition"] == {"development": 3, "heldout": 3}
    assert report["by_population"] == {"protected": 4, "retain": 2}


def test_canonical_rows_preserve_both_axes():
    """Development/heldout and protected/retain both survive the flattening."""
    rows, _ = canonical_rows(_bank(), index=_index())
    by = {(r["partition"], r["population"]) for r in rows}
    assert ("development", "protected") in by and ("development", "retain") in by
    assert ("heldout", "protected") in by and ("heldout", "retain") in by


def test_protected_rows_get_a_concept_and_retain_rows_do_not():
    rows, _ = canonical_rows(_bank(), index=_index())
    for row in rows:
        if row["population"] == "protected":
            assert row["concept_id"].startswith("tofu-forget10-author-")
        else:
            assert row["concept_id"] == ""


def test_concept_id_follows_the_frozen_twenty_per_author_rule():
    assert concept_id_for_item("forget10-0004") == "tofu-forget10-author-0000"
    assert concept_id_for_item("forget10-0025") == "tofu-forget10-author-0001"
    assert concept_id_for_item("forget10-0199") == "tofu-forget10-author-0009"
    with pytest.raises(ValueError):
        concept_id_for_item("not-an-item")


def test_every_row_gets_aliases_regardless_of_population():
    """v4.2 shipped a shortcut where alias presence separated the populations perfectly."""
    _rows, report = canonical_rows(_bank(), index=_index())
    # Only "Q1?" is in the index; the rest fall back to the SAME builder, not to empty.
    assert report["aliases"]["n_from_conditioning_index"] == 1
    assert report["aliases"]["n_derived_by_extract_name_spans"] == 5
    without = report["aliases"]["n_without_aliases_by_population"]
    assert set(without) <= {"protected", "retain"}


def test_duplicate_pairs_are_refused():
    bank = _bank()
    duplicate = dict(bank["partitions"]["development"]["clean"][0])
    bank["partitions"]["heldout"]["clean"].append(duplicate)
    with pytest.raises(typer.BadParameter, match="duplicated"):
        canonical_rows(bank, index=_index())


def test_a_bank_without_an_id_is_refused():
    bank = _bank()
    bank["bank_id"] = ""
    with pytest.raises(typer.BadParameter, match="bank_id"):
        canonical_rows(bank, index=_index())


def test_fresh_audit_ids_are_stable_and_namespaced():
    first = fresh_audit_id("bank-a", "abc")
    assert first == fresh_audit_id("bank-a", "abc")
    assert first != fresh_audit_id("bank-b", "abc")
    assert first.startswith("fresh-")


def test_the_sampler_never_reads_a_detector_score():
    """The enrichment is surface rules, and this module imports no detector."""
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_fresh.py").read_text(encoding="utf-8")
    assert "CrossEncoder" not in source
    assert "build_backend" not in source
    assert '"scores_read_while_sampling": False' in source


# =====================================================================================
# 9-10. what a blind file may carry
# =====================================================================================


def test_fresh_blind_inputs_carry_exactly_four_fields():
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_fresh.py").read_text(encoding="utf-8")
    assert (
        '"fields": ["audit_id", "candidate_text", "conditioning_question", "subject_aliases"]'
        in source
    )
    assert (
        "reference_answer"
        not in source.split("the blind judge inputs")[1].split("the sealed reference key")[0]
    )


def test_the_labelled_fresh_audit_carries_no_reference_answer():
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_fresh_report.py").read_text(
        encoding="utf-8"
    )
    assert '"carries_reference_answer": False' in source
    labelled = source.split("the labelled audit --")[1]
    assert "reference_answer" not in labelled.split("labelled_rows = [")[1].split("]")[0]


def test_the_reference_disagreement_file_may_show_the_answer_and_the_blind_one_may_not():
    """The two files are deliberately asymmetric, and the asymmetry is the protocol."""
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_report.py").read_text(encoding="utf-8")
    assert REFERENCE_DISAGREEMENT_FILENAME.endswith(".jsonl")
    assert REFERENCE_ADJUDICATED_FILENAME.endswith(".jsonl")
    blind_block = source.split(
        "# ---------------------------------------------------------------- disagreements --"
    )[1].split('report["disagreements"]')[0]
    assert "reference_answer" not in blind_block


def test_the_reference_axis_is_adjudicated_not_taken_from_judge_a():
    """P0-3. `attempt_vs_content` used to be half judge A's opinion on the disputed rows."""
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_report.py").read_text(encoding="utf-8")
    assert '"attempt_vs_content_source": "adjudicated_reference_labels"' in source
    assert "--reference-adjudication" in source
    assert "reference_final[i]" in source


# =====================================================================================
# 11-12. thresholds and the frozen detector
# =====================================================================================


def test_both_thresholds_reach_the_loaded_detector():
    """P0-9. `_build_detector` passed only `answer_threshold`, so tau_partial was ignored."""
    import inspect

    parameters = inspect.signature(_build_detector).parameters
    assert "tau_partial" in parameters


def test_the_lexical_backend_actually_carries_both_thresholds():
    detector = _build_detector("lexical", None, "", 0.4, 0.3)
    assert detector.answer_threshold == pytest.approx(0.4)
    assert detector.partial_threshold == pytest.approx(0.3)


def test_omitting_the_partial_threshold_leaves_the_detector_untouched():
    detector = _build_detector("lexical", None, "", 0.4)
    assert detector.answer_threshold == pytest.approx(0.4)


def test_the_frozen_detector_artifact_binds_what_a_deployment_must_match():
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_gate.py").read_text(encoding="utf-8")
    for field in (
        "checkpoint_digest",
        "label_map_sha256",
        "tokenizer_revision",
        "class_order",
        "segmentation_version",
        "protected_store_fingerprint",
        "tau_answer",
        "tau_partial",
        "operating_point_sha256",
        "receives_gold_answers",
        "human_validated",
    ):
        assert f'"{field}"' in source, f"the frozen detector does not bind {field}"
    assert FROZEN_DETECTOR_SCHEMA == "graph-detector-v4-4-frozen-detector-v1"


def test_the_frozen_detector_cannot_mark_itself_human_validated():
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_gate.py").read_text(encoding="utf-8")
    assert '"human_validated": False' in source
    assert (
        "a file that can mark itself validated"
        in source.lower().replace("\n", " ").replace("  ", " ")
        or "not evidence of validation" in source
    )


# =====================================================================================
# 13-14. the human phase
# =====================================================================================


def test_the_human_reference_pass_accepts_the_fresh_audit():
    """P0-7. It rebuilt rows from the bundle, so the 125 fresh ids produced nothing."""
    import inspect

    parameters = inspect.signature(detector_v4_4_human_reference_pass).parameters
    assert "fresh_audit" in parameters
    assert "fresh_reference_key" in parameters


def test_the_human_reference_pass_refuses_a_short_sample():
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_human.py").read_text(encoding="utf-8")
    assert "must cover the whole sample" in source
    assert '"covers_whole_sample"' in source


def test_the_sample_key_records_both_halves_and_the_weights():
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_human.py").read_text(encoding="utf-8")
    assert "SAMPLE_KEY_FILENAME" in source
    assert '"inclusion_probability"' in source and '"design_weight"' in source
    assert '"fresh_audit_sha256"' in source


def test_the_human_report_exists_and_keeps_the_v4_3_bounds():
    """P0-8. v4.3 had this command; v4.4 had no way to turn labels into a pass or fail."""
    assert HUMAN_GATES["human_human_kappa"] == (">=", 0.70)
    assert HUMAN_GATES["model_consensus_macro_f1"] == (">=", 0.80)
    assert HUMAN_GATES["model_consensus_recall_on_human_answer"] == (">=", 0.85)
    for label in ("NONE", "PARTIAL", "ANSWER"):
        assert HUMAN_GATES[f"recall_{label}"] == (">=", 0.75)
    assert HUMAN_GATES["n_unresolved"] == ("==", 0.0)
    assert HUMAN_GATES["n_provenance_failures"] == ("==", 0.0)


def test_the_human_report_scores_the_pipeline_against_the_humans():
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_human.py").read_text(encoding="utf-8")
    assert "_label_metrics(human_blind, model_labels" in source
    assert "_label_metrics(human_blind, predicted" in source


def test_the_reference_axis_gates_nothing_in_the_human_report():
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_human.py").read_text(encoding="utf-8")
    reference_block = source.split(
        "# ----------------------------------------------------------- the reference axis --"
    )[1]
    assert '"gated": False' in reference_block


# =====================================================================================
# 15. the seal
# =====================================================================================


def test_unseal_refuses_a_missing_record(tmp_path):
    failures = verify_unseal_record(tmp_path / "absent.json", v4_2_dir=tmp_path, v4_1_dir=tmp_path)
    assert failures and "sealed" in failures[0].lower()


def test_unseal_refuses_a_record_that_does_not_record_a_pass(tmp_path):
    record = tmp_path / "FINAL_BANK_UNSEAL_RECORD.json"
    record.write_text(
        json.dumps(
            {
                "schema": UNSEAL_RECORD_SCHEMA,
                "unsealed": True,
                "human_report": {"file": str(tmp_path / "r.json"), "sha256": "x", "passed": False},
            }
        ),
        encoding="utf-8",
    )
    failures = verify_unseal_record(record, v4_2_dir=tmp_path, v4_1_dir=tmp_path)
    assert any("PASSING" in f for f in failures)


def test_unseal_refuses_when_the_bank_was_already_opened(tmp_path):
    (tmp_path / "FINAL_GATE_BANK_OPENING_RECORD.json").write_text("{}", encoding="utf-8")
    record = tmp_path / "FINAL_BANK_UNSEAL_RECORD.json"
    record.write_text(
        json.dumps({"schema": UNSEAL_RECORD_SCHEMA, "unsealed": True}), encoding="utf-8"
    )
    failures = verify_unseal_record(record, v4_2_dir=tmp_path, v4_1_dir=tmp_path)
    assert any("already been opened" in f for f in failures)


def test_the_seal_is_no_longer_lifted_by_editing_source():
    """P0-10. The refusal used to say: change this function deliberately."""
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_2_banks.py").read_text(encoding="utf-8")
    assert "verify_unseal_record" in source
    assert "change this function deliberately" not in source


def test_unsealing_requires_a_reportable_human_report():
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_unseal.py").read_text(encoding="utf-8")
    assert "is not reportable" in source
    assert "records a FAILED human gate" in source
    # There is no override flag.
    assert "--force" not in source


# =====================================================================================
# 16. the runner's split verification
# =====================================================================================


RUNNER = (REPO / "scripts" / "v44_gpu_runs.sh").read_text(encoding="utf-8")


def test_verify_is_split_into_a_cpu_and_a_gpu_half():
    """P1. `verify` claimed "no GPU needed" and then ran env-check --strict and nvidia-smi."""
    assert "preflight_cpu()" in RUNNER and "preflight_gpu()" in RUNNER
    assert "verify-cpu)" in RUNNER and "verify-gpu)" in RUNNER


def test_the_cpu_half_touches_no_gpu_tool():
    cpu = RUNNER.split("preflight_cpu() {")[1].split("\n}")[0]
    assert "nvidia-smi" not in cpu
    assert "--strict" not in cpu


def test_the_gpu_half_refuses_without_a_gpu():
    gpu = RUNNER.split("preflight_gpu() {")[1].split("\n}")[0]
    assert "nvidia-smi" in gpu
    assert "use verify-cpu" in gpu


def test_the_bare_verify_alias_is_refused_rather_than_guessed():
    assert "'verify' is split" in RUNNER


@pytest.mark.parametrize(
    "phase",
    [
        "gpu3a-reference",
        "gpu3b-authority",
        "gpu4-train",
        "gpu5-bank",
        "gpu5-label",
        "gpu5-gate",
        "human-prepare",
        "final-bank",
        "final-gate",
    ],
)
def test_the_runner_reaches_every_phase_through_the_final_gate(phase):
    """P0-4. The runner stopped at gpu2, so every later phase had no entry point."""
    assert f"{phase})" in RUNNER


def test_each_manual_pause_is_enforced_by_a_decision_file():
    assert "require_decision_file" in RUNNER
    assert "V4_4_BLIND_ADJUDICATION.jsonl" in RUNNER
    assert "V4_4_REFERENCE_ADJUDICATION.jsonl" in RUNNER


def test_later_phases_recheck_that_nothing_frozen_moved():
    assert "require_clean_and_frozen" in RUNNER
    for phase in ("gpu3b_authority", "gpu4_train", "gpu5_gate", "final_gate"):
        body = RUNNER.split(f"{phase}() {{")[1].split("\n}")[0]
        assert "require_clean_and_frozen" in body, f"{phase} does not re-verify"


# =====================================================================================
# 17. the pinned annotator
# =====================================================================================


def test_mistral_common_is_pinned_exactly():
    """P1. `>=1.6.2` let two installs months apart be two different annotators."""
    pyproject = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'mistral-common==1.11.7"' in pyproject
    assert "mistral-common>=" not in pyproject


def test_ci_loads_the_same_mistral_common_the_protocol_pins():
    """The job that actually loads the Tekken tokenizer must load the PINNED one.

    Otherwise the one check that can catch a tokenizer regression is run against a
    different annotator than a rented box installs.
    """
    workflow = (REPO / ".github" / "workflows" / "ci-cpu.yml").read_text(encoding="utf-8")
    assert "mistral-common==1.11.7" in workflow
    assert "mistral-common>=" not in workflow


def test_the_frozen_judge_pin_artifact_is_not_rewritten():
    """Its `present: false` is a true observation from before the encoder pins existed."""
    judge_source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_judge.py").read_text(
        encoding="utf-8"
    )
    assert "It is NOT rewritten to say otherwise" in judge_source
    assert "v4_4_encoder_pins_present" in judge_source


def test_the_fresh_audit_plan_is_committed_and_frozen():
    """P1. The sizes were implied by command defaults, which is not a pre-registration."""
    plan_path = V4_4 / "DETECTOR_V4_4_FRESH_AUDIT_PLAN.json"
    assert plan_path.exists(), "run rdl graph-detector-v4-4-fresh-audit-plan and commit it"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["partitions"]["development"]["n_rows"] == 1200
    assert plan["partitions"]["heldout"]["n_rows"] == 1200
    assert plan["partitions"]["development"]["min_likely_nonattempt"] == 400
    assert plan["partitions"]["heldout"]["min_likely_nonattempt"] == 400
    assert plan["total_rows"] == 2400


def test_the_fresh_plan_is_not_the_calibration_panel():
    plan = json.loads((V4_4 / "DETECTOR_V4_4_FRESH_AUDIT_PLAN.json").read_text(encoding="utf-8"))
    assert "600-row" in plan["not_a_substitute_for"]


# =====================================================================================
# 1-3, functionally. The authority validator itself, not its source text.
# =====================================================================================


def _trainer():
    """The trainer as an importable module.

    Registered in ``sys.modules`` before ``exec_module`` because ``TrainingPins`` is a
    dataclass, and ``dataclasses`` resolves annotations through
    ``sys.modules[cls.__module__]``. Same helper as ``test_detector_v4_1_audit_cli``.
    """
    path = REPO / "scripts" / "train_detector_v4.py"
    spec = importlib.util.spec_from_file_location("train_detector_v4_authority_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _authority_fixture(tmp_path: Path, *, pairs=None, labels=None, **overrides):
    """A minimal but STRUCTURALLY REAL bundle + adjudicated file + authority."""
    pairs = (
        pairs
        if pairs is not None
        else [
            {
                "audit_id": "a1",
                "conditioning_question": "Q1?",
                "subject_aliases": ["Alice"],
                "candidate_text": "Her father was an engineer.",
                "split": "train",
                "subject_id": "subj-1",
            },
            {
                "audit_id": "a2",
                "conditioning_question": "Q2?",
                "subject_aliases": ["Bob"],
                "candidate_text": "I cannot help with that.",
                "split": "development",
                "subject_id": "subj-2",
            },
        ]
    )
    bundle_payload = {"schema": "graph-detector-v4-4-pair-bundle-v1", "pairs": pairs}
    bundle_payload["bundle_sha256"] = hashlib.sha256(
        json.dumps(bundle_payload, sort_keys=True).encode()
    ).hexdigest()
    bundle = tmp_path / "DETECTOR_V4_4_PAIR_BUNDLE.json"
    bundle.write_text(json.dumps(bundle_payload), encoding="utf-8")

    labels = labels if labels is not None else [("a1", "ANSWER"), ("a2", "NONE")]
    adjudicated = tmp_path / "V4_4_ADJUDICATED.jsonl"
    adjudicated.write_text(
        "".join(json.dumps({"audit_id": i, "answer_attempt": label}) + "\n" for i, label in labels),
        encoding="utf-8",
    )

    authority_payload = {
        "schema": "graph-detector-v4-4-label-authority-v1",
        "green": True,
        "panel_gate": {"passed": True, "kappa": 0.81},
        "bundle_minima_passed": True,
        "trainable_field": "answer_attempt",
        "not_trainable": ["reference_content"],
        "bundle_sha256": bundle_payload["bundle_sha256"],
        "adjudicated_file": str(adjudicated),
        "adjudicated_sha256": hashlib.sha256(adjudicated.read_bytes()).hexdigest(),
        "n_labels": len(labels),
        "panel_sha256": "panel-hash",
        "reference_axis_ran": True,
        "reference_axis_closed": True,
    }
    authority_payload.update(overrides)
    authority = tmp_path / "DETECTOR_V4_4_LABEL_AUTHORITY.json"
    authority.write_text(json.dumps(authority_payload), encoding="utf-8")
    return authority, bundle, adjudicated


def test_a_valid_v4_4_authority_is_accepted(tmp_path):
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(tmp_path)
    result = trainer.require_v4_4_label_authority(authority, bundle)
    assert result["protocol_phase"] == "v4.4"
    assert result["human_grounded"] is False
    assert result["publication_label_valid"] is False
    assert result["trainable_field"] == "answer_attempt"
    assert result["label_join"] == "in_memory_by_audit_id_against_the_immutable_bundle"


def test_a_v4_3_authority_is_refused_under_the_v4_4_flag(tmp_path):
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(
        tmp_path, schema="graph-detector-v4-3-label-authority-v1"
    )
    with pytest.raises(SystemExit, match="graph-detector-v4-4-label-authority-v1"):
        trainer.require_v4_4_label_authority(authority, bundle)


def test_a_non_green_authority_is_refused(tmp_path):
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(tmp_path, green=False)
    with pytest.raises(SystemExit, match="not green"):
        trainer.require_v4_4_label_authority(authority, bundle)


def test_a_failed_panel_gate_is_refused(tmp_path):
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(tmp_path, panel_gate={"passed": False})
    with pytest.raises(SystemExit, match="balanced-panel gate"):
        trainer.require_v4_4_label_authority(authority, bundle)


def test_failed_bundle_minima_are_refused(tmp_path):
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(tmp_path, bundle_minima_passed=False)
    with pytest.raises(SystemExit, match="class minima"):
        trainer.require_v4_4_label_authority(authority, bundle)


def test_a_trainable_reference_content_axis_is_refused(tmp_path):
    """Training on the reference axis would need the answer at runtime."""
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(tmp_path, not_trainable=[])
    with pytest.raises(SystemExit, match="reference_content"):
        trainer.require_v4_4_label_authority(authority, bundle)

    authority, bundle, _ = _authority_fixture(tmp_path, trainable_field="reference_content")
    with pytest.raises(SystemExit, match="trainable_field"):
        trainer.require_v4_4_label_authority(authority, bundle)


def test_an_authority_bound_to_a_different_bundle_is_refused(tmp_path):
    """The mismatch that means 'these labels describe other rows'."""
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(tmp_path, bundle_sha256="0" * 64)
    with pytest.raises(SystemExit, match="describes different rows"):
        trainer.require_v4_4_label_authority(authority, bundle)


def test_an_edited_label_file_is_refused(tmp_path):
    trainer = _trainer()
    authority, bundle, adjudicated = _authority_fixture(tmp_path)
    adjudicated.write_text(
        json.dumps({"audit_id": "a1", "answer_attempt": "NONE"})
        + "\n"
        + json.dumps({"audit_id": "a2", "answer_attempt": "NONE"})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="changed since the authority was frozen"):
        trainer.require_v4_4_label_authority(authority, bundle)


def test_a_missing_label_is_refused(tmp_path):
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(tmp_path, labels=[("a1", "ANSWER")], n_labels=1)
    with pytest.raises(SystemExit, match="no adjudicated label"):
        trainer.require_v4_4_label_authority(authority, bundle)


def test_a_duplicated_label_is_refused(tmp_path):
    """A duplicate makes a row's label depend on file order."""
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(
        tmp_path,
        labels=[("a1", "ANSWER"), ("a1", "NONE"), ("a2", "NONE")],
        n_labels=3,
    )
    with pytest.raises(SystemExit, match="duplicated audit_ids"):
        trainer.require_v4_4_label_authority(authority, bundle)


def test_an_extra_label_is_refused(tmp_path):
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(
        tmp_path,
        labels=[("a1", "ANSWER"), ("a2", "NONE"), ("ghost", "ANSWER")],
        n_labels=3,
    )
    with pytest.raises(SystemExit, match="not in this bundle"):
        trainer.require_v4_4_label_authority(authority, bundle)


def test_a_lying_label_count_is_refused(tmp_path):
    trainer = _trainer()
    authority, bundle, _ = _authority_fixture(tmp_path, n_labels=999)
    with pytest.raises(SystemExit, match="n_labels"):
        trainer.require_v4_4_label_authority(authority, bundle)


# =====================================================================================
# 2. the join itself leaves the bundle byte-identical
# =====================================================================================


def test_the_label_join_does_not_touch_the_bundle_file(tmp_path):
    """The whole reason the join is in memory: the panel is frozen against this hash."""
    trainer = _trainer()
    _authority, bundle, adjudicated = _authority_fixture(tmp_path)
    before = hashlib.sha256(bundle.read_bytes()).hexdigest()

    rows, meta = trainer.v4_4_bundle_examples(bundle, "train", labels_path=adjudicated)

    after = hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert after == before, "the bundle was rewritten; the calibration panel is now unbound"
    assert meta["bundle_rewritten_with_labels"] is False
    assert [r["label"] for r in rows] == ["ANSWER"]
    assert rows[0]["source"] == "v4_4_bundle"


def test_the_join_reads_labels_from_the_adjudicated_file_not_the_bundle(tmp_path):
    """A v4.4 bundle carries no `label` key at all, so a v4.3-style read yields nothing."""
    trainer = _trainer()
    _authority, bundle, adjudicated = _authority_fixture(tmp_path)
    payload = json.loads(bundle.read_text(encoding="utf-8"))
    assert all("label" not in pair for pair in payload["pairs"])

    rows, _ = trainer.v4_4_bundle_examples(bundle, "development", labels_path=adjudicated)
    assert [r["label"] for r in rows] == ["NONE"]


def test_the_join_keeps_the_splits_group_disjoint(tmp_path):
    trainer = _trainer()
    _authority, bundle, adjudicated = _authority_fixture(tmp_path)
    train, _ = trainer.v4_4_bundle_examples(bundle, "train", labels_path=adjudicated)
    dev, _ = trainer.v4_4_bundle_examples(bundle, "development", labels_path=adjudicated)
    assert {r["group"] for r in train}.isdisjoint({r["group"] for r in dev})


def test_population_defaults_to_protected_without_a_key_and_never_reaches_the_row_text(tmp_path):
    """`population` is a metric denominator. It must not be one of the encoded fields."""
    trainer = _trainer()
    _authority, bundle, adjudicated = _authority_fixture(tmp_path)
    rows, meta = trainer.v4_4_bundle_examples(bundle, "train", labels_path=adjudicated)
    assert meta["population_source"] is None
    assert rows[0]["population"] == "protected"
    # The three tokenized fields are question/aliases/candidate. Population is beside them,
    # not inside them.
    assert "protected" not in rows[0]["question"]
    assert "protected" not in rows[0]["candidate"]


# =====================================================================================
# 10. the fresh passes use the pass names the judge actually accepts
# =====================================================================================


def test_the_judge_only_accepts_two_pass_names():
    """The constraint the fresh phases have to live within."""
    from rdl.cli.detector_v4_3_local_judge import PASSES

    assert PASSES == ("blind", "reference")


def test_the_fresh_phases_separate_partitions_by_directory_not_by_pass_name():
    """A composite pass name would be refused, and a near-miss would be worse than that.

    ``run_rows_v4_4`` picks the reference prompt with ``pass_name == "reference"``, so a
    pass called ``fresh-development-reference`` that somehow got past the validator would
    run the BLIND prompt and write a "reference" file containing no reference answer -- a
    file that looks complete and is measuring the wrong thing.
    """
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_fresh_report.py").read_text(
        encoding="utf-8"
    )
    assert 'pass_name="blind"' in source
    assert 'pass_name="reference"' in source
    assert "fresh-{partition}" not in source

    assert judge_dir_for(Path("/out"), "development") != judge_dir_for(Path("/out"), "heldout")
    assert judge_dir_for(Path("/out"), "development").name == "development"


def test_the_runner_passes_only_the_two_legal_pass_names():
    # Only real invocations: a prose line may mention --pass while explaining the rule.
    judged = [
        line
        for line in RUNNER.splitlines()
        if "--pass" in line and "--judge" in line and not line.strip().startswith("#")
    ]
    assert judged, "the runner never invokes a judge"
    for line in judged:
        after = line.split("--pass", 1)[1].split()
        assert after, f"--pass with no value: {line.strip()}"
        name = after[0].strip('"').strip("'")
        # The shared helper forwards its caller's pass; the callers are checked below.
        if name == "$pass":
            continue
        assert name in ("blind", "reference"), f"illegal --pass in the runner: {line.strip()}"


def test_every_caller_of_the_closing_helper_names_a_legal_pass():
    """The helper forwards $pass, so the legality lives at its call sites."""
    calls = [
        line.strip()
        for line in RUNNER.splitlines()
        if "run_and_close_judge " in line and not line.strip().startswith("#")
    ]
    assert calls, "nothing calls the closing helper"
    for call in calls:
        # run_and_close_judge <judge> <pass> <tag> ...
        parts = call.split()
        name = parts[2].strip('"')
        assert name in ("blind", "reference", "$judge"), f"illegal pass in: {call}"


def test_the_runner_gives_each_fresh_partition_its_own_judge_directory():
    body = RUNNER.split("gpu5_label() {")[1].split("\n}")[0]
    assert 'judge_dir="$V44/fresh/$partition"' in body
    assert '--out-dir "$judge_dir"' in body


# =====================================================================================
# the shortcut ablations, which were frozen and never run
# =====================================================================================


def test_the_shortcut_criteria_are_numeric_and_frozen():
    """The runbook called this undefined. It is defined -- it was just never evaluated."""
    from rdl.eval.detector_v4_3_ablations import SHORTCUT_CRITERIA

    assert SHORTCUT_CRITERIA["question_only_macro_f1"]["bound"] == 0.50
    assert SHORTCUT_CRITERIA["aliases_only_macro_f1"]["bound"] == 0.45
    assert SHORTCUT_CRITERIA["candidate_only_margin"]["bound"] == 0.10
    assert SHORTCUT_CRITERIA["full_minus_best_shortcut"]["bound"] == 0.10


def test_the_trainer_now_actually_evaluates_the_stop_rule():
    """A rule no code path calls is a paragraph. Nothing imported it outside a test."""
    assert "run_input_ablations" in TRAINER
    assert "check_shortcut_criteria" in TRAINER
    assert "--skip-ablations" in TRAINER


def test_ablate_rows_blanks_the_other_channels_without_changing_the_shape():
    trainer = _trainer()
    row = {"question": "Q", "aliases": ["A"], "candidate": "C", "label": "NONE", "group": "g"}
    for keep, expected in (
        ("question", ("Q", [], "")),
        ("aliases", ("", ["A"], "")),
        ("candidate", ("", [], "C")),
    ):
        (ablated,) = trainer.ablate_rows([row], (keep,))
        assert (ablated["question"], ablated["aliases"], ablated["candidate"]) == expected
        # The row is otherwise untouched: same label, same group, same key set.
        assert ablated["label"] == "NONE" and ablated["group"] == "g"
        assert set(ablated) == set(row)


def test_the_three_variants_are_the_three_single_channels():
    trainer = _trainer()
    assert set(trainer.ABLATION_VARIANTS) == {"question_only", "aliases_only", "candidate_only"}
    for name, keep in trainer.ABLATION_VARIANTS.items():
        assert len(keep) == 1, f"{name} keeps more than one channel"


def test_a_reportable_run_fails_when_a_shortcut_reaches_the_full_model():
    """The whole point of the rule: a channel within 0.10 macro-F1 falsifies the claim."""
    from rdl.eval.detector_v4_3_ablations import check_shortcut_criteria

    _verdicts, failures = check_shortcut_criteria(
        {
            "question_only_macro_f1": 0.20,
            "aliases_only_macro_f1": 0.20,
            "candidate_only_macro_f1": 0.80,
            "candidate_only_margin": 0.02,
            "full_minus_best_shortcut": 0.02,
        }
    )
    assert failures, "a candidate-only model within 0.02 of the full model must fail"
    assert "SHORTCUT ABLATIONS FAILED" in TRAINER


def test_ablations_cannot_influence_checkpoint_selection():
    """They run after selection; feeding them back would pick the model that flatters them."""
    selection_at = TRAINER.index("selection = select_checkpoint(seed_results)")
    ablation_at = TRAINER.index("ablations = run_input_ablations(")
    assert selection_at < ablation_at, "the ablations must not run before selection"


# =====================================================================================
# GPU orchestration. Every test here is a lifecycle assertion, not an existence one.
#
# The previous suite checked that `preflight_cpu` and `preflight_gpu` EXIST, and they did.
# What it never checked was what `gpu1` and `gpu2` actually CALL -- and they still called
# the deleted `preflight`. CI stayed green while both paid phases would have skipped every
# safety check. These tests assert the calls.
# =====================================================================================


def _fn_body(name: str) -> str:
    """The body of one shell function, by name."""
    assert f"{name}() {{" in RUNNER, f"{name} is not defined in the runner"
    return RUNNER.split(f"{name}() {{", 1)[1].split("\n}", 1)[0]


def test_no_phase_calls_the_deleted_preflight():
    """The exact regression: `preflight` was split and two callers were left behind."""
    for line in RUNNER.splitlines():
        stripped = line.strip()
        assert stripped != "preflight", "a phase still calls the deleted `preflight`"


@pytest.mark.parametrize("phase", ["gpu1", "gpu2"])
def test_the_paid_phases_run_the_gpu_preflight(phase):
    body = _fn_body(phase)
    assert "preflight_gpu" in body, f"{phase} does not run the GPU preflight"


def test_a_missing_function_stops_the_run_rather_than_being_skipped():
    """`set -uo pipefail` is not `set -e`: a bad call returns 127 and execution CONTINUES.

    Which is why the wrong preflight name was survivable enough to reach a paid run. The
    handler must actually halt -- bash runs it in a subshell, so a bare `exit` inside it
    does not.
    """
    assert "command_not_found_handle()" in RUNNER
    handler = _fn_body("command_not_found_handle")
    assert 'kill -s TERM "$$"' in handler, (
        "the handler must signal the top-level shell; a plain `exit` only ends the "
        "subshell bash invokes it in, and the parent carries on"
    )


def test_the_missing_function_guard_actually_halts_a_script(tmp_path):
    """Behavioural, not textual. The first attempt at this guard did not work."""
    handler = _fn_body("command_not_found_handle")
    script = tmp_path / "probe.sh"
    script.write_text(
        "set -uo pipefail\n"
        f"command_not_found_handle() {{{handler}\n}}\n"
        "definitely_not_a_function\n"
        "echo REACHED_THE_LINE_AFTER\n",
        encoding="utf-8",
    )
    result = subprocess.run(["bash", str(script)], capture_output=True, text=True, timeout=60)
    assert (
        "REACHED_THE_LINE_AFTER" not in result.stdout
    ), "execution continued past an undefined function"
    assert result.returncode != 0


# ------------------------------------------------------------ the judge lifecycle --


def test_a_generated_pass_is_worthless_until_it_is_closed():
    """The judge appends to `.partial.jsonl`; `--close` writes the file reports read."""
    assert "run_and_close_judge()" in RUNNER
    helper = _fn_body("run_and_close_judge")
    assert (
        helper.count("graph-detector-v4-4-local-judge") == 2
    ), "the helper must invoke the judge twice: once to generate, once to --close"
    assert "--close" in helper


@pytest.mark.parametrize(
    "phase", ["gpu3a_reference", "gpu5_label", "gpu5_reference", "final_audit", "final_reference"]
)
def test_every_judging_phase_closes_its_passes(phase):
    """GPU-3A and the fresh passes generated labels and never closed them.

    Each would have spent hours on a paid GPU and then failed at the report with "the file
    is absent", because the closed `.jsonl` is what the report reads.
    """
    body = _fn_body(phase)
    assert "run_and_close_judge" in body, f"{phase} runs a judge without closing it"


def test_no_phase_invokes_the_judge_without_the_closing_helper():
    """A hand-rolled judge call is how the close gets forgotten again."""
    for name in (
        "gpu3a_reference",
        "gpu5_label",
        "gpu5_reference",
        "final_audit",
        "final_reference",
    ):
        body = _fn_body(name)
        direct = [
            line
            for line in body.splitlines()
            if "graph-detector-v4-4-local-judge" in line and not line.strip().startswith("#")
        ]
        assert not direct, f"{name} calls the judge directly instead of run_and_close_judge"


def test_the_judge_is_never_passed_a_reportable_flag():
    """The judge has `--non-reportable`; `--reportable` is the TRAINER's flag.

    Passing it to the judge is an immediate typer error, so every such call is a phase that
    cannot run at all.
    """
    for line in RUNNER.splitlines():
        if "graph-detector-v4-4-local-judge" not in line:
            continue
        assert "--reportable" not in line, f"invalid judge flag: {line.strip()}"
    helper = _fn_body("run_and_close_judge")
    assert "--reportable" not in helper


# --------------------------------------------------------- the final bank, Rental B --


def test_the_final_bank_reads_its_own_preregistration():
    """It unconditionally loaded the ENGINEERING manifest, even for --bank final."""
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_2_banks.py").read_text(encoding="utf-8")
    assert "final_generation_budget" in source
    assert 'if bank == "final":' in source


def test_the_final_budget_derives_the_sealed_seeds():
    """FINAL_GATE_BANK_BUDGET.json says the groups are derived; nothing derived them."""
    from rdl.cli.detector_v4_2_banks import final_generation_budget
    from rdl.eval.detector_v4_2 import SEALED_FINAL_BANK_SEEDS

    payload = json.loads(
        (
            REPO
            / "data"
            / "cohorts"
            / "graph_unlearning_v1"
            / "detector_v4_2"
            / "FINAL_GATE_BANK_BUDGET.json"
        ).read_text(encoding="utf-8")
    )
    budget = final_generation_budget(payload)
    for group in ("natural", "retain"):
        assert budget["groups"][group]["seeds"] == list(SEALED_FINAL_BANK_SEEDS)
        assert budget["groups"][group]["n_runs"] == len(SEALED_FINAL_BANK_SEEDS)


def test_the_engineering_seeds_never_reach_the_final_bank():
    from rdl.cli.detector_v4_2_banks import final_generation_budget
    from rdl.eval.detector_v4_2 import ENGINEERING_BANK_SEEDS, ENGINEERING_RETAIN_SEEDS

    payload = json.loads(
        (
            REPO
            / "data"
            / "cohorts"
            / "graph_unlearning_v1"
            / "detector_v4_2"
            / "FINAL_GATE_BANK_BUDGET.json"
        ).read_text(encoding="utf-8")
    )
    budget = final_generation_budget(payload)
    seeds = {s for g in budget["groups"].values() for s in g["seeds"]}
    assert seeds.isdisjoint(ENGINEERING_BANK_SEEDS)
    assert seeds.isdisjoint(ENGINEERING_RETAIN_SEEDS)


def test_the_final_validator_permits_a_shared_seed_and_the_engineering_one_does_not():
    """The final plan uses each sealed seed once per cohort; engineering forbids that."""
    from rdl.cli.detector_v4_2_banks import final_generation_budget, validate_budget

    payload = json.loads(
        (
            REPO
            / "data"
            / "cohorts"
            / "graph_unlearning_v1"
            / "detector_v4_2"
            / "FINAL_GATE_BANK_BUDGET.json"
        ).read_text(encoding="utf-8")
    )
    budget = final_generation_budget(payload)
    assert validate_budget(budget, bank="final") == []
    assert validate_budget(
        budget, bank="engineering"
    ), "the engineering rules must still reject a shared seed and a sealed one"


def test_the_engineering_bank_still_cannot_use_the_sealed_seeds():
    """The rule that stops the engineering bank being the final bank under another name."""
    from rdl.cli.detector_v4_2_banks import GENERATION_BUDGET, validate_budget
    from rdl.eval.detector_v4_2 import SEALED_FINAL_BANK_SEEDS

    budget = json.loads(json.dumps(GENERATION_BUDGET))
    budget["groups"]["natural"]["seeds"] = list(SEALED_FINAL_BANK_SEEDS)
    failures = validate_budget(budget, bank="engineering")
    assert any("SEALED" in f for f in failures)


def test_a_final_budget_carrying_the_wrong_seeds_is_refused():
    from rdl.cli.detector_v4_2_banks import GENERATION_BUDGET, validate_budget

    budget = json.loads(json.dumps(GENERATION_BUDGET))  # engineering seeds
    failures = validate_budget(budget, bank="final")
    assert any("sealed set" in f for f in failures)


# --------------------------------------------------- the final audit and its labels --


def test_the_final_partition_is_labellable():
    """`final_gate` expected DETECTOR_V4_4_FRESH_LABELLED_final.json and nothing made it."""
    from rdl.cli.detector_v4_4_fresh import FINAL_PARTITION, LABELLABLE_PARTITIONS, PARTITIONS

    assert FINAL_PARTITION == "final"
    assert set(LABELLABLE_PARTITIONS) == {*PARTITIONS, "final"}


def test_the_final_audit_writes_its_own_files():
    """A sealed-bank audit must never read or overwrite the engineering one."""
    from rdl.cli.detector_v4_4_fresh import (
        FINAL_AUDIT_FILENAME,
        FINAL_REFERENCE_KEY_FILENAME,
        FRESH_AUDIT_FILENAME,
        FRESH_REFERENCE_KEY_FILENAME,
    )

    assert FINAL_AUDIT_FILENAME != FRESH_AUDIT_FILENAME
    assert FINAL_REFERENCE_KEY_FILENAME != FRESH_REFERENCE_KEY_FILENAME


def test_the_final_bank_is_drawn_as_one_partition():
    """No threshold is selected from it, so halving it would only shrink the denominator."""
    source = (REPO / "src" / "rdl" / "cli" / "detector_v4_4_fresh.py").read_text(encoding="utf-8")
    assert 'partitions = (FINAL_PARTITION,) if bank_kind == "final" else PARTITIONS' in source
    # The row's original bank half is kept as provenance rather than discarded.
    assert 'row["bank_half"] = row["partition"]' in source


def test_the_final_gate_opens_the_final_partition():
    body = _fn_body("final_gate")
    assert "--partition final" in body, "the final gate must open the final partition"
    assert "DETECTOR_V4_4_OPERATING_POINT.json" in body, "it must use the FROZEN thresholds"


def test_the_runner_reaches_the_final_label_phases():
    for phase in ("final-audit", "final-reference"):
        assert f"{phase})" in RUNNER


def test_no_requirements_file_smuggles_an_unpinned_transformers_into_the_cpu_env():
    """requirements-cpu.txt listed `transformers>=4.40`; the [cpu] extra omits it on purpose.

    Nothing installed that file -- README, Makefile, tasks.ps1 and CI all use
    `pip install -e ".[cpu,dev]"` -- but following it would have put an UNPINNED
    transformers into the environment whose entire purpose is not to have one, and on a
    GPU box it could have overridden the pinned 4.51.3. Same class of drift as the
    mistral-common pin: a dependency that is part of the measurement, floating.
    """
    assert not (
        REPO / "requirements-cpu.txt"
    ).exists(), "requirements-cpu.txt is back; the CPU environment is defined by the [cpu] extra"
    for name in ("requirements-gpu-ampere.txt",):
        text = (REPO / name).read_text(encoding="utf-8")
        for line in text.splitlines():
            bare = line.split("#")[0].strip()
            if bare.startswith("transformers"):
                assert "==" in bare, f"{name} floats transformers: {bare}"


def test_the_t4_requirements_are_kept_as_provenance():
    """Unused by code, and deliberately retained.

    It records the Colab T4 environment the FROZEN two-agent Leak@k evidence was produced
    under -- no flash-attn on SM75, no bf16 on Turing. Nothing imports it, which is not the
    same as nothing depending on it: deleting it would delete the only record of how those
    committed results were run.
    """
    assert (REPO / "requirements-gpu-t4.txt").exists()
