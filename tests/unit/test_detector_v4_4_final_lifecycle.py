"""The human and final-bank lifecycle: the bindings that were described but not enforced.

Every test here corresponds to a step the v4.4 protocol requires and that no code path
performed at ``d083c9f``. They fall into three groups:

* the human sample's two companion files -- the frozen detector's predictions on the exact
  250 rows, and the 250 adjudicated model labels joined from the two populations the draw
  comes from. Both were paragraphs in a runner's output asking the operator to produce them
  by hand;
* the final bank's identity and its one-build rule. ``--check-only --bank final`` wrote the
  ENGINEERING verification file, the final bank asserted the engineering ``bank_id`` in its
  own body, and nothing stopped a second build;
* ``finalize``, which did not exist: no artifact read the whole chain and said, in a form a
  machine can check, that every stage referred to the same detector.

Nothing here loads a model or touches a GPU. The detector-scoring path is exercised through
the shared decision rule rather than through a checkpoint, and every other test drives real
CLI functions over synthetic artifacts whose shape is copied from the real ones.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path

import pytest
import typer

from rdl.cli.detector_v4_2_bank_runs import (
    BANK_PLAN_FILENAME,
    BANK_SCRIPT_FILENAME,
    bank_run_commands,
)
from rdl.cli.detector_v4_2_banks import (
    BANK_ID,
    BUILD_RECORD_FILENAME,
    VERIFICATION_FILENAME,
    detector_v4_2_build_bank,
    final_generation_budget,
)
from rdl.cli.detector_v4_4_finalize import (
    FINAL_VALIDATION_FILENAME,
    detector_v4_4_finalize,
)
from rdl.cli.detector_v4_4_human import (
    DETECTOR_PREDICTIONS_FILENAME,
    DETECTOR_PREDICTIONS_MANIFEST,
    MODEL_CONSENSUS_FILENAME,
    SAMPLE_FILENAME,
    SAMPLE_KEY_FILENAME,
    detector_v4_4_human_model_consensus,
    detector_v4_4_human_report,
)
from rdl.eval.detector_v4_2 import SEALED_FINAL_BANK_SEEDS
from rdl.eval.detector_v4_3 import StoreRow, predicted_label

REPO = Path(__file__).resolve().parents[2]
V4_2 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_2"
ENGINEERING_SEEDS = {50241, 50242, 50243, 50244, 51241, 51242, 51243, 51244}


def _sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    return path


# =====================================================================================
# 1. the shared decision rule, and the predictions the human sample is scored into
# =====================================================================================


def test_predicted_label_lets_answer_dominate_partial():
    """ANSWER precedence lived only inside a metric expression, so every other caller
    restated it. A restated rule is a second rule that can drift."""
    over_both = StoreRow(
        audit_id="a", population="protected", gold_label=None, scored=(("s", "f", 0.9, 0.9),)
    )
    assert predicted_label(over_both, tau_answer=0.5, tau_partial=0.5) == "ANSWER"

    partial_only = StoreRow(
        audit_id="b", population="protected", gold_label=None, scored=(("s", "f", 0.1, 0.9),)
    )
    assert predicted_label(partial_only, tau_answer=0.5, tau_partial=0.5) == "PARTIAL"

    neither = StoreRow(
        audit_id="c", population="protected", gold_label=None, scored=(("s", "f", 0.1, 0.1),)
    )
    assert predicted_label(neither, tau_answer=0.5, tau_partial=0.5) == "NONE"


def test_an_unrouted_row_is_none_not_a_crash():
    """A row the router matched to nothing was never scored. It must not read as a low
    score, and it must not raise on the max() of an empty sequence."""
    unrouted = StoreRow(audit_id="d", population="protected", gold_label=None, scored=())
    assert predicted_label(unrouted, tau_answer=0.5, tau_partial=0.5) == "NONE"
    assert unrouted.routed is False


# =====================================================================================
# 2. the 250-row model consensus
# =====================================================================================


def _sample(tmp_path: Path, ids: list[str], *, reportable: bool = True) -> Path:
    """A drawn human sample, with the row-level key the reference pass rebuilds from."""
    key = _write_json(
        tmp_path / SAMPLE_KEY_FILENAME,
        {
            "schema": "graph-detector-v4-4-human-sample-key-v1",
            "n_rows": len(ids),
            "rows": [
                {
                    "audit_id": i,
                    "conditioning_question": f"who is {i}",
                    "subject_aliases": [i],
                    "candidate_text": f"candidate for {i}",
                    "source_stratum": "v4_4_population",
                    "sampling_stratum": "leaking|closed",
                }
                for i in ids
            ],
        },
    )
    return _write_json(
        tmp_path / SAMPLE_FILENAME,
        {
            "schema": "graph-detector-v4-4-human-sample-v1",
            "reportable": reportable,
            "n_rows": len(ids),
            "audit_ids": sorted(ids),
            "raters": ["A", "B"],
            "sample_key": str(key),
            "sample_key_sha256": _sha(key),
            "reference_pass": {"covers_whole_sample": True},
        },
    )


def test_model_consensus_joins_both_populations_for_exactly_the_sample(tmp_path):
    """The 250 is 125 bundle rows + 125 fresh rows, so the model labels live in two files
    and in neither one alone. Nothing built the join."""
    _sample(tmp_path, ["a1", "a2", "b1", "b2"])
    left = _write_jsonl(
        tmp_path / "V4_4_ADJUDICATED.jsonl",
        [
            {"audit_id": "a1", "answer_attempt": "ANSWER"},
            {"audit_id": "a2", "answer_attempt": "NONE"},
        ],
    )
    right = _write_jsonl(
        tmp_path / "FRESH_ADJUDICATED.jsonl",
        [
            {"audit_id": "b1", "answer_attempt": "PARTIAL"},
            {"audit_id": "b2", "answer_attempt": "ANSWER"},
        ],
    )
    detector_v4_4_human_model_consensus(out_dir=tmp_path, adjudicated=[left, right])

    rows = [
        json.loads(line)
        for line in (tmp_path / MODEL_CONSENSUS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [r["audit_id"] for r in rows] == ["a1", "a2", "b1", "b2"]
    assert {r["audit_id"]: r["answer_attempt"] for r in rows}["b1"] == "PARTIAL"


def test_model_consensus_refuses_a_missing_sampled_id(tmp_path):
    """ "250 rows" is satisfied by a file that drops one sampled id and adds another."""
    _sample(tmp_path, ["a1", "a2", "b1"])
    partial = _write_jsonl(
        tmp_path / "V4_4_ADJUDICATED.jsonl",
        [
            {"audit_id": "a1", "answer_attempt": "ANSWER"},
            {"audit_id": "a2", "answer_attempt": "NONE"},
        ],
    )
    with pytest.raises(typer.BadParameter, match="missing"):
        detector_v4_4_human_model_consensus(out_dir=tmp_path, adjudicated=[partial])


def test_model_consensus_keeps_only_the_sampled_rows(tmp_path):
    """The adjudicated files are the FULL label sets -- 1,733 rows on one side -- so rows
    outside the draw are the normal case and are dropped, not refused. What must never
    happen is one of them reaching the output: the human gate compares these labels to the
    humans' on the same rows, and an extra row would score against no human label at all.
    """
    _sample(tmp_path, ["a1"])
    source = _write_jsonl(
        tmp_path / "V4_4_ADJUDICATED.jsonl",
        [
            {"audit_id": "a1", "answer_attempt": "ANSWER"},
            {"audit_id": "zz", "answer_attempt": "ANSWER"},
        ],
    )
    detector_v4_4_human_model_consensus(out_dir=tmp_path, adjudicated=[source])
    written = [
        json.loads(line)
        for line in (tmp_path / MODEL_CONSENSUS_FILENAME).read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [r["audit_id"] for r in written] == ["a1"]


def test_model_consensus_refuses_two_files_that_disagree(tmp_path):
    """Both files were closed by adjudication. Picking one here would be a third
    adjudication, performed by whoever ran the command."""
    _sample(tmp_path, ["a1"])
    left = _write_jsonl(tmp_path / "one.jsonl", [{"audit_id": "a1", "answer_attempt": "ANSWER"}])
    right = _write_jsonl(tmp_path / "two.jsonl", [{"audit_id": "a1", "answer_attempt": "NONE"}])
    with pytest.raises(typer.BadParameter, match="duplicated"):
        detector_v4_4_human_model_consensus(out_dir=tmp_path, adjudicated=[left, right])


def test_model_consensus_requires_its_sources_to_be_named(tmp_path):
    """Defaulting to whatever adjudicated files are on disk would make the consensus depend
    on directory contents at run time."""
    _sample(tmp_path, ["a1"])
    with pytest.raises(typer.BadParameter, match="--adjudicated is required"):
        detector_v4_4_human_model_consensus(out_dir=tmp_path, adjudicated=[])


# =====================================================================================
# 3. the reportable human report refuses an unbound run
# =====================================================================================


def _human_evidence(tmp_path: Path, ids: list[str], *, reportable: bool = True) -> None:
    """Everything the report reads apart from the two bindings under test."""
    _sample(tmp_path, ids, reportable=reportable)
    _write_jsonl(
        tmp_path / "V4_4_HUMAN_BLIND_ADJUDICATED.jsonl",
        [{"audit_id": i, "answer_attempt": "ANSWER"} for i in ids],
    )
    _write_jsonl(
        tmp_path / "V4_4_HUMAN_REFERENCE_ADJUDICATED.jsonl",
        [{"audit_id": i, "reference_content": "YES"} for i in ids],
    )
    _write_json(
        tmp_path / "V4_4_HUMAN_BLIND_AGREEMENT.json",
        {"kappa": 0.9, "raw_agreement": 0.95, "confusion": {}, "rater_file_sha256": {}},
    )
    _write_jsonl(
        tmp_path / MODEL_CONSENSUS_FILENAME,
        [{"audit_id": i, "answer_attempt": "ANSWER"} for i in ids],
    )


def test_reportable_report_refuses_without_the_detector_predictions(tmp_path):
    """P0. Both flags defaulted to None and the report carried on, recording
    '--detector-predictions was not passed' as prose in a file whose whole purpose is to
    say whether the detector and a person agreed."""
    ids = ["a1", "a2"]
    _human_evidence(tmp_path, ids)
    with pytest.raises(typer.BadParameter, match="--detector-predictions"):
        detector_v4_4_human_report(
            out_dir=tmp_path,
            model_consensus=tmp_path / MODEL_CONSENSUS_FILENAME,
            detector_predictions=None,
            frozen_detector=tmp_path / "DETECTOR_V4_4_FROZEN_DETECTOR.json",
        )


def test_reportable_report_refuses_without_the_frozen_detector(tmp_path):
    ids = ["a1", "a2"]
    _human_evidence(tmp_path, ids)
    _write_jsonl(
        tmp_path / DETECTOR_PREDICTIONS_FILENAME,
        [{"audit_id": i, "predicted_label": "ANSWER"} for i in ids],
    )
    with pytest.raises(typer.BadParameter, match="--frozen-detector"):
        detector_v4_4_human_report(
            out_dir=tmp_path,
            model_consensus=tmp_path / MODEL_CONSENSUS_FILENAME,
            detector_predictions=tmp_path / DETECTOR_PREDICTIONS_FILENAME,
            frozen_detector=None,
        )


def test_an_exploratory_sample_may_still_run_without_the_bindings(tmp_path):
    """The flags stay optional for a draw marked non-reportable. That sample fails the gate
    for being non-reportable, which is a different and honest reason."""
    ids = ["a1", "a2"]
    _human_evidence(tmp_path, ids, reportable=False)
    detector_v4_4_human_report(
        out_dir=tmp_path,
        model_consensus=tmp_path / MODEL_CONSENSUS_FILENAME,
        detector_predictions=None,
        frozen_detector=None,
    )
    report = json.loads((tmp_path / "DETECTOR_V4_4_HUMAN_REPORT.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert any("not reportable" in f for f in report["provenance_failures"])


def test_report_records_a_mismatched_prediction_binding_as_a_failure(tmp_path):
    """A present-but-wrong binding is a finding, not an operator slip: the failing report
    IS the evidence, so it is written rather than refused."""
    ids = ["a1", "a2"]
    _human_evidence(tmp_path, ids)
    predictions = _write_jsonl(
        tmp_path / DETECTOR_PREDICTIONS_FILENAME,
        [{"audit_id": i, "predicted_label": "ANSWER"} for i in ids],
    )
    frozen = _write_json(
        tmp_path / "DETECTOR_V4_4_FROZEN_DETECTOR.json",
        {"tau_answer": 0.5, "tau_partial": 0.3, "frozen_detector_sha256": "abc"},
    )
    _write_json(
        tmp_path / DETECTOR_PREDICTIONS_MANIFEST,
        {
            "schema": "graph-detector-v4-4-human-detector-predictions-v1",
            # Scored against a DIFFERENT sample: the draw was repeated after scoring.
            "human_sample_sha256": "a-sample-that-is-not-this-one",
            "human_sample_key_sha256": "also-not-this-one",
            "predictions_sha256": _sha(predictions),
            "frozen_detector_sha256": _sha(frozen),
        },
    )
    detector_v4_4_human_report(
        out_dir=tmp_path,
        model_consensus=tmp_path / MODEL_CONSENSUS_FILENAME,
        detector_predictions=predictions,
        frozen_detector=frozen,
    )
    report = json.loads((tmp_path / "DETECTOR_V4_4_HUMAN_REPORT.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["prediction_bindings"]["matches"] is False
    assert any("human_sample_sha256" in f for f in report["provenance_failures"])


def test_report_flags_predictions_that_do_not_cover_the_sample(tmp_path):
    ids = ["a1", "a2", "a3"]
    _human_evidence(tmp_path, ids)
    predictions = _write_jsonl(
        tmp_path / DETECTOR_PREDICTIONS_FILENAME,
        [{"audit_id": "a1", "predicted_label": "ANSWER"}],
    )
    frozen = _write_json(
        tmp_path / "DETECTOR_V4_4_FROZEN_DETECTOR.json", {"tau_answer": 0.5, "tau_partial": 0.3}
    )
    detector_v4_4_human_report(
        out_dir=tmp_path,
        model_consensus=tmp_path / MODEL_CONSENSUS_FILENAME,
        detector_predictions=predictions,
        frozen_detector=frozen,
    )
    report = json.loads((tmp_path / "DETECTOR_V4_4_HUMAN_REPORT.json").read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert any("detector predictions cover" in f for f in report["provenance_failures"])


# =====================================================================================
# 4. the final run planner
# =====================================================================================


def _final_budget() -> dict:
    """The real frozen final budget, derived the way build-bank derives it."""
    payload = json.loads((V4_2 / "FINAL_GATE_BANK_BUDGET.json").read_text(encoding="utf-8"))
    return final_generation_budget(payload)


def test_final_plan_emits_eight_commands_over_the_sealed_seeds():
    """P0. The planner read the ENGINEERING manifest and emitted engineering run names, so
    the eight final commands were hand-composed on a metered box."""
    commands = bank_run_commands(
        _final_budget(),
        launch=Path("configs/graph/launch.yaml"),
        cohort_dir=Path("data/cohorts/graph_unlearning_v1"),
        output_root=Path("runs/graph"),
        profile="rtx3090_1b",
        bank="final",
    )
    assert len(commands) == 8
    for group in ("natural", "retain"):
        seeds = sorted(c["base_seed"] for c in commands if c["group"] == group)
        assert seeds == sorted(SEALED_FINAL_BANK_SEEDS), f"{group} is not the sealed set"


def test_no_engineering_seed_can_enter_a_final_plan():
    """The engineering and final banks share a directory and a command. A final plan that
    named an engineering seed would be rejected by build-bank only AFTER the seal had been
    lifted and the humans had already passed."""
    commands = bank_run_commands(
        _final_budget(),
        launch=Path("configs/graph/launch.yaml"),
        cohort_dir=Path("data/cohorts/graph_unlearning_v1"),
        output_root=Path("runs/graph"),
        profile="rtx3090_1b",
        bank="final",
    )
    assert not ENGINEERING_SEEDS & {c["base_seed"] for c in commands}
    for command in commands:
        assert "engineering" not in command["output"], command["output"]
        assert command["output"].startswith("runs/graph/final-")


def test_the_two_banks_do_not_share_plan_filenames():
    """A final plan written over ENGINEERING_BANK_RUNS.sh would destroy the record of how
    the engineering bank was generated, at the moment the final bank is being built."""
    assert BANK_PLAN_FILENAME["engineering"] != BANK_PLAN_FILENAME["final"]
    assert BANK_SCRIPT_FILENAME["engineering"] != BANK_SCRIPT_FILENAME["final"]
    assert "FINAL" in BANK_PLAN_FILENAME["final"]


# =====================================================================================
# 5. the final bank's identity and its one-build rule
# =====================================================================================


def test_final_check_only_does_not_overwrite_the_engineering_verification():
    """P0. --check-only wrote ENGINEERING_BANK_VERIFICATION.json whatever --bank said, so
    the operator's last sanity check before their one allowed build destroyed the other
    bank's verification record."""
    assert VERIFICATION_FILENAME["final"] != VERIFICATION_FILENAME["engineering"]
    assert VERIFICATION_FILENAME["engineering"] == "ENGINEERING_BANK_VERIFICATION.json"


def test_the_final_bank_has_its_own_bank_id():
    """The payload hard-coded the engineering id for BOTH banks, so a final bank asserted
    in its own body that it was the engineering one."""
    assert BANK_ID["final"] != BANK_ID["engineering"]
    assert BANK_ID["engineering"] == "detector_v4_2_engineering_v1"
    assert "engineering" not in BANK_ID["final"]


def test_a_second_final_build_is_refused(tmp_path):
    """P0. The final gate was already one-shot; the final BUILD was not. A disappointing
    final result could be answered by rebuilding the bank under the same sealed seeds."""
    (tmp_path / BUILD_RECORD_FILENAME["final"]).write_text(
        json.dumps({"bank": "final"}), encoding="utf-8"
    )
    with pytest.raises(typer.BadParameter, match="already been built"):
        detector_v4_2_build_bank(
            bank="final",
            natural_run=[],
            retain_run=[],
            manifest_dir=tmp_path,
            v4_1_dir=tmp_path,
            output_dir=tmp_path,
            check_only=False,
        )


def test_check_only_is_still_allowed_after_a_final_build(tmp_path):
    """Re-verifying runs against the freeze writes no bank. Refusing it would push the
    operator toward the one command that does write."""
    (tmp_path / BUILD_RECORD_FILENAME["final"]).write_text(
        json.dumps({"bank": "final"}), encoding="utf-8"
    )
    with pytest.raises(typer.BadParameter) as excinfo:
        detector_v4_2_build_bank(
            bank="final",
            natural_run=[],
            retain_run=[],
            manifest_dir=tmp_path,
            v4_1_dir=tmp_path,
            output_dir=tmp_path,
            check_only=True,
        )
    # It gets past the one-build guard and fails later, on the seal -- which is the next
    # thing a --check-only run legitimately has to satisfy.
    assert "already been built" not in str(excinfo.value)


def test_rebuilding_the_engineering_bank_is_not_refused(tmp_path):
    """Engineering data is engineering data. A one-build rule there would block ordinary
    work and teach the operator to delete records to get past it."""
    (tmp_path / BUILD_RECORD_FILENAME["engineering"]).write_text(
        json.dumps({"bank": "engineering"}), encoding="utf-8"
    )
    with pytest.raises(typer.BadParameter) as excinfo:
        detector_v4_2_build_bank(
            bank="engineering",
            natural_run=[],
            retain_run=[],
            manifest_dir=tmp_path,
            v4_1_dir=tmp_path,
            output_dir=tmp_path,
            check_only=False,
        )
    assert "already been built" not in str(excinfo.value)


# =====================================================================================
# 6. finalize
# =====================================================================================


def _chain(tmp_path: Path, **overrides) -> tuple[Path, Path]:
    """A complete, internally consistent v4.4 chain. Overrides break exactly one link."""
    v44 = tmp_path / "v4_4"
    v42 = tmp_path / "v4_2"
    v44.mkdir(parents=True, exist_ok=True)
    v42.mkdir(parents=True, exist_ok=True)

    _write_json(v44 / "DETECTOR_V4_4_LABEL_AUTHORITY.json", {"green": overrides.get("green", True)})
    _write_json(
        v44 / "DETECTOR_V4_MODEL.json",
        {
            "selected_checkpoint": "/workspace/ckpt",
            "checkpoint_digest": overrides.get("model_digest", "ckpt-digest"),
        },
    )
    point = _write_json(
        v44 / "DETECTOR_V4_4_OPERATING_POINT.json",
        {"tau_answer": 0.62, "tau_partial": 0.41, "operating_point_sha256": "op-sha"},
    )
    frozen = _write_json(
        v44 / "DETECTOR_V4_4_FROZEN_DETECTOR.json",
        {
            "tau_answer": 0.62,
            "tau_partial": 0.41,
            "operating_point_sha256": "op-sha",
            "checkpoint_digest": overrides.get("frozen_digest", "ckpt-digest"),
            "frozen_detector_sha256": "fd-content-sha",
            "protected_store_fingerprint": "store-fp",
        },
    )
    _write_json(
        v44 / "DETECTOR_V4_4_HELDOUT_GATE.json",
        {"passed": True, "reopened": False, "measured": {}},
    )
    human = _write_json(
        v44 / "DETECTOR_V4_4_HUMAN_REPORT.json",
        {
            "passed": overrides.get("human_passed", True),
            "reportable": True,
            "provenance_failures": [],
            "human_human": {"kappa": 0.81},
            "model_consensus_vs_human": {"macro_f1": 0.86},
        },
    )
    _write_json(v44 / "DETECTOR_V4_4_FRESH_LABELLED_final.json", {"passed": True})
    _write_json(
        v44 / "DETECTOR_V4_4_FINAL_GATE.json",
        {
            "passed": overrides.get("final_gate_passed", True),
            "reopened": overrides.get("final_gate_reopened", False),
            "operating_point_sha256": "op-sha",
            "measured": {},
        },
    )
    _write_json(
        v42 / "FINAL_BANK_UNSEAL_RECORD.json",
        {
            "schema": "graph-detector-v4-4-final-bank-unseal-v1",
            "unsealed": True,
            "human_report": {"file": str(human), "sha256": _sha(human), "passed": True},
            "frozen_detector": {
                "file": str(frozen),
                "sha256": _sha(frozen),
                "frozen_detector_sha256": "fd-content-sha",
            },
            "operating_point": {"file": str(point), "sha256": _sha(point)},
        },
    )
    _write_json(
        v42 / BUILD_RECORD_FILENAME["final"],
        {"bank": "final", "clean_tree": overrides.get("clean_tree", True)},
    )
    return v44, v42


def _run_finalize(v44: Path, v42: Path) -> dict:
    """Run finalize and return the record it wrote, whatever its exit code.

    The record is written on failure too -- that IS the evidence -- so the SystemExit a
    refusal raises is caught rather than allowed to hide the artifact under test.
    """
    with contextlib.suppress(typer.Exit, SystemExit):
        detector_v4_4_finalize(
            v4_4_dir=v44,
            v4_2_dir=v42,
            v4_1_dir=v42,
            model_artifact=v44 / "DETECTOR_V4_MODEL.json",
            output_dir=v44,
        )
    return json.loads((v44 / FINAL_VALIDATION_FILENAME).read_text(encoding="utf-8"))


def test_finalize_refuses_a_failed_final_gate(tmp_path):
    v44, v42 = _chain(tmp_path, final_gate_passed=False)
    record = _run_finalize(v44, v42)
    assert record["deployable"] is False
    assert record["answerability_v4_ready"] is False
    assert any("final_gate_passed" in f for f in record["failures"])


def test_finalize_refuses_a_missing_final_gate(tmp_path):
    v44, v42 = _chain(tmp_path)
    (v44 / "DETECTOR_V4_4_FINAL_GATE.json").unlink()
    record = _run_finalize(v44, v42)
    assert record["deployable"] is False
    assert any("final_gate" in f and "absent" in f for f in record["failures"])


def test_finalize_refuses_a_reopened_final_gate(tmp_path):
    """A gate opened twice reports a number that is not the first one this detector
    produced on the sealed bank."""
    v44, v42 = _chain(tmp_path, final_gate_reopened=True)
    record = _run_finalize(v44, v42)
    assert record["deployable"] is False
    assert any("REOPENING" in f for f in record["failures"])


def test_finalize_refuses_a_changed_checkpoint_digest(tmp_path):
    """The frozen detector and the model artifact must name the same checkpoint. If they do
    not, the chain describes two detectors and nothing says which one the final number
    belongs to."""
    v44, v42 = _chain(tmp_path, frozen_digest="a-different-checkpoint")
    record = _run_finalize(v44, v42)
    assert record["deployable"] is False
    assert record["bindings"]["checkpoint_the_frozen_detector_names"]["matches"] is False


def test_finalize_refuses_a_human_report_edited_after_the_unseal(tmp_path):
    """The human report is the ONLY thing permitted to unseal the final bank. The unseal
    record hashes it; editing it afterwards must not go unnoticed."""
    v44, v42 = _chain(tmp_path)
    _write_json(
        v44 / "DETECTOR_V4_4_HUMAN_REPORT.json",
        {
            "passed": True,
            "reportable": True,
            "provenance_failures": [],
            "human_human": {"kappa": 0.99},
            "model_consensus_vs_human": {"macro_f1": 0.99},
        },
    )
    record = _run_finalize(v44, v42)
    assert record["deployable"] is False
    assert record["bindings"]["human_report_that_unsealed_the_bank"]["matches"] is False


def test_finalize_refuses_a_failed_human_report(tmp_path):
    v44, v42 = _chain(tmp_path, human_passed=False)
    record = _run_finalize(v44, v42)
    assert record["deployable"] is False
    assert any("human_report_passed" in f for f in record["failures"])


def test_finalize_refuses_a_final_bank_built_from_a_dirty_tree(tmp_path):
    v44, v42 = _chain(tmp_path, clean_tree=False)
    record = _run_finalize(v44, v42)
    assert record["deployable"] is False
    assert any("dirty" in f for f in record["failures"])


def test_finalize_writes_deployable_only_when_the_whole_chain_passes(tmp_path):
    """The positive case. Everything above is one broken link away from this one."""
    v44, v42 = _chain(tmp_path)
    record = _run_finalize(v44, v42)
    assert record["failures"] == []
    assert record["deployable"] is True
    assert record["answerability_v4_ready"] is True
    assert record["detector"]["checkpoint_digest"] == "ckpt-digest"
    assert record["detector"]["tau_answer"] == 0.62
    # Every link is named and hashed, so a reviewer can re-check the chain from the record.
    assert set(record["chain"]) == {
        "label_authority",
        "model_artifact",
        "operating_point",
        "frozen_detector",
        "engineering_heldout_gate",
        "human_report",
        "final_bank_unseal_record",
        "final_bank_build_record",
        "final_labelled_audit",
        "final_gate",
    }
    assert all(link["sha256"] for link in record["chain"].values())


def test_finalize_does_not_treat_two_absent_hashes_as_agreement(tmp_path):
    """Two Nones is not a match. It is two artifacts that both declined to say, and
    treating it as agreement is how an unbound chain reports itself as bound."""
    v44, v42 = _chain(tmp_path)
    _write_json(v44 / "DETECTOR_V4_MODEL.json", {"selected_checkpoint": "/workspace/ckpt"})
    _write_json(
        v44 / "DETECTOR_V4_4_FROZEN_DETECTOR.json",
        {
            "tau_answer": 0.62,
            "tau_partial": 0.41,
            "operating_point_sha256": "op-sha",
            "frozen_detector_sha256": "fd-content-sha",
            "protected_store_fingerprint": "store-fp",
        },
    )
    record = _run_finalize(v44, v42)
    binding = record["bindings"]["checkpoint_the_frozen_detector_names"]
    assert binding["matches"] is False
    assert binding["stated"] is False


def test_finalize_never_edits_an_upstream_artifact(tmp_path):
    """A frozen file amended to say PASS is the failure the whole protocol is built
    against. The verdict goes in a NEW record."""
    v44, v42 = _chain(tmp_path)
    before = {
        path.name: _sha(path) for path in sorted(v44.glob("*.json")) + sorted(v42.glob("*.json"))
    }
    _run_finalize(v44, v42)
    after = {
        path.name: _sha(path)
        for path in sorted(v44.glob("*.json")) + sorted(v42.glob("*.json"))
        if path.name != FINAL_VALIDATION_FILENAME
    }
    for name, digest in after.items():
        assert before[name] == digest, f"{name} was modified by finalize"
