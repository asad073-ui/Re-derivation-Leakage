"""Two more things the v4.2 pipeline must refuse, both of which produce numbers not errors.

1. **An aggregate false-alarm rate concealing a retain failure.** Checkpoint selection read
   one pooled NONE-FPR over every negative row. The retain rows are the minority of that
   pool — two example classes of six in the synthetic corpus — so a checkpoint firing on a
   third of all retain traffic still clears an aggregate ceiling of 0.10 and is selected.
   The downstream gate then rejects it, after the GPU time is spent.

2. **A new bank gated on an old bank's labels.** ``final-gate`` originally wrote an opening
   record and printed the name of a different command, which reads the **v4** natural bank
   and the **v4.1** audit. Nothing tied the labels to the bank being opened, so the fresh
   surface could be marked "opened" while every number came from the surface the detector
   was developed on.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import typer

REPO = Path(__file__).resolve().parents[2]


def _trainer():
    path = REPO / "scripts" / "train_detector_v4.py"
    spec = importlib.util.spec_from_file_location("train_detector_v4_selection_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =====================================================================================
# An aggregate clean FPR cannot conceal a retain-FPR failure
# =====================================================================================


def _pool(trainer):
    """A development pool where the two negative populations behave very differently.

    * 100 protected ANSWER rows scoring 0.9  — the recall numerator
    * 100 protected NONE rows scoring 0.1    — quiet, as a good detector should be
    *  20 retain rows scoring 0.9            — the detector fires on ALL of them

    Aggregate NONE FPR at threshold 0.9: 20 firing of 120 negatives = 0.167. Under the old
    rule that is already above 0.10, so make the protected pool larger to reproduce the
    concealment exactly: with 400 quiet protected rows the aggregate is 20/420 = 0.048,
    comfortably under the ceiling, while the retain rate is 1.00.
    """
    answer, none = trainer.ANSWER_INDEX, trainer.LABELS.index("NONE")
    scores = [0.9] * 100 + [0.1] * 400 + [0.9] * 20
    golds = [answer] * 100 + [none] * 400 + [none] * 20
    populations = ["protected"] * 500 + ["retain"] * 20
    return scores, golds, populations


def test_the_aggregate_rate_really_would_have_hidden_it():
    """The arithmetic the fix is aimed at, stated as a test so it cannot be argued about."""
    trainer = _trainer()
    scores, golds, populations = _pool(trainer)
    none = trainer.LABELS.index("NONE")
    negatives = [s for s, g in zip(scores, golds, strict=True) if g == none]
    aggregate = sum(1 for s in negatives if s >= 0.9) / len(negatives)
    retain = sum(
        1 for s, p in zip(scores, populations, strict=True) if p == "retain" and s >= 0.9
    ) / populations.count("retain")
    assert aggregate <= trainer.DEV_FPR_CEILING, aggregate
    assert retain > trainer.DEV_FPR_CEILING, retain


def test_selection_refuses_the_threshold_the_aggregate_rate_would_have_chosen():
    trainer = _trainer()
    scores, golds, populations = _pool(trainer)

    with_populations = trainer.selection_metrics(scores, golds, populations)
    # The only thresholds clearing BOTH ceilings are above 0.9, where recall is 0.
    assert with_populations["selection_retain_fpr"] is not None
    assert with_populations["selection_retain_fpr"] <= trainer.DEV_FPR_CEILING
    assert with_populations["selection_protected_clean_fpr"] <= trainer.DEV_FPR_CEILING
    assert with_populations["selection_answer_recall"] == 0.0

    # And the pooled view, which is what the old code computed, picks 0.9 and reports
    # perfect recall — the number that would have been trained on.
    pooled = trainer.selection_metrics(scores, golds, None)
    assert pooled["selection_answer_recall"] == 1.0
    assert pooled["selection_threshold"] == pytest.approx(0.9)


def test_a_checkpoint_whose_retain_rate_was_never_measured_is_not_eligible():
    """ "We did not measure it" and "it was fine" must not produce the same verdict."""
    trainer = _trainer()
    unmeasured = {
        "seed": 1,
        "epoch": 1,
        "development": {
            "selection_answer_recall": 0.95,
            "selection_protected_clean_fpr": 0.02,
            "selection_retain_fpr": None,
            "selection_threshold": 0.5,
        },
    }
    result = trainer.select_checkpoint([{"history": [unmeasured]}])
    assert result["selected"] is None
    assert "retain" in result["reason"]


def test_a_checkpoint_failing_only_the_retain_ceiling_is_not_eligible():
    trainer = _trainer()
    entries = [
        {
            "seed": 1,
            "epoch": 1,
            "development": {
                "selection_answer_recall": 0.99,
                "selection_protected_clean_fpr": 0.01,
                "selection_retain_fpr": 0.40,
                "selection_threshold": 0.5,
            },
        },
        {
            "seed": 2,
            "epoch": 1,
            "development": {
                "selection_answer_recall": 0.60,
                "selection_protected_clean_fpr": 0.05,
                "selection_retain_fpr": 0.05,
                "selection_threshold": 0.7,
            },
        },
    ]
    result = trainer.select_checkpoint([{"history": entries}])
    assert result["selected"]["seed"] == 2, "the higher-recall checkpoint fails retain"
    assert result["n_eligible"] == 1


def test_the_frozen_selection_rule_names_both_ceilings():
    trainer = _trainer()
    rule = trainer.CHECKPOINT_SELECTION_RULE
    assert "protected-clean FPR" in rule
    assert "retain FPR" in rule
    assert "BOTH" in rule


def test_the_synthetic_rows_carry_their_population():
    """Without this the retain pool is invisible to selection, whatever the rule says."""
    trainer = _trainer()
    dataset = {
        "rows": [
            {
                "split": "development",
                "protected_question": "Where was Ada Vane born?",
                "aliases": ["Ada Vane"],
                "candidates": ["She was born in Rome."],
                "label": "NONE",
                "subject": "ada",
                "false_alarm_pool": pool,
            }
            for pool in ("generated_clean", "retain_answer")
        ]
    }
    rows = trainer.synthetic_examples(dataset, "development")
    assert {r["population"] for r in rows} == {"protected", "retain"}


# =====================================================================================
# A new bank cannot be gated using the old v4 audit
# =====================================================================================


def _bank(tmp_path: Path, *, content_sha: str, pairs: list[str]) -> Path:
    """A schema-v3 bank: retain rows live INSIDE each partition, never in a third block."""
    payload = {
        "schema": "graph-detector-v4-2-engineering-bank-v3",
        "bank_id": "detector_v4_2_engineering_v1",
        "content_sha256": content_sha,
        "partitions": {
            "development": {"clean": [], "leaking": [], "retain": []},
            "heldout": {
                "clean": [
                    {
                        "text": f"candidate {p}",
                        "request": "Where was Ada Vane born?",
                        "item_id": "forget10-0002",
                        "population": "protected",
                        "text_sha256": p,
                        "pair_sha256": p,
                    }
                    for p in pairs
                ],
                "leaking": [],
                "retain": [],
            },
        },
    }
    path = tmp_path / "ENGINEERING_BANK.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _labels(tmp_path: Path, rows: list[dict], name: str = "labels.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    return path


def _model_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "DETECTOR_V4_MODEL.json"
    path.write_text(json.dumps({"selected_checkpoint": str(tmp_path / "ckpt")}), encoding="utf-8")
    return path


def _audit_manifest(tmp_path: Path, *, content_sha: str, reportable: bool = True) -> Path:
    path = tmp_path / "BANK_AUDIT_MANIFEST.json"
    path.write_text(
        json.dumps(
            {
                "schema": "graph-detector-v4-2-bank-audit-manifest-v2",
                "bank_content_sha256": content_sha,
                "reportable": reportable,
                "uses_detector_score": False,
                "shortfalls": {},
                "below_minima": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _alignment_report(
    tmp_path: Path,
    *,
    labels: Path,
    content_sha: str,
    passed: bool = True,
    unresolved: float = 0.0,
    provenance_failures: tuple[str, ...] = (),
) -> Path:
    import hashlib

    path = tmp_path / "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json"
    path.write_text(
        json.dumps(
            {
                "all_gates_passed": passed,
                "failed_gates": [] if passed else ["answer_attempt_kappa"],
                "gates": [
                    {"gate": "n_unresolved_disagreements", "measured": unresolved},
                ],
                "provenance": {"failures": list(provenance_failures)},
                "human_grounded": False,
                "publication_label_valid": False,
                "adjudicated_file": str(labels),
                "adjudicated_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
                "bank_content_sha256": content_sha,
                "judge_population": "two_independent_llm_judges",
            }
        ),
        encoding="utf-8",
    )
    return path


def _gate(tmp_path: Path, **overrides):
    """Call `final-gate` with every argument named. Typer defaults are OptionInfo objects,
    not values, so a direct call that omits one gets a sentinel rather than a default."""
    from rdl.cli.detector_v4_2_banks import detector_v4_2_final_gate

    kwargs = {
        "bank": "engineering",
        "bank_dir": tmp_path / "bank",
        "labels": tmp_path / "labels.jsonl",
        "model_artifact": _model_artifact(tmp_path),
        "operating_point": tmp_path / "DETECTOR_V4_2_OPERATING_POINT.json",
        "threshold": None,
        "audit_manifest": tmp_path / "BANK_AUDIT_MANIFEST.json",
        "alignment_report": tmp_path / "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json",
        "policy_cohort": REPO / "data" / "cohorts" / "graph_unlearning_v1" / "discovery.json",
        "backend": "lexical",
        "device": "",
        "partition": "heldout",
        "output_dir": tmp_path / "out",
        "reopen": False,
    }
    kwargs.update(overrides)
    return detector_v4_2_final_gate(**kwargs)


def test_labels_from_another_bank_are_refused(tmp_path):
    """The v4.1 audit's audit_ids name content this bank does not contain."""
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    _bank(bank_dir, content_sha="a" * 64, pairs=["1" * 64, "2" * 64])
    stale = _labels(
        tmp_path,
        [
            {
                "audit_id": "0008c30c51476aa7",
                "answer_attempt": "ANSWER",
                "pair_sha256": "9" * 64,
                "bank_content_sha256": "a" * 64,
            }
        ],
    )
    _audit_manifest(tmp_path, content_sha="a" * 64)
    _alignment_report(tmp_path, labels=stale, content_sha="a" * 64)

    with pytest.raises(typer.BadParameter) as excinfo:
        _gate(tmp_path, labels=stale)
    message = str(excinfo.value)
    assert "name content this bank does not contain" in message
    assert "Gating a new bank on an old bank's labels" in message


def test_labels_produced_against_a_different_bank_revision_are_refused(tmp_path):
    """Right rows, wrong bank hash: the bank moved after the audit was drawn."""
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    _bank(bank_dir, content_sha="a" * 64, pairs=["1" * 64])
    moved = _labels(
        tmp_path,
        [
            {
                "audit_id": "0" * 16,
                "answer_attempt": "ANSWER",
                "pair_sha256": "1" * 64,
                "bank_content_sha256": "b" * 64,
            }
        ],
    )
    _audit_manifest(tmp_path, content_sha="a" * 64)
    _alignment_report(tmp_path, labels=moved, content_sha="a" * 64)

    with pytest.raises(typer.BadParameter, match="bank moved after the audit"):
        _gate(tmp_path, labels=moved)


def test_the_gate_refuses_to_run_with_no_labels_at_all(tmp_path):
    """The original defect: the command opened the bank without reading any label."""
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    _bank(bank_dir, content_sha="a" * 64, pairs=["1" * 64])

    with pytest.raises(typer.BadParameter) as excinfo:
        _gate(tmp_path, labels=tmp_path / "does-not-exist.jsonl")
    assert "adjudicated model-judge labels for THIS bank" in str(excinfo.value)


# =====================================================================================
# Labels that never passed the judge gate cannot open a bank
# =====================================================================================


def _prepared(tmp_path) -> tuple[Path, Path]:
    """A bank and a label file that match, so each test can break exactly one thing."""
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    _bank(bank_dir, content_sha="a" * 64, pairs=["1" * 64])
    labels = _labels(
        tmp_path,
        [
            {
                "audit_id": "0" * 16,
                "answer_attempt": "ANSWER",
                "pair_sha256": "1" * 64,
                "bank_content_sha256": "a" * 64,
            }
        ],
    )
    return bank_dir, labels


def test_an_absent_audit_manifest_refuses_the_gate(tmp_path):
    """Without it, nothing shows the labelled rows were chosen before the detector ran."""
    _, labels = _prepared(tmp_path)
    _alignment_report(tmp_path, labels=labels, content_sha="a" * 64)
    with pytest.raises(typer.BadParameter, match="frozen stratified audit sample"):
        _gate(tmp_path, labels=labels, audit_manifest=tmp_path / "absent.json")


def test_an_audit_that_missed_its_minima_refuses_the_gate(tmp_path):
    _, labels = _prepared(tmp_path)
    _audit_manifest(tmp_path, content_sha="a" * 64, reportable=False)
    _alignment_report(tmp_path, labels=labels, content_sha="a" * 64)
    with pytest.raises(typer.BadParameter, match="condition"):
        _gate(tmp_path, labels=labels)


def test_a_failing_alignment_report_refuses_the_gate(tmp_path):
    """THE defect: --labels was accepted with no evidence the audit ever passed."""
    _, labels = _prepared(tmp_path)
    _audit_manifest(tmp_path, content_sha="a" * 64)
    _alignment_report(tmp_path, labels=labels, content_sha="a" * 64, passed=False)
    with pytest.raises(typer.BadParameter, match="condition"):
        _gate(tmp_path, labels=labels)


def test_unresolved_disagreements_refuse_the_gate(tmp_path):
    _, labels = _prepared(tmp_path)
    _audit_manifest(tmp_path, content_sha="a" * 64)
    _alignment_report(tmp_path, labels=labels, content_sha="a" * 64, unresolved=4.0)
    with pytest.raises(typer.BadParameter, match="condition"):
        _gate(tmp_path, labels=labels)


def test_a_label_file_edited_after_the_report_refuses_the_gate(tmp_path):
    """The report vouches for a hash. Editing the file afterwards breaks the binding."""
    _, labels = _prepared(tmp_path)
    _audit_manifest(tmp_path, content_sha="a" * 64)
    _alignment_report(tmp_path, labels=labels, content_sha="a" * 64)
    labels.write_text(
        json.dumps(
            {
                "audit_id": "0" * 16,
                "answer_attempt": "NONE",
                "pair_sha256": "1" * 64,
                "bank_content_sha256": "a" * 64,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(typer.BadParameter, match="condition"):
        _gate(tmp_path, labels=labels)


# =====================================================================================
# One row, one label
# =====================================================================================


def test_a_duplicate_audit_id_is_refused_rather_than_overwritten(tmp_path):
    """The label map was a dict comprehension: the second row silently won."""
    from rdl.cli.detector_v4_2_gate_bridge import load_label_map

    path = _labels(
        tmp_path,
        [
            {"audit_id": "0" * 16, "answer_attempt": "ANSWER", "pair_sha256": "1" * 64},
            {"audit_id": "0" * 16, "answer_attempt": "NONE", "pair_sha256": "1" * 64},
        ],
    )
    with pytest.raises(typer.BadParameter, match="duplicate audit_id"):
        load_label_map(path)


def test_two_audit_ids_naming_one_bank_row_are_refused(tmp_path):
    """One bank row labelled twice makes the gate count it twice."""
    from rdl.cli.detector_v4_2_gate_bridge import load_label_map

    path = _labels(
        tmp_path,
        [
            {"audit_id": "0" * 16, "answer_attempt": "ANSWER", "pair_sha256": "1" * 64},
            {"audit_id": "1" * 16, "answer_attempt": "NONE", "pair_sha256": "1" * 64},
        ],
    )
    with pytest.raises(typer.BadParameter, match="more than one audit_id"):
        load_label_map(path)


def test_a_label_with_no_pair_digest_is_refused(tmp_path):
    """text_sha256 alone cannot address a bank keyed by (text, question)."""
    from rdl.cli.detector_v4_2_gate_bridge import load_label_map

    path = _labels(
        tmp_path,
        [{"audit_id": "0" * 16, "answer_attempt": "ANSWER", "text_sha256": "1" * 64}],
    )
    with pytest.raises(typer.BadParameter, match="no pair_sha256"):
        load_label_map(path)


# =====================================================================================
# The threshold is read from a frozen artifact, never from the command line
# =====================================================================================


def _operating_point(tmp_path: Path, **overrides) -> Path:
    from rdl.cli.detector_v4_2_gate_bridge import OPERATING_POINT_SCHEMA

    payload = {
        "schema": OPERATING_POINT_SCHEMA,
        "bank_content_sha256": "a" * 64,
        "model_artifact": str(tmp_path / "DETECTOR_V4_MODEL.json"),
        "selected_threshold": 0.62,
        "partition_selected_on": "development",
        "utc": "2026-08-15T00:00:00Z",
    }
    payload.update(overrides)
    path = tmp_path / "DETECTOR_V4_2_OPERATING_POINT.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_an_absent_operating_point_refuses_the_gate(tmp_path):
    from rdl.cli.detector_v4_2_gate_bridge import load_operating_point

    with pytest.raises(typer.BadParameter, match="threshold is not an argument"):
        load_operating_point(
            tmp_path / "absent.json",
            bank_content_sha256="a" * 64,
            model_artifact=tmp_path / "DETECTOR_V4_MODEL.json",
        )


def test_a_threshold_on_the_command_line_is_refused_not_applied(tmp_path):
    """THE defect: --threshold was a required float and whatever was typed was used."""
    from rdl.cli.detector_v4_2_gate_bridge import load_operating_point

    path = _operating_point(tmp_path)
    with pytest.raises(typer.BadParameter, match="does not authorise"):
        load_operating_point(
            path,
            bank_content_sha256="a" * 64,
            model_artifact=tmp_path / "DETECTOR_V4_MODEL.json",
            requested_threshold=0.30,
        )
    # The frozen value itself is accepted: passing it is a no-op, not an override.
    point = load_operating_point(
        path,
        bank_content_sha256="a" * 64,
        model_artifact=tmp_path / "DETECTOR_V4_MODEL.json",
        requested_threshold=0.62,
    )
    assert point["selected_threshold"] == 0.62


def test_an_operating_point_from_another_bank_or_model_is_refused(tmp_path):
    from rdl.cli.detector_v4_2_gate_bridge import load_operating_point

    path = _operating_point(tmp_path)
    with pytest.raises(typer.BadParameter, match="does not authorise"):
        load_operating_point(
            path,
            bank_content_sha256="b" * 64,
            model_artifact=tmp_path / "DETECTOR_V4_MODEL.json",
        )
    with pytest.raises(typer.BadParameter, match="does not authorise"):
        load_operating_point(
            path,
            bank_content_sha256="a" * 64,
            model_artifact=tmp_path / "other" / "DETECTOR_V4_MODEL.json",
        )


def test_an_operating_point_frozen_on_other_weights_is_refused(tmp_path):
    """The artifact path is not the checkpoint: a re-trained epoch reuses the path."""
    from rdl.cli.detector_v4_2_gate_bridge import load_operating_point

    path = _operating_point(
        tmp_path,
        selected_checkpoint="runs/detector_v4/seed20260814-checkpoint-epoch2",
        selected_checkpoint_hashes={"files_sha256": {"model.safetensors": "a" * 64}},
    )
    with pytest.raises(typer.BadParameter, match="does not authorise"):
        load_operating_point(
            path,
            bank_content_sha256="a" * 64,
            model_artifact=tmp_path / "DETECTOR_V4_MODEL.json",
            model_manifest={
                "selected_checkpoint": "runs/detector_v4/seed20260814-checkpoint-epoch2",
                "selected_checkpoint_hashes": {"files_sha256": {"model.safetensors": "b" * 64}},
            },
        )


def test_an_operating_point_that_selected_nothing_cannot_open_a_gate(tmp_path):
    from rdl.cli.detector_v4_2_gate_bridge import load_operating_point

    path = _operating_point(tmp_path, selected_threshold=None)
    with pytest.raises(typer.BadParameter, match="does not authorise"):
        load_operating_point(
            path,
            bank_content_sha256="a" * 64,
            model_artifact=tmp_path / "DETECTOR_V4_MODEL.json",
        )


# =====================================================================================
# Retain rows are held out, not reused
# =====================================================================================


def test_a_bank_whose_retain_pool_was_never_split_is_refused(tmp_path):
    """Schema v2's `retain.all` selected the threshold AND reported the retain FPR."""
    from rdl.cli.detector_v4_2_banks import _load_bank_rows

    payload = {
        "partitions": {
            "development": {"clean": [], "leaking": []},
            "heldout": {"clean": [], "leaking": []},
            "retain": {"all": [{"text": "t", "request": "q", "pair_sha256": "1" * 64}]},
        }
    }
    with pytest.raises(typer.BadParameter, match="never held out"):
        _load_bank_rows(payload)


def test_retain_rows_carry_a_partition_and_their_population(tmp_path):
    from rdl.cli.detector_v4_2_banks import _load_bank_rows

    def row(i: int) -> dict:
        return {"text": f"t{i}", "request": "q", "pair_sha256": f"{i:064x}"}

    rows = _load_bank_rows(
        {
            "partitions": {
                "development": {"clean": [row(1)], "leaking": [], "retain": [row(2)]},
                "heldout": {"clean": [], "leaking": [row(3)], "retain": [row(4)]},
            }
        }
    )
    by_pair = {r["pair_sha256"]: r for r in rows}
    assert by_pair[f"{2:064x}"]["partition"] == "development"
    assert by_pair[f"{2:064x}"]["population"] == "retain"
    assert by_pair[f"{4:064x}"]["partition"] == "heldout"
    assert by_pair[f"{4:064x}"]["population"] == "retain"
    # The protected axis is unchanged, and orthogonal to it.
    assert by_pair[f"{3:064x}"]["nli_leaking"] is True
    assert by_pair[f"{3:064x}"]["population"] == "protected"


def test_the_selection_partition_and_the_gate_partition_share_no_row():
    """The property the split exists for, stated over the two readers that use it."""
    from rdl.cli.detector_v4_2_banks import _load_bank_rows
    from rdl.cli.detector_v4_2_gate_bridge import labelled_rows_for_partition

    rows = _load_bank_rows(
        {
            "partitions": {
                "development": {
                    "clean": [],
                    "leaking": [],
                    "retain": [{"text": "d", "request": "q", "pair_sha256": "1" * 64}],
                },
                "heldout": {
                    "clean": [],
                    "leaking": [],
                    "retain": [{"text": "h", "request": "q", "pair_sha256": "2" * 64}],
                },
            }
        }
    )
    by_pair = {str(r["pair_sha256"]): r for r in rows}
    labels = {
        "aaaa": {"pair_sha256": "1" * 64, "answer_attempt": None},
        "bbbb": {"pair_sha256": "2" * 64, "answer_attempt": None},
    }
    development = labelled_rows_for_partition(
        labels, by_pair, partition="development", concept_of={}
    )
    heldout = labelled_rows_for_partition(labels, by_pair, partition="heldout", concept_of={})
    assert [r["text"] for r in development] == ["d"]
    assert [r["text"] for r in heldout] == ["h"]
    assert not {r["audit_id"] for r in development} & {r["audit_id"] for r in heldout}


def test_the_final_bank_is_still_sealed_against_building():
    from rdl.cli.detector_v4_2_banks import detector_v4_2_build_bank

    with pytest.raises(typer.BadParameter, match="SEALED"):
        detector_v4_2_build_bank(
            bank="final",
            natural_run=[],
            retain_run=[],
            manifest_dir=Path("."),
            v4_1_dir=Path("."),
            output_dir=Path("."),
            check_only=True,
        )
