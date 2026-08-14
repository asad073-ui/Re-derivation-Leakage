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
    payload = {
        "bank_id": "detector_v4_2_engineering_v1",
        "content_sha256": content_sha,
        "partitions": {
            "development": {"clean": [], "leaking": []},
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
            },
            "retain": {"all": []},
        },
    }
    path = tmp_path / "ENGINEERING_BANK.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _labels(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    return path


def _model_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "DETECTOR_V4_MODEL.json"
    path.write_text(json.dumps({"selected_checkpoint": str(tmp_path / "ckpt")}), encoding="utf-8")
    return path


def test_labels_from_another_bank_are_refused(tmp_path):
    """The v4.1 audit's audit_ids name content this bank does not contain."""
    from rdl.cli.detector_v4_2_banks import detector_v4_2_final_gate

    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    _bank(bank_dir, content_sha="a" * 64, pairs=["1" * 64, "2" * 64])
    stale = _labels(
        tmp_path,
        [{"audit_id": "0008c30c51476aa7", "answer_attempt": "ANSWER", "text_sha256": "9" * 64}],
    )

    with pytest.raises(typer.BadParameter) as excinfo:
        detector_v4_2_final_gate(
            bank="engineering",
            bank_dir=bank_dir,
            labels=stale,
            model_artifact=_model_artifact(tmp_path),
            threshold=0.5,
            policy_cohort=REPO / "data" / "cohorts" / "graph_unlearning_v1" / "discovery.json",
            backend="lexical",
            device="",
            partition="heldout",
            output_dir=tmp_path / "out",
            reopen=False,
        )
    message = str(excinfo.value)
    assert "name content this bank does not contain" in message
    assert "Gating a new bank on an old bank's labels" in message


def test_labels_produced_against_a_different_bank_revision_are_refused(tmp_path):
    """Right rows, wrong bank hash: the bank moved after the audit was drawn."""
    from rdl.cli.detector_v4_2_banks import detector_v4_2_final_gate

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

    with pytest.raises(typer.BadParameter, match="bank moved after the audit"):
        detector_v4_2_final_gate(
            bank="engineering",
            bank_dir=bank_dir,
            labels=moved,
            model_artifact=_model_artifact(tmp_path),
            threshold=0.5,
            policy_cohort=REPO / "data" / "cohorts" / "graph_unlearning_v1" / "discovery.json",
            backend="lexical",
            device="",
            partition="heldout",
            output_dir=tmp_path / "out",
            reopen=False,
        )


def test_the_gate_refuses_to_run_with_no_labels_at_all(tmp_path):
    """The original defect: the command opened the bank without reading any label."""
    from rdl.cli.detector_v4_2_banks import detector_v4_2_final_gate

    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    _bank(bank_dir, content_sha="a" * 64, pairs=["1" * 64])

    with pytest.raises(typer.BadParameter) as excinfo:
        detector_v4_2_final_gate(
            bank="engineering",
            bank_dir=bank_dir,
            labels=tmp_path / "does-not-exist.jsonl",
            model_artifact=_model_artifact(tmp_path),
            threshold=0.5,
            policy_cohort=REPO / "data" / "cohorts" / "graph_unlearning_v1" / "discovery.json",
            backend="lexical",
            device="",
            partition="heldout",
            output_dir=tmp_path / "out",
            reopen=False,
        )
    assert "adjudicated model-judge labels for THIS bank" in str(excinfo.value)


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
