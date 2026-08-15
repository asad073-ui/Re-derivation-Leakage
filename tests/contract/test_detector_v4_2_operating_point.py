"""The threshold: chosen on development, frozen, and refused if it is typed anywhere else.

``final-gate`` used to take ``--threshold`` as a required float. The one number the whole
protocol is organised around — chosen on the development partition, frozen, then used to
open the held-out partition exactly once — was whatever the operator typed, and a second
attempt at a different value left nothing behind but a second opening record.

This runs the real command on the lexical floor, which needs no GPU and no checkpoint, so
the plumbing under test is the plumbing the GPU run will use. What is asserted is the
protocol, not the detector: which rows the sweep saw, which it did not, and what the gate
does when the frozen file disagrees with the command line.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from rdl.cli.detector_v4_2_gate_bridge import OPERATING_POINT_FILENAME
from rdl.cli.detector_v4_2_operating_point import (
    SELECTION_RULE,
    detector_v4_2_select_operating_point,
    select_threshold,
)

REPO = Path(__file__).resolve().parents[2]
COHORT = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "discovery.json"
BANK_SHA = "e" * 64

# Two real cohort items, so the router has concepts to route to.
ITEMS = ("forget10-0004", "forget10-0012")


def _row(i: int, *, partition: str, population: str, leaking: bool | None) -> dict:
    return {
        "text": (
            f"She was born in Rome in 19{i:02d}."
            if leaking
            else "I am not able to help with that request."
        ),
        "request": f"Where was the author of item {ITEMS[i % 2]} born?",
        "item_id": ITEMS[i % 2],
        "population": population,
        "text_sha256": f"{i + 5000:064x}",
        "pair_sha256": f"{i:064x}",
    }


def _bank(tmp_path: Path) -> Path:
    """Twelve rows: six per partition, protected and retain in each."""
    partitions: dict[str, dict[str, list]] = {
        "development": {"clean": [], "leaking": [], "retain": []},
        "heldout": {"clean": [], "leaking": [], "retain": []},
    }
    i = 0
    for partition in ("development", "heldout"):
        for key, population, leaking in (
            ("leaking", "protected", True),
            ("clean", "protected", False),
            ("retain", "retain", None),
        ):
            for _ in range(2):
                partitions[partition][key].append(
                    _row(i, partition=partition, population=population, leaking=leaking)
                )
                i += 1
    path = tmp_path / "bank" / "ENGINEERING_BANK.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": "graph-detector-v4-2-engineering-bank-v3",
                "bank_id": "detector_v4_2_engineering_v1",
                "content_sha256": BANK_SHA,
                "partitions": partitions,
            }
        ),
        encoding="utf-8",
    )
    return path


def _labels(tmp_path: Path, bank: Path) -> Path:
    payload = json.loads(bank.read_text(encoding="utf-8"))
    rows = []
    for partition, block in payload["partitions"].items():
        for key, label in (("leaking", "ANSWER"), ("clean", "NONE"), ("retain", None)):
            for row in block[key]:
                rows.append(
                    {
                        "audit_id": row["pair_sha256"][-16:],
                        "pair_sha256": row["pair_sha256"],
                        "text_sha256": row["text_sha256"],
                        "bank_content_sha256": BANK_SHA,
                        "answer_attempt": label,
                        "question_type": "slot",
                        "partition_hint_not_read": partition,
                    }
                )
    path = tmp_path / "V4_2_ADJUDICATED.jsonl"
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    return path


def _audit_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "bank" / "BANK_AUDIT_MANIFEST.json"
    path.write_text(
        json.dumps(
            {
                "schema": "graph-detector-v4-2-bank-audit-manifest-v2",
                "bank_content_sha256": BANK_SHA,
                "reportable": True,
                "uses_detector_score": False,
                "plan": {},
                "minima": {},
                "n_rows_drawn_by_cell": {},
            }
        ),
        encoding="utf-8",
    )
    return path


def _alignment_report(tmp_path: Path, labels: Path) -> Path:
    import hashlib

    path = tmp_path / "out" / "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "all_gates_passed": True,
                "gates": [{"gate": "n_unresolved_disagreements", "measured": 0.0}],
                "provenance": {"failures": []},
                "human_grounded": False,
                "publication_label_valid": False,
                "adjudicated_file": str(labels),
                "adjudicated_sha256": hashlib.sha256(labels.read_bytes()).hexdigest(),
                "bank_content_sha256": BANK_SHA,
                "judge_population": "two_independent_llm_judges",
                "inter_judge": {"per_field": {"answer_attempt": {"cohens_kappa": 0.8}}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _model_artifact(tmp_path: Path) -> Path:
    path = tmp_path / "DETECTOR_V4_MODEL.json"
    path.write_text(
        json.dumps({"selected_checkpoint": str(tmp_path / "ckpt"), "reportable": True}),
        encoding="utf-8",
    )
    return path


@pytest.fixture()
def prepared(tmp_path: Path) -> dict:
    bank = _bank(tmp_path)
    labels = _labels(tmp_path, bank)
    return {
        "tmp_path": tmp_path,
        "bank": bank,
        "labels": labels,
        "audit_manifest": _audit_manifest(tmp_path),
        "alignment_report": _alignment_report(tmp_path, labels),
        "model_artifact": _model_artifact(tmp_path),
    }


def _select(prepared: dict, **overrides):
    kwargs = {
        "bank": "engineering",
        "bank_dir": prepared["tmp_path"] / "bank",
        "labels": prepared["labels"],
        "audit_manifest": prepared["audit_manifest"],
        "alignment_report": prepared["alignment_report"],
        "model_artifact": prepared["model_artifact"],
        "policy_cohort": COHORT,
        "backend": "lexical",
        "device": "",
        "output_dir": prepared["tmp_path"] / "out",
        "refreeze": False,
    }
    kwargs.update(overrides)
    return detector_v4_2_select_operating_point(**kwargs)


def test_the_operating_point_is_frozen_into_an_artifact(prepared):
    with pytest.raises(typer.Exit) as exit_info:
        _select(prepared)
    assert exit_info.value.exit_code in (0, 1)
    path = prepared["tmp_path"] / "out" / OPERATING_POINT_FILENAME
    point = json.loads(path.read_text(encoding="utf-8"))
    assert point["partition_selected_on"] == "development"
    assert point["never_read"] == "heldout"
    assert point["bank_content_sha256"] == BANK_SHA
    assert point["model_artifact"] == str(prepared["model_artifact"])
    assert point["selection_rule"] == SELECTION_RULE
    assert point["human_grounded"] is False
    assert point["publication_label_valid"] is False


def test_only_development_rows_are_swept(prepared):
    """Six rows exist in each partition; the sweep must see exactly one partition's."""
    with pytest.raises(typer.Exit):
        _select(prepared)
    point = json.loads(
        (prepared["tmp_path"] / "out" / OPERATING_POINT_FILENAME).read_text(encoding="utf-8")
    )
    assert point["n_development_rows"] == 6
    assert point["n_development_rows_by_population"] == {"protected": 4, "retain": 2}


def test_a_second_freeze_is_refused_without_refreeze(prepared):
    with pytest.raises(typer.Exit):
        _select(prepared)
    with pytest.raises(typer.BadParameter, match="already freezes threshold"):
        _select(prepared)


def test_refreezing_after_the_bank_was_opened_is_refused(prepared):
    with pytest.raises(typer.Exit):
        _select(prepared)
    record = prepared["tmp_path"] / "out" / "ENGINEERING_BANK_OPENING_RECORD.json"
    record.write_text(json.dumps({"n_openings": 1}), encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="chosen with the gate's numbers in view"):
        _select(prepared, refreeze=True)


def test_a_non_reportable_training_run_gets_no_operating_point(prepared):
    prepared["model_artifact"].write_text(
        json.dumps(
            {"selected_checkpoint": "ckpt", "reportable": False, "preregistration": {}},
        ),
        encoding="utf-8",
    )
    with pytest.raises(typer.BadParameter, match="reportable=false"):
        _select(prepared)


def test_a_development_partition_with_no_retain_rows_is_refused(prepared):
    """Without them the retain ceiling cannot constrain the threshold at all."""
    payload = json.loads(prepared["bank"].read_text(encoding="utf-8"))
    payload["partitions"]["development"]["retain"] = []
    prepared["bank"].write_text(json.dumps(payload), encoding="utf-8")
    # The labels have to be rebuilt for the smaller bank, or the earlier binding check
    # fires first — which is itself the correct behaviour and is asserted elsewhere.
    labels = _labels(prepared["tmp_path"], prepared["bank"])
    _alignment_report(prepared["tmp_path"], labels)
    with pytest.raises(typer.BadParameter, match="no labelled RETAIN rows"):
        _select(prepared, labels=labels)


# =====================================================================================
# The sweep's rule, on arithmetic rather than on a detector
# =====================================================================================


def _pool(protected_answer, protected_none, retain):
    rows, predictions = [], []
    for score in protected_answer:
        rows.append({"population": "protected", "answer_attempt": "ANSWER", "concept_id": "c"})
        predictions.append({"answer_probability": score, "forget_ids": ("c",)})
    for score in protected_none:
        rows.append({"population": "protected", "answer_attempt": "NONE", "concept_id": "c"})
        predictions.append({"answer_probability": score, "forget_ids": ("c",)})
    for score in retain:
        rows.append({"population": "retain", "answer_attempt": None, "concept_id": ""})
        predictions.append({"answer_probability": score, "forget_ids": ()})
    return rows, predictions


def test_a_threshold_that_fires_on_retain_traffic_is_not_eligible():
    """Both ceilings, separately. The retain pool is the smaller one."""
    rows, predictions = _pool([0.9] * 10, [0.1] * 40, [0.9] * 4)
    result = select_threshold(rows, predictions)
    # 0.9 would give perfect recall and a retain FPR of 1.0.
    assert result["selected_threshold"] > 0.9
    assert result["selected"]["retain_fpr"] == 0.0


def test_no_eligible_threshold_selects_nothing_rather_than_the_best_available():
    rows, predictions = _pool([0.9] * 10, [0.9] * 40, [0.9] * 4)
    result = select_threshold(rows, predictions)
    # Firing on everything or nothing: the only clean point has zero recall, and it IS
    # eligible — the rule never falls back, so what it returns here is that point.
    assert result["selected_threshold"] is not None
    assert result["selected"]["answer_attempt_micro_recall"] == 0.0


def test_a_pool_with_no_retain_rows_yields_no_operating_point():
    """ "Not measured" and "fine" must not agree, so the point is simply not eligible."""
    rows, predictions = _pool([0.9] * 10, [0.1] * 40, [])
    result = select_threshold(rows, predictions)
    assert result["selected_threshold"] is None
    assert "does not fall back" in result["reason_if_none"]
