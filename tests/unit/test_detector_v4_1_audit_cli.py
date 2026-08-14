"""The blinded audit's construction, and the trainer's readiness gate.

Two things are checked here that no amount of reading the artifacts would establish:

* the annotation set a judge receives cannot carry the labels the audit exists to test —
  blinding that is a convention rather than a property is not blinding;
* ``scripts/train_detector_v4.py`` refuses to start without the audit, and has no flag that
  skips the refusal. v4's ``--force-despite-failed-ceiling`` was a way to keep a mistake,
  and a deleted flag is the only kind that cannot be typed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from rdl.cli.detector_v4_label_audit import (
    ADJUDICATED_FILENAME,
    BLIND_FIELDS,
    KEY_FILENAME,
    REPORT_FILENAME,
    _load_judge,
    select_strata,
)
from rdl.eval.detector_v4_1 import AUDIT_FIELDS, STRATA

REPO = Path(__file__).resolve().parents[2]
V4_1 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_1"


# =====================================================================================
# Sampling
# =====================================================================================


def _pool(prefix: str, n: int, *, request: str = "Where was Ada Vane born?", score=0.0):
    return [
        {
            "audit_id": f"{prefix}{i:04d}",
            "text": "x" * (10 + i),
            "request": request,
            "lexical_score": score + i / 1000.0,
        }
        for i in range(n)
    ]


def test_a_row_belongs_to_exactly_one_stratum():
    """Overlapping strata would make the per-stratum counts add to more than the set, and
    a row annotated twice would be counted twice in every rate."""
    strata, _prov = select_strata(_pool("L", 120), _pool("C", 2000), _pool("R", 500))
    seen: list[str] = []
    for name in STRATA:
        seen.extend(r["audit_id"] for r in strata[name])
    assert len(seen) == len(set(seen))
    assert all(r["stratum"] == name for name in STRATA for r in strata[name])


def test_the_hard_negative_stratum_takes_the_highest_scoring_clean_rows():
    """It is a SAMPLING device: it decides which rows a human reads and enters no rate."""
    clean = _pool("C", 500)
    strata, _prov = select_strata(_pool("L", 120), clean, _pool("R", 500))
    chosen = {r["audit_id"] for r in strata["clean_hard_negative"]}
    top = {r["audit_id"] for r in sorted(clean, key=lambda r: -r["lexical_score"])[:200]}
    assert chosen == top


def test_the_selection_is_reproducible_from_the_bank_alone():
    """No RNG and no seed: a re-run has to produce the same annotation set, or the audit
    is not something a second person can check."""
    args = (_pool("L", 120), _pool("C", 900), _pool("R", 400))
    first, _a = select_strata(*args)
    second, _b = select_strata(*args)
    assert {k: [r["audit_id"] for r in v] for k, v in first.items()} == {
        k: [r["audit_id"] for r in v] for k, v in second.items()
    }


def test_a_shortfall_is_recorded_rather_than_silently_absorbed():
    strata, provenance = select_strata(_pool("L", 3), _pool("C", 900), _pool("R", 400))
    assert len(strata["natural_leaking"]) == 3
    assert provenance["shortfall"]["natural_leaking"] == 117
    assert provenance["pool_sizes"]["leaking"] == 3


def test_the_matched_stratum_prefers_the_leaking_rows_questions():
    leaking = _pool("L", 20, request="Where was Ada Vane born?")
    clean = [
        *_pool("CM", 300, request="Where was Ada Vane born?"),
        *_pool("CO", 300, request="Who publishes Cyril Moss?"),
    ]
    strata, _prov = select_strata(leaking, clean, _pool("R", 400))
    matched = strata["clean_matched"]
    assert matched
    assert all(r["request"] == "Where was Ada Vane born?" for r in matched)


# =====================================================================================
# Blinding
# =====================================================================================


@pytest.mark.skipif(not (V4_1 / "LABEL_AUDIT_JUDGE_A.jsonl").exists(), reason="audit not built")
def test_the_judge_files_carry_a_question_a_message_and_nothing_else():
    """THE blinding property. Every withheld field would anchor a judgement: the NLI label
    is what the audit exists to check, the stratum and the score are the sampling, and the
    concept id is the thing a detector has to infer.
    """
    allowed = {"audit_id", "protected_question", "candidate_text", *BLIND_FIELDS}
    forbidden = {
        "nli_leaking",
        "stratum",
        "lexical_score",
        "item_id",
        "concept_id",
        "bank_partition",
        "reference_answer",
        "leaking",
        "split",
    }
    for judge in ("A", "B"):
        path = V4_1 / f"LABEL_AUDIT_JUDGE_{judge}.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        assert rows
        for row in rows:
            assert set(row) == allowed, set(row) - allowed
            assert not (set(row) & forbidden)
            assert all(row[field] is None for field in BLIND_FIELDS)


@pytest.mark.skipif(not (V4_1 / "LABEL_AUDIT_JUDGE_A.jsonl").exists(), reason="audit not built")
def test_the_two_judges_do_not_see_the_same_order():
    """A shared order lets a shared drift in attention look like agreement."""

    def ids(judge: str) -> list[str]:
        path = V4_1 / f"LABEL_AUDIT_JUDGE_{judge}.jsonl"
        return [
            json.loads(line)["audit_id"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    a, b = ids("A"), ids("B")
    assert set(a) == set(b)
    assert a != b


@pytest.mark.skipif(not (V4_1 / KEY_FILENAME).exists(), reason="audit not built")
def test_the_unblinding_key_is_a_separate_file_that_says_so():
    key = json.loads((V4_1 / KEY_FILENAME).read_text(encoding="utf-8"))
    assert key["judges_must_not_read"] is True
    row = next(iter(key["rows"].values()))
    assert {"stratum", "nli_leaking", "concept_id", "population"} <= set(row)


@pytest.mark.skipif(not (V4_1 / "LABEL_AUDIT_MANIFEST.json").exists(), reason="audit not built")
def test_the_manifest_does_not_claim_to_have_modified_the_frozen_bank():
    manifest = json.loads((V4_1 / "LABEL_AUDIT_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["source_bank_modified"] is False
    assert manifest["scope"]["gpu_used"] is False
    assert manifest["runtime_reads_gold_answers"] is False
    assert manifest["registry"]["role"].startswith("sampling only")


@pytest.mark.skipif(not (V4_1 / "LABEL_AUDIT_JUDGE_A.jsonl").exists(), reason="audit not built")
def test_the_reference_pass_is_a_different_file_from_the_blind_pass():
    """A judge who has seen the reference answer can no longer report whether the text
    ATTEMPTS an answer independently of whether it got the answer right."""
    blind = json.loads(
        (V4_1 / "LABEL_AUDIT_JUDGE_A.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert "reference_answer" not in blind
    reference = V4_1 / "LABEL_AUDIT_REFERENCE_PASS_A.jsonl"
    if reference.exists():
        row = json.loads(reference.read_text(encoding="utf-8").splitlines()[0])
        assert "reference_answer" in row
        assert "answer_attempt" not in row
        assert AUDIT_FIELDS["reference_content"]["reference_answer_visible"] is True


# =====================================================================================
# The report command
# =====================================================================================


def _write_jsonl(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def _labels(answer: str) -> dict:
    return {
        "answer_attempt": answer,
        "reference_content": "NO",
        "subject_only": "no",
        "refusal": "no",
        "question_type": "slot",
    }


def _fixture_audit(tmp_path: Path, n: int = 400) -> Path:
    key = {"schema": "x", "judges_must_not_read": True, "rows": {}}
    a, b = [], []
    for i in range(n):
        audit_id = f"{i:016x}"
        answer = "ANSWER" if i < 150 else ("PARTIAL" if i < 155 else "NONE")
        key["rows"][audit_id] = {
            "text_sha256": f"sha{i}",
            "stratum": "clean_hard_negative" if i % 2 else "clean_matched",
            "population": "protected",
            "nli_leaking": False,
            "concept_id": f"author-{i % 6:04d}",
        }
        a.append({"audit_id": audit_id, **_labels(answer)})
        b.append({"audit_id": audit_id, **_labels(answer)})
    (tmp_path / KEY_FILENAME).write_text(json.dumps(key), encoding="utf-8")
    _write_jsonl(tmp_path / "a.jsonl", a)
    _write_jsonl(tmp_path / "b.jsonl", b)
    return tmp_path


def _run_report(tmp_path: Path, extra: list[str] | None = None):
    from rdl.cli.__main__ import app

    return CliRunner().invoke(
        app,
        [
            "graph-detector-v4-label-report",
            "--judge-a",
            str(tmp_path / "a.jsonl"),
            "--judge-b",
            str(tmp_path / "b.jsonl"),
            "--output-dir",
            str(tmp_path),
            *(extra or []),
        ],
    )


def test_the_report_writes_the_overlay_and_passes_a_clean_audit(tmp_path):
    result = _run_report(_fixture_audit(tmp_path))
    assert result.exit_code == 0, result.output
    report = json.loads((tmp_path / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert report["all_gates_passed"]
    assert report["label_disagreement"]["share_of_nli_clean_that_is_an_answer_attempt"] > 0
    overlay = (tmp_path / ADJUDICATED_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(overlay) == 400
    # The overlay is keyed back to the bank by text hash, and carries no bank row.
    first = json.loads(overlay[0])
    assert first["text_sha256"].startswith("sha")
    assert "candidate_text" not in first


def test_the_report_exits_non_zero_when_the_audit_fails_its_gate(tmp_path):
    tmp = _fixture_audit(tmp_path, n=40)  # far too few ANSWER and NONE rows
    result = _run_report(tmp)
    assert result.exit_code == 1
    report = json.loads((tmp_path / REPORT_FILENAME).read_text(encoding="utf-8"))
    assert not report["all_gates_passed"]
    assert "Do not train" in report["verdict"]


def test_one_judge_is_not_an_audit(tmp_path):
    from rdl.cli.__main__ import app

    tmp = _fixture_audit(tmp_path)
    result = CliRunner().invoke(
        app,
        [
            "graph-detector-v4-label-report",
            "--judge-a",
            str(tmp / "a.jsonl"),
            "--output-dir",
            str(tmp),
        ],
    )
    assert result.exit_code != 0


def test_a_judge_value_outside_the_vocabulary_is_refused(tmp_path):
    tmp = _fixture_audit(tmp_path)
    rows = [json.loads(line) for line in (tmp / "a.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["answer_attempt"] = "MAYBE"
    _write_jsonl(tmp / "a.jsonl", rows)
    with pytest.raises(Exception, match="MAYBE"):
        _load_judge([tmp / "a.jsonl"])


# =====================================================================================
# The trainer's readiness gate
# =====================================================================================


def _trainer_body() -> str:
    """The script's CODE, with its module docstring removed.

    The docstring quotes both the deleted flag and the old ``truncation="only_second"``
    call, because explaining what was wrong is the point of it. Searching the raw file
    would therefore find the thing the test is asserting is gone.
    """
    source = (REPO / "scripts" / "train_detector_v4.py").read_text(encoding="utf-8")
    return source.split('"""', 2)[2]


def _trainer():
    path = REPO / "scripts" / "train_detector_v4.py"
    spec = importlib.util.spec_from_file_location("train_detector_v4_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_force_flag_is_deleted_not_defaulted_off():
    """v4's ``--force-despite-failed-ceiling`` trained despite a failing artifact. The
    correct response to that artifact was to fix its interpretation; a flag that skips the
    fix is a way to keep the mistake, and a deleted flag cannot be typed."""
    body = _trainer_body()
    # Checked against the FLAGS the parser accepts, not the file text: the prose in this
    # script names the deleted flag on purpose, because explaining what was wrong is what
    # keeps the next person from re-adding it.
    flags = [line for line in body.splitlines() if "add_argument" in line]
    assert flags, "the trainer has no argument parser"
    assert not any("force" in line for line in flags), flags
    assert not any("ceiling" in line for line in flags), flags


def test_the_trainer_refuses_to_start_without_the_label_audit(tmp_path):
    module = _trainer()
    with pytest.raises(SystemExit, match=r"LABEL_ALIGNMENT_REPORT\.json is absent"):
        module.require_label_audit(tmp_path)


def test_the_trainer_refuses_an_audit_that_failed_its_gate(tmp_path):
    module = _trainer()
    (tmp_path / "LABEL_ALIGNMENT_REPORT.json").write_text(
        json.dumps(
            {
                "all_gates_passed": False,
                "failed_gates": ["answer_attempt_kappa"],
                "verdict": "the audit does NOT clear its decision gate",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="does NOT clear"):
        module.require_label_audit(tmp_path)


def test_the_trainer_accepts_a_passing_audit(tmp_path):
    """A gate that can only refuse is not a gate."""
    module = _trainer()
    (tmp_path / "LABEL_ALIGNMENT_REPORT.json").write_text(
        json.dumps({"all_gates_passed": True, "verdict": "usable"}), encoding="utf-8"
    )
    assert module.require_label_audit(tmp_path)["all_gates_passed"] is True


def test_the_trainer_encodes_through_the_runtime_budget():
    """Train-time and serve-time segmentation must be the same object, or the checkpoint
    is served under a segmentation it was never trained on."""
    body = _trainer_body()
    assert "from rdl.defenses.cross_encoder_answerability import budget_encode" in body
    assert 'truncation="only_second"' not in body


def test_the_trainer_requires_both_revisions():
    assert "--tokenizer-revision" in _trainer_body()
    module = _trainer()
    assert module.TrainingPins.tokenizer_revision == ""  # no guessed default
    assert module.TrainingPins.model_revision == ""


def test_the_trainer_has_a_loop_a_selection_and_a_saved_checkpoint():
    module = _trainer()
    for name in ("evaluate", "collate", "AnswerabilityDataset", "baseline_zero_shot", "set_seed"):
        assert hasattr(module, name), name
    body = _trainer_body()
    for fragment in (
        "torch.optim.AdamW",
        "scaler.step(optimizer)",
        "scheduler.step()",
        ".backward()",
        "save_pretrained",
        "selected_checkpoint",
        "CrossEntropyLoss",
    ):
        assert fragment in body, fragment


def test_the_baseline_is_a_pretrained_head_not_a_random_one():
    """``microsoft/deberta-v3-base`` has a randomly initialised classification head, so its
    off-the-shelf number measures that head rather than what pretraining knows."""
    module = _trainer()
    assert module.TrainingPins.model_repo_id != module.DEFAULT_BASELINE_REPO
    assert "nli" in module.DEFAULT_BASELINE_REPO


def test_a_held_out_row_cannot_reach_the_training_pool():
    module = _trainer()
    dataset = json.loads(
        (
            REPO
            / "data"
            / "cohorts"
            / "graph_unlearning_v1"
            / "detector_v4"
            / "DETECTOR_V4_DATASET.json"
        ).read_text(encoding="utf-8")
    )
    rows = module.synthetic_examples(dataset, "train")
    assert rows
    assert {r["split"] for r in rows} == {"train"}
