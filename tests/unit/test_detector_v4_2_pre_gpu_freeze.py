"""v4.2.3 — what must be true before an RTX is rented, and what each of these cost.

Every test names something that was checkable and unchecked. None of them would have
raised an error on the box; each would have produced a number, a checkpoint, or a bill.

* A reference pass could run while the blind labels were still revisable.
* Training could read a label file no report vouched for, or the wrong authority's file.
* The held-out gate could report six rates over forty rows.
* A "pinned" revision was any string; the source revision was not recorded at all.
* The engineering bank's eight pre-registered seeds could not be generated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import typer

REPO = Path(__file__).resolve().parents[2]


def _trainer():
    path = REPO / "scripts" / "train_detector_v4.py"
    spec = importlib.util.spec_from_file_location("train_detector_v4_pre_gpu_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =====================================================================================
# The blind labels are frozen before the reference pass, not merely "complete"
# =====================================================================================


def _blind_run(path: Path, *, judge: str, overlay_sha: str) -> None:
    from rdl.eval.detector_v4_2 import JUDGES, PROMPT_VERSION

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "judge": judge,
                "pass": "blind",
                "complete": True,
                "reportable": True,
                "prompt_version": PROMPT_VERSION,
                "requested_model": JUDGES[judge]["requested_model"],
                "output_file_sha256": overlay_sha,
            }
        ),
        encoding="utf-8",
    )


def _blind_overlay(path: Path) -> str:
    from rdl.cli.detector_v4_2_llm_judge import _rows_sha256, _write_jsonl

    rows = [
        {"audit_id": f"{i:016x}", "answer_attempt": "ANSWER", "source": "model"} for i in range(4)
    ]
    _write_jsonl(path, rows)
    return _rows_sha256(rows)


def _blind_state(tmp_path: Path) -> dict[str, str]:
    from rdl.cli.detector_v4_2_llm_judge import OUTPUT_FILENAME, RUN_FILENAME
    from rdl.eval.detector_v4_2 import JUDGES

    hashes = {}
    for role in JUDGES:
        overlay = tmp_path / OUTPUT_FILENAME.format(judge=role, pass_upper="BLIND")
        hashes[role] = _blind_overlay(overlay)
        _blind_run(
            tmp_path / RUN_FILENAME.format(judge=role, pass_upper="BLIND"),
            judge=role,
            overlay_sha=hashes[role],
        )
    return hashes


def _freeze(tmp_path: Path, hashes: dict[str, str], **overrides) -> Path:
    from rdl.cli.detector_v4_2_report import BLIND_FREEZE_FILENAME, BLIND_FREEZE_SCHEMA
    from rdl.eval.detector_v4_2 import PROMPT_VERSION

    payload = {
        "schema": BLIND_FREEZE_SCHEMA,
        "prompt_version": PROMPT_VERSION,
        "blind_overlays": dict(hashes),
        "n_unresolved": 0,
    }
    payload.update(overrides)
    path = tmp_path / BLIND_FREEZE_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_two_complete_blind_runs_are_not_a_freeze(tmp_path):
    """THE defect: "both API runs finished" was accepted as "the labels are settled".

    Two judges can disagree on 200 rows and both be complete. Those rows can still be
    resolved — by someone who has by then read the reference pass's output.
    """
    from rdl.cli.detector_v4_2_llm_judge import _require_frozen_blind_passes

    _blind_state(tmp_path)
    with pytest.raises(typer.BadParameter, match="have not been ADJUDICATED"):
        _require_frozen_blind_passes(tmp_path)


def test_a_freeze_lets_the_reference_pass_run(tmp_path):
    from rdl.cli.detector_v4_2_llm_judge import _require_frozen_blind_passes

    _freeze(tmp_path, _blind_state(tmp_path))
    _require_frozen_blind_passes(tmp_path)  # does not raise


def test_a_freeze_with_unresolved_rows_is_not_a_freeze(tmp_path):
    from rdl.cli.detector_v4_2_llm_judge import _require_frozen_blind_passes

    _freeze(tmp_path, _blind_state(tmp_path), n_unresolved=7)
    with pytest.raises(typer.BadParameter, match="unresolved blind disagreement"):
        _require_frozen_blind_passes(tmp_path)


def test_re_running_a_blind_pass_after_the_freeze_is_caught(tmp_path):
    """The freeze is hash-bound, so "frozen" survives a later blind re-run."""
    from rdl.cli.detector_v4_2_llm_judge import OUTPUT_FILENAME, _require_frozen_blind_passes

    hashes = _blind_state(tmp_path)
    _freeze(tmp_path, hashes)
    # Judge A is re-run and produces different labels.
    from rdl.cli.detector_v4_2_llm_judge import _write_jsonl

    _write_jsonl(
        tmp_path / OUTPUT_FILENAME.format(judge="A", pass_upper="BLIND"),
        [{"audit_id": f"{i:016x}", "answer_attempt": "NONE", "source": "model"} for i in range(4)],
    )
    with pytest.raises(typer.BadParameter, match="re-run after the labels were frozen"):
        _require_frozen_blind_passes(tmp_path)


def test_a_freeze_from_another_prompt_version_is_refused(tmp_path):
    from rdl.cli.detector_v4_2_llm_judge import _require_frozen_blind_passes

    _freeze(tmp_path, _blind_state(tmp_path), prompt_version="v4.2-prompt-1")
    with pytest.raises(typer.BadParameter, match="froze prompt version"):
        _require_frozen_blind_passes(tmp_path)


# =====================================================================================
# Training reads the labels its authority vouches for, verified by hash
# =====================================================================================


def _labels(path: Path, rows: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps({"audit_id": f"{i:016x}", "answer_attempt": "ANSWER"}) for i in range(rows)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _model_report(trainer, v4_2_dir: Path, labels: Path, *, sha: str | None = "auto") -> Path:
    path = v4_2_dir / trainer.MODEL_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "all_gates_passed": True,
        "human_grounded": False,
        "publication_label_valid": False,
        "judge_population": "two_independent_llm_judges",
        "provenance": {"failures": []},
        "judge_independence": {"ok": True},
        "inter_judge": {"per_field": {"answer_attempt": {"cohens_kappa": 0.8}}},
        "adjudicated_file": str(labels),
    }
    if sha == "auto":
        payload["adjudicated_sha256"] = trainer._sha256_path(labels)
    elif sha is not None:
        payload["adjudicated_sha256"] = sha
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_matching_label_file_is_accepted(tmp_path):
    trainer = _trainer()
    v4_1, v4_2 = tmp_path / "v4_1", tmp_path / "v4_2"
    labels = _labels(v4_2 / "V4_2_ADJUDICATED.jsonl")
    _model_report(trainer, v4_2, labels)
    authority = trainer.require_label_audit(v4_1, v4_2)
    assert authority["adjudicated_sha256_verified"] is True
    assert authority["adjudicated_file"] == str(labels)


def test_editing_the_labels_after_the_report_blocks_training(tmp_path):
    """The report recorded a hash and nothing ever compared it to a file."""
    trainer = _trainer()
    v4_1, v4_2 = tmp_path / "v4_1", tmp_path / "v4_2"
    labels = _labels(v4_2 / "V4_2_ADJUDICATED.jsonl")
    _model_report(trainer, v4_2, labels)
    _labels(labels, rows=9)  # someone re-adjudicates by hand
    with pytest.raises(SystemExit, match="changed after the audit passed"):
        trainer.require_label_audit(v4_1, v4_2)


def test_a_report_with_no_hash_cannot_vouch_for_labels(tmp_path):
    trainer = _trainer()
    v4_1, v4_2 = tmp_path / "v4_1", tmp_path / "v4_2"
    labels = _labels(v4_2 / "V4_2_ADJUDICATED.jsonl")
    _model_report(trainer, v4_2, labels, sha=None)
    with pytest.raises(SystemExit, match="records no adjudicated_sha256"):
        trainer.require_label_audit(v4_1, v4_2)


def test_the_model_authority_names_the_model_labels_not_whichever_file_exists(tmp_path):
    """With both audits on disk, the file read must be the authority's file.

    `natural_examples` chose the first adjudicated file that existed — the v4.1 human one
    — even when the authority was the v4.2 model report. The manifest would then name the
    model report while the model had been trained on human labels.
    """
    trainer = _trainer()
    v4_1, v4_2 = tmp_path / "v4_1", tmp_path / "v4_2"
    human_labels = _labels(v4_1 / "LABEL_AUDIT_ADJUDICATED.jsonl", rows=2)
    model_labels = _labels(v4_2 / "V4_2_ADJUDICATED.jsonl", rows=5)
    _model_report(trainer, v4_2, model_labels)

    authority = trainer.require_label_audit(v4_1, v4_2)
    assert authority["adjudicated_file"] == str(model_labels)

    # And the join reads exactly that file: no blinded judge file here, so the result is
    # empty and NAMES the file it looked for rather than silently using the other one.
    _rows, meta = trainer.natural_examples(
        v4_1, v4_2, "train", adjudicated_path=Path(authority["adjudicated_file"])
    )
    assert meta["source"] == str(model_labels)
    assert str(human_labels) not in meta["source"]


# =====================================================================================
# The held-out denominators are enforced on the rows the gate actually scores
# =====================================================================================


def _gate_rows(*, n_answer: int, n_none: int, n_retain: int) -> list[dict]:
    return [
        *[{"population": "protected", "answer_attempt": "ANSWER"} for _ in range(n_answer)],
        *[{"population": "protected", "answer_attempt": "NONE"} for _ in range(n_none)],
        *[{"population": "retain", "answer_attempt": None} for _ in range(n_retain)],
    ]


def test_a_full_heldout_population_passes():
    from rdl.cli.detector_v4_2_gate_bridge import check_heldout_minima

    measured, failures = check_heldout_minima(_gate_rows(n_answer=150, n_none=400, n_retain=400))
    assert failures == []
    assert measured == {
        "n_answer_attempt": 150,
        "n_protected_nonanswer": 400,
        "n_retain": 400,
    }


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({"n_answer": 149, "n_none": 400, "n_retain": 400}, "n_answer_attempt"),
        ({"n_answer": 150, "n_none": 399, "n_retain": 400}, "n_protected_nonanswer"),
        ({"n_answer": 150, "n_none": 400, "n_retain": 399}, "n_retain"),
    ],
)
def test_one_row_short_is_a_failure(counts, expected):
    """149, 399 and 399 are the exact cases the protocol's minima name."""
    from rdl.cli.detector_v4_2_gate_bridge import check_heldout_minima

    _measured, failures = check_heldout_minima(_gate_rows(**counts))
    assert len(failures) == 1
    assert failures[0].startswith(expected)


def test_partial_rows_are_in_neither_denominator():
    """The class that silently shrinks both: PARTIAL is not ANSWER and not NONE."""
    from rdl.cli.detector_v4_2_gate_bridge import heldout_denominators

    rows = [
        *_gate_rows(n_answer=10, n_none=10, n_retain=10),
        *[{"population": "protected", "answer_attempt": "PARTIAL"} for _ in range(50)],
    ]
    measured = heldout_denominators(rows)
    assert measured["n_answer_attempt"] == 10
    assert measured["n_protected_nonanswer"] == 10


# =====================================================================================
# Exact revisions, and the source that produced them
# =====================================================================================


def _args(trainer, tmp_path, **overrides) -> argparse.Namespace:
    defaults = {
        "epochs": trainer.PREREGISTERED_EPOCHS,
        "model_repo_id": trainer.PREREGISTERED_MODEL_REPO,
        "baseline_repo_id": trainer.DEFAULT_BASELINE_REPO,
        "baseline_revision": "c" * 40,
        "skip_baseline": False,
        "extra_train": [],
        "reportable": True,
        "declare_extra_train": False,
        "v4_2_dir": tmp_path,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _pins(tmp_path: Path, trainer, **overrides) -> Path:
    payload = {
        "schema": "graph-detector-v4-2-model-pins-v1",
        "model_repo_id": trainer.PREREGISTERED_MODEL_REPO,
        "model_revision": "a" * 40,
        "tokenizer_revision": "b" * 40,
        "baseline_repo_id": trainer.DEFAULT_BASELINE_REPO,
        "baseline_revision": "c" * 40,
    }
    payload.update(overrides)
    path = tmp_path / trainer.MODEL_PINS_FILENAME
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_reportable_run_without_frozen_pins_is_refused(tmp_path):
    trainer = _trainer()
    pins = trainer.TrainingPins(model_revision="a" * 40, tokenizer_revision="b" * 40)
    with pytest.raises(SystemExit, match="has not been frozen"):
        trainer.enforce_preregistration(_args(trainer, tmp_path), pins)


def test_a_revision_that_is_not_the_frozen_one_is_refused(tmp_path):
    """ "The revision is recorded" was satisfied by any string, including a moved tag."""
    trainer = _trainer()
    _pins(tmp_path, trainer)
    pins = trainer.TrainingPins(model_revision="9" * 40, tokenizer_revision="b" * 40)
    with pytest.raises(SystemExit, match="model_revision"):
        trainer.enforce_preregistration(_args(trainer, tmp_path), pins)


def test_the_frozen_revisions_are_accepted_and_recorded(tmp_path):
    trainer = _trainer()
    _pins(tmp_path, trainer)
    pins = trainer.TrainingPins(model_revision="a" * 40, tokenizer_revision="b" * 40)
    record = trainer.enforce_preregistration(_args(trainer, tmp_path, reportable=False), pins)
    assert record["model_pins"]["model_revision"] == "a" * 40
    assert record["git"]["git_sha"]


def test_the_manifest_records_the_source_revision_and_whether_it_was_clean():
    trainer = _trainer()
    git = trainer.git_state()
    assert set(git) >= {"git_sha", "git_branch", "git_dirty", "git_dirty_paths"}
    # This repository is a git checkout, so the SHA is real rather than None.
    assert git["git_sha"], "the training manifest would record no source revision"


def test_a_dirty_tree_blocks_a_reportable_run(tmp_path, monkeypatch):
    trainer = _trainer()
    _pins(tmp_path, trainer)
    monkeypatch.setattr(
        trainer,
        "git_state",
        lambda: {
            "git_sha": "f" * 40,
            "git_branch": "research/x",
            "git_dirty": True,
            "git_dirty_paths": ["scripts/train_detector_v4.py"],
        },
    )
    pins = trainer.TrainingPins(model_revision="a" * 40, tokenizer_revision="b" * 40)
    with pytest.raises(SystemExit, match="working tree is dirty"):
        trainer.enforce_preregistration(_args(trainer, tmp_path), pins)


def test_a_pin_file_may_not_freeze_a_tag(tmp_path):
    """A tag moves, which is the whole defect; only a 40-char commit sha is a pin."""
    from rdl.cli.detector_v4_2_model_pins import (
        detector_v4_2_freeze_model_pins,
        looks_like_commit_sha,
    )

    assert looks_like_commit_sha("a" * 40)
    assert not looks_like_commit_sha("main")
    assert not looks_like_commit_sha("v1.0")
    with pytest.raises(typer.BadParameter, match="not a tag or a branch"):
        detector_v4_2_freeze_model_pins(
            model_repo_id="microsoft/deberta-v3-base",
            baseline_repo_id="cross-encoder/nli-deberta-v3-base",
            model_revision="main",
            tokenizer_revision="main",
            baseline_revision="main",
            resolve=False,
            output_dir=tmp_path,
            refreeze=False,
        )


# =====================================================================================
# The engineering bank's eight seeds can be generated without editing the frozen study
# =====================================================================================


def test_the_preregistered_seeds_are_accepted_and_others_are_not():
    from rdl.cli.graph_common import _preregistered_seeds
    from rdl.eval.detector_v4_2 import ENGINEERING_BANK_SEEDS, ENGINEERING_RETAIN_SEEDS

    permitted = _preregistered_seeds()
    for seed in (*ENGINEERING_BANK_SEEDS, *ENGINEERING_RETAIN_SEEDS):
        assert seed in permitted
    assert 1729 in permitted, "the study's own seed must remain usable"
    assert 12345 not in permitted, "an arbitrary seed is a draw nobody registered"


def test_an_unregistered_base_seed_is_refused():
    from rdl.cli.graph_common import apply_base_seed, load_config_or_fail

    cfg = load_config_or_fail(Path("configs/graph/launch.yaml"))
    with pytest.raises(typer.BadParameter, match="not pre-registered"):
        apply_base_seed(cfg, 4242)


def test_a_preregistered_base_seed_reaches_the_study_sampling_config():
    """The plumbing that did not exist: GraphLaunchConfig carries only two keys."""
    from rdl.cli.graph_common import apply_base_seed, load_config_or_fail
    from rdl.eval.detector_v4_2 import ENGINEERING_BANK_SEEDS

    cfg = load_config_or_fail(Path("configs/graph/launch.yaml"))
    assert cfg.study.sampling.base_seed == 1729
    moved = apply_base_seed(cfg, ENGINEERING_BANK_SEEDS[0])
    assert moved.study.sampling.base_seed == ENGINEERING_BANK_SEEDS[0]
    # And the original is untouched: the frozen study is not edited, it is overridden.
    assert cfg.study.sampling.base_seed == 1729


def test_every_preregistered_run_gets_a_command(tmp_path):
    from rdl.cli.detector_v4_2_bank_runs import bank_run_commands
    from rdl.cli.detector_v4_2_banks import GENERATION_BUDGET
    from rdl.eval.detector_v4_2 import ENGINEERING_BANK_SEEDS, ENGINEERING_RETAIN_SEEDS

    commands = bank_run_commands(
        GENERATION_BUDGET,
        launch=Path("configs/graph/launch.yaml"),
        cohort_dir=Path("data/cohorts/graph_unlearning_v1"),
        output_root=Path("runs/graph"),
        profile="rtx3090_1b",
    )
    assert len(commands) == 8
    assert {c["base_seed"] for c in commands} == {
        *ENGINEERING_BANK_SEEDS,
        *ENGINEERING_RETAIN_SEEDS,
    }
    # Every command carries its seed, its cohort and its protocol — the three things the
    # operator would otherwise have to keep straight across eight invocations.
    for command in commands:
        assert f"--base-seed {command['base_seed']}" in command["command"]
        assert "--protocol graph_flow" in command["command"]
        assert "--n-samples 32" in command["command"]
    natural = [c for c in commands if c["group"] == "natural"]
    retain = [c for c in commands if c["group"] == "retain"]
    assert all("discovery.json" in c["command"] for c in natural)
    assert all("retain_utility.json" in c["command"] for c in retain)
    # POSIX separators whatever machine froze the bank: this script is written here and
    # run on the rented Linux box, where a backslash is an escape character.
    assert not any("\\" in c["command"] for c in commands), commands[0]["command"]
    # And no two runs write to the same directory.
    assert len({c["output"] for c in commands}) == 8


def test_the_emitted_cohorts_exist():
    """A command naming a cohort file that is not there fails on the box, not here."""
    from rdl.cli.detector_v4_2_bank_runs import COHORT_FILENAME

    for filename in COHORT_FILENAME.values():
        assert (REPO / "data" / "cohorts" / "graph_unlearning_v1" / filename).exists(), filename
