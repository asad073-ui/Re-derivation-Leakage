"""Three training-side defects that produce a checkpoint rather than an error.

**Checkpoint hashes were written and never checked.** ``DETECTOR_V4_MODEL.json`` recorded
``selected_checkpoint_hashes`` and every loader then opened the directory by path. A
re-exported, partially copied, or later-overwritten checkpoint loaded silently and its
numbers were attributed to bytes it did not contain.

**The accumulation tail was under-scaled.** v4.1 discarded the epoch's final partial
accumulation; v4.2.1 stopped discarding it and still divided it by the full accumulation
factor, so one optimizer step per epoch ran on a gradient scaled to a fraction of every
other step's. Neither version shows up in a loss curve.

**"Preregistered" settings were defaults.** ``--seeds``, ``--epochs`` and the model pins
were flags whose defaults happened to be the preregistered values. A run with three
different seeds wrote a manifest recording those seeds beside a selection rule describing
the preregistered ones, and nothing said the plan had been left.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def _trainer():
    path = REPO / "scripts" / "train_detector_v4.py"
    spec = importlib.util.spec_from_file_location("train_detector_v4_guards_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# =====================================================================================
# The accumulation tail is normalized by its own size
# =====================================================================================


def test_a_full_epoch_is_unchanged():
    trainer = _trainer()
    assert trainer.accumulation_windows(8, 2) == [2] * 8


def test_the_tail_is_divided_by_its_own_length():
    """101 batches at factor 2: the last window holds ONE batch, not two."""
    trainer = _trainer()
    windows = trainer.accumulation_windows(101, 2)
    assert len(windows) == 101
    assert windows[:100] == [2] * 100
    assert windows[100] == 1


def test_the_tail_of_a_larger_factor_is_the_real_remainder():
    trainer = _trainer()
    windows = trainer.accumulation_windows(10, 4)
    assert windows == [4] * 8 + [2, 2]


def test_every_window_sums_to_one_full_gradient():
    """The property, not the arithmetic: each window's divisors sum to exactly 1.0.

    This is what "correctly normalized" means. Under the old rule the final window of an
    uneven epoch summed to 0.5 (or 0.33), which is one optimizer step per epoch taken at
    half the intended learning rate.
    """
    trainer = _trainer()
    for n_batches, accumulation in ((101, 2), (10, 4), (7, 3), (5, 5), (1, 8)):
        windows = trainer.accumulation_windows(n_batches, accumulation)
        assert len(windows) == n_batches
        step_boundaries = [
            i for i in range(1, n_batches + 1) if i % accumulation == 0 or i == n_batches
        ]
        start = 0
        for boundary in step_boundaries:
            window = windows[start:boundary]
            assert sum(1 / w for w in window) == pytest.approx(1.0), (n_batches, accumulation)
            start = boundary


def test_the_old_rule_really_did_under_scale_the_tail():
    """Stated as a test so the size of the effect is not a matter of opinion."""
    trainer = _trainer()
    windows = trainer.accumulation_windows(101, 2)
    corrected = sum(1 / w for w in windows[100:])
    old = sum(1 / 2 for _ in windows[100:])
    assert corrected == pytest.approx(1.0)
    assert old == pytest.approx(0.5)


# =====================================================================================
# Checkpoint hashes are re-computed, not trusted
# =====================================================================================


def _checkpoint(tmp_path: Path, *, weights: bytes = b"weights") -> Path:
    path = tmp_path / "seed1-checkpoint-epoch1"
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.safetensors").write_bytes(weights)
    (path / "config.json").write_text(
        json.dumps(
            {
                "id2label": {"0": "NONE", "1": "PARTIAL", "2": "ANSWER"},
                "label2id": {"NONE": 0, "PARTIAL": 1, "ANSWER": 2},
            }
        ),
        encoding="utf-8",
    )
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    return path


def test_matching_bytes_verify(tmp_path):
    from rdl.defenses.checkpoint_digest import checkpoint_hashes, verify_checkpoint_hashes

    checkpoint = _checkpoint(tmp_path)
    recorded = checkpoint_hashes(checkpoint)
    assert verify_checkpoint_hashes(recorded, checkpoint)["n_files_hashed"] == 3


def test_changed_weights_are_refused(tmp_path):
    """The failure the recorded hashes existed for, now actually detected."""
    from rdl.defenses.checkpoint_digest import checkpoint_hashes, verify_checkpoint_hashes

    checkpoint = _checkpoint(tmp_path)
    recorded = checkpoint_hashes(checkpoint)
    (checkpoint / "model.safetensors").write_bytes(b"different weights")
    with pytest.raises(ValueError, match=r"model\.safetensors"):
        verify_checkpoint_hashes(recorded, checkpoint)


def test_a_permuted_label_map_is_refused(tmp_path):
    """Byte-identical weights, every score at a fixed threshold different."""
    from rdl.defenses.checkpoint_digest import checkpoint_hashes, verify_checkpoint_hashes

    checkpoint = _checkpoint(tmp_path)
    recorded = checkpoint_hashes(checkpoint)
    (checkpoint / "config.json").write_text(
        json.dumps(
            {
                "id2label": {"0": "ANSWER", "1": "PARTIAL", "2": "NONE"},
                "label2id": {"ANSWER": 0, "PARTIAL": 1, "NONE": 2},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="label"):
        verify_checkpoint_hashes(recorded, checkpoint)


def test_a_missing_file_is_refused(tmp_path):
    from rdl.defenses.checkpoint_digest import checkpoint_hashes, verify_checkpoint_hashes

    checkpoint = _checkpoint(tmp_path)
    recorded = checkpoint_hashes(checkpoint)
    (checkpoint / "tokenizer.json").unlink()
    with pytest.raises(ValueError, match="absent on disk"):
        verify_checkpoint_hashes(recorded, checkpoint)


def test_a_manifest_with_no_hashes_at_all_is_refused(tmp_path):
    """ "Never hashed" must not read the same as "matches"."""
    from rdl.defenses.checkpoint_digest import verify_checkpoint_hashes

    with pytest.raises(ValueError, match="never hashed"):
        verify_checkpoint_hashes(None, _checkpoint(tmp_path))


def test_the_loader_verifies_before_it_loads(tmp_path):
    """`from_artifact` refuses a moved checkpoint BEFORE transformers is imported.

    The ordering is the point: on a rented box the check has to happen before a minute of
    model loading, and the error must name the mismatch rather than a missing torch.
    """
    from rdl.defenses.checkpoint_digest import checkpoint_hashes
    from rdl.defenses.cross_encoder_answerability import CrossEncoderAnswerabilityDetector

    checkpoint = _checkpoint(tmp_path)
    artifact = tmp_path / "DETECTOR_V4_MODEL.json"
    artifact.write_text(
        json.dumps(
            {
                "pins": {
                    "model_repo_id": "microsoft/deberta-v3-base",
                    "model_revision": "abc",
                    "tokenizer_revision": "abc",
                },
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_hashes": checkpoint_hashes(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    (checkpoint / "model.safetensors").write_bytes(b"someone re-exported this")
    with pytest.raises(ValueError, match="does not match the checkpoint hashes"):
        CrossEncoderAnswerabilityDetector.from_artifact(artifact)


def test_an_artifact_with_no_recorded_hashes_is_refused_by_the_loader(tmp_path):
    from rdl.defenses.cross_encoder_answerability import CrossEncoderAnswerabilityDetector

    checkpoint = _checkpoint(tmp_path)
    artifact = tmp_path / "DETECTOR_V4_MODEL.json"
    artifact.write_text(
        json.dumps(
            {
                "pins": {
                    "model_repo_id": "microsoft/deberta-v3-base",
                    "model_revision": "abc",
                    "tokenizer_revision": "abc",
                },
                "selected_checkpoint": str(checkpoint),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="never hashed"):
        CrossEncoderAnswerabilityDetector.from_artifact(artifact)


# =====================================================================================
# The preregistration is enforced, and leaving it is recorded
# =====================================================================================


# v4.2.3 added two more conditions to a reportable run: the exact commit SHAs must be
# frozen (DETECTOR_V4_2_MODEL_PINS.json) and the working tree must be clean. These
# fixtures supply both, so each test below still isolates the ONE thing it is about — a
# tree that happens to be dirty must not decide whether a seed check passes.
FROZEN = {"model": "a" * 40, "tokenizer": "b" * 40, "baseline": "c" * 40}


@pytest.fixture()
def prereg(tmp_path, monkeypatch):
    """A directory carrying frozen pins, and a clean git state."""
    trainer = _trainer()
    (tmp_path / trainer.MODEL_PINS_FILENAME).write_text(
        json.dumps(
            {
                "model_repo_id": trainer.PREREGISTERED_MODEL_REPO,
                "model_revision": FROZEN["model"],
                "tokenizer_revision": FROZEN["tokenizer"],
                "baseline_repo_id": trainer.DEFAULT_BASELINE_REPO,
                "baseline_revision": FROZEN["baseline"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trainer,
        "git_state",
        lambda: {
            "git_sha": "e" * 40,
            "git_branch": "research/x",
            "git_dirty": False,
            "git_dirty_paths": [],
        },
    )
    return trainer, tmp_path


def _pins(trainer, **overrides):
    defaults = {"model_revision": FROZEN["model"], "tokenizer_revision": FROZEN["tokenizer"]}
    defaults.update(overrides)
    return trainer.TrainingPins(**defaults)


def _args(trainer, tmp_path=None, **overrides) -> argparse.Namespace:
    defaults = {
        "epochs": trainer.PREREGISTERED_EPOCHS,
        "model_repo_id": trainer.PREREGISTERED_MODEL_REPO,
        "baseline_repo_id": trainer.DEFAULT_BASELINE_REPO,
        "baseline_revision": FROZEN["baseline"],
        "skip_baseline": False,
        "extra_train": [],
        "reportable": True,
        "declare_extra_train": False,
        "v4_2_dir": tmp_path,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_the_preregistered_run_is_accepted(prereg):
    trainer, tmp_path = prereg
    pins = _pins(trainer)
    record = trainer.enforce_preregistration(_args(trainer, tmp_path), pins)
    assert record["reportable"] is True
    assert record["deviations"] == []


def test_a_reportable_run_with_other_seeds_is_refused(prereg):
    trainer, tmp_path = prereg
    pins = _pins(trainer, seeds=(7,))
    with pytest.raises(SystemExit, match="does not match the preregistration"):
        trainer.enforce_preregistration(_args(trainer, tmp_path), pins)


def test_a_reportable_run_with_other_epochs_is_refused(prereg):
    trainer, tmp_path = prereg
    pins = _pins(trainer)
    with pytest.raises(SystemExit, match="epochs 12"):
        trainer.enforce_preregistration(_args(trainer, tmp_path, epochs=12), pins)


def test_a_reportable_run_on_another_encoder_is_refused(prereg):
    trainer, tmp_path = prereg
    pins = _pins(trainer, model_repo_id="roberta-base")
    with pytest.raises(SystemExit, match="different encoder is a different experiment"):
        trainer.enforce_preregistration(
            _args(trainer, tmp_path, model_repo_id="roberta-base"), pins
        )


def test_leaving_the_preregistration_is_allowed_only_as_non_reportable(prereg):
    """The escape hatch does not skip the check; it marks the run."""
    trainer, tmp_path = prereg
    pins = _pins(trainer, seeds=(7,))
    record = trainer.enforce_preregistration(_args(trainer, tmp_path, reportable=False), pins)
    assert record["reportable"] is False
    assert any("seeds" in d for d in record["deviations"])


def test_every_extra_train_file_is_hashed(prereg):
    """`--extra-train some.jsonl` used to leave no trace beyond a row count."""
    trainer, tmp_path = prereg
    extra = tmp_path / "extra.jsonl"
    extra.write_text(
        json.dumps({"question": "q", "candidate": "c", "label": "NONE"}) + "\n", encoding="utf-8"
    )
    pins = _pins(trainer)
    record = trainer.enforce_preregistration(
        _args(trainer, tmp_path, extra_train=[extra], reportable=False), pins
    )
    assert record["extra_train_files"][0]["sha256"]
    assert record["extra_train_files"][0]["n_rows"] == 1


def test_a_reportable_run_must_declare_its_extra_training_data(prereg):
    trainer, tmp_path = prereg
    extra = tmp_path / "extra.jsonl"
    extra.write_text("{}\n", encoding="utf-8")
    pins = _pins(trainer)
    with pytest.raises(SystemExit, match="declare-extra-train"):
        trainer.enforce_preregistration(_args(trainer, tmp_path, extra_train=[extra]), pins)

    declared = trainer.enforce_preregistration(
        _args(trainer, tmp_path, extra_train=[extra], declare_extra_train=True), pins
    )
    assert declared["extra_train_declared"] is True


def test_an_extra_train_file_that_does_not_exist_is_refused(prereg):
    trainer, tmp_path = prereg
    pins = _pins(trainer)
    with pytest.raises(SystemExit, match="do not exist"):
        trainer.enforce_preregistration(
            _args(trainer, tmp_path, extra_train=[tmp_path / "absent.jsonl"], reportable=False),
            pins,
        )
