"""Checkpoint hashes: computed by the trainer, and RE-COMPUTED by everything that loads one.

A hash that is written once and never checked again is a comment. v4.2 recorded
``selected_checkpoint_hashes`` in ``DETECTOR_V4_MODEL.json`` and then loaded the checkpoint
by path, so a directory that had been re-exported, re-quantised, partially copied, or
overwritten by a later training run loaded silently and produced numbers under the old
run's hashes. The manifest would still say which bytes were meant to have been used.

This module is the one implementation. ``scripts/train_detector_v4.py`` imports it to write
the hashes and :meth:`CrossEncoderAnswerabilityDetector.from_artifact` imports it to verify
them, so the two cannot drift — a verifier that hashes a different file set than the writer
did reports a mismatch on every honest checkpoint, which is the failure that gets
verification turned off.

It imports nothing but the standard library, deliberately: it runs inside the CPU gate,
inside the trainer on the GPU box, and inside the detector's loading path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "HASHED_FILE_GLOBS",
    "checkpoint_hashes",
    "compare_checkpoint_hashes",
    "verify_checkpoint_hashes",
]

# Everything that changes what the checkpoint computes. The tokenizer files are in here for
# the same reason the tokenizer revision is pinned: a changed subword split changes every
# score at a fixed threshold while the weights stay byte-identical.
HASHED_FILE_GLOBS = (
    "*.safetensors",
    "pytorch_model.bin",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "spm.model",
    "sentencepiece.bpe.model",
    "vocab.txt",
    "special_tokens_map.json",
    "added_tokens.json",
)

LABELS: tuple[str, ...] = ("NONE", "PARTIAL", "ANSWER")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hashes(path: Path, *, labels: tuple[str, ...] = LABELS) -> dict:
    """SHA-256 of everything that makes the checkpoint what it is.

    A path is not a checkpoint: a copied, re-exported or re-uploaded directory has the same
    path in a manifest and different bytes on disk. The label map is hashed separately
    because a permuted ``id2label`` changes every score at a fixed threshold while leaving
    the weights byte-identical.
    """
    path = Path(path)
    files: dict[str, str] = {}
    for pattern in HASHED_FILE_GLOBS:
        for candidate in sorted(path.glob(pattern)):
            if candidate.is_file():
                files[candidate.name] = _sha256_file(candidate)
    label_map = None
    config_path = path / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        label_map = {"id2label": config.get("id2label"), "label2id": config.get("label2id")}
    return {
        "files_sha256": files,
        "n_files_hashed": len(files),
        "label_map": label_map,
        "label_map_sha256": (
            hashlib.sha256(json.dumps(label_map, sort_keys=True).encode("utf-8")).hexdigest()
            if label_map
            else None
        ),
        "expected_label_order": list(labels),
        "why": (
            "a path is not a checkpoint. These hashes let a copied or released directory "
            "be checked against the one that produced the reported numbers, and are "
            "RE-COMPUTED at load time rather than trusted."
        ),
    }


def compare_checkpoint_hashes(recorded: Mapping | None, actual: Mapping) -> list[str]:
    """Every way a checkpoint on disk differs from the one a manifest recorded.

    Returns failure strings rather than raising, so one load names all of them. An absent
    ``recorded`` is itself a failure: a checkpoint whose hashes were never written cannot
    be shown to be the one that produced the numbers, and "not recorded" must not read the
    same as "matches".
    """
    if not recorded:
        return [
            "the manifest records no checkpoint hashes. A checkpoint that was never "
            "hashed cannot be shown to be the one that produced the reported numbers; "
            "re-run the trainer, which writes them."
        ]
    failures: list[str] = []
    recorded_files = dict(recorded.get("files_sha256") or {})
    actual_files = dict(actual.get("files_sha256") or {})
    if not actual_files:
        failures.append(
            "no hashable file was found in the checkpoint directory. It is empty, moved, "
            "or holds a format this protocol does not hash."
        )
    for name in sorted(set(recorded_files) | set(actual_files)):
        want, have = recorded_files.get(name), actual_files.get(name)
        if want is None:
            failures.append(f"{name}: present on disk and absent from the manifest")
        elif have is None:
            failures.append(f"{name}: recorded in the manifest and absent on disk")
        elif want != have:
            failures.append(f"{name}: recorded {want[:16]}…, on disk {have[:16]}…")
    want_map = recorded.get("label_map_sha256")
    have_map = actual.get("label_map_sha256")
    if want_map != have_map:
        failures.append(
            f"label map: recorded {str(want_map)[:16]}…, on disk {str(have_map)[:16]}…. "
            "A permuted id2label changes every score at a fixed threshold while leaving "
            "the weights byte-identical."
        )
    expected_order = list(recorded.get("expected_label_order") or LABELS)
    on_disk = (actual.get("label_map") or {}).get("id2label") or {}
    if on_disk:
        ordered = [str(on_disk[k]) for k in sorted(on_disk, key=lambda k: int(k))]
        if ordered != expected_order:
            failures.append(
                f"label order on disk is {ordered}, and the protocol's order is "
                f"{expected_order}. The label order is part of the checkpoint."
            )
    return failures


def verify_checkpoint_hashes(recorded: Mapping | None, checkpoint: Path) -> dict:
    """Re-hash ``checkpoint`` and raise :class:`ValueError` unless it matches ``recorded``.

    Returns the freshly computed hashes on success, so a caller can record what it
    actually verified rather than echoing the manifest back.
    """
    actual = checkpoint_hashes(Path(checkpoint))
    failures = compare_checkpoint_hashes(recorded, actual)
    if failures:
        raise ValueError(
            f"{checkpoint} does not match the checkpoint hashes recorded for it:\n  "
            + "\n  ".join(failures)
            + "\nRefusing to load. Every number produced from this directory would be "
            "attributed to bytes it is not."
        )
    return actual
