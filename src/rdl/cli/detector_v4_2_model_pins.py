"""``rdl graph-detector-v4-2-freeze-model-pins`` — the three commit SHAs, frozen before training.

``--model-revision`` and ``--tokenizer-revision`` were mandatory and unconstrained: the
trainer refused an empty string and accepted anything else. That pins the *shape* of the
claim rather than the claim. ``microsoft/deberta-v3-base`` at two different commits is two
different encoders and two different subword vocabularies, and two runs a month apart under
a moved tag both satisfy "the revision is recorded" while being different experiments —
with nothing in either manifest able to say so.

This writes ``DETECTOR_V4_2_MODEL_PINS.json``: the repo ids and the exact commit SHAs for

* the encoder that is fine-tuned (``microsoft/deberta-v3-base``),
* its tokenizer, which is a separate pin because a changed subword split changes every
  score at a fixed threshold while leaving the weights identical,
* the pinned NLI cross-encoder the zero-shot baseline is read from.

A reportable training run reads this file and refuses any other revision.

Resolving them
--------------
``--resolve`` asks the Hub for each repo's current ``main`` SHA and writes what it gets;
that needs network and ``huggingface_hub``. On a machine without either, pass the three
SHAs explicitly — they are printed by ``huggingface_hub.HfApi().model_info(repo).sha`` or
read off the repository's "Files and versions" page. Either way the resulting file is
committed, which is what makes the pin a pre-registration rather than a note.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import typer

from ..eval.detector_v4_2 import V4_2_PROTOCOL
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_2_llm_judge import DEFAULT_V4_2_OUT

__all__ = ["MODEL_PINS_FILENAME", "detector_v4_2_freeze_model_pins"]

MODEL_PINS_FILENAME = "DETECTOR_V4_2_MODEL_PINS.json"
MODEL_PINS_SCHEMA = "graph-detector-v4-2-model-pins-v1"

# The repositories themselves are already frozen in the trainer; this command pins the
# commits inside them. Named here so the artifact records both halves together.
DEFAULT_MODEL_REPO = "microsoft/deberta-v3-base"
DEFAULT_BASELINE_REPO = "cross-encoder/nli-deberta-v3-base"

# A Hub commit SHA. Checked so a tag name — `main`, `refs/pr/3`, a date — cannot be frozen
# as though it were a commit: a tag is the thing this file exists to stop.
_SHA_LENGTHS = (40,)


def looks_like_commit_sha(value: str) -> bool:
    text = str(value or "").strip()
    return len(text) in _SHA_LENGTHS and all(c in "0123456789abcdef" for c in text.lower())


def _resolve(repo_id: str) -> str:
    """The repo's current default-branch commit, from the Hub. Needs network."""
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - offline machines pass SHAs explicitly
        raise typer.BadParameter(
            "--resolve needs `huggingface_hub`, which the CPU extra does not install. "
            "Either install it on a networked machine, or pass the three SHAs explicitly."
        ) from exc
    info = HfApi().model_info(repo_id)
    sha = str(getattr(info, "sha", "") or "")
    if not looks_like_commit_sha(sha):
        raise typer.BadParameter(f"{repo_id}: the Hub returned {sha!r}, which is not a commit sha")
    return sha


def detector_v4_2_freeze_model_pins(
    model_repo_id: str = typer.Option(DEFAULT_MODEL_REPO, "--model-repo-id"),
    baseline_repo_id: str = typer.Option(DEFAULT_BASELINE_REPO, "--baseline-repo-id"),
    model_revision: str = typer.Option("", "--model-revision", help="exact commit sha"),
    tokenizer_revision: str = typer.Option("", "--tokenizer-revision", help="exact commit sha"),
    baseline_revision: str = typer.Option("", "--baseline-revision", help="exact commit sha"),
    resolve: bool = typer.Option(
        False, "--resolve", help="ask the Hub for each repo's current sha. NEEDS NETWORK."
    ),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
    refreeze: bool = typer.Option(
        False, "--refreeze", help="overwrite an existing pin file; records the previous one"
    ),
) -> None:
    """Freeze the exact encoder, tokenizer and baseline commits a reportable run may use."""
    path = output_dir / MODEL_PINS_FILENAME
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if previous and not refreeze:
        raise typer.BadParameter(
            f"{path} already freezes model {str(previous.get('model_revision'))[:12]} and "
            f"tokenizer {str(previous.get('tokenizer_revision'))[:12]}. Pass --refreeze "
            "only BEFORE a reportable run: changing the pin after training means the "
            "checkpoint on disk was produced by weights the artifact no longer names."
        )

    if resolve:
        model_revision = model_revision or _resolve(model_repo_id)
        tokenizer_revision = tokenizer_revision or model_revision
        baseline_revision = baseline_revision or _resolve(baseline_repo_id)

    supplied = {
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "baseline_revision": baseline_revision,
    }
    bad = {name: value for name, value in supplied.items() if not looks_like_commit_sha(value)}
    if bad:
        raise typer.BadParameter(
            "every pin must be a 40-character commit sha, not a tag or a branch: "
            + ", ".join(f"{name}={value!r}" for name, value in sorted(bad.items()))
            + ". A tag moves, and a pin that moves is the defect this file exists for. "
            "Pass --resolve on a networked machine to read them from the Hub."
        )

    payload = {
        "schema": MODEL_PINS_SCHEMA,
        "protocol": V4_2_PROTOCOL,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_repo_id": model_repo_id,
        "model_revision": model_revision,
        "tokenizer_repo_id": model_repo_id,
        "tokenizer_revision": tokenizer_revision,
        "baseline_repo_id": baseline_repo_id,
        "baseline_revision": baseline_revision,
        "resolved_from_hub": bool(resolve),
        "why_the_tokenizer_is_pinned_separately": (
            "a changed subword split changes every score at a fixed threshold while the "
            "weights stay byte-identical, so 'the same model' is not enough."
        ),
        "why_this_is_a_file": (
            "the trainer required a non-empty --model-revision and accepted whatever was "
            "typed, which pins the shape of the claim and not the claim. This file is "
            "committed, so the pin is pre-registration rather than a runtime argument."
        ),
        "enforced_by": "scripts/train_detector_v4.py enforce_model_pins(), for reportable runs",
        "previous": previous,
    }
    fingerprinted = sorted({*supplied, "model_repo_id", "baseline_repo_id"})
    payload["pins_sha256"] = hashlib.sha256(
        json.dumps({key: payload[key] for key in fingerprinted}, sort_keys=True).encode("utf-8")
    ).hexdigest()
    atomic_json(path, payload)
    typer.echo(f"wrote {path}")
    typer.echo(f"  encoder   {model_repo_id}@{model_revision}")
    typer.echo(f"  tokenizer {model_repo_id}@{tokenizer_revision}")
    typer.echo(f"  baseline  {baseline_repo_id}@{baseline_revision}")
    typer.echo("")
    typer.echo("commit this file: a reportable training run refuses any other revision.")
