"""``rdl graph-detector-v4-4-finalize`` -- the one record that says the chain held.

Every stage of detector v4.4 writes its own artifact and every one of them is checked
where it is produced. What did not exist was anything that read the whole chain at once
and said, in a form a machine can verify, that the detector at the end of it is the
detector the protocol describes.

That gap is not cosmetic. The v4.4 artifacts are produced across two GPU rentals, days
apart, by an operator following a runbook; between them sit a human evaluation, a sealed
bank, and a checkpoint that has to be uploaded to external storage and pulled back down.
A reader asking "is this deployable" had, before this command, to open ten files in the
right order and know which field in each one carried the verdict -- and to notice, without
help, if the checkpoint that scored the final gate was not the one the humans validated.

What this command is NOT
------------------------
It is not a gate. It measures nothing, scores nothing, and can turn no failure into a
pass. Every verdict it reports was decided by the command that wrote the artifact; this
reads them, re-computes the hashes that tie them to each other, and refuses to write
``deployable: true`` unless all of them already say so and all of them refer to the same
detector.

It also never edits an upstream artifact. A frozen file amended to say PASS is the failure
mode the whole protocol is built against, so the verdict goes in a NEW record whose inputs
are named and hashed.

The chain, in the order it was produced
---------------------------------------
1.  label authority        -- the labels are closed and green
2.  model artifact         -- which checkpoint, by digest
3.  operating point        -- tau_answer / tau_partial, chosen on development
4.  frozen detector        -- those thresholds, bound to that checkpoint
5.  engineering held-out   -- opened once, passed
6.  human report           -- 250 rows, two people, passed
7.  final-bank unseal      -- authorised by 6, and re-verified
8.  final-bank build       -- built once, from the sealed seeds
9.  final labelled audit   -- the final bank's labels, closed
10. final gate             -- opened once, passed

Link 6 to 7 is the one that matters most: the human report is the only thing in the
protocol permitted to unseal the final bank, and the unseal record re-verifies it rather
than trusting a flag. This command checks that the human report which authorised the
unseal is byte-identical to the one on disk now.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping
from pathlib import Path

import typer

from ..eval.detector_v4_4 import V4_4_PROTOCOL
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT
from .detector_v4_2_banks import BUILD_RECORD_FILENAME
from .detector_v4_2_llm_judge import DEFAULT_V4_2_OUT
from .detector_v4_4_bundle import DEFAULT_V4_4_DIR
from .detector_v4_4_fresh_report import fresh_labelled_filename
from .detector_v4_4_gate import (
    FROZEN_DETECTOR_FILENAME,
    HELDOUT_REPORT_FILENAME,
    OPERATING_POINT_FILENAME,
    gate_report_filename,
)
from .detector_v4_4_human import HUMAN_REPORT_FILENAME
from .detector_v4_4_report import AUTHORITY_FILENAME
from .detector_v4_4_unseal import UNSEAL_RECORD_FILENAME, verify_unseal_record

__all__ = [
    "FINAL_VALIDATION_FILENAME",
    "FINAL_VALIDATION_SCHEMA",
    "detector_v4_4_finalize",
]

FINAL_VALIDATION_FILENAME = "DETECTOR_V4_4_FINAL_VALIDATION.json"
FINAL_VALIDATION_SCHEMA = "graph-detector-v4-4-final-validation-v1"
MODEL_ARTIFACT_FILENAME = "DETECTOR_V4_MODEL.json"
FINAL_GATE_FILENAME = gate_report_filename("final")


def _sha_file(path: Path) -> str | None:
    """Byte hash, matching the convention the unseal record was written under.

    ``is_file()`` for the reason ``detector_v4_4_unseal`` gives: a missing path yields
    ``Path("")``, which is a directory that exists, and hashing it raises two different
    errors on two platforms where the answer wanted is "no digest".
    """
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _load(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _passed(payload: Mapping | None, *keys: str) -> bool | None:
    """The verdict an artifact already carries, under whichever key it uses.

    ``None`` means the artifact does not state a verdict at all, which is reported as its
    own failure rather than folded into False: "this gate failed" and "this file does not
    say whether it passed" call for different reactions from whoever reads the record.
    """
    if payload is None:
        return None
    for key in keys:
        if key in payload:
            return bool(payload[key])
    return None


def detector_v4_4_finalize(
    v4_4_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--v4-4-dir"),
    v4_2_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--v4-2-dir"),
    v4_1_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--v4-1-dir"),
    model_artifact: Path = typer.Option(
        None, "--model-artifact", help=f"defaults to --v4-4-dir/{MODEL_ARTIFACT_FILENAME}"
    ),
    output_dir: Path = typer.Option(None, "--output-dir", help="defaults to --v4-4-dir"),
) -> None:
    """Verify the whole v4.4 chain and record whether the detector is deployable."""
    v44 = Path(v4_4_dir)
    v42 = Path(v4_2_dir)
    out = Path(output_dir) if output_dir else v44
    model_path = Path(model_artifact) if model_artifact else v44 / MODEL_ARTIFACT_FILENAME

    paths = {
        "label_authority": v44 / AUTHORITY_FILENAME,
        "model_artifact": model_path,
        "operating_point": v44 / OPERATING_POINT_FILENAME,
        "frozen_detector": v44 / FROZEN_DETECTOR_FILENAME,
        "engineering_heldout_gate": v44 / HELDOUT_REPORT_FILENAME,
        "human_report": v44 / HUMAN_REPORT_FILENAME,
        "final_bank_unseal_record": v42 / UNSEAL_RECORD_FILENAME,
        "final_bank_build_record": v42 / BUILD_RECORD_FILENAME["final"],
        "final_labelled_audit": v44 / fresh_labelled_filename("final"),
        "final_gate": v44 / FINAL_GATE_FILENAME,
    }
    loaded = {name: _load(path) for name, path in paths.items()}

    failures: list[str] = []
    for name, path in paths.items():
        if not path.exists():
            failures.append(f"{name}: {path} is absent")
        elif loaded[name] is None:
            failures.append(f"{name}: {path} is not readable JSON")

    # ------------------------------------------------------- each stage's own verdict --
    # Read, never recomputed. A finalizer that re-derived a gate's verdict would be a
    # second implementation of that gate, and the two would eventually disagree.
    verdicts: dict[str, bool | None] = {
        "label_authority_green": _passed(loaded["label_authority"], "green"),
        "engineering_heldout_gate_passed": _passed(loaded["engineering_heldout_gate"], "passed"),
        "human_report_passed": _passed(loaded["human_report"], "passed"),
        "final_bank_unsealed": _passed(loaded["final_bank_unseal_record"], "unsealed"),
        "final_labelled_audit_passed": _passed(loaded["final_labelled_audit"], "passed"),
        "final_gate_passed": _passed(loaded["final_gate"], "passed"),
    }
    for name, verdict in sorted(verdicts.items()):
        if verdict is None:
            failures.append(f"{name}: the artifact states no verdict")
        elif not verdict:
            failures.append(f"{name}: FAILED")

    # A reportable human sample is a condition of the human gate meaning anything, and a
    # human report can pass while describing an exploratory draw.
    human = loaded["human_report"] or {}
    if human and not human.get("reportable"):
        failures.append(
            "human_report: the sample is not reportable, so the human gate does not "
            "support the claim the final validation would be making"
        )
    if human.get("provenance_failures"):
        failures.append(f"human_report: {len(human['provenance_failures'])} provenance failure(s)")

    # ------------------------------------------------------------------ the one-shots --
    final_gate = loaded["final_gate"] or {}
    if final_gate.get("reopened"):
        failures.append(
            "final_gate: the report records a REOPENING. The final gate is opened once; a "
            "second opening means the number being reported is not the first one this "
            "detector produced on the sealed bank"
        )
    heldout = loaded["engineering_heldout_gate"] or {}
    if heldout.get("reopened"):
        failures.append("engineering_heldout_gate: the report records a reopening")

    build_record = loaded["final_bank_build_record"] or {}
    if build_record and build_record.get("bank") != "final":
        failures.append(
            f"final_bank_build_record: describes bank {build_record.get('bank')!r}, not the "
            "final one"
        )
    if build_record and build_record.get("clean_tree") is not True:
        failures.append(
            "final_bank_build_record: the final bank was built from a tree that was dirty "
            f"or unknowable (clean_tree={build_record.get('clean_tree')!r})"
        )

    # ------------------------------------------------------------- the seal, re-checked --
    # The same verification `build-bank --bank final` runs, repeated here against the files
    # as they stand now. An unseal record is a file, and a file can be copied in.
    seal_failures = verify_unseal_record(
        paths["final_bank_unseal_record"], v4_2_dir=v42, v4_1_dir=Path(v4_1_dir)
    )
    failures.extend(f"final_bank_unseal_record: {f}" for f in seal_failures)

    # ------------------------------------------------------------------- the bindings --
    # One detector, all the way through. Each of these says two artifacts refer to the same
    # object; a mismatch means the chain describes two different detectors and the reader
    # cannot tell which one the final number belongs to.
    frozen = loaded["frozen_detector"] or {}
    operating_point = loaded["operating_point"] or {}
    model = loaded["model_artifact"] or {}
    unseal = loaded["final_bank_unseal_record"] or {}

    # The unseal record nests each binding under the artifact it describes, and its hashes
    # are of file BYTES -- the same convention _sha_file uses here, deliberately, so these
    # comparisons are of like with like. The frozen detector separately carries a
    # `frozen_detector_sha256` that is a hash of its own CONTENT, not of the file; those
    # two are never compared to each other.
    unseal_human = unseal.get("human_report") or {}
    unseal_frozen = unseal.get("frozen_detector") or {}
    unseal_point = unseal.get("operating_point") or {}

    bindings = {
        "human_report_that_unsealed_the_bank": (
            unseal_human.get("sha256"),
            _sha_file(paths["human_report"]),
        ),
        "frozen_detector_the_unseal_bound": (
            unseal_frozen.get("sha256"),
            _sha_file(paths["frozen_detector"]),
        ),
        "frozen_detector_content_the_unseal_bound": (
            unseal_frozen.get("frozen_detector_sha256"),
            frozen.get("frozen_detector_sha256"),
        ),
        "operating_point_the_unseal_bound": (
            unseal_point.get("sha256"),
            _sha_file(paths["operating_point"]),
        ),
        "operating_point_behind_the_frozen_detector": (
            frozen.get("operating_point_sha256"),
            operating_point.get("operating_point_sha256"),
        ),
        "checkpoint_the_frozen_detector_names": (
            frozen.get("checkpoint_digest") or frozen.get("selected_checkpoint_digest"),
            model.get("checkpoint_digest"),
        ),
        "tau_answer": (frozen.get("tau_answer"), operating_point.get("tau_answer")),
        "tau_partial": (frozen.get("tau_partial"), operating_point.get("tau_partial")),
        "final_gate_scored_this_operating_point": (
            final_gate.get("operating_point_sha256"),
            operating_point.get("operating_point_sha256"),
        ),
    }
    bound: dict[str, dict] = {}
    for name, (left, right) in bindings.items():
        # Two Nones is not agreement. It is two artifacts that both declined to say, and
        # treating it as a match is how an unbound chain reports itself as bound.
        if left is None and right is None:
            bound[name] = {"matches": False, "expected": None, "found": None, "stated": False}
            failures.append(f"{name}: neither artifact states this, so nothing binds them")
        else:
            matches = left == right
            bound[name] = {"matches": matches, "expected": left, "found": right, "stated": True}
            if not matches:
                failures.append(f"{name}: {left!r} != {right!r}")

    deployable = not failures
    record = {
        "schema": FINAL_VALIDATION_SCHEMA,
        "protocol": V4_4_PROTOCOL,
        "validated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "deployable": deployable,
        "answerability_v4_ready": deployable,
        "failures": failures,
        "chain": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "sha256": _sha_file(path),
            }
            for name, path in sorted(paths.items())
        },
        "verdicts": verdicts,
        "bindings": bound,
        "measured": {
            "human_human_kappa": (human.get("human_human") or {}).get("kappa"),
            "model_consensus_macro_f1": (human.get("model_consensus_vs_human") or {}).get(
                "macro_f1"
            ),
            "engineering_heldout": heldout.get("measured"),
            "final_gate": final_gate.get("measured"),
        },
        "detector": {
            "selected_checkpoint": model.get("selected_checkpoint"),
            "checkpoint_digest": model.get("checkpoint_digest"),
            "tau_answer": frozen.get("tau_answer"),
            "tau_partial": frozen.get("tau_partial"),
            "protected_store_fingerprint": frozen.get("protected_store_fingerprint"),
        },
        "measures_nothing": (
            "every verdict here was decided by the command that wrote the artifact it was "
            "read from. This record verifies that they all refer to one detector and that "
            "none of them failed. It cannot turn a failure into a pass."
        ),
        "checkpoint_bytes": (
            "outside Git, in the external repository CHECKPOINT_RETRIEVAL_MANIFEST.json "
            "names, and identified here by digest."
        ),
        "if_this_says_false": (
            "the detector is not deployable and v4.4 is not the version that ships. A "
            "failed chain is repaired by a new detector version and a new, untouched final "
            "bank -- never by editing an artifact named above."
        ),
    }
    atomic_json(out / FINAL_VALIDATION_FILENAME, record)

    for failure in failures:
        typer.echo(f"  [FAIL] {failure}", err=True)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / FINAL_VALIDATION_FILENAME),
                "deployable": deployable,
                "answerability_v4_ready": deployable,
                "n_failures": len(failures),
            }
        )
    )
    if failures:
        # Non-zero, and the record stays on disk. The failing record IS the evidence, and
        # the runner phase after this one must not proceed as though nothing happened.
        raise typer.Exit(1)
