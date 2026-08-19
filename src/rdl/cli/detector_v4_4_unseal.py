"""``rdl graph-detector-v4-4-unseal-final-bank`` -- the only thing that opens the seal.

``graph-detector-v4-2-build-bank --bank final`` refused unconditionally, and its refusal
said: *"To lift this, record the decision in DECISIONS.md and change this function
deliberately."* That is a source edit made **after** the human result is known, by the
person who wants the bank, and it is exactly the shape of decision a seal exists to take out
of anybody's hands. The seal cannot be a comment.

So the seal is now machine-enforced. This command writes ``FINAL_BANK_UNSEAL_RECORD.json``
and refuses unless it can verify, from files on disk:

* a v4.4 human report that exists, is reportable, and **passed** every gate;
* the frozen detector artifact, and its self-recorded digest;
* the frozen operating point, and that the frozen detector was built from that exact one;
* that the final bank has **not** already been opened;
* the final-bank manifest and budget, by hash.

``build-bank --bank final`` then looks for that record and re-verifies the same hashes
rather than trusting the file's existence. Two checks of one fact, because the record is a
file and files can be copied from somewhere else.

What this command cannot do
---------------------------
It cannot pass a failing human report, and there is no flag that makes it. If the human gate
failed, the correct move is a new detector version against a new fresh bank -- not a reopened
seal, and not a rerun of the humans until they agree. It also cannot unseal a bank that has
already been opened: "opened exactly once" is enforced by the presence of the opening record,
and an unseal that ignored it would make the count unbounded.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import typer

from ..eval.detector_v4_4 import V4_4_PROTOCOL
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT, GATE_BANK_MANIFEST_FILENAME
from .detector_v4_2_llm_judge import DEFAULT_V4_2_OUT
from .detector_v4_4_bundle import DEFAULT_V4_4_DIR
from .detector_v4_4_gate import FROZEN_DETECTOR_FILENAME, OPERATING_POINT_FILENAME
from .detector_v4_4_human import HUMAN_REPORT_FILENAME, HUMAN_REPORT_SCHEMA

__all__ = [
    "UNSEAL_RECORD_FILENAME",
    "UNSEAL_RECORD_SCHEMA",
    "detector_v4_4_unseal_final_bank",
    "verify_unseal_record",
]

UNSEAL_RECORD_FILENAME = "FINAL_BANK_UNSEAL_RECORD.json"
UNSEAL_RECORD_SCHEMA = "graph-detector-v4-4-final-bank-unseal-v1"
FINAL_BUDGET_FILENAME = "FINAL_GATE_BANK_BUDGET.json"
FINAL_OPENING_RECORD_FILENAME = "FINAL_GATE_BANK_OPENING_RECORD.json"


def _sha_file(path: Path) -> str | None:
    """``None`` unless ``path`` is a readable FILE.

    ``is_file()`` rather than ``exists()``: a record with a missing ``file`` key yields
    ``Path("")``, which is the current directory, and that directory exists. Hashing it
    raises PermissionError on Windows and IsADirectoryError elsewhere -- two different
    crashes where the answer wanted is "no digest".
    """
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def verify_unseal_record(record_path: Path, *, v4_2_dir: Path, v4_1_dir: Path) -> list[str]:
    """``[]`` if the seal is genuinely lifted, else the reasons it is not.

    Re-verifies rather than trusts. The record names hashes; this recomputes them from the
    files it names, so a record copied from another checkout, or one whose human report was
    edited afterwards, fails here rather than at the point where a final bank already
    exists.
    """
    failures: list[str] = []
    if not record_path.exists():
        return [
            f"{record_path} is absent. The final bank is sealed until "
            "`rdl graph-detector-v4-4-unseal-final-bank` verifies a PASSING human report."
        ]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if str(record.get("schema")) != UNSEAL_RECORD_SCHEMA:
        failures.append(f"{record_path} carries schema {record.get('schema')!r}")
    if not record.get("unsealed"):
        failures.append(f"{record_path} does not record a lifted seal")

    for name in ("human_report", "frozen_detector", "operating_point"):
        entry = record.get(name) or {}
        named = str(entry.get("file") or "")
        if not named:
            failures.append(f"the unseal record names no {name} file")
            continue
        path = Path(named)
        if not path.is_file():
            failures.append(f"{name} {path} named by the unseal record is absent")
            continue
        if _sha_file(path) != entry.get("sha256"):
            failures.append(
                f"{name} {path} has changed since the seal was lifted: the record binds "
                f"{str(entry.get('sha256'))[:16]}..., the file hashes to "
                f"{str(_sha_file(path))[:16]}..."
            )

    human = record.get("human_report") or {}
    if not human.get("passed"):
        failures.append("the unseal record does not record a PASSING human report")

    opening = v4_2_dir / FINAL_OPENING_RECORD_FILENAME
    if opening.exists():
        failures.append(
            f"{opening} exists: the final bank has already been opened. Unsealing it again "
            "would make 'opened exactly once' unbounded."
        )

    for label, path in (
        ("final bank manifest", v4_1_dir / GATE_BANK_MANIFEST_FILENAME),
        ("final bank budget", v4_2_dir / FINAL_BUDGET_FILENAME),
    ):
        entry = (record.get("final_bank") or {}).get(label.replace(" ", "_")) or {}
        if _sha_file(path) != entry.get("sha256"):
            failures.append(f"the {label} has changed since the seal was lifted")
    return failures


def detector_v4_4_unseal_final_bank(
    v4_4_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--v4-4-dir"),
    v4_2_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--v4-2-dir"),
    v4_1_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--v4-1-dir"),
    human_report: Path = typer.Option(None, "--human-report"),
    frozen_detector: Path = typer.Option(None, "--frozen-detector"),
    operating_point: Path = typer.Option(None, "--operating-point"),
) -> None:
    """Lift the final-bank seal, IF a passing human report and a frozen detector exist."""
    report_path = Path(human_report) if human_report else Path(v4_4_dir) / HUMAN_REPORT_FILENAME
    detector_path = (
        Path(frozen_detector) if frozen_detector else Path(v4_4_dir) / FROZEN_DETECTOR_FILENAME
    )
    point_path = (
        Path(operating_point) if operating_point else Path(v4_4_dir) / OPERATING_POINT_FILENAME
    )

    for label, path in (
        ("the v4.4 human report", report_path),
        ("the frozen detector artifact", detector_path),
        ("the frozen operating point", point_path),
    ):
        if not path.exists():
            raise typer.BadParameter(
                f"{label} ({path}) is absent. The final bank stays sealed until all three "
                "exist: a human result, the detector it validated, and the operating point "
                "that detector was frozen at."
            )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if str(report.get("schema")) != HUMAN_REPORT_SCHEMA:
        raise typer.BadParameter(
            f"{report_path} carries schema {report.get('schema')!r}, not "
            f"{HUMAN_REPORT_SCHEMA!r}."
        )
    if not report.get("reportable"):
        raise typer.BadParameter(
            f"{report_path} is not reportable: its sample was drawn from the v4.4 bundle "
            "alone. A human validation that never saw a fresh row cannot authorise the "
            "final bank."
        )
    if not report.get("passed"):
        raise typer.BadParameter(
            f"{report_path} records a FAILED human gate: {report.get('failures')}. The "
            "final bank stays sealed. Diagnose on the engineering surface and create a new "
            "detector version with a new fresh bank; the bound is not weakened to fit the "
            "result, and the humans are not re-run until they agree."
        )
    if report.get("provenance_failures"):
        raise typer.BadParameter(
            f"{report_path} records provenance failures: {report.get('provenance_failures')}."
        )

    detector = json.loads(detector_path.read_text(encoding="utf-8"))
    point = json.loads(point_path.read_text(encoding="utf-8"))
    if detector.get("operating_point_sha256") != point.get("operating_point_sha256"):
        raise typer.BadParameter(
            "the frozen detector was not built from this operating point. Its thresholds "
            "would be a different pair than the one the gate is about to be run at."
        )
    for name in ("tau_answer", "tau_partial"):
        if detector.get(name) is None:
            raise typer.BadParameter(
                f"the frozen detector carries no {name}. A detector missing one of its two "
                "thresholds is not frozen: the runtime would supply a default."
            )

    opening = Path(v4_2_dir) / FINAL_OPENING_RECORD_FILENAME
    if opening.exists():
        raise typer.BadParameter(
            f"{opening} exists: the final bank has already been opened once. There is no "
            "second opening; preserve the first result and create a new bank."
        )

    manifest_path = Path(v4_1_dir) / GATE_BANK_MANIFEST_FILENAME
    budget_path = Path(v4_2_dir) / FINAL_BUDGET_FILENAME
    for label, path in (("manifest", manifest_path), ("budget", budget_path)):
        if not path.exists():
            raise typer.BadParameter(f"the final bank {label} ({path}) is absent.")

    record: dict = {
        "schema": UNSEAL_RECORD_SCHEMA,
        "protocol": V4_4_PROTOCOL,
        "unsealed": True,
        "unsealed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "human_report": {
            "file": str(report_path),
            "sha256": _sha_file(report_path),
            "passed": True,
            "reportable": True,
            "human_human_kappa": (report.get("human_human") or {}).get("kappa"),
            "model_consensus_macro_f1": (report.get("model_consensus_vs_human") or {}).get(
                "macro_f1"
            ),
        },
        "frozen_detector": {
            "file": str(detector_path),
            "sha256": _sha_file(detector_path),
            "frozen_detector_sha256": detector.get("frozen_detector_sha256"),
            "tau_answer": detector.get("tau_answer"),
            "tau_partial": detector.get("tau_partial"),
            "protected_store_fingerprint": detector.get("protected_store_fingerprint"),
        },
        "operating_point": {
            "file": str(point_path),
            "sha256": _sha_file(point_path),
            "operating_point_sha256": point.get("operating_point_sha256"),
        },
        "final_bank": {
            "final_bank_manifest": {
                "file": str(manifest_path),
                "sha256": _sha_file(manifest_path),
            },
            "final_bank_budget": {"file": str(budget_path), "sha256": _sha_file(budget_path)},
            "not_yet_opened": True,
        },
        "authorises": (
            "exactly one `graph-detector-v4-2-build-bank --bank final`, followed by exactly "
            "one final gate at the thresholds bound above. No retraining, no recalibration, "
            "no new seed, no threshold change."
        ),
        "does_not_authorise": (
            "a second opening, a rebuilt bank after a failed gate, or any change to the "
            "frozen detector. If the final gate fails, the detector is not deployable and "
            "the failure is preserved."
        ),
    }
    atomic_json(Path(v4_2_dir) / UNSEAL_RECORD_FILENAME, record)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(Path(v4_2_dir) / UNSEAL_RECORD_FILENAME),
                "unsealed": True,
                "human_report": str(report_path),
                "human_human_kappa": record["human_report"]["human_human_kappa"],
                "tau_answer": detector.get("tau_answer"),
                "tau_partial": detector.get("tau_partial"),
                "next": "graph-detector-v4-2-build-bank --bank final (once)",
            }
        )
    )
