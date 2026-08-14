"""``rdl graph-detector-v4-2-label-report`` — adjudicate the two model judges.

Writes ``DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json``. **Not**
``LABEL_ALIGNMENT_REPORT.json`` — that name belongs to the human audit of
``DETECTOR_V4_1_PROTOCOL.md`` and no v4.2 command writes it. Feeding model output into
the v4.1 path would leave an artifact whose schema says "human judges" and whose contents
are two LLMs, and every downstream reader would inherit the mistake.

Two runs, one command
---------------------
Run it once with no ``--adjudication``: it computes pre-adjudication κ, writes
``V4_2_DISAGREEMENTS_{BLIND,REFERENCE}.jsonl`` with the fields to resolve left blank, and
exits non-zero because unresolved disagreements are a gate condition. Resolve them blind —
the file carries the same question and candidate the judges saw and nothing else — and run
it again with ``--adjudication``.

κ is computed on the raw judge files, before adjudication, for the reason v4.1 gives:
agreement measured after adjudication describes the adjudicator.

What is excluded from κ, and why
--------------------------------
Reference-pass rows the runner forced to ``UNCERTAIN`` because the reference answer was
empty (300 of 1,019 — the retain stratum) are excluded from the reference-pass agreement.
They are still labels and still adjudicated; they are simply not evidence that two judges
agreed, because neither judge was asked. Including them would add 300 trivially identical
pairs to a κ that is supposed to measure whether the rubric is legible.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import typer

from ..eval.detector_v4_1 import AUDIT_FIELDS, adjudicate
from ..eval.detector_v4_2 import (
    BLIND_FIELDS,
    JUDGES,
    REFERENCE_FIELDS,
    disagreements,
    model_alignment_report,
)
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT
from .detector_v4_2_llm_judge import (
    DEFAULT_V4_2_OUT,
    OUTPUT_FILENAME,
    RUN_FILENAME,
    _read_jsonl,
    _write_jsonl,
)
from .detector_v4_label_audit import KEY_FILENAME

__all__ = ["detector_v4_2_label_report"]

ADJUDICATED_FILENAME = "V4_2_ADJUDICATED.jsonl"
DISAGREEMENT_FILENAME = "V4_2_DISAGREEMENTS_{pass_upper}.jsonl"
REPORT_FILENAME = "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json"

# A row the runner forced rather than asked for. It is a label; it is not agreement.
FORCED_SOURCES = frozenset({"protocol_rule_empty_reference"})


def _load_overlay(path: Path) -> tuple[dict[str, dict], dict[str, dict], int]:
    """``(labels, agreement_labels, n_failed)`` from one judge/pass overlay.

    ``labels`` carries every resolved value; ``agreement_labels`` carries only values the
    model actually produced, and is what κ is computed on.
    """
    labels: dict[str, dict] = {}
    agreement: dict[str, dict] = {}
    n_failed = 0
    for row in _read_jsonl(path):
        audit_id = str(row.get("audit_id", ""))
        if not audit_id:
            raise typer.BadParameter(f"{path}: a row has no audit_id")
        if row.get("source") == "failed" or row.get("error"):
            n_failed += 1
            continue
        forced = row.get("source") in FORCED_SOURCES
        for field in AUDIT_FIELDS:
            value = row.get(field)
            if value is None or value == "":
                continue
            if value not in AUDIT_FIELDS[field]["values"]:
                raise typer.BadParameter(
                    f"{path}: {audit_id}.{field} = {value!r}; "
                    f"allowed: {list(AUDIT_FIELDS[field]['values'])}"
                )
            labels.setdefault(audit_id, {})[field] = value
            if not forced:
                agreement.setdefault(audit_id, {})[field] = value
    return labels, agreement, n_failed


def _merge(into: dict[str, dict], other: Mapping[str, Mapping]) -> dict[str, dict]:
    for audit_id, values in other.items():
        into.setdefault(audit_id, {}).update(values)
    return into


def detector_v4_2_label_report(
    judge_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--judge-dir"),
    audit_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--audit-dir"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
    adjudication: Path | None = typer.Option(
        None, "--adjudication", help="the resolved disagreement file(s), JSONL"
    ),
    require_reference_pass: bool = typer.Option(True, "--require-reference-pass/--blind-pass-only"),
) -> None:
    """Adjudicate the two model judges and write the v4.2 model-label report."""
    key_path = audit_dir / KEY_FILENAME
    if not key_path.exists():
        raise typer.BadParameter(f"{key_path} is absent; run `rdl graph-detector-v4-label-audit`")
    key = json.loads(key_path.read_text(encoding="utf-8")).get("rows", {})

    wanted = ["blind"] + (["reference"] if require_reference_pass else [])
    labels: dict[str, dict[str, dict]] = {"A": {}, "B": {}}
    agreement: dict[str, dict[str, dict]] = {"A": {}, "B": {}}
    runs: list[dict] = []
    n_failed = 0

    for pass_name in wanted:
        for role in JUDGES:
            overlay = judge_dir / OUTPUT_FILENAME.format(judge=role, pass_upper=pass_name.upper())
            if not overlay.exists():
                raise typer.BadParameter(
                    f"{overlay} is absent. Run `rdl graph-detector-v4-2-llm-judge "
                    f"--judge {role} --pass {pass_name}` first, in the order frozen in "
                    f"DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md section 7."
                )
            row_labels, row_agreement, failed = _load_overlay(overlay)
            _merge(labels[role], row_labels)
            _merge(agreement[role], row_agreement)
            n_failed += failed
            run_path = judge_dir / RUN_FILENAME.format(judge=role, pass_upper=pass_name.upper())
            if run_path.exists():
                runs.append(json.loads(run_path.read_text(encoding="utf-8")))

    # ---------------------------------------------------------- the disagreements --
    written: list[str] = []
    for pass_name, fields in (("blind", BLIND_FIELDS), ("reference", REFERENCE_FIELDS)):
        if pass_name not in wanted:
            continue
        rows = disagreements(agreement["A"], agreement["B"], fields=fields)
        path = output_dir / DISAGREEMENT_FILENAME.format(pass_upper=pass_name.upper())
        _write_jsonl(path, rows)
        written.append(f"{path} ({len(rows)} rows)")

    resolutions: dict[str, dict] = {}
    if adjudication:
        for row in _read_jsonl(adjudication):
            audit_id = str(row.get("audit_id", ""))
            entry = resolutions.setdefault(audit_id, {})
            for field in AUDIT_FIELDS:
                value = row.get(field)
                if value in (None, ""):
                    continue
                if value not in AUDIT_FIELDS[field]["values"]:
                    raise typer.BadParameter(
                        f"{adjudication}: {audit_id}.{field} = {value!r}; "
                        f"allowed: {list(AUDIT_FIELDS[field]['values'])}"
                    )
                entry[field] = value

    adjudicated, unresolved = adjudicate(labels["A"], labels["B"], resolutions)
    _write_jsonl(
        output_dir / ADJUDICATED_FILENAME,
        [
            {**row, "text_sha256": key.get(row["audit_id"], {}).get("text_sha256", "")}
            for row in adjudicated
        ],
    )

    n_expected = max(
        (int(run.get("n_rows_in_input", 0)) for run in runs),
        default=None,
    )
    n_returned = len(set(labels["A"]) & set(labels["B"]))
    report = model_alignment_report(
        adjudicated,
        key,
        judge_a=agreement["A"],
        judge_b=agreement["B"],
        unresolved=unresolved,
        runs=runs,
        n_malformed=n_failed,
        n_missing=max(0, (n_expected or n_returned) - n_returned),
        n_expected_rows=n_expected,
    )
    report["inputs"] = {
        "judge_dir": str(judge_dir),
        "key": str(key_path),
        "adjudication": str(adjudication) if adjudication else None,
        "passes": wanted,
    }
    report["excluded_from_agreement"] = {
        "protocol_forced_reference_rows": sum(
            1
            for run in runs
            if run.get("pass") == "reference"
            for _ in range(int(run.get("n_forced_by_protocol_rule", 0)))
        ),
        "why": (
            "rows whose reference answer was empty were labelled UNCERTAIN by rule, not "
            "by a judge. They remain labels and are adjudicated; they are excluded from "
            "the reference-pass agreement because two judges neither of which was asked "
            "are not two judges who agreed."
        ),
    }

    out = output_dir / REPORT_FILENAME
    atomic_json(out, report)

    typer.echo(f"wrote {out}")
    for line in written:
        typer.echo(f"wrote {line}")
    typer.echo("")
    typer.echo(f"judge population: {report['judge_population']}")
    typer.echo(f"human_grounded: {report['human_grounded']}")
    typer.echo(f"publication_label_valid: {report['publication_label_valid']}")
    typer.echo(f"verdict: {report['verdict']}")
    for gate in report["gates"]:
        mark = "PASS" if gate["passed"] else ("n/a " if gate["passed"] is None else "FAIL")
        typer.echo(
            f"  [{mark}] {gate['gate']}: {gate['measured']} "
            f"(need {gate['comparison']} {gate['bound']})"
        )
    raise typer.Exit(0 if report["all_gates_passed"] else 1)
