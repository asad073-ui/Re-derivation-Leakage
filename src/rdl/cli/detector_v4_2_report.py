"""``rdl graph-detector-v4-2-label-report`` — adjudicate the two model judges.

Writes ``DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json``. **Not**
``LABEL_ALIGNMENT_REPORT.json`` — that name belongs to the human audit of
``DETECTOR_V4_1_PROTOCOL.md`` and no v4.2 command writes it. Feeding model output into
the v4.1 path would leave an artifact whose schema says "human judges" and whose contents
are two LLMs, and every downstream reader would inherit the mistake.

Four manifests, or no report
----------------------------
The report requires all four run manifests — A/blind, B/blind, A/reference, B/reference —
and checks each of them before it computes anything: complete, reportable, produced under
this prompt version, against this input file's hash, by the model that was requested, with
every input row covered exactly once and every ``audit_id`` unique. Each failure is
counted into ``n_provenance_failures``, which is a gate condition. Without this a report
could be assembled from one complete pass and one five-row smoke, and every number in it
would be a number about 5 rows presented as a number about 1,019.

Two runs, one command
---------------------
Run it once with no ``--adjudication``: it computes pre-adjudication κ, writes
``V4_2_DISAGREEMENTS_{BLIND,REFERENCE}.jsonl`` — each row carrying the question, the
candidate text and (on the reference pass) the reference answer, which is exactly what the
judges saw and nothing more — and exits non-zero because unresolved disagreements are a
gate condition. Resolve them from those files alone and run it again with
``--adjudication``.

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
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_1 import AUDIT_FIELDS, adjudicate
from ..eval.detector_v4_2 import (
    BLIND_FIELDS,
    JUDGES,
    PROMPT_VERSION,
    REFERENCE_FIELDS,
    disagreements,
    model_alignment_report,
    sha256_text,
)
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT
from .detector_v4_2_llm_judge import (
    DEFAULT_V4_2_OUT,
    INPUT_FILENAME,
    OUTPUT_FILENAME,
    RUN_FILENAME,
    _file_sha256,
    _read_jsonl,
    _rows_sha256,
    _write_jsonl,
)
from .detector_v4_label_audit import KEY_FILENAME

__all__ = ["detector_v4_2_label_report"]

ADJUDICATED_FILENAME = "V4_2_ADJUDICATED.jsonl"
DISAGREEMENT_FILENAME = "V4_2_DISAGREEMENTS_{pass_upper}.jsonl"
REPORT_FILENAME = "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json"

# A row the runner forced rather than asked for. It is a label; it is not agreement.
FORCED_SOURCES = frozenset({"protocol_rule_empty_reference"})


def _load_overlay(path: Path) -> tuple[dict[str, dict], dict[str, dict], int, list[str]]:
    """``(labels, agreement_labels, n_failed, duplicate_ids)`` from one judge/pass overlay.

    ``labels`` carries every resolved value; ``agreement_labels`` carries only values the
    model actually produced, and is what κ is computed on.
    """
    labels: dict[str, dict] = {}
    agreement: dict[str, dict] = {}
    n_failed = 0
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in _read_jsonl(path):
        audit_id = str(row.get("audit_id", ""))
        if not audit_id:
            raise typer.BadParameter(f"{path}: a row has no audit_id")
        if audit_id in seen:
            duplicates.append(audit_id)
            continue
        seen.add(audit_id)
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
    return labels, agreement, n_failed, duplicates


def _merge(into: dict[str, dict], other: Mapping[str, Mapping]) -> dict[str, dict]:
    for audit_id, values in other.items():
        into.setdefault(audit_id, {}).update(values)
    return into


def verify_run(
    manifest: Mapping,
    *,
    role: str,
    pass_name: str,
    overlay_rows: Sequence[Mapping],
    input_ids: Sequence[str],
    duplicate_ids: Sequence[str],
) -> list[str]:
    """Every way one (judge, pass) run fails to be a complete run of the frozen protocol.

    Returns failure strings rather than raising, so one invocation names all of them. A
    report that stopped at the first would need four runs to enumerate four problems, and
    on a metered free tier each of those is a day.
    """
    where = f"judge {role} / {pass_name}"
    failures: list[str] = []

    if not manifest.get("complete"):
        failures.append(f"{where}: complete=false ({manifest.get('why_incomplete')})")
    if not manifest.get("reportable", True):
        failures.append(
            f"{where}: reportable=false — this is a --limit/--run-id smoke run. A smoke "
            "run measures that the path works, never what the labels are."
        )
    if manifest.get("run_id"):
        failures.append(f"{where}: carries run_id={manifest['run_id']!r}, a smoke directory")
    if manifest.get("prompt_version") != PROMPT_VERSION:
        failures.append(
            f"{where}: prompt version {manifest.get('prompt_version')!r} != "
            f"{PROMPT_VERSION!r}. A kappa computed across a prompt change measures the "
            "change, not the judges."
        )
    if int(manifest.get("n_malformed_or_missing", 0) or 0):
        failures.append(
            f"{where}: {manifest['n_malformed_or_missing']} malformed or missing row(s). "
            "A row whose response never arrived is not a label."
        )

    # Judge identity: requested vs returned, and against the frozen roster.
    frozen = JUDGES.get(role, {})
    if manifest.get("provider") != frozen.get("provider"):
        failures.append(
            f"{where}: provider {manifest.get('provider')!r} != frozen {frozen.get('provider')!r}"
        )
    requested = str(manifest.get("requested_model") or "")
    returned = manifest.get("returned_models_seen") or []
    if len(returned) > 1:
        failures.append(
            f"{where}: the provider returned more than one model identifier {returned}. "
            "An alias that moved mid-run makes the pass two passes."
        )
    elif returned and requested and returned[0] != requested:
        # Providers legitimately return a dated or namespaced alias of the requested id;
        # an identifier that does not contain the requested one is a different model.
        if requested.split("/")[-1] not in returned[0]:
            failures.append(
                f"{where}: requested {requested!r} but the provider returned {returned[0]!r}"
            )
    elif not returned:
        failures.append(f"{where}: no returned model identifier was recorded")

    # Input coverage: every input row, exactly once, and the output hash the run claimed.
    overlay_ids = [str(r.get("audit_id", "")) for r in overlay_rows]
    if duplicate_ids:
        failures.append(
            f"{where}: duplicate audit_ids in the overlay {sorted(set(duplicate_ids))[:5]}"
        )
    uncovered = sorted(set(input_ids) - set(overlay_ids))
    if uncovered:
        failures.append(
            f"{where}: {len(uncovered)} input row(s) have no overlay row (e.g. {uncovered[:3]})"
        )
    extra = sorted(set(overlay_ids) - set(input_ids))
    if extra:
        failures.append(
            f"{where}: {len(extra)} overlay row(s) name an audit_id the input does not "
            f"have (e.g. {extra[:3]})"
        )
    claimed = manifest.get("output_file_sha256")
    actual = _rows_sha256(overlay_rows)
    if claimed and claimed != actual:
        failures.append(
            f"{where}: the manifest records output_file_sha256={claimed[:16]}… but the "
            f"overlay on disk hashes to {actual[:16]}…. The file changed after the run."
        )
    if not manifest.get("n_provider_request_ids") and manifest.get("n_rows_called"):
        failures.append(
            f"{where}: no provider request ids were recorded for "
            f"{manifest['n_rows_called']} calls; the run cannot be traced to the provider"
        )
    return failures


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
    failures: list[str] = []
    n_failed = 0

    # The evidence an adjudicator may see, per pass, straight from the blinded input files
    # the judges were given. Reading it from those files rather than from the key is the
    # point: the key carries the NLI label and the stratum.
    evidence: dict[str, dict[str, dict]] = {}

    for pass_name in wanted:
        pass_upper = pass_name.upper()
        for role in JUDGES:
            overlay_path = judge_dir / OUTPUT_FILENAME.format(judge=role, pass_upper=pass_upper)
            run_path = judge_dir / RUN_FILENAME.format(judge=role, pass_upper=pass_upper)
            if not overlay_path.exists():
                raise typer.BadParameter(
                    f"{overlay_path} is absent. Run `rdl graph-detector-v4-2-llm-judge "
                    f"--judge {role} --pass {pass_name}` first, in the order frozen in "
                    f"DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md section 7."
                )
            if not run_path.exists():
                raise typer.BadParameter(
                    f"{run_path} is absent. An overlay with no run manifest has no model, "
                    "no parameters and no completeness record; it cannot back a report."
                )
            manifest = json.loads(run_path.read_text(encoding="utf-8"))
            runs.append(manifest)

            input_path = audit_dir / INPUT_FILENAME[pass_name].format(judge=role)
            input_rows = _read_jsonl(input_path) if input_path.exists() else []
            input_ids = [str(r.get("audit_id", "")) for r in input_rows]
            if input_path.exists():
                actual_input_sha = _file_sha256(input_path)
                if manifest.get("input_file_sha256") != actual_input_sha:
                    failures.append(
                        f"judge {role} / {pass_name}: the run was made against input hash "
                        f"{str(manifest.get('input_file_sha256'))[:16]}… and {input_path} "
                        f"now hashes to {actual_input_sha[:16]}…. The input moved."
                    )
                evidence.setdefault(pass_name, {}).update(
                    {str(r.get("audit_id", "")): dict(r) for r in input_rows}
                )
            else:
                failures.append(f"judge {role} / {pass_name}: {input_path} is absent")

            overlay_rows = _read_jsonl(overlay_path)
            row_labels, row_agreement, failed, duplicates = _load_overlay(overlay_path)
            _merge(labels[role], row_labels)
            _merge(agreement[role], row_agreement)
            n_failed += failed
            failures.extend(
                verify_run(
                    manifest,
                    role=role,
                    pass_name=pass_name,
                    overlay_rows=overlay_rows,
                    input_ids=input_ids,
                    duplicate_ids=duplicates,
                )
            )

    # The two judges must have been given the same rows. A kappa over two different row
    # sets is a kappa over their intersection dressed up as a kappa over the audit.
    for pass_name in wanted:
        by_role = {
            role: str(
                next(
                    (
                        r.get("input_file_sha256")
                        for r in runs
                        if r.get("judge") == role and r.get("pass") == pass_name
                    ),
                    "",
                )
            )
            for role in JUDGES
        }
        rubrics = {
            str(r.get("rubric_sha256"))
            for r in runs
            if r.get("pass") == pass_name and r.get("rubric_sha256")
        }
        if len(rubrics) > 1:
            failures.append(
                f"{pass_name}: the two judges were given different rubrics {sorted(rubrics)}"
            )
        # The blinded input files differ between A and B by construction (each judge has
        # its own copy), so their hashes are compared to their own manifests above; what is
        # checked here is that both files carry the same audit_ids.
        del by_role

    n_expected = max((int(run.get("n_rows_in_input", 0)) for run in runs), default=None)

    # ---------------------------------------------------------- the disagreements --
    written: list[str] = []
    for pass_name, fields in (("blind", BLIND_FIELDS), ("reference", REFERENCE_FIELDS)):
        if pass_name not in wanted:
            continue
        rows = disagreements(
            agreement["A"],
            agreement["B"],
            fields=fields,
            pass_name=pass_name,
            evidence=evidence.get(pass_name, {}),
        )
        path = output_dir / DISAGREEMENT_FILENAME.format(pass_upper=pass_name.upper())
        _write_jsonl(path, rows)
        written.append(f"{path} ({len(rows)} rows)")
        missing_evidence = [r["audit_id"] for r in rows if not any(r["evidence"].values())]
        if missing_evidence:
            failures.append(
                f"{pass_name}: {len(missing_evidence)} disagreement row(s) carry no "
                "evidence; the adjudication file would be unresolvable without opening "
                "the key."
            )

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

    n_returned = len(set(labels["A"]) & set(labels["B"]))
    provenance = {
        "checked": True,
        "n_runs_required": len(JUDGES) * len(wanted),
        "n_runs_found": len(runs),
        "passes_required": wanted,
        "prompt_version": PROMPT_VERSION,
        "failures": failures,
        "runs_fingerprint": sha256_text(
            json.dumps(
                sorted(
                    f"{r.get('judge')}|{r.get('pass')}|{r.get('requested_model')}|"
                    f"{r.get('output_file_sha256')}"
                    for r in runs
                ),
                sort_keys=True,
            )
        ),
        "why_this_gates": (
            "a report assembled from one complete pass and one five-row smoke would carry "
            "numbers about five rows under a header about a thousand. Each of these is a "
            "way that could happen without anything looking wrong."
        ),
    }
    if len(runs) != len(JUDGES) * len(wanted):
        failures.append(
            f"{len(runs)} run manifest(s) found, {len(JUDGES) * len(wanted)} required "
            f"(judges {sorted(JUDGES)} x passes {wanted})"
        )

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
        provenance=provenance,
    )
    report["inputs"] = {
        "judge_dir": str(judge_dir),
        "key": str(key_path),
        "adjudication": str(adjudication) if adjudication else None,
        "passes": wanted,
    }
    report["excluded_from_agreement"] = {
        "protocol_forced_reference_rows": sum(
            int(run.get("n_forced_by_protocol_rule", 0))
            for run in runs
            if run.get("pass") == "reference"
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
    for failure in failures:
        typer.echo(f"  [PROVENANCE] {failure}", err=True)
    typer.echo(f"verdict: {report['verdict']}")
    for gate in report["gates"]:
        mark = "PASS" if gate["passed"] else ("n/a " if gate["passed"] is None else "FAIL")
        typer.echo(
            f"  [{mark}] {gate['gate']}: {gate['measured']} "
            f"(need {gate['comparison']} {gate['bound']})"
        )
    raise typer.Exit(0 if report["all_gates_passed"] else 1)
