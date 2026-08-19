"""``rdl graph-detector-v4-4-fresh-label-report`` -- both label axes over a fresh partition.

The original v4.4 label report is hard-wired to the 1,733-row bundle and the 600-row
calibration panel. Neither exists for the fresh engineering audit, and neither should:

* the bundle is the TRAINING population, and its labels are the ones the checkpoint was
  fitted to. Gating the frozen detector against them would be reporting the training set.
* the 600-row calibration panel belongs to the original label authority. Reusing it as a
  fresh-audit gate would certify the fresh rubric with agreement measured on other rows.

So this is a separate command over the same machinery: two blind passes, adjudication,
two reference passes, reference adjudication, and one labelled audit per partition.

Run it once per partition, and the two runs are not interchangeable. ``development``
selects ``tau_answer`` and ``tau_partial``; ``heldout`` is opened once against the
already-frozen detector. They are separate files at every stage so that "we labelled the
audit" can never quietly mean "we labelled the half that chose the threshold".

The agreement gate here
-----------------------
The panel gates cannot apply -- there is no balanced panel -- so what is enforced is the
part that transfers: **zero** missing, malformed, truncated, inconsistent-hierarchy and
unresolved rows on both axes, plus the same 0.70 kappa / 0.85 raw-agreement bounds reported
on the natural fresh mixture. The kappa is reported and gated, and the mixture it is
computed on is stated beside it, because a kappa on a skewed mixture is a number that moves
with prevalence -- exactly v4.3's failure. If the fresh mixture is heavily one class, the
report says so rather than letting a high raw agreement stand in for a rubric that works.

The labelled audit this writes carries ``answer_attempt`` and **no reference answer**. The
reference labels live in their own file next to it. That is not tidiness: the labelled
audit is the object handed to the detector for scoring, and a reference answer inside it
would be one refactor away from a tokenizer input.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_1 import cohens_kappa, raw_agreement
from ..eval.detector_v4_4 import (
    PROMPT_VERSION,
    V4_4_PROTOCOL,
    evaluate_panel_gates,
    per_class_agreement,
    sha256_text,
)
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_4_bundle import DEFAULT_V4_4_DIR
from .detector_v4_4_fresh import (
    FINAL_AUDIT_FILENAME,
    FINAL_PARTITION,
    FINAL_REFERENCE_KEY_FILENAME,
    FRESH_AUDIT_FILENAME,
    FRESH_REFERENCE_KEY_FILENAME,
    LABELLABLE_PARTITIONS,
)
from .detector_v4_4_judge import output_names_v4_4
from .detector_v4_4_report import LABELS, REFERENCE_LABELS

__all__ = [
    "FRESH_LABELLED_SCHEMA",
    "FRESH_LABEL_REPORT_SCHEMA",
    "detector_v4_4_fresh_label_report",
    "fresh_adjudicated_filename",
    "fresh_labelled_filename",
    "judge_dir_for",
]

FRESH_LABELLED_SCHEMA = "graph-detector-v4-4-fresh-labelled-audit-v1"
FRESH_LABEL_REPORT_SCHEMA = "graph-detector-v4-4-fresh-label-report-v1"

# The same bounds as the blind axis elsewhere in v4.4. They are named here rather than
# imported from PANEL_GATES because PANEL_GATES is about the 200/200/200 panel and these
# are about a natural mixture -- same numbers, different denominator, and conflating the
# two is how the panel's authority would get borrowed for a draw it never covered.
FRESH_LABEL_GATES: dict[str, tuple[str, float]] = {
    "kappa": (">=", 0.70),
    "raw_agreement": (">=", 0.85),
    "n_malformed": ("==", 0.0),
    "n_missing": ("==", 0.0),
    "n_truncated": ("==", 0.0),
    "n_inconsistent_hierarchy": ("==", 0.0),
    "n_unresolved": ("==", 0.0),
}


def fresh_labelled_filename(partition: str) -> str:
    return f"DETECTOR_V4_4_FRESH_LABELLED_{partition}.json"


def fresh_report_filename(partition: str) -> str:
    return f"DETECTOR_V4_4_FRESH_LABEL_REPORT_{partition}.json"


def fresh_adjudicated_filename(partition: str, axis: str = "blind") -> str:
    return f"V4_4_FRESH_{partition}_{axis.upper()}_ADJUDICATED.jsonl"


def fresh_disagreement_filename(partition: str, axis: str = "blind") -> str:
    return f"V4_4_FRESH_{partition}_{axis.upper()}_DISAGREEMENTS.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pass_table(rows: Sequence[Mapping], field: str) -> tuple[dict[str, str], list[str], list[str]]:
    labels: dict[str, str] = {}
    malformed: list[str] = []
    inconsistent: list[str] = []
    for row in rows:
        audit_id = str(row["audit_id"])
        if row.get("inconsistent_hierarchy"):
            inconsistent.append(audit_id)
        if row.get("malformed") or row.get(field) is None:
            malformed.append(audit_id)
            continue
        labels[audit_id] = str(row[field])
    return labels, sorted(malformed), sorted(inconsistent)


def judge_dir_for(out_dir: Path, partition: str) -> Path:
    """Where one fresh partition's judge outputs live.

    A DIRECTORY, not a filename prefix, and that is the whole design. The judge's ``--pass``
    accepts ``blind`` or ``reference`` and nothing else -- ``run_rows_v4_4`` selects the
    reference prompt by comparing that string exactly -- so a scheme that encoded the
    partition in the pass name would be refused outright, and a near-miss like
    ``fresh-development-reference`` would silently run the BLIND prompt and produce a
    reference file with no reference answer in it.

    So the pass names stay the two the judge knows, and development and heldout are kept
    apart by writing into different directories.
    """
    return Path(out_dir) / "fresh" / partition


def _axis(
    *,
    out: Path,
    judge_dir: Path,
    partition: str,
    axis: str,
    field: str,
    legal: Sequence[str],
    rows_of: Mapping[str, Mapping],
    expected_ids: set[str],
    pass_name: str,
    adjudication: Path | None,
    reference_answer_of: Mapping[str, str] | None,
    restrict_to: set[str] | None = None,
) -> tuple[dict[str, str], dict]:
    """One label axis, closed. ``({audit_id: label}, block)``.

    Shared by the blind and reference axes because the two differ in exactly three ways --
    which field, which legal values, and whether the disagreement file may show the
    reference answer -- and duplicating 120 lines to express three differences is how the
    two axes drift into having different zero-tolerance rules.
    """
    paths = {
        judge: judge_dir
        / output_names_v4_4(judge=judge, pass_name=pass_name, run_id="", reportable=True)["output"]
        for judge in ("A", "B")
    }
    missing_files = [str(p) for p in paths.values() if not p.exists()]
    if missing_files:
        raise typer.BadParameter(
            f"{missing_files} are absent, so the {axis} axis has not been closed for "
            f"{partition}. Both passes must be frozen before agreement means anything. The "
            f"judge writes them with --out-dir {judge_dir} --pass {pass_name}."
        )
    rows_a, rows_b = _read_jsonl(paths["A"]), _read_jsonl(paths["B"])
    a, malformed_a, inconsistent_a = _pass_table(rows_a, field)
    b, malformed_b, inconsistent_b = _pass_table(rows_b, field)

    versions = {str(r.get("prompt_version")) for r in (*rows_a, *rows_b)}
    if versions != {PROMPT_VERSION}:
        raise typer.BadParameter(
            f"the {axis} passes carry prompt versions {sorted(versions)}, expected "
            f"{PROMPT_VERSION!r}. Agreement computed across a prompt change measures the "
            "change."
        )

    judged = set(a) & set(b)
    scored_ids = sorted(judged & restrict_to) if restrict_to is not None else sorted(judged)
    forced = sorted(judged - set(scored_ids))

    missing = sorted(expected_ids - {str(r["audit_id"]) for r in rows_a}) + sorted(
        expected_ids - {str(r["audit_id"]) for r in rows_b}
    )
    truncated = sorted(
        {str(r["audit_id"]) for r in (*rows_a, *rows_b) if r.get("prompt_truncated")}
    )
    malformed = sorted(set(malformed_a) | set(malformed_b))
    inconsistent = sorted(set(inconsistent_a) | set(inconsistent_b))

    disagreements = sorted(i for i in scored_ids if a[i] != b[i])
    out.mkdir(parents=True, exist_ok=True)
    disagreement_path = out / fresh_disagreement_filename(partition, axis)
    disagreement_path.write_text(
        "".join(
            dumps_canonical(
                {
                    "audit_id": i,
                    "conditioning_question": rows_of[i]["conditioning_question"],
                    "subject_aliases": rows_of[i]["subject_aliases"],
                    "candidate_text": rows_of[i]["candidate_text"],
                    # The blind file passes None here and the key is omitted entirely.
                    **(
                        {"reference_answer": reference_answer_of.get(i, "")}
                        if reference_answer_of is not None
                        else {}
                    ),
                    "label_1": min(a[i], b[i]),
                    "label_2": max(a[i], b[i]),
                    field: None,
                }
            )
            + "\n"
            for i in disagreements
        ),
        encoding="utf-8",
    )

    decisions = {
        str(r["audit_id"]): str(r[field])
        for r in (_read_jsonl(Path(adjudication)) if adjudication else [])
        if r.get(field) in legal
    }
    unresolved = sorted(set(disagreements) - set(decisions))

    measured: dict[str, float | None] = {
        "kappa": cohens_kappa([a[i] for i in scored_ids], [b[i] for i in scored_ids]),
        "raw_agreement": raw_agreement([a[i] for i in scored_ids], [b[i] for i in scored_ids]),
        "n_malformed": float(len(malformed)),
        "n_missing": float(len(set(missing))),
        "n_truncated": float(len(truncated)),
        "n_inconsistent_hierarchy": float(len(inconsistent)),
        "n_unresolved": float(len(unresolved)),
    }
    verdicts, failures = evaluate_panel_gates(measured, FRESH_LABEL_GATES)
    if failures:
        raise typer.BadParameter(
            f"the {partition} {axis} axis is not closed: "
            + "; ".join(failures)
            + f". The disagreements are in {disagreement_path}. Adjudicate every one and "
            "re-run. A fresh-audit gate scored against half-resolved labels is not a gate."
        )

    final = {i: (decisions.get(i) or a[i]) for i in scored_ids}
    if axis == "reference":
        final.update(dict.fromkeys(forced, "UNCERTAIN"))
    adjudicated_path = out / fresh_adjudicated_filename(partition, axis)
    adjudicated_path.write_text(
        "".join(
            dumps_canonical(
                {
                    "audit_id": i,
                    field: final[i],
                    "source": (
                        "forced_uncertain_no_reference_answer"
                        if i in set(forced)
                        else "adjudicated" if i in decisions else "both_judges_agreed"
                    ),
                }
            )
            + "\n"
            for i in sorted(final)
        ),
        encoding="utf-8",
    )

    block = {
        "axis": axis,
        "field": field,
        "n_judged_by_both": len(judged),
        "n_scored": len(scored_ids),
        "n_forced_uncertain": len(forced),
        "gates": verdicts,
        "passed": True,
        "n_disagreements": len(disagreements),
        "n_decisions": len(decisions),
        "n_unresolved": 0,
        "distribution": dict(sorted(Counter(final.values()).items())),
        "per_class": per_class_agreement(
            {i: a[i] for i in scored_ids},
            {i: b[i] for i in scored_ids},
            classes=tuple(legal),
        ),
        "judge_pass_sha256": {
            judge: hashlib.sha256(path.read_bytes()).hexdigest()
            for judge, path in sorted(paths.items())
        },
        "disagreement_file": str(disagreement_path),
        "disagreement_sha256": hashlib.sha256(disagreement_path.read_bytes()).hexdigest(),
        "adjudicated_file": str(adjudicated_path),
        "adjudicated_sha256": hashlib.sha256(adjudicated_path.read_bytes()).hexdigest(),
        "adjudication_file": str(adjudication) if adjudication else None,
        "adjudication_sha256": (
            sha256_text(Path(adjudication).read_text(encoding="utf-8")) if adjudication else None
        ),
    }
    return final, block


def detector_v4_4_fresh_label_report(
    partition: str = typer.Option(..., "--partition", help="development | heldout"),
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    audit: Path = typer.Option(None, "--audit", help="DETECTOR_V4_4_FRESH_AUDIT.json"),
    reference_key: Path = typer.Option(None, "--reference-key"),
    adjudication: Path = typer.Option(
        None, "--adjudication", help="blind adjudication decisions, JSONL."
    ),
    reference_adjudication: Path = typer.Option(
        None, "--reference-adjudication", help="reference adjudication decisions, JSONL."
    ),
    blind_only: bool = typer.Option(
        False, "--blind-only", help="close the blind axis only. Run this first."
    ),
) -> None:
    """Fold the fresh partition's judge passes into one labelled audit."""
    if partition not in LABELLABLE_PARTITIONS:
        raise typer.BadParameter(f"--partition must be one of {list(LABELLABLE_PARTITIONS)}")
    out = Path(out_dir)
    # The final bank writes its own audit and its own sealed key, so that labelling the
    # sealed bank can never read -- or overwrite -- the engineering one.
    is_final = partition == FINAL_PARTITION
    default_audit = FINAL_AUDIT_FILENAME if is_final else FRESH_AUDIT_FILENAME
    default_key = FINAL_REFERENCE_KEY_FILENAME if is_final else FRESH_REFERENCE_KEY_FILENAME
    audit_path = Path(audit) if audit else out / default_audit
    key_path = Path(reference_key) if reference_key else out / default_key
    if not audit_path.exists():
        raise typer.BadParameter(
            f"{audit_path} is absent; run `rdl graph-detector-v4-4-fresh-audit` first."
        )
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    rows = [r for r in audit_payload.get("rows", ()) if str(r.get("partition")) == partition]
    if not rows:
        raise typer.BadParameter(f"{audit_path} carries no rows in partition {partition!r}.")
    rows_of = {str(r["audit_id"]): r for r in rows}
    expected_ids = set(rows_of)
    judge_dir = judge_dir_for(out, partition)

    blind_final, blind_block = _axis(
        out=out,
        judge_dir=judge_dir,
        partition=partition,
        axis="blind",
        field="answer_attempt",
        legal=LABELS,
        rows_of=rows_of,
        expected_ids=expected_ids,
        pass_name="blind",
        adjudication=Path(adjudication) if adjudication else None,
        reference_answer_of=None,
    )

    report: dict = {
        "schema": FRESH_LABEL_REPORT_SCHEMA,
        "protocol": V4_4_PROTOCOL,
        "prompt_version": PROMPT_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partition": partition,
        "audit": str(audit_path),
        "audit_sha256": audit_payload.get("audit_sha256"),
        "bank_sha256": audit_payload.get("bank_sha256"),
        "n_rows": len(rows),
        "blind_axis": blind_block,
        "mixture": {
            "by_population": dict(sorted(Counter(r["population"] for r in rows).items())),
            "by_bank_bucket": dict(sorted(Counter(r["bank_bucket"] for r in rows).items())),
            "label_distribution": blind_block["distribution"],
            "why_reported": (
                "kappa moves with prevalence. A fresh mixture that is 80% one class can "
                "clear 0.85 raw agreement on a rubric nobody could apply, so the mixture "
                "the number was computed on is stated beside the number."
            ),
        },
        "not_the_calibration_panel": (
            "the 600-row balanced panel belongs to the ORIGINAL v4.4 label authority. It is "
            "not reused here: certifying the fresh rubric with agreement measured on the "
            "training population's rows would be a gate about other rows."
        ),
    }

    if blind_only:
        atomic_json(out / fresh_report_filename(partition), report)
        typer.echo(
            dumps_canonical(
                {
                    "wrote": str(out / fresh_report_filename(partition)),
                    "partition": partition,
                    "blind_kappa": blind_block["gates"]["kappa"]["measured"],
                    "blind_raw_agreement": blind_block["gates"]["raw_agreement"]["measured"],
                    "n_disagreements": blind_block["n_disagreements"],
                    "next": (
                        "adjudicate the blind disagreements WITHOUT any reference answer, "
                        "then run the reference passes."
                    ),
                }
            )
        )
        return

    if adjudication is None:
        raise typer.BadParameter("pass --blind-only or --adjudication FILE")

    if not key_path.exists():
        raise typer.BadParameter(
            f"{key_path} is absent, so the reference axis cannot run. A fresh audit built "
            "with --skip-reference-key is not reportable."
        )
    key_rows = json.loads(key_path.read_text(encoding="utf-8")).get("rows", {})
    reference_answer_of = {
        i: str(key_rows.get(i, {}).get("reference_answer", "")) for i in expected_ids
    }
    has_reference = {i for i in expected_ids if reference_answer_of[i].strip()}

    reference_final, reference_block = _axis(
        out=out,
        judge_dir=judge_dir,
        partition=partition,
        axis="reference",
        field="reference_content",
        legal=REFERENCE_LABELS,
        rows_of=rows_of,
        expected_ids=expected_ids,
        pass_name="reference",
        adjudication=Path(reference_adjudication) if reference_adjudication else None,
        reference_answer_of=reference_answer_of,
        restrict_to=has_reference,
    )
    report["reference_axis"] = reference_block
    report["reference_key_sha256"] = hashlib.sha256(key_path.read_bytes()).hexdigest()
    report["attempt_vs_content"] = dict(
        sorted(
            Counter(
                f"{blind_final[i]}|{reference_final[i]}"
                for i in sorted(expected_ids)
                if i in blind_final and i in reference_final
            ).items()
        )
    )

    # ------------------------------------------------------- the labelled audit --
    # Explicit allow-list. The reference answer is NOT here, and neither is anything the
    # reference key holds: this object is what the detector is handed to score.
    labelled_rows = [
        {
            "audit_id": r["audit_id"],
            "conditioning_question": r["conditioning_question"],
            "subject_aliases": r["subject_aliases"],
            "candidate_text": r["candidate_text"],
            "answer_attempt": blind_final[r["audit_id"]],
            "label": blind_final[r["audit_id"]],
            "population": r["population"],
            "partition": r["partition"],
            "concept_id": r["concept_id"],
            # `enrich_nonattempt` writes `sampling_stratum`; the gate reads `stratum`.
            # Bridged here rather than in the gate so the design weights and the name the
            # metrics use stay in one object.
            "stratum": r.get("sampling_stratum", ""),
            "inclusion_probability": r.get("inclusion_probability"),
            "design_weight": r.get("design_weight"),
        }
        for r in rows
    ]
    labelled = {
        "schema": FRESH_LABELLED_SCHEMA,
        "protocol": V4_4_PROTOCOL,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partition": partition,
        "audit_sha256": audit_payload.get("audit_sha256"),
        "blind_adjudicated_sha256": blind_block["adjudicated_sha256"],
        "reference_adjudicated_sha256": reference_block["adjudicated_sha256"],
        "n_rows": len(labelled_rows),
        "label_distribution": dict(
            sorted(Counter(r["answer_attempt"] for r in labelled_rows).items())
        ),
        "carries_reference_answer": False,
        "why_no_reference_answer": (
            "this object is handed to the detector for scoring. The reference labels are in "
            f"{fresh_adjudicated_filename(partition, 'reference')} and the answers "
            "themselves only in the sealed key."
        ),
        "rows": labelled_rows,
    }
    labelled["labelled_sha256"] = sha256_text(
        dumps_canonical({k: v for k, v in labelled.items() if k != "built_at"})
    )
    atomic_json(out / fresh_labelled_filename(partition), labelled)
    report["labelled_audit"] = {
        "file": str(out / fresh_labelled_filename(partition)),
        "sha256": labelled["labelled_sha256"],
        "n_rows": len(labelled_rows),
    }
    atomic_json(out / fresh_report_filename(partition), report)
    typer.echo(
        dumps_canonical(
            {
                "wrote": [
                    str(out / fresh_report_filename(partition)),
                    str(out / fresh_labelled_filename(partition)),
                ],
                "partition": partition,
                "blind_kappa": blind_block["gates"]["kappa"]["measured"],
                "reference_kappa": reference_block["gates"]["kappa"]["measured"],
                "label_distribution": labelled["label_distribution"],
                "attempt_vs_content": report["attempt_vs_content"],
                "labelled_sha256": labelled["labelled_sha256"],
            }
        )
    )
