"""``rdl graph-detector-v4-4-label-report`` -- two v4.4 passes into one label authority.

Two invocations, in order, exactly as v4.3:

``--blind-only``
    Reads both blind passes, computes the **balanced-panel** gate and the full-mixture
    diagnostic, and writes the disagreements. The disagreement file carries the question,
    the aliases and the candidate and **never the reference answer** -- the researcher
    adjudicates blind, exactly as the judges did.

``--adjudication FILE [--reference-adjudication FILE] [--require-reference-pass]``
    Folds the adjudicated decisions in, freezes the blind labels, checks the adjudicated
    bundle's class minima, and -- once both reference passes exist -- computes
    reference-assisted kappa and the two-axis cross-tabulation.

The reference axis is adjudicated too, and separately
-----------------------------------------------------
``reference_content`` used to be reported from judge A wherever the two reference passes
disagreed. That is not an adjudication, it is a coin already flipped: the cross-tabulation
that GU-0048 needed would have been half judge A's opinion on exactly the rows the two
judges could not agree about. The reference axis now writes its own disagreement file,
takes its own ``--reference-adjudication``, and refuses to build an authority while any
reference row is missing, malformed, truncated or unresolved -- the same zero-tolerance the
blind axis has always had.

The two disagreement files are NOT symmetric, and deliberately so.
``V4_4_BLIND_DISAGREEMENTS.jsonl`` never carries the reference answer, because a researcher
who has seen it cannot un-see it and the blind label is the trainable one.
``V4_4_REFERENCE_DISAGREEMENTS.jsonl`` does carry it, because deciding "does this convey the
reference answer" without the reference answer is not a task. That file is an offline
diagnostic artifact and never reaches training or runtime.

What changed from v4.3, and why
-------------------------------
**The primary kappa is computed on the frozen 600-row panel, not the natural mixture.**
v4.3's 0.404 sat under 74.7% raw agreement because 65-81% of rows were one class. Balancing
the denominator is not a fix for that on its own -- it raises kappa mechanically -- so it is
paired with four other requirements that a merely-rebalanced protocol would fail: raw
agreement >= 0.85, per-class floors, the mixture kappa reported beside it, and the
``A=PARTIAL/B=ANSWER`` cell reported by question type. **The bound did not move.** It is
still 0.70. What moved is the distribution it is evaluated on, and that was fixed in the
panel file before any judge ran.

**Which panel is used is not negotiable at report time.** The report reads
``DETECTOR_V4_4_CALIBRATION_PANEL.json``, checks its ``bundle_sha256`` against the bundle,
and fails if they disagree. A kappa computed on whichever subset happened to agree is not a
gate.

**Open-ended agreement is broken out.** v4.3's damage was localised: raw agreement 0.695 on
open-ended questions against 0.840 on slot ones, kappa 0.350 against 0.546. If v4.4's
rubric repair worked, that gap closes; if the panel passes while the gap stays open, the
report says so and the rubric is not fixed, whatever the headline number reads.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_1 import cohens_kappa, raw_agreement
from ..eval.detector_v4_4 import (
    AUTHORITY_SCHEMA,
    BUNDLE_LABEL_MINIMA,
    PROMPT_VERSION,
    V4_4_PROTOCOL,
    evaluate_panel_gates,
    panel_agreement,
    per_class_agreement,
    sha256_text,
)
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_store import DEFAULT_OUT_DIR, EVAL_KEY_FILENAME
from .detector_v4_4_bundle import BUNDLE_FILENAME, DEFAULT_V4_4_DIR, PANEL_FILENAME
from .detector_v4_4_judge import output_names_v4_4

__all__ = [
    "ADJUDICATED_FILENAME",
    "DISAGREEMENT_FILENAME",
    "REFERENCE_ADJUDICATED_FILENAME",
    "REFERENCE_DISAGREEMENT_FILENAME",
    "REPORT_FILENAME",
    "detector_v4_4_label_report",
]

REPORT_FILENAME = "DETECTOR_V4_4_LABEL_REPORT.json"
ADJUDICATED_FILENAME = "V4_4_ADJUDICATED.jsonl"
DISAGREEMENT_FILENAME = "V4_4_BLIND_DISAGREEMENTS.jsonl"
REFERENCE_ADJUDICATED_FILENAME = "V4_4_REFERENCE_ADJUDICATED.jsonl"
REFERENCE_DISAGREEMENT_FILENAME = "V4_4_REFERENCE_DISAGREEMENTS.jsonl"
AUTHORITY_FILENAME = "DETECTOR_V4_4_LABEL_AUTHORITY.json"

LABELS = ("NONE", "PARTIAL", "ANSWER")
# The reference axis is ternary and its third value is not "no opinion" -- UNCERTAIN is
# assigned BY RULE when the row has no reference answer to convey. An adjudicator may also
# choose it, which is why it is a legal decision and not only a rule output.
REFERENCE_LABELS = ("YES", "NO", "UNCERTAIN")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pass_table(rows: Sequence[Mapping], field: str) -> tuple[dict[str, str], list[str], list[str]]:
    """``({audit_id: value}, malformed ids, internally-contradictory ids)``.

    A malformed row never becomes a label. The third return value is the v4.4 addition: of
    the malformed rows, which were *contradictory* rather than unparseable. They are counted
    into the same zero-tolerance gate, and reported apart because they mean different
    things about the rubric.
    """
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


def _by_question_type(
    a: Mapping[str, str],
    b: Mapping[str, str],
    question_type_of: Mapping[str, str],
) -> dict:
    """Agreement split by the judges' own ``question_type``, with the PARTIAL/ANSWER cell.

    The cell is named explicitly rather than left to be read off a confusion matrix, because
    ``A=PARTIAL, B=ANSWER`` at 184 of 258 was v4.3's whole diagnosis and a number that has
    to be derived is a number nobody derives.
    """
    out: dict[str, dict] = {}
    for qtype in sorted({str(v) for v in question_type_of.values()}):
        ids = [i for i in sorted(set(a) & set(b)) if question_type_of.get(i) == qtype]
        if not ids:
            continue
        left = [a[i] for i in ids]
        right = [b[i] for i in ids]
        out[qtype] = {
            "n_rows": len(ids),
            "raw_agreement": raw_agreement(left, right),
            "kappa": cohens_kappa(left, right),
            "n_partial_vs_answer": sum(1 for i in ids if {a[i], b[i]} == {"PARTIAL", "ANSWER"}),
            "share_of_disagreements_that_are_partial_vs_answer": (
                sum(1 for i in ids if {a[i], b[i]} == {"PARTIAL", "ANSWER"})
                / sum(1 for i in ids if a[i] != b[i])
                if sum(1 for i in ids if a[i] != b[i])
                else None
            ),
            "per_class": per_class_agreement({i: a[i] for i in ids}, {i: b[i] for i in ids}),
        }
    return out


def detector_v4_4_label_report(
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    bundle: Path = typer.Option(None, "--bundle"),
    panel: Path = typer.Option(None, "--panel"),
    eval_key: Path = typer.Option(DEFAULT_OUT_DIR / EVAL_KEY_FILENAME, "--eval-key"),
    blind_only: bool = typer.Option(
        False,
        "--blind-only",
        help="compute the panel gate and write the disagreements. Run this first.",
    ),
    adjudication: Path = typer.Option(
        None, "--adjudication", help="a JSONL of blind adjudication decisions."
    ),
    reference_adjudication: Path = typer.Option(
        None,
        "--reference-adjudication",
        help=(
            "a JSONL of reference adjudication decisions. Required whenever both reference "
            "passes exist and they disagree anywhere."
        ),
    ),
    require_reference_pass: bool = typer.Option(
        False,
        "--require-reference-pass",
        help="fail unless both reference passes exist. Set for the final authority.",
    ),
) -> None:
    """Fold the two v4.4 blind passes into a report, and then into a label authority."""
    out = Path(out_dir)
    bundle_path = Path(bundle) if bundle else out / BUNDLE_FILENAME
    panel_path = Path(panel) if panel else out / PANEL_FILENAME
    for path in (bundle_path, panel_path):
        if not path.exists():
            raise typer.BadParameter(f"{path} is absent.")

    bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    panel_payload = json.loads(panel_path.read_text(encoding="utf-8"))
    if panel_payload.get("bundle_sha256") != bundle_payload.get("bundle_sha256"):
        raise typer.BadParameter(
            "the calibration panel was frozen against a different bundle "
            f"({str(panel_payload.get('bundle_sha256'))[:12]} vs "
            f"{str(bundle_payload.get('bundle_sha256'))[:12]}). The panel names row ids; "
            "against a different bundle those ids may hold different text, and the primary "
            "kappa would be computed over whatever subset happened to join."
        )

    pairs = {str(p["audit_id"]): p for p in bundle_payload.get("pairs", ())}
    panel_rows = list(panel_payload.get("rows", ()))
    panel_ids = [str(r["audit_id"]) for r in panel_rows]
    intended_of = {str(r["audit_id"]): str(r["intended_class"]) for r in panel_rows}

    # ------------------------------------------------------------- the two passes --
    passes: dict[str, list[dict]] = {}
    for judge in ("A", "B"):
        names = output_names_v4_4(judge=judge, pass_name="blind", run_id="", reportable=True)
        path = out / names["output"]
        if not path.exists():
            raise typer.BadParameter(
                f"{path} is absent, so judge {judge}'s blind pass has not been closed. Both "
                "passes must be frozen before agreement means anything."
            )
        passes[judge] = _read_jsonl(path)

    versions = {str(row.get("prompt_version")) for rows in passes.values() for row in rows}
    if versions != {PROMPT_VERSION}:
        raise typer.BadParameter(
            f"the two passes carry prompt versions {sorted(versions)}, expected "
            f"{PROMPT_VERSION!r}. A kappa computed across a prompt change measures the "
            "change, not the judges -- and a v4.3 label is a different annotation of the "
            "same row, not an earlier version of the same one."
        )

    attempt_a, malformed_a, inconsistent_a = _pass_table(passes["A"], "answer_attempt")
    attempt_b, malformed_b, inconsistent_b = _pass_table(passes["B"], "answer_attempt")
    qtype_a, _, _ = _pass_table(passes["A"], "question_type")
    qtype_b, _, _ = _pass_table(passes["B"], "question_type")

    judged_ids = set(attempt_a) & set(attempt_b)
    missing = sorted(set(pairs) - {str(r["audit_id"]) for r in passes["A"]}) + sorted(
        set(pairs) - {str(r["audit_id"]) for r in passes["B"]}
    )
    truncated = sorted(
        str(r["audit_id"]) for rows in passes.values() for r in rows if r.get("prompt_truncated")
    )

    # ------------------------------------------------------ the primary panel gate --
    panel_present = [i for i in panel_ids if i in judged_ids]
    panel_report = panel_agreement(
        {i: attempt_a[i] for i in panel_present},
        {i: attempt_b[i] for i in panel_present},
        intended={i: intended_of[i] for i in panel_present},
    )

    # ------------------------------------------------- the full-mixture diagnostic --
    mixture_report = panel_agreement(
        {i: attempt_a[i] for i in sorted(judged_ids)},
        {i: attempt_b[i] for i in sorted(judged_ids)},
    )

    # Judges' own question_type, adjudicated by agreement: where they disagree the row is
    # excluded from the breakdown rather than assigned to one, because "which kind of
    # question is this" is exactly what the breakdown conditions on.
    question_type_of = {i: qtype_a[i] for i in judged_ids if qtype_a.get(i) == qtype_b.get(i)}

    measured: dict[str, float | None] = {
        "panel_kappa": panel_report["kappa"],
        "panel_raw_agreement": panel_report["raw_agreement"],
        "panel_agreement_NONE": panel_report["per_class"]["NONE"]["agreement"],
        "panel_agreement_PARTIAL": panel_report["per_class"]["PARTIAL"]["agreement"],
        "panel_agreement_ANSWER": panel_report["per_class"]["ANSWER"]["agreement"],
        "n_malformed": float(len(set(malformed_a) | set(malformed_b))),
        "n_missing": float(len(set(missing))),
        "n_truncated": float(len(set(truncated))),
        "n_unresolved": float(len(panel_ids) - len(panel_present)),
        "n_provenance_failures": 0.0,
    }
    verdicts, failures = evaluate_panel_gates(measured)

    report: dict = {
        "schema": "graph-detector-v4-4-label-report-v1",
        "protocol": V4_4_PROTOCOL,
        "prompt_version": PROMPT_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle_sha256": bundle_payload.get("bundle_sha256"),
        "panel_sha256": panel_payload.get("panel_sha256"),
        "n_bundle_rows": len(pairs),
        "n_judged_by_both": len(judged_ids),
        "primary_gate": {
            "layer": "balanced_calibration_panel",
            "n_panel_rows": len(panel_present),
            "agreement": panel_report,
            "gates": verdicts,
            "failures": failures,
            "passed": not failures,
            "why_this_denominator": (
                "the bound is unchanged at 0.70; the DISTRIBUTION it is evaluated on is "
                "the 200/200/200 panel frozen before any judge ran. v4.3's 0.404 sat under "
                "74.7% raw agreement because 65-81% of rows were one class, and a kappa "
                "that moves with prevalence cannot certify a rubric. Balancing alone would "
                "be measuring the sampler, which is why raw agreement, the per-class "
                "floors, the mixture diagnostic and the question-type breakdown all have "
                "to hold at the same time."
            ),
        },
        "mixture_diagnostic": {
            "layer": "full_natural_mixture",
            "agreement": mixture_report,
            "gated": False,
            "why_not_gated": (
                "distribution-dependent. It is the number to compare against v4.3's 0.404, "
                "and a panel that passes while this stays near 0.40 means the classes are "
                "separable when balanced and the deployment mixture is still dominated by "
                "one of them -- worth knowing, and not the same failure as an unclear rubric."
            ),
        },
        "by_question_type": _by_question_type(attempt_a, attempt_b, question_type_of),
        "v4_3_comparison": {
            "v4_3_blind_kappa": 0.4038,
            "v4_3_raw_agreement": 0.7468,
            "v4_3_n_none_rows": {"judge_A": 69, "judge_B": 17},
            "v4_3_open_ended_raw_agreement": 0.695,
            "v4_3_slot_raw_agreement": 0.840,
            "v4_3_partial_vs_answer_disagreements": 184,
            "v4_3_total_disagreements": 258,
            "why_they_are_not_pooled": (
                "different prompt version, six fields against four, three question types "
                "against two. These are different annotations of overlapping rows, not two "
                "runs of one protocol, and pooling them would produce a kappa about the "
                "prompt change."
            ),
        },
        "counts": {
            "n_malformed_A": len(malformed_a),
            "n_malformed_B": len(malformed_b),
            "n_inconsistent_hierarchy_A": len(inconsistent_a),
            "n_inconsistent_hierarchy_B": len(inconsistent_b),
            "n_missing": len(set(missing)),
            "n_truncated": len(set(truncated)),
            "distribution_A": dict(sorted(Counter(attempt_a.values()).items())),
            "distribution_B": dict(sorted(Counter(attempt_b.values()).items())),
            "why_inconsistent_is_reported_apart": (
                "an unparseable response means the model ignored the format; a "
                "contradictory object means it followed the format and could not hold the "
                "hierarchy together. Both are refused and both count toward the "
                "zero-malformed gate, but only the second says the rubric is unclear."
            ),
        },
        "by_source_subtype": {
            subtype: {
                "n_rows": len(ids),
                "raw_agreement": raw_agreement(
                    [attempt_a[i] for i in ids], [attempt_b[i] for i in ids]
                ),
                "distribution_A": dict(sorted(Counter(attempt_a[i] for i in ids).items())),
                "distribution_B": dict(sorted(Counter(attempt_b[i] for i in ids).items())),
            }
            for subtype, ids in sorted(
                _group_by(judged_ids, lambda i: str(pairs[i].get("source_subtype", ""))).items()
            )
        },
    }

    # ---------------------------------------------------------------- disagreements --
    disagreements = sorted(i for i in judged_ids if attempt_a[i] != attempt_b[i])
    out.mkdir(parents=True, exist_ok=True)
    (out / DISAGREEMENT_FILENAME).write_text(
        "".join(
            dumps_canonical(
                {
                    "audit_id": i,
                    # The adjudicator sees exactly what the judges saw. No reference answer,
                    # no population, no stratum, no intended class, no judge identity -- the
                    # two labels are given as "label_1"/"label_2" in sorted order so that
                    # "judge B usually wins" cannot become a tie-break heuristic.
                    "conditioning_question": pairs[i]["conditioning_question"],
                    "subject_aliases": pairs[i]["subject_aliases"],
                    "candidate_text": pairs[i]["candidate_text"],
                    "label_1": min(attempt_a[i], attempt_b[i]),
                    "label_2": max(attempt_a[i], attempt_b[i]),
                    "answer_attempt": None,
                }
            )
            + "\n"
            for i in disagreements
        ),
        encoding="utf-8",
    )
    report["disagreements"] = {
        "n": len(disagreements),
        "file": str(out / DISAGREEMENT_FILENAME),
        "cells": dict(
            sorted(
                Counter(
                    f"{min(attempt_a[i], attempt_b[i])}|{max(attempt_a[i], attempt_b[i])}"
                    for i in disagreements
                ).items()
            )
        ),
        "blinded": "no reference answer, no population, no stratum, no intended class",
    }

    if blind_only:
        atomic_json(out / REPORT_FILENAME, report)
        typer.echo(
            dumps_canonical(
                {
                    "wrote": str(out / REPORT_FILENAME),
                    "panel_gate_passed": not failures,
                    "panel_kappa": panel_report["kappa"],
                    "panel_raw_agreement": panel_report["raw_agreement"],
                    "panel_per_class": {
                        k: v["agreement"] for k, v in panel_report["per_class"].items()
                    },
                    "mixture_kappa": mixture_report["kappa"],
                    "failures": failures,
                    "n_disagreements": len(disagreements),
                    "next": (
                        "adjudicate the disagreement file blind, then re-run with " "--adjudication"
                        if not failures
                        else "the panel gate FAILED. Return to the rubric or the data. Do "
                        "not adjudicate a failed panel and call it passed."
                    ),
                }
            )
        )
        return

    if failures:
        raise typer.BadParameter(
            "the balanced-panel gate failed: "
            + "; ".join(failures)
            + ". Adjudication is refused. A label authority built on a failed agreement "
            "gate is a set of labels nobody has grounds to trust, and adjudicating the "
            "disagreements would hide the failure behind a resolved file."
        )

    # -------------------------------------------------------------- adjudication --
    if adjudication is None:
        raise typer.BadParameter("pass --blind-only or --adjudication FILE")
    decisions = {
        str(r["audit_id"]): str(r["answer_attempt"])
        for r in _read_jsonl(Path(adjudication))
        if r.get("answer_attempt") in LABELS
    }
    unresolved = sorted(set(disagreements) - set(decisions))
    if unresolved:
        raise typer.BadParameter(
            f"{len(unresolved)} disagreements are unadjudicated (first {unresolved[:3]}). "
            "The gate requires zero unresolved rows, and a row silently dropped is a row "
            "that leaves the denominator without anybody deciding it should."
        )

    final = {i: (decisions.get(i) or attempt_a[i]) for i in sorted(judged_ids)}
    distribution = Counter(final.values())
    minima_measured = {
        "n_none_rows": float(distribution["NONE"]),
        "n_partial_rows": float(distribution["PARTIAL"]),
        "n_answer_rows": float(distribution["ANSWER"]),
    }
    minima_verdicts, minima_failures = evaluate_panel_gates(minima_measured, BUNDLE_LABEL_MINIMA)

    # ----------------------------------------------------------- reference axis --
    reference_block: dict = {"ran": False}
    reference_paths = {
        judge: out
        / output_names_v4_4(judge=judge, pass_name="reference", run_id="", reportable=True)[
            "output"
        ]
        for judge in ("A", "B")
    }
    reference_adjudication_block: dict | None = None
    if all(p.exists() for p in reference_paths.values()):
        ref_rows_a = _read_jsonl(reference_paths["A"])
        ref_rows_b = _read_jsonl(reference_paths["B"])
        ref_a, ref_malformed_a, _ = _pass_table(ref_rows_a, "reference_content")
        ref_b, ref_malformed_b, _ = _pass_table(ref_rows_b, "reference_content")
        key_rows = json.loads(Path(eval_key).read_text(encoding="utf-8")).get("rows", {})
        judged_ref = set(ref_a) & set(ref_b)
        has_reference = {
            i for i in judged_ref if str(key_rows.get(i, {}).get("reference_answer", "")).strip()
        }
        forced = sorted(judged_ref - has_reference)
        scored = sorted(has_reference)

        # Same zero-tolerance the blind axis has. A reference row that never came back is
        # not a row that agreed.
        ref_missing = sorted(
            (set(pairs) - {str(r["audit_id"]) for r in ref_rows_a})
            | (set(pairs) - {str(r["audit_id"]) for r in ref_rows_b})
        )
        ref_truncated = sorted(
            {
                str(r["audit_id"])
                for rows in (ref_rows_a, ref_rows_b)
                for r in rows
                if r.get("prompt_truncated")
            }
        )
        ref_malformed = sorted(set(ref_malformed_a) | set(ref_malformed_b))

        # ------------------------------------------- the reference disagreement file --
        # Written before any refusal, so the pause it forces is one the researcher can act
        # on. Unlike the blind file this one SHOWS the reference answer: deciding whether a
        # candidate conveys it is not a question that can be asked blind.
        ref_disagreements = sorted(i for i in scored if ref_a[i] != ref_b[i])
        out.mkdir(parents=True, exist_ok=True)
        (out / REFERENCE_DISAGREEMENT_FILENAME).write_text(
            "".join(
                dumps_canonical(
                    {
                        "audit_id": i,
                        "conditioning_question": pairs[i]["conditioning_question"],
                        "subject_aliases": pairs[i]["subject_aliases"],
                        "candidate_text": pairs[i]["candidate_text"],
                        "reference_answer": str(key_rows.get(i, {}).get("reference_answer", "")),
                        "label_1": min(ref_a[i], ref_b[i]),
                        "label_2": max(ref_a[i], ref_b[i]),
                        "reference_content": None,
                    }
                )
                + "\n"
                for i in ref_disagreements
            ),
            encoding="utf-8",
        )

        ref_decisions = {
            str(r["audit_id"]): str(r["reference_content"])
            for r in (_read_jsonl(Path(reference_adjudication)) if reference_adjudication else [])
            if r.get("reference_content") in REFERENCE_LABELS
        }
        ref_unresolved = sorted(set(ref_disagreements) - set(ref_decisions))
        ref_failures = []
        if ref_missing:
            ref_failures.append(f"{len(ref_missing)} reference rows are missing")
        if ref_malformed:
            ref_failures.append(f"{len(ref_malformed)} reference rows are malformed")
        if ref_truncated:
            ref_failures.append(f"{len(ref_truncated)} reference prompts were truncated")
        if ref_unresolved:
            ref_failures.append(
                f"{len(ref_unresolved)} reference disagreements are unadjudicated "
                f"(first {ref_unresolved[:3]})"
            )
        if ref_failures:
            raise typer.BadParameter(
                "the reference axis is not closed: "
                + "; ".join(ref_failures)
                + f". The disagreements are in {out / REFERENCE_DISAGREEMENT_FILENAME}; "
                "resolve every one WITH the reference answer visible and re-run with "
                "--reference-adjudication. Reporting judge A's label on the rows the two "
                "judges could not agree about is not an adjudication."
            )

        # The adjudicated reference label. Forced-UNCERTAIN rows keep UNCERTAIN by rule.
        reference_final = dict.fromkeys(forced, "UNCERTAIN")
        reference_final.update(
            {i: (ref_decisions.get(i) or ref_a[i]) for i in scored},
        )
        (out / REFERENCE_ADJUDICATED_FILENAME).write_text(
            "".join(
                dumps_canonical(
                    {
                        "audit_id": i,
                        "reference_content": reference_final[i],
                        "source": (
                            "forced_uncertain_no_reference_answer"
                            if i in set(forced)
                            else "adjudicated" if i in ref_decisions else "both_judges_agreed"
                        ),
                    }
                )
                + "\n"
                for i in sorted(reference_final)
            ),
            encoding="utf-8",
        )
        reference_adjudication_block = {
            "file": str(reference_adjudication) if reference_adjudication else None,
            "sha256": (
                sha256_text(Path(reference_adjudication).read_text(encoding="utf-8"))
                if reference_adjudication
                else None
            ),
            "n_decisions": len(ref_decisions),
            "n_unresolved": 0,
            "adjudicated_file": str(out / REFERENCE_ADJUDICATED_FILENAME),
            "adjudicated_sha256": hashlib.sha256(
                (out / REFERENCE_ADJUDICATED_FILENAME).read_bytes()
            ).hexdigest(),
            "disagreement_file": str(out / REFERENCE_DISAGREEMENT_FILENAME),
            "disagreement_sha256": hashlib.sha256(
                (out / REFERENCE_DISAGREEMENT_FILENAME).read_bytes()
            ).hexdigest(),
            "why_this_file_may_show_the_reference_answer": (
                "it is an offline adjudication artifact. The blind file must never show it, "
                "because the blind label is the trainable one; this axis is never trained "
                "on and never reaches runtime, so withholding the reference answer here "
                "would only make the decision impossible."
            ),
        }
        reference_block = {
            "ran": True,
            "n_rows": len(judged_ref),
            "n_scored_excluding_forced": len(scored),
            "n_forced_uncertain": len(forced),
            "kappa_excluding_forced": cohens_kappa(
                [ref_a[i] for i in scored], [ref_b[i] for i in scored]
            ),
            "raw_agreement_excluding_forced": raw_agreement(
                [ref_a[i] for i in scored], [ref_b[i] for i in scored]
            ),
            "n_malformed": len(ref_malformed),
            "n_missing": len(ref_missing),
            "n_truncated": len(ref_truncated),
            "n_disagreements": len(ref_disagreements),
            "n_unresolved": 0,
            "why_forced_rows_are_excluded": (
                "the reference rubric assigns UNCERTAIN by rule when the reference answer "
                "is empty. Those rows were not judged, they were assigned, and two "
                "annotators agreeing because a rule told them both the same thing is not "
                "evidence that they agree. Every supplement row is in this set: its "
                "candidate was generated for this protocol and there is no answer it was "
                "supposed to convey."
            ),
            # From the ADJUDICATED reference label, not judge A. On exactly the rows the
            # two judges disagreed about, judge A's label was never a decision anybody made.
            "attempt_vs_content": dict(
                sorted(
                    Counter(
                        f"{final[i]}|{reference_final[i]}" for i in scored if i in final
                    ).items()
                )
            ),
            "attempt_vs_content_source": "adjudicated_reference_labels",
            "why_this_axis_is_never_trained_on": (
                "the runtime input does not contain the reference answer. A classifier "
                "trained on reference_content would be a different detector and would "
                "break the answer-free claim. These labels exist to measure actual "
                "correct-content leakage offline, after the checkpoint is frozen."
            ),
        }
    elif require_reference_pass:
        raise typer.BadParameter(
            "--require-reference-pass was set but "
            f"{[str(p) for p in reference_paths.values() if not p.exists()]} are absent."
        )

    (out / ADJUDICATED_FILENAME).write_text(
        "".join(
            dumps_canonical(
                {
                    "audit_id": i,
                    "answer_attempt": final[i],
                    "source": "adjudicated" if i in decisions else "both_judges_agreed",
                }
            )
            + "\n"
            for i in sorted(final)
        ),
        encoding="utf-8",
    )

    report["adjudication"] = {
        "file": str(adjudication),
        "sha256": sha256_text(Path(adjudication).read_text(encoding="utf-8")),
        "n_decisions": len(decisions),
        "n_unresolved": 0,
        "final_distribution": dict(sorted(distribution.items())),
        "bundle_minima": minima_verdicts,
        "bundle_minima_failures": minima_failures,
    }
    report["reference_axis"] = reference_block
    if reference_adjudication_block is not None:
        report["reference_adjudication"] = reference_adjudication_block
    atomic_json(out / REPORT_FILENAME, report)

    authority = {
        "schema": AUTHORITY_SCHEMA,
        "protocol": V4_4_PROTOCOL,
        "prompt_version": PROMPT_VERSION,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bundle_sha256": bundle_payload.get("bundle_sha256"),
        "panel_sha256": panel_payload.get("panel_sha256"),
        "adjudicated_file": str(out / ADJUDICATED_FILENAME),
        "adjudicated_sha256": hashlib.sha256((out / ADJUDICATED_FILENAME).read_bytes()).hexdigest(),
        "blind_adjudication_sha256": report["adjudication"]["sha256"],
        "blind_disagreement_sha256": hashlib.sha256(
            (out / DISAGREEMENT_FILENAME).read_bytes()
        ).hexdigest(),
        # Both axes are bound by hash. The reference files are bound even though the axis
        # is untrainable: an authority that names them without pinning them cannot later
        # prove which reference labels its cross-tabulation was computed from.
        "reference_adjudicated_file": (
            reference_adjudication_block["adjudicated_file"]
            if reference_adjudication_block
            else None
        ),
        "reference_adjudicated_sha256": (
            reference_adjudication_block["adjudicated_sha256"]
            if reference_adjudication_block
            else None
        ),
        "reference_adjudication_sha256": (
            reference_adjudication_block["sha256"] if reference_adjudication_block else None
        ),
        "reference_disagreement_sha256": (
            reference_adjudication_block["disagreement_sha256"]
            if reference_adjudication_block
            else None
        ),
        "n_labels": len(final),
        "distribution": dict(sorted(distribution.items())),
        "panel_gate": {"passed": True, "kappa": panel_report["kappa"]},
        "bundle_minima_passed": not minima_failures,
        "reference_axis_ran": reference_block["ran"],
        "reference_axis_closed": reference_adjudication_block is not None,
        "trainable_field": "answer_attempt",
        "not_trainable": ["reference_content"],
        "green": not minima_failures,
    }
    atomic_json(out / AUTHORITY_FILENAME, authority)
    typer.echo(
        dumps_canonical(
            {
                "wrote": [str(out / REPORT_FILENAME), str(out / AUTHORITY_FILENAME)],
                "final_distribution": dict(sorted(distribution.items())),
                "bundle_minima_failures": minima_failures,
                "reference_axis_ran": reference_block["ran"],
                "authority_green": authority["green"],
            }
        )
    )


def _group_by(ids, key):
    out: dict[str, list[str]] = defaultdict(list)
    for i in sorted(ids):
        out[key(i)].append(i)
    return out
