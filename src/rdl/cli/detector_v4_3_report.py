"""``rdl graph-detector-v4-3-label-report`` -- two judges' passes into one label authority.

v4.3 could produce judge outputs and could consume adjudicated labels, and had nothing in
between. This is the missing middle: agreement, gates, a disagreement file, adjudication,
and an authority artifact that the trainer refuses to run without.

Two invocations, in order
-------------------------
``--blind-only``
    Reads both blind passes, computes kappa, and writes the disagreements. The
    disagreement file carries the question, the aliases and the candidate and **never the
    reference answer** -- the researcher adjudicates blind, exactly as the judges did.

``--adjudication FILE [--require-reference-pass]``
    Folds the adjudicated decisions in, freezes the blind labels, and -- once both
    reference passes exist -- computes reference-assisted kappa and applies every gate.

Why the reference kappa excludes forced rows
--------------------------------------------
The reference rubric says an empty reference answer is UNCERTAIN by rule. Those rows were
not judged, they were *assigned*, and two annotators agreeing because a rule told them both
the same thing is not evidence that they agree. Counting them inflates kappa exactly where
the data is weakest, so they are excluded and counted separately.

Why a malformed row is not a missing row
----------------------------------------
Both are refused, but they are different failures and the report says which: a malformed
row means the model produced something unparseable, a missing row means it produced nothing
at all. Collapsing them hides whether the problem is the prompt or the run.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_human import cohens_kappa
from .detector_v4_3_local_judge import output_names
from .detector_v4_3_store import DEFAULT_OUT_DIR, EVAL_KEY_FILENAME

__all__ = [
    "ADJUDICATED_FILENAME",
    "AUTHORITY_SCHEMA",
    "LABEL_GATES",
    "REPORT_FILENAME",
    "detector_v4_3_label_report",
    "evaluate_gates",
]

REPORT_FILENAME = "DETECTOR_V4_3_LABEL_REPORT.json"
ADJUDICATED_FILENAME = "V4_3_ADJUDICATED.jsonl"
DISAGREEMENT_FILENAME = "V4_3_BLIND_DISAGREEMENTS.jsonl"

AUTHORITY_SCHEMA = "graph-detector-v4-3-label-authority-v1"

LABELS = ("NONE", "PARTIAL", "ANSWER")

# §6.1 of the protocol, restated as machine-checkable bounds. Frozen here before any v4.3
# label exists; a failing gate is reported as failed and never re-run with a moved bound.
LABEL_GATES: dict[str, tuple[str, float]] = {
    "blind_kappa": (">=", 0.70),
    "reference_kappa_excluding_forced": (">=", 0.70),
    "n_answer_rows": (">=", 100.0),
    "n_none_rows": (">=", 200.0),
    "n_strata_with_answer_rows": (">=", 2.0),
    "n_concepts_with_answer_rows": (">=", 3.0),
    "n_unresolved": ("==", 0.0),
    "n_malformed": ("==", 0.0),
    "n_missing": ("==", 0.0),
    "n_provenance_failures": ("==", 0.0),
}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def evaluate_gates(measured: Mapping[str, float | None]) -> tuple[dict, list[str]]:
    """``(per-gate verdicts, failures)``. A gate whose input is None fails, not passes."""
    verdicts: dict[str, dict] = {}
    failures: list[str] = []
    for name, (operator, bound) in sorted(LABEL_GATES.items()):
        value = measured.get(name)
        if value is None:
            ok = False
            detail = "not measured"
        else:
            ok = value >= bound if operator == ">=" else value == bound
            detail = f"{value} {operator} {bound}"
        verdicts[name] = {"ok": ok, "measured": value, "bound": bound, "operator": operator}
        if not ok:
            failures.append(f"{name}: {detail}")
    return verdicts, failures


def _judge_table(
    rows: Sequence[Mapping], field: str
) -> tuple[dict[str, str], list[str], list[str]]:
    """``({audit_id: label}, malformed ids, unparseable ids)`` for one closed pass."""
    labels: dict[str, str] = {}
    malformed: list[str] = []
    for row in rows:
        audit_id = str(row["audit_id"])
        if row.get("malformed") or row.get(field) is None:
            malformed.append(audit_id)
            continue
        labels[audit_id] = str(row[field])
    return labels, sorted(malformed), []


def detector_v4_3_label_report(
    bundle: Path = typer.Option(DEFAULT_OUT_DIR / "DETECTOR_V4_3_PAIR_BUNDLE.json", "--bundle"),
    judge_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--judge-dir"),
    eval_key: Path = typer.Option(DEFAULT_OUT_DIR / EVAL_KEY_FILENAME, "--eval-key"),
    output_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--output-dir"),
    blind_only: bool = typer.Option(
        False, "--blind-only", help="stop after blind agreement and the disagreement file."
    ),
    adjudication: Path = typer.Option(
        None, "--adjudication", help="JSONL of {audit_id, answer_attempt} for disagreements."
    ),
    require_reference_pass: bool = typer.Option(
        False, "--require-reference-pass", help="also verify both reference passes and gate."
    ),
) -> None:
    """Agreement, gates, adjudication and the v4.3 label authority."""
    bundle_path = Path(bundle)
    if not bundle_path.exists():
        raise typer.BadParameter(f"{bundle_path} is absent")
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    pairs = list(payload.get("pairs", ()))
    expected_ids = [str(p["audit_id"]) for p in pairs]

    # The evaluation key is opened HERE, in an offline reporting command, for strata and
    # concepts only. It never reaches a judge prompt or a model input.
    key_rows = json.loads(Path(eval_key).read_text(encoding="utf-8")).get("rows", {})

    provenance_failures: list[str] = []
    blind: dict[str, dict[str, str]] = {}
    malformed_ids: set[str] = set()
    missing_ids: set[str] = set()

    for role in ("A", "B"):
        names = output_names(judge=role, pass_name="blind", run_id="", reportable=True)
        path = Path(judge_dir) / names["output"]
        if not path.exists():
            raise typer.BadParameter(
                f"{path} is absent: judge {role}'s blind pass has not been closed. Run "
                f"`rdl graph-detector-v4-3-local-judge --judge {role} --pass blind --close`."
            )
        rows = _read_jsonl(path)
        run_path = Path(judge_dir) / names["run"]
        if run_path.exists():
            run = json.loads(run_path.read_text(encoding="utf-8"))
            if not run.get("reportable"):
                provenance_failures.append(f"judge {role} blind run is marked non-reportable")
            if str(run.get("bundle_sha256") or run.get("source", {}).get("bundle_sha256")) not in (
                str(payload.get("bundle_sha256")),
                "None",
            ):
                provenance_failures.append(
                    f"judge {role} blind run names a different bundle than {bundle_path.name}"
                )
        else:
            provenance_failures.append(f"judge {role} blind run manifest is absent")

        labels, malformed, _ = _judge_table(rows, "answer_attempt")
        blind[role] = labels
        malformed_ids |= set(malformed)
        missing_ids |= set(expected_ids) - set(labels) - set(malformed)

    shared = [i for i in expected_ids if i in blind["A"] and i in blind["B"]]
    a = [blind["A"][i] for i in shared]
    b = [blind["B"][i] for i in shared]
    blind_kappa = cohens_kappa(a, b, labels=LABELS)
    disagreements = [i for i in shared if blind["A"][i] != blind["B"][i]]

    stratum_of = {k: str(v.get("stratum", "")) for k, v in key_rows.items()}
    concept_of = {k: str(v.get("concept_id", "")) for k, v in key_rows.items()}
    by_stratum: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for i in shared:
        by_stratum[stratum_of.get(i, "")].append((blind["A"][i], blind["B"][i]))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pair_of = {str(p["audit_id"]): p for p in pairs}

    if blind_only:
        path = out / DISAGREEMENT_FILENAME
        path.write_text(
            "".join(
                dumps_canonical(
                    {
                        "audit_id": i,
                        "conditioning_question": pair_of[i]["conditioning_question"],
                        "subject_aliases": pair_of[i]["subject_aliases"],
                        "candidate_text": pair_of[i]["candidate_text"],
                        "judge_a": blind["A"][i],
                        "judge_b": blind["B"][i],
                        "answer_attempt": None,
                    }
                )
                + "\n"
                for i in disagreements
            ),
            encoding="utf-8",
        )
        typer.echo(
            dumps_canonical(
                {
                    "stage": "blind_only",
                    "n_rows": len(shared),
                    "blind_kappa": blind_kappa,
                    "n_disagreements": len(disagreements),
                    "wrote_disagreements": str(path),
                    "n_malformed": len(malformed_ids),
                    "n_missing": len(missing_ids),
                    "carries_reference_answers": False,
                    "next": (
                        "adjudicate the disagreement file BLIND (question, aliases, "
                        "candidate, rubric only), then rerun with --adjudication."
                    ),
                }
            )
        )
        return

    # ------------------------------------------------------------- adjudication --
    resolved: dict[str, str] = {}
    if adjudication is not None:
        for row in _read_jsonl(Path(adjudication)):
            label = row.get("answer_attempt")
            if label not in LABELS:
                raise typer.BadParameter(
                    f"{adjudication}: row {row.get('audit_id')} has answer_attempt="
                    f"{label!r}, not one of {list(LABELS)}. An unresolved row is not a NONE."
                )
            resolved[str(row["audit_id"])] = str(label)

    unresolved = sorted(i for i in disagreements if i not in resolved)
    final: list[dict] = []
    for i in shared:
        if blind["A"][i] == blind["B"][i]:
            final.append({"audit_id": i, "answer_attempt": blind["A"][i], "source": "agreed"})
        elif i in resolved:
            final.append({"audit_id": i, "answer_attempt": resolved[i], "source": "adjudicated"})

    distribution = Counter(r["answer_attempt"] for r in final)
    answer_ids = [r["audit_id"] for r in final if r["answer_attempt"] == "ANSWER"]

    # ---------------------------------------------------- reference-assisted pass --
    reference_kappa = None
    n_forced = 0
    reference_detail: dict = {"ran": False}
    if require_reference_pass:
        reference: dict[str, dict[str, str]] = {}
        for role in ("A", "B"):
            names = output_names(judge=role, pass_name="reference", run_id="", reportable=True)
            path = Path(judge_dir) / names["output"]
            if not path.exists():
                raise typer.BadParameter(
                    f"{path} is absent: judge {role}'s reference pass has not been closed."
                )
            labels, malformed, _ = _judge_table(_read_jsonl(path), "reference_content")
            reference[role] = labels
            malformed_ids |= set(malformed)

        reference_answer_of = {k: str(v.get("reference_answer", "")) for k, v in key_rows.items()}
        both = [i for i in expected_ids if i in reference["A"] and i in reference["B"]]
        forced = [i for i in both if not reference_answer_of.get(i, "").strip()]
        judged = [i for i in both if i not in set(forced)]
        n_forced = len(forced)
        reference_kappa = cohens_kappa(
            [reference["A"][i] for i in judged],
            [reference["B"][i] for i in judged],
            labels=("YES", "NO", "UNCERTAIN"),
        )
        reference_detail = {
            "ran": True,
            "n_rows": len(both),
            "n_forced_excluded": n_forced,
            "n_judged": len(judged),
            "kappa_excluding_forced": reference_kappa,
            "why_excluded": (
                "an empty reference answer is UNCERTAIN by rule. Those rows were assigned, "
                "not judged, and two annotators agreeing because a rule told them both the "
                "same thing inflates kappa exactly where the data is weakest."
            ),
        }

    measured = {
        "blind_kappa": blind_kappa,
        "reference_kappa_excluding_forced": reference_kappa if require_reference_pass else None,
        "n_answer_rows": float(distribution.get("ANSWER", 0)),
        "n_none_rows": float(distribution.get("NONE", 0)),
        "n_strata_with_answer_rows": float(len({stratum_of.get(i, "") for i in answer_ids})),
        "n_concepts_with_answer_rows": float(
            len({concept_of.get(i, "") for i in answer_ids if concept_of.get(i)})
        ),
        "n_unresolved": float(len(unresolved)),
        "n_malformed": float(len(malformed_ids)),
        "n_missing": float(len(missing_ids)),
        "n_provenance_failures": float(len(provenance_failures)),
    }
    verdicts, failures = evaluate_gates(measured)

    adjudicated_path = out / ADJUDICATED_FILENAME
    adjudicated_path.write_text("".join(dumps_canonical(r) + "\n" for r in final), encoding="utf-8")

    report = {
        "schema": AUTHORITY_SCHEMA,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "annotator_population": "two_independent_local_open_weight_judges",
        "human_grounded": False,
        "publication_label_valid": False,
        "why_not_publication_valid": (
            "these are engineering labels. They become publication-valid only if the "
            "250-row human validation passes its own gates."
        ),
        "bundle": str(bundle_path),
        "bundle_sha256": payload.get("bundle_sha256"),
        "adjudicated_file": str(adjudicated_path),
        "adjudicated_sha256": hashlib.sha256(adjudicated_path.read_bytes()).hexdigest(),
        "n_rows": len(shared),
        "blind": {
            "kappa": blind_kappa,
            "raw_agreement": (
                sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a) if a else None
            ),
            "n_disagreements": len(disagreements),
            "n_adjudicated": len(resolved),
            "n_unresolved": len(unresolved),
            "unresolved": unresolved[:20],
            "kappa_by_stratum": {
                stratum: cohens_kappa([x for x, _ in rows], [y for _, y in rows], labels=LABELS)
                for stratum, rows in sorted(by_stratum.items())
            },
        },
        "reference": reference_detail,
        "label_distribution": dict(sorted(distribution.items())),
        "n_malformed": len(malformed_ids),
        "n_missing": len(missing_ids),
        "provenance_failures": provenance_failures,
        "gates": verdicts,
        "gate_failures": failures,
        "gates_passed": not failures,
    }
    atomic_json(out / REPORT_FILENAME, report)

    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / REPORT_FILENAME),
                "adjudicated": str(adjudicated_path),
                "blind_kappa": blind_kappa,
                "reference_kappa_excluding_forced": reference_kappa,
                "n_forced_excluded": n_forced,
                "label_distribution": dict(sorted(distribution.items())),
                "gates_passed": not failures,
                "gate_failures": failures,
                "next": (
                    (
                        "rebuild a LABELLED bundle with `rdl graph-detector-v4-3-bundle "
                        f"--labels {adjudicated_path}` into a NEW --out-dir; never overwrite "
                        "the unlabelled input freeze."
                    )
                    if not failures
                    else "a gate failed. Do not train and do not reinterpret the bound."
                ),
            }
        )
    )
