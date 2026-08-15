"""``rdl graph-detector-v4-2-select-operating-point`` — choose the threshold, once, on development.

The threshold was a required ``--threshold`` float on ``final-gate``. That is the whole
protocol expressed as an argument: the number is supposed to be chosen on the development
partition, frozen, and then used to open the held-out partition exactly once, and instead it
was whatever was typed — with a second attempt at a different value leaving no trace beyond
a second opening record nobody reads.

This command is the missing half. It reads the bank, the adjudicated labels for that bank,
the audit manifest and the passing alignment report; it scores the **development partition
only**; it sweeps the exact score breakpoints; it takes the highest ANSWER-recall point that
clears BOTH false-alarm ceilings; and it writes ``DETECTOR_V4_2_OPERATING_POINT.json`` bound
to the bank's content hash and the model artifact. ``final-gate`` reads that file and
refuses any other value.

Why the sweep never touches the held-out rows
---------------------------------------------
It cannot: :func:`labelled_rows_for_partition` is called with ``development`` and there is
no argument that widens it. The previous arrangement was worse than a convention — the
retain pool was a single undivided block, so the sweep enforced the retain ceiling on the
same rows the gate then reported the retain FPR over, and that reported rate was a
description of the selection. The bank now halves the retain rows too.

Both ceilings, separately
-------------------------
``protected_nonanswer_fpr <= 0.10`` AND ``retain_fpr <= 0.10``, never their union. The
retain pool is the smaller one, so an aggregate of 0.10 is satisfiable with a retain rate
far above it — and firing on retain traffic is the utility cost the whole defence is
measured against.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_1 import (
    GOAL_A_GATES,
    goal_a_gate_inputs,
    goal_a_summarise,
    score_goal_a,
)
from ..eval.detector_v4_2 import V4_2_PROTOCOL
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_2_banks import BANK_FILENAME, _load_bank_rows
from .detector_v4_2_gate_bridge import (
    ALIGNMENT_REPORT_FILENAME,
    BANK_AUDIT_MANIFEST_FILENAME,
    OPERATING_POINT_FILENAME,
    OPERATING_POINT_SCHEMA,
    bind_labels_to_bank,
    concept_index,
    labelled_rows_for_partition,
    load_label_map,
    require_audit_gate,
    score_with_backend,
    sha_file,
)
from .detector_v4_2_llm_judge import DEFAULT_V4_2_OUT

__all__ = ["detector_v4_2_select_operating_point", "select_threshold"]

SELECTION_RULE = (
    "The highest answer_attempt micro-recall point on the DEVELOPMENT partition, among "
    "the exact score breakpoints, subject to BOTH protected_nonanswer_fpr <= "
    f"{GOAL_A_GATES['protected_nonanswer_fpr'][1]} AND retain_fpr <= "
    f"{GOAL_A_GATES['retain_fpr'][1]}. Ties break toward the LOWER threshold, which is the "
    "conservative direction for a detector: it fires more, and the false-alarm ceilings "
    "are already binding. A point whose retain rate could not be measured — no retain rows "
    "in the development partition — is not eligible, because 'not measured' and 'fine' "
    "must not agree."
)


def breakpoints(scores: Sequence[float]) -> tuple[float, ...]:
    """Every distinct observed score, plus one above the maximum for 'fire on nothing'."""
    distinct = sorted({float(s) for s in scores})
    if not distinct:
        return ()
    return (*distinct, distinct[-1] + 1e-9)


def select_threshold(rows: Sequence[dict], predictions: Sequence[dict]) -> dict:
    """Sweep the development rows and apply :data:`SELECTION_RULE`."""
    grid = [
        goal_a_summarise(rows, predictions, threshold=threshold)
        for threshold in breakpoints([p["answer_probability"] for p in predictions])
    ]
    eligible = [
        point
        for point in grid
        if point["answer_attempt_micro_recall"] is not None
        and point["protected_nonanswer_fpr"] is not None
        and point["protected_nonanswer_fpr"] <= GOAL_A_GATES["protected_nonanswer_fpr"][1]
        # `retain_fpr is None` means the development partition carried no retain rows. That
        # is not a retain rate of zero; a threshold chosen without one is a threshold whose
        # utility cost was never looked at.
        and point["retain_fpr"] is not None and point["retain_fpr"] <= GOAL_A_GATES["retain_fpr"][1]
    ]
    chosen = max(
        eligible,
        key=lambda point: (point["answer_attempt_micro_recall"], -point["threshold"]),
        default=None,
    )
    return {
        "grid": grid,
        "n_points": len(grid),
        "n_eligible": len(eligible),
        "selected": chosen,
        "selected_threshold": float(chosen["threshold"]) if chosen else None,
        "rule": SELECTION_RULE,
        "reason_if_none": (
            None
            if chosen
            else (
                "no development threshold cleared both false-alarm ceilings with both "
                "measured. The rule does not fall back to an aggregate rate and does not "
                "fall back to the best recall available: a threshold chosen by a different "
                "rule than the frozen one is not the operating point the protocol "
                "authorises."
            )
        ),
    }


def detector_v4_2_select_operating_point(
    bank: str = typer.Option("engineering", "--bank", help="engineering | final"),
    bank_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--bank-dir"),
    labels: Path = typer.Option(..., "--labels", help="V4_2_ADJUDICATED.jsonl for THIS bank"),
    audit_manifest: Path | None = typer.Option(
        None, "--audit-manifest", help="BANK_AUDIT_MANIFEST.json for this bank"
    ),
    alignment_report: Path | None = typer.Option(
        None, "--alignment-report", help="the passing model-label alignment report"
    ),
    model_artifact: Path = typer.Option(..., "--model-artifact"),
    policy_cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/discovery.json"), "--policy-cohort"
    ),
    backend: str = typer.Option("cross_encoder", "--backend"),
    device: str = typer.Option("", "--device"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
    refreeze: bool = typer.Option(
        False,
        "--refreeze",
        help="overwrite an existing operating point. Recorded, and refused once the "
        "held-out partition has been opened at the old one.",
    ),
) -> None:
    """Select and FREEZE the operating point on the development partition."""
    if bank not in BANK_FILENAME:
        raise typer.BadParameter(f"--bank must be engineering or final, got {bank!r}")
    bank_path = bank_dir / BANK_FILENAME[bank]
    if not bank_path.exists():
        raise typer.BadParameter(
            f"{bank_path} is absent; run `rdl graph-detector-v4-2-build-bank` first"
        )
    out_path = output_dir / OPERATING_POINT_FILENAME
    if out_path.exists() and not refreeze:
        previous = json.loads(out_path.read_text(encoding="utf-8"))
        raise typer.BadParameter(
            f"{out_path} already freezes threshold {previous.get('selected_threshold')} "
            f"(written {previous.get('utc')}). Pass --refreeze only if the held-out "
            "partition has NOT been opened at it; re-selecting a threshold after a gate "
            "has been opened is selecting on the gate."
        )
    if out_path.exists() and refreeze:
        from .detector_v4_2_banks import OPENING_RECORD_FILENAME

        record_path = output_dir / OPENING_RECORD_FILENAME[bank]
        if record_path.exists():
            raise typer.BadParameter(
                f"{record_path} exists: the {bank} bank's held-out partition has already "
                "been opened at the frozen threshold. A new threshold now would be chosen "
                "with the gate's numbers in view. The response is a new bank."
            )

    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    bank_sha = str(payload.get("content_sha256"))
    manifest_path = audit_manifest or (bank_dir / BANK_AUDIT_MANIFEST_FILENAME)
    report_path = alignment_report or (output_dir / ALIGNMENT_REPORT_FILENAME)
    gate_record = require_audit_gate(
        audit_manifest=manifest_path,
        alignment_report=report_path,
        labels=labels,
        bank_payload=payload,
    )

    label_rows = load_label_map(labels)
    bank_rows = _load_bank_rows(payload)
    by_pair = bind_labels_to_bank(label_rows, bank_rows, payload, labels_path=labels)

    rows = labelled_rows_for_partition(
        label_rows,
        by_pair,
        partition="development",
        concept_of=concept_index(policy_cohort),
    )
    if not rows:
        raise typer.BadParameter(
            "the development partition carries no labelled rows. The threshold cannot be "
            "chosen on the partition it must be chosen on, and choosing it anywhere else "
            "is choosing it on the gate."
        )
    if not any(r["population"] == "retain" for r in rows):
        raise typer.BadParameter(
            "the development partition carries no labelled RETAIN rows, so the retain "
            "false-alarm ceiling cannot constrain the threshold. Re-run the bank audit: "
            "its plan draws retain rows in both partitions for exactly this reason."
        )

    predictions, compute = score_with_backend(
        rows,
        backend=backend,
        model_artifact=model_artifact if backend == "cross_encoder" else None,
        device=device,
        policy_cohort=policy_cohort,
    )
    selection = select_threshold(rows, predictions)

    model_manifest = json.loads(Path(model_artifact).read_text(encoding="utf-8"))
    if model_manifest.get("reportable") is False:
        raise typer.BadParameter(
            f"{model_artifact} records reportable=false: it was trained outside the "
            "preregistration (see its `preregistration.deviations`). No operating point "
            "is frozen for a checkpoint whose training run may not be reported."
        )

    artifact = {
        "schema": OPERATING_POINT_SCHEMA,
        "protocol": V4_2_PROTOCOL,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bank": bank,
        "bank_path": str(bank_path),
        "bank_content_sha256": bank_sha,
        "partition_selected_on": "development",
        "never_read": "heldout",
        "selected_threshold": selection["selected_threshold"],
        "selection_rule": SELECTION_RULE,
        "selection": selection,
        "gates_applied": {
            "protected_nonanswer_fpr": list(GOAL_A_GATES["protected_nonanswer_fpr"]),
            "retain_fpr": list(GOAL_A_GATES["retain_fpr"]),
        },
        "development_summary": selection["selected"],
        "development_gate_preview": (
            score_goal_a(goal_a_gate_inputs(selection["selected"]))
            if selection["selected"]
            else None
        ),
        "development_gate_preview_note": (
            "the development partition's own gate arithmetic, for diagnosis only. It is "
            "not a result: these are the rows the threshold was chosen on."
        ),
        "n_development_rows": len(rows),
        "n_development_rows_by_population": {
            population: sum(1 for r in rows if r["population"] == population)
            for population in sorted({r["population"] for r in rows})
        },
        "model_artifact": str(model_artifact),
        "selected_checkpoint": model_manifest.get("selected_checkpoint"),
        "selected_checkpoint_hashes": model_manifest.get("selected_checkpoint_hashes"),
        "model_artifact_sha256": sha_file(Path(model_artifact)),
        "training_reportable": model_manifest.get("reportable"),
        "label_gate": gate_record,
        "compute": compute,
        "judge_population": "two_independent_llm_judges",
        "human_grounded": False,
        "publication_label_valid": False,
        "refrozen": bool(refreeze),
        "next": (
            "rdl graph-detector-v4-2-final-gate --bank "
            f"{bank} --labels {labels} --model-artifact {model_artifact} "
            "--partition heldout"
        ),
    }
    atomic_json(out_path, artifact)
    typer.echo(f"wrote {out_path}")
    typer.echo("")
    typer.echo(f"development rows: {len(rows)} ({artifact['n_development_rows_by_population']})")
    typer.echo(f"breakpoints swept: {selection['n_points']}, eligible: {selection['n_eligible']}")
    if selection["selected_threshold"] is None:
        typer.echo(f"NO OPERATING POINT: {selection['reason_if_none']}", err=True)
        raise typer.Exit(1)
    chosen = selection["selected"]
    typer.echo(f"frozen threshold: {selection['selected_threshold']}")
    typer.echo(
        f"  development recall {chosen['answer_attempt_micro_recall']}, "
        f"protected-clean FPR {chosen['protected_nonanswer_fpr']}, "
        f"retain FPR {chosen['retain_fpr']}"
    )
    typer.echo("")
    typer.echo(
        "the held-out partition has not been read. `final-gate` reads this file and "
        "refuses any other threshold."
    )
    raise typer.Exit(0)
