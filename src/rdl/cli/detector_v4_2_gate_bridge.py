"""The bridge between a labelled bank and a gate: what must be true before either opens.

Two commands read a bank and its adjudicated labels — ``select-operating-point``, which
chooses the threshold on the development partition, and ``final-gate``, which opens the
held-out partition once at that threshold. Everything they must agree about lives here, so
"the labels belong to this bank" and "the audit passed its gate" cannot mean one thing in
one command and something weaker in the other.

The four things this module refuses
-----------------------------------
**A label file that was never gated.** ``final-gate`` took ``--labels`` and checked that the
rows named content the bank contained. It never asked whether those labels came from an
audit that *passed* — a κ of 0.2, four unresolved disagreements and a five-row smoke pass
would sail through, and every number downstream would be computed over labels the protocol
says are not labels. The audit manifest and a passing model-alignment report are now
required, bound to this bank by content hash, and the report's own hash of the label file
must match the file being read.

**Duplicate labels.** The label map was an unchecked dict comprehension keyed by
``audit_id``: a file containing one row twice, or two rows for one bank pair, silently kept
whichever came last. Two adjudications of one row is a disagreement about that row, and the
one that "wins" is the one that happens to sit lower in the file.

**Labels from another bank.** Kept from the previous revision and strengthened: the pair
digest is required, not merely preferred. ``text_sha256`` identifies a text, and the bank is
keyed by (text, question) — so matching on the text hash matches the wrong row whenever one
candidate appears under two protected questions, which is the case the pair digest exists
for.

**A threshold typed on the command line.** ``--threshold`` was a required float. The one
number the whole protocol is about — chosen on development, frozen, then opened once — was
whatever the operator typed, and a second attempt at a different value left no trace.
:func:`load_operating_point` reads the frozen artifact and refuses any other value.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

__all__ = [
    "OPERATING_POINT_FILENAME",
    "OPERATING_POINT_SCHEMA",
    "bind_labels_to_bank",
    "labelled_rows_for_partition",
    "load_label_map",
    "load_operating_point",
    "require_audit_gate",
    "score_with_backend",
    "sha_file",
]

OPERATING_POINT_FILENAME = "DETECTOR_V4_2_OPERATING_POINT.json"
OPERATING_POINT_SCHEMA = "graph-detector-v4-2-operating-point-v1"
ALIGNMENT_REPORT_FILENAME = "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json"
BANK_AUDIT_MANIFEST_FILENAME = "BANK_AUDIT_MANIFEST.json"

# Two thresholds compared for equality after a JSON round trip. Floats that differ in the
# last bit are the same operating point; floats that differ at 1e-9 are not, and a gate
# that accepted "close enough" would accept a hand-typed 0.42 for a frozen 0.4200001.
THRESHOLD_TOLERANCE = 1e-12


def sha_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- labels --


def load_label_map(path: Path) -> dict[str, dict]:
    """``{audit_id: row}`` from an adjudicated label file. Duplicates are refused.

    Three distinct failures, all of which the previous dict comprehension resolved by
    keeping the last row:

    * the same ``audit_id`` twice — one row adjudicated twice;
    * two ``audit_id``s naming the same ``pair_sha256`` — one bank row labelled twice under
      different ids, which makes the gate's denominator wrong by one and its numerator
      wrong by zero or one depending on which copy won;
    * a row with no pair digest at all, which cannot be bound to a bank row.
    """
    if not path.exists():
        raise typer.BadParameter(f"{path} is absent")
    rows: dict[str, dict] = {}
    pair_owner: dict[str, str] = {}
    duplicates: list[str] = []
    conflicts: list[str] = []
    unkeyed: list[str] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        audit_id = str(row.get("audit_id", ""))
        if not audit_id:
            raise typer.BadParameter(f"{path}:{number} has no audit_id")
        if audit_id in rows:
            duplicates.append(audit_id)
            continue
        pair = str(row.get("pair_sha256") or "")
        if not pair:
            unkeyed.append(audit_id)
        elif pair in pair_owner and pair_owner[pair] != audit_id:
            conflicts.append(f"{pair[:16]}… labelled as both {pair_owner[pair]} and {audit_id}")
        else:
            pair_owner[pair] = audit_id
        rows[audit_id] = row

    failures: list[str] = []
    if duplicates:
        failures.append(
            f"{len(duplicates)} duplicate audit_id(s) {sorted(set(duplicates))[:5]}. One "
            "row adjudicated twice is a disagreement about that row, and a dict keyed by "
            "audit_id resolves it by file order."
        )
    if conflicts:
        failures.append(
            f"{len(conflicts)} bank pair(s) labelled under more than one audit_id: "
            f"{conflicts[:3]}. The gate would count one bank row twice."
        )
    if unkeyed:
        failures.append(
            f"{len(unkeyed)} label row(s) carry no pair_sha256 (e.g. {unkeyed[:3]}). A "
            "label that names only a text hash cannot be bound to a bank row: the bank is "
            "keyed by (text, question), and one candidate text can appear under two "
            "protected questions."
        )
    if failures:
        raise typer.BadParameter(
            f"{path} cannot be used as a label map:\n  " + "\n  ".join(failures)
        )
    if not rows:
        raise typer.BadParameter(f"{path} carries no label rows")
    return rows


def bind_labels_to_bank(
    label_rows: Mapping[str, Mapping],
    bank_rows: Sequence[Mapping],
    bank_payload: Mapping,
    *,
    labels_path: Path,
) -> dict[str, dict]:
    """``{pair_sha256: bank row}``, after proving the labels describe THIS bank."""
    by_pair: dict[str, dict] = {}
    for row in bank_rows:
        pair = str(row.get("pair_sha256") or "")
        if not pair:
            raise typer.BadParameter(
                "a bank row carries no pair_sha256. The bank predates the (text, question) "
                "digest and cannot be bound to labels; rebuild it."
            )
        by_pair[pair] = dict(row)

    unmatched = [
        audit_id
        for audit_id, row in label_rows.items()
        if str(row.get("pair_sha256") or "") not in by_pair
    ]
    if unmatched:
        raise typer.BadParameter(
            f"{len(unmatched)} label row(s) name content this bank does not contain "
            f"(e.g. {unmatched[:3]}). These labels were produced for a different bank. "
            "Gating a new bank on an old bank's labels is the failure this check exists "
            "for: every number would describe the old surface."
        )
    declared = {
        str(r.get("bank_content_sha256"))
        for r in label_rows.values()
        if r.get("bank_content_sha256")
    }
    bank_sha = str(bank_payload.get("content_sha256"))
    if not declared:
        raise typer.BadParameter(
            f"{labels_path} records no bank_content_sha256 on any row. A label file that "
            "does not say which bank revision it was drawn against cannot be shown to "
            "describe this one; re-run `rdl graph-detector-v4-2-label-report`, which "
            "carries the binding through."
        )
    if declared != {bank_sha}:
        raise typer.BadParameter(
            f"{labels_path} was produced against bank content {sorted(declared)} and this "
            f"bank hashes to {bank_sha}. The bank moved after the audit; the labels "
            "describe rows that are no longer these rows."
        )
    return by_pair


# ------------------------------------------------------------------- the audit gate --


def require_audit_gate(
    *,
    audit_manifest: Path,
    alignment_report: Path,
    labels: Path,
    bank_payload: Mapping,
) -> dict:
    """Prove the labels came from an audit that PASSED, over THIS bank. Returns the record.

    Five conditions, each of which was previously unchecked and each of which produces
    numbers rather than an error when it is false:

    1. the audit manifest exists, is reportable, and names this bank's content hash;
    2. the alignment report exists and cleared its decision gate;
    3. it recorded zero unresolved disagreements and zero provenance failures;
    4. the label file it names hashes to the file being read;
    5. the report is a model-judge report that still says so — ``human_grounded`` false and
       ``publication_label_valid`` false, unedited.
    """
    failures: list[str] = []
    bank_sha = str(bank_payload.get("content_sha256"))

    if not audit_manifest.exists():
        raise typer.BadParameter(
            f"{audit_manifest} is absent. The gate is scored over a frozen stratified "
            "audit sample of THIS bank; without its manifest there is no evidence that "
            "the labelled rows were chosen before the detector scored anything. Run "
            "`rdl graph-detector-v4-2-bank-audit` first."
        )
    manifest = json.loads(audit_manifest.read_text(encoding="utf-8"))
    if str(manifest.get("bank_content_sha256")) != bank_sha:
        failures.append(
            f"{audit_manifest} was drawn from bank content "
            f"{str(manifest.get('bank_content_sha256'))[:16]}… and this bank hashes to "
            f"{bank_sha[:16]}…"
        )
    if not manifest.get("reportable", False):
        failures.append(
            f"{audit_manifest} is marked reportable=false "
            f"(shortfalls {manifest.get('shortfalls')}, "
            f"below_minima {manifest.get('below_minima')}). A sample that did not meet its "
            "pre-registered minima cannot support the numbers those minima were set for."
        )
    if manifest.get("uses_detector_score") is not False:
        failures.append(
            f"{audit_manifest} does not assert uses_detector_score=false. A sample drawn "
            "on the detector's own score makes every recall number a measurement of the "
            "sampler."
        )

    if not alignment_report.exists():
        raise typer.BadParameter(
            f"{alignment_report} is absent. A label file with no alignment report is a "
            "file of opinions: nothing establishes that the two judges agreed, that the "
            "disagreements were resolved, or that four complete passes ran. Run "
            "`rdl graph-detector-v4-2-label-report`."
        )
    report = json.loads(alignment_report.read_text(encoding="utf-8"))
    if not report.get("all_gates_passed"):
        failures.append(
            f"{alignment_report} did NOT clear its decision gate: "
            f"{report.get('failed_gates')}. Labels that failed the audit gate are not "
            "labels; fix the label protocol, not the model."
        )
    measured = {g["gate"]: g.get("measured") for g in report.get("gates", ())}
    if measured.get("n_unresolved_disagreements") not in (0, 0.0):
        failures.append(
            f"{alignment_report} records "
            f"{measured.get('n_unresolved_disagreements')} unresolved disagreement(s). An "
            "unresolved row has two labels and no label."
        )
    n_provenance = len(list((report.get("provenance") or {}).get("failures", ())))
    if n_provenance:
        failures.append(
            f"{alignment_report} records {n_provenance} provenance failure(s): the four "
            "judge runs are not four complete runs of the frozen protocol."
        )
    if report.get("human_grounded") is not False:
        failures.append(
            f"{alignment_report} claims human_grounded={report.get('human_grounded')!r}. "
            "The v4.2 schema is model-judge output and that flag is false by "
            "construction; this file has been edited."
        )
    if report.get("publication_label_valid") is not False:
        failures.append(
            f"{alignment_report} claims publication_label_valid="
            f"{report.get('publication_label_valid')!r}. Two model judges agreeing is "
            "consistency evidence, not correctness."
        )

    # The report names the label file it produced, and its hash. Without this a passing
    # report authorises whatever file is passed as --labels, including one edited after
    # the report was written.
    declared_labels = report.get("adjudicated_file")
    declared_sha = report.get("adjudicated_sha256")
    actual_sha = sha_file(labels)
    if not declared_sha:
        failures.append(
            f"{alignment_report} records no adjudicated_sha256, so it cannot vouch for "
            f"{labels}. Re-run the label report; it now carries the hash of the file it "
            "wrote."
        )
    elif str(declared_sha) != actual_sha:
        failures.append(
            f"{labels} hashes to {actual_sha[:16]}… and {alignment_report} vouches for "
            f"{str(declared_sha)[:16]}… ({declared_labels}). The label file changed after "
            "the audit passed."
        )
    if report.get("bank_content_sha256") and str(report["bank_content_sha256"]) != bank_sha:
        failures.append(
            f"{alignment_report} was computed over bank content "
            f"{str(report['bank_content_sha256'])[:16]}…, not {bank_sha[:16]}…"
        )

    if failures:
        for failure in failures:
            typer.echo(f"  [GATE] {failure}", err=True)
        raise typer.BadParameter(
            f"{len(failures)} condition(s) between the bank, its audit and its labels are "
            "not met. Refusing to score: a gate that reads labels which never passed the "
            "judge gate is a gate on nothing."
        )
    return {
        "audit_manifest": str(audit_manifest),
        "audit_manifest_sha256": sha_file(audit_manifest),
        "audit_reportable": True,
        "alignment_report": str(alignment_report),
        "alignment_report_sha256": sha_file(alignment_report),
        "alignment_report_all_gates_passed": True,
        "answer_attempt_kappa": (
            report.get("inter_judge", {}).get("per_field", {}).get("answer_attempt", {})
        ).get("cohens_kappa"),
        "labels_file": str(labels),
        "labels_sha256": actual_sha,
        "bank_content_sha256": bank_sha,
        "judge_population": report.get("judge_population"),
        "human_grounded": False,
        "publication_label_valid": False,
        "n_unresolved_disagreements": measured.get("n_unresolved_disagreements"),
        "n_provenance_failures": n_provenance,
        "audit_plan": manifest.get("plan"),
        "audit_minima": manifest.get("minima"),
        "audit_drawn_by_cell": manifest.get("n_rows_drawn_by_cell"),
    }


# ------------------------------------------------------- the rows, and scoring them --


def labelled_rows_for_partition(
    label_rows: Mapping[str, Mapping],
    by_pair: Mapping[str, Mapping],
    *,
    partition: str,
    concept_of: Mapping[str, str],
) -> list[dict]:
    """The scorable rows of one partition: a bank row joined to its adjudicated label.

    One partition, never a union. ``development`` chooses the threshold and ``heldout``
    reports it, and a function that could return both at once is a function one caller
    will eventually call with the wrong one — which is how a retain FPR gets selected on
    and reported from the same rows.

    A protected row with no ``answer_attempt`` is dropped: it has no Goal A label, so it
    belongs in neither the recall numerator nor the false-alarm denominator. A retain row
    is kept regardless, because its denominator is "retain rows, all of them" and dropping
    the unlabelled ones would shrink the denominator of the number that measures utility.
    """
    out: list[dict] = []
    for audit_id, label in sorted(label_rows.items()):
        source = by_pair[str(label.get("pair_sha256") or "")]
        if str(source.get("partition")) != partition:
            continue
        population = str(source.get("population", "protected"))
        if label.get("answer_attempt") is None and population != "retain":
            continue
        out.append(
            {
                "audit_id": audit_id,
                "request": source["request"],
                "text": source["text"],
                "item_id": source.get("item_id"),
                "concept_id": concept_of.get(str(source.get("item_id", "")), ""),
                "population": population,
                "partition": partition,
                "nli_leaking": source.get("nli_leaking"),
                "answer_attempt": label.get("answer_attempt"),
                "reference_content": label.get("reference_content"),
                "question_type": label.get("question_type"),
            }
        )
    return out


def score_with_backend(
    rows: Sequence[Mapping],
    *,
    backend: str,
    model_artifact: Path | None,
    device: str,
    policy_cohort: Path,
) -> tuple[list[dict], dict]:
    """``(predictions, compute record)`` for one partition's rows, under routing.

    The detector is built here so that both commands load the checkpoint exactly once and
    through the same path — including the hash re-verification inside ``from_artifact``,
    which is skipped by anything that constructs a detector some other way.
    """
    from ..defenses.concept_registry import ConceptPolicy, ConceptRegistry
    from ..defenses.identity_router import alias_index
    from ..studies.graph_leak.cohort import load_cohort
    from .detector_v4_gates import _natural_scores, _protected_questions, build_backend

    cohort = load_cohort(policy_cohort, exclusions_path=None)
    registry = ConceptRegistry.from_questions(
        [
            {"item_id": e.item_id, "concept_id": e.concept_id, "question": q}
            for e, q in (
                (e, next((r["request"] for r in rows if r["item_id"] == e.item_id), ""))
                for e in cohort.items
            )
            if q
        ],
        policy=ConceptPolicy(),
    )
    questions_by_concept: dict[str, list[str]] = {}
    for row in rows:
        concept = str(row.get("concept_id") or "")
        if concept:
            questions_by_concept.setdefault(concept, []).append(str(row["request"]))
    questions = _protected_questions(registry, questions_by_concept)
    index = alias_index(registry)

    detector = build_backend(
        backend, model_artifact if backend == "cross_encoder" else None, device=device
    )
    described = detector.to_dict()
    parameter_device = str(described.get("device_of_parameters") or "unknown")
    if device.startswith("cuda") and not parameter_device.startswith("cuda"):
        raise typer.BadParameter(
            f"--device {device} was requested but the model's parameters are on "
            f"{parameter_device!r}. Refusing: the record would assert a GPU that never ran."
        )
    predictions = _natural_scores(rows, detector, questions, index)
    compute = {
        "backend": backend,
        "model_artifact": str(model_artifact) if model_artifact else None,
        "device_requested": device or None,
        "device_of_parameters": parameter_device,
        "gpu_used": parameter_device.startswith("cuda"),
        "checkpoint_hashes_verified": bool(described.get("checkpoint_hashes_verified")),
        "verified_label_map_sha256": described.get("verified_label_map_sha256"),
        "registry_fingerprint": registry.fingerprint(),
        "n_concepts": len(registry),
        "cohort_fingerprint": cohort.fingerprint(),
    }
    return predictions, compute


def concept_index(policy_cohort: Path) -> dict[str, str]:
    """``{item_id: concept_id}`` from the frozen cohort. Never from a bank row."""
    from ..studies.graph_leak.cohort import load_cohort

    cohort = load_cohort(policy_cohort, exclusions_path=None)
    return {e.item_id: e.concept_id for e in cohort.items}


# --------------------------------------------------------------- the frozen threshold --


def load_operating_point(
    path: Path,
    *,
    bank_content_sha256: str,
    model_artifact: Path,
    requested_threshold: float | None = None,
    model_manifest: Mapping | None = None,
) -> dict:
    """The frozen operating point, or a refusal. Never a threshold from the command line.

    ``requested_threshold`` exists so an operator who passes ``--threshold`` gets an error
    naming both numbers rather than a silent override. Passing the same value is allowed
    and is a no-op; passing a different one is the thing this function exists to stop.
    """
    if not path.exists():
        raise typer.BadParameter(
            f"{path} is absent. The threshold is not an argument: it is chosen on the "
            "DEVELOPMENT partition by `rdl graph-detector-v4-2-select-operating-point`, "
            "frozen into that file, and read from it here. A gate opened at a threshold "
            "typed on the command line is a gate at whichever threshold worked."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema")) != OPERATING_POINT_SCHEMA:
        raise typer.BadParameter(
            f"{path} carries schema {payload.get('schema')!r}, expected "
            f"{OPERATING_POINT_SCHEMA!r}"
        )
    failures: list[str] = []
    if str(payload.get("bank_content_sha256")) != str(bank_content_sha256):
        failures.append(
            f"the operating point was selected on bank content "
            f"{str(payload.get('bank_content_sha256'))[:16]}… and this bank hashes to "
            f"{str(bank_content_sha256)[:16]}…"
        )
    if str(payload.get("model_artifact")) != str(model_artifact):
        failures.append(
            f"the operating point was selected for model artifact "
            f"{payload.get('model_artifact')!r}, and this gate was given "
            f"{str(model_artifact)!r}. A threshold belongs to the checkpoint it was "
            "chosen on."
        )
    # The artifact path is not the checkpoint. A threshold belongs to the BYTES it was
    # chosen on, and a training run that re-selected a different epoch writes the same
    # manifest path with different hashes inside it.
    if model_manifest is not None:
        if str(payload.get("selected_checkpoint") or "") != str(
            model_manifest.get("selected_checkpoint") or ""
        ):
            failures.append(
                f"the operating point was selected on checkpoint "
                f"{payload.get('selected_checkpoint')!r} and the manifest now names "
                f"{model_manifest.get('selected_checkpoint')!r}"
            )
        frozen_files = (payload.get("selected_checkpoint_hashes") or {}).get("files_sha256")
        current_files = (model_manifest.get("selected_checkpoint_hashes") or {}).get("files_sha256")
        if frozen_files != current_files:
            failures.append(
                "the operating point's checkpoint hashes do not match the model "
                "manifest's. The threshold was chosen on different weights than the ones "
                "this gate would score with."
            )
    if payload.get("selected_threshold") is None:
        failures.append(
            "the operating point records no selected_threshold: no threshold on the "
            "development partition cleared both false-alarm ceilings. There is nothing to "
            "open the held-out partition at."
        )
    threshold = payload.get("selected_threshold")
    if (
        requested_threshold is not None
        and threshold is not None
        and abs(float(requested_threshold) - float(threshold)) > THRESHOLD_TOLERANCE
    ):
        failures.append(
            f"--threshold {requested_threshold} was passed and the frozen operating point "
            f"is {threshold}. The frozen value is the one the held-out partition may be "
            "opened at; a different one is a second choice made after the first."
        )
    if failures:
        for failure in failures:
            typer.echo(f"  [THRESHOLD] {failure}", err=True)
        raise typer.BadParameter(f"{path} does not authorise this gate. Refusing to open the bank.")
    return payload
