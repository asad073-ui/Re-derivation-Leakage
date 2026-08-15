"""``rdl graph-detector-v4-2-bank-audit`` — freeze an audit sample of a bank, and its inputs.

A bank holds up to ~24,000 candidate rows. Labelling all of them is two judges x two
passes x 24,000 = ~96,000 calls: days of quota, or real money, for labels the gate does not
need. What the gate needs is enough labelled rows *per stratum, in the partition the number
is read on*, and those come from a sample.

The cell is (partition, stratum), not stratum
---------------------------------------------
The first version drew 300/500/400 across the whole bank and checked the pre-registered
minima against that draw. The minima are conditions on the **held-out gate population**, so
a draw that satisfies every one of them can still leave the held-out partition with forty
retain rows — the check passed before the filter that decides which rows the gate actually
sees. Sampling is now per (partition, stratum), the plan and the minima are both per cell,
and a shortfall in one cell is a shortfall.

The sample is frozen before the detector scores anything
--------------------------------------------------------
The stratum of a row is decided by generation metadata alone: which population it came from
(protected or retain) and what the run's own pinned NLI+ROUGE scorer said about it. The
trained detector's score is never consulted, and this command never loads a checkpoint.
Sampling on the detector's score would make every recall number a measurement of the
sampler: draw the rows the detector already fires on and recall is high by construction.

The draw itself is content-addressed — ordered by ``sha256(bank_id || partition || stratum
|| pair_sha256)`` — so it is reproducible from the bank alone, does not depend on the order
rows happen to sit in the file, and does not move when unrelated rows are added.

What it writes
--------------
``BANK_AUDIT_JUDGE_{A,B}.jsonl``      blind inputs: question + candidate, labels blank
``BANK_AUDIT_REFERENCE_PASS_{A,B}.jsonl``  reference inputs: the above + reference answer
``BANK_AUDIT_KEY.json``               the offline key: stratum, population, NLI label,
                                      partition. **No judge ever sees this file.**
``BANK_AUDIT_MANIFEST.json``          the frozen plan, the hashes, the shortfalls, and a
                                      ``bundle`` block naming every file above

The ``bundle`` block is why the report can read this audit at all. It named its files
``BANK_AUDIT_*`` while the report opened ``LABEL_AUDIT_*``, so the report either failed to
find a key or — worse, pointed at the v4.1 directory — produced a report about the 1,019-row
training audit under a header about the fresh bank.

If a cell comes up short
------------------------
Recorded as a shortfall and the command exits non-zero. The response is a pre-registered
extension that draws more rows under a recorded seed — not a quiet reduction of the plan,
and never a second draw chosen after seeing which rows the first one produced.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_2 import (
    AUDIT_PARTITIONS,
    AUDIT_SAMPLE_MINIMA,
    AUDIT_SAMPLE_PLAN,
    AUDIT_SAMPLE_RULE,
    AUDIT_STRATA,
    BLIND_FIELDS,
    JUDGES,
    REFERENCE_FIELDS,
    V4_2_PROTOCOL,
    audit_sample_total,
)
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT
from .detector_v4_2_banks import BANK_FILENAME, _load_bank_rows
from .detector_v4_2_llm_judge import DEFAULT_V4_2_OUT, _write_jsonl

__all__ = ["assign_stratum", "detector_v4_2_bank_audit", "draw_sample"]

BLIND_FILENAME = "BANK_AUDIT_JUDGE_{judge}.jsonl"
REFERENCE_FILENAME = "BANK_AUDIT_REFERENCE_PASS_{judge}.jsonl"
KEY_FILENAME = "BANK_AUDIT_KEY.json"
MANIFEST_FILENAME = "BANK_AUDIT_MANIFEST.json"
MANIFEST_SCHEMA = "graph-detector-v4-2-bank-audit-manifest-v2"

STRATA: tuple[str, ...] = AUDIT_STRATA


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign_stratum(row: Mapping) -> str | None:
    """The row's audit stratum, from generation metadata alone.

    ``None`` for a protected row the run's scorer never labelled: it belongs in neither
    the likely-answer nor the clean stratum, and putting it in either would make one of
    them a mixture of "the scorer said so" and "the scorer said nothing".
    """
    if str(row.get("population")) == "retain" or row.get("partition") == "retain":
        return "retain"
    leaking = row.get("nli_leaking")
    if leaking is True:
        return "protected_likely_answer"
    if leaking is False:
        return "protected_clean"
    return None


def assign_partition(row: Mapping) -> str | None:
    """``development`` or ``heldout``. Anything else is not a cell this plan draws from."""
    partition = str(row.get("partition") or "")
    return partition if partition in AUDIT_PARTITIONS else None


def draw_sample(
    rows: Sequence[Mapping],
    *,
    bank_id: str,
    plan: Mapping[str, Mapping[str, int]],
) -> tuple[dict[str, dict[str, list[dict]]], dict[str, int]]:
    """``(rows by partition and stratum, shortfalls)``. Deterministic, content-addressed.

    Seedless on purpose. A seed is a number somebody chose, and a number chosen after
    looking at a draw is not a seed. The ordering key is a hash of the bank's own id, the
    cell, and the row's own content, so the same bank always yields the same sample and a
    different bank yields a different one.

    Shortfalls are keyed ``"partition/stratum"`` because that is the unit the plan is
    written in. A bank-wide count cannot express "the held-out retain pool is 40 rows",
    which is the exact condition the gate's retain FPR needs and the previous per-stratum
    check could not see.
    """
    buckets: dict[tuple[str, str], list[Mapping]] = {
        (partition, stratum): [] for partition, cells in plan.items() for stratum in cells
    }
    for row in rows:
        cell = (assign_partition(row), assign_stratum(row))
        if cell in buckets:
            buckets[cell].append(row)

    drawn: dict[str, dict[str, list[dict]]] = {p: {} for p in plan}
    shortfalls: dict[str, int] = {}
    for partition, cells in plan.items():
        for stratum, wanted in cells.items():
            pool = sorted(
                buckets[(partition, stratum)],
                key=lambda r: _sha(
                    f"{bank_id}\x00{partition}\x00{stratum}\x00{r.get('pair_sha256', '')}"
                ),
            )
            drawn[partition][stratum] = [dict(r) for r in pool[:wanted]]
            if len(pool) < wanted:
                shortfalls[f"{partition}/{stratum}"] = wanted - len(pool)
    return drawn, shortfalls


def check_minima(
    drawn: Mapping[str, Mapping[str, Sequence]],
    minima: Mapping[str, Mapping[str, int]] = AUDIT_SAMPLE_MINIMA,
) -> list[str]:
    """Every cell whose DRAWN row count is below its pre-registered minimum.

    Checked on the drawn rows per cell rather than on the bank-wide totals. The minima are
    statements about the population each reported number is computed over, and a total
    that satisfies them while one partition does not is the failure this replaces.
    """
    failures: list[str] = []
    for partition, cells in minima.items():
        for stratum, floor in cells.items():
            have = len(drawn.get(partition, {}).get(stratum, ()))
            if have < floor:
                failures.append(
                    f"{partition}/{stratum}: {have} row(s) drawn, minimum {floor}. "
                    f"The {partition} {stratum} number would be computed over {have} rows."
                )
    return failures


def detector_v4_2_bank_audit(
    bank: str = typer.Option("engineering", "--bank", help="engineering | final"),
    bank_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--bank-dir"),
    bank_path: Path | None = typer.Option(None, "--bank-path", help="override the bank file"),
    policy_cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/discovery.json"), "--policy-cohort"
    ),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
    v4_1_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--v4-1-dir"),
    allow_shortfall: bool = typer.Option(
        False,
        "--allow-shortfall",
        help="write the files even when a cell is short. Marks the audit non-reportable.",
    ),
) -> None:
    """Freeze a stratified audit sample of a bank and write every file the judges need."""
    del v4_1_dir  # kept for signature stability with the other v4.2 commands
    if bank not in BANK_FILENAME:
        raise typer.BadParameter(f"--bank must be engineering or final, got {bank!r}")
    path = bank_path or (bank_dir / BANK_FILENAME[bank])
    if not path.exists():
        raise typer.BadParameter(
            f"{path} is absent; run `rdl graph-detector-v4-2-build-bank` first"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    bank_id = str(payload.get("bank_id") or path.stem)
    bank_sha = str(payload.get("content_sha256") or "")
    rows = _load_bank_rows(payload)
    if not rows:
        raise typer.BadParameter(f"{path} carries no rows")

    # Reference answers come from the pinned dataset revision, exactly as the v4.1 audit's
    # reference pass does. They are written ONLY into the reference-pass files and the key,
    # never into a blind file — the contract test over the prompt bytes is what enforces
    # that, not this comment.
    from .detector_v4_data import _question_bank

    _questions, answers, provenance = _question_bank(policy_cohort)

    drawn, shortfalls = draw_sample(rows, bank_id=bank_id, plan=AUDIT_SAMPLE_PLAN)
    below_minima = check_minima(drawn)

    sampled: list[dict] = []
    for partition, cells in drawn.items():
        for stratum, subset in cells.items():
            for row in subset:
                pair = str(row.get("pair_sha256") or row.get("text_sha256") or "")
                sampled.append(
                    {**row, "stratum": stratum, "partition": partition, "pair_sha256": pair}
                )

    # Ordered by a hash that encodes neither the stratum nor the partition, so both are
    # interleaved in the file the judges read. A file whose first 300 rows are all leaking
    # is a file whose first 300 rows a judge can learn the answer to.
    sampled.sort(key=lambda r: _sha(f"order\x00{bank_id}\x00{r['pair_sha256']}"))

    audit_ids: dict[str, str] = {}
    for row in sampled:
        audit_id = _sha(f"{bank_id}\x00{row['pair_sha256']}")[:16]
        if audit_id in audit_ids:  # pragma: no cover - a 64-bit collision
            raise typer.BadParameter(f"audit_id collision on {audit_id}; widen the id")
        audit_ids[audit_id] = row["pair_sha256"]
        row["audit_id"] = audit_id

    blind_rows = [
        {
            "audit_id": r["audit_id"],
            "protected_question": r["request"],
            "candidate_text": r["text"],
            **dict.fromkeys(BLIND_FIELDS, ""),
        }
        for r in sampled
    ]
    reference_rows = [
        {
            "audit_id": r["audit_id"],
            "protected_question": r["request"],
            "candidate_text": r["text"],
            # A retain row has no protected reference answer, and the empty string is the
            # signal the runner turns into a forced UNCERTAIN without spending a call.
            "reference_answer": (
                "" if r["stratum"] == "retain" else str(answers.get(str(r.get("item_id", "")), ""))
            ),
            **dict.fromkeys(REFERENCE_FIELDS, ""),
        }
        for r in sampled
    ]

    written: list[str] = []
    bundle: dict[str, dict[str, str]] = {"blind": {}, "reference": {}}
    for role in JUDGES:
        blind_path = output_dir / BLIND_FILENAME.format(judge=role)
        reference_path = output_dir / REFERENCE_FILENAME.format(judge=role)
        _write_jsonl(blind_path, blind_rows)
        _write_jsonl(reference_path, reference_rows)
        bundle["blind"][role] = str(blind_path)
        bundle["reference"][role] = str(reference_path)
        written += [str(blind_path), str(reference_path)]

    key = {
        "schema": "graph-detector-v4-2-bank-audit-key-v2",
        "judges_must_not_read": True,
        "why": (
            "this file carries the stratum, the population, the run's NLI+ROUGE verdict "
            "and the bank partition. Those are the things the audit exists to check; a "
            "judge that saw them would be confirming them."
        ),
        "bank_id": bank_id,
        "bank_content_sha256": bank_sha,
        "rows": {
            r["audit_id"]: {
                "stratum": r["stratum"],
                "population": r.get("population", "protected"),
                "partition": r.get("partition"),
                "nli_leaking": r.get("nli_leaking"),
                "item_id": r.get("item_id"),
                "text_sha256": r.get("text_sha256"),
                "pair_sha256": r["pair_sha256"],
                "bank_content_sha256": bank_sha,
                "n_chars": len(str(r.get("text", ""))),
            }
            for r in sampled
        },
    }
    key_path = output_dir / KEY_FILENAME
    atomic_json(key_path, key)

    available: dict[str, dict[str, int]] = {
        partition: {
            stratum: sum(
                1 for r in rows if assign_partition(r) == partition and assign_stratum(r) == stratum
            )
            for stratum in STRATA
        }
        for partition in AUDIT_PARTITIONS
    }
    reportable = not shortfalls and not below_minima
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "protocol": V4_2_PROTOCOL,
        "bundle_id": f"{bank_id}-audit",
        "bank": bank,
        "bank_path": str(path),
        "bank_id": bank_id,
        "bank_content_sha256": bank_sha,
        "reportable": reportable,
        # The block every consumer resolves its inputs from. Without it the report opened
        # the v4.1 audit's filenames and read a different audit's rows.
        "bundle": {"key": str(key_path), **bundle},
        "sampling_rule": AUDIT_SAMPLE_RULE,
        "uses_detector_score": False,
        "why_not": (
            "a sample drawn on the detector's own score makes every recall number a "
            "measurement of the sampler. The stratum comes from generation metadata."
        ),
        "sampling_cell": "(partition, stratum)",
        "why_the_cell_is_the_partition_too": (
            "the pre-registered minima are conditions on the population each number is "
            "computed over, and the gate's numbers are computed on the held-out "
            "partition. A bank-wide draw can satisfy every minimum and leave the held-out "
            "retain pool at 40 rows, because the check ran before the filter."
        ),
        "plan": {p: dict(cells) for p, cells in AUDIT_SAMPLE_PLAN.items()},
        "minima": {p: dict(cells) for p, cells in AUDIT_SAMPLE_MINIMA.items()},
        "n_rows_planned_total": audit_sample_total(),
        "n_rows_available_by_cell": available,
        "n_rows_drawn_by_cell": {
            partition: {stratum: len(v) for stratum, v in sorted(cells.items())}
            for partition, cells in sorted(drawn.items())
        },
        "n_rows_sampled": len(sampled),
        "n_rows_in_bank": len(rows),
        "shortfalls": shortfalls,
        "below_minima": below_minima,
        "on_shortfall": (
            "draw more rows under a pre-registered extension with a recorded seed, and "
            "record it in DECISIONS.md. Never reduce the plan to fit the draw: a plan "
            "changed after seeing the draw is a plan the draw chose."
        ),
        "audit_ids_sha256": _sha(
            json.dumps(sorted(audit_ids.items()), sort_keys=True, separators=(",", ":"))
        ),
        "n_audit_ids": len(audit_ids),
        "raw_bank_preserved": True,
        "raw_bank_note": (
            "the full bank is untouched on disk. This command selects which rows are "
            "LABELLED; it removes nothing."
        ),
        "blind_fields": list(BLIND_FIELDS),
        "reference_fields": list(REFERENCE_FIELDS),
        "files": [*sorted(written), str(key_path)],
        "reference_answer_provenance": provenance,
        "judge_population": "two_independent_llm_judges",
        "human_grounded": False,
        "publication_label_valid": False,
        "next": (
            "rdl graph-detector-v4-2-llm-judge --judge A --pass blind "
            f"--input {output_dir / BLIND_FILENAME.format(judge='A')}"
        ),
    }
    atomic_json(output_dir / MANIFEST_FILENAME, manifest)

    typer.echo(f"wrote {output_dir / MANIFEST_FILENAME}")
    for name in written:
        typer.echo(f"wrote {name}")
    typer.echo(f"wrote {key_path}")
    typer.echo("")
    for partition in AUDIT_PARTITIONS:
        for stratum in STRATA:
            typer.echo(
                f"  {partition:<12} {stratum:<26} "
                f"drawn {len(drawn.get(partition, {}).get(stratum, ())):>5}  "
                f"available {available[partition][stratum]:>6}  "
                f"planned {AUDIT_SAMPLE_PLAN[partition][stratum]:>5}"
            )
    if shortfalls or below_minima:
        typer.echo("", err=True)
        for cell, missing in sorted(shortfalls.items()):
            typer.echo(f"  [SHORT] {cell}: {missing} row(s) short of the plan", err=True)
        for failure in below_minima:
            typer.echo(f"  [MINIMUM] {failure}", err=True)
        if not allow_shortfall:
            raise typer.BadParameter(
                "the bank cannot supply the pre-registered audit sample. Generate more "
                "rows under a pre-registered extension rather than reducing the plan; "
                "pass --allow-shortfall only to write a NON-REPORTABLE audit for "
                "development."
            )
    raise typer.Exit(0 if reportable else 1)
