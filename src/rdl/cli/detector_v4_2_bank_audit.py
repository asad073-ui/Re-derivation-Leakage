"""``rdl graph-detector-v4-2-bank-audit`` — freeze an audit sample of a bank, and its inputs.

A bank holds up to ~24,000 candidate rows. Labelling all of them is two judges x two
passes x 24,000 = ~96,000 calls: days of free-tier quota, or real money, for labels the
gate does not need. What the gate needs is enough labelled rows *per stratum* — 150 ANSWER,
400 protected NONE, 400 retain — and those come from a sample.

The sample is frozen before the detector scores anything
--------------------------------------------------------
The stratum of a row is decided by generation metadata alone: which group it came from
(natural or retain) and what the run's own pinned NLI+ROUGE scorer said about it. The
trained detector's score is never consulted, and this command never loads a checkpoint.
Sampling on the detector's score would make every recall number a measurement of the
sampler: draw the rows the detector already fires on and recall is high by construction.

The draw itself is content-addressed — ordered by ``sha256(bank_id || stratum ||
pair_sha256)`` — so it is reproducible from the bank alone, does not depend on the order
rows happen to sit in the file, and does not move when unrelated rows are added.

What it writes
--------------
``BANK_AUDIT_JUDGE_{A,B}.jsonl``      blind inputs: question + candidate, labels blank
``BANK_AUDIT_REFERENCE_PASS_{A,B}.jsonl``  reference inputs: the above + reference answer
``BANK_AUDIT_KEY.json``               the offline key: stratum, population, NLI label,
                                      partition. **No judge ever sees this file.**
``BANK_AUDIT_MANIFEST.json``          the frozen plan, the hashes, and the shortfalls

The blind and reference files are the same shape as the v4.1 audit's, so the same judge
runner reads them with ``--input`` and no second code path exists to drift.

If a stratum comes up short
---------------------------
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
    AUDIT_SAMPLE_MINIMA,
    AUDIT_SAMPLE_PLAN,
    AUDIT_SAMPLE_RULE,
    BLIND_FIELDS,
    JUDGES,
    REFERENCE_FIELDS,
    V4_2_PROTOCOL,
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

STRATA: tuple[str, ...] = ("protected_likely_answer", "protected_clean", "retain")


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


def draw_sample(
    rows: Sequence[Mapping], *, bank_id: str, plan: Mapping[str, int]
) -> tuple[dict[str, list[dict]], dict[str, int]]:
    """``(rows by stratum, shortfalls)``. Deterministic, content-addressed, seedless.

    Seedless on purpose. A seed is a number somebody chose, and a number chosen after
    looking at a draw is not a seed. The ordering key is a hash of the bank's own id and
    the row's own content, so the same bank always yields the same sample and a different
    bank yields a different one.
    """
    buckets: dict[str, list[Mapping]] = {name: [] for name in plan}
    for row in rows:
        stratum = assign_stratum(row)
        if stratum in buckets:
            buckets[stratum].append(row)

    drawn: dict[str, list[dict]] = {}
    shortfalls: dict[str, int] = {}
    for stratum, wanted in plan.items():
        pool = sorted(
            buckets[stratum],
            key=lambda r: _sha(f"{bank_id}\x00{stratum}\x00{r.get('pair_sha256', '')}"),
        )
        drawn[stratum] = [dict(r) for r in pool[:wanted]]
        if len(pool) < wanted:
            shortfalls[stratum] = wanted - len(pool)
    return drawn, shortfalls


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
        help="write the files even when a stratum is short. Marks the audit non-reportable.",
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

    sampled: list[dict] = []
    for stratum, subset in drawn.items():
        for row in subset:
            pair = str(row.get("pair_sha256") or row.get("text_sha256") or "")
            sampled.append({**row, "stratum": stratum, "pair_sha256": pair})

    # Ordered by a hash that does not encode the stratum, so the strata are interleaved in
    # the file the judges read. A file whose first 300 rows are all leaking is a file whose
    # first 300 rows a judge can learn the answer to.
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
    for role in JUDGES:
        blind_path = output_dir / BLIND_FILENAME.format(judge=role)
        reference_path = output_dir / REFERENCE_FILENAME.format(judge=role)
        _write_jsonl(blind_path, blind_rows)
        _write_jsonl(reference_path, reference_rows)
        written += [str(blind_path), str(reference_path)]

    key = {
        "schema": "graph-detector-v4-2-bank-audit-key-v1",
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
    atomic_json(output_dir / KEY_FILENAME, key)

    available = {
        stratum: sum(1 for r in rows if assign_stratum(r) == stratum) for stratum in STRATA
    }
    reportable = not shortfalls
    manifest = {
        "schema": "graph-detector-v4-2-bank-audit-manifest-v1",
        "protocol": V4_2_PROTOCOL,
        "bank": bank,
        "bank_path": str(path),
        "bank_id": bank_id,
        "bank_content_sha256": bank_sha,
        "reportable": reportable,
        "sampling_rule": AUDIT_SAMPLE_RULE,
        "uses_detector_score": False,
        "why_not": (
            "a sample drawn on the detector's own score makes every recall number a "
            "measurement of the sampler. The stratum comes from generation metadata."
        ),
        "plan": dict(AUDIT_SAMPLE_PLAN),
        "minima": dict(AUDIT_SAMPLE_MINIMA),
        "n_rows_available_by_stratum": available,
        "n_rows_drawn_by_stratum": {s: len(v) for s, v in sorted(drawn.items())},
        "n_rows_sampled": len(sampled),
        "n_rows_in_bank": len(rows),
        "shortfalls": shortfalls,
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
        "files": [*sorted(written), str(output_dir / KEY_FILENAME)],
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
    typer.echo(f"wrote {output_dir / KEY_FILENAME}")
    typer.echo("")
    for stratum in STRATA:
        typer.echo(
            f"  {stratum:<26} drawn {len(drawn.get(stratum, ())):>5}  "
            f"available {available[stratum]:>6}  planned {AUDIT_SAMPLE_PLAN[stratum]:>5}"
        )
    if shortfalls:
        typer.echo("", err=True)
        for stratum, missing in sorted(shortfalls.items()):
            typer.echo(f"  [SHORT] {stratum}: {missing} row(s) short of the plan", err=True)
        if not allow_shortfall:
            raise typer.BadParameter(
                "the bank cannot supply the pre-registered audit sample. Generate more "
                "rows under a pre-registered extension rather than reducing the plan; "
                "pass --allow-shortfall only to write a NON-REPORTABLE audit for "
                "development."
            )
    raise typer.Exit(0 if reportable else 1)
