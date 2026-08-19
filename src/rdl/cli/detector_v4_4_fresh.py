"""The fresh engineering audit: the bridge from a v4.2 bank to a v4.4 judgeable row.

The v4.4 protocol says the frozen detector is gated on a *fresh* engineering bank, drawn
after training, judged by the same two pinned judges, and opened once. Every piece of that
existed except the join, and the join is not cosmetic -- the two file formats do not
overlap on a single field:

==========================  ==========================================================
``ENGINEERING_BANK.json``   ``partitions.{development,heldout}.{clean,leaking,retain}``,
                            rows of ``{text, request, item_id, population, text_sha256,
                            pair_sha256}``
v4.4 judge / gate           a flat list of ``{audit_id, conditioning_question,
                            subject_aliases, candidate_text}``, plus ``population``,
                            ``concept_id`` and a gold label as evaluation metadata
==========================  ==========================================================

``graph-detector-v4-4-audit-sample`` reads top-level ``rows`` or ``pairs``. A bank has
neither, so it drew zero rows and said the bank was empty -- which is the failure mode this
module exists to remove.

Three things are derived rather than carried, and each is derived the way the rest of the
protocol already derives it, so that the fresh rows and the v4.4 bundle rows are the same
kind of object:

``audit_id``
    ``fresh-`` plus a digest of ``bank_id`` and the row's ``pair_sha256``. Stable across
    re-runs of this command, distinct from any v4.4 bundle id, and derived from the pair
    digest rather than from the text so that the same refusal string under two different
    protected questions stays two rows.

``subject_aliases``
    from the all-row, population-blind conditioning index, by question text -- the same
    index the v4.4 bundle uses. A question the index does not carry gets its aliases from
    the SAME builder (``extract_name_spans``) rather than an empty tuple, because "retain
    rows have no aliases" is precisely the shortcut v4.2 shipped and v4.3 removed.

``concept_id``
    from TOFU's twenty-questions-per-author layout, via the rule
    :mod:`rdl.cli.detector_v4_3_store` already froze and checked against ``concept_id`` on
    the rows that have one. Protected rows must have it -- macro recall and correct-concept
    precision are computed per concept, and a protected row with no concept is a row that
    cannot enter either.

What must not happen here
-------------------------
The sampler never reads a detector score. Drawing the rows the detector is evaluated on by
asking the detector which rows to draw makes the gate a measurement of the sampler, and the
non-attempt enrichment is exactly where that temptation lives -- so enrichment runs off
frozen surface rules (:func:`~rdl.cli.detector_v4_4_gate.enrich_nonattempt`) and this module
imports no detector at all.

The blind judge inputs carry four fields. No reference answer, no population, no partition,
no concept, no bank bucket. Those live in the sealed reference key and in the audit
manifest, and the reference key is a separate file for the same reason v4.3's is: a file
that cannot be opened by the blind path cannot leak into it.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

import typer

from ..defenses.concept_registry import extract_name_spans
from ..defenses.protected_store import ConditioningIndex
from ..eval.detector_v4_4 import V4_4_PROTOCOL, sha256_text
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_store import (
    CONDITIONING_INDEX_FILENAME,
    TOFU_QUESTIONS_PER_AUTHOR,
)
from .detector_v4_3_store import (
    DEFAULT_OUT_DIR as V4_3_DIR,
)
from .detector_v4_4_bundle import DEFAULT_V4_4_DIR
from .detector_v4_4_gate import enrich_nonattempt

__all__ = [
    "FRESH_AUDIT_FILENAME",
    "FRESH_PLAN_FILENAME",
    "FRESH_PLAN_SCHEMA",
    "FRESH_REFERENCE_KEY_FILENAME",
    "PARTITIONS",
    "blind_filename",
    "canonical_rows",
    "concept_id_for_item",
    "detector_v4_4_fresh_audit",
    "detector_v4_4_fresh_audit_plan",
    "flatten_bank",
    "fresh_audit_id",
]

FRESH_PLAN_FILENAME = "DETECTOR_V4_4_FRESH_AUDIT_PLAN.json"
FRESH_AUDIT_FILENAME = "DETECTOR_V4_4_FRESH_AUDIT.json"
FRESH_REFERENCE_KEY_FILENAME = "DETECTOR_V4_4_FRESH_REFERENCE_KEY.json"

FRESH_PLAN_SCHEMA = "graph-detector-v4-4-fresh-audit-plan-v1"
FRESH_AUDIT_SCHEMA = "graph-detector-v4-4-fresh-audit-v1"
FRESH_REFERENCE_KEY_SCHEMA = "graph-detector-v4-4-fresh-reference-key-v1"

PARTITIONS: tuple[str, ...] = ("development", "heldout")
# The bank's own leak buckets. Carried as diagnostic metadata and never as a label: the
# bank's `leaking` flag is a v4.2 heuristic over generated text, and the v4.4 gold label is
# what two pinned judges say about the answer-attempt axis.
BUCKETS: tuple[str, ...] = ("clean", "leaking", "retain")

# Frozen defaults, per partition. Named here and written into the plan file so the sizes
# are a committed artifact rather than whatever the command defaults happened to be on the
# day it ran -- the difference between a pre-registered draw and a described one.
DEFAULT_N_ROWS = 1200
DEFAULT_MIN_NONATTEMPT = 400
DEFAULT_BASE_RATE = 0.25


def blind_filename(partition: str) -> str:
    return f"V4_4_FRESH_BLIND_{partition}.jsonl"


def fresh_audit_id(bank_id: str, pair_sha256: str) -> str:
    """A stable id for a fresh row: ``fresh-`` + digest(bank_id, pair digest).

    Prefixed so a fresh id can never be confused with a v4.4 bundle id in a joined file,
    and derived from the PAIR digest so that one candidate text under two protected
    questions remains two rows with two ids.
    """
    return "fresh-" + hashlib.sha256(f"{bank_id}\x1f{pair_sha256}".encode()).hexdigest()[:16]


def concept_id_for_item(item_id: str) -> str:
    """``tofu-forget10-author-0003`` for ``forget10-0069``, or ``""`` for a retain item.

    The same twenty-questions-per-author rule ``detector_v4_3_store`` froze. It is applied
    here only to rows the bank already marked ``population == "protected"``; retain rows
    get no concept, because on the retain split there is nothing to check the rule against
    and a fabricated concept would put unrelated rows in one recall denominator.
    """
    text = str(item_id or "").strip()
    prefix, _, index = text.rpartition("-")
    if not prefix or not index.isdigit():
        raise ValueError(
            f"cannot derive a concept from item_id {item_id!r}: expected '<split>-<index>'. "
            "A protected row whose concept cannot be derived cannot enter macro recall or "
            "correct-concept precision, and inventing one would silently merge concepts."
        )
    return f"tofu-{prefix}-author-{int(index) // TOFU_QUESTIONS_PER_AUTHOR:04d}"


def flatten_bank(payload: Mapping) -> list[dict]:
    """``partitions.<partition>.<bucket>[]`` -> one flat list carrying both keys.

    The nesting is the bank's whole design -- development and heldout are separate draws
    and must never be pooled -- so it is flattened by *carrying* the partition onto each
    row, never by concatenating the two.
    """
    partitions = payload.get("partitions") or {}
    out: list[dict] = []
    for partition in PARTITIONS:
        block = partitions.get(partition) or {}
        for bucket in BUCKETS:
            for row in block.get(bucket) or ():
                out.append({**row, "partition": partition, "bank_bucket": bucket})
    return out


def canonical_rows(
    bank_payload: Mapping,
    *,
    index: ConditioningIndex,
    max_aliases: int = 8,
) -> tuple[list[dict], dict]:
    """Flatten a bank into v4.4-shaped rows. ``(rows, report)``.

    Refuses rather than repairs on every condition that would corrupt a denominator:
    duplicate ids, duplicate pairs, a protected row with no derivable concept.
    """
    flat = flatten_bank(bank_payload)
    if not flat:
        raise typer.BadParameter(
            "the bank carries no rows under partitions.{development,heldout}."
            "{clean,leaking,retain}. This command reads the NESTED bank layout; a flat "
            "`rows` list is a different artifact."
        )
    bank_id = str(bank_payload.get("bank_id") or "")
    if not bank_id:
        raise typer.BadParameter("the bank carries no bank_id, so no stable audit_id exists.")

    rows: list[dict] = []
    n_alias_from_index = 0
    n_alias_derived = 0
    for row in flat:
        question = str(row["request"])
        record = index.for_question(question)
        if record is not None:
            aliases = list(record.subject_aliases)[:max_aliases]
            n_alias_from_index += 1
        else:
            # Same builder, same cap. NOT an empty tuple: a population whose rows carry no
            # aliases is separable from one whose rows do, without reading the candidate.
            aliases = list(extract_name_spans(question))[:max_aliases]
            n_alias_derived += 1
        population = str(row.get("population") or "protected")
        rows.append(
            {
                "audit_id": fresh_audit_id(bank_id, str(row["pair_sha256"])),
                # ---- the tokenized triple, identical in shape for both populations ----
                "conditioning_question": question,
                "subject_aliases": aliases,
                "candidate_text": str(row["text"]),
                # ---- evaluation metadata. Never tokenized, never a judge prompt field ----
                "population": population,
                "partition": str(row["partition"]),
                "concept_id": (
                    concept_id_for_item(str(row["item_id"])) if population == "protected" else ""
                ),
                "item_id": str(row["item_id"]),
                "bank_bucket": str(row["bank_bucket"]),
                "pair_sha256": str(row["pair_sha256"]),
                "text_sha256": str(row.get("text_sha256") or sha256_text(str(row["text"]))),
            }
        )

    ids = Counter(r["audit_id"] for r in rows)
    duplicate_ids = sorted(i for i, n in ids.items() if n > 1)
    if duplicate_ids:
        raise typer.BadParameter(
            f"{len(duplicate_ids)} duplicated audit_ids (first {duplicate_ids[:3]}). Two "
            "rows sharing an id would be judged once and counted twice."
        )
    pairs = Counter(r["pair_sha256"] for r in rows)
    duplicate_pairs = sorted(p for p, n in pairs.items() if n > 1)
    if duplicate_pairs:
        raise typer.BadParameter(
            f"{len(duplicate_pairs)} duplicated (text, question) pairs survived bank "
            "deduplication. The same pair in two partitions would put correlated rows on "
            "both sides of the one-shot boundary."
        )
    missing_concept = [
        r["audit_id"] for r in rows if r["population"] == "protected" and not r["concept_id"]
    ]
    if missing_concept:
        raise typer.BadParameter(
            f"{len(missing_concept)} protected rows have no concept_id (first "
            f"{missing_concept[:3]}). Macro recall and correct-concept precision are "
            "computed per concept."
        )

    report = {
        "n_rows": len(rows),
        "bank_id": bank_id,
        "by_partition": dict(sorted(Counter(r["partition"] for r in rows).items())),
        "by_population": dict(sorted(Counter(r["population"] for r in rows).items())),
        "by_bank_bucket": dict(sorted(Counter(r["bank_bucket"] for r in rows).items())),
        "n_concepts": len({r["concept_id"] for r in rows if r["concept_id"]}),
        "aliases": {
            "n_from_conditioning_index": n_alias_from_index,
            "n_derived_by_extract_name_spans": n_alias_derived,
            "n_without_aliases": sum(1 for r in rows if not r["subject_aliases"]),
            "n_without_aliases_by_population": dict(
                sorted(Counter(r["population"] for r in rows if not r["subject_aliases"]).items())
            ),
            "why_derived_rather_than_empty": (
                "a question the index does not carry still gets aliases from the same "
                "builder. Leaving them empty would make 'has aliases' separate the "
                "populations, which is the v4.2 shortcut v4.3 removed."
            ),
        },
    }
    return rows, report


# =====================================================================================
# the plan
# =====================================================================================


def detector_v4_4_fresh_audit_plan(
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    n_development: int = typer.Option(DEFAULT_N_ROWS, "--n-development"),
    n_heldout: int = typer.Option(DEFAULT_N_ROWS, "--n-heldout"),
    min_nonattempt_development: int = typer.Option(
        DEFAULT_MIN_NONATTEMPT, "--min-nonattempt-development"
    ),
    min_nonattempt_heldout: int = typer.Option(DEFAULT_MIN_NONATTEMPT, "--min-nonattempt-heldout"),
    base_rate: float = typer.Option(DEFAULT_BASE_RATE, "--base-rate"),
) -> None:
    """Freeze the fresh audit's sizes BEFORE the bank exists. Generates nothing.

    The sizes were previously implied by the sampler's defaults, which means they were
    whatever the command line said on the day it ran. A minimum that can be chosen after
    seeing the draw is not a minimum, so it is written here, hashed, and checked by
    ``graph-detector-v4-4-fresh-audit``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan: dict = {
        "schema": FRESH_PLAN_SCHEMA,
        "protocol": V4_4_PROTOCOL,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partitions": {
            "development": {
                "n_rows": int(n_development),
                "min_likely_nonattempt": int(min_nonattempt_development),
                "usage": "threshold selection ONLY -- tau_answer and tau_partial",
            },
            "heldout": {
                "n_rows": int(n_heldout),
                "min_likely_nonattempt": int(min_nonattempt_heldout),
                "usage": "opened ONCE, against the already-frozen detector",
            },
        },
        "base_rate": float(base_rate),
        "total_rows": int(n_development) + int(n_heldout),
        "enrichment": {
            "rule": "frozen surface rules only (detector_v4_4_gate.enrich_nonattempt)",
            "never": (
                "no detector score is read while sampling. A sampler that used the learned "
                "score to find non-attempts would draw the rows the detector already "
                "handles, and the gate would measure the sampler."
            ),
        },
        "why_two_partitions_in_one_file": (
            "two invocations that both wrote DETECTOR_V4_4_AUDIT_SAMPLE.json would leave "
            "the second overwriting the first, and the heldout draw is the one that may be "
            "opened exactly once. One manifest, both partitions, one hash."
        ),
        "not_a_substitute_for": (
            "the 600-row v4.4 calibration panel, which belongs to the ORIGINAL label "
            "authority and is not a fresh-audit gate."
        ),
    }
    plan["plan_sha256"] = sha256_text(
        dumps_canonical({k: v for k, v in plan.items() if k != "frozen_at"})
    )
    atomic_json(out / FRESH_PLAN_FILENAME, plan)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / FRESH_PLAN_FILENAME),
                "development": plan["partitions"]["development"],
                "heldout": plan["partitions"]["heldout"],
                "plan_sha256": plan["plan_sha256"],
            }
        )
    )


# =====================================================================================
# the draw
# =====================================================================================


def _reference_answers(cohort: Path) -> tuple[dict[str, str], dict]:
    """``({item_id: answer}, provenance)`` from the pinned TOFU revision.

    Imported lazily: it resolves the dataset, and neither the plan command nor any of the
    pure helpers above should acquire that dependency.
    """
    from .detector_v4_data import _question_bank

    _questions, answers, provenance = _question_bank(cohort)
    return answers, provenance


def detector_v4_4_fresh_audit(
    bank: Path = typer.Option(..., "--bank", help="ENGINEERING_BANK.json, closed and hashed."),
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    plan: Path = typer.Option(None, "--plan", help="the frozen fresh-audit plan."),
    conditioning_index: Path = typer.Option(
        V4_3_DIR / CONDITIONING_INDEX_FILENAME, "--conditioning-index"
    ),
    cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/discovery.json"),
        "--cohort",
        help="the pinned cohort the reference answers come from.",
    ),
    skip_reference_key: bool = typer.Option(
        False,
        "--skip-reference-key",
        help=(
            "build the blind inputs only, without resolving TOFU. The audit is marked "
            "non-reportable: the reference pass cannot run without the key."
        ),
    ),
) -> None:
    """Draw BOTH fresh partitions from the engineering bank, before any scoring."""
    out = Path(out_dir)
    plan_path = Path(plan) if plan else out / FRESH_PLAN_FILENAME
    if not plan_path.exists():
        raise typer.BadParameter(
            f"{plan_path} is absent. Freeze the sizes with "
            "`rdl graph-detector-v4-4-fresh-audit-plan` BEFORE the bank exists; a minimum "
            "chosen after the draw is not a minimum."
        )
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    if str(plan_payload.get("schema")) != FRESH_PLAN_SCHEMA:
        raise typer.BadParameter(f"{plan_path} carries schema {plan_payload.get('schema')!r}.")

    bank_payload = json.loads(Path(bank).read_text(encoding="utf-8"))
    index = ConditioningIndex.load(Path(conditioning_index))
    rows, report = canonical_rows(bank_payload, index=index)

    by_partition: dict[str, list[dict]] = {p: [] for p in PARTITIONS}
    for row in rows:
        by_partition[row["partition"]].append(row)

    drawn_by_partition: dict[str, list[dict]] = {}
    designs: dict[str, dict] = {}
    for partition in PARTITIONS:
        spec = (plan_payload.get("partitions") or {}).get(partition) or {}
        wanted = int(spec.get("n_rows") or 0)
        minimum = int(spec.get("min_likely_nonattempt") or 0)
        available = by_partition[partition]
        if not available:
            raise typer.BadParameter(
                f"the bank supplied no {partition} rows. The two partitions are separate "
                "draws and neither may be filled from the other."
            )
        drawn, design = enrich_nonattempt(
            available,
            n_wanted=wanted,
            base_rate=float(plan_payload.get("base_rate") or DEFAULT_BASE_RATE),
            salt=f"v4.4-fresh|{partition}",
        )
        n_likely = design["strata"]["likely_nonattempt"]["n_drawn"]
        if n_likely < minimum:
            raise typer.BadParameter(
                f"the {partition} draw contains {n_likely} likely-non-attempt rows against "
                f"a pre-registered minimum of {minimum}. Either the bank does not contain "
                "non-attempt messages -- in which case generate one that does -- or the "
                "minimum was wrong, in which case amend the PLAN and say so. Lowering it "
                "now to fit the draw is how a gate stops being a gate."
            )
        # Route coverage: a partition that reaches only some concepts cannot report a
        # zero-recall-concepts count that means anything.
        concepts_available = {r["concept_id"] for r in available if r["concept_id"]}
        concepts_drawn = {r["concept_id"] for r in drawn if r["concept_id"]}
        if concepts_available and not concepts_drawn:
            raise typer.BadParameter(
                f"the {partition} draw reached none of the {len(concepts_available)} "
                "protected concepts in that partition."
            )
        drawn_by_partition[partition] = drawn
        designs[partition] = {
            **design,
            "n_available": len(available),
            "n_drawn": len(drawn),
            "min_likely_nonattempt": minimum,
            "n_likely_nonattempt": n_likely,
            "n_concepts_available": len(concepts_available),
            "n_concepts_drawn": len(concepts_drawn),
        }

    all_drawn = [r for p in PARTITIONS for r in drawn_by_partition[p]]
    overlap = {r["audit_id"] for r in drawn_by_partition["development"]} & {
        r["audit_id"] for r in drawn_by_partition["heldout"]
    }
    if overlap:
        raise typer.BadParameter(
            f"{len(overlap)} audit_ids appear in BOTH partitions. Development chooses the "
            "threshold and heldout reports the result; a shared row makes the second a "
            "description of the first."
        )

    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------- the blind judge inputs --
    # Four fields. This is the entire surface a judge sees, and it is written from an
    # explicit dict rather than by deleting keys from the row -- a filter that stops
    # filtering is invisible, an allow-list that gains a field is a diff.
    blind_files: dict[str, dict] = {}
    for partition in PARTITIONS:
        path = out / blind_filename(partition)
        path.write_text(
            "".join(
                dumps_canonical(
                    {
                        "audit_id": r["audit_id"],
                        "conditioning_question": r["conditioning_question"],
                        "subject_aliases": r["subject_aliases"],
                        "candidate_text": r["candidate_text"],
                    }
                )
                + "\n"
                for r in drawn_by_partition[partition]
            ),
            encoding="utf-8",
        )
        blind_files[partition] = {
            "file": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "n_rows": len(drawn_by_partition[partition]),
            "fields": ["audit_id", "candidate_text", "conditioning_question", "subject_aliases"],
        }

    # ---------------------------------------------------- the sealed reference key --
    reference_block: dict = {"built": False, "why": "--skip-reference-key was set"}
    if not skip_reference_key:
        answers, provenance = _reference_answers(Path(cohort))
        key_rows = {
            r["audit_id"]: {
                "item_id": r["item_id"],
                "concept_id": r["concept_id"],
                "population": r["population"],
                "partition": r["partition"],
                "reference_answer": answers.get(r["item_id"], ""),
            }
            for r in all_drawn
        }
        key = {
            "schema": FRESH_REFERENCE_KEY_SCHEMA,
            "protocol": V4_4_PROTOCOL,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "source_cohort": str(cohort),
            "provenance": provenance,
            "n_rows": len(key_rows),
            "n_reference_answers": sum(1 for v in key_rows.values() if v["reference_answer"]),
            "never_loadable_by": (
                "rdl.defenses. This file carries the reference answers and the population; "
                "it is read by evaluators and by the reference judge pass, and by nothing "
                "that builds a runtime or training input."
            ),
            "rows": key_rows,
        }
        key_path = out / FRESH_REFERENCE_KEY_FILENAME
        atomic_json(key_path, key)
        reference_block = {
            "built": True,
            "file": str(key_path),
            "sha256": hashlib.sha256(key_path.read_bytes()).hexdigest(),
            "n_reference_answers": key["n_reference_answers"],
            "provenance": provenance,
        }

    manifest = {
        "schema": FRESH_AUDIT_SCHEMA,
        "protocol": V4_4_PROTOCOL,
        "drawn_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bank": str(bank),
        "bank_sha256": hashlib.sha256(Path(bank).read_bytes()).hexdigest(),
        "bank_content_sha256": bank_payload.get("content_sha256"),
        "plan": str(plan_path),
        "plan_sha256": plan_payload.get("plan_sha256"),
        "conditioning_index": str(conditioning_index),
        "conditioning_index_fingerprint_sha256": index.fingerprint(),
        "canonicalisation": report,
        "designs": designs,
        "blind_inputs": blind_files,
        "reference_key": reference_block,
        "reportable": bool(reference_block["built"]),
        "n_rows": len(all_drawn),
        "scores_read_while_sampling": False,
        "labels_are_not_here": (
            "this file carries no answer_attempt label. The gate is scored against the "
            "adjudicated labels `rdl graph-detector-v4-4-fresh-label-report` produces from "
            "two blind and two reference judge passes over the files above."
        ),
        "rows": all_drawn,
    }
    manifest["audit_sha256"] = sha256_text(
        dumps_canonical({k: v for k, v in manifest.items() if k != "drawn_at"})
    )
    atomic_json(out / FRESH_AUDIT_FILENAME, manifest)
    typer.echo(
        dumps_canonical(
            {
                "wrote": [
                    str(out / FRESH_AUDIT_FILENAME),
                    *[b["file"] for b in blind_files.values()],
                    *([reference_block["file"]] if reference_block["built"] else []),
                ],
                "n_rows": len(all_drawn),
                "by_partition": {p: len(drawn_by_partition[p]) for p in PARTITIONS},
                "n_likely_nonattempt": {p: designs[p]["n_likely_nonattempt"] for p in PARTITIONS},
                "n_concepts": report["n_concepts"],
                "reportable": manifest["reportable"],
                "audit_sha256": manifest["audit_sha256"],
            }
        )
    )
