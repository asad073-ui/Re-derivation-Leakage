"""``rdl graph-detector-v4-3-build-store`` -- the three v4.3 artifacts, split by audience.

One command, three files, because the three have different readers and exactly one of them
may be read at runtime:

``PROTECTED_STORE_RUNTIME.json``
    The protected set a deployed GraphForget holds. Questions, safe aliases, policy
    actions. No answers, no answer digests. Loadable by ``rdl.defenses``.

``DETECTOR_V4_3_CONDITIONING_INDEX.json``
    Every question in the 1,019-row audit -- protected *and* retain -- under one alias
    builder. Answer-free and population-free. Loadable by the bundle builder, which turns
    it into tokenizer inputs.

``PROTECTED_STORE_EVAL_KEY.json``
    The sealed side: item ids, concept ids, population, stratum, subject groups, and the
    reference answers themselves. Read by evaluators and by nothing else. Its schema is
    deliberately unknown to :mod:`rdl.defenses.protected_store`, which contains no code
    that can open it.

Where the subject group comes from, and why it is checked rather than trusted
-----------------------------------------------------------------------------
v4.2 split the natural rows by ``audit_id`` parity and recorded ``group=audit_id``, so two
questions about one author could land on opposite sides of the train/development boundary.
A model that has seen "Hsiao Yun-Hwa's father was a civil engineer" in training is not
being tested on a held-out concept when development asks about Hsiao Yun-Hwa's father's
profession, and the resulting number describes memorisation rather than a learned
question--answer relation.

The fix needs a subject group, and TOFU supplies one structurally: its items run twenty
consecutive questions per author, so ``item_index // 20`` *is* the author. This module
derives the group that way for every row, protected and retain alike, and then **checks it
against ``concept_id``** on the rows that have one. On the frozen audit the check passes
exactly -- all 20 protected blocks map to ``tofu-forget10-author-{block:04d}`` and no block
spans two concepts -- which is what licenses using the same rule on the 300 retain rows,
where ``concept_id`` is empty and there is nothing to check against. An independent check
agrees: across the 45 retain blocks, no extracted name span appears in two blocks.

If the check ever fails the command refuses. A silently wrong group is worse than no group,
because it produces a split that *looks* concept-disjoint in the manifest.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

import typer

from ..defenses.concept_registry import extract_name_spans
from ..defenses.protected_store import (
    ConditioningIndex,
    ConditioningRecord,
    ProtectedScope,
    ProtectedStore,
    subject_id_for,
)
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json

__all__ = [
    "CONDITIONING_INDEX_FILENAME",
    "EVAL_KEY_FILENAME",
    "EVAL_KEY_SCHEMA",
    "RUNTIME_STORE_FILENAME",
    "TOFU_QUESTIONS_PER_AUTHOR",
    "detector_v4_3_build_store",
    "subject_groups",
]

RUNTIME_STORE_FILENAME = "PROTECTED_STORE_RUNTIME.json"
EVAL_KEY_FILENAME = "PROTECTED_STORE_EVAL_KEY.json"
CONDITIONING_INDEX_FILENAME = "DETECTOR_V4_3_CONDITIONING_INDEX.json"

EVAL_KEY_SCHEMA = "graph-detector-v4-3-protected-store-eval-key-v1"

# TOFU is laid out as twenty consecutive questions per author. This is the whole basis of
# the subject group, so it is named once and checked against `concept_id` below rather
# than assumed at each use.
TOFU_QUESTIONS_PER_AUTHOR = 20

DEFAULT_AUDIT_DIR = Path("data/cohorts/graph_unlearning_v1/detector_v4_1")
DEFAULT_OUT_DIR = Path("data/cohorts/graph_unlearning_v1/detector_v4_3")

# The dataset the protected set is drawn from. Recorded in every runtime scope so a store
# can be told apart from a store built over a different forget split.
DEFAULT_DATASET_ID = "locuslab/TOFU"
DEFAULT_FORGET_SPLIT = "forget10"


def _item_block(item_id: str) -> tuple[str, int]:
    """``(split_prefix, author_block)`` for a TOFU item id like ``forget10-0004``.

    Raises rather than defaulting. An item id this function cannot parse is an item id
    whose author is unknown, and inventing a group for it would put an unknown number of
    rows about one author on both sides of the split -- the exact failure this exists to
    prevent.
    """
    text = str(item_id or "").strip()
    prefix, _, index = text.rpartition("-")
    if not prefix or not index.isdigit():
        raise typer.BadParameter(
            f"cannot derive a subject group from item_id {item_id!r}: expected "
            "'<split>-<index>' such as 'forget10-0004'. The subject group is what makes "
            "the train/development split concept-disjoint; guessing one would produce a "
            "split that only looks disjoint in the manifest."
        )
    return prefix, int(index) // TOFU_QUESTIONS_PER_AUTHOR


def subject_groups(key_rows: Mapping[str, Mapping]) -> tuple[dict[str, str], dict]:
    """``({audit_id: group_key}, check report)`` for every audited row.

    One rule for both populations: the TOFU author block. The report records whether that
    rule agreed with ``concept_id`` wherever a ``concept_id`` exists, and the caller
    refuses on disagreement.
    """
    groups: dict[str, str] = {}
    block_to_concepts: dict[str, set[str]] = defaultdict(set)
    concept_to_blocks: dict[str, set[str]] = defaultdict(set)

    for audit_id, row in key_rows.items():
        prefix, block = _item_block(str(row.get("item_id", "")))
        group = f"{prefix}#{block:04d}"
        groups[audit_id] = group
        concept = str(row.get("concept_id") or "")
        if concept:
            block_to_concepts[group].add(concept)
            concept_to_blocks[concept].add(group)

    split_concepts = {c: sorted(b) for c, b in concept_to_blocks.items() if len(b) > 1}
    merged_blocks = {g: sorted(c) for g, c in block_to_concepts.items() if len(c) > 1}
    populations = Counter(str(r.get("population", "protected")) for r in key_rows.values())
    return groups, {
        "rule": f"item index // {TOFU_QUESTIONS_PER_AUTHOR}, per split prefix",
        "n_rows": len(groups),
        "n_groups": len(set(groups.values())),
        "n_groups_with_a_concept_id": len(block_to_concepts),
        "concepts_spanning_more_than_one_group": split_concepts,
        "groups_spanning_more_than_one_concept": merged_blocks,
        "agrees_with_concept_id": not split_concepts and not merged_blocks,
        "populations": dict(sorted(populations.items())),
        "why": (
            "the same rule is applied to protected and retain rows. It is CHECKED against "
            "concept_id on the protected rows, which is what licenses trusting it on the "
            "retain rows, where concept_id is empty."
        ),
    }


def _load_blind(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise typer.BadParameter(
            f"{path} is absent. The blinded judge file carries the question text; the key "
            "carries only its hash, so without it no store can be built."
        )
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {str(r["audit_id"]): r for r in rows}


def _load_reference_answers(path: Path) -> dict[str, str]:
    """``{audit_id: reference_answer}`` from a reference pass, if one has been run.

    This file is answer-BEARING, which is why it is read here -- in the offline key builder
    -- and nowhere near the runtime store or a blind judge prompt. Absent is fine: the
    eval key then records that the answers were not available, rather than pretending
    the audit had none.
    """
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        answer = str(row.get("reference_answer") or "")
        if answer:
            out[str(row["audit_id"])] = answer
    return out


def detector_v4_3_build_store(
    audit_dir: Path = typer.Option(
        DEFAULT_AUDIT_DIR,
        "--audit-dir",
        help="the frozen v4.1 label audit: LABEL_AUDIT_KEY.json plus LABEL_AUDIT_JUDGE_A.jsonl.",
    ),
    out_dir: Path = typer.Option(
        DEFAULT_OUT_DIR, "--out-dir", help="where the three v4.3 artifacts are written."
    ),
    dataset_id: str = typer.Option(DEFAULT_DATASET_ID, "--dataset-id"),
    dataset_revision: str = typer.Option(
        "",
        "--dataset-revision",
        help="the immutable dataset revision. Recorded verbatim; empty is recorded as empty.",
    ),
    forget_split: str = typer.Option(DEFAULT_FORGET_SPLIT, "--forget-split"),
    policy_version: str = typer.Option("forget-policy-v1", "--policy-version"),
    max_aliases: int = typer.Option(4, "--max-aliases"),
) -> None:
    """Build the runtime store, the conditioning index and the sealed evaluation key."""
    key_path = Path(audit_dir) / "LABEL_AUDIT_KEY.json"
    blind_path = Path(audit_dir) / "LABEL_AUDIT_JUDGE_A.jsonl"
    reference_path = Path(audit_dir) / "LABEL_AUDIT_REFERENCE_PASS_A.jsonl"
    if not key_path.exists():
        raise typer.BadParameter(f"{key_path} is absent")

    key_rows: dict[str, dict] = dict(
        json.loads(key_path.read_text(encoding="utf-8")).get("rows", {})
    )
    blind_rows = _load_blind(blind_path)

    missing = sorted(set(key_rows) - set(blind_rows))
    if missing:
        raise typer.BadParameter(
            f"{len(missing)} audited row(s) have a key entry but no blinded text, first "
            f"{missing[:3]}. Every one of the 1,019 rows must join, or the bundle silently "
            "describes a subset."
        )

    groups, group_report = subject_groups(key_rows)
    if not group_report["agrees_with_concept_id"]:
        raise typer.BadParameter(
            "the subject-group rule disagrees with concept_id: "
            f"concepts spanning >1 group {group_report['concepts_spanning_more_than_one_group']}, "
            f"groups spanning >1 concept {group_report['groups_spanning_more_than_one_concept']}. "
            "Refusing to write a store whose groups would make a train/development split "
            "look concept-disjoint without being it."
        )

    # ------------------------------------------------------------ conditioning index --
    #
    # Keyed by DISTINCT question. Two audit rows that ask the same question of different
    # candidates share one conditioning record, which is what makes the record a property
    # of the question rather than of the row.
    question_of = {aid: str(blind_rows[aid]["protected_question"]) for aid in key_rows}
    group_of_question: dict[str, set[str]] = defaultdict(set)
    for audit_id, question in question_of.items():
        group_of_question[question].add(groups[audit_id])

    ambiguous = {q: sorted(g) for q, g in group_of_question.items() if len(g) > 1}
    if ambiguous:
        raise typer.BadParameter(
            f"{len(ambiguous)} question(s) map to more than one subject group, e.g. "
            f"{list(ambiguous.items())[:2]}. A question in two groups cannot be assigned "
            "to one side of a group-disjoint split."
        )

    records: list[ConditioningRecord] = []
    for question in sorted(group_of_question):
        group = next(iter(group_of_question[question]))
        spans = extract_name_spans(question)[:max_aliases]
        records.append(
            ConditioningRecord(
                conditioning_id="cond-" + _digest(question),
                subject_id=subject_id_for(group),
                conditioning_question=question,
                subject_aliases=tuple(spans),
                relation="",
                template_id="extract_name_spans/v1",
            )
        )
    index = ConditioningIndex(records)

    # ------------------------------------------------------------------ runtime store --
    #
    # Protected scopes only. A retain question conditions training through the index above
    # and must never become a runtime scope: putting it in the store would make a retain
    # request route, which is precisely the end-to-end false alarm the protocol measures.
    protected_questions: dict[str, set[str]] = defaultdict(set)
    for audit_id, row in key_rows.items():
        if str(row.get("population", "protected")) == "retain":
            continue
        concept = str(row.get("concept_id") or "")
        if not concept:
            raise typer.BadParameter(
                f"protected row {audit_id} carries no concept_id, so the forget_id "
                "enforcement acts on is unknown."
            )
        protected_questions[concept].add(question_of[audit_id])

    scopes: list[ProtectedScope] = []
    for forget_id in sorted(protected_questions):
        for i, question in enumerate(sorted(protected_questions[forget_id])):
            record = index.for_question(question)
            assert record is not None  # every question was indexed above
            scopes.append(
                ProtectedScope(
                    dataset_id=str(dataset_id),
                    dataset_revision=dataset_revision,
                    forget_split=forget_split,
                    policy_version=policy_version,
                    forget_id=forget_id,
                    scope_id=f"{forget_id}#{i:03d}",
                    subject_id=record.subject_id,
                    question=question,
                    aliases=record.subject_aliases,
                    relation="",
                    template_id="v4.3-audit-question",
                    allow_refusal=True,
                    allow_persistent_write=False,
                    allow_edge_release=False,
                    allow_retrieval=False,
                    provenance_sha256=_digest_full(f"{forget_id}|{question}"),
                )
            )
    store = ProtectedStore(scopes)

    # -------------------------------------------------------------------- eval key --
    reference_answers = _load_reference_answers(reference_path)
    eval_rows = {
        audit_id: {
            "item_id": str(row.get("item_id", "")),
            "concept_id": str(row.get("concept_id") or ""),
            "population": str(row.get("population", "protected")),
            "stratum": str(row.get("stratum", "")),
            "bank_partition": str(row.get("bank_partition", "")),
            "subject_group": groups[audit_id],
            "subject_id": subject_id_for(groups[audit_id]),
            "conditioning_id": "cond-" + _digest(question_of[audit_id]),
            "text_sha256": str(row.get("text_sha256", "")),
            "reference_answer": reference_answers.get(audit_id, ""),
        }
        for audit_id, row in sorted(key_rows.items())
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    store_fingerprint = store.save(out / RUNTIME_STORE_FILENAME)
    index_fingerprint = index.save(out / CONDITIONING_INDEX_FILENAME)

    # Alias coverage BY POPULATION. The pooled number cannot show the v4.2 defect: an
    # index that gave every protected row aliases and no retain row any would report 70%
    # coverage and look unremarkable.
    by_population: dict[str, Counter] = defaultdict(Counter)
    for audit_id, row in key_rows.items():
        record = index.for_question(question_of[audit_id])
        assert record is not None
        population = str(row.get("population", "protected"))
        by_population[population]["n"] += 1
        by_population[population]["with_aliases"] += int(bool(record.subject_aliases))
    alias_parity = {
        population: {
            "n_rows": counts["n"],
            "n_with_aliases": counts["with_aliases"],
            "alias_coverage": counts["with_aliases"] / counts["n"] if counts["n"] else None,
        }
        for population, counts in sorted(by_population.items())
    }

    atomic_json(
        out / EVAL_KEY_FILENAME,
        {
            "schema": EVAL_KEY_SCHEMA,
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "never_loadable_by": "rdl.defenses.*",
            "why": (
                "the sealed half of the store. Reference answers, item ids, population and "
                "strata live here so that no runtime module can reach them: "
                "rdl.defenses.protected_store contains no code that opens this schema, "
                "and a contract test asserts it never will."
            ),
            "source_key": str(key_path),
            "source_reference_pass": str(reference_path) if reference_answers else None,
            "n_reference_answers": len(reference_answers),
            "runtime_store_fingerprint_sha256": store_fingerprint,
            "conditioning_index_fingerprint_sha256": index_fingerprint,
            "subject_groups": group_report,
            "alias_parity_by_population": alias_parity,
            "rows": eval_rows,
        },
    )

    typer.echo(
        dumps_canonical(
            {
                "wrote": {
                    "runtime_store": str(out / RUNTIME_STORE_FILENAME),
                    "conditioning_index": str(out / CONDITIONING_INDEX_FILENAME),
                    "eval_key": str(out / EVAL_KEY_FILENAME),
                },
                "n_audited_rows": len(key_rows),
                "n_scopes": len(store),
                "n_forget_ids": len(store.forget_ids()),
                "n_conditioning_records": len(index),
                "n_subject_groups": group_report["n_groups"],
                "subject_group_rule_agrees_with_concept_id": True,
                "alias_parity_by_population": alias_parity,
                "runtime_store_fingerprint_sha256": store_fingerprint,
                "conditioning_index_fingerprint_sha256": index_fingerprint,
            }
        )
    )


def _digest(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _digest_full(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
