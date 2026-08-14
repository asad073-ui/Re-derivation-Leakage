"""``rdl graph-detector-v4-label-audit`` / ``…-label-report`` — the blinded label audit.

Why an audit rather than another detector
-----------------------------------------
Detector v4 measured a *policy* question — "does this candidate attempt to answer the
protected question?" — against a *truth-content* label — the run's NLI+ROUGE verdict on
whether the candidate reproduces the reference answer. Those labels disagree on the exact
population Goal A cares most about: "X was born in Rome" for an author born in Madrid is
ANSWER under the policy and *clean* under the scorer. v4's operating point was selected
under a ceiling on ``generated_clean_fpr``, so the detector was penalised for obeying its
own specification, and the penalty is what chose the threshold.

Neither the scorer nor a detector can settle which label is right. A human can. This is the
instrument that asks one.

Blinding, and why it is structural
----------------------------------
A judge sees a protected question and a candidate message. Not the NLI label, not the
partition, not any detector score, not the item id, not the concept id — none of which a
person needs to answer "does this attempt to answer that", and all of which would anchor
the answer. The unblinding key is written to its own file, marked ``judges_must_not_read``,
in the same relationship ``DETECTOR_V4_ANSWER_KEY.json`` has to ``DETECTOR_V4_DATASET.json``.

The reference answer is a special case, and gets its own pass. ``answer_attempt`` must be
judged *without* it, because a judge who has seen it can no longer separate "attempts an
answer" from "gets the answer right" — that separation is the entire content of the
correction. ``reference_content`` is judged *with* it, in a second file.

The hard-negative stratum
-------------------------
200 of the clean rows are the highest-scoring under the lexical detector. That score is a
**sampling device only**: it decides which rows a human looks at and contributes nothing to
any reported rate. It is there because wrong answer attempts — the mislabelled population —
concentrate where a content detector already fires and the scorer said clean. The stratum
is named in every artifact so the selection is visible.

Nothing here modifies ``DETECTOR_V4_NATURAL_BANK.json``. The judgements are an overlay
keyed by ``text_sha256``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..defenses.answerability_detector import LexicalAnswerabilityDetector
from ..defenses.concept_registry import ConceptPolicy, ConceptRegistry
from ..defenses.detection_context import ProtectedQuestion, build_context
from ..defenses.identity_router import alias_index
from ..eval.detector_v4_1 import (
    AUDIT_FIELDS,
    AUDIT_SCHEMA,
    CEILING_REINTERPRETATION,
    MANIFEST_SCHEMA,
    STRATA,
    STRATUM_SIZES,
    adjudicate,
    alignment_report,
)
from ..studies.graph_leak.cohort import load_cohort
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT
from .detector_v4_data import ANSWER_KEY_FILENAME, NATURAL_BANK_FILENAME
from .detector_v4_data import DEFAULT_OUT as V4_DIR

__all__ = ["detector_v4_label_audit", "detector_v4_label_report"]

MANIFEST_FILENAME = "LABEL_AUDIT_MANIFEST.json"
KEY_FILENAME = "LABEL_AUDIT_KEY.json"
JUDGE_FILENAME = "LABEL_AUDIT_JUDGE_{judge}.jsonl"
REFERENCE_PASS_FILENAME = "LABEL_AUDIT_REFERENCE_PASS_{judge}.jsonl"
ADJUDICATED_FILENAME = "LABEL_AUDIT_ADJUDICATED.jsonl"
REPORT_FILENAME = "LABEL_ALIGNMENT_REPORT.json"

# The four fields judged without the reference answer, and the one judged with it.
BLIND_FIELDS = tuple(f for f, s in AUDIT_FIELDS.items() if not s["reference_answer_visible"])
REFERENCE_FIELDS = tuple(f for f, s in AUDIT_FIELDS.items() if s["reference_answer_visible"])


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _audit_id(text_sha256: str, request: str) -> str:
    return _sha(f"v4.1-audit|{text_sha256}|{request}")[:16]


def _order(rows: Sequence[Mapping], salt: str) -> list[Mapping]:
    """Deterministic, content-addressed ordering. No RNG, so no seed to lose.

    Salted per consumer so the two judges do not see the same sequence: a shared order
    lets a shared drift in attention look like agreement.
    """
    return sorted(rows, key=lambda r: _sha(f"{salt}|{r['audit_id']}"))


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temp.replace(path)


def _read_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


# ------------------------------------------------------------------------ sampling --


def _decile_bounds(lengths: Sequence[int]) -> list[int]:
    """Nine cut points over the observed lengths. Empty input gives no cuts."""
    ordered = sorted(lengths)
    if not ordered:
        return []
    return [ordered[min(len(ordered) - 1, (len(ordered) * i) // 10)] for i in range(1, 10)]


def _decile(length: int, bounds: Sequence[int]) -> int:
    return sum(1 for bound in bounds if length >= bound)


def select_strata(
    leaking: Sequence[Mapping],
    clean: Sequence[Mapping],
    retain: Sequence[Mapping],
    *,
    sizes: Mapping[str, int] | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Partition the bank into the five pre-registered strata.

    Pure and deterministic: every choice is a sha256 bucket of the row's own key, so the
    annotation set is reproducible from the bank alone and there is no seed to record.
    Rows already taken by an earlier stratum are never re-offered, so a row belongs to
    exactly one stratum and the per-stratum counts add up.
    """
    sizes = dict(sizes or STRATUM_SIZES)
    taken: set[str] = set()
    out: dict[str, list[dict]] = {}
    shortfall: dict[str, int] = {}

    def take(name: str, pool: Sequence[Mapping], ordered: Sequence[Mapping] | None = None) -> None:
        want = sizes.get(name, 0)
        source = list(ordered if ordered is not None else _order(pool, name))
        chosen: list[dict] = []
        for row in source:
            if len(chosen) >= want:
                break
            if row["audit_id"] in taken:
                continue
            taken.add(row["audit_id"])
            chosen.append({**row, "stratum": name})
        out[name] = chosen
        if len(chosen) < want:
            shortfall[name] = want - len(chosen)

    # 1. Every row the scorer called leaking. Not a sample: 120 is the population.
    take("natural_leaking", leaking)

    # 2. Where the mislabelling concentrates — the content detector already fires and the
    #    scorer said clean.
    take(
        "clean_hard_negative",
        clean,
        ordered=sorted(clean, key=lambda r: (-float(r.get("lexical_score", 0.0)), r["audit_id"])),
    )

    # 3. Matched to the leaking rows on protected question and length decile, so a
    #    difference between the two pools is not a difference in what was asked or in how
    #    much was said.
    bounds = _decile_bounds([len(str(r["text"])) for r in clean])
    by_key: dict[tuple[str, int], list[Mapping]] = {}
    for row in _order(clean, "matched"):
        by_key.setdefault((str(row["request"]), _decile(len(str(row["text"])), bounds)), []).append(
            row
        )
    matched: list[Mapping] = []
    cursors: dict[tuple[str, int], int] = {}
    targets = [
        (str(r["request"]), _decile(len(str(r["text"])), bounds)) for r in _order(leaking, "target")
    ]
    for _round in range(sizes.get("clean_matched", 0)):
        progressed = False
        for key in targets:
            if len(matched) >= sizes.get("clean_matched", 0):
                break
            pool = by_key.get(key, ())
            i = cursors.get(key, 0)
            while i < len(pool) and pool[i]["audit_id"] in taken:
                i += 1
            cursors[key] = i + 1
            if i < len(pool):
                matched.append(pool[i])
                taken.add(pool[i]["audit_id"])
                progressed = True
        if not progressed or len(matched) >= sizes.get("clean_matched", 0):
            break
    # Exact matches can run out before the quota; fall back to the same question at any
    # length rather than dropping the stratum, and say so in the manifest.
    n_exact = len(matched)
    if len(matched) < sizes.get("clean_matched", 0):
        wanted_questions = {q for q, _d in targets}
        for row in _order(clean, "matched-fallback"):
            if len(matched) >= sizes.get("clean_matched", 0):
                break
            if row["audit_id"] in taken or str(row["request"]) not in wanted_questions:
                continue
            taken.add(row["audit_id"])
            matched.append(row)
    out["clean_matched"] = [{**r, "stratum": "clean_matched"} for r in matched]
    if len(matched) < sizes.get("clean_matched", 0):
        shortfall["clean_matched"] = sizes.get("clean_matched", 0) - len(matched)

    # 4. Uniform over what is left, so the two hard strata cannot be mistaken for the
    #    population they were drawn from.
    take("clean_random", clean)

    # 5. Retain-author traffic: the retain-FPR population.
    take("retain", retain)

    provenance = {
        "sizes_requested": sizes,
        "sizes_selected": {k: len(v) for k, v in out.items()},
        "shortfall": shortfall,
        # A shortfall is usually a DUPLICATE, not a missing row: the bank holds the same
        # message under the same request more than once, and one text gets one judgement.
        # Recorded per pool so "119 of 120 leaking" reads as arithmetic rather than a bug.
        "pool_sizes": {
            "leaking": len(leaking),
            "clean": len(clean),
            "retain": len(retain),
        },
        "distinct_audit_ids": {
            "leaking": len({r["audit_id"] for r in leaking}),
            "clean": len({r["audit_id"] for r in clean}),
            "retain": len({r["audit_id"] for r in retain}),
        },
        "clean_matched_exact": n_exact,
        "clean_matched_fallback": len(matched) - n_exact,
        "ordering": "sha256(salt|audit_id); no RNG and no seed",
        "disjoint": "a row belongs to exactly one stratum; earlier strata claim first",
        "length_decile_bounds_chars": bounds,
    }
    return out, provenance


# ------------------------------------------------------------------ bank -> rows --


def _registry_from_bank(bank: Mapping, cohort_path: Path) -> tuple[ConceptRegistry, dict, dict]:
    """Registry and ``{item_id: concept_id}``, built offline from committed files only.

    The gate command derives its registry from the pinned TOFU revision. This one derives
    it from the bank's own request strings joined to the frozen cohort's item→concept map,
    which needs no dataset load and therefore no network. The two can differ on items the
    bank never sampled; that is acceptable *here* and nowhere else, because the registry is
    used only to compute the hard-negative sampling score.
    """
    cohort = load_cohort(cohort_path, exclusions_path=None)
    concept_of = {e.item_id: e.concept_id for e in cohort.items}
    seen: dict[tuple[str, str], str] = {}
    for partition in ("development", "heldout"):
        block = bank["partitions"].get(partition, {})
        for keyname in ("clean", "leaking"):
            for row in block.get(keyname, ()):
                item_id = str(row.get("item_id", ""))
                concept = concept_of.get(item_id)
                if concept and row.get("request"):
                    seen[(item_id, str(row["request"]))] = concept
    rows = [
        {"item_id": item_id, "concept_id": concept, "question": question}
        for (item_id, question), concept in sorted(seen.items())
    ]
    registry = ConceptRegistry.from_questions(rows, policy=ConceptPolicy())
    meta = {
        "source": "natural bank request strings joined to the frozen cohort item->concept map",
        "offline": True,
        "cohort": str(cohort_path),
        "cohort_fingerprint": cohort.fingerprint(),
        "registry_fingerprint": registry.fingerprint(),
        "n_concepts": len(registry),
        "n_questions": len(rows),
        "role": "sampling only; contributes to no reported rate",
        "stores_gold_answers": False,
    }
    return registry, concept_of, meta


def _protected_questions(registry: ConceptRegistry, bank_rows: Sequence[Mapping]) -> list:
    by_concept: dict[str, list[str]] = {}
    for row in bank_rows:
        concept = str(row.get("concept_id", ""))
        request = str(row.get("request", ""))
        if concept and request:
            by_concept.setdefault(concept, []).append(request)
    out: list[ProtectedQuestion] = []
    for entry in registry.concepts():
        for i, question in enumerate(dict.fromkeys(by_concept.get(entry.forget_id, ()))):
            out.append(
                ProtectedQuestion(
                    scope_id=f"{entry.forget_id}#{i:03d}",
                    forget_id=entry.forget_id,
                    question=question,
                    aliases=tuple(entry.aliases),
                    template_id="cohort",
                )
            )
    return out


def _lexical_scores(rows: Sequence[Mapping], questions: Sequence, index: Mapping) -> list[float]:
    """One routing decision per distinct request, reused across that request's messages."""
    detector = LexicalAnswerabilityDetector()
    by_request: dict[str, list[int]] = {}
    for i, row in enumerate(rows):
        by_request.setdefault(str(row.get("request", "")), []).append(i)
    scores = [0.0] * len(rows)
    for request, idxs in by_request.items():
        context = build_context(request, protected_questions=questions, alias_index=index)
        for i in idxs:
            scores[i] = detector.score_batch([str(rows[i]["text"])], context=context)[
                0
            ].answer_probability
    return scores


def _bank_rows(
    bank: Mapping, concept_of: Mapping[str, str]
) -> tuple[list[dict], list[dict], list[dict]]:
    """``(leaking, clean, retain)`` with the hidden fields still attached."""

    def rows_of(partition: str, keyname: str, *, leaking: bool | None) -> list[dict]:
        block = bank["partitions"].get(partition, {})
        out = []
        for row in block.get(keyname, ()):
            text = str(row["text"])
            request = str(row.get("request", ""))
            text_sha = str(row.get("text_sha256") or _sha(text))
            item_id = str(row.get("item_id", ""))
            out.append(
                {
                    "audit_id": _audit_id(text_sha, request),
                    "text_sha256": text_sha,
                    "text": text,
                    "request": request,
                    "item_id": item_id,
                    "concept_id": concept_of.get(item_id, ""),
                    "nli_leaking": leaking,
                    "population": "retain" if partition == "retain" else "protected",
                    "bank_partition": partition,
                }
            )
        return out

    leaking = rows_of("development", "leaking", leaking=True) + rows_of(
        "heldout", "leaking", leaking=True
    )
    clean = rows_of("development", "clean", leaking=False) + rows_of(
        "heldout", "clean", leaking=False
    )
    retain = rows_of("retain", "all", leaking=None)
    return leaking, clean, retain


# --------------------------------------------------------------------- build audit --


def detector_v4_label_audit(
    data_dir: Path = typer.Option(V4_DIR, "--data-dir"),
    output_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--output-dir"),
    policy_cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/discovery.json"), "--policy-cohort"
    ),
    reference_pass: bool = typer.Option(
        True,
        "--reference-pass/--no-reference-pass",
        help="also emit the second-pass file that shows the reference answer",
    ),
) -> None:
    """Build the blinded annotation set. Reads the natural bank; modifies nothing."""
    bank_path = data_dir / NATURAL_BANK_FILENAME
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    registry, concept_of, registry_meta = _registry_from_bank(bank, policy_cohort)

    leaking, clean, retain = _bank_rows(bank, concept_of)
    questions = _protected_questions(registry, [*leaking, *clean])
    index = alias_index(registry)
    for row, score in zip(clean, _lexical_scores(clean, questions, index), strict=True):
        row["lexical_score"] = score

    strata, sampling = select_strata(leaking, clean, retain)
    selected = [row for name in STRATA for row in strata.get(name, ())]

    # The blinded row: a question and a message. Everything else is in the key file.
    blind = [
        {
            "audit_id": row["audit_id"],
            "protected_question": row["request"],
            "candidate_text": row["text"],
            **dict.fromkeys(BLIND_FIELDS),
        }
        for row in selected
    ]
    by_id = {row["audit_id"]: row for row in blind}

    for judge in ("A", "B"):
        path = output_dir / JUDGE_FILENAME.format(judge=judge)
        _write_jsonl(path, [by_id[r["audit_id"]] for r in _order(selected, f"judge-{judge}")])
        typer.echo(f"wrote {path}  ({len(selected)} rows)")

    reference_written: list[str] = []
    answers: dict[str, str] = {}
    key_path = data_dir / ANSWER_KEY_FILENAME
    if reference_pass and key_path.exists():
        answers = dict(json.loads(key_path.read_text(encoding="utf-8")).get("natural_answers", {}))
        for judge in ("A", "B"):
            path = output_dir / REFERENCE_PASS_FILENAME.format(judge=judge)
            _write_jsonl(
                path,
                [
                    {
                        "audit_id": row["audit_id"],
                        "protected_question": row["request"],
                        "candidate_text": row["text"],
                        "reference_answer": answers.get(row["item_id"], ""),
                        **dict.fromkeys(REFERENCE_FIELDS),
                    }
                    for row in _order(selected, f"reference-{judge}")
                ],
            )
            reference_written.append(str(path))
            typer.echo(f"wrote {path}  (SECOND PASS — shows the reference answer)")

    # The unblinding key. Same relationship to the judge files that
    # DETECTOR_V4_ANSWER_KEY.json has to DETECTOR_V4_DATASET.json.
    key = {
        "schema": "graph-detector-v4-1-label-audit-key-v1",
        "judges_must_not_read": True,
        "why": (
            "every field here would anchor a judgement: the NLI label is the thing the "
            "audit exists to check, the partition and the detector score are the "
            "sampling, and the concept id is what a detector has to infer."
        ),
        "rows": {
            row["audit_id"]: {
                "text_sha256": row["text_sha256"],
                "stratum": row["stratum"],
                "population": row["population"],
                "bank_partition": row["bank_partition"],
                "nli_leaking": row["nli_leaking"],
                "concept_id": row["concept_id"],
                "item_id": row["item_id"],
                "lexical_score": round(float(row.get("lexical_score", 0.0)), 6),
                "n_chars": len(str(row["text"])),
            }
            for row in selected
        },
    }
    atomic_json(output_dir / KEY_FILENAME, key)
    typer.echo(f"wrote {output_dir / KEY_FILENAME}  (UNBLINDING KEY)")

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "audit_schema": AUDIT_SCHEMA,
        "protocol": "docs/graph_unlearning/DETECTOR_V4_1_PROTOCOL.md",
        "why": (
            "Goal A's label is 'does this attempt to answer the protected question'. The "
            "natural bank's label is 'does this reproduce the reference answer'. This "
            "audit measures how far apart they are, on rows a human has read."
        ),
        "ceiling_reinterpretation": CEILING_REINTERPRETATION,
        "source_bank": str(bank_path),
        "source_bank_content_sha256": bank.get("content_sha256"),
        "source_bank_modified": False,
        "overlay_key": "text_sha256 (audit_id = sha256('v4.1-audit|<text_sha256>|<question>')[:16])",
        "n_rows": len(selected),
        "strata": {name: len(strata.get(name, ())) for name in STRATA},
        "stratum_definitions": {
            "natural_leaking": "every row the run's NLI+ROUGE scorer called leaking",
            "clean_hard_negative": (
                "highest lexical answerability score among NLI-clean rows. A SAMPLING "
                "DEVICE: it decides which rows a human reads and enters no reported rate."
            ),
            "clean_matched": "matched to the leaking rows on question and length decile",
            "clean_random": "uniform over the remaining clean rows",
            "retain": "retain-author traffic; the retain-FPR population",
        },
        "sampling": sampling,
        "fields": {
            field: {
                "values": list(spec["values"]),
                "reference_answer_visible": spec["reference_answer_visible"],
                "question": spec["question"],
            }
            for field, spec in AUDIT_FIELDS.items()
        },
        "passes": {
            "blind": {"fields": list(BLIND_FIELDS), "files": ["LABEL_AUDIT_JUDGE_{A,B}.jsonl"]},
            "reference": {
                "fields": list(REFERENCE_FIELDS),
                "files": reference_written,
                "why_separate": (
                    "a judge who has seen the reference answer can no longer report "
                    "whether the text ATTEMPTS an answer independently of whether it got "
                    "the answer right, and that independence is the whole correction."
                ),
                "written": bool(reference_written),
            },
        },
        "judges": {
            "n": 2,
            "independent": True,
            "row_order": "salted per judge, so a shared drift in attention is not agreement",
            "adjudication": "disagreements need an explicit third-pass resolution",
        },
        "registry": registry_meta,
        "blinding": {
            "judge_files_contain": ["audit_id", "protected_question", "candidate_text"],
            "withheld": [
                "nli_leaking",
                "bank_partition",
                "lexical_score",
                "item_id",
                "concept_id",
                "stratum",
            ],
            "key_file": KEY_FILENAME,
        },
        "scope": {
            "model_trained": False,
            "graph_generation_run": False,
            "frozen_v1_v2_v3_artifacts_modified": False,
            "gpu_used": False,
        },
        "runtime_reads_gold_answers": False,
    }
    atomic_json(output_dir / MANIFEST_FILENAME, manifest)
    typer.echo(f"wrote {output_dir / MANIFEST_FILENAME}")
    typer.echo("")
    typer.echo("next: two judges annotate the JUDGE files independently, then")
    typer.echo("      rdl graph-detector-v4-label-report --judge-a ... --judge-b ...")


# -------------------------------------------------------------------- the report --


def _load_judge(paths: Sequence[Path]) -> dict[str, dict]:
    """Merge one judge's pass files into ``{audit_id: {field: value}}``, validating values."""
    out: dict[str, dict] = {}
    for path in paths:
        for row in _read_jsonl(path):
            audit_id = str(row.get("audit_id", ""))
            if not audit_id:
                raise typer.BadParameter(f"{path}: a row has no audit_id")
            entry = out.setdefault(audit_id, {})
            for field, spec in AUDIT_FIELDS.items():
                value = row.get(field)
                if value is None or value == "":
                    continue
                if value not in spec["values"]:
                    raise typer.BadParameter(
                        f"{path}: {audit_id}.{field} = {value!r}; allowed: {spec['values']}"
                    )
                entry[field] = value
    return out


def detector_v4_label_report(
    judge_a: list[Path] = typer.Option([], "--judge-a", help="judge A's returned file(s)"),
    judge_b: list[Path] = typer.Option([], "--judge-b", help="judge B's returned file(s)"),
    adjudication: Path | None = typer.Option(None, "--adjudication"),
    output_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--output-dir"),
) -> None:
    """Adjudicate the two judge files and write the alignment report.

    Exits non-zero when the audit does not clear its pre-registered decision gate. That is
    the point of the command: it is the last thing run before a GPU is rented, and "we did
    not measure it" must not exit the same way "it was fine" does.
    """
    if not judge_a or not judge_b:
        raise typer.BadParameter("both --judge-a and --judge-b are required; the audit needs two")
    key_path = output_dir / KEY_FILENAME
    if not key_path.exists():
        raise typer.BadParameter(f"{key_path} is absent; run `rdl graph-detector-v4-label-audit`")
    key = json.loads(key_path.read_text(encoding="utf-8")).get("rows", {})

    a = _load_judge(judge_a)
    b = _load_judge(judge_b)
    resolutions = {str(r["audit_id"]): r for r in _read_jsonl(adjudication)} if adjudication else {}
    adjudicated, unresolved = adjudicate(a, b, resolutions)

    _write_jsonl(
        output_dir / ADJUDICATED_FILENAME,
        [
            {**row, "text_sha256": key.get(row["audit_id"], {}).get("text_sha256", "")}
            for row in adjudicated
        ],
    )
    typer.echo(f"wrote {output_dir / ADJUDICATED_FILENAME}  ({len(adjudicated)} rows)")

    report = alignment_report(adjudicated, key, judge_a=a, judge_b=b, unresolved=unresolved)
    report["inputs"] = {
        "judge_a": [str(p) for p in judge_a],
        "judge_b": [str(p) for p in judge_b],
        "adjudication": str(adjudication) if adjudication else None,
        "key": str(key_path),
    }
    report["scope"] = {
        "model_trained": False,
        "graph_generation_run": False,
        "frozen_v1_v2_v3_artifacts_modified": False,
        "gpu_used": False,
    }
    report["runtime_reads_gold_answers"] = False
    atomic_json(output_dir / REPORT_FILENAME, report)
    typer.echo(f"wrote {output_dir / REPORT_FILENAME}")
    typer.echo(f"verdict: {report['verdict']}")
    for gate in report["gates"]:
        mark = "PASS" if gate["passed"] else ("n/a " if gate["passed"] is None else "FAIL")
        typer.echo(
            f"  [{mark}] {gate['gate']}: {gate['measured']} "
            f"(need {gate['comparison']} {gate['bound']})"
        )
    raise typer.Exit(0 if report["all_gates_passed"] else 1)
