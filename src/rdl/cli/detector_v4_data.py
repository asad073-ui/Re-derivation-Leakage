"""``rdl graph-detector-v4-build-data`` — the frozen v4 datasets.

Writes three files into ``data/cohorts/graph_unlearning_v1/detector_v4/``:

``DETECTOR_V4_DATASET.json``
    The synthetic answerability corpus: invented subjects, invented relation values, the
    twelve Goal A example classes, split three ways (subject, surface variant, relation).
    A detector may see every field of it except ``label``.

``DETECTOR_V4_ANSWER_KEY.json``
    The offline oracle's sidecar. Marked ``runtime_forbidden``.

``DETECTOR_V4_NATURAL_BANK.json``
    Real text, and the reason this command touches run directories at all. Detector v3's
    generated-clean population came from ``memory_reentry``, which is not the flow the
    natural study reports, so its false-alarm rate was not comparable to anything a
    natural run would produce (``DETECTOR_V3_IDENTITY_PROBE.json`` says so in its own
    caveat). The v4 bank is collected from the NATURAL flow, from the unguarded arm only,
    and is split into a development half and a gate half by content hash before anyone
    looks at either.

Eligibility rules, unchanged from the v2 corpus (GU-0031): unguarded arm only, because a
guarded arm's text is clean partly because the guard removed the rest of it; the run's own
pinned NLI scorer supplies every label; nothing is rescored here.

Offline except for the TOFU questions, which come from the pinned dataset revision the
cohorts name and are hash-verified against the frozen cohort before use.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from ..eval.detector_v4 import build_dataset
from ..eval.tofu_data import load_items
from ..studies.graph_leak.cohort import load_cohort, resolve_cohort
from ..studies.graph_leak.evidence import atomic_json, read_nli_cache, read_shards, scorer_label_fn

__all__ = ["detector_v4_build_data"]

DEFAULT_OUT = Path("data/cohorts/graph_unlearning_v1/detector_v4")
DATASET_FILENAME = "DETECTOR_V4_DATASET.json"
ANSWER_KEY_FILENAME = "DETECTOR_V4_ANSWER_KEY.json"
NATURAL_BANK_FILENAME = "DETECTOR_V4_NATURAL_BANK.json"
UNGUARDED_ARM = "multi_agent_leak"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collect(run: Path, *, limit: int) -> tuple[list[dict], dict]:
    """Unguarded-arm agent messages from one run, each carrying its own scorer label.

    Returns ``([{request, text, leaking, item_id}, ...], provenance)``. ``item_id`` is
    recorded so the gate runner can look up the PROTECTED QUESTION the trajectory was
    given — never to look up the concept, which the detector has to infer.
    """
    scoring = run / "scores" / "SCORING.json"
    if not scoring.exists():
        return [], {"run": run.name, "usable": False, "reason": "no SCORING.json"}
    version = str(json.loads(scoring.read_text(encoding="utf-8")).get("scorer_version", ""))
    cache = read_nli_cache(run / "scores" / "nli-cache.jsonl", version)
    if not cache:
        return [], {"run": run.name, "usable": False, "reason": "empty NLI cache"}
    label = scorer_label_fn(cache, version)

    out: list[dict] = []
    n_rows = 0
    truncated = False
    for row in read_shards(run / "generations"):
        if str(row.get("arm", "")) != UNGUARDED_ARM:
            continue
        n_rows += 1
        request = str(row.get("question", ""))
        reference = str(row.get("reference_answer", ""))
        trajectory_id = str(row.get("trajectory_id", ""))
        parts = trajectory_id.split(":")
        item_id = parts[2] if len(parts) >= 4 else ""
        for text in (str(t) for t in (row.get("raw_outputs", {}) or {}).get("agent_messages", [])):
            if not text.strip():
                continue
            verdict = label(reference, text)
            out.append(
                {
                    "text": text,
                    "request": request,
                    "item_id": item_id,
                    # ``None`` when the scorer had no cached verdict for this pair. Kept
                    # rather than coerced: a row whose label is unknown must not be
                    # counted as clean, which is how a false-alarm rate gets understated.
                    "leaking": None if verdict is None else bool(verdict),
                }
            )
            if len(out) >= limit:
                truncated = True
                break
        if truncated:
            break
    return out, {
        "run": run.name,
        "usable": True,
        "scorer_version": version,
        "n_unguarded_rows_seen": n_rows,
        "n_texts": len(out),
        "truncated_at_limit": truncated,
    }


def _halve(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Content-addressed halving. Threshold selection may see only the first half.

    By hash of the TEXT, not by position: the shards are ordered by trajectory, so a
    positional split would put whole items on one side and make the two halves differ in
    which authors they contain.
    """
    dev = [r for r in rows if int(_sha(r["text"])[:2], 16) % 2 == 0]
    gate = [r for r in rows if int(_sha(r["text"])[:2], 16) % 2 == 1]
    return dev, gate


def _question_bank(cohort_path: Path) -> tuple[dict[str, str], dict[str, str], dict]:
    """``({item_id: question}, {item_id: answer}, provenance)`` from the pinned revision.

    The answers are returned for the ORACLE ONLY and this function's caller is responsible
    for keeping them out of the runtime dataset — which is why they land in the answer-key
    file and never in ``DETECTOR_V4_NATURAL_BANK.json``.
    """
    cohort = load_cohort(cohort_path, exclusions_path=None)
    source, _ = load_items(
        dataset=cohort.dataset,
        split=cohort.dataset_config,
        n_items=None,
        revision=cohort.dataset_revision,
    )
    resolved = resolve_cohort(cohort, source)
    questions = {i.item_id: i.question for i in resolved}
    answers = {i.item_id: i.answer for i in resolved}
    verified = sum(
        1
        for entry in cohort.items
        if entry.frozen and _sha(questions.get(entry.item_id, "")) == entry.question_sha256
    )
    return (
        questions,
        answers,
        {
            "cohort": str(cohort_path),
            "cohort_fingerprint": cohort.fingerprint(),
            "dataset_revision": cohort.dataset_revision,
            "n_items": len(resolved),
            "n_frozen_items_hash_verified": verified,
            "all_frozen_items_verified": verified == sum(1 for e in cohort.items if e.frozen),
        },
    )


def detector_v4_build_data(
    output_dir: Path = typer.Option(DEFAULT_OUT, "--output-dir"),
    natural_run: list[Path] = typer.Option(
        [], "--natural-run", help="graph_flow NATURAL run directory"
    ),
    retain_run: list[Path] = typer.Option(
        [], "--retain-run", help="graph_flow RETAIN run directory"
    ),
    policy_cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/discovery.json"), "--policy-cohort"
    ),
    max_per_run: int = typer.Option(6000, "--max-per-run"),
    n_subjects: int = typer.Option(60, "--n-subjects"),
) -> None:
    """Build and freeze the Detector-v4 datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, answer_key = build_dataset(n_subjects=n_subjects)
    atomic_json(output_dir / DATASET_FILENAME, dataset)
    typer.echo(f"wrote {output_dir / DATASET_FILENAME}  ({dataset['n_rows']} rows)")

    natural_rows: list[dict] = []
    natural_sources: list[dict] = []
    for run in natural_run:
        rows, meta = _collect(run, limit=max_per_run)
        natural_rows.extend(rows)
        natural_sources.append(meta)
    retain_rows: list[dict] = []
    retain_sources: list[dict] = []
    for run in retain_run:
        rows, meta = _collect(run, limit=max_per_run)
        retain_rows.extend(rows)
        retain_sources.append(meta)

    questions: dict[str, str] = {}
    answers: dict[str, str] = {}
    provenance: dict = {"measured": False, "reason": "no natural or retain run supplied"}
    if natural_rows or retain_rows:
        questions, answers, provenance = _question_bank(policy_cohort)
        provenance["measured"] = True

    leaking = [r for r in natural_rows if r["leaking"] is True]
    clean = [r for r in natural_rows if r["leaking"] is False]
    unlabelled = [r for r in natural_rows if r["leaking"] is None]
    clean_dev, clean_gate = _halve(clean)
    leak_dev, leak_gate = _halve(leaking)

    def strip(rows: list[dict]) -> list[dict]:
        """Bank rows carry no answer and no concept label — only what a detector may see."""
        return [
            {
                "text": r["text"],
                "request": questions.get(r["item_id"], r["request"]),
                "item_id": r["item_id"],
                "text_sha256": _sha(r["text"]),
            }
            for r in rows
        ]

    bank = {
        "schema": "graph-detector-v4-natural-bank-v1",
        "bank_id": "detector_v4_natural_v1",
        "uses_gold_answers": False,
        "arm": UNGUARDED_ARM,
        "challenge": "natural",
        "protocol": "graph_flow",
        "why_natural_only": (
            "the v3 probe's generated-clean population came from memory_reentry, which is "
            "not the flow the natural study reports; its false-alarm rate was therefore "
            "not comparable to what a natural run produces. This bank is collected from "
            "the natural flow so the v4 gate's clean FPR describes the deployed setting."
        ),
        "partitions": {
            "development": {
                "clean": strip(clean_dev),
                "leaking": strip(leak_dev),
                "usage": "threshold selection ONLY",
            },
            "heldout": {
                "clean": strip(clean_gate),
                "leaking": strip(leak_gate),
                "usage": "opened once, after the threshold is frozen",
            },
            "retain": {
                "all": strip(retain_rows),
                "usage": "retain-answer false alarms; bounded above by the router",
            },
            "split_rule": "sha256(text)[:2] parity — content-addressed, computed before inspection",
        },
        "counts": {
            "n_natural_texts": len(natural_rows),
            "n_leaking": len(leaking),
            "n_clean": len(clean),
            "n_unlabelled": len(unlabelled),
            "n_clean_development": len(clean_dev),
            "n_clean_heldout": len(clean_gate),
            "n_leaking_development": len(leak_dev),
            "n_leaking_heldout": len(leak_gate),
            "n_retain": len(retain_rows),
        },
        "unlabelled_policy": (
            "texts with no cached scorer verdict are counted in n_unlabelled and are in "
            "neither pool. Treating them as clean would understate the false-alarm rate."
        ),
        "request_provenance": provenance,
        "sources": {"natural": natural_sources, "retain": retain_sources},
    }
    bank["content_sha256"] = _sha(
        json.dumps(bank["partitions"], sort_keys=True, separators=(",", ":"))
    )
    atomic_json(output_dir / NATURAL_BANK_FILENAME, bank)
    typer.echo(
        f"wrote {output_dir / NATURAL_BANK_FILENAME}  "
        f"(leaking {len(leaking)}, clean {len(clean)}, retain {len(retain_rows)}, "
        f"unlabelled {len(unlabelled)})"
    )

    # The answer key is written LAST and separately, so that a partially failed run can
    # never leave a dataset with answers glued into it.
    answer_key["natural_answers"] = dict(sorted(answers.items()))
    answer_key["natural_answers_note"] = (
        "TOFU forget answers at the pinned revision, keyed by item id. Used ONLY by "
        "`rdl graph-detector-v4-oracle`, which is an offline evaluator. No module under "
        "rdl.defenses may read this file, and the CPU gate asserts it."
    )
    answer_key["content_sha256"] = _sha(
        json.dumps(
            {"answers": answer_key["answers"], "natural": answer_key["natural_answers"]},
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    atomic_json(output_dir / ANSWER_KEY_FILENAME, answer_key)
    typer.echo(f"wrote {output_dir / ANSWER_KEY_FILENAME}  (RUNTIME FORBIDDEN)")
