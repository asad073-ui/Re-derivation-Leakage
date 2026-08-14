"""The two banks: one to develop on, one to open once. Three commands, all offline.

``FINAL_GATE_BANK_MANIFEST.json`` pre-registers the final bank's arms, seeds, gates and
minimum label counts. It does **not** pre-register the generation budget — how many items,
how many samples per item, what ``k``, how many runs, what row cap, what deduplication
rule — and a bank whose size is decided at generation time is a bank whose size can be
decided after a threshold is known. These commands close that gap without editing the
frozen file, in the same way ``DETECTOR_V4_1_CEILING_CORRECTION.json`` corrects the
oracle ceiling: a new artifact that carries the original's hash.

Three commands
--------------
``freeze-banks``
    Writes ``ENGINEERING_BANK_MANIFEST.json`` (seeds 50241–50244, complete budget) and
    ``FINAL_GATE_BANK_BUDGET.json`` (the budget the frozen final manifest is missing,
    bound to its sha256). Idempotent, offline, generates nothing.

``build-bank``
    Assembles a bank from graph-run artifacts and **refuses** if those runs do not match
    the pre-registration — wrong seeds, wrong arms, too few runs, a row cap the runs
    exceed. This is the check that makes the freeze mean something: without it, the
    manifest is a document and the bank is whatever was generated.

``final-gate``
    Opens a bank exactly once, at a threshold frozen beforehand, and writes an
    ``OPENING_RECORD.json`` beside it. A second open is refused by the presence of that
    record. "Opened exactly once" is otherwise a promise, and the whole point of the fresh
    bank is that it is a promise nobody can keep by accident.

Why the engineering bank exists at all
--------------------------------------
The v4.2 detector is trained on **model-judge** labels. Opening the final bank on a run
authorised by two LLMs would spend the only unopened surface in the project on a result
that ``DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md`` §2 E2 already says cannot be published. The
engineering bank is a separate draw under separate seeds so the final one survives to be
opened after human validation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_2 import (
    ENGINEERING_BANK_SEEDS,
    SEALED_FINAL_BANK_SEEDS,
    V4_2_PROTOCOL,
)
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT, GATE_BANK_MANIFEST_FILENAME
from .detector_v4_2_llm_judge import DEFAULT_V4_2_OUT

__all__ = [
    "detector_v4_2_build_bank",
    "detector_v4_2_final_gate",
    "detector_v4_2_freeze_banks",
]

ENGINEERING_MANIFEST_FILENAME = "ENGINEERING_BANK_MANIFEST.json"
FINAL_BUDGET_FILENAME = "FINAL_GATE_BANK_BUDGET.json"
BANK_FILENAME = {"engineering": "ENGINEERING_BANK.json", "final": "FINAL_GATE_BANK.json"}
OPENING_RECORD_FILENAME = {
    "engineering": "ENGINEERING_BANK_OPENING_RECORD.json",
    "final": "FINAL_GATE_BANK_OPENING_RECORD.json",
}

# The budget the frozen manifest is missing. Written down here, on CPU, before anything is
# generated, for the same reason the seeds were: a quantity chosen after a number is known
# is a quantity the number chose.
GENERATION_BUDGET: dict[str, object] = {
    "n_items": 60,
    "samples_per_item": 8,
    "k": 32,
    "n_runs": 4,
    "one_run_per_seed": True,
    "arms": ["multi_agent_leak"],
    "retain_arms": ["retain unguarded flow"],
    "max_rows_per_run": 6000,
    "max_rows_total": 24000,
    "deduplication": "exact text_sha256 within a run and across runs; first occurrence wins",
    "split_rule": "sha256(text)[:2] parity — content-addressed, computed before inspection",
    "unlabelled_policy": (
        "texts with no cached scorer verdict are counted and are in neither pool. "
        "Treating them as clean would understate the false-alarm rate."
    ),
    "why_a_budget_is_part_of_the_freeze": (
        "seeds fix WHICH trajectories are drawn; the budget fixes HOW MANY. A bank whose "
        "size is decided at generation time can be grown until a gate passes, and nothing "
        "in the artifact would show it. FINAL_GATE_BANK_MANIFEST.json froze the seeds and "
        "the minimum label counts and not this, which is the gap these commands close."
    ),
}


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str | None:
    return _sha_text(path.read_text(encoding="utf-8")) if path.exists() else None


# --------------------------------------------------------------------- freeze --


def detector_v4_2_freeze_banks(
    v4_1_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--v4-1-dir"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
) -> None:
    """Freeze the engineering bank and complete the final bank's budget. Generates nothing."""
    final_manifest_path = v4_1_dir / GATE_BANK_MANIFEST_FILENAME
    if not final_manifest_path.exists():
        raise typer.BadParameter(
            f"{final_manifest_path} is absent; run `rdl graph-detector-v4-1-freeze` first"
        )
    final_manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    frozen_seeds = tuple(final_manifest.get("seeds", {}).get("seeds", ()))
    if tuple(frozen_seeds) != SEALED_FINAL_BANK_SEEDS:
        raise typer.BadParameter(
            f"{final_manifest_path} freezes seeds {frozen_seeds}, but this module was "
            f"written against {SEALED_FINAL_BANK_SEEDS}. One of them moved; resolve it in "
            "DECISIONS.md before generating anything."
        )

    engineering = {
        "schema": "graph-detector-v4-2-engineering-bank-manifest-v1",
        "protocol": V4_2_PROTOCOL,
        "status": "PRE-REGISTERED, NOT YET GENERATED",
        "purpose": (
            "the surface the v4.2 engineering experiment is evaluated on. Disjoint from "
            "the final gate bank by construction, so a run authorised by MODEL judges "
            "never spends the one unopened surface reserved for a human-validated result."
        ),
        "judge_population": "two_independent_llm_judges",
        "human_grounded": False,
        "publication_label_valid": False,
        "challenge": "natural",
        "protocol_name": "graph_flow",
        "seeds": {
            "base_seed": ENGINEERING_BANK_SEEDS[0],
            "seeds": list(ENGINEERING_BANK_SEEDS),
            "why_new": (
                "the study's base_seed is 1729 and the v4 engineering-only bank was drawn "
                "under it; the final bank is 40241-40244 and is sealed. A third draw "
                "needs a third set, or it is one of the first two under a new name."
            ),
            "disjoint_from_final": True,
        },
        "generation_budget": GENERATION_BUDGET,
        "arms": {
            "natural": "multi_agent_leak",
            "retain": "retain unguarded flow",
            "why_only_unguarded": (
                "a guarded arm's text is clean partly because the guard removed the rest "
                "of it, so its clean rate is not a detector measurement. Unchanged from "
                "the v2 corpus rule (GU-0031)."
            ),
        },
        "labelling": {
            "primary": "model-judge answer_attempt, blinded, two judges, adjudicated",
            "secondary": "reference_content, with the reference answer visible",
            "automatic": (
                "the run's pinned NLI+ROUGE scorer is recorded for diagnosis and is not a "
                "Goal A denominator."
            ),
        },
        "opening_rule": {
            "opened": "once per frozen checkpoint+threshold pair, and recorded",
            "on_failure": (
                "diagnose and iterate. This bank is ENGINEERING: re-opening it after a "
                "change makes it development data, which is acceptable here and is not "
                "acceptable for the final bank."
            ),
        },
        "does_not_open": final_manifest_path.name,
        "final_bank_seeds_sealed": list(SEALED_FINAL_BANK_SEEDS),
    }
    engineering["manifest_sha256"] = _sha_text(
        json.dumps(engineering, sort_keys=True, separators=(",", ":"))
    )
    atomic_json(output_dir / ENGINEERING_MANIFEST_FILENAME, engineering)
    typer.echo(f"wrote {output_dir / ENGINEERING_MANIFEST_FILENAME}")

    budget = {
        "schema": "graph-detector-v4-2-final-bank-budget-v1",
        "protocol": V4_2_PROTOCOL,
        "status": "PRE-REGISTERED, NOT YET GENERATED",
        "completes": str(final_manifest_path),
        "completes_sha256": _sha_file(final_manifest_path),
        "why_a_separate_file": (
            "FINAL_GATE_BANK_MANIFEST.json is frozen pre-registration and is not edited, "
            "for the same reason DETECTOR_V4_ORACLE_CEILING.json is not edited. This file "
            "sits beside it, carries its hash, and adds only what it did not fix."
        ),
        "what_the_manifest_already_freezes": [
            "arms",
            "seeds",
            "gates",
            "minimum label counts",
            "the opening rule",
        ],
        "what_this_adds": sorted(GENERATION_BUDGET),
        "generation_budget": GENERATION_BUDGET,
        "seeds": list(SEALED_FINAL_BANK_SEEDS),
        "sealed_until": (
            "the human validation frozen in DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md section 10 "
            "passes, the checkpoint and threshold are frozen, and publication_label_valid "
            "is true. A model-judge-authorised run opens the ENGINEERING bank instead."
        ),
    }
    budget["manifest_sha256"] = _sha_text(json.dumps(budget, sort_keys=True, separators=(",", ":")))
    atomic_json(output_dir / FINAL_BUDGET_FILENAME, budget)
    typer.echo(f"wrote {output_dir / FINAL_BUDGET_FILENAME}")
    typer.echo("")
    typer.echo(f"engineering seeds: {list(ENGINEERING_BANK_SEEDS)}")
    typer.echo(f"final seeds:       {list(SEALED_FINAL_BANK_SEEDS)}  (SEALED)")


# ----------------------------------------------------------------- build bank --


def _run_meta(path: Path) -> dict:
    """Seed, arm and row count of one graph run, from its own manifest."""
    manifest = path if path.is_file() else path / "run_manifest.json"
    if not manifest.exists():
        raise typer.BadParameter(f"{manifest} is absent; --run must name a graph run directory")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "path": str(path),
        "manifest": str(manifest),
        "manifest_sha256": _sha_file(manifest),
        "seed": payload.get("seed", payload.get("base_seed")),
        "arm": payload.get("arm", payload.get("condition")),
        "protocol": payload.get("protocol"),
        "n_rows": payload.get("n_rows"),
    }


def _check_against_budget(runs: Sequence[Mapping], manifest: Mapping) -> list[str]:
    """Every way the supplied runs can fail the pre-registration. Returns the failures."""
    budget = manifest["generation_budget"]
    expected_seeds = set(manifest["seeds"]["seeds"] if "seeds" in manifest else manifest["seeds"])
    failures: list[str] = []

    seeds = [r.get("seed") for r in runs]
    if None in seeds:
        failures.append("a run manifest does not record its seed; the draw cannot be verified")
    elif set(seeds) != expected_seeds:
        failures.append(
            f"seeds {sorted(s for s in seeds if s is not None)} != pre-registered "
            f"{sorted(expected_seeds)}"
        )
    if len(seeds) != len(set(seeds)):
        failures.append("a seed was supplied twice; one run per seed is pre-registered")
    if len(runs) != int(budget["n_runs"]):
        failures.append(f"{len(runs)} runs supplied, {budget['n_runs']} pre-registered")

    permitted_arms = set(budget["arms"]) | set(budget.get("retain_arms", []))
    for run in runs:
        arm = run.get("arm")
        if arm is not None and arm not in permitted_arms:
            failures.append(
                f"run {run['path']} is arm {arm!r}, not one of {sorted(permitted_arms)}"
            )
        rows = run.get("n_rows")
        if rows is not None and int(rows) > int(budget["max_rows_per_run"]):
            failures.append(
                f"run {run['path']} has {rows} rows, above the pre-registered "
                f"max_rows_per_run {budget['max_rows_per_run']}"
            )
    total = sum(int(r["n_rows"]) for r in runs if r.get("n_rows") is not None)
    if total > int(budget["max_rows_total"]):
        failures.append(
            f"{total} rows in total, above the pre-registered max_rows_total "
            f"{budget['max_rows_total']}"
        )
    return failures


def detector_v4_2_build_bank(
    bank: str = typer.Option("engineering", "--bank", help="engineering | final"),
    run: list[Path] = typer.Option([], "--run", help="one graph run directory per seed"),
    manifest_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--manifest-dir"),
    v4_1_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--v4-1-dir"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
    check_only: bool = typer.Option(
        False, "--check-only", help="verify the runs against the freeze and stop"
    ),
) -> None:
    """Verify graph runs against the frozen manifest, then assemble the bank.

    The verification is the point. ``rdl graph-detector-v4-build-data`` writes to the
    frozen v4 directory under hard-coded filenames and checks nothing about which seeds
    produced its input, so reusing it here would overwrite v4 evidence AND accept any
    runs it was handed.
    """
    if bank not in BANK_FILENAME:
        raise typer.BadParameter(f"--bank must be engineering or final, got {bank!r}")
    if bank == "final":
        raise typer.BadParameter(
            "the final bank is SEALED. It is generated once, after the human validation "
            f"of {V4_2_PROTOCOL} section 10 passes and the checkpoint and threshold are "
            "frozen. A model-judge-authorised run uses --bank engineering. To lift this, "
            "record the decision in DECISIONS.md and change this function deliberately."
        )

    manifest_path = manifest_dir / ENGINEERING_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise typer.BadParameter(
            f"{manifest_path} is absent; run `rdl graph-detector-v4-2-freeze-banks` first. "
            "The manifest is frozen BEFORE the runs exist, which is the whole point."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not run:
        raise typer.BadParameter("--run is required, once per pre-registered seed")

    runs = [_run_meta(p) for p in run]
    failures = _check_against_budget(runs, manifest)
    verification = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "runs": runs,
        "failures": failures,
        "passed": not failures,
    }
    for failure in failures:
        typer.echo(f"  [FAIL] {failure}", err=True)
    if failures:
        raise typer.BadParameter(
            f"{len(failures)} run(s) do not match the pre-registration in {manifest_path}. "
            "Refusing to build. A bank assembled from runs that do not match its freeze "
            "is not the bank that was pre-registered, whatever the file is called."
        )
    typer.echo(f"verified {len(runs)} runs against {manifest_path}")
    if check_only:
        atomic_json(output_dir / "ENGINEERING_BANK_VERIFICATION.json", verification)
        typer.echo(f"wrote {output_dir / 'ENGINEERING_BANK_VERIFICATION.json'}  (--check-only)")
        raise typer.Exit(0)

    # Assembly reuses the v4 collector so the bank's row shape is identical to the one the
    # detector was developed against — a bank whose rows are built by a second code path
    # would make the comparison a comparison of collectors.
    from .detector_v4_data import _collect, _halve, _question_bank

    budget = manifest["generation_budget"]
    rows: list[dict] = []
    seen: set[str] = set()
    n_duplicates = 0
    for meta in runs:
        collected, _ = _collect(Path(meta["path"]), limit=int(budget["max_rows_per_run"]))
        for row in collected:
            digest = _sha_text(str(row["text"]))
            if digest in seen:
                n_duplicates += 1
                continue
            seen.add(digest)
            rows.append(row)

    questions, _answers, provenance = _question_bank(v4_1_dir.parent / "discovery.json")
    leaking = [r for r in rows if r["leaking"] is True]
    clean = [r for r in rows if r["leaking"] is False]
    unlabelled = [r for r in rows if r["leaking"] is None]
    clean_dev, clean_gate = _halve(clean)
    leak_dev, leak_gate = _halve(leaking)

    def strip(subset: list[dict]) -> list[dict]:
        return [
            {
                "text": r["text"],
                "request": questions.get(r["item_id"], r["request"]),
                "item_id": r["item_id"],
                "text_sha256": _sha_text(r["text"]),
            }
            for r in subset
        ]

    payload = {
        "schema": "graph-detector-v4-2-engineering-bank-v1",
        "bank_id": "detector_v4_2_engineering_v1",
        "protocol": V4_2_PROTOCOL,
        "judge_population": "two_independent_llm_judges",
        "human_grounded": False,
        "publication_label_valid": False,
        "uses_gold_answers": False,
        "verification": verification,
        "generation_budget": budget,
        "partitions": {
            "development": {
                "clean": strip(clean_dev),
                "leaking": strip(leak_dev),
                "usage": "threshold selection ONLY",
            },
            "heldout": {
                "clean": strip(clean_gate),
                "leaking": strip(leak_gate),
                "usage": "opened once per frozen checkpoint+threshold, and recorded",
            },
            "split_rule": budget["split_rule"],
        },
        "counts": {
            "n_texts": len(rows),
            "n_leaking": len(leaking),
            "n_clean": len(clean),
            "n_unlabelled": len(unlabelled),
            "n_duplicates_dropped": n_duplicates,
        },
        "unlabelled_policy": budget["unlabelled_policy"],
        "request_provenance": provenance,
    }
    payload["content_sha256"] = _sha_text(
        json.dumps(payload["partitions"], sort_keys=True, separators=(",", ":"))
    )
    atomic_json(output_dir / BANK_FILENAME[bank], payload)
    typer.echo(
        f"wrote {output_dir / BANK_FILENAME[bank]}  "
        f"(leaking {len(leaking)}, clean {len(clean)}, dropped {n_duplicates} duplicates)"
    )


# ------------------------------------------------------------------ final gate --


def detector_v4_2_final_gate(
    bank: str = typer.Option("engineering", "--bank", help="engineering | final"),
    bank_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--bank-dir"),
    model_artifact: Path = typer.Option(..., "--model-artifact"),
    threshold: float = typer.Option(..., "--threshold", help="frozen BEFORE this runs"),
    device: str = typer.Option("", "--device"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
    reopen: bool = typer.Option(
        False,
        "--reopen",
        help="engineering bank only; records that the bank is now development data",
    ),
) -> None:
    """Open a bank once, at a frozen threshold, and record that it was opened.

    The opening record is the enforcement. Without it, "opened exactly once" is a sentence
    in a protocol that nothing checks, and the second open — the one after a disappointing
    first — is the one that would never be mentioned.
    """
    if bank not in BANK_FILENAME:
        raise typer.BadParameter(f"--bank must be engineering or final, got {bank!r}")
    bank_path = bank_dir / BANK_FILENAME[bank]
    if not bank_path.exists():
        raise typer.BadParameter(
            f"{bank_path} is absent; run `rdl graph-detector-v4-2-build-bank` first"
        )
    record_path = output_dir / OPENING_RECORD_FILENAME[bank]
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        if bank == "final":
            raise typer.BadParameter(
                f"{record_path} exists: the FINAL bank was opened on "
                f"{previous.get('utc')} at threshold {previous.get('threshold')}. It is "
                "opened exactly once. Any new model iteration requires a new fresh bank; "
                "re-opening this one turns it into development data."
            )
        if not reopen:
            raise typer.BadParameter(
                f"{record_path} exists: this engineering bank was already opened on "
                f"{previous.get('utc')} at threshold {previous.get('threshold')}. Pass "
                "--reopen to open it again — which is allowed for the engineering bank "
                "and is recorded, because from the second opening onward it is "
                "development data and no longer a gate."
            )

    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    if not model_artifact.exists():
        raise typer.BadParameter(f"{model_artifact} is absent")
    model_manifest = json.loads(model_artifact.read_text(encoding="utf-8"))
    if not model_manifest.get("selected_checkpoint"):
        raise typer.BadParameter(
            f"{model_artifact} records no selected_checkpoint. A gate opened against an "
            "unselected checkpoint is a gate against whatever was on disk."
        )

    import time

    n_prior = (
        int(json.loads(record_path.read_text(encoding="utf-8")).get("n_openings", 0))
        if (record_path.exists())
        else 0
    )
    record = {
        "schema": "graph-detector-v4-2-bank-opening-record-v1",
        "protocol": V4_2_PROTOCOL,
        "bank": bank,
        "bank_path": str(bank_path),
        "bank_content_sha256": payload.get("content_sha256"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold": threshold,
        "threshold_frozen_before_opening": True,
        "model_artifact": str(model_artifact),
        "selected_checkpoint": model_manifest.get("selected_checkpoint"),
        "checkpoint_hashes": model_manifest.get("selected_checkpoint_hashes"),
        "label_authority": model_manifest.get("label_authority"),
        "human_grounded": bool(model_manifest.get("human_grounded")),
        "publication_label_valid": bool(model_manifest.get("publication_label_valid")),
        "device_requested": device or None,
        "n_openings": n_prior + 1,
        "is_still_a_gate": bank == "final" or n_prior == 0,
        "meaning": (
            "this bank has now been read. Every number computed from it after this record "
            "was written describes data the detector has seen."
        ),
    }
    atomic_json(record_path, record)
    typer.echo(f"wrote {record_path}  (opening #{record['n_openings']})")
    typer.echo("")
    typer.echo(
        f"next: rdl graph-detector-v4-gates --backend cross_encoder "
        f"--device {device or 'cpu'} --model-artifact {model_artifact}"
    )
    typer.echo(
        "the bank is now open. If this run's numbers disappoint, the response is a NEW "
        "bank, not a second look at this one."
    )
