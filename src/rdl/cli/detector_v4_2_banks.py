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
    Writes ``ENGINEERING_BANK_MANIFEST.json`` (two seed groups, complete budget) and
    ``FINAL_GATE_BANK_BUDGET.json`` (the budget the frozen final manifest is missing,
    bound to its sha256). Idempotent, offline, generates nothing.

``build-bank``
    Assembles a bank from graph-run artifacts and **refuses** if those runs do not match
    the pre-registration — wrong seeds, wrong arms, wrong cohort, too few runs, a sampling
    design that cannot measure the ``k`` it claims. This is the check that makes the freeze
    mean something: without it, the manifest is a document and the bank is whatever was
    generated.

``final-gate``
    Loads a bank AND its adjudicated labels, scores the frozen checkpoint at the frozen
    threshold, writes the result and an ``OPENING_RECORD.json`` atomically. A second open
    is refused by the presence of that record. "Opened exactly once" is otherwise a
    promise, and the whole point of the fresh bank is that it is a promise nobody can keep
    by accident.

Why the budget is what it is
----------------------------
The first version of this file froze ``n_items=60, samples_per_item=8, k=32``. That design
cannot be generated: the discovery cohort has 50 items, not 60, and **8 samples cannot
measure k=32** — ``k`` is the number of samples the success-at-k statistic is read over, so
``n_samples >= primary_k`` is an arithmetic precondition, not a preference. Every real run
in ``runs/graph`` that reports ``primary_k=32`` carries ``n_samples=32``. The budget below
is the design those runs actually realise, and :func:`validate_budget` refuses the old one.

Why the run manifest is read the way it is
------------------------------------------
Graph runs write ``RUN_MANIFEST.json`` — upper case — and record the seed at
``sampling.base_seed``, the arms as a list of objects under ``arms``, and the cohort under
``evaluation_cohort``. The first version of this file opened ``run_manifest.json`` and read
top-level ``seed`` / ``arm`` / ``n_rows``, none of which exist. Because every lookup
returned ``None`` and every check skipped ``None``, it accepted any directory that happened
to contain a file by that name and rejected every real run for not having one. A validator
that cannot read the artifact it validates is worse than no validator: it reports PASS.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_2 import (
    ENGINEERING_BANK_SEEDS,
    ENGINEERING_RETAIN_SEEDS,
    SEALED_FINAL_BANK_SEEDS,
    V4_2_PROTOCOL,
)
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT, GATE_BANK_MANIFEST_FILENAME
from .detector_v4_2_llm_judge import DEFAULT_V4_2_OUT

__all__ = [
    "GENERATION_BUDGET",
    "detector_v4_2_build_bank",
    "detector_v4_2_final_gate",
    "detector_v4_2_freeze_banks",
    "run_meta",
    "validate_budget",
]

ENGINEERING_MANIFEST_FILENAME = "ENGINEERING_BANK_MANIFEST.json"
# v3: the retain pool is halved into the two partitions instead of sitting in a third
# block that both selected the threshold and reported the rate.
BANK_SCHEMA = "graph-detector-v4-2-engineering-bank-v3"
FINAL_BUDGET_FILENAME = "FINAL_GATE_BANK_BUDGET.json"
BANK_FILENAME = {"engineering": "ENGINEERING_BANK.json", "final": "FINAL_GATE_BANK.json"}
OPENING_RECORD_FILENAME = {
    "engineering": "ENGINEERING_BANK_OPENING_RECORD.json",
    "final": "FINAL_GATE_BANK_OPENING_RECORD.json",
}
GATE_RESULT_FILENAME = {
    "engineering": "ENGINEERING_BANK_GATE_RESULT.json",
    "final": "FINAL_GATE_BANK_GATE_RESULT.json",
}
RUN_MANIFEST_FILENAME = "RUN_MANIFEST.json"

# The arm the bank is drawn from. Unchanged from the v2 corpus rule (GU-0031): a guarded
# arm's text is clean partly because the guard removed the rest of it, so its clean rate is
# not a detector measurement.
UNGUARDED_ARM = "multi_agent_leak"

GROUPS: tuple[str, ...] = ("natural", "retain")

# The budget the frozen manifest is missing. Written down here, on CPU, before anything is
# generated, for the same reason the seeds were: a quantity chosen after a number is known
# is a quantity the number chose.
#
# Two groups, not one four-run set. A retain run and a natural run are different cohorts
# (retain90 vs forget10), different item counts, and different roles in the gate — the
# natural group supplies the recall numerator and the protected-clean false-alarm pool, the
# retain group supplies the retain false-alarm pool. Collapsing them into "four runs" makes
# "four" ambiguous between four natural runs and two of each, and lets a retain run and a
# natural run share a seed, which is the same draw wearing two labels.
GENERATION_BUDGET: dict[str, object] = {
    "groups": {
        "natural": {
            "role": "recall numerator and the protected-clean false-alarm pool",
            "n_items": 50,
            "n_samples": 32,
            "primary_k": 32,
            "n_runs": 4,
            "seeds": list(ENGINEERING_BANK_SEEDS),
            "challenge": "natural",
            "protocol": "graph_flow",
            "cohort_split": "discovery",
            "is_retain": False,
            "required_arm": UNGUARDED_ARM,
            "max_rows_per_run": 6000,
        },
        "retain": {
            "role": "the retain false-alarm pool, which the aggregate clean FPR cannot express",
            "n_items": 45,
            "n_samples": 32,
            "primary_k": 32,
            "n_runs": 4,
            "seeds": list(ENGINEERING_RETAIN_SEEDS),
            "challenge": "natural",
            "protocol": "graph_flow",
            "cohort_split": "retain_utility",
            "is_retain": True,
            "required_arm": UNGUARDED_ARM,
            "max_rows_per_run": 6000,
        },
    },
    "max_rows_total": 24000,
    "deduplication": (
        "by (candidate text, protected question), first occurrence wins. Text alone is "
        "wrong: the same refusal string is a legitimately distinct row under two different "
        "protected questions, and dropping the second silently deletes one question's "
        "false-alarm evidence."
    ),
    "split_rule": "sha256(text || question)[:2] parity — content-addressed, before inspection",
    # New in v4.2.2, and the fix for a number that was not held out. The retain pool was
    # one undivided block: the threshold sweep read it to enforce the retain ceiling, and
    # the gate then reported the retain FPR over the same rows. A rate measured on the rows
    # that chose the threshold is a description of the choice. The retain rows are halved
    # by the same content-addressed rule as the protected ones, and the two halves are
    # separate partitions everywhere downstream.
    "retain_split_rule": (
        "the same sha256(text || question)[:2] parity as the protected rows. Retain rows "
        "live inside `development` and `heldout` rather than in a third block, so 'the "
        "held-out partition' means the same thing for both populations and no reader can "
        "select on one and report on the other."
    ),
    "why_retain_is_split": (
        "retain/all supplied the retain ceiling that constrained threshold selection AND "
        "the retain FPR the gate reported. Those are the same rows, so the reported "
        "false-alarm rate was fitted, not held out — and it is the number the utility "
        "claim rests on."
    ),
    "unlabelled_policy": (
        "texts with no cached scorer verdict are counted and are in neither pool. "
        "Treating them as clean would understate the false-alarm rate."
    ),
    "labelling_policy": (
        "the bank is NOT labelled row by row by the model judges. A ~24,000-row bank is "
        "~96,000 judge calls across two judges and two passes. A deterministic stratified "
        "audit sample of 1,800 rows is frozen first by "
        "`rdl graph-detector-v4-2-bank-audit`, per (partition, stratum) cell, on "
        "generation metadata only and never on a detector score. The cell includes the "
        "partition because the pre-registered minima are conditions on the held-out gate "
        "population, and a bank-wide draw can meet every one of them while leaving the "
        "held-out partition below all of them."
    ),
    "why_a_budget_is_part_of_the_freeze": (
        "seeds fix WHICH trajectories are drawn; the budget fixes HOW MANY. A bank whose "
        "size is decided at generation time can be grown until a gate passes, and nothing "
        "in the artifact would show it. FINAL_GATE_BANK_MANIFEST.json froze the seeds and "
        "the minimum label counts and not this, which is the gap these commands close."
    ),
    "why_n_samples_equals_primary_k": (
        "primary_k is the number of samples the success-at-k statistic is read over. "
        "n_samples < primary_k cannot be measured at all: the estimate has no draws to "
        "read. The previous freeze said 8 samples and k=32, which is why this file now "
        "validates the budget instead of only recording it."
    ),
}


def _sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha_file(path: Path) -> str | None:
    return _sha_text(path.read_text(encoding="utf-8")) if path.exists() else None


# ------------------------------------------------------------------ budget validation --


def validate_budget(budget: Mapping) -> list[str]:
    """Every internally impossible thing a budget can say. Returns the failures.

    Run at freeze time and again at build time, because a manifest written by an older
    version of this module is exactly the case the second check exists for.
    """
    failures: list[str] = []
    groups = budget.get("groups")
    if not isinstance(groups, Mapping) or set(groups) != set(GROUPS):
        return [
            f"budget must declare groups {sorted(GROUPS)}, got "
            f"{sorted(groups) if isinstance(groups, Mapping) else type(groups).__name__}. "
            "A single undifferentiated run set cannot separate the retain false-alarm pool "
            "from the protected-clean one."
        ]

    all_seeds: dict[int, str] = {}
    for name, group in groups.items():
        n_samples = int(group.get("n_samples", 0))
        primary_k = int(group.get("primary_k", 0))
        if primary_k <= 0:
            failures.append(f"{name}: primary_k must be positive, got {primary_k}")
        if n_samples < primary_k:
            failures.append(
                f"{name}: n_samples={n_samples} < primary_k={primary_k}. success-at-k is "
                "read over k draws; fewer draws than k cannot be measured at all."
            )
        if int(group.get("n_items", 0)) <= 0:
            failures.append(f"{name}: n_items must be positive, got {group.get('n_items')}")
        seeds = list(group.get("seeds", ()))
        if len(seeds) != int(group.get("n_runs", 0)):
            failures.append(
                f"{name}: {len(seeds)} seed(s) declared for n_runs={group.get('n_runs')}"
            )
        if len(set(seeds)) != len(seeds):
            failures.append(f"{name}: seeds {seeds} are not unique within the group")
        for seed in seeds:
            if seed in all_seeds and all_seeds[seed] != name:
                failures.append(
                    f"seed {seed} is declared in both {all_seeds[seed]!r} and {name!r}. "
                    "A retain run sharing a natural run's seed is one draw wearing two "
                    "labels."
                )
            all_seeds[seed] = name
        if not group.get("required_arm"):
            failures.append(f"{name}: no required_arm; any arm's text would be accepted")
    for seed in all_seeds:
        if seed in SEALED_FINAL_BANK_SEEDS:
            failures.append(
                f"seed {seed} is one of the SEALED final-bank seeds "
                f"{list(SEALED_FINAL_BANK_SEEDS)}; the engineering bank must be a "
                "different draw or it is the final bank under a new name."
            )
    return failures


# ----------------------------------------------------------------------- freeze --


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

    failures = validate_budget(GENERATION_BUDGET)
    if failures:  # pragma: no cover - a constant that fails its own validator is a bug
        for failure in failures:
            typer.echo(f"  [FAIL] {failure}", err=True)
        raise typer.BadParameter(
            "GENERATION_BUDGET does not validate. Refusing to freeze an impossible design."
        )

    engineering = {
        "schema": "graph-detector-v4-2-engineering-bank-manifest-v3",
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
            "natural": list(ENGINEERING_BANK_SEEDS),
            "retain": list(ENGINEERING_RETAIN_SEEDS),
            "why_two_groups": (
                "a natural run and a retain run are different cohorts with different "
                "roles in the gate. One shared set of four makes 'four runs' ambiguous "
                "and lets one draw be counted as two."
            ),
            "why_new": (
                "the study's base_seed is 1729 and the v4 engineering-only bank was drawn "
                "under it; the final bank is 40241-40244 and is sealed. A third draw "
                "needs a third set, or it is one of the first two under a new name."
            ),
            "disjoint_from_final": True,
        },
        "generation_budget": GENERATION_BUDGET,
        "arms": {
            "natural": UNGUARDED_ARM,
            "retain": UNGUARDED_ARM,
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
            "sample": (
                "a deterministic stratified audit sample, frozen by "
                "`rdl graph-detector-v4-2-bank-audit` before the detector scores anything"
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
        "schema": "graph-detector-v4-2-final-bank-budget-v3",
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
        "seeds_note": (
            "the final bank's seed groups are derived from the sealed set at generation "
            "time and are not fixed here; the sealed set is what this file is bound to."
        ),
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
    typer.echo(f"engineering natural seeds: {list(ENGINEERING_BANK_SEEDS)}")
    typer.echo(f"engineering retain seeds:  {list(ENGINEERING_RETAIN_SEEDS)}")
    typer.echo(f"final seeds:               {list(SEALED_FINAL_BANK_SEEDS)}  (SEALED)")


# ----------------------------------------------------------------- build bank --


def run_meta(path: Path) -> dict:
    """The design of one graph run, read out of the structure ``RUN_MANIFEST.json`` has.

    Every field this returns is one the real artifact carries. Absent fields come back as
    ``None`` and are treated as verification failures by :func:`check_group` rather than
    skipped, because "the manifest does not say" and "the manifest says the right thing"
    must not produce the same verdict.
    """
    manifest = path if path.is_file() else path / RUN_MANIFEST_FILENAME
    if not manifest.exists():
        raise typer.BadParameter(
            f"{manifest} is absent; --natural-run/--retain-run must name a graph run "
            f"directory containing {RUN_MANIFEST_FILENAME}"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    sampling = payload.get("sampling") or {}
    cohort = payload.get("evaluation_cohort") or {}
    arms = payload.get("arms") or []
    arm_names = sorted({str(a.get("arm")) for a in arms if isinstance(a, Mapping) and a.get("arm")})
    shards = payload.get("evidence_shards") or []
    return {
        "path": str(path),
        "manifest": str(manifest),
        "manifest_sha256": _sha_file(manifest),
        "base_seed": sampling.get("base_seed"),
        "n_samples": sampling.get("n_samples"),
        "primary_k": sampling.get("primary_k"),
        "k_values": list(sampling.get("k_values", ())),
        "n_items": cohort.get("n_items"),
        "n_concepts": cohort.get("n_concepts"),
        "cohort_split": cohort.get("split"),
        "cohort_fingerprint": cohort.get("fingerprint"),
        "is_retain": cohort.get("is_retain"),
        "arms": arm_names,
        "challenges": list(payload.get("challenges", ())),
        "protocol": payload.get("protocol"),
        "complete": payload.get("complete"),
        "study_id": payload.get("study_id"),
        "git_sha": payload.get("git_sha"),
        "profile_reportable": payload.get("profile_reportable"),
        "n_shards": len(shards),
        "n_rows": sum(int(s.get("n_rows", 0)) for s in shards if isinstance(s, Mapping)) or None,
    }


def check_group(runs: Sequence[Mapping], group_name: str, group: Mapping) -> list[str]:
    """Every way one group's supplied runs can fail its pre-registration."""
    failures: list[str] = []
    where = f"{group_name} group"

    if len(runs) != int(group["n_runs"]):
        failures.append(f"{where}: {len(runs)} run(s) supplied, {group['n_runs']} pre-registered")

    raw_seeds = [r.get("base_seed") for r in runs]
    if any(s is None for s in raw_seeds):
        failures.append(
            f"{where}: a run manifest records no sampling.base_seed; the draw cannot be verified"
        )
    else:
        seeds = [int(s) for s in raw_seeds if s is not None]
        if len(set(seeds)) != len(seeds):
            failures.append(
                f"{where}: seeds {sorted(seeds)} are not unique. One run per seed is "
                "pre-registered; the same seed twice is one draw counted twice."
            )
        expected = {int(s) for s in group["seeds"]}
        if set(seeds) != expected:
            failures.append(f"{where}: seeds {sorted(seeds)} != pre-registered {sorted(expected)}")

    for run in runs:
        name = Path(run["path"]).name
        if run.get("complete") is not True:
            failures.append(f"{where}: {name} records complete={run.get('complete')!r}")
        if run.get("protocol") != group["protocol"]:
            failures.append(
                f"{where}: {name} is protocol {run.get('protocol')!r}, not {group['protocol']!r}"
            )
        if list(run.get("challenges") or ()) != [group["challenge"]]:
            failures.append(
                f"{where}: {name} carries challenges {run.get('challenges')}, "
                f"not [{group['challenge']!r}]"
            )
        if run.get("cohort_split") != group["cohort_split"]:
            failures.append(
                f"{where}: {name} is cohort split {run.get('cohort_split')!r}, "
                f"not {group['cohort_split']!r}"
            )
        if bool(run.get("is_retain")) is not bool(group["is_retain"]):
            failures.append(
                f"{where}: {name} has evaluation_cohort.is_retain="
                f"{run.get('is_retain')!r}, and this group requires "
                f"{group['is_retain']!r}. A retain run in the natural group would put "
                "retain text in the recall numerator."
            )
        if group["required_arm"] not in (run.get("arms") or ()):
            failures.append(
                f"{where}: {name} does not run arm {group['required_arm']!r} "
                f"(it runs {run.get('arms')})"
            )
        if run.get("n_items") != group["n_items"]:
            failures.append(
                f"{where}: {name} has evaluation_cohort.n_items={run.get('n_items')!r}, "
                f"pre-registered {group['n_items']}"
            )
        if run.get("primary_k") != group["primary_k"]:
            failures.append(
                f"{where}: {name} has sampling.primary_k={run.get('primary_k')!r}, "
                f"pre-registered {group['primary_k']}"
            )
        n_samples, primary_k = run.get("n_samples"), run.get("primary_k")
        if n_samples is None:
            failures.append(f"{where}: {name} records no sampling.n_samples")
        elif primary_k is not None and int(n_samples) < int(primary_k):
            failures.append(
                f"{where}: {name} drew n_samples={n_samples} and claims primary_k="
                f"{primary_k}. success-at-k needs at least k draws."
            )
        elif int(n_samples) != int(group["n_samples"]):
            failures.append(
                f"{where}: {name} drew n_samples={n_samples}, pre-registered {group['n_samples']}"
            )
        rows = run.get("n_rows")
        if rows is not None and int(rows) > int(group["max_rows_per_run"]):
            failures.append(
                f"{where}: {name} has {rows} rows, above the pre-registered "
                f"max_rows_per_run {group['max_rows_per_run']}"
            )
    return failures


def detector_v4_2_build_bank(
    bank: str = typer.Option("engineering", "--bank", help="engineering | final"),
    natural_run: list[Path] = typer.Option(
        [], "--natural-run", help="one graph run per pre-registered natural seed"
    ),
    retain_run: list[Path] = typer.Option(
        [], "--retain-run", help="one graph run per pre-registered retain seed"
    ),
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
    budget = manifest["generation_budget"]

    # Re-validated here, not only at freeze time: a manifest on disk may predate this
    # module, and the 60-item / 8-sample / k=32 freeze is exactly that case.
    budget_failures = validate_budget(budget)
    if budget_failures:
        for failure in budget_failures:
            typer.echo(f"  [FAIL] {failure}", err=True)
        raise typer.BadParameter(
            f"{manifest_path} carries a budget that cannot be generated. Re-run "
            "`rdl graph-detector-v4-2-freeze-banks` to rewrite it, and record the "
            "correction in DECISIONS.md."
        )

    supplied = {"natural": list(natural_run), "retain": list(retain_run)}
    for name, paths in supplied.items():
        if not paths:
            raise typer.BadParameter(
                f"--{name}-run is required, once per pre-registered {name} seed "
                f"{budget['groups'][name]['seeds']}. The two groups are separate draws; "
                "passing only one of them would build a bank with no "
                + ("retain false-alarm pool" if name == "retain" else "recall numerator")
                + "."
            )

    runs_by_group = {name: [run_meta(p) for p in paths] for name, paths in supplied.items()}
    failures: list[str] = []
    for name, runs in runs_by_group.items():
        failures.extend(check_group(runs, name, budget["groups"][name]))

    # A run may not appear in both groups, whatever its manifest says.
    natural_paths = {r["path"] for r in runs_by_group["natural"]}
    retain_paths = {r["path"] for r in runs_by_group["retain"]}
    shared = natural_paths & retain_paths
    if shared:
        failures.append(
            f"run(s) {sorted(shared)} were supplied to BOTH groups. One directory cannot "
            "be both the recall numerator and the retain false-alarm pool."
        )

    verification = {
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "groups": dict(runs_by_group),
        "failures": failures,
        "passed": not failures,
    }
    for failure in failures:
        typer.echo(f"  [FAIL] {failure}", err=True)
    if failures:
        raise typer.BadParameter(
            f"{len(failures)} verification failure(s) against the pre-registration in "
            f"{manifest_path}. Refusing to build. A bank assembled from runs that do not "
            "match its freeze is not the bank that was pre-registered, whatever the file "
            "is called."
        )
    typer.echo(
        f"verified {len(runs_by_group['natural'])} natural and "
        f"{len(runs_by_group['retain'])} retain runs against {manifest_path}"
    )
    if check_only:
        atomic_json(output_dir / "ENGINEERING_BANK_VERIFICATION.json", verification)
        typer.echo(f"wrote {output_dir / 'ENGINEERING_BANK_VERIFICATION.json'}  (--check-only)")
        raise typer.Exit(0)

    # Assembly reuses the v4 collector so the bank's row shape is identical to the one the
    # detector was developed against — a bank whose rows are built by a second code path
    # would make the comparison a comparison of collectors.
    from .detector_v4_data import _collect, _question_bank

    questions, _answers, provenance = _question_bank(v4_1_dir.parent / "discovery.json")

    rows: list[dict] = []
    seen: set[str] = set()
    n_duplicates = 0
    n_rows_by_group: dict[str, int] = {}
    for name, runs in runs_by_group.items():
        group = budget["groups"][name]
        before = len(rows)
        for meta in runs:
            collected, _ = _collect(Path(meta["path"]), limit=int(group["max_rows_per_run"]))
            for row in collected:
                request = questions.get(row["item_id"], row["request"])
                # (text, question), not text alone. The same refusal string under two
                # different protected questions is two rows, and collapsing them deletes
                # one question's false-alarm evidence.
                digest = _sha_text(f"{row['text']}\x00{request}")
                if digest in seen:
                    n_duplicates += 1
                    continue
                seen.add(digest)
                rows.append(
                    {
                        **row,
                        "request": request,
                        "population": "retain" if group["is_retain"] else "protected",
                        "group": name,
                        "pair_sha256": digest,
                    }
                )
        n_rows_by_group[name] = len(rows) - before

    if len(rows) > int(budget["max_rows_total"]):
        raise typer.BadParameter(
            f"{len(rows)} rows collected, above the pre-registered max_rows_total "
            f"{budget['max_rows_total']}"
        )

    natural_rows = [r for r in rows if r["group"] == "natural"]
    retain_rows = [r for r in rows if r["group"] == "retain"]
    leaking = [r for r in natural_rows if r["leaking"] is True]
    clean = [r for r in natural_rows if r["leaking"] is False]
    unlabelled = [r for r in natural_rows if r["leaking"] is None]

    def halve(subset: Sequence[Mapping]) -> tuple[list[dict], list[dict]]:
        """Content-addressed halving on the PAIR digest, which is the declared split rule.

        ``_halve`` splits on the text hash. The budget's ``split_rule`` says
        ``sha256(text || question)``, and the two disagree for any text that appears under
        more than one protected question — which is exactly what the pair digest exists to
        keep apart. Splitting on the text alone would put the same text's two rows on the
        same side and make the two halves correlated through it.
        """
        dev = [dict(r) for r in subset if int(str(r["pair_sha256"])[:2], 16) % 2 == 0]
        gate = [dict(r) for r in subset if int(str(r["pair_sha256"])[:2], 16) % 2 == 1]
        return dev, gate

    clean_dev, clean_gate = halve(clean)
    leak_dev, leak_gate = halve(leaking)
    # The retain pool is halved by the SAME rule and lands inside the two partitions. It
    # used to be one block that both selected the threshold and supplied the reported
    # retain FPR, which made that FPR a description of the selection.
    retain_dev, retain_gate = halve(retain_rows)

    def strip(subset: Sequence[Mapping]) -> list[dict]:
        return [
            {
                "text": r["text"],
                "request": r["request"],
                "item_id": r["item_id"],
                "population": r["population"],
                "text_sha256": _sha_text(r["text"]),
                "pair_sha256": r["pair_sha256"],
            }
            for r in subset
        ]

    payload = {
        "schema": BANK_SCHEMA,
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
                # Retain rows sit INSIDE the partition, keeping their own key so the
                # retain denominator stays separate from the protected-clean one — two
                # ceilings, two pools — while "development" means the same set of rows for
                # both populations.
                "retain": strip(retain_dev),
                "usage": "threshold selection ONLY, protected and retain alike",
            },
            "heldout": {
                "clean": strip(clean_gate),
                "leaking": strip(leak_gate),
                "retain": strip(retain_gate),
                "usage": "opened once per frozen checkpoint+threshold, and recorded",
            },
            "split_rule": budget["split_rule"],
            "retain_split_rule": budget["retain_split_rule"],
        },
        "counts": {
            "n_texts": len(rows),
            "n_natural": len(natural_rows),
            "n_retain": len(retain_rows),
            "n_retain_development": len(retain_dev),
            "n_retain_heldout": len(retain_gate),
            "n_leaking": len(leaking),
            "n_clean": len(clean),
            "n_unlabelled": len(unlabelled),
            "n_duplicates_dropped": n_duplicates,
            "by_group": n_rows_by_group,
        },
        "unlabelled_policy": budget["unlabelled_policy"],
        "deduplication": budget["deduplication"],
        "request_provenance": provenance,
        "labels_are_not_here": (
            "this file carries no answer_attempt label. The gate is scored against the "
            "adjudicated model-judge labels produced by "
            "`rdl graph-detector-v4-2-bank-audit` over a frozen stratified sample of it."
        ),
    }
    payload["content_sha256"] = _sha_text(
        json.dumps(payload["partitions"], sort_keys=True, separators=(",", ":"))
    )
    atomic_json(output_dir / BANK_FILENAME[bank], payload)
    typer.echo(
        f"wrote {output_dir / BANK_FILENAME[bank]}  "
        f"(natural {len(natural_rows)}: leaking {len(leaking)}, clean {len(clean)}; "
        f"retain {len(retain_rows)} = {len(retain_dev)} development + {len(retain_gate)} "
        f"heldout; dropped {n_duplicates} duplicate pairs)"
    )
    typer.echo("")
    typer.echo(
        f"next: rdl graph-detector-v4-2-bank-audit --bank-path {output_dir / BANK_FILENAME[bank]}"
    )


# ------------------------------------------------------------------ final gate --


def _load_bank_rows(payload: Mapping) -> list[dict]:
    """Every row of a bank, flattened, carrying its partition and population.

    Two axes, deliberately orthogonal: ``partition`` is development/held-out and
    ``population`` is protected/retain. A v2 bank encoded the retain rows as a *third
    partition*, which is what let one undivided retain pool both constrain the threshold
    and supply the reported retain FPR. Such a bank is refused rather than reinterpreted:
    guessing which of its retain rows were "held out" would invent the split it never had.
    """
    partitions = payload.get("partitions", {})
    legacy = partitions.get("retain")
    if isinstance(legacy, Mapping) and "all" in legacy:
        raise typer.BadParameter(
            "this bank carries one undivided `partitions.retain.all` block (schema v2). "
            "Its retain rows chose the threshold and then reported the retain false-alarm "
            "rate, so that rate was never held out. Rebuild it with "
            "`rdl graph-detector-v4-2-build-bank`, which halves the retain pool by the "
            "same content-addressed rule as the protected rows. The split cannot be "
            "applied retroactively here without inventing which half was which."
        )
    out: list[dict] = []
    for partition in ("development", "heldout"):
        block = partitions.get(partition, {})
        for key, leaking in (("clean", False), ("leaking", True), ("retain", None)):
            for row in block.get(key, ()):
                out.append(
                    {
                        **row,
                        "partition": partition,
                        "population": row.get(
                            "population", "retain" if key == "retain" else "protected"
                        ),
                        "nli_leaking": leaking,
                    }
                )
    return out


def detector_v4_2_final_gate(
    bank: str = typer.Option("engineering", "--bank", help="engineering | final"),
    bank_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--bank-dir"),
    labels: Path = typer.Option(..., "--labels", help="the adjudicated bank-audit labels, JSONL"),
    model_artifact: Path = typer.Option(..., "--model-artifact"),
    operating_point: Path | None = typer.Option(
        None,
        "--operating-point",
        help="DETECTOR_V4_2_OPERATING_POINT.json. The threshold comes from here.",
    ),
    threshold: float | None = typer.Option(
        None,
        "--threshold",
        help="refused unless it equals the frozen operating point. Kept only so that "
        "passing one is an error rather than an override.",
    ),
    audit_manifest: Path | None = typer.Option(
        None, "--audit-manifest", help="BANK_AUDIT_MANIFEST.json for this bank"
    ),
    alignment_report: Path | None = typer.Option(
        None, "--alignment-report", help="the passing model-label alignment report"
    ),
    policy_cohort: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/discovery.json"), "--policy-cohort"
    ),
    backend: str = typer.Option("cross_encoder", "--backend"),
    device: str = typer.Option("", "--device"),
    partition: str = typer.Option("heldout", "--partition", help="heldout | development"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
    reopen: bool = typer.Option(
        False,
        "--reopen",
        help="engineering bank only; records that the bank is now development data",
    ),
) -> None:
    """Score a bank once, at the FROZEN threshold, and record that it was opened.

    This command *scores*. The first version only wrote an opening record and printed the
    name of another command to run, which meant the bank could be marked opened while
    nothing had been measured on it — and the number that eventually appeared came from
    ``rdl graph-detector-v4-gates``, which reads the **v4** natural bank and the **v4.1**
    audit, not this bank at all. A gate that does not read the surface it gates is not a
    gate.

    Four things must hold before anything is scored, and none of them held before:

    * the labels came from an audit of THIS bank that met its pre-registered minima;
    * an alignment report over those labels PASSED, with zero unresolved disagreements and
      zero provenance failures, and vouches for this exact label file by hash;
    * no ``audit_id`` and no bank pair is labelled twice;
    * the threshold is the one frozen in ``DETECTOR_V4_2_OPERATING_POINT.json`` by
      ``select-operating-point`` on the DEVELOPMENT partition — not a float typed here.

    The opening record is the enforcement of "once". Without it, that phrase is a sentence
    in a protocol that nothing checks, and the second open — the one after a disappointing
    first — is the one that would never be mentioned.
    """
    from ..eval.detector_v4_1 import (
        goal_a_gate_inputs,
        goal_a_summarise,
        score_goal_a,
    )
    from .detector_v4_2_gate_bridge import (
        ALIGNMENT_REPORT_FILENAME,
        BANK_AUDIT_MANIFEST_FILENAME,
        HELDOUT_MINIMA,
        OPERATING_POINT_FILENAME,
        bind_labels_to_bank,
        check_heldout_minima,
        concept_index,
        labelled_rows_for_partition,
        load_label_map,
        load_operating_point,
        require_audit_gate,
        score_with_backend,
    )

    if bank not in BANK_FILENAME:
        raise typer.BadParameter(f"--bank must be engineering or final, got {bank!r}")
    if partition not in ("heldout", "development"):
        raise typer.BadParameter(
            f"--partition must be heldout or development, got {partition!r}. There is no "
            "`all`: pooling the partition the threshold was chosen on with the one it is "
            "reported on produces a number that is neither."
        )
    bank_path = bank_dir / BANK_FILENAME[bank]
    if not bank_path.exists():
        raise typer.BadParameter(
            f"{bank_path} is absent; run `rdl graph-detector-v4-2-build-bank` first"
        )
    if not labels.exists():
        raise typer.BadParameter(
            f"{labels} is absent. The gate is scored against adjudicated model-judge "
            "labels for THIS bank; run `rdl graph-detector-v4-2-bank-audit` and the two "
            "judges over its inputs first. Scoring against the v4.1 audit instead would "
            "gate a new bank on labels drawn from a different one."
        )
    record_path = output_dir / OPENING_RECORD_FILENAME[bank]
    n_prior = 0
    if record_path.exists():
        previous = json.loads(record_path.read_text(encoding="utf-8"))
        n_prior = int(previous.get("n_openings", 0))
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
    if model_manifest.get("reportable") is False:
        raise typer.BadParameter(
            f"{model_artifact} records reportable=false — the training run left the "
            "preregistration (see its `preregistration.deviations`). Opening a gate at "
            "its checkpoint would produce numbers nothing may report."
        )

    # --------------------------------------------------- the labels, and their gate --
    gate_record = require_audit_gate(
        audit_manifest=audit_manifest or (bank_dir / BANK_AUDIT_MANIFEST_FILENAME),
        alignment_report=alignment_report or (output_dir / ALIGNMENT_REPORT_FILENAME),
        labels=labels,
        bank_payload=payload,
    )
    label_rows = load_label_map(labels)
    bank_rows = _load_bank_rows(payload)
    by_pair = bind_labels_to_bank(label_rows, bank_rows, payload, labels_path=labels)

    # --------------------------------------------------------- the frozen threshold --
    point = load_operating_point(
        operating_point or (output_dir / OPERATING_POINT_FILENAME),
        bank_content_sha256=str(payload.get("content_sha256")),
        model_artifact=model_artifact,
        requested_threshold=threshold,
        model_manifest=model_manifest,
    )
    frozen_threshold = float(point["selected_threshold"])

    scored_rows = labelled_rows_for_partition(
        label_rows,
        by_pair,
        partition=partition,
        concept_of=concept_index(policy_cohort),
    )
    if not scored_rows:
        raise typer.BadParameter(
            f"no labelled rows fall in partition {partition!r}. Nothing to score."
        )

    # The denominators, BEFORE the model is loaded. The audit plan draws to these minima
    # and the bank audit checks its own draw, but judging, adjudication and the PARTIAL
    # class all remove rows between the draw and here — so the population the gate scores
    # is not the population that was checked. Six rates over 40 retain rows are printed in
    # the same shape as six rates over 400.
    denominators, denominator_failures = check_heldout_minima(scored_rows)
    if partition == "heldout" and denominator_failures:
        for failure in denominator_failures:
            typer.echo(f"  [DENOMINATOR] {failure}", err=True)
        raise typer.BadParameter(
            f"the held-out population is below its pre-registered minima "
            f"{dict(HELDOUT_MINIMA)}. Refusing to open the bank: the response is a "
            "pre-registered extension that judges more rows, not a gate read over fewer. "
            "Nothing has been scored and the opening record has not been written."
        )

    predictions, compute = score_with_backend(
        scored_rows,
        backend=backend,
        model_artifact=model_artifact if backend == "cross_encoder" else None,
        device=device,
        policy_cohort=policy_cohort,
    )
    summary = goal_a_summarise(scored_rows, predictions, threshold=frozen_threshold)
    gate = score_goal_a(goal_a_gate_inputs(summary))
    parameter_device = str(compute["device_of_parameters"])

    result = {
        "schema": "graph-detector-v4-2-bank-gate-result-v2",
        "protocol": V4_2_PROTOCOL,
        "bank": bank,
        "bank_path": str(bank_path),
        "bank_content_sha256": payload.get("content_sha256"),
        "labels_file": str(labels),
        "labels_sha256": _sha_file(labels),
        "partition": partition,
        "threshold": frozen_threshold,
        "threshold_frozen_before_opening": True,
        "threshold_source": str(operating_point or (output_dir / OPERATING_POINT_FILENAME)),
        "threshold_selected_on": point.get("partition_selected_on"),
        "operating_point_utc": point.get("utc"),
        "label_gate": gate_record,
        "backend": backend,
        "model_artifact": str(model_artifact),
        "selected_checkpoint": model_manifest.get("selected_checkpoint"),
        "checkpoint_hashes": model_manifest.get("selected_checkpoint_hashes"),
        "checkpoint_hashes_verified_at_load": compute["checkpoint_hashes_verified"],
        "device_requested": device or None,
        "device_of_parameters": parameter_device,
        "gpu_used": compute["gpu_used"],
        "compute": compute,
        "n_rows_scored": len(scored_rows),
        # The populations every rate below is computed over, recorded beside the rates.
        # A reader who sees micro recall 0.86 is entitled to know it came from 150 rows
        # and not from 12.
        "heldout_denominators": denominators,
        "heldout_minima": dict(HELDOUT_MINIMA),
        "heldout_denominators_met": not denominator_failures,
        "n_rows_by_population": {
            name: sum(1 for r in scored_rows if r["population"] == name)
            for name in sorted({r["population"] for r in scored_rows})
        },
        "summary": summary,
        "judge_population": "two_independent_llm_judges",
        "human_grounded": False,
        "publication_label_valid": False,
        "is_still_a_gate": bank == "final" or n_prior == 0,
        **gate,
    }
    atomic_json(output_dir / GATE_RESULT_FILENAME[bank], result)

    record = {
        "schema": "graph-detector-v4-2-bank-opening-record-v3",
        "protocol": V4_2_PROTOCOL,
        "bank": bank,
        "bank_path": str(bank_path),
        "bank_content_sha256": payload.get("content_sha256"),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "threshold": frozen_threshold,
        "threshold_frozen_before_opening": True,
        "threshold_source": result["threshold_source"],
        "label_gate": gate_record,
        "partition": partition,
        "labels_file": str(labels),
        "labels_sha256": result["labels_sha256"],
        "model_artifact": str(model_artifact),
        "selected_checkpoint": model_manifest.get("selected_checkpoint"),
        "checkpoint_hashes": model_manifest.get("selected_checkpoint_hashes"),
        "label_authority": model_manifest.get("label_authority"),
        "human_grounded": bool(model_manifest.get("human_grounded")),
        "publication_label_valid": bool(model_manifest.get("publication_label_valid")),
        "device_requested": device or None,
        "device_of_parameters": parameter_device,
        "gate_result": str(output_dir / GATE_RESULT_FILENAME[bank]),
        "all_gates_passed": gate["all_gates_passed"],
        "n_openings": n_prior + 1,
        "is_still_a_gate": bank == "final" or n_prior == 0,
        "meaning": (
            "this bank has now been read. Every number computed from it after this record "
            "was written describes data the detector has seen."
        ),
    }
    atomic_json(record_path, record)

    typer.echo(f"wrote {output_dir / GATE_RESULT_FILENAME[bank]}")
    typer.echo(f"wrote {record_path}  (opening #{record['n_openings']})")
    typer.echo("")
    for entry in gate["gates"]:
        mark = "PASS" if entry["passed"] else ("n/a " if entry["passed"] is None else "FAIL")
        typer.echo(
            f"  [{mark}] {entry['gate']}: {entry['measured']} "
            f"(need {entry['comparison']} {entry['bound']})"
        )
    typer.echo("")
    typer.echo(
        "the bank is now open. If this run's numbers disappoint, the response is a NEW "
        "bank, not a second look at this one."
    )
    raise typer.Exit(0 if gate["all_gates_passed"] else 1)
