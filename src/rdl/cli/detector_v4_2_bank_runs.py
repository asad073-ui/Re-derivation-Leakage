"""``rdl graph-detector-v4-2-plan-bank-runs`` — the eight generation runs, from the freeze.

The engineering bank pre-registers eight draws: natural seeds 50241-50244 over the
discovery cohort and retain seeds 51241-51244 over retain_utility, all ``graph_flow``,
all the unguarded arm, 32 samples at ``primary_k=32``. ``build-bank`` refuses runs that do
not match that, which is the point of the freeze — and until v4.2.3 there was no way to
*produce* runs that did.

``sampling.base_seed`` lives in the frozen study file and ``GraphLaunchConfig`` carries only
``study`` and ``active_profile``, so no override could reach it. The only route to a second
draw was editing the frozen study between runs: eight hand-edits, none of them recorded, on
the file whose whole job is to be the thing that did not change. ``graph-run --base-seed``
now takes a pre-registered seed, and this command writes the exact eight invocations out of
the manifest so the operator copies rather than composes them.

It generates nothing and calls nothing. On the GPU box the output is a shell script and a
JSON plan; every command it emits is one ``graph-run``, and their outputs are what
``build-bank`` then verifies against the same manifest this read.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from ..eval.detector_v4_2 import V4_2_PROTOCOL
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_2_banks import ENGINEERING_MANIFEST_FILENAME, validate_budget
from .detector_v4_2_llm_judge import DEFAULT_V4_2_OUT

__all__ = ["bank_run_commands", "detector_v4_2_plan_bank_runs"]

PLAN_FILENAME = "ENGINEERING_BANK_RUN_PLAN.json"
SCRIPT_FILENAME = "ENGINEERING_BANK_RUNS.sh"

# Which cohort manifest each group's runs read. The cohort split is already in the budget;
# this maps it to the file `graph-run --cohort` wants, so the emitted command is complete
# rather than nearly complete.
COHORT_FILENAME = {
    "discovery": "discovery.json",
    "retain_utility": "retain_utility.json",
}


def _posix(path: Path) -> str:
    """A path as the GPU box will read it, whatever machine wrote the plan."""
    return Path(path).as_posix()


def bank_run_commands(
    budget: dict,
    *,
    launch: Path,
    cohort_dir: Path,
    output_root: Path,
    profile: str | None,
) -> list[dict]:
    """One command per pre-registered seed, in a stable order."""
    commands: list[dict] = []
    for group_name in sorted(budget["groups"]):
        group = budget["groups"][group_name]
        cohort = COHORT_FILENAME.get(str(group["cohort_split"]))
        if cohort is None:
            raise typer.BadParameter(
                f"{group_name}: no cohort file is known for split "
                f"{group['cohort_split']!r}. Add it to COHORT_FILENAME rather than "
                "letting the operator guess."
            )
        for seed in group["seeds"]:
            out = output_root / f"engineering-{group_name}-seed{seed}"
            # POSIX separators, always. This plan is written on whatever machine froze the
            # bank — here, Windows — and executed on the rented Linux box, where a
            # backslash is an escape character and `data\cohorts\...` is not a path.
            argv = [
                "python -m rdl.cli graph-run",
                f"--launch {_posix(launch)}",
                *([f"--profile {profile}"] if profile else []),
                f"--cohort {_posix(cohort_dir / cohort)}",
                f"--protocol {group['protocol']}",
                f"--challenges {group['challenge']}",
                f"--n-samples {group['n_samples']}",
                f"--base-seed {seed}",
                f"--output {_posix(out)}",
            ]
            commands.append(
                {
                    "group": group_name,
                    "base_seed": int(seed),
                    "cohort_split": group["cohort_split"],
                    "is_retain": bool(group["is_retain"]),
                    "required_arm": group["required_arm"],
                    "n_items": group["n_items"],
                    "n_samples": group["n_samples"],
                    "primary_k": group["primary_k"],
                    "output": _posix(out),
                    "command": " ".join(argv),
                }
            )
    return commands


def detector_v4_2_plan_bank_runs(
    manifest_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--manifest-dir"),
    launch: Path = typer.Option(Path("configs/graph/launch.yaml"), "--launch"),
    profile: str | None = typer.Option(
        "rtx3090_1b", "--profile", help="the graph profile these runs are generated under"
    ),
    cohort_dir: Path = typer.Option(Path("data/cohorts/graph_unlearning_v1"), "--cohort-dir"),
    output_root: Path = typer.Option(Path("runs/graph"), "--output-root"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
) -> None:
    """Write the exact eight ``graph-run`` invocations the frozen bank manifest implies."""
    manifest_path = manifest_dir / ENGINEERING_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise typer.BadParameter(
            f"{manifest_path} is absent; run `rdl graph-detector-v4-2-freeze-banks` first. "
            "The runs are derived from the freeze, not the other way round."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    budget = manifest["generation_budget"]
    failures = validate_budget(budget)
    if failures:
        for failure in failures:
            typer.echo(f"  [FAIL] {failure}", err=True)
        raise typer.BadParameter(
            f"{manifest_path} carries a budget that cannot be generated; refusing to emit "
            "commands for it."
        )

    commands = bank_run_commands(
        budget,
        launch=launch,
        cohort_dir=cohort_dir,
        output_root=output_root,
        profile=profile,
    )
    plan = {
        "schema": "graph-detector-v4-2-bank-run-plan-v1",
        "protocol": V4_2_PROTOCOL,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "manifest": str(manifest_path),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "profile": profile,
        "n_runs": len(commands),
        "runs": commands,
        "generates_nothing": True,
        "why": (
            "the frozen study fixes base_seed=1729 and the bank pre-registers eight other "
            "seeds. Before `graph-run --base-seed`, producing them meant editing the "
            "frozen study file between runs — eight unrecorded edits to the artifact whose "
            "purpose is to be the thing that did not change."
        ),
        "next": (
            "run every command, then `rdl graph-detector-v4-2-build-bank` with "
            "--natural-run/--retain-run pointing at their outputs. build-bank verifies "
            "each run's manifest against this same freeze and refuses a mismatch."
        ),
    }
    atomic_json(output_dir / PLAN_FILENAME, plan)

    script = output_dir / SCRIPT_FILENAME
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "# GENERATED by `rdl graph-detector-v4-2-plan-bank-runs`. Do not edit:",
                "# every seed here is pre-registered in ENGINEERING_BANK_MANIFEST.json and",
                "# `build-bank` verifies the runs against it.",
                "set -euo pipefail",
                "",
                *[f"# {c['group']} seed {c['base_seed']}\n{c['command']}\n" for c in commands],
            ]
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    typer.echo(f"wrote {output_dir / PLAN_FILENAME}")
    typer.echo(f"wrote {script}")
    typer.echo("")
    for command in commands:
        typer.echo(f"  {command['group']:<8} seed {command['base_seed']}  -> {command['output']}")
    typer.echo("")
    typer.echo(f"{len(commands)} runs. Nothing has been generated; this command calls nothing.")
