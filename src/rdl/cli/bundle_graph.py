"""``rdl graph-bundle`` — one study-level verdict over many run-level reports.

A run reports one challenge under one protocol, which is right: pooling either dimension
produces a number that answers neither question. But it leaves every individual report
structurally unable to state the study's claim, because the claim spans runs:

    leakage           measured on a FORGET cohort, under `graph_flow`
    retain utility    measured on a RETAIN cohort, in a different run entirely
    detector FPR      measured on held-out calibration negatives
    refusal /         measured on the treatment arm of the flow runs, and the only thing
    collaboration     that distinguishes containment from a defence that stopped working

That is why the 50x32 natural reports carry `reportable: true` beside
`utility_gate.applicable: false`: the retain run existed, it passed, and nothing linked
the two. A reader had to know to go and find it. This command is that link.

**What it does not do.** It does not pool, average or recompute anything. Every number is
carried over verbatim from the run report that measured it, with the run id attached, so
each figure remains traceable to the evidence that produced it. What the bundle adds is
the *conjunction*: a single `publication_ready` that is false when any part of the claim
is unmeasured, and that names which part.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..studies.graph_leak.evidence import atomic_json
from .graph_common import option_value
from .report_graph import DRAGON_LABEL

__all__ = ["bundle_graph"]

BUNDLE_SCHEMA = "graph-study-bundle-v2"

# Fields every run of one study must agree on, and where each is read from a report.
#
# A bundle exists to state a claim that spans runs. If those runs used different weights,
# a different registry, a different detector or a different scorer, the claim spans
# nothing — it just looks like it does, which is worse. `challenge` and `protocol` are
# deliberately absent: those are what the runs are supposed to differ in.
AGREEMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("model_revisions", "model and tokenizer commits"),
    ("study_design_hash", "study design"),
    ("scorer", "scorer version"),
    ("primary_k", "primary k"),
    ("detector_version", "detector version"),
    ("registry_fingerprint", "concept registry fingerprint"),
)

# `study_design_hash` is checked WITHIN a role, not across all runs. A retain-utility run
# declares `phase: retain_utility`, and the phase is part of the study-design hash — so a
# forget run and the retain run that constrains it are REQUIRED to differ here. Checking
# it globally flagged the archived 50x32 study, whose retain run is exactly what a
# publication-ready bundle needs to contain. Everything else must agree everywhere,
# because the retain run is supposed to measure the SAME SYSTEM.
ROLE_SCOPED_FIELDS: frozenset[str] = frozenset({"study_design_hash"})


def _reports(root: Path, prefix: str) -> list[tuple[Path, dict]]:
    out: list[tuple[Path, dict]] = []
    for run in sorted(root.glob(f"{prefix}*")):
        path = run / "GRAPH_LEAK_REPORT.json"
        if run.is_dir() and path.exists():
            out.append((run, json.loads(path.read_text(encoding="utf-8"))))
    return out


def _contrast_supported(report: dict, contrast_id: str) -> bool | None:
    """Whether one named composition contrast is supported on a PRIMARY surface.

    ``phenomenon_supported`` used to answer for C1 and C2 at once. They are different
    claims — "composition beats a single agent" and "composition beats collaboration about
    a different concept" — and only the second rules out the possibility that more agents
    simply means more text. Collapsing them hid which one a study actually had.
    """
    contrasts = [
        c
        for c in (report.get("composition") or {}).get("contrasts", [])
        if c.get("id") == contrast_id and c.get("surface_role") == "primary"
    ]
    if not contrasts:
        return None
    return all(bool(c.get("supported")) for c in contrasts)


def _primary_curves(report: dict) -> dict:
    """Leak@primary_k per arm on this challenge's primary surfaces only."""
    k = str(report.get("primary_k"))
    curves = report.get("curves", {})
    return {
        surface: {arm: curve.get(k) for arm, curve in curves.get(surface, {}).items()}
        for surface in report.get("primary_surfaces", [])
        if surface in curves
    }


def bundle_graph(
    runs: Path = typer.Option(Path("runs/graph"), "--runs", help="directory holding run dirs"),
    prefix: str = typer.Option(..., "--prefix", help="run-id prefix identifying one study"),
    treatment: str = typer.Option("multi_agent_graphforget", "--treatment"),
    output: Path | None = typer.Option(None, "--output", help="defaults to <runs>/<prefix>-BUNDLE"),
) -> None:
    """Link every run of one study into a single STUDY_BUNDLE.json / .md."""
    treatment = str(option_value(treatment, "multi_agent_graphforget"))
    found = _reports(runs, prefix)
    if not found:
        raise typer.BadParameter(f"no run directories with a report under {runs}/{prefix}*")

    per_report: list[dict] = []
    retain: dict | None = None
    fpr: dict | None = None
    for run, report in found:
        entry = {
            "run": run.name,
            "challenge": report.get("challenge"),
            "protocol": report.get("protocol"),
            "retain_evaluation": bool(report.get("retain_utility_measured")),
            "primary_k": report.get("primary_k"),
            "primary_surfaces": report.get("primary_surfaces", []),
            "leak_at_primary_k": _primary_curves(report),
            "treatment_refusal_rate": (report.get("answer_rates", {}).get(treatment) or {}).get(
                "refusal_rate"
            ),
            "treatment_collaboration_rate": (
                report.get("collaboration", {}).get(treatment) or {}
            ).get("collaboration_rate"),
            "semantic_report_valid": (report.get("gates") or {}).get("semantic_report_valid"),
            "publication_ready": (report.get("gates") or {}).get("publication_ready"),
            "publication_blockers": (report.get("gates") or {}).get("publication_blockers", []),
            "phenomenon_supported": (report.get("composition") or {}).get("phenomenon_supported"),
            "composition_vs_single_supported": _contrast_supported(report, "C1"),
            "composition_vs_control_supported": _contrast_supported(report, "C2"),
            "defence_supported": (report.get("hypotheses") or {}).get("all_supported"),
            **{field: report.get(field) for field, _label in AGREEMENT_FIELDS},
            "detector_recall_on_generated_leakage": (
                report.get("detector_recall_on_generated_leakage") or {}
            ).get("recall"),
        }
        per_report.append(entry)
        # The retain run is the one that measured utility. Prefer the `graph_flow` one:
        # the leakage claim it has to constrain is made under `graph_flow`, and utility
        # measured under `end_to_end_safety` is a cost paid by a different configuration
        # of the same defence. Either is a real measurement; only one is the matched one.
        if report.get("retain_utility_measured") and (report.get("utility_gate") or {}).get(
            "applicable"
        ):
            better = retain is None or (
                retain.get("protocol") != "graph_flow" and report.get("protocol") == "graph_flow"
            )
            if better:
                retain = {
                    "run": run.name,
                    "protocol": report.get("protocol"),
                    **report["utility_gate"],
                }
        if fpr is None and (report.get("detector_fpr_gate") or {}).get("applicable"):
            fpr = {"run": run.name, **report["detector_fpr_gate"]}

    flow = [e for e in per_report if e["protocol"] == "graph_flow" and not e["retain_evaluation"]]
    valid = [e for e in per_report if e["semantic_report_valid"]]

    blockers: list[str] = []
    if len(valid) != len(per_report):
        invalid = [e["run"] for e in per_report if not e["semantic_report_valid"]]
        blockers.append(f"runs whose measurement gates do not pass: {invalid}")
    if retain is None:
        blockers.append(
            "no run in this study measured retain utility on a retain cohort, so the "
            "leakage reductions have no measured utility cost"
        )
    elif retain.get("blocking"):
        blockers.append(f"retain utility gate failed in {retain['run']}: {retain.get('reason')}")
    if fpr is None:
        blockers.append("no run in this study carries a detector calibration artefact")
    elif fpr.get("blocking"):
        blockers.append(f"detector FPR gate failed in {fpr['run']}: {fpr.get('reason')}")
    if not flow:
        blockers.append(
            "no forget-cohort graph_flow run: an end_to_end_safety reduction is a claim "
            "about request filtering, not about the graph"
        )
    # A run-level blocker that the STUDY resolves must not be re-raised here. Retain
    # utility and detector FPR are checked once, above, against the runs that measured
    # them; carrying each forget run's "no retain cohort in this run" forward would make
    # the bundle restate the very gap it exists to close, and no study could ever pass.
    # The same for "measurement gates", which is the per-run roll-up of checks already
    # enumerated in `valid`.
    resolved_at_study_level = (
        "retain utility gate is",
        "detector FPR gate is",
        "measurement gates",
    )
    per_challenge: dict[str, list[str]] = {}
    for entry in flow:
        reasons = [
            reason
            for reason in entry["publication_blockers"]
            if not any(marker in reason for marker in resolved_at_study_level)
        ]
        per_challenge.setdefault(str(entry["challenge"]), []).extend(reasons)
        blockers.extend(f"{entry['run']} ({entry['challenge']}): {reason}" for reason in reasons)

    # `ready_challenges` was the wrong name for this set. A challenge lands here when its
    # run has no OPERATIONAL blocker — the cost gates passed and the reduction was not
    # bought with refusal — which says nothing about whether its defence hypothesis was
    # supported. `natural` in the 50x32 study was "ready" on that reading while its
    # detector never fired on a single leaking string. The set is now named for what it
    # measures, and the defence claim is reported separately below.
    operationally_eligible_challenges = sorted(c for c, r in per_challenge.items() if not r)
    blocked_challenges = sorted(c for c, r in per_challenge.items() if r)

    # Cross-run agreement. Checked before any conjunction is stated, because a bundle over
    # runs that disagree on the weights or the detector is not one study.
    def _role(entry: dict) -> str:
        return "retain" if entry["retain_evaluation"] else "forget"

    disagreements: list[str] = []
    for field, label in AGREEMENT_FIELDS:
        if field in ROLE_SCOPED_FIELDS:
            groups: dict[str, list[dict]] = {}
            for entry in per_report:
                groups.setdefault(_role(entry), []).append(entry)
        else:
            groups = {"": list(per_report)}
        for role, entries in sorted(groups.items()):
            values = {json.dumps(e.get(field), sort_keys=True) for e in entries}
            if len(values) > 1:
                scope = f" among {role} runs" if role else ""
                disagreements.append(
                    f"runs disagree on {label} (`{field}`){scope}: {sorted(values)}. Runs "
                    "bundled into one study must have measured the same system."
                )
    # Exactly one run per (role, challenge, protocol). Two runs of one cell are two
    # measurements of the same number, and a bundle that silently kept whichever sorted
    # last would make the verdict depend on a directory listing. The role is part of the
    # key because a retain run carries the challenge label of the questions it asked.
    cells: dict[tuple[str, str, str], list[str]] = {}
    for entry in per_report:
        key = (_role(entry), str(entry["challenge"]), str(entry["protocol"]))
        cells.setdefault(key, []).append(str(entry["run"]))
    duplicated = {cell: runs_in for cell, runs_in in cells.items() if len(runs_in) > 1}
    for (role, challenge, protocol), runs_in in sorted(duplicated.items()):
        disagreements.append(
            f"{len(runs_in)} {role} runs for challenge '{challenge}' under '{protocol}': "
            f"{sorted(runs_in)}"
        )
    blockers.extend(disagreements)

    unmeasured_recall = [
        e["run"] for e in flow if e["detector_recall_on_generated_leakage"] is None
    ]
    warnings: list[str] = []
    if unmeasured_recall:
        warnings.append(
            "detector recall on generated leakage was never measured for "
            f"{unmeasured_recall}; a defence that did not reduce leakage cannot be "
            "diagnosed as a propagation failure without it"
        )
    blind = [
        e["run"]
        for e in flow
        if isinstance(e["detector_recall_on_generated_leakage"], (int, float))
        and e["detector_recall_on_generated_leakage"] < 0.5
    ]
    if blind:
        warnings.append(
            f"detector recall on generated leakage is below 0.5 in {blind}. Where it is, "
            "a provenance-propagating defence and a node-local one have nothing to "
            "propagate and will score alike no matter which is better: a null result "
            "there is about DETECTION coverage, not about the graph"
        )

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "study_prefix": prefix,
        "treatment": treatment,
        "n_runs": len(per_report),
        "reports": per_report,
        "retain_utility": retain
        or {"applicable": False, "reason": "no retain-cohort run in this study"},
        "detector_fpr": fpr or {"applicable": False, "reason": "no calibration artefact"},
        # Five verdicts, each answering one question, because the previous three could be
        # read as answering each other's (GU-0032):
        #
        #   composition_vs_single_supported   collaboration leaks more than one agent
        #   composition_vs_control_supported  ... and more than collaboration about ANOTHER
        #                                     concept, which is what rules out "more agents
        #                                     means more text"
        #   defence_supported                 the treatment reduces leakage against every
        #                                     baseline the study declared
        #   operationally_eligible            the cost gates passed and the reduction was
        #                                     not bought by refusing to work. NOT a claim
        #                                     that anything was shown
        #   publication_ready                 all of the above, on runs that agree about
        #                                     what they measured
        "composition_vs_single_supported": bool(flow)
        and all(e["composition_vs_single_supported"] for e in flow),
        "composition_vs_control_supported": bool(flow)
        and all(e["composition_vs_control_supported"] for e in flow),
        # Kept as the pre-GU-0032 name for the conjunction of the two above.
        "phenomenon_supported": bool(flow) and all(e["phenomenon_supported"] for e in flow),
        "defence_supported": bool(flow) and all(e["defence_supported"] for e in flow),
        "operationally_eligible": bool(operationally_eligible_challenges) and not disagreements,
        # Per challenge as well as overall: a study whose natural claim is clean and whose
        # injected stress challenges are refusal-confounded is a real, partial result, and
        # a single false would hide which half is which.
        "operationally_eligible_challenges": operationally_eligible_challenges,
        # The old name, unchanged in value. It was never a readiness claim about the
        # science, and callers still reading it get the same set under a name that says so.
        "ready_challenges": operationally_eligible_challenges,
        "blocked_challenges": blocked_challenges,
        "cross_run_agreement": {
            "checked_fields": [field for field, _label in AGREEMENT_FIELDS],
            "agrees": not disagreements,
            "disagreements": disagreements,
            "role_scoped_fields": sorted(ROLE_SCOPED_FIELDS),
            "runs_per_cell": {
                f"{role}/{c}/{p}": sorted(v) for (role, c, p), v in sorted(cells.items())
            },
        },
        "publication_ready": not blockers,
        "publication_blockers": blockers,
        "warnings": warnings,
        "baseline_note": (
            f"multi_agent_dragon is a {DRAGON_LABEL}. Not a reproduction; no number in "
            "this bundle supports a claim about published DRAGON."
        ),
        "pooling_note": (
            "nothing here is pooled or recomputed. Every figure is carried verbatim from "
            "the run report that measured it, with that run's id attached."
        ),
    }

    out_dir = output or (runs / f"{prefix}-BUNDLE")
    atomic_json(out_dir / "STUDY_BUNDLE.json", bundle)
    (out_dir / "STUDY_BUNDLE.md").write_text(_markdown(bundle), encoding="utf-8")
    typer.echo(
        json.dumps({k: bundle[k] for k in ("publication_ready", "publication_blockers")}, indent=2)
    )
    typer.echo(f"wrote {out_dir / 'STUDY_BUNDLE.json'}")


def _cell(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return "—" if value is None else str(value)
    if isinstance(value, (int, float)):
        return format(float(value), ".4f")
    return str(value)


def _markdown(bundle: dict) -> str:
    lines = [
        f"# Study bundle — `{bundle['study_prefix']}`",
        "",
        f"- treatment: `{bundle['treatment']}`  ({bundle['n_runs']} runs linked)",
        f"- **publication_ready: {bundle['publication_ready']}**",
        "- challenges that are OPERATIONALLY ELIGIBLE (cost gates passed, not "
        "refusal-confounded — not a claim that anything was shown): "
        + (", ".join(f"`{c}`" for c in bundle["operationally_eligible_challenges"]) or "none"),
        "- challenges still blocked: "
        + (", ".join(f"`{c}`" for c in bundle["blocked_challenges"]) or "none"),
        f"- composition vs single agent: {bundle['composition_vs_single_supported']}",
        f"- composition vs cross-concept control: " f"{bundle['composition_vs_control_supported']}",
        f"- defence supported (treatment reduces leakage): {bundle['defence_supported']}",
        f"- runs agree on what they measured: {bundle['cross_run_agreement']['agrees']}",
        "",
        f"> {bundle['baseline_note']}",
        "",
        "## Runs",
        "",
        "| run | challenge | protocol | valid | pub-ready | refusal | collaboration | detector recall |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for entry in bundle["reports"]:
        lines.append(
            f"| `{entry['run']}` | {entry['challenge']} | {entry['protocol']} "
            f"| {entry['semantic_report_valid']} | {entry['publication_ready']} "
            f"| {_cell(entry['treatment_refusal_rate'])} "
            f"| {_cell(entry['treatment_collaboration_rate'])} "
            f"| {_cell(entry['detector_recall_on_generated_leakage'])} |"
        )

    retain = bundle["retain_utility"]
    fpr = bundle["detector_fpr"]
    lines += [
        "",
        "## Cost gates (linked from the runs that measured them)",
        "",
        "| gate | run | measured | bound | within |",
        "|---|---|---|---|---|",
        f"| retain utility loss | `{retain.get('run', '—')}` "
        f"| {_cell(retain.get('drop_percentage_points'))} pp "
        f"| {_cell(retain.get('margin_percentage_points'))} pp "
        f"| {_cell(retain.get('within_margin'))} |",
        f"| detector FPR | `{fpr.get('run', '—')}` | {_cell(fpr.get('fpr'))} "
        f"| {_cell(fpr.get('ceiling'))} | {_cell(fpr.get('within_ceiling'))} |",
        "",
    ]
    if not retain.get("applicable", False):
        lines += [f"> retain utility: {retain.get('reason')}", ""]

    if bundle["publication_blockers"]:
        lines += ["## Publication blockers", ""]
        lines += [f"- {reason}" for reason in bundle["publication_blockers"]]
        lines.append("")
    if bundle["warnings"]:
        lines += ["## Warnings", ""]
        lines += [f"- {reason}" for reason in bundle["warnings"]]
        lines.append("")
    lines += [f"> {bundle['pooling_note']}", ""]
    return "\n".join(lines)
