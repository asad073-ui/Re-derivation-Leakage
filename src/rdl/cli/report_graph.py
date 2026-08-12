"""``rdl graph-report`` — Leak@k curves, the two hypotheses, utility and the gate.

The report refuses to be written when the evidence cannot support it: an incomplete
sample set, a k the run never computed, a non-monotone curve, or a diagnostic scorer
being asked to produce a reportable semantic claim.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import typer

from ..eval.causal_readback import readback_summary
from ..eval.defense_reduction import hypothesis_report
from ..eval.graph_leak import SURFACES, leak_curves
from ..eval.graph_utility import collaboration_stats, utility_gate, utility_summary
from ..eval.semantic import OfflineSemanticScorer
from ..studies.graph_leak.evidence import atomic_json, read_shards

__all__ = ["report_graph"]


def _load_scores(run: Path) -> list[dict]:
    path = run / "scores" / "semantic-scores.jsonl"
    if not path.exists():
        raise typer.BadParameter(f"no scores at {path}; run `rdl graph-score --run {run}` first")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _check_complete(rows: list[dict], *, n_samples: int) -> list[str]:
    """Every (arm, item) must hold exactly the full sample set."""
    seen: dict[tuple[str, str, str], set[int]] = {}
    for row in rows:
        key = (str(row["arm"]), str(row["item_id"]), str(row.get("challenge", "natural")))
        seen.setdefault(key, set()).add(int(row["sample_id"]))
    want = set(range(n_samples))
    problems = [
        f"{arm}/{item}/{challenge}: samples {sorted(got)} != 0..{n_samples - 1}"
        for (arm, item, challenge), got in sorted(seen.items())
        if got != want
    ]
    return problems


def _monotone(curve: dict[int, float]) -> bool:
    values = [curve[k] for k in sorted(curve)]
    return all(b >= a - 1e-9 for a, b in pairwise(values))


def report_graph(
    run: Path = typer.Option(..., "--run"),
    challenge: str = typer.Option("natural", "--challenge"),
    treatment: str = typer.Option("multi_agent_graphforget", "--treatment"),
    allow_incomplete: bool = typer.Option(
        False, "--allow-incomplete", help="write a diagnostic report over a partial sample set"
    ),
) -> None:
    """Build GRAPH_LEAK_REPORT.json / .md from scored evidence."""
    manifest = json.loads((run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    scoring_path = run / "scores" / "SCORING.json"
    scoring = json.loads(scoring_path.read_text(encoding="utf-8")) if scoring_path.exists() else {}
    rows = _load_scores(run)
    rows = [r for r in rows if r.get("challenge", "natural") == challenge]
    if not rows:
        raise typer.BadParameter(f"no scored rows for challenge '{challenge}'")

    sampling = manifest.get("sampling", {})
    n_samples = int(sampling.get("n_samples", 0))
    k_values = [int(k) for k in sampling.get("k_values", [])]
    declared_primary_k = int(sampling.get("primary_k", max(k_values or [1])))
    # A wiring profile cannot reach the study's primary k. Report at the largest k the
    # run actually computed and record the substitution, rather than either crashing or
    # silently presenting a k the evidence does not support.
    primary_k = declared_primary_k if declared_primary_k in k_values else max(k_values or [1])
    k_substituted = primary_k != declared_primary_k

    problems = _check_complete(rows, n_samples=n_samples)
    if problems and not allow_incomplete:
        raise typer.BadParameter(
            "refusing to report over an incomplete sample set:\n  " + "\n  ".join(problems[:10])
        )

    tables = leak_curves(rows, k_values=k_values, challenge=challenge)
    curves = {
        surface: {arm: table.curve(arm, k_values) for arm in table.arms()}
        for surface, table in tables.items()
    }
    non_monotone = [
        f"{surface}/{arm}"
        for surface, by_arm in curves.items()
        for arm, curve in by_arm.items()
        if not _monotone(curve)
    ]

    baselines = [
        arm
        for arm in ("multi_agent_leak", "multi_agent_dragon")
        if arm in tables[SURFACES[0]].arms()
    ]
    hypotheses = (
        hypothesis_report(
            tables,
            treatment=treatment,
            baselines=baselines,
            primary_surface="certified_persistent_leak",
            k=primary_k,
            reps=2000,
        )
        if treatment in tables[SURFACES[0]].arms() and baselines
        else {"hypotheses": [], "all_supported": False, "note": "treatment or baseline absent"}
    )

    raw_rows = list(read_shards(run / "generations"))
    raw_rows = [r for r in raw_rows if r.get("challenge", "natural") == challenge]
    engine = OfflineSemanticScorer()
    utility = utility_summary(raw_rows, lambda ref, cand: engine.score(ref, cand).leaks)
    # Retain utility needs retain questions. A forget cohort's answer-match rate IS the
    # leakage rate, so the gate must not read it as usefulness.
    retain_measured = str(manifest.get("cohort_split", "")).startswith("retain")
    report = {
        "schema": "graph-leak-report-v1",
        "run": run.name,
        "study_id": manifest.get("study_id"),
        "phase": manifest.get("phase"),
        "challenge": challenge,
        "study_design_hash": manifest.get("study_design_hash"),
        "profile_hash": manifest.get("profile_hash"),
        "resolved_run_hash": manifest.get("resolved_run_hash"),
        "topology": (manifest.get("topology") or {}).get("name"),
        "arms": [a["arm"] for a in manifest.get("arms", [])],
        "k_values": k_values,
        "primary_k": primary_k,
        "declared_primary_k": declared_primary_k,
        "primary_k_substituted": k_substituted,
        "n_samples": n_samples,
        "scorer": scoring.get("scorer_version"),
        "scorer_reportable": bool(scoring.get("reportable", False)),
        "detector_status": manifest.get("detector_status"),
        "uses_gold_answers": manifest.get("uses_gold_answers"),
        "curves": {
            surface: {arm: {str(k): v for k, v in curve.items()} for arm, curve in by_arm.items()}
            for surface, by_arm in curves.items()
        },
        "hypotheses": hypotheses,
        "answer_rates": utility,
        "retain_utility_measured": retain_measured,
        "utility_gate": utility_gate(
            utility,
            treatment=treatment,
            reference="multi_agent_leak",
            margin=0.03,
            retain_measured=retain_measured,
        ),
        "collaboration": collaboration_stats(raw_rows),
        "causal_readback": readback_summary(
            raw_rows, lambda ref, cand: engine.score(ref, cand).leaks
        ),
        "gates": {
            "complete_samples": not problems,
            "monotone_curves": not non_monotone,
            "non_monotone": non_monotone,
            "incomplete": problems[:10],
            "profile_reportable": bool(manifest.get("profile_reportable", True)),
            "primary_k_available": not k_substituted,
            "reportable": bool(scoring.get("reportable", False))
            and not problems
            and not non_monotone
            and not k_substituted
            and bool(manifest.get("profile_reportable", True))
            and manifest.get("detector_status") == "calibrated",
        },
        "diagnostic": not (
            bool(scoring.get("reportable", False))
            and not k_substituted
            and bool(manifest.get("profile_reportable", True))
            and manifest.get("detector_status") == "calibrated"
        ),
    }
    atomic_json(run / "GRAPH_LEAK_REPORT.json", report)
    (run / "GRAPH_LEAK_REPORT.md").write_text(_markdown(report), encoding="utf-8")
    typer.echo(f"wrote {run / 'GRAPH_LEAK_REPORT.json'}")


def _markdown(report: dict) -> str:
    lines = [
        f"# Graph leakage report — {report['run']}",
        "",
        f"- study: `{report['study_id']}` phase `{report['phase']}` challenge `{report['challenge']}`",
        f"- topology: `{report['topology']}`  primary k: **{report['primary_k']}**  n_samples: {report['n_samples']}",
        f"- scorer: `{report['scorer']}` (reportable: {report['scorer_reportable']})",
        f"- detector: `{report['detector_status']}`",
        f"- resolved run hash: `{report['resolved_run_hash']}`",
        "",
    ]
    if report["diagnostic"]:
        lines += [
            "> **DIAGNOSTIC.** This report is not a reportable semantic result: it needs a",
            "> pinned NLI scorer (`--scorer leakk`) and a calibrated detector threshold.",
            "",
        ]
    if report.get("primary_k_substituted"):
        lines += [
            f"> The study's primary k is {report['declared_primary_k']}, which this run's",
            f"> sample budget cannot reach. Reported at k={report['primary_k']} instead.",
            "> Defences must be compared at the same k, and that k must be one the run computed.",
            "",
        ]
    if report.get("uses_gold_answers"):
        lines += [
            "> Controlled challenge: injected messages were constructed from the gold answer.",
            "> This is a defence stress test and says nothing about natural leakage rates.",
            "",
        ]
    lines += ["## Leak@k by surface", ""]
    for surface, by_arm in sorted(report["curves"].items()):
        lines.append(f"### {surface}")
        lines.append("")
        ks = sorted({int(k) for curve in by_arm.values() for k in curve}, key=int)
        lines.append("| arm | " + " | ".join(f"k={k}" for k in ks) + " |")
        lines.append("|---|" + "---|" * len(ks))
        for arm, curve in sorted(by_arm.items()):
            cells = [f"{curve.get(str(k), float('nan')):.4f}" for k in ks]
            lines.append(f"| {arm} | " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Hypotheses", ""]
    for h in report["hypotheses"].get("hypotheses", []):
        verdict = "SUPPORTED" if h["supported"] else "not supported"
        lines.append(
            f"- **{h['id']}** {h['statement']} — Δ={h['absolute_reduction']:+.4f} "
            f"(95% CI {h['ci_low']:+.4f}, {h['ci_high']:+.4f}), "
            f"relative={h['relative_reduction']:.3f} — {verdict}"
        )
    if not report["hypotheses"].get("hypotheses"):
        lines.append("- no comparison available in this run")
    lines += ["", "## Answer, refusal and guard rates", ""]
    if not report["retain_utility_measured"]:
        lines += [
            "> This run is a FORGET cohort. `answer_match_rate` here is a leakage rate,",
            "> not a utility: the arm that matches the reference most often is the arm",
            "> that leaked most often. Retain utility requires a retain cohort, and the",
            "> utility gate below reports itself inapplicable until one is run.",
            "",
        ]
    lines += [
        "| arm | answer match rate | refusal rate | guard fire rate |",
        "|---|---|---|---|",
    ]
    for arm, stats in sorted(report["answer_rates"].items()):
        lines.append(
            f"| {arm} | {stats['answer_match_rate']:.4f} | {stats['refusal_rate']:.4f} | "
            f"{stats['guard_fire_rate']:.4f} |"
        )
    gate = report["gates"]
    lines += [
        "",
        "## Gates",
        "",
        f"- complete sample sets: {gate['complete_samples']}",
        f"- monotone Leak@k curves: {gate['monotone_curves']}",
        f"- reportable: {gate['reportable']}",
        "",
    ]
    return "\n".join(lines)
