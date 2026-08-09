"""CPU-only command for blinded semantic rescoring of preserved Day-2 handoffs."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from ..eval.openrouter_semantic import run_openrouter_rescore
from ..paths import results_dir


def _report(payload: dict) -> str:
    summary = payload["summary"]
    c3c, c3s = summary["conditions"]["C3C"], summary["conditions"]["C3S"]
    delta = summary["semantic_accuracy_c3c_minus_c3s"]
    audit = summary["judgment_audit"]
    return "\n".join(
        [
            "# POST-HOC DAY-2 SEMANTIC SENSITIVITY ANALYSIS",
            "",
            "> **NOT THE PREREGISTERED RESULT.** This analysis evaluates preserved C3C/C3S responses with blinded external judges. It cannot retroactively create the missing per-item memory certificates.",
            "",
            "## Semantic outcomes",
            "",
            "| metric | C3C | C3S |",
            "|---|---:|---:|",
            f"| semantic accuracy (B correct) | {c3c['semantic_accuracy']:.3f} | {c3s['semantic_accuracy']:.3f} |",
            f"| potential reconstruction (B correct, A not correct) | {c3c['potential_reconstruction']:.3f} | {c3s['potential_reconstruction']:.3f} |",
            f"| peer-context harm (A correct, B wrong) | {c3c['peer_context_harm']:.3f} | {c3s['peer_context_harm']:.3f} |",
            f"| same-false-claim propagation | {c3c['same_false_claim_propagation']:.3f} | {c3s['same_false_claim_propagation']:.3f} |",
            "",
            "## Paired primary sensitivity",
            "",
            f"Semantic accuracy C3C - C3S: {delta['delta']:.3f}; 95% clustered-by-author CI [{delta['ci95'][0]:.3f}, {delta['ci95'][1]:.3f}] ({delta['n_pairs']} pairs, {delta['n_units']} author clusters).",
            f"Potential reconstruction C3C - C3S: {summary['potential_reconstruction_c3c_minus_c3s']['delta']:.3f}; 95% clustered-by-author CI [{summary['potential_reconstruction_c3c_minus_c3s']['ci95'][0]:.3f}, {summary['potential_reconstruction_c3c_minus_c3s']['ci95'][1]:.3f}].",
            "",
            "## Judge audit",
            "",
            f"Scored answers: {audit['scored_answers']}; primary label disagreements: {audit['primary_label_disagreements']}; adjudications: {audit['adjudications']}; direct same-false-claim adjudications: {audit['same_false_claim_adjudications']}.",
            "",
            "Same-false-claim propagation is a direct blinded adjudication of wrong-handoff/wrong-final pairs, not a string-overlap proxy.",
            "The original strict-substring result and all original artifacts remain unchanged.",
            "",
        ]
    )


def rescore_day2(
    c3c: Path = typer.Option(..., "--c3c", exists=True, readable=True),
    c3s: Path = typer.Option(..., "--c3s", exists=True, readable=True),
    judge_1_model: str = typer.Option(..., "--judge-1-model", help="first OpenRouter model"),
    judge_2_model: str = typer.Option(..., "--judge-2-model", help="independent OpenRouter model"),
    adjudicator_model: str = typer.Option(
        ..., "--adjudicator-model", help="OpenRouter adjudicator model"
    ),
    temperature: float = typer.Option(0.0, "--temperature", help="must be zero for this protocol"),
    output: Path | None = typer.Option(None, "--output"),
    cache: Path | None = typer.Option(None, "--cache"),
) -> None:
    """Rescore C3C/C3S with two blinded semantic judges; never overwrite raw evidence."""
    destination = output or (results_dir() / "posthoc_day2_semantic.json")
    if temperature != 0.0:
        raise typer.BadParameter("semantic judges must run at temperature 0")
    if judge_1_model == judge_2_model:
        raise typer.BadParameter("judge 1 and judge 2 must use distinct model identities")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise typer.BadParameter(
            "set OPENROUTER_API_KEY in the environment; never pass it as an argument"
        )
    if destination.exists():
        raise typer.BadParameter(f"refusing to overwrite existing post-hoc analysis: {destination}")
    cache_path = cache or (results_dir() / "posthoc_day2_semantic_judges.jsonl")
    payload = run_openrouter_rescore(
        c3c_path=c3c,
        c3s_path=c3s,
        cache_path=cache_path,
        models=[judge_1_model, judge_2_model, adjudicator_model],
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = destination.with_name("POSTHOC_DAY2_ANALYSIS.md")
    report.write_text(_report(payload), encoding="utf-8")
    typer.echo(f"wrote {destination}")
    typer.echo(f"wrote {report}")
