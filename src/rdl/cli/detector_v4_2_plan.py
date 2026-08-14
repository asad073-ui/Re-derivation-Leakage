"""``rdl graph-detector-v4-2-judge-plan`` — how many calls, how many tokens, how many days.

Both v4.2 judges run on free tiers with published per-minute and per-day ceilings. The
question that decides whether the audit can be run at all is arithmetic: how many calls
does this audit need, how many tokens is that, and does it fit inside a day's quota. This
command answers it offline, from the actual input files and the actual prompt builder, and
never calls an API.

It exists because the alternative is finding out at row 700. A run that discovers its
per-day ceiling two thirds of the way through has not failed cheaply — on Gemini's free
tier it has consumed the day, and on Groq's it has consumed the day for every model.

Token counts are estimated from the real prompts
------------------------------------------------
The prompts are built by :func:`rdl.eval.detector_v4_2.build_prompt`, the same pure
function the runner uses, and measured with the provider-agnostic ~4-characters-per-token
approximation. That is an estimate and is labelled one. It is accurate enough for the
question being asked — "does 1.8M tokens fit in a 200k/day ceiling" does not turn on ±15%
— and a measured count from a live run replaces it in the run manifest afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_2 import (
    FREE_TIER_LIMITS,
    JUDGES,
    PASSES,
    PROVIDERS,
    V4_2_PROTOCOL,
    build_prompt,
    required_reference_label,
)
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT
from .detector_v4_2_llm_judge import (
    DEFAULT_V4_2_OUT,
    INPUT_FILENAME,
    PAID_FALLBACK_USD_PER_MTOK,
    PRICES_USD_PER_MTOK,
    _read_jsonl,
)

__all__ = ["detector_v4_2_judge_plan", "estimate_tokens"]

PLAN_FILENAME = "DETECTOR_V4_2_JUDGE_PLAN.json"

# Measured on both judges against the real rubric at protocol freeze: gpt-oss-120b returned
# 130 completion tokens of which 94 were reasoning; gemini-3.7-flash returned 42. 160 is the
# planning figure, deliberately above both, because a plan that underestimates output is a
# plan that runs out of quota early.
ASSUMED_OUTPUT_TOKENS = 160
CHARS_PER_TOKEN = 4.0


def estimate_tokens(text: str) -> int:
    """~4 characters per token. An estimate, labelled one everywhere it is reported."""
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def _pass_load(rows: Sequence[Mapping], pass_name: str) -> dict:
    """Calls and input tokens for one pass over one input file."""
    n_calls = 0
    input_tokens = 0
    n_forced = 0
    for row in rows:
        if pass_name == "reference" and required_reference_label(row.get("reference_answer")):
            n_forced += 1
            continue
        system, user = build_prompt(row, pass_name=pass_name)
        input_tokens += estimate_tokens(system) + estimate_tokens(user)
        n_calls += 1
    return {
        "n_rows": len(rows),
        "n_calls": n_calls,
        "n_forced_no_call": n_forced,
        "estimated_input_tokens": input_tokens,
        "estimated_output_tokens": n_calls * ASSUMED_OUTPUT_TOKENS,
    }


def _feasibility(model: str, n_calls: int, n_tokens: int) -> dict:
    limits = FREE_TIER_LIMITS.get(model, {})
    rpd = limits.get("requests_per_day")
    tpd = limits.get("tokens_per_day")
    rpm = limits.get("requests_per_minute")
    days_by_requests = math.ceil(n_calls / rpd) if rpd else 1
    days_by_tokens = math.ceil(n_tokens / tpd) if tpd else 1
    days = max(days_by_requests, days_by_tokens, 1)
    return {
        "published_limits": limits or None,
        "days_by_request_ceiling": days_by_requests if rpd else None,
        "days_by_token_ceiling": days_by_tokens if tpd else None,
        "free_tier_days_required": days,
        "minimum_wall_clock_minutes_at_rpm": round(n_calls / rpm, 1) if rpm else None,
        "binding_constraint": (
            "tokens_per_day"
            if tpd and days_by_tokens >= days_by_requests
            else "requests_per_day" if rpd else "none published"
        ),
        "fits_in_one_free_day": days <= 1,
    }


def detector_v4_2_judge_plan(
    audit_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--audit-dir"),
    blind_input: Path | None = typer.Option(None, "--blind-input", help="override, per judge"),
    reference_input: Path | None = typer.Option(None, "--reference-input"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
) -> None:
    """Cost the four judge passes against the published free-tier ceilings. Calls nothing."""
    per_judge: dict[str, dict] = {}
    totals = {"n_calls": 0, "input_tokens": 0, "output_tokens": 0}

    for role, spec in JUDGES.items():
        model = spec["requested_model"]
        provider = PROVIDERS[spec["provider"]]
        passes: dict[str, dict] = {}
        judge_calls = 0
        judge_input = 0
        judge_output = 0
        for pass_name in PASSES:
            override = blind_input if pass_name == "blind" else reference_input
            path = override or (audit_dir / INPUT_FILENAME[pass_name].format(judge=role))
            if not path.exists():
                passes[pass_name] = {"input_file": str(path), "present": False}
                continue
            load = _pass_load(_read_jsonl(path), pass_name)
            load["input_file"] = str(path)
            load["present"] = True
            passes[pass_name] = load
            judge_calls += load["n_calls"]
            judge_input += load["estimated_input_tokens"]
            judge_output += load["estimated_output_tokens"]

        prices = PRICES_USD_PER_MTOK.get(model, {})
        paid = PAID_FALLBACK_USD_PER_MTOK.get(model)
        per_judge[role] = {
            "provider": spec["provider"],
            "family": provider["family"],
            "requested_model": model,
            "billing": provider["billing"],
            "api_key_env": provider["api_key_env"],
            "passes": passes,
            "n_calls": judge_calls,
            "estimated_input_tokens": judge_input,
            "estimated_output_tokens": judge_output,
            "estimated_total_tokens": judge_input + judge_output,
            "estimated_cost_usd": (
                round(
                    judge_input / 1e6 * (prices.get("input") or 0.0)
                    + judge_output / 1e6 * (prices.get("output") or 0.0),
                    4,
                )
                if prices.get("input") is not None
                else None
            ),
            "paid_fallback_cost_usd": (
                round(judge_input / 1e6 * paid["input"] + judge_output / 1e6 * paid["output"], 4)
                if paid
                else None
            ),
            "free_tier": _feasibility(model, judge_calls, judge_input + judge_output),
        }
        totals["n_calls"] += judge_calls
        totals["input_tokens"] += judge_input
        totals["output_tokens"] += judge_output

    days = max((j["free_tier"]["free_tier_days_required"] for j in per_judge.values()), default=1)
    plan = {
        "schema": "graph-detector-v4-2-judge-plan-v1",
        "protocol": V4_2_PROTOCOL,
        "calls_nothing": True,
        "token_estimate_method": (
            f"~{CHARS_PER_TOKEN} characters per token over the real prompts, plus "
            f"{ASSUMED_OUTPUT_TOKENS} output tokens per call (measured envelope for both "
            "judges at freeze was 42 and 130, the latter including 94 reasoning tokens). "
            "An estimate, not a measurement; the run manifest records the real counts."
        ),
        "judges": per_judge,
        "totals": {
            **totals,
            "estimated_total_tokens": totals["input_tokens"] + totals["output_tokens"],
            "estimated_cost_usd": sum(j["estimated_cost_usd"] or 0.0 for j in per_judge.values()),
            "paid_fallback_cost_usd": sum(
                j["paid_fallback_cost_usd"] or 0.0 for j in per_judge.values()
            ),
        },
        "free_tier_days_required": days,
        "verdict": (
            "the whole audit fits inside one day of both free tiers"
            if days <= 1
            else f"the audit needs {days} day(s) of free-tier quota. Run it with --resume: "
            "the runner checkpoints every row, stops cleanly on a per-day ceiling, and "
            "picks up where it stopped. The paid fallback rate is recorded beside each "
            "judge if waiting is not worth it."
        ),
    }
    atomic_json(output_dir / PLAN_FILENAME, plan)
    typer.echo(f"wrote {output_dir / PLAN_FILENAME}")
    typer.echo("")
    for role, entry in sorted(per_judge.items()):
        typer.echo(
            f"  judge {role} {entry['requested_model']:<24} "
            f"{entry['n_calls']:>6} calls  "
            f"{entry['estimated_total_tokens']:>9,} tok  "
            f"{entry['free_tier']['free_tier_days_required']:>2} free day(s)  "
            f"paid fallback ${entry['paid_fallback_cost_usd'] or 0.0:.2f}"
        )
    typer.echo("")
    typer.echo(f"verdict: {plan['verdict']}")
