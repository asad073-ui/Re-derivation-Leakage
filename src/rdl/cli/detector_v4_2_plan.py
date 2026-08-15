"""``rdl graph-detector-v4-2-judge-plan`` — how many calls, how many tokens, how many days.

One judge is billed and one runs on a free plan, so the question that decides whether the
audit can be run at all has two halves: what does it COST, and does it FIT inside a day's
quota. This command answers both offline, from the actual input files and the actual prompt
builder, and never calls an API.

It exists because the alternative is finding out at row 700. A run that discovers its
per-day ceiling two thirds of the way through has not failed cheaply — on Groq's free plan
it has consumed the day for every model — and a run that discovers its rate was double what
the repository recorded has spent money nobody budgeted.

Costs are read at the tier the runner actually uses (`REQUEST_MODE`), which is Standard
synchronous. The Batch/Flex rates are half of Standard and this pipeline does not pay them;
recording one unlabelled number is how v4.2.2 reported Gemini at half price.

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
    JUDGES,
    PAID_FALLBACK_USD_PER_MTOK,
    PASSES,
    PRICE_NOTE,
    PRICING_AS_OF,
    PROVIDERS,
    RATE_LIMITS,
    V4_2_PROTOCOL,
    build_prompt,
    list_price_of,
    price_of,
    required_reference_label,
)
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT
from .detector_v4_2_bundle import load_bundle
from .detector_v4_2_llm_judge import DEFAULT_V4_2_OUT, _read_jsonl

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
    """Days of quota this judge needs, and whether it is on a free plan at all.

    A judge that is NOT on a free plan has no per-day ceiling to wait out: it is billed and
    it runs in one sitting. `free_tier_used` rather than `free_tier_exists` is what this
    keys on — Gemini has a free tier and the protocol refuses it for data-handling reasons,
    and a schedule computed as though it were free would be a schedule for a run nobody is
    allowed to make.
    """
    limits = dict(RATE_LIMITS.get(model, {}))
    # Whether the free plan is USED, not whether one exists. Gemini has a free tier and
    # the protocol refuses it; a schedule computed as though it were free would be a
    # schedule for a run nobody is allowed to make.
    free = bool(limits.get("free_tier_used"))
    rpd = limits.get("requests_per_day") if free else None
    tpd = limits.get("tokens_per_day") if free else None
    rpm = limits.get("requests_per_minute")
    days_by_requests = math.ceil(n_calls / int(rpd)) if rpd else 1
    days_by_tokens = math.ceil(n_tokens / int(tpd)) if tpd else 1
    days = max(days_by_requests, days_by_tokens, 1)
    return {
        "free_tier_exists": bool(limits.get("free_tier_exists")),
        "free_tier_used": free,
        "why_not_free": limits.get("why_not_free"),
        "billing": limits.get("billing"),
        "plan_limits": limits or None,
        "quota_source": limits.get("quota_source"),
        "quota_verification": (
            "plan documentation, NOT verified against this account. The run manifest "
            "records the provider's own x-ratelimit-* headers as observed_rate_limits, "
            "which is the account-verified figure."
        ),
        "days_by_request_ceiling": days_by_requests if rpd else None,
        "days_by_token_ceiling": days_by_tokens if tpd else None,
        "free_tier_days_required": days if free else None,
        "minimum_wall_clock_minutes_at_rpm": round(n_calls / int(rpm), 1) if rpm else None,
        "binding_constraint": (
            "none — this judge is billed per token, so money is the constraint and quota " "is not"
            if not free
            else (
                "tokens_per_day"
                if tpd and days_by_tokens >= days_by_requests
                else "requests_per_day" if rpd else "none published"
            )
        ),
        "fits_in_one_free_day": days <= 1 if free else None,
    }


def detector_v4_2_judge_plan(
    audit_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--audit-dir"),
    audit_manifest: Path | None = typer.Option(
        None,
        "--audit-manifest",
        help="BANK_AUDIT_MANIFEST.json, to cost a BANK audit rather than the v4.1 one",
    ),
    blind_input: Path | None = typer.Option(None, "--blind-input", help="override, per judge"),
    reference_input: Path | None = typer.Option(None, "--reference-input"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
) -> None:
    """Cost the four judge passes: money first, then quota days. Calls nothing."""
    bundle = load_bundle(audit_dir=audit_dir, manifest=audit_manifest, require_reference=False)
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
            path = override or bundle.input_for(judge=role, pass_name=pass_name)
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

        paid = PAID_FALLBACK_USD_PER_MTOK.get(model)
        per_judge[role] = {
            "provider": spec["provider"],
            "family": provider["family"],
            "requested_model": model,
            "billing": RATE_LIMITS.get(model, {}).get("billing", provider["billing"]),
            "api_key_env": provider["api_key_env"],
            "passes": passes,
            "n_calls": judge_calls,
            "estimated_input_tokens": judge_input,
            "estimated_output_tokens": judge_output,
            "estimated_total_tokens": judge_input + judge_output,
            # What this judge actually bills: zero on a free plan, list rate otherwise.
            "estimated_cost_usd": price_of(model, judge_input, judge_output),
            # What it would cost at list rate regardless of plan. For Gemini these two are
            # the same number, which is the point: it has no free tier.
            "list_price_cost_usd": list_price_of(model, judge_input, judge_output),
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

    days = max(
        (j["free_tier"]["free_tier_days_required"] or 1 for j in per_judge.values()),
        default=1,
    )
    billed = round(sum(j["estimated_cost_usd"] or 0.0 for j in per_judge.values()), 4)
    # What it costs to not wait: the billed total plus the paid rate for every judge whose
    # free plan is the thing making the audit take days.
    fallback = round(
        billed + sum(j["paid_fallback_cost_usd"] or 0.0 for j in per_judge.values()), 4
    )
    plan = {
        "schema": "graph-detector-v4-2-judge-plan-v2",
        "protocol": V4_2_PROTOCOL,
        "calls_nothing": True,
        "audit_bundle": bundle.to_dict(),
        "pricing_as_of": PRICING_AS_OF,
        "price_note": PRICE_NOTE,
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
            "estimated_cost_usd": billed,
            "list_price_cost_usd": round(
                sum(j["list_price_cost_usd"] or 0.0 for j in per_judge.values()), 4
            ),
            "paid_fallback_cost_usd": round(
                sum(j["paid_fallback_cost_usd"] or 0.0 for j in per_judge.values()), 4
            ),
            "budget_with_retries_usd": round(billed * 2.0, 2),
            "cost_to_skip_the_free_plan_wait_usd": fallback,
            "budget_note": (
                "twice the estimate. Retries, a re-run after a rubric fix, and the ~15% "
                "the 4-characters-per-token approximation can be out by all land on the "
                "same card."
            ),
        },
        "free_tier_days_required": days,
        "verdict": (
            f"the audit bills approximately ${billed:.2f} "
            f"(budget ${billed * 2:.2f} with retries)"
            + (
                ". It fits inside one day of every free plan involved"
                if days <= 1
                else f". It needs {days} day(s) of free-plan quota on the judges that have "
                f"one, or ${fallback:.2f} to skip the wait — the free plan's per-day token "
                "ceiling is the binding constraint, not the money. Waiting is fine: the "
                "runner checkpoints every row, stops cleanly on a per-day ceiling, and "
                "`--resume` picks up where it stopped."
            )
        ),
    }
    atomic_json(output_dir / PLAN_FILENAME, plan)
    typer.echo(f"wrote {output_dir / PLAN_FILENAME}")
    typer.echo("")
    for role, entry in sorted(per_judge.items()):
        free = entry["free_tier"]
        free_days = free["free_tier_days_required"]
        typer.echo(
            f"  judge {role} {entry['requested_model']:<24} "
            f"{entry['n_calls']:>6} calls  "
            f"{entry['estimated_total_tokens']:>9,} tok  "
            f"bills ${entry['estimated_cost_usd'] or 0.0:>6.2f}  "
            + (
                f"{free_days:>2} free day(s)"
                if free_days
                else (
                    "free tier refused by protocol"
                    if free.get("free_tier_exists")
                    else "no free tier"
                )
            )
        )
    typer.echo("")
    typer.echo(f"verdict: {plan['verdict']}")
