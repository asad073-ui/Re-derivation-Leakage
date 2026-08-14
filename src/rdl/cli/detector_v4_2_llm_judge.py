"""``rdl graph-detector-v4-2-llm-judge`` — run one model judge over one pass.

One judge, one pass, one invocation. Four invocations produce the v4.2 labels, in the
order frozen in ``DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md`` §7. The command deliberately does
*not* loop over judges: running both from one process would make it trivially easy to
re-run judge A after seeing judge B's κ, and the ordering is the only thing keeping the
two annotations independent.

What it sends
-------------
One stateless request per row: no conversation history, no tools, no browsing, no memory,
no retrieval, no repository access. A judge that could read this repository could read
``LABEL_AUDIT_KEY.json``, which carries the NLI label the audit exists to check.

The request is built by :func:`rdl.eval.detector_v4_2.build_prompt`, a pure function, so
``tests/contract/test_detector_v4_2_blinding.py`` can assert that no forbidden field's
value appears in one. That is the blinding: not a convention about which file to open, but
a test over the bytes that leave the process.

Sampling parameters
-------------------
Not ``temperature=0``. ``claude-sonnet-5`` rejects non-default ``temperature``, ``top_p``
and ``top_k`` with a 400, so Judge B runs at its supported defaults with an explicit
thinking configuration and effort, and the run manifest records what was actually sent
rather than what would have been convenient to claim. Judge A runs at a fixed reasoning
effort. Neither run is deterministic and no artifact says it is.

Failure is not a label
----------------------
A row whose response is missing, malformed after its retries, or schema-invalid is written
with a ``null`` label and an ``error``, counted, and **blocks the report**. It never
defaults to ``NONE``. Defaulting would convert an API outage into a label distribution,
and ``NONE`` is the majority class — the failure would look like a result.

Credentials
-----------
``OPENAI_API_KEY`` and ``ANTHROPIC_API_KEY``, from the environment only. They are never
accepted as arguments, never written to a manifest, and never logged. ``--dry-run`` needs
neither: it builds and writes the prompts, calls nothing, and is what ``make cpu-all``
exercises.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import typer

from ..eval.detector_v4_2 import (
    JUDGE_SCHEMA,
    JUDGES,
    PASSES,
    RUN_SCHEMA,
    V4_2_PROTOCOL,
    build_prompt,
    fields_for,
    parse_judgement,
    prompt_sha256,
    required_reference_label,
    response_schema,
    rubric_for,
    sha256_text,
)
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_1_freeze import DEFAULT_V4_1_OUT

__all__ = ["detector_v4_2_llm_judge"]

DEFAULT_V4_2_OUT = Path("data/cohorts/graph_unlearning_v1/detector_v4_2")

INPUT_FILENAME = {
    "blind": "LABEL_AUDIT_JUDGE_{judge}.jsonl",
    "reference": "LABEL_AUDIT_REFERENCE_PASS_{judge}.jsonl",
}
OUTPUT_FILENAME = "V4_2_JUDGE_{judge}_{pass_upper}.jsonl"
RUN_FILENAME = "V4_2_JUDGE_RUN_{judge}_{pass_upper}.json"
PROMPT_PREVIEW_FILENAME = "V4_2_PROMPT_PREVIEW_{judge}_{pass_upper}.jsonl"

MAX_RETRIES = 2

# Recorded, not billed. USD per million tokens, as published at protocol freeze
# (2026-08-14). A model whose list price was not established here is left None rather than
# guessed: a fabricated unit price would make the cost field look like a measurement.
# Token counts are always recorded, so a price can be applied afterwards.
PRICES_USD_PER_MTOK: dict[str, dict[str, float | None]] = {
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-opus-5": {"input": 5.00, "output": 25.00},
    "gpt-5.6-sol": {"input": None, "output": None},
}
PRICE_NOTE = (
    "list prices recorded at protocol freeze (2026-08-14); claude-sonnet-5 is the "
    "introductory rate in effect through 2026-08-31. This is an estimate from token "
    "counts, not a bill, and a null unit price means the rate was not established here."
)

# --model exists, and choosing a cheaper judge is a real decision rather than a free one.
# The audit spends roughly 3,476 calls at ~1.1k input tokens each — the whole Anthropic
# side is single-digit dollars at Sonnet 5 and about half that at Haiku 4.5. What a weaker
# judge risks is not the bill but the gate: kappa >= 0.70 between the two judges is the
# pre-registered condition, and a judge that applies the rubric less consistently fails it
# — or, worse, passes it while both judges share a systematic error the detector then
# learns. The saving is a few dollars; the exposure is the whole run.
CHEAPER_JUDGE_NOTE = (
    "The model is a frozen protocol parameter, recorded here and in the report. A "
    "cheaper judge saves a few dollars on a run whose per-judge cost is single-digit "
    "USD, and risks the kappa >= 0.70 gate that the entire experiment is conditioned on. "
    "If a cheaper judge is used, it must be chosen BEFORE the first call and named in "
    "DECISIONS.md — swapping it after seeing a kappa is a protocol violation."
)


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _write_jsonl(path: Path, rows: Sequence[Mapping]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")
    temp.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


# ------------------------------------------------------------------ provider adapters --


class JudgeCallError(RuntimeError):
    """A transport or provider error. Retried; never silently turned into a label."""


class _Adapter:
    """One stateless classification call. Returns ``(text, returned_model, usage)``."""

    provider = ""

    def parameters(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    def call(self, system: str, user: str, schema: dict) -> tuple[str, str, dict]:
        raise NotImplementedError  # pragma: no cover - overridden


class OpenAIAdapter(_Adapter):
    """``gpt-5.6-sol`` via the official OpenAI SDK, strict structured output.

    ``reasoning_effort`` is sent as a fixed low value and is a *frozen protocol
    parameter*: if the provider rejects it, the run fails and the operator re-freezes it
    in DECISIONS.md. It is not dropped and retried, because a run whose parameters
    silently differ from the recorded ones is a run nobody can reproduce.
    """

    provider = "openai"

    def __init__(self, model: str, *, reasoning_effort: str | None, max_output_tokens: int):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self._client: Any = None

    def parameters(self) -> dict:
        return {
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": self.max_output_tokens,
            "response_format": "json_schema (strict)",
            "temperature": "not set — provider default",
            "stateless": True,
            "tools": [],
        }

    def _ensure(self) -> Any:
        if self._client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise typer.BadParameter(
                    "OPENAI_API_KEY is not set. Judge A is an OpenAI model; export the "
                    "key in the environment. It is never passed as an argument."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - offline gate never reaches here
                raise typer.BadParameter(
                    "the `openai` package is not installed. `pip install -e '.[judges]'`"
                ) from exc
            self._client = OpenAI()
        return self._client

    def call(self, system: str, user: str, schema: dict) -> tuple[str, str, dict]:
        client = self._ensure()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_completion_tokens": self.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "audit_judgement",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise JudgeCallError(f"{type(exc).__name__}: {exc}") from exc
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise JudgeCallError("response truncated at max_completion_tokens")
        text = choice.message.content or ""
        usage = getattr(response, "usage", None)
        return (
            text,
            str(getattr(response, "model", "") or self.model),
            {
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            },
        )


class AnthropicAdapter(_Adapter):
    """``claude-sonnet-5`` via the official Anthropic SDK, strict structured output.

    No ``temperature``, ``top_p`` or ``top_k``: Sonnet 5 rejects non-default values with a
    400, so this run is explicitly not a temperature-zero run and no artifact claims it
    is. Depth is controlled by ``output_config.effort``, which is the supported lever.
    """

    provider = "anthropic"

    def __init__(self, model: str, *, effort: str, thinking: str, max_tokens: int):
        self.model = model
        self.effort = effort
        self.thinking = thinking
        self.max_tokens = max_tokens
        self._client: Any = None

    def parameters(self) -> dict:
        return {
            "thinking": {"type": self.thinking},
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "output_config.format": "json_schema",
            "temperature": (
                "not set — claude-sonnet-5 rejects non-default temperature/top_p/top_k "
                "with a 400, so this run is not temperature-zero"
            ),
            "stateless": True,
            "tools": [],
        }

    def _ensure(self) -> Any:
        if self._client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise typer.BadParameter(
                    "ANTHROPIC_API_KEY is not set. Judge B is an Anthropic model; export "
                    "the key in the environment. It is never passed as an argument."
                )
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise typer.BadParameter(
                    "the `anthropic` package is not installed. `pip install -e '.[judges]'`"
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def call(self, system: str, user: str, schema: dict) -> tuple[str, str, dict]:
        client = self._ensure()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                thinking={"type": self.thinking},
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema},
                },
            )
        except Exception as exc:
            raise JudgeCallError(f"{type(exc).__name__}: {exc}") from exc
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            raise JudgeCallError("stop_reason=refusal; the request was declined")
        if stop == "max_tokens":
            raise JudgeCallError("response truncated at max_tokens")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), "")
        usage = getattr(response, "usage", None)
        return (
            text,
            str(getattr(response, "model", "") or self.model),
            {
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            },
        )


def build_adapter(judge: str, *, model: str, effort: str, max_output_tokens: int) -> _Adapter:
    if JUDGES[judge]["provider"] == "openai":
        return OpenAIAdapter(model, reasoning_effort=effort, max_output_tokens=max_output_tokens)
    return AnthropicAdapter(model, effort=effort, thinking="adaptive", max_tokens=max_output_tokens)


# ------------------------------------------------------------------------ the runner --


def judge_row(
    adapter: _Adapter,
    row: Mapping,
    *,
    pass_name: str,
    schema: dict,
    max_retries: int = MAX_RETRIES,
) -> dict:
    """One row -> one overlay record. Never raises; failures become recorded failures."""
    system, user = build_prompt(row, pass_name=pass_name)
    attempts: list[str] = []
    for attempt in range(max_retries + 1):
        try:
            text, returned_model, usage = adapter.call(system, user, schema)
            labels = parse_judgement(text, pass_name=pass_name)
        except (JudgeCallError, ValueError) as exc:
            attempts.append(f"attempt {attempt + 1}: {exc}")
            continue
        return {
            **labels,
            "returned_model": returned_model,
            "n_retries": attempt,
            "usage": usage,
            "error": None,
            "source": "model",
        }
    return {
        **dict.fromkeys(fields_for(pass_name)),
        "returned_model": None,
        "n_retries": max_retries,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        # Truncated so one pathological provider message cannot dominate the overlay.
        "error": " | ".join(a[:400] for a in attempts),
        "source": "failed",
    }


def _cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated USD, or ``None`` when this model's list price was not frozen here.

    ``None`` rather than 0.0, and rather than a guessed rate: a zero would read as a free
    run and a guess would read as a measurement. The token counts are recorded either way,
    so a rate can be applied to them later.
    """
    prices = PRICES_USD_PER_MTOK.get(model, {})
    per_input, per_output = prices.get("input"), prices.get("output")
    if per_input is None or per_output is None:
        return None
    return round(input_tokens / 1e6 * per_input + output_tokens / 1e6 * per_output, 4)


def detector_v4_2_llm_judge(
    judge: str = typer.Option(..., "--judge", help="A (OpenAI) or B (Anthropic)"),
    judge_pass: str = typer.Option("blind", "--pass", help=f"one of {PASSES}"),
    audit_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--audit-dir"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
    input_file: Path | None = typer.Option(None, "--input", help="override the pass file"),
    model: str = typer.Option(
        "",
        "--model",
        help=(
            "override the judge's model (e.g. claude-haiku-4-5). Must be chosen before "
            "the first call and named in DECISIONS.md; it is recorded in the manifest."
        ),
    ),
    effort: str = typer.Option("low", "--effort", help="frozen; recorded in the manifest"),
    max_output_tokens: int = typer.Option(4096, "--max-output-tokens"),
    limit: int | None = typer.Option(
        None, "--limit", help="smoke run over the first N rows. Marks the run incomplete."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="build and write the prompts, call nothing. Needs no API key.",
    ),
) -> None:
    """Run one judge over one pass. Writes an overlay keyed by ``audit_id``."""
    if judge not in JUDGES:
        raise typer.BadParameter(f"--judge must be one of {sorted(JUDGES)}, got {judge!r}")
    if judge_pass not in PASSES:
        raise typer.BadParameter(f"--pass must be one of {PASSES}, got {judge_pass!r}")

    spec = dict(JUDGES[judge])
    if model:
        spec["requested_model"] = model
    source = input_file or (audit_dir / INPUT_FILENAME[judge_pass].format(judge=judge))
    if not source.exists():
        raise typer.BadParameter(f"{source} is absent; run `rdl graph-detector-v4-label-audit`")

    rows = _read_jsonl(source)
    n_available = len(rows)
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    pass_upper = judge_pass.upper()
    fields = fields_for(judge_pass)
    schema = response_schema(judge_pass)
    rubric_sha = sha256_text(rubric_for(judge_pass))
    input_sha = _file_sha256(source)

    # ------------------------------------------------------------------- dry run --
    # Deliberately the same prompt-building path, so what cpu-all inspects is what the
    # network run sends. A preview built by a second code path would preview nothing.
    if dry_run:
        preview = []
        for r in rows:
            system, user = build_prompt(r, pass_name=judge_pass)
            preview.append(
                {
                    "audit_id": str(r.get("audit_id", "")),
                    "system": system,
                    "user": user,
                    "prompt_sha256": prompt_sha256(system, user),
                }
            )
        path = output_dir / PROMPT_PREVIEW_FILENAME.format(judge=judge, pass_upper=pass_upper)
        _write_jsonl(path, preview)
        typer.echo(f"wrote {path}  ({len(preview)} prompts, no API call made)")
        raise typer.Exit(0)

    adapter = build_adapter(
        judge,
        model=spec["requested_model"],
        effort=effort,
        max_output_tokens=max_output_tokens,
    )
    started = time.time()
    out: list[dict] = []
    n_forced = 0
    n_malformed = 0
    totals = {"input_tokens": 0, "output_tokens": 0}
    returned_models: set[str] = set()

    for i, row in enumerate(rows, start=1):
        audit_id = str(row.get("audit_id", ""))
        if not audit_id:
            raise typer.BadParameter(f"{source}: row {i} has no audit_id")

        # §4. An absent reference answer is a question the judge was not asked. Forcing
        # UNCERTAIN here is both the protocol rule and 300 calls per judge not spent —
        # and the forced rows are flagged so the report can keep them out of the
        # reference-pass agreement, where 300 trivially-agreeing rows would inflate kappa.
        forced = (
            required_reference_label(row.get("reference_answer"))
            if judge_pass == "reference"
            else None
        )
        if forced is not None:
            n_forced += 1
            record = {
                **dict.fromkeys(fields),
                "reference_content": forced,
                "returned_model": None,
                "n_retries": 0,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "error": None,
                "source": "protocol_rule_empty_reference",
            }
        else:
            record = judge_row(adapter, row, pass_name=judge_pass, schema=schema)
            totals["input_tokens"] += int(record["usage"]["input_tokens"])
            totals["output_tokens"] += int(record["usage"]["output_tokens"])
            if record["returned_model"]:
                returned_models.add(str(record["returned_model"]))
            if record["source"] == "failed":
                n_malformed += 1
                typer.echo(f"  [FAIL] {audit_id}: {record['error']}", err=True)

        system, user = build_prompt(row, pass_name=judge_pass)
        out.append(
            {
                "schema": JUDGE_SCHEMA,
                "audit_id": audit_id,
                "judge": judge,
                "pass": judge_pass,
                "provider": spec["provider"],
                "requested_model": spec["requested_model"],
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "prompt_sha256": prompt_sha256(system, user),
                "rubric_sha256": rubric_sha,
                "input_file_sha256": input_sha,
                **record,
            }
        )
        if i % 50 == 0:
            typer.echo(f"  {i}/{len(rows)} rows ({n_malformed} failed)")

    out_path = output_dir / OUTPUT_FILENAME.format(judge=judge, pass_upper=pass_upper)
    _write_jsonl(out_path, out)

    complete = limit is None and len(rows) == n_available and n_malformed == 0
    manifest = {
        "schema": RUN_SCHEMA,
        "protocol": V4_2_PROTOCOL,
        "judge": judge,
        "role": f"Model Judge {judge}",
        "judge_population": "two_independent_llm_judges",
        "human_grounded": False,
        "pass": judge_pass,
        "fields": list(fields),
        "provider": spec["provider"],
        "requested_model": spec["requested_model"],
        "protocol_default_model": JUDGES[judge]["requested_model"],
        "model_overridden": bool(model),
        "model_choice_note": CHEAPER_JUDGE_NOTE,
        "returned_model": sorted(returned_models)[0] if len(returned_models) == 1 else None,
        "returned_models_seen": sorted(returned_models),
        "parameters": adapter.parameters(),
        "max_retries": MAX_RETRIES,
        "input_file": str(source),
        "input_file_sha256": input_sha,
        "rubric_sha256": rubric_sha,
        "response_schema_sha256": sha256_text(json.dumps(schema, sort_keys=True)),
        "output_file": str(out_path),
        "n_rows_in_input": n_available,
        "n_rows_processed": len(rows),
        "n_rows_called": len(rows) - n_forced,
        "n_forced_by_protocol_rule": n_forced,
        "forced_rule": (
            "an empty reference_answer is labelled UNCERTAIN without a call, and is "
            "excluded from the reference-pass agreement: 300 trivially agreeing rows "
            "would inflate a kappa neither judge earned."
        ),
        "n_malformed_or_missing": n_malformed,
        "complete": complete,
        "why_incomplete": (
            None
            if complete
            else "a --limit smoke run, or rows that failed after their retries. A run "
            "that is not complete cannot back a report; n_malformed_or_missing == 0 is "
            "a pre-registered gate condition."
        ),
        "token_usage": totals,
        "estimated_cost_usd": _cost(
            spec["requested_model"], totals["input_tokens"], totals["output_tokens"]
        ),
        "price_note": PRICE_NOTE,
        "wall_clock_seconds": round(time.time() - started, 1),
        "isolation": {
            "stateless_per_row": True,
            "conversation_history": False,
            "tools_enabled": False,
            "web_access": False,
            "repository_access": False,
            "other_judge_output_visible": False,
        },
        "credentials": "from OPENAI_API_KEY / ANTHROPIC_API_KEY; never recorded here",
        "scope": {
            "model_trained": False,
            "graph_generation_run": False,
            "frozen_v1_v2_v3_v4_v4_1_artifacts_modified": False,
            "gpu_used": False,
            "final_gate_bank_opened": False,
        },
    }
    atomic_json(output_dir / RUN_FILENAME.format(judge=judge, pass_upper=pass_upper), manifest)
    typer.echo(f"wrote {out_path}  ({len(out)} rows)")
    typer.echo(
        f"judge {judge} / {judge_pass}: {len(rows) - n_forced} calls, "
        f"{n_forced} forced, {n_malformed} failed, complete={complete}"
    )
    raise typer.Exit(0 if n_malformed == 0 else 1)
