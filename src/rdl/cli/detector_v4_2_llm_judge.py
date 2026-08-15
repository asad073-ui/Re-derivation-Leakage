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

Every row is written when it returns
------------------------------------
The first version of this runner accumulated 1,019 responses in a list and wrote the file
at the end. A crash at row 900 lost 900 responses and, on a metered free tier, 900 of that
day's quota. Now each successful row is appended to ``…​.partial.jsonl`` as it lands and a
progress manifest is written atomically after each batch, so ``--resume`` re-reads what is
already there and asks only for what is missing. A resumed row whose stored prompt hash
disagrees with the prompt this invocation would build is a conflict and is refused, not
overwritten: silently re-judging a row under a changed rubric is how two passes end up
measuring two different questions.

Rate limits are the schedule, not an error
------------------------------------------
Both judges run on free tiers with published per-minute and per-day ceilings. The runner
paces itself against them, retries 429 and 5xx with exponential backoff, and stops
cleanly with the partial file intact when a per-day ceiling is hit — which is a reason to
run ``--resume`` tomorrow, not a reason to have lost today's work. Gemini's free tier in
particular returns 503 UNAVAILABLE under load often enough that a runner without backoff
mostly records failures.

Smoke runs cannot touch a real run
----------------------------------
``--limit`` requires ``--run-id``, which changes the output directory. A five-row smoke
that wrote ``V4_2_JUDGE_A_BLIND.jsonl`` into the same directory as the real pass could be
mistaken for it, or could overwrite it. Limited runs are marked ``reportable: false`` and
the report refuses them.

Failure is not a label
----------------------
A row whose response is missing, malformed after its retries, or schema-invalid is written
with a ``null`` label and an ``error``, counted, and **blocks the report**. It never
defaults to ``NONE``. Defaulting would convert an API outage into a label distribution,
and ``NONE`` is the majority class.

Credentials
-----------
``GEMINI_API_KEY`` and ``GROQ_API_KEY``, from the environment only (see ``.env.example``).
They are never accepted as arguments, never written to a manifest, and never logged.
``--dry-run`` needs neither: it builds and writes the prompts, calls nothing, and is what
``make cpu-all`` exercises.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import typer

from ..eval.detector_v4_2 import (
    JUDGE_SCHEMA,
    JUDGES,
    PAID_FALLBACK_USD_PER_MTOK,
    PASSES,
    PRICE_NOTE,
    PROMPT_VERSION,
    PROVIDERS,
    RATE_LIMITS,
    RUN_SCHEMA,
    V4_2_PROTOCOL,
    build_prompt,
    fields_for,
    judge_families_are_independent,
    parse_judgement,
    price_of,
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
PARTIAL_FILENAME = "V4_2_JUDGE_{judge}_{pass_upper}.partial.jsonl"
PROGRESS_FILENAME = "V4_2_JUDGE_PROGRESS_{judge}_{pass_upper}.json"
RUN_FILENAME = "V4_2_JUDGE_RUN_{judge}_{pass_upper}.json"
PROMPT_PREVIEW_FILENAME = "V4_2_PROMPT_PREVIEW_{judge}_{pass_upper}.jsonl"

MAX_RETRIES = 5
RETRY_BASE_SECONDS = 2.0
RETRY_MAX_SECONDS = 60.0
# The provider says "not now", not "not ever". Retried with backoff rather than recorded as
# a failed row: a 429 turned into a null label is a quota limit turned into a result.
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
# A per-day ceiling ends the run, it does not fail it. The partial file is already on disk.
DAILY_QUOTA_MARKERS = ("quota", "per day", "rpd", "tpd", "daily limit", "exceeded your current")

# The response is four enum values. 4,096 was room for a small essay, and on a reasoning
# model it is room for a lot of reasoning tokens that are billed and thrown away; the
# measured envelope for both judges is under 200 completion tokens including gpt-oss's
# reasoning trace. Truncation is a recorded failure, so an under-budget cap is loud.
DEFAULT_MAX_OUTPUT_TOKENS = 1024
MIN_MAX_OUTPUT_TOKENS = 256
MAX_MAX_OUTPUT_TOKENS = 2048

# Concurrency is bounded by the smaller of this and the provider's requests-per-minute
# pacing. Unbounded fan-out over a free tier is a way to spend a day's quota on 429s.
DEFAULT_CONCURRENCY = 4
MAX_CONCURRENCY = 16

CHEAPER_JUDGE_NOTE = (
    "The model is a frozen protocol parameter, recorded here and in the report. Both "
    "judges are free-tier; what a weaker judge risks is not the bill but the gate. "
    "kappa >= 0.70 between the two judges is the pre-registered condition the entire "
    "experiment is conditioned on, and a judge that applies the rubric less consistently "
    "fails it — or, worse, passes it while both judges share a systematic error the "
    "detector then learns. If a different judge is used, it must be chosen BEFORE the "
    "first call and named in DECISIONS.md; swapping it after seeing a kappa is a protocol "
    "violation. Two judges from one model family are not two judges."
)

USER_AGENT = "rdl-detector-v4-2-judge/1.0"


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


def _rows_sha256(rows: Sequence[Mapping]) -> str:
    """Content hash of an overlay, independent of row order.

    Order-independent on purpose: with bounded concurrency the completion order is not the
    input order, and a hash that changed when two rows landed in a different sequence would
    flag every concurrent run as corrupt.
    """
    lines = sorted(json.dumps(r, sort_keys=True, ensure_ascii=False) for r in rows)
    return sha256_text("\n".join(lines))


# ------------------------------------------------------------------ provider adapters --


class JudgeCallError(RuntimeError):
    """A transport or provider error. Retried; never silently turned into a label."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


class DailyQuotaExhausted(RuntimeError):
    """The provider's per-day ceiling. Ends the run; the partial file stays on disk."""


class _Adapter:
    """One stateless classification call. Returns a record of what was sent and returned."""

    provider = ""

    def parameters(self) -> dict:  # pragma: no cover - overridden
        raise NotImplementedError

    def call(self, system: str, user: str, schema: dict) -> dict:
        raise NotImplementedError  # pragma: no cover - overridden

    def sdk_version(self) -> str:  # pragma: no cover - overridden
        return "unknown"


class OpenAIWireAdapter(_Adapter):
    """Any provider speaking OpenAI chat-completions: Google, Groq, OpenAI itself.

    One adapter rather than three because the differences between these providers are
    entirely data — base URL, key variable, which field caps output, whether
    ``temperature`` and ``reasoning_effort`` are accepted — and that data lives in
    :data:`rdl.eval.detector_v4_2.PROVIDERS`. Three near-identical classes would be three
    places for a parameter to drift out of the manifest.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        reasoning_effort: str | None,
        max_output_tokens: int,
        temperature: float | None,
        timeout: float,
    ):
        self.provider = provider
        self.spec = PROVIDERS[provider]
        self.model = model
        self.reasoning_effort = reasoning_effort if self.spec["supports_reasoning_effort"] else None
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature if self.spec["supports_temperature"] else None
        self.timeout = timeout
        self._client: Any = None

    def parameters(self) -> dict:
        return {
            "provider": self.provider,
            "family": self.spec["family"],
            "wire": self.spec["wire"],
            "base_url": self.spec["base_url"],
            "reasoning_effort": self.reasoning_effort,
            self.spec["max_tokens_field"]: self.max_output_tokens,
            "temperature": self.temperature,
            "response_format": "json_schema (strict)",
            "timeout_seconds": self.timeout,
            "stateless": True,
            "tools": [],
        }

    def sdk_version(self) -> str:
        try:
            import openai

            return f"openai=={openai.__version__}"
        except Exception:  # pragma: no cover - reporting, not control flow
            return "openai==unknown"

    def _ensure(self) -> Any:
        if self._client is None:
            env = self.spec["api_key_env"]
            key = os.environ.get(env)
            if not key:
                raise typer.BadParameter(
                    f"{env} is not set. Judge {self.model!r} is a {self.provider} model; "
                    f"export the key in the environment (see .env.example). It is never "
                    "passed as an argument."
                )
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - offline gate never reaches here
                raise typer.BadParameter(
                    "the `openai` package is not installed. `pip install -e '.[judges]'`. "
                    "It is the client for every OpenAI-wire provider here, including "
                    "Gemini's compatibility endpoint and Groq."
                ) from exc
            headers = {"User-Agent": USER_AGENT} if self.spec.get("requires_user_agent") else None
            self._client = OpenAI(
                api_key=key,
                base_url=self.spec["base_url"],
                timeout=self.timeout,
                max_retries=0,  # retried here, so every attempt is counted and recorded
                default_headers=headers,
            )
        return self._client

    def call(self, system: str, user: str, schema: dict) -> dict:
        client = self._ensure()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            self.spec["max_tokens_field"]: self.max_output_tokens,
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
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        try:
            response = client.chat.completions.with_raw_response.create(**kwargs)
            request_id = response.headers.get("x-request-id") or response.headers.get(
                "x-groq-request-id"
            )
            # The account's actual ceilings, as the provider reports them. A plan's
            # documented limits are a claim about the plan; these are a measurement of
            # this key, and they are what the manifest calls account-verified.
            rate_limit_headers = {
                name: value
                for name, value in response.headers.items()
                if name.lower().startswith("x-ratelimit")
            }
            parsed = response.parse()
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            message = f"{type(exc).__name__}: {exc}"
            if status is None or int(status) in RETRYABLE_STATUS:
                if _looks_like_daily_quota(message):
                    raise DailyQuotaExhausted(message) from exc
                raise JudgeCallError(message, status=status, retryable=True) from exc
            raise JudgeCallError(message, status=status, retryable=False) from exc

        choice = parsed.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            raise JudgeCallError(
                f"response truncated at {self.spec['max_tokens_field']}={self.max_output_tokens}",
                retryable=False,
            )
        text = choice.message.content or ""
        usage = getattr(parsed, "usage", None)
        return {
            "text": text,
            "returned_model": str(getattr(parsed, "model", "") or self.model),
            "provider_request_id": str(request_id or getattr(parsed, "id", "") or ""),
            "rate_limit_headers": rate_limit_headers,
            "usage": {
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            },
        }


def _looks_like_daily_quota(message: str) -> bool:
    lowered = message.lower()
    return "429" in lowered and any(marker in lowered for marker in DAILY_QUOTA_MARKERS)


def build_adapter(
    judge: str,
    *,
    model: str,
    effort: str,
    max_output_tokens: int,
    temperature: float | None,
    timeout: float,
) -> _Adapter:
    provider = JUDGES[judge]["provider"]
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise typer.BadParameter(f"judge {judge} names unknown provider {provider!r}")
    if spec["wire"] != "openai-chat-completions":
        raise typer.BadParameter(
            f"provider {provider!r} speaks {spec['wire']!r}, for which this runner has no "
            "adapter. Add one deliberately and record it in DECISIONS.md."
        )
    return OpenAIWireAdapter(
        provider,
        model,
        reasoning_effort=effort,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        timeout=timeout,
    )


# ------------------------------------------------------------------------- pacing --


class RateLimiter:
    """A shared token-bucket over wall clock, so N workers together stay under one RPM.

    Per-worker sleeping does not bound a fleet: four workers each pausing 2s issue 120
    requests a minute, not 30. One lock and one next-slot clock does bound it.
    """

    def __init__(self, requests_per_minute: int | None):
        self.interval = 60.0 / requests_per_minute if requests_per_minute else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait:
            time.sleep(wait)


# ------------------------------------------------------------------------ the runner --


def judge_row(
    adapter: _Adapter,
    row: Mapping,
    *,
    pass_name: str,
    schema: dict,
    max_retries: int = MAX_RETRIES,
    limiter: RateLimiter | None = None,
    sleep=time.sleep,
) -> dict:
    """One row -> one overlay record. Never raises except on a per-day ceiling.

    ``DailyQuotaExhausted`` is the one exception that propagates: it is not a property of
    this row, and recording it as this row's failure would mark a perfectly good row
    permanently failed while the real cause was the calendar.
    """
    system, user = build_prompt(row, pass_name=pass_name)
    attempts: list[str] = []
    for attempt in range(max_retries + 1):
        if limiter is not None:
            limiter.acquire()
        try:
            result = adapter.call(system, user, schema)
            labels = parse_judgement(result["text"], pass_name=pass_name)
        except DailyQuotaExhausted:
            raise
        except JudgeCallError as exc:
            attempts.append(f"attempt {attempt + 1}: {exc}")
            if not exc.retryable or attempt >= max_retries:
                break
            # Full jitter. A fixed backoff synchronises every worker onto the same retry
            # instant, which is how a fleet turns one 429 into a thundering herd.
            delay = min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * (2**attempt))
            sleep(random.uniform(0.0, delay))
            continue
        except ValueError as exc:
            attempts.append(f"attempt {attempt + 1}: {exc}")
            if attempt >= max_retries:
                break
            sleep(min(RETRY_MAX_SECONDS, RETRY_BASE_SECONDS * attempt))
            continue
        return {
            **labels,
            "returned_model": result["returned_model"],
            "provider_request_id": result["provider_request_id"] or None,
            "rate_limit_headers": result.get("rate_limit_headers") or {},
            "n_retries": attempt,
            "usage": result["usage"],
            "raw_response_sha256": sha256_text(result["text"]),
            "error": None,
            "source": "model",
        }
    return {
        **dict.fromkeys(fields_for(pass_name)),
        "returned_model": None,
        "provider_request_id": None,
        "n_retries": max_retries,
        "usage": {"input_tokens": 0, "output_tokens": 0},
        "raw_response_sha256": None,
        # Truncated so one pathological provider message cannot dominate the overlay.
        "error": " | ".join(a[:400] for a in attempts),
        "source": "failed",
    }


def _cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Billed USD, or ``None`` when this model's list price was not established.

    Reads the plan and the rate separately. The previous table priced every judge at 0.0
    "by rate, not by omission", which was true of Groq's free plan and false of Gemini —
    which has no free tier — so the artifact said the audit was free while it was billing.
    """
    return price_of(model, input_tokens, output_tokens)


def _paid_fallback_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """What the same tokens cost without the free plan, for models that have one."""
    prices = PAID_FALLBACK_USD_PER_MTOK.get(model)
    if not prices:
        return None
    return round(input_tokens / 1e6 * prices["input"] + output_tokens / 1e6 * prices["output"], 4)


def load_resumable(
    partial_path: Path,
    rows: Sequence[Mapping],
    *,
    pass_name: str,
) -> tuple[dict[str, dict], list[str]]:
    """``(reusable rows by audit_id, conflicts)`` from a partial file.

    Three ways a stored row is not reusable, all of them refusals rather than repairs:

    * it names an ``audit_id`` the current input does not have — the input moved;
    * it appears twice with different content — the partial file is corrupt;
    * its ``prompt_sha256`` is not the hash this invocation would build for that row — the
      rubric, the schema or the row's text changed, so the stored label answers a
      different question than the one now being asked.

    A failed row is never reusable: the point of resuming is to retry it.
    """
    if not partial_path.exists():
        return {}, []
    wanted = {str(r.get("audit_id", "")): r for r in rows}
    reusable: dict[str, dict] = {}
    seen: dict[str, str] = {}
    conflicts: list[str] = []
    for stored in _read_jsonl(partial_path):
        audit_id = str(stored.get("audit_id", ""))
        if not audit_id:
            conflicts.append(f"{partial_path.name}: a stored row has no audit_id")
            continue
        digest = sha256_text(json.dumps(stored, sort_keys=True, ensure_ascii=False))
        if audit_id in seen:
            if seen[audit_id] != digest:
                conflicts.append(
                    f"{audit_id}: stored twice with different content; the partial file "
                    "cannot be trusted. Delete it and re-run without --resume."
                )
                # And the first copy is discarded too. Keeping it would mean returning a
                # row as reusable that this function has just declared untrustworthy —
                # whichever of the two copies happened to be written first.
                reusable.pop(audit_id, None)
            continue
        seen[audit_id] = digest
        source = wanted.get(audit_id)
        if source is None:
            conflicts.append(
                f"{audit_id}: present in {partial_path.name} but not in the input file. "
                "The input moved under a partial run; these are not the same pass."
            )
            continue
        if stored.get("source") == "failed" or stored.get("error"):
            continue  # retry it, which is what resuming is for
        if stored.get("prompt_version") != PROMPT_VERSION:
            conflicts.append(
                f"{audit_id}: stored under prompt version "
                f"{stored.get('prompt_version')!r}, this run is {PROMPT_VERSION!r}."
            )
            continue
        system, user = build_prompt(source, pass_name=pass_name)
        if stored.get("prompt_sha256") != prompt_sha256(system, user):
            conflicts.append(
                f"{audit_id}: stored prompt hash does not match the prompt this run "
                "would build. The rubric or the row's text changed; the stored label "
                "answers a different question."
            )
            continue
        reusable[audit_id] = stored
    return reusable, conflicts


def detector_v4_2_llm_judge(
    judge: str = typer.Option(..., "--judge", help="A (Gemini) or B (Groq gpt-oss)"),
    judge_pass: str = typer.Option("blind", "--pass", help=f"one of {PASSES}"),
    audit_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--audit-dir"),
    output_dir: Path = typer.Option(DEFAULT_V4_2_OUT, "--output-dir"),
    input_file: Path | None = typer.Option(None, "--input", help="override the pass file"),
    model: str = typer.Option(
        "",
        "--model",
        help=(
            "override the judge's model. Must be chosen before the first call and named "
            "in DECISIONS.md; it is recorded in the manifest."
        ),
    ),
    effort: str = typer.Option("low", "--effort", help="frozen; recorded in the manifest"),
    temperature: float = typer.Option(
        0.0, "--temperature", help="frozen; ignored by providers that reject it"
    ),
    max_output_tokens: int = typer.Option(DEFAULT_MAX_OUTPUT_TOKENS, "--max-output-tokens"),
    concurrency: int = typer.Option(DEFAULT_CONCURRENCY, "--concurrency"),
    requests_per_minute: int = typer.Option(
        0, "--rpm", help="0 uses the model's published free-tier ceiling"
    ),
    timeout: float = typer.Option(120.0, "--timeout", help="seconds per request"),
    resume: bool = typer.Option(
        False, "--resume", help="reuse rows already in the partial file; refuse conflicts"
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="smoke run over the first N rows. Requires --run-id."
    ),
    run_id: str = typer.Option(
        "",
        "--run-id",
        help="writes into <output-dir>/<run-id>/. Required for --limit.",
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
    if not MIN_MAX_OUTPUT_TOKENS <= max_output_tokens <= MAX_MAX_OUTPUT_TOKENS:
        raise typer.BadParameter(
            f"--max-output-tokens must be between {MIN_MAX_OUTPUT_TOKENS} and "
            f"{MAX_MAX_OUTPUT_TOKENS}, got {max_output_tokens}. The response is four enum "
            "values; a larger cap only buys room for reasoning tokens nobody reads."
        )
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise typer.BadParameter(f"--concurrency must be 1..{MAX_CONCURRENCY}, got {concurrency}")

    # A limited run is a smoke run, and a smoke run that can land on a real run's path is
    # one mistyped flag away from being mistaken for it.
    if limit is not None and not run_id:
        raise typer.BadParameter(
            "--limit requires --run-id: a limited run is a smoke run and must not share a "
            "directory with the pass it is smoke-testing. Try --run-id smoke-a-blind."
        )
    if run_id:
        if not run_id.replace("-", "").replace("_", "").isalnum():
            raise typer.BadParameter(f"--run-id must be alphanumeric/-/_ , got {run_id!r}")
        output_dir = output_dir / run_id

    spec = dict(JUDGES[judge])
    provider_spec = PROVIDERS[spec["provider"]]
    if model:
        spec["requested_model"] = model
    source = input_file or (audit_dir / INPUT_FILENAME[judge_pass].format(judge=judge))
    if not source.exists():
        raise typer.BadParameter(f"{source} is absent; run `rdl graph-detector-v4-label-audit`")

    rows = _read_jsonl(source)
    n_available = len(rows)
    ids = [str(r.get("audit_id", "")) for r in rows]
    if "" in ids:
        raise typer.BadParameter(f"{source}: a row has no audit_id")
    if len(set(ids)) != len(ids):
        duplicated = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise typer.BadParameter(
            f"{source}: duplicate audit_ids {duplicated}. An overlay keyed by audit_id "
            "cannot represent two rows with one id, and the second would silently win."
        )
    if limit is not None:
        rows = rows[: max(0, int(limit))]
    pass_upper = judge_pass.upper()
    fields = fields_for(judge_pass)
    schema = response_schema(judge_pass)
    rubric_sha = sha256_text(rubric_for(judge_pass))
    schema_sha = sha256_text(json.dumps(schema, sort_keys=True))
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
                    "prompt_version": PROMPT_VERSION,
                    "prompt_sha256": prompt_sha256(system, user),
                }
            )
        path = output_dir / PROMPT_PREVIEW_FILENAME.format(judge=judge, pass_upper=pass_upper)
        _write_jsonl(path, preview)
        typer.echo(f"wrote {path}  ({len(preview)} prompts, no API call made)")
        raise typer.Exit(0)

    # §7. The reference pass sees the answer. Running it before the blind passes are
    # complete and frozen would mean the blind labels could still be regenerated by
    # somebody who has now seen the reference-pass output, and "blind" would describe the
    # prompt rather than the procedure.
    if judge_pass == "reference" and not run_id:
        _require_frozen_blind_passes(output_dir)

    ok, reason = judge_families_are_independent()
    if not ok:
        raise typer.BadParameter(reason)

    partial_path = output_dir / PARTIAL_FILENAME.format(judge=judge, pass_upper=pass_upper)
    out_path = output_dir / OUTPUT_FILENAME.format(judge=judge, pass_upper=pass_upper)
    progress_path = output_dir / PROGRESS_FILENAME.format(judge=judge, pass_upper=pass_upper)

    reusable: dict[str, dict] = {}
    conflicts: list[str] = []
    if resume:
        reusable, conflicts = load_resumable(partial_path, rows, pass_name=judge_pass)
        if conflicts:
            for conflict in conflicts:
                typer.echo(f"  [CONFLICT] {conflict}", err=True)
            raise typer.BadParameter(
                f"{len(conflicts)} row(s) in {partial_path} conflict with this invocation. "
                "Refusing to resume: overwriting them would mix labels produced under "
                "different inputs into one pass."
            )
        typer.echo(f"resuming: {len(reusable)} of {len(rows)} rows already recorded")
    elif partial_path.exists():
        raise typer.BadParameter(
            f"{partial_path} exists from an earlier run. Pass --resume to continue it "
            "(paid-for rows are reused), or delete it to start over. Silently "
            "overwriting it would discard responses that were already spent."
        )

    adapter = build_adapter(
        judge,
        model=spec["requested_model"],
        effort=effort,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        timeout=timeout,
    )
    plan_limits = dict(RATE_LIMITS.get(spec["requested_model"], {}))
    rpm = requests_per_minute or int(plan_limits.get("requests_per_minute") or 0)
    limiter = RateLimiter(rpm or None)
    # The provider's own account-level ceilings, as it reports them on the responses. This
    # is the account-verified number; the table above is a plan-level claim that can be
    # wrong in either direction for any particular key.
    observed_limits: dict[str, str] = {}

    started = time.time()
    lock = threading.Lock()
    completed: dict[str, dict] = {}
    n_forced = 0
    n_malformed = 0
    n_reused = 0
    totals = {"input_tokens": 0, "output_tokens": 0}
    returned_models: set[str] = set()
    request_ids: list[str] = []
    quota_stop: list[str] = []

    def envelope(row: Mapping, record: Mapping) -> dict:
        system, user = build_prompt(row, pass_name=judge_pass)
        return {
            "schema": JUDGE_SCHEMA,
            "audit_id": str(row.get("audit_id", "")),
            "judge": judge,
            "pass": judge_pass,
            "provider": spec["provider"],
            "provider_family": provider_spec["family"],
            "requested_model": spec["requested_model"],
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": prompt_sha256(system, user),
            "rubric_sha256": rubric_sha,
            "response_schema_sha256": schema_sha,
            "input_file_sha256": input_sha,
            **record,
        }

    def flush() -> None:
        """Append-safe checkpoint: rewrite the partial file and the progress manifest.

        A whole-file rewrite rather than an append, because the partial file is small
        (~1k short rows) and `atomic_json`-style replace is the only way to be sure a
        crash mid-write leaves the previous complete file rather than half a line.
        """
        with lock:
            snapshot = list(completed.values())
        _write_jsonl(partial_path, snapshot)
        atomic_json(
            progress_path,
            {
                "schema": "graph-detector-v4-2-judge-progress-v1",
                "judge": judge,
                "pass": judge_pass,
                "requested_model": spec["requested_model"],
                "prompt_version": PROMPT_VERSION,
                "input_file": str(source),
                "input_file_sha256": input_sha,
                "partial_file": str(partial_path),
                "partial_file_sha256": _rows_sha256(snapshot),
                "n_rows_in_input": n_available,
                "n_rows_planned": len(rows),
                "n_rows_recorded": len(snapshot),
                "n_rows_remaining": len(rows) - len(snapshot),
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "resume_with": (
                    f"rdl graph-detector-v4-2-llm-judge --judge {judge} "
                    f"--pass {judge_pass} --resume"
                ),
            },
        )

    for audit_id, stored in reusable.items():
        completed[audit_id] = stored
        n_reused += 1
        usage = stored.get("usage") or {}
        totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
        totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
        if stored.get("returned_model"):
            returned_models.add(str(stored["returned_model"]))
        if stored.get("provider_request_id"):
            request_ids.append(str(stored["provider_request_id"]))

    pending = [r for r in rows if str(r.get("audit_id", "")) not in completed]

    # §4. An absent reference answer is a question the judge was not asked. Forcing
    # UNCERTAIN here is both the protocol rule and 300 calls per judge not spent — and the
    # forced rows are flagged so the report can keep them out of the reference-pass
    # agreement, where 300 trivially-agreeing rows would inflate kappa.
    def forced_record(row: Mapping) -> dict | None:
        if judge_pass != "reference":
            return None
        forced = required_reference_label(row.get("reference_answer"))
        if forced is None:
            return None
        return {
            **dict.fromkeys(fields),
            "reference_content": forced,
            "returned_model": None,
            "provider_request_id": None,
            "n_retries": 0,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "raw_response_sha256": None,
            "error": None,
            "source": "protocol_rule_empty_reference",
        }

    to_call: list[Mapping] = []
    for row in pending:
        record = forced_record(row)
        if record is None:
            to_call.append(row)
            continue
        completed[str(row.get("audit_id", ""))] = envelope(row, record)
        n_forced += 1
    if n_forced:
        flush()

    def work(row: Mapping) -> tuple[str, dict | None]:
        audit_id = str(row.get("audit_id", ""))
        if quota_stop:
            return audit_id, None
        try:
            record = judge_row(
                adapter,
                row,
                pass_name=judge_pass,
                schema=schema,
                limiter=limiter,
            )
        except DailyQuotaExhausted as exc:
            with lock:
                if not quota_stop:
                    quota_stop.append(str(exc))
            return audit_id, None
        return audit_id, envelope(row, record)

    n_done = 0
    if to_call:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            # Batched so a checkpoint lands regularly without one flush per row: the
            # partial file is rewritten whole, and rewriting it 1,019 times is 1,019
            # fsyncs to save at most one row of work.
            batch_size = max(concurrency * 5, 20)
            for start in range(0, len(to_call), batch_size):
                if quota_stop:
                    break
                batch = to_call[start : start + batch_size]
                for audit_id, record in pool.map(work, batch):
                    if record is None:
                        continue
                    with lock:
                        completed[audit_id] = record
                    usage = record.get("usage") or {}
                    totals["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
                    totals["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
                    if record.get("returned_model"):
                        returned_models.add(str(record["returned_model"]))
                    if record.get("provider_request_id"):
                        request_ids.append(str(record["provider_request_id"]))
                    # Last writer wins: the provider's remaining-quota headers change
                    # every call, and what the manifest records is the ceiling, which does
                    # not.
                    observed_limits.update(
                        {k: str(v) for k, v in (record.get("rate_limit_headers") or {}).items()}
                    )
                    if record.get("source") == "failed":
                        n_malformed += 1
                        typer.echo(f"  [FAIL] {audit_id}: {record['error']}", err=True)
                n_done += len(batch)
                flush()
                typer.echo(
                    f"  {len(completed)}/{len(rows)} rows recorded "
                    f"({n_malformed} failed, {n_reused} reused)"
                )

    flush()
    out_rows = [completed[i] for i in ids if i in completed]
    _write_jsonl(out_path, out_rows)
    output_sha = _rows_sha256(out_rows)

    n_recorded = len(out_rows)
    n_missing = len(rows) - n_recorded
    reportable = limit is None and not run_id
    complete = reportable and n_recorded == n_available and n_malformed == 0 and n_missing == 0
    why_incomplete: list[str] = []
    if not reportable:
        why_incomplete.append(
            "a --limit / --run-id smoke run. Smoke runs are never reportable: they "
            "measure that the path works, not what the labels are."
        )
    if n_missing:
        why_incomplete.append(f"{n_missing} row(s) were never attempted")
    if n_malformed:
        why_incomplete.append(f"{n_malformed} row(s) failed after their retries")
    if quota_stop:
        why_incomplete.append(
            f"the provider's per-day ceiling ended the run ({quota_stop[0][:200]}). "
            "The partial file is intact; re-run with --resume when the quota resets."
        )

    manifest = {
        "schema": RUN_SCHEMA,
        "protocol": V4_2_PROTOCOL,
        "judge": judge,
        "role": f"Model Judge {judge}",
        "judge_population": "two_independent_llm_judges",
        "human_grounded": False,
        "pass": judge_pass,
        "fields": list(fields),
        "reportable": reportable,
        "run_id": run_id or None,
        "provider": spec["provider"],
        "provider_family": provider_spec["family"],
        "provider_base_url": provider_spec["base_url"],
        "provider_data_retention_note": provider_spec.get("data_retention_note", ""),
        "requested_model": spec["requested_model"],
        "protocol_default_model": JUDGES[judge]["requested_model"],
        "model_overridden": bool(model),
        "model_choice_note": CHEAPER_JUDGE_NOTE,
        "returned_model": sorted(returned_models)[0] if len(returned_models) == 1 else None,
        "returned_models_seen": sorted(returned_models),
        "sdk_version": adapter.sdk_version(),
        "n_provider_request_ids": len(request_ids),
        "provider_request_id_sample": request_ids[:3],
        "parameters": adapter.parameters(),
        "max_retries": MAX_RETRIES,
        "retry_policy": (
            f"{MAX_RETRIES} retries with full-jitter exponential backoff on "
            f"{sorted(RETRYABLE_STATUS)}; a per-day quota ends the run rather than "
            "failing the row"
        ),
        "concurrency": concurrency,
        "requests_per_minute": rpm or None,
        # Two different kinds of claim, kept apart. The plan limits are documentation; the
        # observed ones came off this account's own responses. Before v4.2.2 there was one
        # field called `published_free_tier_limits`, which asserted a free tier Gemini does
        # not offer and account limits nobody had checked.
        "plan_rate_limits": plan_limits or None,
        "observed_rate_limits": observed_limits or None,
        "rate_limit_verification": (
            "account-verified from the provider's x-ratelimit-* response headers"
            if observed_limits
            else "not verified: the provider returned no x-ratelimit-* headers on this run"
        ),
        "free_tier_available": plan_limits.get("free_tier_available"),
        "billing": plan_limits.get("billing"),
        "input_file": str(source),
        "input_file_sha256": input_sha,
        "prompt_version": PROMPT_VERSION,
        "rubric_sha256": rubric_sha,
        "response_schema_sha256": schema_sha,
        "output_file": str(out_path),
        "output_file_sha256": output_sha,
        "partial_file": str(partial_path),
        "progress_file": str(progress_path),
        "n_rows_in_input": n_available,
        "n_rows_processed": len(rows),
        "n_rows_recorded": n_recorded,
        "n_rows_reused_from_partial": n_reused,
        "n_rows_called": max(0, n_recorded - n_forced - n_reused),
        "n_rows_missing": n_missing,
        "n_forced_by_protocol_rule": n_forced,
        "forced_rule": (
            "an empty reference_answer is labelled UNCERTAIN without a call, and is "
            "excluded from the reference-pass agreement: trivially agreeing rows would "
            "inflate a kappa neither judge earned."
        ),
        "n_malformed_or_missing": n_malformed + n_missing,
        "complete": complete,
        "why_incomplete": " ".join(why_incomplete) or None,
        "token_usage": totals,
        "estimated_cost_usd": _cost(
            spec["requested_model"], totals["input_tokens"], totals["output_tokens"]
        ),
        "paid_fallback_cost_usd": _paid_fallback_cost(
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
        "credentials": (
            f"from {provider_spec['api_key_env']}; never recorded here or in any artifact"
        ),
        "scope": {
            "model_trained": False,
            "graph_generation_run": False,
            "frozen_v1_v2_v3_v4_v4_1_artifacts_modified": False,
            "gpu_used": False,
            "final_gate_bank_opened": False,
        },
    }
    atomic_json(output_dir / RUN_FILENAME.format(judge=judge, pass_upper=pass_upper), manifest)
    typer.echo(f"wrote {out_path}  ({n_recorded} rows)")
    typer.echo(
        f"judge {judge} / {judge_pass}: {manifest['n_rows_called']} calls, "
        f"{n_reused} reused, {n_forced} forced, {n_malformed} failed, "
        f"{n_missing} missing, complete={complete}"
    )
    if quota_stop:
        typer.echo("", err=True)
        typer.echo(
            "the provider's per-day quota ended this run. Nothing was lost: re-run the "
            "same command with --resume once the quota resets.",
            err=True,
        )
    raise typer.Exit(0 if complete else 1)


def _require_frozen_blind_passes(output_dir: Path) -> None:
    """Refuse a reference pass until both blind manifests exist and are complete.

    The reference pass shows the judge the answer. If it can run first, then the blind
    labels can be produced afterwards by an operator who has already seen reference-pass
    output, and the blindness is a property of one prompt rather than of the procedure.
    """
    missing: list[str] = []
    for role in JUDGES:
        path = output_dir / RUN_FILENAME.format(judge=role, pass_upper="BLIND")
        if not path.exists():
            missing.append(f"{path} is absent")
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            missing.append(f"{path} records complete=false ({manifest.get('why_incomplete')})")
        if not manifest.get("reportable", True):
            missing.append(f"{path} is a smoke run and is not reportable")
        if manifest.get("prompt_version") != PROMPT_VERSION:
            missing.append(
                f"{path} was produced under prompt version "
                f"{manifest.get('prompt_version')!r}, this run is {PROMPT_VERSION!r}"
            )
    if missing:
        raise typer.BadParameter(
            "the reference pass is refused until BOTH blind passes are complete and "
            "frozen (DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md section 7): "
            + "; ".join(missing)
            + ". The reference pass shows the judge the answer; running it first would "
            "let the blind labels be produced by an operator who has already seen it."
        )


def _iter_pending(rows: Iterable[Mapping], done: Mapping[str, object]) -> list[Mapping]:
    """Rows not yet recorded, in input order. Small, but named so tests can use it."""
    return [r for r in rows if str(r.get("audit_id", "")) not in done]
