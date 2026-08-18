"""``rdl graph-detector-v4-3-local-judge`` -- one open-weight judge, one process, one pass.

The v4.2 judge runner talks to two hosted APIs over HTTP. This one loads weights, and that
difference changes what can go wrong: instead of rate limits and quota exhaustion the
failure modes are OOM, a quantizer that silently falls back, a chat template that is not
the one that was pinned, and a model that fences its JSON. Each is recorded rather than
absorbed.

Three properties carried over from v4.2 unchanged, because they are what make a labelling
run resumable and auditable:

**Durable, content-addressed resume.** Every completed row is appended to a partial file
keyed by ``audit_id``, and a restart skips what is already there. A run that dies at row
800 of 1,019 resumes at 801 rather than re-annotating 800 rows with a model whose sampling
is only as deterministic as its seed.

**Malformed is a failure, not a default.** A response that does not parse is retried and
then recorded as malformed. It never becomes a label -- the gate requires zero malformed
rows, and a count of zero means nothing if the parser fills in NONE when it is confused.

**Blind means structurally blind.** The prompt is built by
:func:`~rdl.eval.detector_v4_3_judges.blind_prompt`, whose signature has no parameter for
population, stratum or the reference answer. The reference pass is a different command
path and refuses to run until the blind outputs are frozen.

``--fake-model``
----------------
Loads nothing and answers from a deterministic rule. It exists so that resume, malformed
handling, prompt injection, the single-model guard and the output schema are all tested on
CPU, before the 3090 is rented -- these are the parts that waste GPU hours when they break,
and none of them needs a GPU to be exercised.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_3_judges import (
    LOCAL_JUDGE_ROSTER,
    PROMPT_VERSION,
    JudgeRunRecord,
    MalformedJudgement,
    assert_single_model_process,
    blind_prompt,
    parse_strict_json,
    response_hashes,
)
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_store import DEFAULT_OUT_DIR

__all__ = [
    "MAX_ATTEMPTS",
    "OUTPUT_FILENAME",
    "PARTIAL_FILENAME",
    "detector_v4_3_local_judge",
    "fake_judgement",
    "run_rows",
]

OUTPUT_FILENAME = "V4_3_LOCAL_JUDGE_{judge}_{pass_upper}.jsonl"
PARTIAL_FILENAME = "V4_3_LOCAL_JUDGE_{judge}_{pass_upper}.partial.jsonl"
RUN_FILENAME = "V4_3_LOCAL_JUDGE_RUN_{judge}_{pass_upper}.json"

# Retries per row before it is recorded as malformed. Three, not the API runner's five:
# the failures here are parsing failures rather than transient network ones, and a model
# that has emitted unparseable output three times at temperature 0 will do it again.
MAX_ATTEMPTS = 3


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append_jsonl(path: Path, row: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(dumps_canonical(row) + "\n")


def fake_judgement(prompt: str) -> str:
    """A deterministic stand-in, so the harness is testable without weights.

    Deliberately crude and deliberately not always well-formed: one row in every seventeen
    comes back as prose, so the malformed path is exercised by an ordinary ``--fake-model``
    run rather than only by a test that reaches in and breaks something.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if int(digest[:2], 16) % 17 == 0:
        return "I think this one is hard to call, honestly."
    label = ("NONE", "PARTIAL", "ANSWER")[int(digest[2:4], 16) % 3]
    question_type = ("slot", "open-ended")[int(digest[4:6], 16) % 2]
    return json.dumps(
        {
            "answer_attempt": label,
            "subject_only": "no",
            "refusal": "no",
            "question_type": question_type,
        }
    )


def run_rows(
    rows: Sequence[Mapping],
    generate,
    *,
    partial_path: Path,
    role: str,
    max_attempts: int = MAX_ATTEMPTS,
) -> dict:
    """Judge ``rows``, appending each result durably, skipping what is already done.

    ``generate`` is ``(prompt) -> response text``. Keeping it a parameter is what lets the
    fake model, the real model and the tests share one control flow -- the resume logic and
    the malformed accounting are the parts worth testing, and they must not have a separate
    implementation for the case that never runs on CPU.
    """
    done = {str(r["audit_id"]) for r in _read_jsonl(partial_path)}
    counts = {"n_judged": 0, "n_skipped": len(done), "n_malformed": 0, "n_retries": 0}

    for row in rows:
        audit_id = str(row["audit_id"])
        if audit_id in done:
            continue
        prompt = blind_prompt(
            conditioning_question=str(row["conditioning_question"]),
            subject_aliases=list(row.get("subject_aliases", ())),
            candidate_text=str(row["candidate_text"]),
        )
        judgement: dict | None = None
        last_error = ""
        raw = ""
        for attempt in range(max_attempts):
            raw = generate(prompt)
            try:
                judgement = parse_strict_json(raw)
                break
            except MalformedJudgement as error:
                last_error = str(error)
                counts["n_retries"] += int(attempt + 1 < max_attempts)

        record = {
            "audit_id": audit_id,
            "judge": role,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            **response_hashes(raw),
        }
        if judgement is None:
            # Recorded, not dropped and not defaulted. A row with no label must be visible
            # as a row with no label, because the gate counts it.
            counts["n_malformed"] += 1
            record.update(
                {
                    "answer_attempt": None,
                    "subject_only": None,
                    "refusal": None,
                    "question_type": None,
                    "malformed": True,
                    "error": last_error,
                }
            )
        else:
            counts["n_judged"] += 1
            record.update({**judgement, "malformed": False})
        _append_jsonl(partial_path, record)

    return counts


def _build_generator(pin, *, fake: bool, device: str):
    """``(generate, description)``. The only place weights are touched."""
    if fake:
        return fake_judgement, {
            "loaded": False,
            "device": "cpu",
            "dtype": "n/a",
            "why": (
                "--fake-model: no weights are loaded and no network is used. Exercises "
                "resume, malformed handling, injection framing and the output schema."
            ),
        }

    # Imported here so the CPU test suite never needs transformers or bitsandbytes.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    assert_single_model_process(pin.repo_id)
    if not pin.pinned:
        raise typer.BadParameter(
            f"judge {pin.role} has revision {pin.revision!r}, which is not a 40-character "
            "commit sha. Freeze the pins first: a tag can move, and a labelling run under "
            "a moved tag is a different annotator with the same name."
        )
    if pin.quantization == "bitsandbytes-8bit":
        quantization = BitsAndBytesConfig(load_in_8bit=True)
    elif pin.quantization == "bitsandbytes-4bit":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=getattr(torch, pin.compute_dtype),
            bnb_4bit_quant_type="nf4",
        )
    else:
        raise typer.BadParameter(f"unknown quantization {pin.quantization!r}")

    tokenizer = AutoTokenizer.from_pretrained(pin.repo_id, revision=pin.revision)
    model = AutoModelForCausalLM.from_pretrained(
        pin.repo_id,
        revision=pin.revision,
        quantization_config=quantization,
        device_map=device,
    )
    model.eval()

    def generate(prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {"add_generation_prompt": True, "tokenize": False}
        if pin.thinking_mode is False and "Qwen3" in pin.repo_id:
            # Qwen3's template takes an explicit switch. Passing it only for Qwen keeps the
            # call identical to the model card's documented form for each family.
            template_kwargs["enable_thinking"] = False
        text = tokenizer.apply_chat_template(messages, **template_kwargs)
        batch = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **batch,
                max_new_tokens=pin.max_new_tokens,
                do_sample=pin.temperature > 0,
                temperature=pin.temperature or None,
                top_p=pin.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(output[0][batch["input_ids"].shape[1] :], skip_special_tokens=True)

    return generate, {
        "loaded": True,
        "device": str(model.device),
        "dtype": str(getattr(model, "dtype", "")),
        "quantization": pin.quantization,
    }


def detector_v4_3_local_judge(
    judge: str = typer.Option(..., "--judge", help="A (Qwen) or B (Mistral)."),
    bundle: Path = typer.Option(
        DEFAULT_OUT_DIR / "DETECTOR_V4_3_PAIR_BUNDLE.json",
        "--bundle",
        help="the pair bundle whose rows are judged.",
    ),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir"),
    device: str = typer.Option("cuda:0", "--device"),
    limit: int = typer.Option(0, "--limit", help="judge at most N rows. 0 means all."),
    fake_model: bool = typer.Option(
        False, "--fake-model", help="load nothing; exercise the harness on CPU."
    ),
    close: bool = typer.Option(
        False,
        "--close",
        help="promote the partial file to the frozen output and hash it. Refuses if any "
        "row is missing or malformed.",
    ),
) -> None:
    """Run one blind judging pass with one local model."""
    if judge not in LOCAL_JUDGE_ROSTER:
        raise typer.BadParameter(f"--judge must be one of {sorted(LOCAL_JUDGE_ROSTER)}")
    pin = LOCAL_JUDGE_ROSTER[judge]
    if not Path(bundle).exists():
        raise typer.BadParameter(f"{bundle} is absent. Run `rdl graph-detector-v4-3-bundle`.")

    payload = json.loads(Path(bundle).read_text(encoding="utf-8"))
    rows = list(payload.get("pairs", ()))
    if limit:
        rows = rows[:limit]

    out = Path(out_dir)
    partial_path = out / PARTIAL_FILENAME.format(judge=judge, pass_upper="BLIND")
    output_path = out / OUTPUT_FILENAME.format(judge=judge, pass_upper="BLIND")

    if close:
        return _close(partial_path, output_path, rows=rows, judge=judge, out=out)

    generate, described = _build_generator(pin, fake=fake_model, device=device)
    started = time.time()
    counts = run_rows(rows, generate, partial_path=partial_path, role=judge)

    peak = 0
    if described.get("loaded"):
        import torch

        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    record = JudgeRunRecord(
        role=judge,
        repo_id=pin.repo_id,
        revision=pin.revision,
        quantization=pin.quantization,
        device=str(described.get("device", "")),
        dtype=str(described.get("dtype", "")),
        peak_vram_bytes=peak,
        n_rows=len(rows),
        n_malformed=counts["n_malformed"],
        n_retries=counts["n_retries"],
        seconds=round(time.time() - started, 1),
        extra={
            "bundle": str(bundle),
            "bundle_sha256": payload.get("bundle_sha256"),
            "partial_file": str(partial_path),
            "fake_model": bool(fake_model),
            "reportable": not fake_model,
            "counts": counts,
            "pin": pin.to_dict(),
            "generator": described,
        },
    )
    atomic_json(out / RUN_FILENAME.format(judge=judge, pass_upper="BLIND"), record.to_dict())
    typer.echo(dumps_canonical(record.to_dict()))


def _close(
    partial_path: Path, output_path: Path, *, rows: Sequence[Mapping], judge: str, out: Path
) -> None:
    """Freeze a complete pass. Refuses an incomplete or malformed one."""
    judged = _read_jsonl(partial_path)
    by_id = {str(r["audit_id"]): r for r in judged}
    missing = [str(r["audit_id"]) for r in rows if str(r["audit_id"]) not in by_id]
    malformed = sorted(k for k, v in by_id.items() if v.get("malformed"))
    if missing or malformed:
        raise typer.BadParameter(
            f"cannot close judge {judge}: {len(missing)} row(s) unjudged and "
            f"{len(malformed)} malformed. The label gate requires zero of each, and a "
            "closed file that quietly omitted them would satisfy the gate by having fewer "
            f"rows. First missing {missing[:3]}, first malformed {malformed[:3]}."
        )
    ordered = [by_id[str(r["audit_id"])] for r in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(dumps_canonical(r) + "\n" for r in ordered), encoding="utf-8")
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    typer.echo(
        dumps_canonical(
            {
                "closed": str(output_path),
                "judge": judge,
                "n_rows": len(ordered),
                "sha256": digest,
                "n_malformed": 0,
            }
        )
    )
