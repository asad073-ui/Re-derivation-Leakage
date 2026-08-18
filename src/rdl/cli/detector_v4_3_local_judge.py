"""``rdl graph-detector-v4-3-local-judge`` -- one open-weight judge, one process, one pass.

The v4.2 judge runner talks to two hosted APIs over HTTP. This one loads weights, and that
difference changes what can go wrong: instead of rate limits and quota exhaustion the
failure modes are OOM, a quantizer that silently falls back to fp16 or CPU offload, an auto
class that does not match the checkpoint, a chat template that is not the pinned one, and a
model that fences its JSON. Each is checked or recorded rather than absorbed.

Properties that make a labelling run resumable and auditable:

**Durable, content-addressed resume.** Every completed row is appended to a partial file
keyed by ``audit_id``, and a restart skips what is already there.

**Malformed is a failure, not a default.** A response that does not parse is retried and
then recorded with a null label. It never becomes a label -- the gate requires zero
malformed rows, and that count means nothing if the parser fills in NONE when confused.

**Blind means structurally blind.** :func:`~rdl.eval.detector_v4_3_judges.blind_prompt`
takes exactly question, aliases and candidate. The reference pass is a different function
with a different rubric, and ``--pass reference`` refuses to start until the blind pass for
that judge has been closed -- a reference label produced before the blind labels are frozen
could have been revised in the light of the answer.

**The annotator comes from a committed file.** ``--pins`` is read for the repo, commit,
quantizer, dtype, template, thinking mode and generation parameters. Nothing that changes
an output token is a runtime flag.

**Reportable and smoke runs cannot collide.** ``--non-reportable --run-id X`` namespaces
every output file and stamps ``reportable: false``. Without it a ``--limit 50`` run wrote
the reportable filenames with 50 rows in them, which is a truncated reportable pass wearing
a smoke's clothes. The GPU-1 smoke additionally uses a fixture that is disjoint from the
1,019 -- see ``rdl graph-detector-v4-3-judge-smoke-fixture``.
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
    reference_prompt,
    response_hashes,
)
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_pins import JUDGE_PINS_FILENAME, load_judge_pins
from .detector_v4_3_store import DEFAULT_OUT_DIR

__all__ = [
    "MAX_ATTEMPTS",
    "PASSES",
    "REFERENCE_VALUES",
    "detector_v4_3_local_judge",
    "fake_judgement",
    "output_names",
    "run_rows",
]

PASSES = ("blind", "reference")

# The reference pass answers ONE field, with its own enum. Kept beside the blind schema
# rather than merged into it: a parser that accepted either shape would accept a blind
# response to a reference prompt and nobody would notice.
REFERENCE_VALUES = {"reference_content": {"YES", "NO", "UNCERTAIN"}}

MAX_ATTEMPTS = 3


def output_names(*, judge: str, pass_name: str, run_id: str, reportable: bool) -> dict[str, str]:
    """Filenames for one pass. A non-reportable run cannot occupy a reportable name."""
    stem = (
        f"V4_3_LOCAL_JUDGE_{judge}_{pass_name.upper()}"
        if reportable
        else f"V4_3_SMOKE_{run_id}_JUDGE_{judge}_{pass_name.upper()}"
    )
    return {
        "output": f"{stem}.jsonl",
        "partial": f"{stem}.partial.jsonl",
        "run": f"{stem}_RUN.json",
    }


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

    Deliberately not always well-formed: one row in every seventeen comes back as prose, so
    the malformed path is exercised by an ordinary ``--fake-model`` run rather than only by
    a test that reaches in and breaks something.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if int(digest[:2], 16) % 17 == 0:
        return "I think this one is hard to call, honestly."
    if "<reference_answer>" in prompt:
        return json.dumps(
            {"reference_content": ("YES", "NO", "UNCERTAIN")[int(digest[2:4], 16) % 3]}
        )
    return json.dumps(
        {
            "answer_attempt": ("NONE", "PARTIAL", "ANSWER")[int(digest[2:4], 16) % 3],
            "subject_only": "no",
            "refusal": "no",
            "question_type": ("slot", "open-ended")[int(digest[4:6], 16) % 2],
        }
    )


def run_rows(
    rows: Sequence[Mapping],
    generate,
    *,
    partial_path: Path,
    role: str,
    pass_name: str = "blind",
    reference_of: Mapping[str, str] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    measure_tokens=None,
) -> dict:
    """Judge ``rows``, appending each result durably, skipping what is already done.

    ``generate`` is ``(prompt) -> response``. Keeping it a parameter is what lets the fake
    model, the real model and the tests share one control flow: resume and malformed
    accounting are the parts worth testing, and they must not have a second implementation
    for the case that only runs on GPU.

    ``measure_tokens`` is ``(prompt) -> (n_tokens, limit)``. Supplied only by the real
    generator; a prompt that would be truncated is recorded and counted, because a judged
    row whose candidate was cut off is a label about different text.
    """
    expected = REFERENCE_VALUES if pass_name == "reference" else None
    done = {str(r["audit_id"]) for r in _read_jsonl(partial_path)}
    counts = {
        "n_judged": 0,
        "n_skipped": len(done),
        "n_malformed": 0,
        "n_retries": 0,
        "n_truncated": 0,
    }

    for row in rows:
        audit_id = str(row["audit_id"])
        if audit_id in done:
            continue
        if pass_name == "reference":
            prompt = reference_prompt(
                conditioning_question=str(row["conditioning_question"]),
                candidate_text=str(row["candidate_text"]),
                reference_answer=str((reference_of or {}).get(audit_id, "")),
            )
        else:
            prompt = blind_prompt(
                conditioning_question=str(row["conditioning_question"]),
                subject_aliases=list(row.get("subject_aliases", ())),
                candidate_text=str(row["candidate_text"]),
            )

        truncated = False
        n_tokens = None
        if measure_tokens is not None:
            n_tokens, limit = measure_tokens(prompt)
            truncated = bool(limit) and n_tokens > limit
            counts["n_truncated"] += int(truncated)

        judgement: dict | None = None
        last_error = ""
        raw = ""
        for attempt in range(max_attempts):
            raw = generate(prompt)
            try:
                judgement = parse_strict_json(raw, expected=expected)
                break
            except MalformedJudgement as error:
                last_error = str(error)
                counts["n_retries"] += int(attempt + 1 < max_attempts)

        record = {
            "audit_id": audit_id,
            "judge": role,
            "pass": pass_name,
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_tokens": n_tokens,
            "prompt_truncated": truncated,
            **response_hashes(raw),
        }
        if judgement is None:
            counts["n_malformed"] += 1
            record.update(
                {
                    **dict.fromkeys(expected or _BLIND_FIELDS),
                    "malformed": True,
                    "error": last_error,
                }
            )
        else:
            counts["n_judged"] += 1
            record.update({**judgement, "malformed": False})
        _append_jsonl(partial_path, record)

    return counts


_BLIND_FIELDS = ("answer_attempt", "subject_only", "refusal", "question_type")


def _resolve_auto_class(repo_id: str, revision: str):
    """Pick the model class from the CHECKPOINT'S OWN config, never from a guess.

    v4.3 hard-coded ``AutoModelForCausalLM``. That is right for Qwen3-14B and wrong for
    Mistral-Small-3.2-24B-Instruct-2506, whose config declares
    ``Mistral3ForConditionalGeneration`` -- a conditional-generation architecture that
    ``AutoModelForCausalLM`` refuses with an unrecognised-configuration error. Hard-coding
    either class makes one of the two judges unloadable.

    Reading ``config.architectures`` and dispatching means the runner is correct for both
    without this file having to encode a claim about a model card it cannot check offline.
    The class actually used is recorded in the run manifest.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(repo_id, revision=revision)
    architectures = list(getattr(config, "architectures", None) or [])

    for name in architectures:
        if "ConditionalGeneration" in name or "ImageTextToText" in name:
            try:
                from transformers import AutoModelForImageTextToText

                return AutoModelForImageTextToText, name
            except ImportError as error:  # pragma: no cover - depends on transformers version
                raise typer.BadParameter(
                    f"{repo_id} declares {name}, which needs a transformers new enough to "
                    "expose AutoModelForImageTextToText. Upgrade transformers in the gpu "
                    "extra rather than forcing AutoModelForCausalLM, which will refuse "
                    "this configuration."
                ) from error
    return AutoModelForCausalLM, (architectures[0] if architectures else "unknown")


def _verify_quantization(model, pin) -> dict:
    """Confirm the loaded model is the annotator the pin names. Raises when it is not.

    A ``BitsAndBytesConfig`` is a *request*. If bitsandbytes is missing, or the dtype is
    unsupported, or the device map spills to CPU, the load can still succeed and produce a
    model that is not what the manifest claims. That model may well label differently, and
    the run would be attributed to a quantizer that never ran.
    """
    config = getattr(model, "config", None)
    quantization = getattr(config, "quantization_config", None)
    described: dict = {"requested": pin.quantization, "config_present": quantization is not None}
    if quantization is not None:
        as_dict = quantization.to_dict() if hasattr(quantization, "to_dict") else dict(quantization)
        described["loaded"] = {
            k: v
            for k, v in as_dict.items()
            if k
            in ("load_in_4bit", "load_in_8bit", "bnb_4bit_quant_type", "bnb_4bit_compute_dtype")
        }
        wanted_8bit = pin.quantization.endswith("8bit")
        actual_8bit = bool(as_dict.get("load_in_8bit"))
        actual_4bit = bool(as_dict.get("load_in_4bit"))
        if wanted_8bit != actual_8bit or wanted_8bit == actual_4bit:
            raise typer.BadParameter(
                f"the pin names {pin.quantization} but the loaded model reports "
                f"load_in_8bit={actual_8bit}, load_in_4bit={actual_4bit}. A nominal 4/8-bit "
                "run that silently loaded otherwise is a different annotator than the "
                "manifest names."
            )
    else:
        raise typer.BadParameter(
            f"the pin names {pin.quantization} but the loaded model carries no "
            "quantization_config. bitsandbytes probably fell back; refusing to label with "
            "an annotator the manifest cannot describe."
        )

    devices = {str(p.device) for p in model.parameters()}
    described["parameter_devices"] = sorted(devices)
    offloaded = sorted(d for d in devices if not d.startswith("cuda"))
    if offloaded:
        raise typer.BadParameter(
            f"parameters are on {offloaded}. CPU/disk offload changes throughput and can "
            "change numerics; it must be preregistered rather than discovered."
        )
    return described


class _ChatTokenizer:
    """One prompt-to-token-ids interface over two tokenizer families.

    The runner needs four things from a tokenizer -- apply the family's chat template,
    count the resulting tokens, feed them to ``generate``, and decode only the completion.
    Qwen3 and Mistral-Small-3.2 expose those through APIs that share no method names, so
    the difference is absorbed here rather than branching inside the generation loop.
    """

    def __init__(self, *, kind: str, max_length: int, eos_id: int | None, describe: dict):
        self.kind = kind
        self.max_length = max_length
        self.eos_id = eos_id
        self.describe = describe

    def encode_chat(self, prompt: str) -> list[int]:  # pragma: no cover - overridden
        raise NotImplementedError

    def decode(self, ids) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class _HFChatTokenizer(_ChatTokenizer):
    """``AutoTokenizer`` plus ``apply_chat_template``. Correct for Qwen3-14B."""

    def __init__(self, tokenizer, pin):
        limit = int(getattr(tokenizer, "model_max_length", 0) or 0)
        if limit > 1_000_000:  # some tokenizers use a sentinel rather than a real bound
            limit = 0
        super().__init__(
            kind="transformers",
            max_length=limit,
            eos_id=tokenizer.eos_token_id,
            describe={
                "tokenizer_class": type(tokenizer).__name__,
                "source": "transformers.AutoTokenizer",
                "chat_template": "tokenizer.apply_chat_template",
            },
        )
        self._tokenizer = tokenizer
        self._pin = pin

    def encode_chat(self, prompt: str) -> list[int]:
        kwargs: dict = {"add_generation_prompt": True, "tokenize": False}
        if "Qwen3" in self._pin.repo_id:
            # Qwen3's template takes an explicit switch; passing it only for Qwen keeps the
            # call identical to each family's documented form.
            kwargs["enable_thinking"] = bool(self._pin.thinking_mode)
        text = self._tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
        return list(self._tokenizer(text)["input_ids"])

    def decode(self, ids) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=True)


class _MistralCommonChatTokenizer(_ChatTokenizer):
    """``mistral-common``, which is the only tokenizer Mistral-Small-3.2 ships for.

    ``mistralai/Mistral-Small-3.2-24B-Instruct-2506`` publishes ``tekken.json`` and no
    ``tokenizer.json`` or ``tokenizer_config.json``, and transformers 4.51 has no
    ``Mistral3Config`` entry in its tokenizer mapping. ``AutoTokenizer.from_pretrained``
    therefore raises ``KeyError`` on that repository -- not a warning, not a slow path.
    The model card's own instruction is to tokenize through ``mistral-common``, so the
    documented path is the implemented one, and the template markers come from the
    checkpoint rather than from a template this repository would otherwise have to guess.
    """

    def __init__(self, tokenizer, pin):
        inner = tokenizer.instruct_tokenizer.tokenizer
        super().__init__(
            kind="mistral-common",
            max_length=int(getattr(inner, "n_words", 0) or 0),
            eos_id=int(inner.eos_id),
            describe={
                "tokenizer_class": type(inner).__name__,
                "source": "mistral_common.MistralTokenizer.from_hf_hub",
                "chat_template": "encode_chat_completion",
            },
        )
        self._tokenizer = tokenizer
        self._inner = inner
        self._pin = pin

    def encode_chat(self, prompt: str) -> list[int]:
        from mistral_common.protocol.instruct.messages import UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        request = ChatCompletionRequest(messages=[UserMessage(content=prompt)])
        return list(self._tokenizer.encode_chat_completion(request).tokens)

    def decode(self, ids) -> str:
        from mistral_common.tokens.tokenizers.base import SpecialTokenPolicy

        return self._inner.decode(list(ids), special_token_policy=SpecialTokenPolicy.IGNORE)


def _resolve_tokenizer(pin):
    """Pick the tokenizer family from what the checkpoint actually publishes.

    Dispatch is on the declared architecture, for the same reason ``_resolve_auto_class``
    dispatches there: it is a fact about the checkpoint rather than a claim this file
    makes about a model card it cannot read offline.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(pin.repo_id, revision=pin.revision)
    architectures = list(getattr(config, "architectures", None) or [])
    needs_mistral_common = any(name.startswith("Mistral3") for name in architectures)

    if needs_mistral_common:
        try:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        except ImportError as error:
            raise typer.BadParameter(
                f"{pin.repo_id} declares {architectures} and ships only tekken.json, so it "
                "tokenizes through mistral-common. Install the gpu extra, which pins "
                "mistral-common>=1.6.2. AutoTokenizer raises KeyError on this repository."
            ) from error
        return _MistralCommonChatTokenizer(
            MistralTokenizer.from_hf_hub(pin.repo_id, revision=pin.revision), pin
        )

    from transformers import AutoTokenizer

    return _HFChatTokenizer(AutoTokenizer.from_pretrained(pin.repo_id, revision=pin.revision), pin)


def _build_generator(pin, *, fake: bool, device: str):
    """``(generate, measure_tokens, description)``. The only place weights are touched."""
    if fake:
        return (
            fake_judgement,
            None,
            {
                "loaded": False,
                "device": "cpu",
                "dtype": "n/a",
                "why": (
                    "--fake-model: no weights, no network. Exercises resume, malformed "
                    "handling, injection framing, pass separation and the output schema."
                ),
            },
        )

    import torch
    from transformers import BitsAndBytesConfig

    assert_single_model_process(pin.repo_id)

    # Seeding, which v4.3 recorded in provenance and never applied. Greedy decoding makes
    # it mostly moot, but a manifest that claims a control the code does not apply is the
    # kind of thing that is only discovered when a run fails to reproduce.
    torch.manual_seed(pin.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(pin.seed)

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

    auto_class, architecture = _resolve_auto_class(pin.repo_id, pin.revision)
    tokenizer = _resolve_tokenizer(pin)
    model = auto_class.from_pretrained(
        pin.repo_id,
        revision=pin.revision,
        quantization_config=quantization,
        device_map=device,
    )
    model.eval()
    quantization_report = _verify_quantization(model, pin)

    limit = tokenizer.max_length

    def measure_tokens(prompt: str) -> tuple[int, int]:
        # The templated length, not the bare prompt's: the template is part of what the
        # context has to hold, and a row that overflows it is a label about cut-off text.
        return len(tokenizer.encode_chat(prompt)), limit

    def generate(prompt: str) -> str:
        ids = tokenizer.encode_chat(prompt)
        batch = torch.tensor([ids], device=model.device)
        with torch.inference_mode():
            output = model.generate(
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                max_new_tokens=pin.max_new_tokens,
                do_sample=pin.temperature > 0,
                temperature=pin.temperature or None,
                top_p=pin.top_p,
                pad_token_id=tokenizer.eos_id,
            )
        return tokenizer.decode(output[0][len(ids) :])

    return (
        generate,
        measure_tokens,
        {
            "loaded": True,
            "device": str(model.device),
            "dtype": str(getattr(model, "dtype", "")),
            "auto_class": auto_class.__name__,
            "architecture": architecture,
            "quantization": quantization_report,
            "tokenizer": tokenizer.describe,
            "tokenizer_model_max_length": limit,
            "seeded_with": pin.seed,
        },
    )


def _load_input(path: Path, limit: int) -> tuple[list[dict], dict]:
    """Rows plus provenance, from a pair bundle or a plain JSONL fixture."""
    path = Path(path)
    if not path.exists():
        raise typer.BadParameter(f"{path} is absent")
    if path.suffix == ".jsonl":
        rows = _read_jsonl(path)
        provenance = {
            "input": str(path),
            "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kind": "jsonl fixture",
        }
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = list(payload.get("pairs", ()))
        provenance = {
            "input": str(path),
            "bundle_sha256": payload.get("bundle_sha256"),
            "kind": "pair bundle",
        }
    for row in rows:
        for field in ("audit_id", "conditioning_question", "candidate_text"):
            if field not in row:
                raise typer.BadParameter(f"{path}: a row is missing {field!r}")
    return (rows[:limit] if limit else rows), provenance


def detector_v4_3_local_judge(
    judge: str = typer.Option(..., "--judge", help="A (Qwen) or B (Mistral)."),
    pass_name: str = typer.Option("blind", "--pass", help="blind or reference."),
    input_path: Path = typer.Option(
        None, "--input", help="a JSONL fixture. Overrides --bundle when given."
    ),
    bundle: Path = typer.Option(DEFAULT_OUT_DIR / "DETECTOR_V4_3_PAIR_BUNDLE.json", "--bundle"),
    eval_key: Path = typer.Option(
        None,
        "--eval-key",
        help="PROTECTED_STORE_EVAL_KEY.json. Required for --pass reference, refused for blind.",
    ),
    pins: Path = typer.Option(DEFAULT_OUT_DIR / JUDGE_PINS_FILENAME, "--pins"),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir"),
    device: str = typer.Option("cuda:0", "--device"),
    limit: int = typer.Option(0, "--limit", help="judge at most N rows. 0 means all."),
    run_id: str = typer.Option("", "--run-id", help="names a non-reportable run."),
    non_reportable: bool = typer.Option(
        False,
        "--non-reportable",
        help="a smoke run. Namespaces every output file so it cannot occupy a reportable name.",
    ),
    fake_model: bool = typer.Option(False, "--fake-model", help="load nothing; CPU harness test."),
    close: bool = typer.Option(False, "--close", help="freeze the pass. Refuses if incomplete."),
) -> None:
    """Run one judging pass with one local model."""
    if judge not in LOCAL_JUDGE_ROSTER:
        raise typer.BadParameter(f"--judge must be one of {sorted(LOCAL_JUDGE_ROSTER)}")
    if pass_name not in PASSES:
        raise typer.BadParameter(f"--pass must be one of {list(PASSES)}")
    if limit and not non_reportable:
        raise typer.BadParameter(
            "--limit without --non-reportable would write the REPORTABLE filenames with a "
            "partial pass in them. Pass --non-reportable --run-id <name> for a smoke, or "
            "drop --limit for the real run."
        )
    if non_reportable and not run_id:
        raise typer.BadParameter("--non-reportable needs --run-id to name its files")

    out = Path(out_dir)
    names = output_names(
        judge=judge, pass_name=pass_name, run_id=run_id, reportable=not non_reportable
    )
    partial_path = out / names["partial"]
    output_path = out / names["output"]

    source = input_path if input_path is not None else bundle
    rows, provenance = _load_input(source, limit)

    if close:
        return _close(partial_path, output_path, rows=rows, judge=judge, pass_name=pass_name)

    # ------------------------------------------------- pass separation, enforced --
    reference_of: dict[str, str] = {}
    if pass_name == "reference":
        if eval_key is None:
            raise typer.BadParameter("--pass reference needs --eval-key")
        blind_names = output_names(
            judge=judge, pass_name="blind", run_id=run_id, reportable=not non_reportable
        )
        if not (out / blind_names["output"]).exists():
            raise typer.BadParameter(
                f"{out / blind_names['output']} does not exist, so judge {judge}'s blind "
                "pass has not been closed. The reference pass runs only after the blind "
                "labels are frozen: a reference label produced first could have been "
                "revised in the light of the answer, and the two passes would no longer be "
                "independent annotations."
            )
        key = json.loads(Path(eval_key).read_text(encoding="utf-8"))
        reference_of = {
            audit_id: str(entry.get("reference_answer", ""))
            for audit_id, entry in key.get("rows", {}).items()
        }
    elif eval_key is not None:
        raise typer.BadParameter(
            "--eval-key is refused for --pass blind. The blind pass must not be able to "
            "reach a reference answer, and the way that is guaranteed is that the file is "
            "never opened."
        )

    pin = LOCAL_JUDGE_ROSTER[judge] if fake_model else load_judge_pins(pins, judge)
    generate, measure_tokens, described = _build_generator(pin, fake=fake_model, device=device)

    started = time.time()
    counts = run_rows(
        rows,
        generate,
        partial_path=partial_path,
        role=judge,
        pass_name=pass_name,
        reference_of=reference_of,
        measure_tokens=measure_tokens,
    )

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
        n_truncated=counts["n_truncated"],
        seconds=round(time.time() - started, 1),
        extra={
            "pass": pass_name,
            "run_id": run_id or None,
            "reportable": not (fake_model or non_reportable),
            "source": provenance,
            "partial_file": str(partial_path),
            "fake_model": bool(fake_model),
            "pins_file": None if fake_model else str(pins),
            "counts": counts,
            "pin": pin.to_dict(),
            "generator": described,
        },
    )
    atomic_json(out / names["run"], record.to_dict())
    typer.echo(dumps_canonical(record.to_dict()))


def _close(
    partial_path: Path,
    output_path: Path,
    *,
    rows: Sequence[Mapping],
    judge: str,
    pass_name: str = "blind",
) -> None:
    """Freeze a complete pass. Refuses an incomplete, malformed or truncated one."""
    judged = _read_jsonl(partial_path)
    by_id = {str(r["audit_id"]): r for r in judged}
    missing = [str(r["audit_id"]) for r in rows if str(r["audit_id"]) not in by_id]
    malformed = sorted(k for k, v in by_id.items() if v.get("malformed"))
    truncated = sorted(k for k, v in by_id.items() if v.get("prompt_truncated"))
    if missing or malformed or truncated:
        raise typer.BadParameter(
            f"cannot close judge {judge} {pass_name}: {len(missing)} unjudged, "
            f"{len(malformed)} malformed, {len(truncated)} truncated. The label gate "
            "requires zero of each, and a closed file that quietly omitted them would "
            f"satisfy the gate by having fewer rows. First missing {missing[:3]}, first "
            f"malformed {malformed[:3]}, first truncated {truncated[:3]}."
        )
    ordered = [by_id[str(r["audit_id"])] for r in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(dumps_canonical(r) + "\n" for r in ordered), encoding="utf-8")
    typer.echo(
        dumps_canonical(
            {
                "closed": str(output_path),
                "judge": judge,
                "pass": pass_name,
                "n_rows": len(ordered),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "n_malformed": 0,
                "n_truncated": 0,
            }
        )
    )
