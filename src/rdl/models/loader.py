"""The single place a real model is loaded.

Design invariant 2: no bare `AutoModelForCausalLM.from_pretrained` exists anywhere else
in this package. Everything that differs between a T4, a 3090, and an H100 is decided
here and nowhere else.

The four traps this file exists to close, in order of how quietly they corrupt results:

1. **dtype.** T4 has no bf16. A config that asks for bf16 must be overridden to fp16
   for eval. The decision comes from `HardwareProfile`, never from the YAML, unless the
   YAML sets an explicit `dtype_override`.

2. **attn_implementation.** FlashAttention-2 is SM80+. Requesting it on a T4 either
   fails to import or falls back silently depending on the transformers version.
   Same rule: `HardwareProfile` decides.

3. **padding_side.** Left-pad for generation, right-pad for scoring. Getting this
   backwards does not crash — it silently degrades ES/EM. We set it explicitly per
   call and assert it.

4. **chat template.** The TOFU checkpoints are `-Instruct` derivatives. If our prompt
   format does not match what open-unlearning used, the numbers will not reproduce and
   the mismatch looks like an unlearning effect. We assert the template exists.

Related determinism note: batched generation with left-padding changes greedy outputs
under fp16. `batch_size=1` for any number that goes in the paper — this is a real
reproducibility trap, not defensive paranoia.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

from ..config import ModelConfig
from ..hardware import HardwareProfile
from ..logging_utils import get_logger
from .registry import resolve
from .stub import GenerationRequest, LMHandle, StubLM

__all__ = [
    "ChatTemplateMissingError",
    "HFLMHandle",
    "clean_leakk_official_output",
    "leakk_official_inputs",
    "load_lm",
    "resolve_attn",
    "resolve_dtype",
]

log = get_logger(__name__)


class ChatTemplateMissingError(RuntimeError):
    """The tokenizer has no chat template but the model config requires one."""


LEAKK_OFFICIAL_GENERATE_KWARGS: dict[str, Any] = {
    "max_new_tokens": 200,
    "do_sample": True,
    "num_return_sequences": 1,
    "top_p": 1.0,
    "temperature": 1.0,
}


def leakk_official_inputs(tokenizer: Any, question: str) -> tuple[str, list[int], list[int]]:
    """Reproduce Leak-k e544af6's evaluator prompt round-trip exactly.

    The released evaluator turns its template token IDs back into text with special
    tokens removed, then tokenizes that text again for generation.  Returning both
    ID sequences lets the network golden test detect a drift in either side.
    """
    template_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
        ],
        tokenize=True,
        add_generation_prompt=True,
        date_string="10 Apr 2025",
    )
    serialized = tokenizer.decode(
        template_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    retokenized_ids = tokenizer([serialized], padding=True, add_special_tokens=True)["input_ids"][0]
    return serialized, list(template_ids), list(retokenized_ids)


def clean_leakk_official_output(serialized_prompt: str, decoded_output: str) -> str:
    """Apply the released evaluator's full-prompt removal rule."""
    return (
        decoded_output.split(serialized_prompt)[-1].strip()
        if serialized_prompt in decoded_output
        else decoded_output.strip()
    )


_DTYPES = ("float32", "float16", "bfloat16")


def resolve_dtype(model_cfg: ModelConfig, hw: HardwareProfile, *, training: bool = False) -> str:
    """Decide the dtype. Hardware wins unless the config sets an explicit override."""
    if model_cfg.dtype_override is not None:
        if model_cfg.dtype_override == "bfloat16" and not hw.supports_bf16:
            raise ValueError(
                f"model '{model_cfg.name}' sets dtype_override=bfloat16 but "
                f"{hw.name} (cc={hw.compute_capability}) has no bf16 support. "
                "Remove the override and let hardware.py choose float16."
            )
        return model_cfg.dtype_override
    return hw.recommended_train_dtype if training else hw.recommended_dtype


def resolve_attn(model_cfg: ModelConfig, hw: HardwareProfile) -> str:
    """Decide the attention implementation. Hardware wins unless explicitly overridden."""
    if model_cfg.attn_override is not None:
        if model_cfg.attn_override == "flash_attention_2" and not hw.supports_flash_attn2:
            raise ValueError(
                f"model '{model_cfg.name}' sets attn_override=flash_attention_2 but "
                f"{hw.name} (cc={hw.compute_capability}) is pre-Ampere. "
                "FlashAttention-2 requires SM80+. Use 'sdpa'."
            )
        return model_cfg.attn_override
    return hw.recommended_attn


class HFLMHandle(LMHandle):
    """Thin wrapper over (model, tokenizer). Three methods, as per `LMHandle`."""

    def __init__(self, model: Any, tokenizer: Any, *, model_id: str, dtype: str, attn: str) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.dtype = dtype
        self.attn = attn
        self._closed = False
        self._last_generation_provenance: dict[str, Any] = {}

    # ---------------------------------------------------------------- prompting --

    def _apply_chat_template(self, prompt: str, system: str | None = None) -> str:
        """Render the user turn through the model's own chat template.

        The TOFU finetunes are -Instruct derivatives; the prompt format must match what
        open-unlearning used or the numbers will not reproduce.
        """
        tok = self.tokenizer
        if getattr(tok, "chat_template", None) is None:
            return prompt
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # --------------------------------------------------------------- generation --

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        *,
        system: str | None = None,
        apply_template: bool = True,
        request: GenerationRequest | None = None,
    ) -> str:
        import torch

        text = self._apply_chat_template(prompt, system) if apply_template else prompt
        req = request or GenerationRequest()

        # LEFT pad for generation. Asserted, not assumed.
        self.tokenizer.padding_side = "left"
        assert self.tokenizer.padding_side == "left", "generation requires left padding"

        enc = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        self._last_generation_provenance = {
            "semantic_user_prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "serialized_chat_prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "input_ids_sha256": hashlib.sha256(
                enc["input_ids"].detach().cpu().numpy().tobytes()
            ).hexdigest(),
        }
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": req.do_sample,
            "num_beams": 1,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if req.do_sample:
            if req.temperature is not None:
                kwargs["temperature"] = req.temperature
            if req.top_p is not None:
                kwargs["top_p"] = req.top_p
            if req.top_k is not None:
                kwargs["top_k"] = req.top_k
        else:
            kwargs.update({"temperature": None, "top_p": None, "top_k": None})

        # ``transformers.GenerationMixin.generate`` does not support a public
        # per-call ``generator`` kwarg across the pinned versions this project runs.
        # Passing it is rejected as an unused model kwarg on common releases.  Scope
        # the global RNG instead, restoring it immediately afterwards so a sampled
        # trajectory is reproducible without leaking its seed into the next call.
        devices = [self.model.device.index] if self.model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices, enabled=req.seed is not None):
            if req.seed is not None:
                torch.manual_seed(req.seed)
                if self.model.device.type == "cuda":
                    torch.cuda.manual_seed_all(req.seed)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    **kwargs,
                )

        gen_ids = out[0][enc["input_ids"].shape[-1] :]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    def generate_leakk_official(
        self, question: str, max_new_tokens: int, *, seed: int | None = None
    ) -> str:
        """Execute the released evaluator's prompt round-trip and decoding call.

        Leak-k first decodes the OpenUnlearning evaluator input IDs with special
        tokens removed, then tokenizes that resulting string again before
        ``model.generate``.  This apparently redundant route is part of its released
        distribution and is intentionally not shared with the agent chat path.
        """
        import torch

        serialized, template_ids, retokenized_ids = leakk_official_inputs(self.tokenizer, question)
        self.tokenizer.padding_side = "left"
        enc = self.tokenizer(
            [serialized], return_tensors="pt", padding=True, add_special_tokens=True
        )
        self._last_generation_provenance = {
            "semantic_user_prompt_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
            "serialized_chat_prompt_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
            "input_ids_sha256": hashlib.sha256(
                enc["input_ids"].detach().cpu().numpy().tobytes()
            ).hexdigest(),
            "leakk_official": {
                "semantic_prompt": question,
                "serialized_prompt": serialized,
                "template_input_ids": template_ids,
                "input_ids": retokenized_ids,
                "generate_kwargs": {
                    "pad_token_id": self.tokenizer.eos_token_id,
                    **LEAKK_OFFICIAL_GENERATE_KWARGS,
                    "max_new_tokens": max_new_tokens,
                },
            },
        }
        enc = {key: value.to(self.model.device) for key, value in enc.items()}
        kwargs = self._last_generation_provenance["leakk_official"]["generate_kwargs"]
        devices = [self.model.device.index] if self.model.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices, enabled=seed is not None):
            if seed is not None:
                torch.manual_seed(seed)
                if self.model.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], **kwargs
                )
        decoded = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return clean_leakk_official_output(serialized, decoded)

    def generation_provenance(self) -> dict[str, Any]:
        return dict(self._last_generation_provenance)

    def logprobs(
        self,
        prompt: str,
        continuation: str,
        *,
        system: str | None = None,
        apply_template: bool = True,
    ) -> Any:
        """Per-token logprobs of `continuation` given `prompt`. Returns a torch.Tensor."""
        import torch

        text = self._apply_chat_template(prompt, system) if apply_template else prompt

        # RIGHT pad for scoring. The opposite of generation, and asserted for the same
        # reason: getting it wrong degrades ES/EM without ever raising.
        self.tokenizer.padding_side = "right"
        assert self.tokenizer.padding_side == "right", "scoring requires right padding"

        prompt_ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ]
        cont_ids = self.tokenizer(continuation, return_tensors="pt", add_special_tokens=False)[
            "input_ids"
        ]
        input_ids = torch.cat([prompt_ids, cont_ids], dim=-1).to(self.model.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids).logits

        n_cont = cont_ids.shape[-1]
        if n_cont == 0:
            return torch.tensor([], device=logits.device)

        # Predicting token t uses logits at position t-1.
        target = input_ids[0, -n_cont:]
        pred = logits[0, -n_cont - 1 : -1, :].float()
        logprobs = torch.log_softmax(pred, dim=-1)
        return logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            import torch

            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:  # pragma: no cover
            pass

    def __repr__(self) -> str:  # pragma: no cover
        return f"HFLMHandle({self.model_id!r}, dtype={self.dtype}, attn={self.attn})"


def load_lm(
    model_cfg: ModelConfig,
    hw: HardwareProfile,
    *,
    token: str | None = None,
    training: bool = False,
    device_map: str | None = None,
) -> LMHandle:
    """Load a model. The ONLY place this happens.

    Returns a `StubLM` for `kind == "stub"`, which is why the whole system is testable
    on CPU with no network.
    """
    if model_cfg.kind == "stub":
        return StubLM(
            answers=model_cfg.stub_answers,
            knowledge_mask=model_cfg.stub_knowledge_mask,
            abstention_text=model_cfg.stub_abstention_text,
            paraphrase_mode=model_cfg.stub_paraphrase_mode,
            scripted_samples=model_cfg.stub_scripted_samples,
            model_id=model_cfg.name,
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    repo_id = resolve(model_cfg.repo_id or model_cfg.name)
    dtype_name = resolve_dtype(model_cfg, hw, training=training)
    attn = resolve_attn(model_cfg, hw)
    torch_dtype = getattr(torch, dtype_name)

    log.info("loading %s dtype=%s attn=%s device=%s", repo_id, dtype_name, attn, hw.device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.tokenizer_repo_id or repo_id,
        revision=model_cfg.revision,
        token=token,
        trust_remote_code=model_cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        # Llama-3.2 ships no pad token. eos is the conventional stand-in and is what
        # open-unlearning uses; inventing a new one would shift the embedding table.
        tokenizer.pad_token = tokenizer.eos_token

    if model_cfg.chat_template_required and getattr(tokenizer, "chat_template", None) is None:
        raise ChatTemplateMissingError(
            f"{repo_id}: tokenizer has no chat_template, but the model config requires "
            "one. The TOFU checkpoints are -Instruct derivatives and the prompt format "
            "must match what open-unlearning used, or the metrics will not reproduce. "
            "Set chat_template_required: false only if you know the eval is raw-text."
        )

    # Annotated Any: transformers' `.to()` overloads do not type-check against a plain
    # device string, and narrowing it here would buy nothing — HFLMHandle is the typed
    # boundary that callers actually see.
    model: Any = AutoModelForCausalLM.from_pretrained(
        repo_id,
        revision=model_cfg.revision,
        torch_dtype=torch_dtype,
        attn_implementation=attn,
        token=token,
        trust_remote_code=model_cfg.trust_remote_code,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    if device_map is None:
        model = model.to("cuda" if hw.is_cuda else "cpu")
    model.eval()

    return HFLMHandle(model, tokenizer, model_id=repo_id, dtype=dtype_name, attn=attn)


def load_many(
    model_cfgs: Sequence[ModelConfig], hw: HardwareProfile, **kwargs: Any
) -> dict[str, LMHandle]:
    """Load several models keyed by config name. Sequential on purpose — a T4 has 16 GB."""
    return {cfg.name: load_lm(cfg, hw, **kwargs) for cfg in model_cfgs}
