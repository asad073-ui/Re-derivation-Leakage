"""Restore fp32 `output.logits`, which transformers 4.46 stopped providing.

THE DEFECT
----------
`third_party/open-unlearning/src/evals/metrics/utils.py` computes its own
cross-entropy from `output.logits` and then converts the result with
`.cpu().numpy()`:

    losses      = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
    avg_losses  = losses / num_token_gt
    avg_losses  = avg_losses.cpu().numpy().tolist()     # <-- line 98

`Tensor.numpy()` has no bfloat16 dtype to convert into, so when `logits` is bf16 this
raises `TypeError: Got unsupported ScalarType BFloat16` and the eval dies partway
through the first metric. `src/evals/metrics/utility.py:62` does the same thing to
`F.softmax(outputs.logits, ...)` and would fail immediately afterwards.

WHY IT ONLY APPEARS NOW
-----------------------
Up to and including transformers 4.45.1, `LlamaForCausalLM.forward` ended with an
unconditional ``logits = logits.float()``. Every consumer therefore saw fp32 logits no
matter what dtype the weights were in, and upstream's code was correct by accident.
transformers 4.46 removed that upcast to save memory.

The dates line up exactly, and they are the whole argument:

    2025-07-20  open-unlearning docs/repro.md last updated  (transformers==4.45.1)
    (4.46)      transformers removes the logits.float() upcast
    2026-03-07  open-unlearning a456aa2 bumps the pin to transformers==4.51.3

That bump touched `src/trainer/*`, `requirements.txt` and the docs. It did not touch
`src/evals/` at all, so the evaluation path was never re-run against the new pin. The
pinned SHA `4ad738a` is upstream HEAD, so there is no upstream fix to move to.

The consequence: open-unlearning's DEFAULT eval configuration
(`configs/model/Llama-3.2-1B-Instruct.yaml` pins `torch_dtype: bfloat16`) cannot run
at its OWN pinned dependency set. This is not specific to our bridge, our overrides or
an RTX 3090 — it reproduces with a 1 MB random Llama on CPU.

WHAT THIS SHIM DOES
-------------------
Re-applies exactly the cast transformers used to apply: the logits returned by the
causal-LM forward pass are upcast to fp32 before anything downstream sees them.

That is the point. It is not "a cast that makes the crash go away" — casting at
line 98 instead would leave upstream's cross-entropy running in bf16, which has about
three decimal digits of mantissa, and `exp(-avg_loss)` would then differ from the
published probabilities by far more than the +/-0.01 gate tolerance. Upcasting at the
source reproduces the numerical environment the published numbers were produced in:
the loss, the softmax and the truth ratios are all computed in fp32, exactly as they
were under 4.45.1.

The model WEIGHTS stay bf16 and FlashAttention-2 stays on, so `torch_dtype=bfloat16`
and `attn_implementation=flash_attention_2` remain true of the run and
`published_parity` is not compromised.

Cost: the logits tensor is materialised in fp32, doubling its footprint. That is the
memory transformers 4.46 was reclaiming, and it is what 4.45.1 spent. TOFU sequences
are short and it fits comfortably in 24 GB at batch 32.
"""

from __future__ import annotations

from typing import Any

__all__ = ["ShimNotApplied", "install", "is_installed"]

_INSTALLED: dict[str, Any] = {}


class ShimNotApplied(RuntimeError):
    """Raised when the shim cannot be applied.

    Deliberately fatal. The failure mode this whole module exists to prevent is a run
    that quietly produces numbers under conditions its report does not describe, and a
    shim that failed to attach but let the process continue would be precisely that.
    """


def is_installed() -> bool:
    return bool(_INSTALLED)


def install() -> dict[str, Any]:
    """Patch causal-LM forwards to return fp32 logits. Idempotent.

    Returns a small record describing what was patched, for the run report.
    """
    if _INSTALLED:
        return dict(_INSTALLED)

    try:
        import torch
        from transformers.models.llama.modeling_llama import LlamaForCausalLM
    except ImportError as exc:  # pragma: no cover - the eval env always has both
        raise ShimNotApplied(
            f"cannot import the class to patch: {exc}. The fp32-logits shim is required "
            "whenever open-unlearning is evaluated in bfloat16."
        ) from exc

    original = LlamaForCausalLM.forward

    def forward_with_fp32_logits(self, *args: Any, **kwargs: Any) -> Any:
        out = original(self, *args, **kwargs)
        logits = getattr(out, "logits", None)
        if logits is not None and logits.dtype != torch.float32:
            # ModelOutput.__setattr__ keeps the mapping view in sync, so downstream
            # code reaching the field either way sees the upcast tensor.
            out.logits = logits.float()
        return out

    forward_with_fp32_logits.__doc__ = (
        "transformers <= 4.45.1 behaviour: upcast logits to fp32 before returning. "
        "See rdl.compat.fp32_logits."
    )
    # Marked so a second install() through a different import path is still a no-op.
    forward_with_fp32_logits._rdl_fp32_logits_shim = True  # type: ignore[attr-defined]

    if getattr(original, "_rdl_fp32_logits_shim", False):
        _INSTALLED.update({"target": "LlamaForCausalLM.forward", "already": True})
        return dict(_INSTALLED)

    LlamaForCausalLM.forward = forward_with_fp32_logits  # type: ignore[method-assign]

    import transformers

    _INSTALLED.update(
        {
            "shim": "fp32_logits",
            "target": "transformers.models.llama.modeling_llama.LlamaForCausalLM.forward",
            "transformers_version": transformers.__version__,
            "reason": (
                "transformers >= 4.46 removed the unconditional logits.float() upcast; "
                "open-unlearning's eval metrics call .cpu().numpy() on tensors derived "
                "from logits, which has no bfloat16 conversion"
            ),
        }
    )
    return dict(_INSTALLED)
