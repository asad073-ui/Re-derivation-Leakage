"""Model loading.

Design invariant 2: **every model call goes through `loader.load_lm`.** There is no
bare `AutoModelForCausalLM.from_pretrained` anywhere else in this package. That single
choke point is what makes the T4 / 3090 / H100 split survivable — dtype, attention
implementation, padding side, and chat templating are decided in exactly one file.

`stub.StubLM` implements the same `LMHandle` interface with no weights and no network,
which is what makes the whole system testable on a laptop.
"""

from __future__ import annotations

from .registry import KNOWN_MODELS, resolve, verify_exists
from .stub import LMHandle, StubLM

__all__ = ["KNOWN_MODELS", "LMHandle", "StubLM", "load_lm", "resolve", "verify_exists"]


def load_lm(*args, **kwargs):
    """Lazy re-export so importing `rdl.models` does not require transformers/torch."""
    from .loader import load_lm as _load_lm

    return _load_lm(*args, **kwargs)
