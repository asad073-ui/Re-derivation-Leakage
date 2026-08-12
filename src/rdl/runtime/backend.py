"""The generation interface every backend implements.

Batch-first on purpose. The historical runner generated one prompt at a time inside
nested item/sample loops, which is why a 50x32 pilot took a whole GPU session. Here the
unit of work is "every node ready at this graph depth, across every trajectory in
flight", and a backend that can only do one at a time simply loops internally.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

__all__ = ["GenRequest", "GenResponse", "GenerationBackend", "decoding_key"]


@dataclass(frozen=True)
class GenRequest:
    """One generation, fully specified.

    ``seed`` is per request rather than ambient global state: matched arms must be able
    to prove they differed only in the intended intervention, and a global RNG makes
    that unprovable the moment two arms issue a different number of calls.
    """

    request_id: str
    prompt: str
    system: str | None = None
    max_new_tokens: int = 128
    seed: int = 0
    do_sample: bool = True
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    top_k: int | None = 0
    model: str = "primary"

    def decoding(self) -> dict[str, object]:
        return {
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "max_new_tokens": self.max_new_tokens,
        }


@dataclass(frozen=True)
class GenResponse:
    request_id: str
    text: str
    backend: str = "unknown"
    model_revision: str | None = None
    cached: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


def decoding_key(request: GenRequest) -> str:
    """Hash of everything that shapes the distribution, excluding the seed.

    The seed is provenance for one draw; the decoding key is what two arms must share
    before their Leak@k numbers may be compared at all.
    """
    return hashlib.sha256(
        json.dumps(request.decoding(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@runtime_checkable
class GenerationBackend(Protocol):
    name: str

    def generate(self, requests: Sequence[GenRequest]) -> list[GenResponse]:
        """Generate for a batch, returning responses in the same order as `requests`."""
        ...

    def model_revision(self, model: str) -> str | None:
        """Identity of the physical checkpoint serving `model`. Goes in the manifest."""
        ...

    def close(self) -> None: ...
