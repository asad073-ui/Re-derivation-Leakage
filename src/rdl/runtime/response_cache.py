"""Reuse of identical generations across arms.

The five arms share a great deal of work — SA and every multi-agent arm ask the root
node the same question with the same seed — and re-decoding it is pure cost. But a
cache is also the easiest way to destroy an experiment: serve one arm a response that
was produced under a different prompt and the arms stop differing only in the defence.

Hence the key covers **everything that could change the output**: model revision,
tokenizer revision, the complete serialized prompt (system message included), every
decoding parameter, the seed, and the backend and its version. A defence that rewrote
the prompt therefore gets a different key automatically.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator

from ..logging_utils import dumps_canonical
from .backend import GenRequest, GenResponse

__all__ = ["ResponseCache", "cache_key"]


def cache_key(
    request: GenRequest,
    *,
    model_revision: str | None,
    tokenizer_revision: str | None,
    backend: str,
    backend_version: str,
) -> str:
    payload = {
        "backend": backend,
        "backend_version": backend_version,
        "model": request.model,
        "model_revision": model_revision,
        "tokenizer_revision": tokenizer_revision,
        "system": request.system,
        "prompt": request.prompt,
        "decoding": request.decoding(),
        "seed": request.seed,
    }
    return hashlib.sha256(dumps_canonical(payload).encode("utf-8")).hexdigest()


class ResponseCache:
    """In-memory, per-run. Deliberately not persisted across runs.

    A cross-run cache would let a response generated under yesterday's checkpoint
    survive into today's manifest, and the manifest is what a reviewer trusts.
    """

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._entries: dict[str, GenResponse] = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> GenResponse | None:
        if not self.enabled:
            return None
        hit = self._entries.get(key)
        if hit is None:
            self.misses += 1
            return None
        self.hits += 1
        return GenResponse(
            request_id=hit.request_id,
            text=hit.text,
            backend=hit.backend,
            model_revision=hit.model_revision,
            cached=True,
            prompt_tokens=hit.prompt_tokens,
            completion_tokens=hit.completion_tokens,
        )

    def put(self, key: str, response: GenResponse) -> None:
        if self.enabled:
            self._entries.setdefault(key, response)

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses, "entries": len(self._entries)}

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)
