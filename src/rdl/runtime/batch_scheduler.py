"""Batching, cache lookup and dispatch for one graph depth.

The scheduler sits between the executor and the backend so that neither has to know
about the other's constraints: the executor thinks in graph layers, the backend thinks
in `max_num_seqs`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .backend import GenerationBackend, GenRequest, GenResponse
from .response_cache import ResponseCache, cache_key

__all__ = ["BatchScheduler"]


class BatchScheduler:
    def __init__(
        self,
        backend: GenerationBackend,
        *,
        cache: ResponseCache | None = None,
        max_batch_size: int = 16,
        tokenizer_revision: str | None = None,
        backend_version: str = "0",
        observer: Callable[[str, str], None] | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        self.backend = backend
        self.cache = cache if cache is not None else ResponseCache()
        self.max_batch_size = max_batch_size
        self.tokenizer_revision = tokenizer_revision
        self.backend_version = backend_version
        # Told which arm is running, so the response bank can record which arms shared a
        # cache entry. That record is the evidence that reuse was byte-identical.
        self.observer = observer
        self.tag = ""
        self.dispatched = 0
        self.served_from_cache = 0

    def key_for(self, request: GenRequest) -> str:
        return cache_key(
            request,
            model_revision=self.backend.model_revision(request.model),
            tokenizer_revision=self.tokenizer_revision,
            backend=self.backend.name,
            backend_version=self.backend_version,
        )

    def run(self, requests: Sequence[GenRequest]) -> dict[str, GenResponse]:
        """Resolve a whole layer. Returns ``{request_id: response}``.

        Duplicate request ids in one call are a programming error, not a merge
        opportunity: two graph nodes that produced the same id would silently share an
        output and one node's generation would vanish from the evidence.
        """
        ids = [r.request_id for r in requests]
        if len(set(ids)) != len(ids):
            duplicates = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate request ids in one batch: {duplicates}")

        out: dict[str, GenResponse] = {}
        pending: list[GenRequest] = []
        keys: dict[str, str] = {}
        for request in requests:
            key = self.key_for(request)
            keys[request.request_id] = key
            if self.observer is not None:
                self.observer(key, self.tag)
            hit = self.cache.get(key)
            if hit is not None:
                self.served_from_cache += 1
                out[request.request_id] = GenResponse(
                    request_id=request.request_id,
                    text=hit.text,
                    backend=hit.backend,
                    model_revision=hit.model_revision,
                    cached=True,
                    prompt_tokens=hit.prompt_tokens,
                    completion_tokens=hit.completion_tokens,
                )
            else:
                pending.append(request)

        for start in range(0, len(pending), self.max_batch_size):
            chunk = pending[start : start + self.max_batch_size]
            responses = self.backend.generate(chunk)
            if len(responses) != len(chunk):
                raise RuntimeError(
                    f"backend '{self.backend.name}' returned {len(responses)} responses "
                    f"for {len(chunk)} requests"
                )
            self.dispatched += len(chunk)
            for request, response in zip(chunk, responses, strict=True):
                if response.request_id != request.request_id:
                    raise RuntimeError(
                        "backend returned responses out of order: expected "
                        f"{request.request_id}, got {response.request_id}"
                    )
                self.cache.put(keys[request.request_id], response)
                out[request.request_id] = response
        return out

    def stats(self) -> dict[str, int]:
        return {
            "dispatched": self.dispatched,
            "served_from_cache": self.served_from_cache,
            **self.cache.stats(),
        }
