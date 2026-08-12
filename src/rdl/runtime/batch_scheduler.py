"""Batching, cache lookup and dispatch for one graph depth.

The scheduler sits between the executor and the backend so that neither has to know
about the other's constraints: the executor thinks in graph layers, the backend thinks
in `max_num_seqs`.

**Counters are per purpose** (GU-0028). One total conflated the graph's own model calls
with the later-episode readback probes, and the run manifest then published that total
under the name ``actual_graph_generations``. A 2x1 preflight planning 42 graph
generations and 20 probes reported "27 model calls dispatched", which is neither number
and cannot be checked against either. Every dispatch is now attributed to the purpose
that asked for it:

    requested    prompts handed to the scheduler
    dispatched   prompts that reached the backend
    cache_hits   prompts answered from the response cache

``requested == dispatched + cache_hits`` per purpose, which is what makes the pair
verifiable from outside.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .backend import GenerationBackend, GenRequest, GenResponse
from .response_cache import ResponseCache, cache_key

__all__ = ["PURPOSES", "BatchScheduler"]

# The two things a graph run generates. `graph` is the agents' own calls; `probe` is the
# later-episode readback, which is measurement rather than execution.
PURPOSES: tuple[str, ...] = ("graph", "probe")


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
        self.by_purpose: dict[str, dict[str, int]] = {
            purpose: {"requested": 0, "dispatched": 0, "cache_hits": 0} for purpose in PURPOSES
        }

    def key_for(self, request: GenRequest) -> str:
        return cache_key(
            request,
            model_revision=self.backend.model_revision(request.model),
            tokenizer_revision=self.tokenizer_revision,
            backend=self.backend.name,
            backend_version=self.backend_version,
        )

    def run(
        self, requests: Sequence[GenRequest], *, purpose: str = "graph"
    ) -> dict[str, GenResponse]:
        """Resolve a whole layer. Returns ``{request_id: response}``.

        ``purpose`` attributes the batch to ``graph`` (the agents' own calls) or ``probe``
        (the later-episode readback). It only affects counters, and it is a hard error
        rather than a default so that a new call site cannot quietly land in the wrong
        bucket.

        Duplicate request ids in one call are a programming error, not a merge
        opportunity: two graph nodes that produced the same id would silently share an
        output and one node's generation would vanish from the evidence.
        """
        if purpose not in self.by_purpose:
            raise ValueError(f"unknown generation purpose '{purpose}'; expected {PURPOSES}")
        counters = self.by_purpose[purpose]
        counters["requested"] += len(requests)

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
                counters["cache_hits"] += 1
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
            counters["dispatched"] += len(chunk)
            for request, response in zip(chunk, responses, strict=True):
                if response.request_id != request.request_id:
                    raise RuntimeError(
                        "backend returned responses out of order: expected "
                        f"{request.request_id}, got {response.request_id}"
                    )
                self.cache.put(keys[request.request_id], response)
                out[request.request_id] = response
        return out

    def stats(self) -> dict:
        """Per-purpose counters, plus the flat totals earlier runs recorded.

        The nested shape is the one to read. The flat ``dispatched`` /
        ``served_from_cache`` keys are the sum over purposes and are kept so an old
        PERFORMANCE.json and a new one can still be compared; nothing should gate on
        them, because a total cannot tell an idle graph with busy probes from the
        reverse.
        """
        purposes = {purpose: dict(counts) for purpose, counts in self.by_purpose.items()}
        return {
            **purposes,
            "total": {
                key: sum(counts[key] for counts in purposes.values())
                for key in ("requested", "dispatched", "cache_hits")
            },
            "dispatched": self.dispatched,
            "served_from_cache": self.served_from_cache,
            **self.cache.stats(),
        }
