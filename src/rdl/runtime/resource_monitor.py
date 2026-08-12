"""Throughput and GPU accounting for the run's PERFORMANCE.json.

Best-effort and never raises: a missing torch or a driver that will not report memory
must not fail a finished run. Recorded because "did the 3090 plan actually hold" is a
question the next run has to answer from evidence rather than memory.
"""

from __future__ import annotations

import time

__all__ = ["ResourceMonitor"]


class ResourceMonitor:
    def __init__(self, *, clock=time.perf_counter) -> None:
        self._clock = clock
        self.started = self._clock()
        self.generation_seconds = 0.0
        self.n_generations = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.peak_vram_bytes: int | None = None

    def record_batch(
        self, *, seconds: float, n: int, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        self.generation_seconds += max(0.0, seconds)
        self.n_generations += n
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self._sample_vram()

    def _sample_vram(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                peak = int(torch.cuda.max_memory_allocated())
                self.peak_vram_bytes = max(self.peak_vram_bytes or 0, peak)
        except Exception:
            return

    def to_dict(self) -> dict:
        wall = max(1e-9, self._clock() - self.started)
        gen = max(1e-9, self.generation_seconds)
        return {
            "wall_seconds": round(wall, 3),
            "generation_seconds": round(self.generation_seconds, 3),
            "n_generations": self.n_generations,
            "generations_per_second": round(self.n_generations / gen, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "completion_tokens_per_second": round(self.completion_tokens / gen, 3),
            "peak_vram_bytes": self.peak_vram_bytes,
        }
