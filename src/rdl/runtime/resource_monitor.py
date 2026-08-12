"""Throughput and GPU accounting for the run's PERFORMANCE.json.

Best-effort and never raises: a missing torch or a driver that will not report memory
must not fail a finished run. Recorded because "did the 3090 plan actually hold" is a
question the next run has to answer from evidence rather than memory.

**Why NVML and not just ``torch.cuda.max_memory_allocated``** (GU-0025). That counter
reports live tensor bytes in *this process's torch caching allocator*. Under the vLLM
backend it is close to meaningless: vLLM pre-reserves the KV cache to
``gpu_memory_utilization`` of the card, builds CUDA graph pools, and keeps activation
workspaces outside the caching allocator's accounting, and on ``tensor_parallel_size>1``
the weights live in worker processes torch cannot see from here at all. A 3090 readiness
decision taken from that number would read "4 GB peak" on a card that is in fact 20 GB
full and one long prompt away from an OOM.

So the device-level figure is the one that gates the run, sampled in this order:

  1. **NVML** (``pynvml`` / ``nvidia-ml-py``, which vLLM already depends on). Gives both
     whole-device used bytes and, via the compute-process list, the bytes attributable to
     *this* PID and its vLLM workers.
  2. **``torch.cuda.mem_get_info``**, the driver's own ``cudaMemGetInfo``. Whole-device
     only, no per-process attribution, but needs no extra package.
  3. Nothing. The fields stay ``None`` rather than reporting a torch-allocator number
     under a device-level name.

The torch-allocator peak is still recorded, under a name that says what it is.
"""

from __future__ import annotations

import os
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
        # This process's torch caching allocator. Kept for continuity; NOT the gate.
        self.peak_vram_bytes: int | None = None
        # The whole card, which is what a 24 GB budget is a claim about.
        self.peak_device_vram_bytes: int | None = None
        # This PID and its vLLM workers, when NVML can attribute it.
        self.peak_process_vram_bytes: int | None = None
        self.device_total_vram_bytes: int | None = None
        self.device_name: str | None = None
        self.vram_source: str = "unsampled"
        self._nvml: object | None = None
        self._nvml_state: str = "untried"

    def record_batch(
        self, *, seconds: float, n: int, prompt_tokens: int = 0, completion_tokens: int = 0
    ) -> None:
        self.generation_seconds += max(0.0, seconds)
        self.n_generations += n
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.sample_vram()

    # ----------------------------------------------------------------------- vram --

    def sample_vram(self) -> None:
        """One device-memory sample. Never raises."""
        self._sample_torch_allocator()
        if self._sample_nvml():
            return
        self._sample_driver()

    def _sample_torch_allocator(self) -> None:
        try:
            import torch

            if torch.cuda.is_available():
                peak = int(torch.cuda.max_memory_allocated())
                self.peak_vram_bytes = max(self.peak_vram_bytes or 0, peak)
        except Exception:
            return

    def _load_nvml(self) -> object | None:
        if self._nvml_state != "untried":
            return self._nvml
        try:
            import pynvml  # nvidia-ml-py; a vLLM dependency on the GPU box

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_state = "ready"
        except Exception as exc:
            self._nvml = None
            self._nvml_state = f"unavailable: {type(exc).__name__}"
        return self._nvml

    def _visible_indices(self) -> list[int]:
        """NVML indexes physical cards; ``CUDA_VISIBLE_DEVICES`` renumbers them.

        Sampling NVML device 0 while the run is pinned to card 3 would report an idle
        card as the run's memory, so the mapping is applied explicitly. Non-integer
        entries (UUIDs) are skipped rather than guessed at.
        """
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not visible or not visible.strip():
            return []
        out: list[int] = []
        for part in visible.split(","):
            part = part.strip()
            if part.isdigit():
                out.append(int(part))
        return out

    def _sample_nvml(self) -> bool:
        pynvml = self._load_nvml()
        if pynvml is None:
            return False
        try:
            count = int(pynvml.nvmlDeviceGetCount())  # type: ignore[attr-defined]
            indices = [i for i in self._visible_indices() if i < count] or list(range(count))
            pids = {os.getpid(), *_child_pids()}
            device_used = 0
            device_total = 0
            process_used = 0
            names: list[str] = []
            for index in indices:
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)  # type: ignore[attr-defined]
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]
                device_used += int(info.used)
                device_total += int(info.total)
                names.append(_decode(pynvml.nvmlDeviceGetName(handle)))  # type: ignore[attr-defined]
                try:
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)  # type: ignore[attr-defined]
                except Exception:
                    procs = []
                for proc in procs:
                    used = getattr(proc, "usedGpuMemory", None)
                    if used is not None and int(getattr(proc, "pid", -1)) in pids:
                        process_used += int(used)
        except Exception:
            self._nvml_state = "sample_failed"
            return False
        self.peak_device_vram_bytes = max(self.peak_device_vram_bytes or 0, device_used)
        self.device_total_vram_bytes = device_total or None
        if process_used:
            self.peak_process_vram_bytes = max(self.peak_process_vram_bytes or 0, process_used)
        self.device_name = ", ".join(sorted(set(names))) or None
        self.vram_source = "nvml"
        return True

    def _sample_driver(self) -> bool:
        """``cudaMemGetInfo`` through torch: whole device, no per-process split."""
        try:
            import torch

            if not torch.cuda.is_available():
                return False
            free, total = torch.cuda.mem_get_info()
            used = int(total) - int(free)
            self.peak_device_vram_bytes = max(self.peak_device_vram_bytes or 0, used)
            self.device_total_vram_bytes = int(total)
            self.device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        except Exception:
            return False
        self.vram_source = "torch.cuda.mem_get_info"
        return True

    # --------------------------------------------------------------------- report --

    def to_dict(self) -> dict:
        wall = max(1e-9, self._clock() - self.started)
        gen = max(1e-9, self.generation_seconds)
        total = self.device_total_vram_bytes
        peak = self.peak_device_vram_bytes
        return {
            "wall_seconds": round(wall, 3),
            "generation_seconds": round(self.generation_seconds, 3),
            "n_generations": self.n_generations,
            "generations_per_second": round(self.n_generations / gen, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "completion_tokens_per_second": round(self.completion_tokens / gen, 3),
            # Named for what it is: this process's torch caching allocator, which under
            # vLLM does NOT include the KV cache reservation or the CUDA graph pools.
            "peak_torch_allocator_bytes": self.peak_vram_bytes,
            "peak_vram_bytes": self.peak_vram_bytes,
            "peak_device_vram_bytes": peak,
            "peak_process_vram_bytes": self.peak_process_vram_bytes,
            "device_total_vram_bytes": total,
            "device_vram_utilization": (round(peak / total, 4) if peak and total else None),
            "device_name": self.device_name,
            "vram_source": self.vram_source,
            "nvml_state": self._nvml_state,
        }


def _decode(value: object) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _child_pids() -> set[int]:
    """Direct children of this process — vLLM's workers under tensor parallelism.

    ``psutil`` is not a dependency, so its absence is normal and silent; the whole-device
    figure still holds, only the per-process attribution narrows.
    """
    try:
        import psutil

        return {c.pid for c in psutil.Process().children(recursive=True)}
    except Exception:
        return set()
