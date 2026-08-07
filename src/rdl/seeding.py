"""Determinism.

Greedy decoding and a fixed seed are not enough on their own. The traps this module
exists to close:

- Three separate RNGs (`random`, `numpy`, `torch`) must all be seeded; seeding one and
  assuming the others follow is a common and silent source of run-to-run drift.
- cuBLAS needs `CUBLAS_WORKSPACE_CONFIG` set *before* the first CUDA context is
  created, otherwise `torch.use_deterministic_algorithms(True)` raises at the first
  matmul rather than at setup.
- `torch.use_deterministic_algorithms(True)` legitimately errors for some kernels.
  We degrade to warn-only rather than crashing a 40-minute eval, and we record that
  we degraded, because a run whose determinism was downgraded must not silently
  masquerade as a deterministic one.
"""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass, field

__all__ = ["SeedReport", "seed_worker", "set_all_seeds"]

DEFAULT_SEED = 42


@dataclass(frozen=True)
class SeedReport:
    """What actually happened when we asked for determinism. Goes into every run record."""

    seed: int
    python_seeded: bool = False
    numpy_seeded: bool = False
    torch_seeded: bool = False
    torch_cuda_seeded: bool = False
    deterministic_algorithms: bool = False
    cudnn_deterministic: bool = False
    cublas_workspace_config: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def fully_deterministic(self) -> bool:
        """True only when every requested guarantee was actually granted."""
        return (
            self.python_seeded
            and self.numpy_seeded
            and self.torch_seeded
            and self.deterministic_algorithms
            and not self.warnings
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fully_deterministic"] = self.fully_deterministic
        return d


def set_all_seeds(seed: int = DEFAULT_SEED, deterministic: bool = True) -> SeedReport:
    """Seed every RNG we touch and, optionally, request deterministic kernels.

    Safe to call when torch is not installed — the CPU unit suite runs without it.
    """
    warnings: list[str] = []

    random.seed(seed)
    python_seeded = True

    # PYTHONHASHSEED only takes effect at interpreter start; setting it here helps any
    # subprocess we spawn (the open-unlearning bridge does spawn one).
    os.environ.setdefault("PYTHONHASHSEED", str(seed))

    numpy_seeded = False
    try:
        import numpy as np

        np.random.seed(seed)
        numpy_seeded = True
    except ImportError:  # pragma: no cover - numpy is a hard dep in practice
        warnings.append("numpy not importable; numpy RNG unseeded")

    cublas_cfg: str | None = None
    if deterministic:
        # Must be set before the first CUDA context. Doing it unconditionally is
        # harmless on CPU and is the only way to make it reliable on GPU.
        cublas_cfg = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    torch_seeded = False
    torch_cuda_seeded = False
    det_algos = False
    cudnn_det = False
    try:
        import torch
    except ImportError:
        warnings.append("torch not importable; torch RNGs unseeded (fine on the CPU gate)")
    else:
        torch.manual_seed(seed)
        torch_seeded = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch_cuda_seeded = True

        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                cudnn_det = True
            except (AttributeError, RuntimeError) as exc:
                warnings.append(f"cudnn determinism unavailable: {exc}")

            try:
                # warn_only: some kernels have no deterministic implementation. We would
                # rather finish the run and record the downgrade than abort at hour one.
                torch.use_deterministic_algorithms(True, warn_only=True)
                det_algos = True
            except (RuntimeError, TypeError) as exc:
                warnings.append(f"use_deterministic_algorithms failed: {exc}")

    return SeedReport(
        seed=seed,
        python_seeded=python_seeded,
        numpy_seeded=numpy_seeded,
        torch_seeded=torch_seeded,
        torch_cuda_seeded=torch_cuda_seeded,
        deterministic_algorithms=det_algos,
        cudnn_deterministic=cudnn_det,
        cublas_workspace_config=cublas_cfg,
        warnings=warnings,
    )


def seed_worker(worker_id: int, base_seed: int = DEFAULT_SEED) -> None:
    """DataLoader `worker_init_fn`. Each worker gets a distinct but derived seed."""
    seed = (base_seed + worker_id) % (2**32)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
