"""Single source of truth for device capability.

Nothing else in the codebase is allowed to ask torch what the GPU can do. The reason
is that the three target devices differ in ways that silently corrupt results rather
than crashing:

    device      cc      bf16   FA2    consequence if ignored
    ----------  ------  -----  -----  --------------------------------------------
    Tesla T4    (7,5)   no     no     bf16 configs fall back to fp32 or error;
                                      flash-attn fails to build or fails at runtime
    RTX 3090    (8,6)   yes    yes    fine
    H100        (9,0)   yes    yes    fine

The refusal in `assert_training_allowed` is the important part of this module.
Gradient-ascent-family objectives (GA, NPO) push loss upward without bound. Under fp16
the `GradScaler` NaNs on them. That is silent corruption: you get a finished run and a
plausible-looking number that means nothing.
"""

from __future__ import annotations

import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = [
    "HardwareProfile",
    "TrainingOnFp16Error",
    "assert_training_allowed",
    "detect",
]

# FlashAttention-2 requires Ampere or newer.
_FA2_MIN_CC = (8, 0)


class TrainingOnFp16Error(RuntimeError):
    """Raised when a training run is attempted on a device without bf16 support."""


@dataclass(frozen=True)
class HardwareProfile:
    device: str  # "cpu" | "cuda"
    name: str
    compute_capability: tuple[int, int] | None
    supports_bf16: bool
    supports_flash_attn2: bool
    total_vram_gb: float
    recommended_dtype: str  # eval dtype
    recommended_train_dtype: str
    recommended_attn: str
    torch_version: str | None
    cuda_version: str | None
    python_version: str
    platform: str
    free_disk_gb: float
    cpu_count: int | None

    # ---------------------------------------------------------------- convenience --

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def is_turing_or_older(self) -> bool:
        return self.compute_capability is not None and self.compute_capability < _FA2_MIN_CC

    def to_dict(self) -> dict:
        d = asdict(self)
        d["compute_capability"] = list(self.compute_capability) if self.compute_capability else None
        return d

    def summary(self) -> str:
        cc = ".".join(map(str, self.compute_capability)) if self.compute_capability else "-"
        return (
            f"{self.device}:{self.name} cc={cc} vram={self.total_vram_gb}GB "
            f"bf16={self.supports_bf16} fa2={self.supports_flash_attn2} "
            f"-> dtype={self.recommended_dtype} attn={self.recommended_attn}"
        )


def _free_disk_gb(where: Path | None = None) -> float:
    for candidate in (where, Path("/content"), Path.cwd()):
        if candidate is None:
            continue
        try:
            return round(shutil.disk_usage(str(candidate)).free / 1e9, 1)
        except OSError:
            continue
    return 0.0


def detect(disk_probe: Path | None = None) -> HardwareProfile:
    """Probe the current device. Never raises; a torch-less CPU box is a valid answer."""
    import os

    py = platform.python_version()
    plat = f"{platform.system()}-{platform.release()}-{platform.machine()}"
    free_disk = _free_disk_gb(disk_probe)
    cpu_count = os.cpu_count()

    try:
        import torch
    except ImportError:
        return HardwareProfile(
            device="cpu",
            name="cpu (torch not installed)",
            compute_capability=None,
            supports_bf16=False,
            supports_flash_attn2=False,
            total_vram_gb=0.0,
            recommended_dtype="float32",
            recommended_train_dtype="float32",
            recommended_attn="eager",
            torch_version=None,
            cuda_version=None,
            python_version=py,
            platform=plat,
            free_disk_gb=free_disk,
            cpu_count=cpu_count,
        )

    torch_version = torch.__version__
    cuda_version = getattr(torch.version, "cuda", None)

    if not torch.cuda.is_available():
        return HardwareProfile(
            device="cpu",
            name=platform.processor() or "cpu",
            compute_capability=None,
            supports_bf16=False,
            supports_flash_attn2=False,
            total_vram_gb=0.0,
            # fp16 on CPU is slower than fp32 and unsupported by many kernels.
            recommended_dtype="float32",
            recommended_train_dtype="float32",
            recommended_attn="eager",
            torch_version=torch_version,
            cuda_version=cuda_version,
            python_version=py,
            platform=plat,
            free_disk_gb=free_disk,
            cpu_count=cpu_count,
        )

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    cc = (props.major, props.minor)

    try:
        bf16 = bool(torch.cuda.is_bf16_supported())
    except (RuntimeError, AttributeError):
        bf16 = cc >= _FA2_MIN_CC

    fa2 = cc >= _FA2_MIN_CC

    eval_dtype = "bfloat16" if bf16 else "float16"
    # Training in fp16 on a GA/NPO objective is unsound; fp32 is the only safe
    # fallback, and `assert_training_allowed` will still block by default.
    train_dtype = "bfloat16" if bf16 else "float32"
    attn = "flash_attention_2" if fa2 else "sdpa"

    return HardwareProfile(
        device="cuda",
        name=props.name,
        compute_capability=cc,
        supports_bf16=bf16,
        supports_flash_attn2=fa2,
        total_vram_gb=round(props.total_memory / 1e9, 1),
        recommended_dtype=eval_dtype,
        recommended_train_dtype=train_dtype,
        recommended_attn=attn,
        torch_version=torch_version,
        cuda_version=cuda_version,
        python_version=py,
        platform=plat,
        free_disk_gb=free_disk,
        cpu_count=cpu_count,
    )


def assert_training_allowed(hw: HardwareProfile, allow_fp16_training: bool = False) -> None:
    """Refuse to start training on a device without bf16 unless explicitly overridden.

    See the module docstring: this is a silent-corruption guard, not a performance one.
    """
    if hw.supports_bf16:
        return
    if allow_fp16_training:
        return
    raise TrainingOnFp16Error(
        f"Refusing to train on {hw.name} (compute capability {hw.compute_capability}): "
        "no bfloat16 support. Gradient-ascent-family objectives (GA, NPO) push loss "
        "upward without bound and fp16 GradScaler NaNs on them — this fails silently, "
        "producing a finished run with a meaningless number.\n"
        "  -> Train on Ampere or newer (RTX 3090, H100).\n"
        "  -> Or pass --allow-fp16-training if you accept an uninterpretable result."
    )
