"""Single source of truth for device capability.

Nothing else in the codebase is allowed to ask torch what the GPU can do. The reason
is that the target devices differ in ways that silently corrupt results rather than
crashing:

    device      cc      bf16   FA2 hw  consequence if ignored
    ----------  ------  -----  ------  --------------------------------------------
    Tesla T4    (7,5)   no     no      bf16 configs fall back to fp32 or error;
                                       flash-attn fails to build or fails at runtime
    RTX 3090    (8,6)   yes    yes     fine
    H100        (9,0)   yes    yes     fine

**Hardware capability is not package availability.** `supports_flash_attn2` says the SM
version is >= 8.0; `flash_attn_installed` says the `flash_attn` wheel is importable in
*this* interpreter. Only when both hold is `flash_attention_2` recommended. An Ampere box
without the wheel — which is every fresh Vast.ai/RunPod image that lacks nvcc — used to
get `recommended_attn=flash_attention_2` and then die inside
`AutoModelForCausalLM.from_pretrained`, forty minutes into a session, after the
checkpoint had already been downloaded. Upstream's own
`configs/model/Llama-3.2-1B-Instruct.yaml` hard-codes `attn_implementation:
flash_attention_2`, so the bridge's override is the only thing standing between that
default and the model load.

The refusal in `assert_training_allowed` is the other important part of this module.
Gradient-ascent-family objectives (GA, NPO) push loss upward without bound. Under fp16
the `GradScaler` NaNs on them. That is silent corruption: you get a finished run and a
plausible-looking number that means nothing.
"""

from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, no import cycle at runtime
    from .config import EnvConfig

__all__ = [
    "EnvHardwareMismatch",
    "HardwareProfile",
    "TrainingOnFp16Error",
    "assert_env_matches_hardware",
    "assert_training_allowed",
    "check_env_against_hardware",
    "detect",
    "flash_attn_available",
]

# FlashAttention-2 requires Ampere or newer.
_FA2_MIN_CC = (8, 0)


class TrainingOnFp16Error(RuntimeError):
    """Raised when a training run is attempted on a device without bf16 support."""


class EnvHardwareMismatch(RuntimeError):
    """Raised when the requested environment profile does not match the detected device.

    Renting the wrong GPU and discovering it from a results table is the failure this
    prevents.
    """


@lru_cache(maxsize=1)
def flash_attn_available() -> bool:
    """True iff `import flash_attn` actually SUCCEEDS in a fresh interpreter.

    Two steps, and both are load-bearing:

    1. `find_spec` is the cheap negative. On the common case — no wheel at all — it
       answers in microseconds and no subprocess is spawned.
    2. If a spec exists, the module is imported **in a subprocess**. `find_spec` only
       proves Python can locate the package; `flash_attn` is a thin wrapper around a
       compiled CUDA extension, and a wheel built against a different torch or CUDA
       imports with an `ImportError: undefined symbol` at first use. That failure would
       otherwise surface inside `from_pretrained`, after the checkpoint downloaded —
       the exact failure mode this whole predicate exists to prevent.

    The subprocess is what keeps a broken extension from poisoning this process: a
    mismatched CUDA ext can abort the interpreter, not merely raise. Cached, because
    `detect()` runs at the top of every command and the answer cannot change mid-process.
    """
    import importlib.util

    try:
        if importlib.util.find_spec("flash_attn") is None:
            return False
    except (ImportError, ValueError):  # partially installed / broken metadata
        return False

    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import flash_attn"],
            capture_output=True,
            timeout=120,  # a cold CUDA-extension import is slow, but not this slow
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0


@dataclass(frozen=True)
class HardwareProfile:
    device: str  # "cpu" | "cuda"
    name: str
    compute_capability: tuple[int, int] | None
    supports_bf16: bool
    supports_flash_attn2: bool  # HARDWARE capability (SM >= 8.0)
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
    # PACKAGE availability. Defaulted so existing constructions stay valid; `detect()`
    # always passes it explicitly.
    flash_attn_installed: bool = False

    # ---------------------------------------------------------------- convenience --

    @property
    def is_cuda(self) -> bool:
        return self.device == "cuda"

    @property
    def is_turing_or_older(self) -> bool:
        return self.compute_capability is not None and self.compute_capability < _FA2_MIN_CC

    @property
    def can_use_flash_attn2(self) -> bool:
        """Hardware supports it AND the wheel is installed. The only honest predicate."""
        return self.supports_flash_attn2 and self.flash_attn_installed

    def to_dict(self) -> dict:
        d = asdict(self)
        d["compute_capability"] = list(self.compute_capability) if self.compute_capability else None
        return d

    def summary(self) -> str:
        cc = ".".join(map(str, self.compute_capability)) if self.compute_capability else "-"
        return (
            f"{self.device}:{self.name} cc={cc} vram={self.total_vram_gb}GB "
            f"bf16={self.supports_bf16} fa2_hardware={self.supports_flash_attn2} "
            f"fa2_installed={self.flash_attn_installed} "
            f"-> dtype={self.recommended_dtype} attn={self.recommended_attn}"
        )


def _free_disk_gb(where: Path | None = None) -> float:
    # /workspace is the Vast.ai / RunPod persistent volume; /content is Colab's. Probing
    # the wrong one reports the container's tiny overlay and looks like a full disk.
    for candidate in (where, Path("/workspace"), Path("/content"), Path.cwd()):
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
    fa2_installed = flash_attn_available()

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
            flash_attn_installed=fa2_installed,
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
            flash_attn_installed=fa2_installed,
        )

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    cc = (props.major, props.minor)

    try:
        bf16 = bool(torch.cuda.is_bf16_supported())
    except (RuntimeError, AttributeError):
        bf16 = cc >= _FA2_MIN_CC

    fa2_hardware = cc >= _FA2_MIN_CC

    eval_dtype = "bfloat16" if bf16 else "float16"
    # Training in fp16 on a GA/NPO objective is unsound; fp32 is the only safe
    # fallback, and `assert_training_allowed` will still block by default.
    train_dtype = "bfloat16" if bf16 else "float32"
    # Both halves, or sdpa. An Ampere card without the wheel is the common Vast.ai case
    # and recommending FA2 there produces a crash at model-load time, not a fallback.
    attn = "flash_attention_2" if (fa2_hardware and fa2_installed) else "sdpa"

    return HardwareProfile(
        device="cuda",
        name=props.name,
        compute_capability=cc,
        supports_bf16=bf16,
        supports_flash_attn2=fa2_hardware,
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
        flash_attn_installed=fa2_installed,
    )


# ------------------------------------------------------- environment vs. hardware --


def check_env_against_hardware(env: EnvConfig, hw: HardwareProfile) -> list[str]:
    """Return every way the requested environment profile disagrees with this box.

    Pure: returns reasons rather than raising, so `env-check` can print all of them and
    `run-condition` can refuse on the same list. The profile is a claim about the machine
    the results were produced on — `vast_rtx3090` in a report that ran on a T4 is a
    provenance error, not a performance one.
    """
    problems: list[str] = []

    if env.device == "cuda" and not hw.is_cuda:
        problems.append(
            f"env '{env.name}' requires CUDA, but no CUDA device was detected "
            f"(torch reports {hw.name}). Renting a GPU box is not the same as torch "
            "seeing it: check `nvidia-smi` and that torch was installed from a CUDA index."
        )
    if env.device == "cpu" and hw.is_cuda:
        problems.append(
            f"env '{env.name}' pins device=cpu but a CUDA device ({hw.name}) is present. "
            "That is almost certainly the wrong profile for this box."
        )
    if env.dtype == "bfloat16" and not hw.supports_bf16:
        problems.append(
            f"env '{env.name}' requires bfloat16, but {hw.name} "
            f"(cc={hw.compute_capability}) has no bf16 datapath."
        )
    if env.attn_implementation == "flash_attention_2" and not hw.can_use_flash_attn2:
        why = (
            "the GPU is pre-Ampere"
            if not hw.supports_flash_attn2
            else "the `flash_attn` package is not installed in this interpreter"
        )
        problems.append(
            f"env '{env.name}' pins attn_implementation=flash_attention_2, but {why}. "
            "Install it (`pip install --no-build-isolation flash-attn==2.6.3`, needs "
            "nvcc) or set `attn_implementation: auto` and let hardware.py fall back to "
            "sdpa."
        )
    if env.min_vram_gb is not None and hw.is_cuda and hw.total_vram_gb < env.min_vram_gb:
        problems.append(
            f"env '{env.name}' expects at least {env.min_vram_gb:.0f} GB of VRAM, but "
            f"{hw.name} reports {hw.total_vram_gb:.1f} GB. This is the wrong instance — "
            "stop it before the grid starts rather than after."
        )
    if env.min_free_disk_gb is not None and hw.free_disk_gb < env.min_free_disk_gb:
        problems.append(
            f"env '{env.name}' expects at least {env.min_free_disk_gb:.0f} GB free disk, "
            f"but only {hw.free_disk_gb:.1f} GB is available. Two 1B checkpoints plus the "
            "TOFU eval logs do not fit, and disk cannot be resized after an instance is "
            "created."
        )
    # Provenance. VRAM and bf16 alone would also accept a 4090, an A5000 or an H100:
    # all fine to evaluate on, none of them the card a report stamped with this profile
    # says it ran on.
    if (
        env.expected_gpu_name_regex
        and hw.is_cuda
        and not re.search(env.expected_gpu_name_regex, hw.name, re.IGNORECASE)
    ):
        problems.append(
            f"env '{env.name}' expects a GPU matching /{env.expected_gpu_name_regex}/, "
            f"but this box reports '{hw.name}'. If that card is what you meant to "
            "rent, use the generic profile (--env rtx3090 / h100) so the report does "
            "not claim hardware it did not run on."
        )
    if env.expected_compute_capability is not None and hw.compute_capability is not None:
        want = tuple(env.expected_compute_capability)
        if tuple(hw.compute_capability) != want:
            problems.append(
                f"env '{env.name}' expects compute capability {want[0]}.{want[1]}, but "
                f"this GPU reports {hw.compute_capability[0]}.{hw.compute_capability[1]}."
            )
    if env.min_python is not None and _version_lt(hw.python_version, env.min_python):
        problems.append(
            f"env '{env.name}' requires Python >= {env.min_python}, but this interpreter "
            f"is {hw.python_version}. open-unlearning declares `python_requires >= 3.11`; "
            "installing the submodule under an older interpreter fails, and picking an "
            "image without checking is how that is discovered an hour in."
        )
    return problems


def _version_lt(have: str, want: str) -> bool:
    """`have < want` on dotted numeric prefixes. Tolerant of '3.11.9+local' and friends."""

    def parts(v: str) -> tuple[int, ...]:
        out: list[int] = []
        for chunk in v.split("."):
            digits = ""
            for ch in chunk:
                if not ch.isdigit():
                    break
                digits += ch
            if not digits:
                break
            out.append(int(digits))
        return tuple(out)

    a, b = parts(have), parts(want)
    if not a:  # unparseable version: do not invent a failure
        return False
    n = max(len(a), len(b))
    return a + (0,) * (n - len(a)) < b + (0,) * (n - len(b))


def assert_env_matches_hardware(env: EnvConfig, hw: HardwareProfile) -> None:
    """`check_env_against_hardware`, as a refusal. Called before anything is downloaded."""
    problems = check_env_against_hardware(env, hw)
    if problems:
        raise EnvHardwareMismatch(
            f"environment profile '{env.name}' does not match the detected hardware:\n  - "
            + "\n  - ".join(problems)
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
