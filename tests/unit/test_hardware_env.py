"""Hardware capability, package availability, and the env-profile precondition.

Two failure modes are pinned here, both of which cost a rented GPU hour rather than a
test run:

1. **FA2 capability is not FA2 availability.** Every SM80+ card used to be handed
   `recommended_attn=flash_attention_2` whether or not the `flash_attn` wheel existed in
   the interpreter. A fresh Vast.ai/RunPod image usually has no nvcc and therefore no
   wheel, so the recommendation was a crash inside `from_pretrained` — after the 2.5 GB
   checkpoint had downloaded. Upstream's own model config hard-codes
   `attn_implementation: flash_attention_2`, so nothing else would have caught it.

2. **A rented instance is not the instance you asked for.** `min_vram_gb` turns "this is
   a 12 GB card" into a refusal before anything downloads, instead of a results table
   that claims a 3090.
"""

from __future__ import annotations

import pytest

from rdl.config import EnvConfig, load_env
from rdl.hardware import (
    EnvHardwareMismatch,
    HardwareProfile,
    assert_env_matches_hardware,
    check_env_against_hardware,
    detect,
)


def _profile(**kw) -> HardwareProfile:
    base = {
        "device": "cuda",
        "name": "NVIDIA GeForce RTX 3090",
        "compute_capability": (8, 6),
        "supports_bf16": True,
        "supports_flash_attn2": True,
        "flash_attn_installed": True,
        "total_vram_gb": 25.4,
        "recommended_dtype": "bfloat16",
        "recommended_train_dtype": "bfloat16",
        "recommended_attn": "flash_attention_2",
        "torch_version": "2.4.1",
        "cuda_version": "12.1",
        "python_version": "3.11.9",
        "platform": "Linux",
        "free_disk_gb": 100.0,
        "cpu_count": 16,
    }
    base.update(kw)
    return HardwareProfile(**base)  # type: ignore[arg-type]


# =====================================================================================
# capability vs availability
# =====================================================================================


def test_capability_and_availability_are_separate_fields():
    ampere_without_wheel = _profile(flash_attn_installed=False, recommended_attn="sdpa")
    assert ampere_without_wheel.supports_flash_attn2 is True
    assert ampere_without_wheel.flash_attn_installed is False
    assert ampere_without_wheel.can_use_flash_attn2 is False


def test_summary_reports_both_halves():
    """The operator has to be able to see WHY sdpa was chosen on an Ampere card."""
    s = _profile(flash_attn_installed=False, recommended_attn="sdpa").summary()
    assert "fa2_hardware=True" in s
    assert "fa2_installed=False" in s
    assert "attn=sdpa" in s


def test_detect_never_recommends_fa2_without_the_package():
    """Whatever this box is, the recommendation must be self-consistent."""
    hw = detect()
    if hw.recommended_attn == "flash_attention_2":
        assert hw.supports_flash_attn2 and hw.flash_attn_installed
    if hw.supports_flash_attn2 and not hw.flash_attn_installed:
        assert hw.recommended_attn == "sdpa"


def test_a_profile_pinning_fa2_is_refused_when_the_wheel_is_missing():
    env = EnvConfig(name="fa2_pinned", device="cuda", attn_implementation="flash_attention_2")
    problems = check_env_against_hardware(env, _profile(flash_attn_installed=False))
    assert problems and "flash_attn" in problems[0]

    # ...and accepted when it is there.
    assert check_env_against_hardware(env, _profile()) == []


# =====================================================================================
# the rented-instance preconditions
# =====================================================================================


def test_vast_profile_accepts_a_real_3090():
    assert check_env_against_hardware(load_env("vast_rtx3090"), _profile()) == []


def test_vast_profile_refuses_a_smaller_card():
    problems = check_env_against_hardware(
        load_env("vast_rtx3090"), _profile(name="RTX 3080", total_vram_gb=11.8)
    )
    assert any("VRAM" in p for p in problems)


def test_vast_profile_refuses_a_t4():
    """Wrong instance rented: no bf16, and the profile demands it."""
    t4 = _profile(
        name="Tesla T4",
        compute_capability=(7, 5),
        supports_bf16=False,
        supports_flash_attn2=False,
        flash_attn_installed=False,
        total_vram_gb=15.8,
        recommended_dtype="float16",
        recommended_attn="sdpa",
    )
    problems = check_env_against_hardware(load_env("vast_rtx3090"), t4)
    assert any("bfloat16" in p for p in problems)
    assert any("VRAM" in p for p in problems)


def test_vast_profile_refuses_a_cpu_box():
    cpu = _profile(
        device="cpu",
        name="cpu",
        compute_capability=None,
        supports_bf16=False,
        supports_flash_attn2=False,
        flash_attn_installed=False,
        total_vram_gb=0.0,
    )
    problems = check_env_against_hardware(load_env("vast_rtx3090"), cpu)
    assert any("CUDA" in p for p in problems)


def test_vast_profile_refuses_a_disk_too_small_for_two_checkpoints():
    problems = check_env_against_hardware(load_env("vast_rtx3090"), _profile(free_disk_gb=12.0))
    assert any("disk" in p for p in problems)


def test_assert_raises_with_every_reason_at_once():
    """One refusal listing all of it — not one restart per problem."""
    with pytest.raises(EnvHardwareMismatch) as exc:
        assert_env_matches_hardware(
            load_env("vast_rtx3090"),
            _profile(supports_bf16=False, total_vram_gb=8.0, free_disk_gb=5.0),
        )
    msg = str(exc.value)
    assert "bfloat16" in msg and "VRAM" in msg and "disk" in msg


def test_cpu_profile_still_works_on_a_cpu_box():
    cpu_hw = _profile(
        device="cpu",
        name="cpu",
        compute_capability=None,
        supports_bf16=False,
        supports_flash_attn2=False,
        flash_attn_installed=False,
        total_vram_gb=0.0,
        recommended_dtype="float32",
        recommended_attn="eager",
    )
    assert check_env_against_hardware(load_env("local_cpu"), cpu_hw) == []


def test_colab_t4_profile_is_still_selectable_and_still_correct():
    """Kept, not deleted: a result that only exists on one device is uncheckable."""
    t4 = _profile(
        name="Tesla T4",
        compute_capability=(7, 5),
        supports_bf16=False,
        supports_flash_attn2=False,
        flash_attn_installed=False,
        total_vram_gb=15.8,
        recommended_dtype="float16",
        recommended_attn="sdpa",
    )
    env = load_env("colab_t4")
    assert env.dtype == "auto" and env.attn_implementation == "auto"
    assert check_env_against_hardware(env, t4) == []
