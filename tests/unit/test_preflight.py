"""`env-check --strict`: everything that must hold before a GPU hour is spent.

The diagnostic form of `env-check` printed a missing token, a blocked licence, a
mismatched submodule pin and an unreachable checkpoint — and exited 0 on all of them.
Only the env-profile mismatch failed. So the one command whose entire purpose is "find
out now instead of at hour six" could not stop a session.

`strict_blockers` is pure so that every one of those conditions is testable on a laptop
with no token, no GPU and no network.
"""

from __future__ import annotations

from rdl.cli.env_check import strict_blockers
from rdl.hardware import flash_attn_available


def _ok_info(**overrides) -> dict:
    """An environment with nothing wrong with it."""
    info = {
        "packages": {"torch": "2.4.1", "transformers": "4.51.3", "numpy": "2.2.3"},
        "huggingface": {
            "token_present": True,
            "user": "someone",
            "meta-llama/Llama-3.2-1B-Instruct": "ok",
            "open-unlearning/tofu_Llama-3.2-1B-Instruct_full": "ok @ 88e31200b97e",
        },
        "submodule": {"present": True, "matches_pin": True, "head_sha": "4ad738a"},
        "retain_logs": {"present": True, "path": "/w/TOFU_EVAL.json"},
        "env_profile": {"requested": "vast_rtx3090", "matches_hardware": True, "problems": []},
    }
    info.update(overrides)
    return info


def test_a_healthy_box_has_no_blockers():
    assert strict_blockers(_ok_info()) == []


def test_missing_token_blocks():
    info = _ok_info(huggingface={"token_present": False})
    assert any("HF_TOKEN" in b for b in strict_blockers(info))


def test_blocked_llama_licence_blocks():
    """The gated base model supplies the tokenizer and chat template every eval uses.
    Approval is instant; hitting it at hour six is not."""
    info = _ok_info(
        huggingface={
            "token_present": True,
            "meta-llama/Llama-3.2-1B-Instruct": "BLOCKED (GatedRepoError)",
        }
    )
    blockers = strict_blockers(info)
    assert any("meta-llama/Llama-3.2-1B-Instruct" in b for b in blockers)


def test_a_checkpoint_unreachable_at_its_pinned_revision_blocks():
    """Repo-level reachability is the weaker test: a pin at a removed commit resolves as
    a healthy repo and 404s inside from_pretrained later."""
    info = _ok_info(
        huggingface={
            "token_present": True,
            "meta-llama/Llama-3.2-1B-Instruct": "ok",
            "open-unlearning/unlearn_tofu_x_NPO": "BLOCKED @ 94ed64eb73bc (RevisionNotFoundError)",
        }
    )
    assert any("unlearn_tofu_x_NPO" in b for b in strict_blockers(info))


def test_a_missing_package_blocks():
    info = _ok_info(packages={"torch": "2.4.1", "transformers": "MISSING"})
    assert any("transformers" in b for b in strict_blockers(info))


def test_a_missing_submodule_blocks():
    info = _ok_info(submodule={"present": False})
    assert any("submodule" in b for b in strict_blockers(info))


def test_a_submodule_off_its_pin_blocks():
    """Every published number is quoted against the pinned SHA."""
    info = _ok_info(
        submodule={
            "present": True,
            "matches_pin": False,
            "head_sha": "deadbee",
            "pinned_sha": "4ad738a",
        }
    )
    assert any("pins" in b for b in strict_blockers(info))


def test_missing_retain_logs_block():
    """forget_quality is computed against them; absent, the metric is silently
    unavailable rather than loudly missing."""
    info = _ok_info(retain_logs={"present": False, "path": "/w/x.json", "hint": "setup_data.py"})
    assert any("forget_quality" in b for b in strict_blockers(info))


def test_env_profile_problems_are_carried_into_the_preflight():
    info = _ok_info(
        env_profile={
            "requested": "vast_rtx3090",
            "matches_hardware": False,
            "problems": ["env 'vast_rtx3090' expects at least 20 GB of VRAM, but ... 11.8 GB"],
        }
    )
    assert any("VRAM" in b for b in strict_blockers(info))


def test_every_problem_is_reported_at_once():
    """One refusal listing everything — not one restart per discovery."""
    info = _ok_info(
        huggingface={"token_present": False},
        packages={"torch": "MISSING"},
        submodule={"present": False},
        retain_logs={"present": False, "path": "/w/x.json"},
    )
    assert len(strict_blockers(info)) >= 4


# =====================================================================================
# flash_attn: locating the module is not importing it
# =====================================================================================


def test_flash_attn_probe_is_honest_about_this_box():
    """`find_spec` only proves Python can locate the package. flash_attn wraps a compiled
    CUDA extension, and a wheel built against a different torch imports with an
    `undefined symbol` error — which would otherwise surface inside from_pretrained,
    after the checkpoint downloaded. The probe therefore really imports it, in a
    subprocess so a broken extension cannot abort this interpreter."""
    result = flash_attn_available()
    assert isinstance(result, bool)
    if result:
        import flash_attn  # noqa: F401  — if the probe says yes, this must not raise


def test_flash_attn_probe_is_cached():
    """`detect()` runs at the top of every command; the answer cannot change mid-process
    and the subprocess must not be spawned repeatedly."""
    assert flash_attn_available() is flash_attn_available()
    assert flash_attn_available.cache_info().hits >= 1
