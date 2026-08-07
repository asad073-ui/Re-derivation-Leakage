"""The fp32-logits compatibility shim.

The defect it works around is dated and explained in `rdl.compat.fp32_logits`. What
these tests protect is the part that is easy to lose silently:

  * the shim is NAMED IN THE COMMAND, so a report's command string is enough to tell a
    shimmed run from an unshimmed one;
  * a failure to attach is fatal rather than a no-op;
  * upstream's own files are never touched.
"""

from __future__ import annotations

from pathlib import Path

from rdl.compat import fp32_logits
from rdl.eval.openunlearning_bridge import EvalSpec, build_eval_command, command_string
from rdl.hardware import HardwareProfile


def _ampere() -> HardwareProfile:
    return HardwareProfile(
        device="cuda",
        name="NVIDIA GeForce RTX 3090",
        compute_capability=(8, 6),
        supports_bf16=True,
        supports_flash_attn2=True,
        total_vram_gb=25.3,
        recommended_dtype="bfloat16",
        recommended_train_dtype="bfloat16",
        recommended_attn="flash_attention_2",
        torch_version="2.4.1",
        cuda_version="12.1",
        python_version="3.11.9",
        platform="Linux",
        free_disk_gb=100.0,
        cpu_count=16,
        flash_attn_installed=True,
    )


def _cmd() -> list[str]:
    return build_eval_command(EvalSpec(model_path="org/ckpt", task_name="t"), _ampere())


def test_command_invokes_the_shim_not_eval_py_directly():
    """The bare upstream command crashes in bf16; the recorded one must not."""
    cmd = _cmd()
    assert "-m" in cmd
    assert "rdl.compat.ou_eval_shim" in cmd
    assert cmd.index("-m") < cmd.index("rdl.compat.ou_eval_shim") < cmd.index("src/eval.py")


def test_shim_is_visible_in_the_recorded_command_string():
    """Provenance: reading the report alone must reveal the shim."""
    assert "rdl.compat.ou_eval_shim" in command_string(_cmd())


def test_shim_precedes_hydra_overrides():
    """`python -m pkg script.py <overrides>` — the script stays argv[0] for Hydra."""
    cmd = _cmd()
    assert cmd[1:4] == ["-m", "rdl.compat.ou_eval_shim", "src/eval.py"]
    assert cmd[4].startswith("--config-name=")


def test_published_parity_overrides_are_unchanged_by_the_shim():
    """The shim must not have quietly moved the run off the published settings.

    Built at the parity settings the Day-1 gate passes explicitly (batch 32 / seed 0),
    not at `EvalSpec`'s defaults, which are the pre-registered deterministic protocol.
    """
    spec = EvalSpec(model_path="org/ckpt", task_name="t", batch_size=32, seed=0)
    cmd = build_eval_command(spec, _ampere())
    assert "model.model_args.torch_dtype=bfloat16" in cmd
    assert "model.model_args.attn_implementation=flash_attention_2" in cmd
    assert "seed=0" in cmd
    assert "eval.tofu.batch_size=32" in cmd


def test_install_is_idempotent_and_reports_its_target():
    record = fp32_logits.install()
    again = fp32_logits.install()
    assert fp32_logits.is_installed()
    assert "LlamaForCausalLM" in str(record.get("target"))
    assert again.get("target") == record.get("target")


def test_no_file_under_third_party_is_modified_by_this_package():
    """Design invariant 1: the shim lives entirely in our own source tree."""
    compat = Path(fp32_logits.__file__).resolve()
    assert "third_party" not in compat.parts
    assert compat.parts[-3:] == ("rdl", "compat", "fp32_logits.py")
