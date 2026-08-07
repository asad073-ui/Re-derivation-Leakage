"""Config composition, strict validation, and hashing.

Two properties carry weight:

- **Unknown keys are an error.** A typo'd `writepolicy.mdoe` must fail at config-parse
  time, not forty minutes into a GPU run when the write path turns out to have silently
  used its default.
- **Two runs with the same hash are byte-identical.** The hash covers the *resolved*
  tree, so it must be insensitive to YAML formatting and sensitive to any value change.
"""

from __future__ import annotations

import re

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from rdl.config import (
    ConfigError,
    EnvConfig,
    RDLConfig,
    compose,
    config_hash,
    load_config,
    load_env,
    validate,
)
from rdl.paths import configs_dir, repo_root

CONDITIONS = configs_dir(repo_root()) / "conditions"


# ------------------------------------------------------------------- composition --


@pytest.mark.parametrize("name", ["C0", "C1", "C2", "C3"])
def test_every_shipped_condition_loads(name):
    cfg = load_config(CONDITIONS / f"{name}.yaml")
    assert cfg.condition == name
    assert cfg.env.name
    assert cfg.agent_a.agent_id == "A"
    assert cfg.agent_a.model in cfg.models


def test_composition_pulls_in_the_referenced_fragments():
    cfg = load_config(CONDITIONS / "C3.yaml")
    assert cfg.env.name == "vast_rtx3090"
    assert cfg.memory.name == "sbu_id_blocklist"
    assert cfg.memory.blocklist == "id"
    assert cfg.writepolicy.mode == "framework_default"
    assert cfg.agent_b is not None and cfg.agent_b.agent_id == "B"


def test_missing_required_group_is_an_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("condition: C0\nmemory: none\nwritepolicy: disabled\nagent_a: A_unlearned\n")
    with pytest.raises(ConfigError, match="missing required key 'env'"):
        compose(bad)


def test_missing_fragment_file_is_an_error(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "condition: C0\nenv: does_not_exist\nmemory: none\n"
        "writepolicy: disabled\nagent_a: A_unlearned\n"
    )
    with pytest.raises(ConfigError, match="config file not found"):
        compose(bad)


# --------------------------------------------------------------------- overrides --


def test_cli_overrides_are_applied_last():
    cfg = load_config(CONDITIONS / "C3.yaml", ["episode.retrieval_k=11", "seed=7"])
    assert cfg.episode.retrieval_k == 11
    assert cfg.seed == 7


def test_override_can_change_the_env():
    cfg = load_config(CONDITIONS / "C3.yaml", ["env.name=local_cpu", "env.device=cpu"])
    assert cfg.env.device == "cpu"


# ----------------------------------------------------------------- env selection --


@pytest.mark.parametrize("name", ["local_cpu", "colab_t4", "rtx3090", "vast_rtx3090", "h100"])
def test_every_shipped_env_profile_validates(name):
    """A profile that only fails to parse on the box it was written for is worthless."""
    env = load_env(name)
    assert env.name == name


def test_env_override_replaces_the_whole_group():
    """The reason `--env` exists rather than `--set env=...`.

    By the time a dotlist override is applied, `env` is a fully populated mapping. A
    string override cannot replace it, and a per-key one leaves the previous profile's
    values in place under a new name — e.g. Colab's /content HF cache surviving a switch
    to the 3090 profile, on a box where /content does not exist.
    """
    t4 = load_config(CONDITIONS / "C3D.yaml", env_override="colab_t4")
    vast = load_config(CONDITIONS / "C3D.yaml", env_override="vast_rtx3090")

    assert t4.env.name == "colab_t4"
    assert t4.env.hf_home == "/content/hf"
    assert vast.env.name == "vast_rtx3090"
    assert vast.env.hf_home == "/workspace/hf"
    assert vast.env.min_vram_gb == 20.0
    # No leakage from the profile that was NOT selected.
    assert t4.env.min_vram_gb is None


def test_env_override_changes_nothing_scientific():
    """Hardware moves; the condition must not. Everything except `env` is identical."""
    a = load_config(CONDITIONS / "C3D.yaml", env_override="colab_t4").model_dump(mode="json")
    b = load_config(CONDITIONS / "C3D.yaml", env_override="vast_rtx3090").model_dump(mode="json")
    a.pop("env")
    b.pop("env")
    assert a == b


def test_env_override_beats_the_condition_file():
    default = load_config(CONDITIONS / "C1W.yaml")
    assert default.env.name == "vast_rtx3090", "the grid now runs on the 3090 by default"
    assert load_config(CONDITIONS / "C1W.yaml", env_override="local_cpu").env.name == "local_cpu"


def test_a_dotlist_cannot_switch_the_env_group():
    """Demonstrates why `--env` is a separate mechanism rather than sugar over `--set`.

    `--set env=rtx3090` replaces a populated mapping with a bare string, which fails
    validation. The dangerous variant is `--set env.name=rtx3090`, which succeeds and
    leaves every OTHER key from the old profile in place — a run labelled rtx3090 that is
    still pointing HF_HOME at Colab's /content.
    """
    with pytest.raises(ConfigError):
        load_config(CONDITIONS / "C0.yaml", ["env=colab_t4"])

    mislabelled = load_config(CONDITIONS / "C0.yaml", ["env.name=colab_t4"])
    assert mislabelled.env.name == "colab_t4"
    assert mislabelled.env.hf_home == "/workspace/hf", "the old profile's values survived"


def test_unknown_env_override_is_an_error():
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(CONDITIONS / "C0.yaml", env_override="a_gpu_nobody_owns")


def test_env_override_changes_the_config_hash():
    """Two runs on different hardware are not the same run, and the id must say so."""
    a = load_config(CONDITIONS / "C3D.yaml", env_override="colab_t4")
    b = load_config(CONDITIONS / "C3D.yaml", env_override="vast_rtx3090")
    assert config_hash(a) != config_hash(b)


# -------------------------------------------------------------- strict validation --


def test_unknown_key_is_rejected():
    """The whole point of the Pydantic layer."""
    raw = compose(CONDITIONS / "C3.yaml")
    raw.writepolicy.mdoe = "framework_default"  # typo
    with pytest.raises(ConfigError):
        validate(raw)


def test_unknown_top_level_key_is_rejected():
    raw = compose(CONDITIONS / "C0.yaml")
    raw.totally_made_up = 1
    with pytest.raises(ConfigError):
        validate(raw)


def test_config_is_frozen():
    cfg = load_config(CONDITIONS / "C0.yaml")
    with pytest.raises(ValidationError):
        cfg.seed = 999


# ------------------------------------------------------------ cross-field checks --


def test_c0_rejects_a_second_agent():
    raw = compose(CONDITIONS / "C0.yaml")
    raw.agent_b = OmegaConf.load(configs_dir() / "agents" / "B_full.yaml")
    raw.models.tofu_llama32_1b_full = OmegaConf.load(
        configs_dir() / "models" / "tofu_llama32_1b_full.yaml"
    )
    with pytest.raises(ConfigError, match="single-agent"):
        validate(raw)


def test_two_agent_condition_requires_agent_b():
    raw = compose(CONDITIONS / "C3.yaml")
    del raw["agent_b"]
    with pytest.raises(ConfigError, match="agent_b is required"):
        validate(raw)


def test_c1_must_have_write_back_disabled():
    """C1 IS the leakage floor. Enabling write-back would silently destroy the baseline."""
    raw = compose(CONDITIONS / "C1.yaml")
    raw.writepolicy.mode = "framework_default"
    with pytest.raises(ConfigError, match=re.escape("C1 requires writepolicy.mode=disabled")):
        validate(raw)


def test_agent_referencing_an_uncomposed_model_is_rejected():
    raw = compose(CONDITIONS / "C3.yaml")
    raw.agent_b.model = "some_model_nobody_composed"
    with pytest.raises(ConfigError, match="not composed"):
        validate(raw)


def test_bf16_override_is_representable_but_flagged_at_load_time():
    """The config layer accepts it; loader.resolve_dtype is what refuses it on a T4."""
    raw = compose(CONDITIONS / "C3.yaml")
    raw.models.tofu_llama32_1b_npo_forget10.dtype_override = "bfloat16"
    cfg = validate(raw)
    assert cfg.models["tofu_llama32_1b_npo_forget10"].dtype_override == "bfloat16"


# -------------------------------------------------------------------- hashing --


def test_hash_is_stable_across_calls():
    cfg = load_config(CONDITIONS / "C3.yaml")
    assert config_hash(cfg) == config_hash(cfg)


def test_hash_is_insensitive_to_yaml_formatting(tmp_path):
    a = load_config(CONDITIONS / "C3.yaml")
    b = load_config(CONDITIONS / "C3.yaml", [])
    assert config_hash(a) == config_hash(b)


def test_hash_changes_when_any_value_changes():
    base = load_config(CONDITIONS / "C3.yaml")
    changed = load_config(CONDITIONS / "C3.yaml", ["episode.retrieval_k=99"])
    assert config_hash(base) != config_hash(changed)


def test_hash_distinguishes_conditions():
    hashes = {config_hash(load_config(CONDITIONS / f"{c}.yaml")) for c in ("C0", "C1", "C2", "C3")}
    assert len(hashes) == 4


def test_hash_accepts_plain_dicts():
    assert config_hash({"a": 1}) == config_hash({"a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


# ------------------------------------------------------------------- sub-models --


def test_env_batch_size_must_be_positive():
    with pytest.raises(ValidationError):
        EnvConfig(name="x", batch_size=0)


def test_rdl_config_model_names():
    cfg = load_config(CONDITIONS / "C2.yaml")
    assert cfg.model_names() == sorted(cfg.models)
    assert "tofu_llama32_1b_full" in cfg.model_names()


def test_config_round_trips_through_json():
    import json

    cfg = load_config(CONDITIONS / "C3.yaml")
    blob = json.dumps(cfg.model_dump(mode="json"))
    back = RDLConfig.model_validate(json.loads(blob))
    assert config_hash(back) == config_hash(cfg)
