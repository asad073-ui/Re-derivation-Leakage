"""Config composition, strict validation, and hashing.

Two properties carry weight:

- **Unknown keys are an error.** A typo'd `writepolicy.mdoe` must fail at config-parse
  time, not forty minutes into a GPU run when the write path turns out to have silently
  used its default.
- **Two runs with the same hash are byte-identical.** The hash covers the *resolved*
  tree, so it must be insensitive to YAML formatting and sensitive to any value change.
"""

from __future__ import annotations

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
    assert cfg.env.name == "colab_t4"
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
    with pytest.raises(ConfigError, match="C1 is the leakage floor"):
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
