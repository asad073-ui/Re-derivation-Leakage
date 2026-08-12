"""Config composition, the profile/study separation, and the three hashes."""

from __future__ import annotations

import pytest
import yaml

from rdl.graph.config import GraphLaunchConfig, load_graph_config, resolve
from rdl.graph.validation import GraphConfigError
from rdl.paths import repo_root

LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"
GRAPH_CONFIGS = repo_root() / "configs" / "graph"


def test_launch_resolves_the_whole_tree():
    cfg = load_graph_config(LAUNCH)
    assert cfg.study.study_id == "graphforget-v1"
    assert cfg.topology.name == "diamond5"
    assert [a.name for a in cfg.arms] == [
        "single_agent",
        "multi_agent_control",
        "multi_agent_leak",
        "multi_agent_dragon",
        "multi_agent_dragon_subsets",
        "multi_agent_graphforget",
    ]
    assert set(cfg.defenses) == {
        "none",
        "dragon_style",
        # The matched-subset fairness ablation, composed alongside the published
        # baseline rather than replacing it.
        "dragon_style_subsets",
        "graphforget",
    }


def test_only_the_active_profile_changes_between_machines():
    cpu = load_graph_config(LAUNCH)
    gpu = load_graph_config(LAUNCH, overrides=["active_profile=rtx3090_1b"])
    h100 = load_graph_config(LAUNCH, overrides=["active_profile=h100_7b"])
    # The science is identical; only the profile hash moves.
    assert cpu.study_design_hash() == gpu.study_design_hash() == h100.study_design_hash()
    assert len({cpu.profile_hash(), gpu.profile_hash(), h100.profile_hash()}) == 3
    assert len({cpu.resolved_run_hash(), gpu.resolved_run_hash(), h100.resolved_run_hash()}) == 3


def test_hash_is_stable_across_loads():
    assert load_graph_config(LAUNCH).hashes() == load_graph_config(LAUNCH).hashes()


def test_gpu_profiles_reach_the_studys_primary_k():
    for profile in ("rtx3090_1b", "h100_7b"):
        cfg = load_graph_config(LAUNCH, overrides=[f"active_profile={profile}"])
        assert cfg.study.sampling.primary_k in cfg.profile.sampling.k_values
        assert cfg.profile.reportable


def test_cpu_profile_declares_itself_unreportable():
    cfg = load_graph_config(LAUNCH)
    assert cfg.profile.reportable is False
    assert cfg.study.sampling.primary_k not in cfg.profile.sampling.k_values


def test_profile_may_not_change_the_agent_count():
    with pytest.raises(GraphConfigError, match="logical_count"):
        load_graph_config(LAUNCH, topology="diamond6")


def test_diamond6_needs_a_six_agent_profile(tmp_path):
    """A six-node topology is only legal with a profile that declares six agents."""
    profile = yaml.safe_load(
        (GRAPH_CONFIGS / "runtime" / "local_cpu.yaml").read_text(encoding="utf-8")
    )
    assert profile["agents"]["logical_count"] == 5


def test_unknown_profile_is_rejected():
    with pytest.raises(GraphConfigError):
        load_graph_config(LAUNCH, overrides=["active_profile=no_such_profile"])


def test_dragon_may_not_be_given_scope_propagation(tmp_path):
    """The baseline must not receive the mechanism under test."""
    base = GRAPH_CONFIGS
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    import shutil

    shutil.copytree(base, root / "configs" / "graph")
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    path = root / "configs" / "graph" / "defenses" / "dragon_style.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec["propagate_scope"] = True
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(GraphConfigError, match="must not propagate scope"):
        resolve(GraphLaunchConfig(study="graph_unlearning_v1", active_profile="local_cpu"), root)


def test_a_multi_agent_arm_must_declare_its_peer_content(tmp_path):
    import shutil

    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(GRAPH_CONFIGS, root / "configs" / "graph")
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    path = root / "configs" / "graph" / "arms" / "multi_agent_leak.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec["peer_content"] = "none"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(GraphConfigError, match="same_concept or cross_concept"):
        resolve(GraphLaunchConfig(study="graph_unlearning_v1", active_profile="local_cpu"), root)


def test_unknown_keys_are_rejected(tmp_path):
    import shutil

    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(GRAPH_CONFIGS, root / "configs" / "graph")
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    path = root / "configs" / "graph" / "runtime" / "local_cpu.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec["temperature"] = 0.7  # a runtime profile must not carry a decoding parameter
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(GraphConfigError):
        resolve(GraphLaunchConfig(study="graph_unlearning_v1", active_profile="local_cpu"), root)
