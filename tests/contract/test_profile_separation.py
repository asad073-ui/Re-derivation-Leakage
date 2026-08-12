"""A machine changes the profile. It must not be able to change the experiment."""

from __future__ import annotations

import yaml

from rdl.graph.config import load_graph_config
from rdl.paths import repo_root

LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"
RUNTIME_DIR = repo_root() / "configs" / "graph" / "runtime"
STUDY = repo_root() / "configs" / "graph" / "studies" / "graph_unlearning_v1.yaml"

# Everything that defines the experiment. None of it may appear in a runtime profile.
SCIENTIFIC_KEYS = {
    "temperature",
    "top_p",
    "top_k",
    "max_new_tokens",
    "primary_k",
    "base_seed",
    "arms",
    "challenge_modes",
    "controlled_challenges",
    "memory",
    "detector",
    "evaluation",
    "data",
    "primary_topology",
    "phase",
}


def test_no_runtime_profile_carries_a_scientific_key():
    for path in sorted(RUNTIME_DIR.glob("*.yaml")):
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        overlap = SCIENTIFIC_KEYS & set(profile)
        assert not overlap, f"{path.name} carries scientific keys {sorted(overlap)}"


def test_launch_file_has_exactly_the_two_switches():
    launch = yaml.safe_load(LAUNCH.read_text(encoding="utf-8"))
    assert set(launch) <= {"study", "active_profile", "phase"}
    assert launch["study"] == "graph_unlearning_v1"


def test_switching_profile_preserves_every_scientific_setting():
    cpu = load_graph_config(LAUNCH)
    gpu = load_graph_config(LAUNCH, overrides=["active_profile=rtx3090_1b"])
    h100 = load_graph_config(LAUNCH, overrides=["active_profile=h100_7b"])
    for other in (gpu, h100):
        assert other.study.sampling == cpu.study.sampling
        assert other.study.memory == cpu.study.memory
        assert other.study.detector == cpu.study.detector
        assert other.study.evaluation == cpu.study.evaluation
        assert other.arms == cpu.arms
        assert other.defenses == cpu.defenses
        assert other.topology == cpu.topology


def test_both_gpu_profiles_use_the_same_dragon_implementation():
    """Otherwise model size and baseline strength become confounded."""
    gpu = load_graph_config(LAUNCH, overrides=["active_profile=rtx3090_1b"])
    h100 = load_graph_config(LAUNCH, overrides=["active_profile=h100_7b"])
    assert gpu.defenses["dragon_style"] == h100.defenses["dragon_style"]


def test_all_profiles_share_one_model_handle():
    for profile in ("local_cpu", "rtx3090_1b", "h100_7b"):
        cfg = load_graph_config(LAUNCH, overrides=[f"active_profile={profile}"])
        assert cfg.profile.agents.share_model_handle
        assert {n.model for n in cfg.topology.nodes} == {"primary"}


def test_h100_derives_lower_k_from_one_sample_bank():
    """128 draws once, every k derived from them — not one experiment per k."""
    h100 = load_graph_config(LAUNCH, overrides=["active_profile=h100_7b"])
    k_values = h100.profile.sampling.k_values
    assert max(k_values) == h100.profile.sampling.n_samples == 128
    assert h100.study.sampling.primary_k in k_values


def test_the_study_declares_which_profiles_are_legal():
    study = yaml.safe_load(STUDY.read_text(encoding="utf-8"))
    assert set(study["profiles"]) == {"local_cpu", "rtx3090_1b", "h100_7b"}
