"""The GPU-readiness gate, as executable checks.

Every test here corresponds to a blocker found in review of the first graph PR. They
exist because each failure mode is silent: a run under any of them completes, writes a
manifest, and produces numbers that look exactly like the right ones.
"""

from __future__ import annotations

import json

import pytest
import yaml

from rdl.graph.config import (
    GraphModelConfig,
    assert_model_provenance,
    load_graph_config,
)
from rdl.graph.validation import GraphConfigError
from rdl.paths import repo_root

LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"
MODELS = repo_root() / "configs" / "graph" / "models"
RUNTIME = repo_root() / "configs" / "graph" / "runtime"


# --------------------------------------------------------------- checkpoint identity --


def test_the_rtx_target_is_the_unlearned_checkpoint_not_the_full_one():
    """The blocker: the graph config pointed at `..._full` with `revision: null`.

    A run under that config measures leakage from a model that never forgot anything.
    """
    cfg = load_graph_config(LAUNCH, overrides=["active_profile=rtx3090_1b"])
    assert cfg.model.repo_id == "OptimAI-Lab/TOFU-forget10_RULE-NPO"
    assert "full" not in (cfg.model.repo_id or "").lower()
    assert cfg.model.expect_unlearned


def test_the_rtx_target_pins_model_and_tokenizer():
    cfg = load_graph_config(LAUNCH, overrides=["active_profile=rtx3090_1b"])
    assert cfg.model.revision == "afe117e41a876f815bbd0f336d5036ced666ab06"
    assert cfg.model.tokenizer_revision == cfg.model.revision
    assert cfg.model.pinned
    assert_model_provenance(cfg.model)


def test_the_graph_target_matches_the_two_agent_repos_pinned_checkpoint():
    """One checkpoint identity across both experiment families, or they are not comparable."""
    old = yaml.safe_load(
        (repo_root() / "configs" / "models" / "tofu_forget10_rule_npo.yaml").read_text(
            encoding="utf-8"
        )
    )
    new = yaml.safe_load((MODELS / "rule_npo_1b.yaml").read_text(encoding="utf-8"))
    assert new["repo_id"] == old["repo_id"]
    assert new["revision"] == old["revision"]


def test_a_config_named_for_an_unlearning_method_cannot_name_a_base_checkpoint():
    with pytest.raises(ValueError, match="NOT unlearned"):
        GraphModelConfig(
            name="rule_npo_1b",
            kind="hf",
            repo_id="open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
            revision="abc",
            tokenizer_revision="abc",
            expect_unlearned=True,
        )


def test_a_method_named_config_must_opt_into_the_provenance_checks():
    with pytest.raises(ValueError, match="expect_unlearned"):
        GraphModelConfig(name="rule_npo_1b", kind="hf", repo_id="some/repo")


@pytest.mark.parametrize("marker", ["_full", "-full", "_base", "retain90"])
def test_every_not_unlearned_marker_is_caught(marker):
    with pytest.raises(ValueError, match="NOT unlearned"):
        GraphModelConfig(
            name="npo_target",
            kind="hf",
            repo_id=f"org/model{marker}",
            revision="abc",
            tokenizer_revision="abc",
            expect_unlearned=True,
        )


def test_an_unpinned_target_cannot_start_a_run():
    """The 7B target is deliberately unselected, and must refuse rather than default."""
    cfg = load_graph_config(LAUNCH, overrides=["active_profile=h100_7b"])
    with pytest.raises(GraphConfigError, match="cannot start a run"):
        assert_model_provenance(cfg.model)


def test_every_shipped_graph_model_is_either_stub_or_declares_unlearning():
    for path in sorted(MODELS.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        if spec.get("kind") == "stub":
            continue
        assert spec.get("expect_unlearned") is True, path.name


# ------------------------------------------------------------------ plan == run cost --


def test_plan_and_run_share_one_sample_budget_function():
    from rdl.cli import plan_graph_run, run_graph
    from rdl.cli.graph_common import apply_sample_budget

    assert plan_graph_run.apply_sample_budget is apply_sample_budget
    assert run_graph.apply_sample_budget is apply_sample_budget


def test_the_rtx_profile_with_two_samples_plans_the_smoke_not_the_full_run():
    """The blocker: `graph-plan` ignored --n-samples, so it printed 2 688, not 168."""
    from rdl.cli.graph_common import apply_sample_budget
    from rdl.graph.scheduler import planned_generations

    cfg = load_graph_config(LAUNCH, overrides=["active_profile=rtx3090_1b"])
    assert cfg.profile.sampling.n_samples == 32

    reduced = apply_sample_budget(cfg, 2)
    assert reduced.profile.sampling.n_samples == 2
    assert reduced.profile.sampling.k_values == (1, 2)

    arms = tuple(a.name for a in reduced.arms)
    plan = planned_generations(
        reduced.topology,
        n_items=4,
        n_samples=reduced.profile.sampling.n_samples,
        arms=arms,
        single_agent_arms=("single_agent",),
    )
    assert plan["planned_generations"] == 168
    assert plan["planned_trajectories"] == 40


def test_reducing_below_the_primary_k_marks_the_run_unreportable():
    from rdl.cli.graph_common import apply_sample_budget

    cfg = load_graph_config(LAUNCH, overrides=["active_profile=rtx3090_1b"])
    assert cfg.profile.reportable
    assert not apply_sample_budget(cfg, 2).profile.reportable
    # The full budget still reaches primary_k and stays reportable.
    assert apply_sample_budget(cfg, 32).profile.reportable


def test_the_budget_may_only_be_reduced():
    import typer

    from rdl.cli.graph_common import apply_sample_budget

    cfg = load_graph_config(LAUNCH, overrides=["active_profile=rtx3090_1b"])
    with pytest.raises(typer.BadParameter, match="may reduce, never raise"):
        apply_sample_budget(cfg, 128)


# ------------------------------------------------------------------ dataset pinning --


def test_load_tofu_passes_the_revision_to_load_dataset():
    """The blocker: the revision was recorded in the manifest and never downloaded."""
    import inspect

    from rdl.eval import tofu_data

    assert "revision" in inspect.signature(tofu_data.load_tofu).parameters
    assert "revision" in inspect.signature(tofu_data.load_items).parameters
    source = inspect.getsource(tofu_data.load_tofu)
    assert "revision=revision" in source, "the parameter must reach load_dataset"


def test_freezing_against_real_data_requires_a_revision():
    import inspect

    from rdl.cli import freeze_graph_cohort

    assert "--dataset-revision is required" in inspect.getsource(
        freeze_graph_cohort.freeze_graph_cohort
    )


# ------------------------------------------------------------------ detector honesty --


def test_no_runtime_profile_declares_a_detector_device():
    """The blocker: `detector_device: cuda` named a code path that does not exist."""
    for path in sorted(RUNTIME.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "detector_device" not in spec, path.name
        assert "detector_batch_size" not in spec, path.name


def test_the_detector_backend_is_declared_where_the_science_is():
    cfg = load_graph_config(LAUNCH)
    assert cfg.study.detector.backend == "hashing64"
    assert cfg.study.detector.status == "diagnostic"


# --------------------------------------------------------------------------- vLLM --


def test_vllm_is_pinned_for_the_gpu_environment():
    text = (repo_root() / "requirements-gpu-ampere.txt").read_text(encoding="utf-8")
    pins = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("vllm") and "==" in line
    ]
    assert pins, "the RTX profile uses the vllm backend and nothing installs it"
    assert len(pins) == 1


def test_the_vllm_backend_accepts_and_forwards_the_tokenizer_revision():
    from rdl.runtime.vllm_backend import VllmBackend

    backend = VllmBackend(
        "org/model", revision="modelsha", tokenizer="org/model", tokenizer_revision="toksha"
    )
    assert backend._engine_kwargs["tokenizer_revision"] == "toksha"
    assert backend._engine_kwargs["revision"] == "modelsha"


def test_the_vllm_backend_never_holds_a_token():
    """Authentication comes from the environment so a credential cannot reach a manifest."""
    import inspect

    from rdl.runtime.vllm_backend import VllmBackend

    params = set(inspect.signature(VllmBackend.__init__).parameters)
    assert not {"token", "hf_token", "use_auth_token"} & params

    credentials = {"token", "hf_token", "use_auth_token", "api_key", "auth"}
    backend = VllmBackend("org/model", revision="r", tokenizer_revision="t")
    # The engine kwargs are the closest thing to a serialisable record of how the model
    # was loaded, so a secret must never be among them.
    assert not credentials & set(backend._engine_kwargs)
    assert not credentials & set(vars(backend))
    assert not credentials & set(backend.resolved_revisions())


def test_resolved_revisions_are_reported_separately_from_requested_ones():
    from rdl.runtime.vllm_backend import VllmBackend

    backend = VllmBackend("org/model", revision="asked", tokenizer_revision="asked-tok")
    resolved = backend.resolved_revisions()
    assert resolved["requested_model_revision"] == "asked"
    assert resolved["requested_tokenizer_revision"] == "asked-tok"
    # Either it resolved them or it recorded why it could not; never silence.
    assert "resolved_model_sha" in resolved or "error" in resolved


# ------------------------------------------------------------------------ protocols --


def test_the_study_must_enable_the_graph_flow_protocol():
    cfg = load_graph_config(LAUNCH)
    assert "graph_flow" in cfg.study.protocols
    assert "end_to_end_safety" in cfg.study.protocols


def test_dropping_graph_flow_is_refused(tmp_path):
    import shutil

    from rdl.graph.config import GraphLaunchConfig, resolve

    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    shutil.copytree(repo_root() / "configs" / "graph", root / "configs" / "graph")
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    path = root / "configs" / "graph" / "studies" / "graph_unlearning_v1.yaml"
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    spec["protocols"] = ["end_to_end_safety"]
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(GraphConfigError, match="graph_flow must be enabled"):
        resolve(GraphLaunchConfig(study="graph_unlearning_v1", active_profile="local_cpu"), root)


# ------------------------------------------------------------------ retain utility --


def test_a_retain_cohort_exists_and_is_reachable_by_phase():
    cfg = load_graph_config(LAUNCH)
    path = repo_root() / cfg.study.data.cohort_dir / cfg.study.data.retain_utility_manifest
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset_config"] == "retain90"
    assert payload["hashes_frozen"]


def test_the_retain_phase_selects_the_retain_manifest():
    from rdl.cli.graph_common import _MANIFEST_FOR_PHASE

    assert _MANIFEST_FOR_PHASE["retain_utility"] == "retain_utility_manifest"


# ------------------------------------------------------------------- dragon bracket --


def test_the_primary_dragon_arm_is_the_prompt_guard():
    cfg = load_graph_config(LAUNCH)
    assert cfg.defenses["dragon_style"].guard_action == "guard_prompt"


def test_the_refusal_upper_bound_exists_but_is_not_a_default_arm():
    cfg = load_graph_config(LAUNCH)
    assert "multi_agent_dragon_refuse" not in [a.name for a in cfg.arms]
    spec = yaml.safe_load(
        (repo_root() / "configs/graph/defenses/dragon_style_refuse.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert spec["guard_action"] == "refuse"
    assert "upper bound" in spec["description"]
