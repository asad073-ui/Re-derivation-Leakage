"""The forget policy is never derived from the evaluation cohort. GU-0027.

THE BUG. `GraphRunner` built its concept registry and its deleted baseline memory from
`self.items` — the questions being asked. That is correct for a forget cohort and
catastrophic for a retain one: `phase: retain_utility` would have registered the 45
retained authors as forgotten concepts, so the guard would have fired on exactly the
behaviour the run exists to measure. Retain utility would have measured over-blocking of
concepts the run itself declared forbidden, and the detector's false-positive rate would
have been computed against its own positives. Both numbers would have looked plausible.

Nothing downstream can detect that, which is why the checks are structural and why they
refuse rather than warn.
"""

from __future__ import annotations

import pytest

from rdl.cli.graph_common import _forget_policy_phase, resolve_run_cohorts
from rdl.graph.config import GraphLaunchConfig, load_graph_config, resolve
from rdl.paths import repo_root
from rdl.studies.graph_leak.cohort import (
    Cohort,
    CohortError,
    CohortItem,
    assert_forget_policy_cohort,
    assert_policy_excludes_evaluation_concepts,
    load_cohort,
    resolve_cohort,
)
from rdl.studies.graph_leak.runner import GraphRunner

CONFIGS = repo_root() / "configs" / "graph"
LAUNCH = CONFIGS / "launch.yaml"
COHORTS = repo_root() / "data" / "cohorts" / "graph_unlearning_v1"


def _cohort(split: str, dataset_config: str, concepts: tuple[str, ...]) -> Cohort:
    return Cohort(
        split=split,
        dataset="TOFU",
        dataset_config=dataset_config,
        dataset_revision="deadbeef",
        items=tuple(
            CohortItem(
                item_id=f"{dataset_config}-{i:04d}",
                concept_id=concept,
                index=i,
                question_sha256="q" * 64,
                answer_sha256="a" * 64,
            )
            for i, concept in enumerate(concepts)
        ),
    )


# ------------------------------------------------------------------ the cohorts --


def test_the_committed_retain_cohort_is_recognised_as_retain():
    retain = load_cohort(COHORTS / "retain_utility.json")
    engineering = load_cohort(COHORTS / "engineering.json")
    assert retain.is_retain is True
    assert engineering.is_retain is False


def test_a_retain_cohort_is_refused_as_the_forget_policy():
    retain = load_cohort(COHORTS / "retain_utility.json")
    with pytest.raises(CohortError, match="cannot be the forget-policy cohort"):
        assert_forget_policy_cohort(retain)


def test_a_forget_cohort_is_accepted_as_the_forget_policy():
    engineering = load_cohort(COHORTS / "engineering.json")
    assert assert_forget_policy_cohort(engineering) is engineering


def test_the_committed_retain_and_forget_cohorts_share_no_author():
    """retain90 and forget10 are disjoint author sets, and the loader proves it."""
    retain = load_cohort(COHORTS / "retain_utility.json")
    engineering = load_cohort(COHORTS / "engineering.json")
    assert not set(retain.concept_ids) & set(engineering.concept_ids)
    assert_policy_excludes_evaluation_concepts(engineering, retain)


def test_an_overlapping_retain_cohort_is_refused():
    shared = ("author-a", "author-b")
    policy = _cohort("engineering", "forget10", shared)
    evaluation = _cohort("retain_utility", "retain90", shared)
    with pytest.raises(CohortError, match="shares concepts"):
        assert_policy_excludes_evaluation_concepts(policy, evaluation)


# ------------------------------------------------------------------ the runner --


def test_the_registry_holds_only_forget_authors_never_retain_authors(tofu_items, graph_backend):
    """THE regression test. Retain authors must not appear in the runtime registry.

    The registry's concepts are the concepts the guard withholds. A retain author in
    there is the system being told to forget a question it is required to answer.
    """
    cfg = load_graph_config(LAUNCH)
    forget = load_cohort(COHORTS / "cpu_stub.json").limited(4)
    forget_items = resolve_cohort(forget, tofu_items)

    # A retain cohort built over the SAME fixture items but different concept ids, so the
    # only thing distinguishing it is its role in the run.
    retain = Cohort(
        split="retain_utility",
        dataset="fixture",
        dataset_config="retain90",
        dataset_revision=None,
        items=tuple(
            CohortItem(
                item_id=entry.item_id,
                concept_id=f"retain-{entry.concept_id}",
                index=entry.index,
                question_sha256=entry.question_sha256,
                answer_sha256=entry.answer_sha256,
                usage="retain_utility",
            )
            for entry in forget.items
        ),
    )
    retain_items = resolve_cohort(retain, tofu_items)

    runner = GraphRunner(
        cfg=cfg,
        items=retain_items,
        cohort=retain,
        policy_cohort=forget,
        policy_items=forget_items,
        backend=graph_backend,
        output=repo_root() / "runs" / "graph" / "never-written",
        tokenizer_revision="stub",
    )

    registry_concepts = set(runner.registry.ids())
    assert registry_concepts == set(forget.concept_ids)
    assert not registry_concepts & set(retain.concept_ids)
    assert not any(c.startswith("retain-") for c in registry_concepts)


def test_a_retain_cohort_alone_cannot_start_a_run(tofu_items, graph_backend):
    """No policy cohort plus a retain evaluation cohort is a refusal, not a fallback."""
    cfg = load_graph_config(LAUNCH)
    forget = load_cohort(COHORTS / "cpu_stub.json").limited(4)
    retain = Cohort(
        split="retain_utility",
        dataset="fixture",
        dataset_config="retain90",
        dataset_revision=None,
        items=forget.items,
    )
    with pytest.raises(CohortError, match="cannot be the forget-policy cohort"):
        GraphRunner(
            cfg=cfg,
            items=resolve_cohort(retain, tofu_items),
            cohort=retain,
            backend=graph_backend,
            output=repo_root() / "runs" / "graph" / "never-written",
            tokenizer_revision="stub",
        )


def test_the_manifest_records_both_cohort_fingerprints(tofu_items, graph_backend, tmp_path):
    cfg = load_graph_config(LAUNCH)
    cohort = load_cohort(COHORTS / "cpu_stub.json").limited(4)
    manifest = GraphRunner(
        cfg=cfg,
        items=resolve_cohort(cohort, tofu_items),
        cohort=cohort,
        backend=graph_backend,
        output=tmp_path / "run",
        tokenizer_revision="stub",
    ).run()

    assert manifest["forget_policy_fingerprint"] == cohort.fingerprint()
    assert manifest["cohort_fingerprint"] == cohort.fingerprint()
    assert manifest["forget_policy_cohort"]["n_concepts"] == len(cohort.concept_ids)
    assert manifest["evaluation_cohort"]["is_retain"] is False
    assert manifest["retain_evaluation"] is False
    assert (tmp_path / "run" / "FORGET_POLICY_COHORT.json").exists()


def test_the_forget_policy_is_immutable_on_resume():
    assert "forget_policy_fingerprint" in GraphRunner.IMMUTABLE_ON_RESUME
    assert "forget_policy_split" in GraphRunner.IMMUTABLE_ON_RESUME


# ------------------------------------------------------------ the launch files --


def test_the_retain_launch_file_names_the_engineering_forget_policy():
    cfg = load_graph_config(CONFIGS / "launch_rtx3090_retain.yaml")
    assert cfg.phase == "retain_utility"
    assert cfg.launch.forget_policy_phase == "engineering"
    assert _forget_policy_phase(cfg) == "engineering"


def test_the_engineering_launch_file_is_its_own_forget_policy():
    cfg = load_graph_config(CONFIGS / "launch_rtx3090_engineering.yaml")
    assert cfg.phase == "engineering"
    assert cfg.launch.forget_policy_phase is None
    assert _forget_policy_phase(cfg) == "engineering"


def test_the_two_rtx_launch_files_describe_the_same_study_and_profile():
    """Same science, same hardware; only the cohort roles differ."""
    engineering = load_graph_config(CONFIGS / "launch_rtx3090_engineering.yaml")
    retain = load_graph_config(CONFIGS / "launch_rtx3090_retain.yaml")
    assert engineering.profile.name == retain.profile.name == "rtx3090_1b"
    assert engineering.model.name == retain.model.name
    assert engineering.topology.name == retain.topology.name
    assert [a.name for a in engineering.arms] == [a.name for a in retain.arms]


def test_a_retain_phase_without_a_forget_policy_phase_is_refused():
    import typer

    launch = GraphLaunchConfig(
        study="graph_unlearning_v1", active_profile="local_cpu", phase="retain_utility"
    )
    cfg = resolve(launch)
    with pytest.raises(typer.BadParameter, match="cannot also supply the forget policy"):
        _forget_policy_phase(cfg)


def test_retain_utility_may_not_be_named_as_the_forget_policy():
    with pytest.raises(ValueError, match="cannot be 'retain_utility'"):
        GraphLaunchConfig(
            study="graph_unlearning_v1",
            active_profile="local_cpu",
            forget_policy_phase="retain_utility",
        )


def test_the_limit_narrows_the_questions_and_never_the_forget_policy(monkeypatch):
    """A question budget is not a statement about what the deployment forgot."""
    cfg = load_graph_config(LAUNCH, overrides=["phase=engineering"])

    seen: list[tuple[str | None, int | None]] = []
    real = resolve_run_cohorts.__globals__["resolve_cohort_items"]

    def spy(config, **kwargs):
        seen.append((kwargs.get("phase"), kwargs.get("limit")))
        return real(config, **kwargs)

    monkeypatch.setitem(resolve_run_cohorts.__globals__, "resolve_cohort_items", spy)
    cohorts = resolve_run_cohorts(cfg, limit=2)

    assert len(cohorts.evaluation_items) == 2
    # One resolution only, because evaluation and policy are the same cohort here — and
    # crucially it is the EVALUATION resolution that carried the limit.
    assert seen == [(None, 2)]
    assert cohorts.policy is cohorts.evaluation
