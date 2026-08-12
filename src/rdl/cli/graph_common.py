"""Shared wiring for the ``graph-*`` commands: config, cohort, backend.

Kept out of the individual commands so that ``plan``, ``run``, ``score`` and ``report``
resolve the identical configuration and cohort. A planning command that resolved things
slightly differently from the runner would predict a cost for an experiment nobody ran.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import typer

from ..eval.tofu_data import TofuItem, load_items
from ..graph.config import ResolvedGraphConfig, assert_model_provenance, load_graph_config
from ..graph.validation import GraphConfigError
from ..models.stub import StubLM
from ..paths import repo_root
from ..runtime.backend import GenerationBackend
from ..runtime.stub_backend import StubBackend
from ..studies.graph_leak.cohort import (
    Cohort,
    CohortError,
    assert_forget_policy_cohort,
    assert_policy_excludes_evaluation_concepts,
    load_cohort,
    resolve_cohort,
)

__all__ = [
    "DEFAULT_LAUNCH",
    "RunCohorts",
    "apply_sample_budget",
    "build_backend",
    "load_config_or_fail",
    "resolve_cohort_items",
    "resolve_run_cohorts",
]

DEFAULT_LAUNCH = "configs/graph/launch.yaml"

T = TypeVar("T")

_MANIFEST_FOR_PHASE = {
    "smoke": "smoke_manifest",
    "engineering": "engineering_manifest",
    "discovery": "discovery_manifest",
    "validation": "validation_manifest",
    "retain_utility": "retain_utility_manifest",
}


def apply_sample_budget(cfg: ResolvedGraphConfig, n_samples: int | None) -> ResolvedGraphConfig:
    """Reduce the profile's sample budget, identically for `graph-plan` and `graph-run`.

    ONE implementation, deliberately. `graph-run` used to accept `--n-samples` while
    `graph-plan` did not, so planning a smoke against the RTX profile printed the cost
    of the profile's full 32 draws — 2 688 graph generations — while the run that
    followed did 168. A plan that describes a different experiment from the run is worse
    than no plan, because it is trusted.

    Reducing only: a profile's budget is the hardware's ceiling, not a suggestion.
    """
    if n_samples is None:
        return cfg
    if n_samples > cfg.profile.sampling.n_samples:
        raise typer.BadParameter(
            f"--n-samples {n_samples} exceeds the profile's budget of "
            f"{cfg.profile.sampling.n_samples}. It may reduce, never raise."
        )
    budget = cfg.profile.sampling.model_copy(
        update={
            "n_samples": n_samples,
            "k_values": tuple(k for k in cfg.profile.sampling.k_values if k <= n_samples) or (1,),
        }
    )
    profile = cfg.profile.model_copy(update={"sampling": budget})
    # A reduced budget that can no longer reach the study's primary k makes the run a
    # wiring check, and it has to say so rather than silently reporting at a smaller k.
    if cfg.study.sampling.primary_k not in budget.k_values:
        profile = profile.model_copy(update={"reportable": False})
    return cfg.model_copy(update={"profile": profile})


def load_config_or_fail(
    launch: Path,
    *,
    overrides: Sequence[str] | None = None,
    topology: str | None = None,
    root: Path | None = None,
) -> ResolvedGraphConfig:
    try:
        return load_graph_config(launch, root, overrides=overrides, topology=topology)
    except GraphConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc


def resolve_cohort_items(
    cfg: ResolvedGraphConfig,
    *,
    root: Path | None = None,
    fixture: Path | None = None,
    cohort_path: Path | None = None,
    phase: str | None = None,
    token: str | None = None,
    limit: int | None = None,
) -> tuple[Cohort, list[TofuItem]]:
    """Load one frozen cohort and attach real question/answer text.

    ``phase`` selects which manifest, defaulting to the run's own phase; the forget-policy
    resolution passes a different one so a retain run can load the forget cohort that
    defines its policy. The exclusion list applies either way.

    The validation split additionally refuses to load while its concepts intersect the
    exclusion list — which, on the current checkpoint, they always do. That refusal is
    the point: 'the next 50 questions' are new questions, not new forgotten concepts.
    """
    base = (root or repo_root()) / cfg.study.data.cohort_dir
    active_phase = phase or cfg.phase
    manifest_key = _MANIFEST_FOR_PHASE[active_phase]
    path = cohort_path or base / getattr(cfg.study.data, manifest_key)
    exclusions = base / cfg.study.data.exclusion_manifest

    try:
        cohort = load_cohort(
            path,
            exclusions_path=exclusions if exclusions.exists() else None,
            require_frozen=cfg.study.data.require_frozen_hashes,
            forbid_excluded_items=True,
            forbid_excluded_concepts=active_phase == "validation",
        )
    except CohortError as exc:
        raise typer.BadParameter(str(exc)) from exc

    cohort = cohort.limited(limit)
    source, _provenance = load_items(
        dataset="stub" if fixture else "tofu",
        split=cohort.dataset_config,
        n_items=None,
        fixture=fixture,
        allow_fixture=bool(fixture),
        token=token,
        # The cohort's own frozen revision drives the download, so the run loads exactly
        # the commit its manifest names rather than whatever the branch head holds.
        revision=cohort.dataset_revision,
    )
    try:
        items = resolve_cohort(cohort, source)
    except CohortError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return cohort, items


@dataclass(frozen=True)
class RunCohorts:
    """The two cohorts one run needs, resolved together so they cannot diverge.

    ``evaluation`` supplies the questions. ``policy`` supplies the forget policy — the
    concept registry and the deleted baseline memory. They are the same object for every
    forget-cohort run and different objects for a retain-utility run.
    """

    evaluation: Cohort
    evaluation_items: list[TofuItem]
    policy: Cohort
    policy_items: list[TofuItem]

    @property
    def separated(self) -> bool:
        return self.policy.fingerprint() != self.evaluation.fingerprint()

    def to_dict(self) -> dict:
        return {
            "evaluation_cohort_split": self.evaluation.split,
            "evaluation_cohort_fingerprint": self.evaluation.fingerprint(),
            "evaluation_cohort_dataset_config": self.evaluation.dataset_config,
            "evaluation_n_items": len(self.evaluation_items),
            "evaluation_n_concepts": len(self.evaluation.concept_ids),
            "evaluation_is_retain": self.evaluation.is_retain,
            "forget_policy_split": self.policy.split,
            "forget_policy_fingerprint": self.policy.fingerprint(),
            "forget_policy_dataset_config": self.policy.dataset_config,
            "forget_policy_n_items": len(self.policy_items),
            "forget_policy_n_concepts": len(self.policy.concept_ids),
            "cohorts_separated": self.separated,
        }


def _forget_policy_phase(cfg: ResolvedGraphConfig) -> str:
    """Which phase's cohort defines what was forgotten.

    A retain phase has no forget cohort of its own, so it must name one. Falling back to
    the evaluated phase there is precisely the bug this function exists to make
    impossible: it would build the registry from retain questions.
    """
    declared = cfg.launch.forget_policy_phase
    if declared:
        return str(declared)
    if cfg.phase == "retain_utility":
        raise typer.BadParameter(
            "phase 'retain_utility' evaluates questions the system is SUPPOSED to answer, "
            "so it cannot also supply the forget policy. Set `forget_policy_phase` in the "
            "launch file to the frozen forget cohort this deployment forgot, e.g.\n"
            "    forget_policy_phase: engineering\n"
            "Without it the concept registry and the deleted baseline memory would be "
            "built from retain authors, and every retain-utility and false-positive "
            "number would be invalid."
        )
    return str(cfg.phase)


def resolve_run_cohorts(
    cfg: ResolvedGraphConfig,
    *,
    root: Path | None = None,
    fixture: Path | None = None,
    cohort_path: Path | None = None,
    policy_cohort_path: Path | None = None,
    token: str | None = None,
    limit: int | None = None,
) -> RunCohorts:
    """Resolve the evaluation cohort and the forget-policy cohort for one run.

    ``limit`` narrows the EVALUATION cohort only. A limit is a question budget; it is not
    a statement that the deployment forgot fewer things, and a registry that shrank with
    it would make the guard's scope depend on how much GPU time was bought.
    """
    evaluation, evaluation_items = resolve_cohort_items(
        cfg, root=root, fixture=fixture, cohort_path=cohort_path, token=token, limit=limit
    )
    policy_phase = _forget_policy_phase(cfg)

    if policy_cohort_path is None and policy_phase == str(cfg.phase) and cohort_path is not None:
        # An explicit --cohort with no separate policy cohort keeps both roles on that
        # one manifest, which is right for the CPU stub and for any ad-hoc forget split.
        policy, policy_items = evaluation, evaluation_items
    elif policy_cohort_path is None and policy_phase == str(cfg.phase):
        policy, policy_items = evaluation, evaluation_items
    else:
        policy, policy_items = resolve_cohort_items(
            cfg,
            root=root,
            fixture=fixture,
            cohort_path=policy_cohort_path,
            phase=policy_phase,
            token=token,
            limit=None,
        )

    try:
        assert_forget_policy_cohort(policy)
        assert_policy_excludes_evaluation_concepts(policy, evaluation)
    except CohortError as exc:
        raise typer.BadParameter(str(exc)) from exc

    return RunCohorts(
        evaluation=evaluation,
        evaluation_items=evaluation_items,
        policy=policy,
        policy_items=policy_items,
    )


def option_value(value: object, fallback: T) -> T:
    """Resolve a Typer default when a command is called as a plain Python function.

    The tests in this repository call the command functions directly rather than through
    a CliRunner, which is a deliberate convention: it keeps them fast and lets them assert
    on return values. The cost is that any parameter the caller omits arrives as a
    ``typer.models.OptionInfo`` sentinel rather than its default, and a new option that
    reaches a comparison or a JSON payload then fails in a way that has nothing to do with
    the behaviour under test. One helper, applied at the top of the commands that grew
    new options, is cheaper than converting every caller.
    """
    return fallback if isinstance(value, typer.models.OptionInfo) else cast(T, value)


def stub_source_items(cohorts: RunCohorts) -> list[TofuItem]:
    """Every item the stub backend may be asked about, evaluation and policy alike.

    Only the stub backend uses this: it answers from a lookup table, and a retain run
    whose table held the retain questions alone would have no answer for the forgotten
    content the memory baseline is built from.
    """
    seen: dict[str, TofuItem] = {}
    for item in [*cohorts.policy_items, *cohorts.evaluation_items]:
        seen.setdefault(item.item_id, item)
    return list(seen.values())


# THE PREFLIGHT IS 2 ITEMS x 1 SAMPLE. One item cannot produce a cross-concept control:
# `cross_author_mapping` needs two authors to rotate between, so `--limit 1` raises deep
# inside the runner with a message about C3S. The runbook said 1x1 for months and the
# operator had to discover 2x1 on a rented GPU. This turns that into a parameter error
# with the right number in it, before anything is loaded.
MIN_ITEMS_FOR_CROSS_CONCEPT_CONTROL = 2


def assert_control_arm_has_enough_items(
    cfg: ResolvedGraphConfig, items: Sequence[TofuItem]
) -> None:
    needs_control = any(arm.peer_content == "cross_concept" for arm in cfg.arms)
    if not needs_control or len(items) >= MIN_ITEMS_FOR_CROSS_CONCEPT_CONTROL:
        return
    raise typer.BadParameter(
        f"{len(items)} evaluation item(s), but this study includes a cross-concept "
        "control arm, which needs at least "
        f"{MIN_ITEMS_FOR_CROSS_CONCEPT_CONTROL} items from different concepts to rotate "
        "between. The minimal preflight is 2 items x 1 sample:\n"
        "    --limit 2 --n-samples 1"
    )


def build_backend(
    cfg: ResolvedGraphConfig,
    items: Sequence[TofuItem],
    *,
    token: str | None = None,
    unlearned: bool = True,
) -> tuple[GenerationBackend, str | None]:
    """One physical handle, shared by every logical agent profile.

    Returns ``(backend, tokenizer_revision)``. The tokenizer revision is part of the
    response-cache key, so it has to come back with the backend rather than be looked up
    again somewhere else.
    """
    profiles = {node.model for node in cfg.topology.nodes}
    backend_kind = cfg.profile.runtime.backend

    # The gate that would have caught `..._full` with `revision: null`. Checked for every
    # real backend, before a single byte is downloaded.
    if backend_kind != "stub":
        try:
            assert_model_provenance(cfg.model)
        except GraphConfigError as exc:
            raise typer.BadParameter(str(exc)) from exc

    if backend_kind == "stub":
        answers = {item.question: item.answer for item in items}
        stub = StubLM(
            answers,
            knowledge_mask=list(answers) if unlearned else None,
            model_id=cfg.model.name,
            qid_to_question={item.item_id: item.question for item in items},
        )
        # Every profile maps to the SAME object: five logical agents, one backend.
        return StubBackend(dict.fromkeys(profiles, stub)), "stub"

    if backend_kind == "vllm":
        from ..runtime.vllm_backend import VllmBackend

        if not cfg.model.repo_id:
            raise typer.BadParameter("the vllm backend needs model.repo_id")
        if token:
            # vLLM authenticates from the environment; passing a token as an argument
            # would let it reach engine kwargs and from there a manifest.
            os.environ.setdefault("HF_TOKEN", token)
        return (
            VllmBackend(
                cfg.model.repo_id,
                profiles=tuple(sorted(profiles)),
                revision=cfg.model.revision,
                tokenizer=cfg.model.tokenizer_repo_id,
                tokenizer_revision=cfg.model.tokenizer_revision,
                dtype=cfg.profile.runtime.dtype,
                gpu_memory_utilization=cfg.profile.runtime.gpu_memory_utilization,
                max_model_len=cfg.profile.runtime.max_model_len,
                max_num_seqs=cfg.profile.runtime.max_num_seqs,
                max_num_batched_tokens=cfg.profile.runtime.max_num_batched_tokens,
                tensor_parallel_size=cfg.profile.runtime.tensor_parallel_size,
            ),
            cfg.model.tokenizer_revision,
        )

    if backend_kind == "transformers":
        from ..config import ModelConfig
        from ..hardware import detect
        from ..models.loader import load_lm
        from ..runtime.transformers_backend import TransformersBackend

        if not cfg.model.repo_id:
            raise typer.BadParameter("the transformers backend needs model.repo_id")
        model_cfg = ModelConfig(
            name=cfg.model.name,
            kind="hf",
            repo_id=cfg.model.repo_id,
            revision=cfg.model.revision,
            tokenizer_repo_id=cfg.model.tokenizer_repo_id,
        )
        handle = load_lm(model_cfg, detect(), token=token)
        return (
            TransformersBackend(
                dict.fromkeys(profiles, handle),
                revisions=dict.fromkeys(
                    profiles, f"{cfg.model.repo_id}@{cfg.model.revision or 'unresolved'}"
                ),
            ),
            cfg.model.tokenizer_revision,
        )

    raise typer.BadParameter(f"unknown backend '{backend_kind}'")
