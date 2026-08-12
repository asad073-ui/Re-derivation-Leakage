"""Shared wiring for the ``graph-*`` commands: config, cohort, backend.

Kept out of the individual commands so that ``plan``, ``run``, ``score`` and ``report``
resolve the identical configuration and cohort. A planning command that resolved things
slightly differently from the runner would predict a cost for an experiment nobody ran.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import typer

from ..eval.tofu_data import TofuItem, load_items
from ..graph.config import ResolvedGraphConfig, assert_model_provenance, load_graph_config
from ..graph.validation import GraphConfigError
from ..models.stub import StubLM
from ..paths import repo_root
from ..runtime.backend import GenerationBackend
from ..runtime.stub_backend import StubBackend
from ..studies.graph_leak.cohort import Cohort, CohortError, load_cohort, resolve_cohort

__all__ = [
    "DEFAULT_LAUNCH",
    "apply_sample_budget",
    "build_backend",
    "load_config_or_fail",
    "resolve_cohort_items",
]

DEFAULT_LAUNCH = "configs/graph/launch.yaml"

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
    token: str | None = None,
    limit: int | None = None,
) -> tuple[Cohort, list[TofuItem]]:
    """Load the phase's frozen cohort and attach real question/answer text.

    The validation split additionally refuses to load while its concepts intersect the
    exclusion list — which, on the current checkpoint, they always do. That refusal is
    the point: 'the next 50 questions' are new questions, not new forgotten concepts.
    """
    base = (root or repo_root()) / cfg.study.data.cohort_dir
    manifest_key = _MANIFEST_FOR_PHASE[cfg.phase]
    path = cohort_path or base / getattr(cfg.study.data, manifest_key)
    exclusions = base / cfg.study.data.exclusion_manifest

    try:
        cohort = load_cohort(
            path,
            exclusions_path=exclusions if exclusions.exists() else None,
            require_frozen=cfg.study.data.require_frozen_hashes,
            forbid_excluded_items=True,
            forbid_excluded_concepts=cfg.phase == "validation",
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
