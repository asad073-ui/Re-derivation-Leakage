"""Config loading, validation, and hashing.

Two layers, on purpose:

  OmegaConf   composes YAML fragments (env / models / agents / memory / writepolicy /
              condition) and applies CLI dotlist overrides last. It is good at merging
              and interpolation and bad at telling you a key is misspelled.

  Pydantic    validates the merged tree into a frozen model where **unknown keys are an
              error, not a warning**. A typo'd `writepolicy.mdoe` must fail at config
              parse time, not forty minutes into a GPU run when the write path turns
              out to have silently used its default.

`config_hash` closes invariant 6: two runs with the same hash must be byte-identical,
so the hash is taken over the *resolved, canonicalised* config, not the source YAML.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import DictConfig, ListConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .logging_utils import dumps_canonical
from .paths import configs_dir

__all__ = [
    "AbstentionConfig",
    "AgentConfig",
    "ConfigError",
    "DataConfig",
    "DelegationConfig",
    "EnvConfig",
    "EpisodeConfig",
    "MemoryConfig",
    "ModelConfig",
    "RDLConfig",
    "WritePolicyConfig",
    "compose",
    "config_hash",
    "load_config",
    "load_env",
    "validate",
]

# Groups are resolved as configs/<group>/<name>.yaml. Order matters: later groups may
# reference earlier ones, and CLI overrides are applied after all of them.
GROUPS = ("env", "models", "agents", "memory", "writepolicy")

SourceKindLiteral = Literal["ingest", "agent_answer", "summary", "tool"]


def _default_write_source_kinds() -> list[SourceKindLiteral]:
    """Only the assistant turn is persisted by default. See writer.py."""
    return ["agent_answer"]


class ConfigError(ValueError):
    """Raised for any malformed, missing, or unknown-key configuration."""


class _Base(BaseModel):
    """Frozen, strict. Every config model inherits this."""

    model_config = ConfigDict(frozen=True, extra="forbid", validate_assignment=True)


# ---------------------------------------------------------------------------- env --


class EnvConfig(_Base):
    name: str
    device: Literal["cpu", "cuda", "auto"] = "auto"
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    attn_implementation: Literal["auto", "eager", "sdpa", "flash_attention_2"] = "auto"
    batch_size: int = 1
    max_new_tokens: int = 128
    allow_fp16_training: bool = False
    hf_home: str | None = None
    index_backend: Literal["numpy", "faiss"] = "numpy"
    # Preconditions on the machine, checked by hardware.check_env_against_hardware before
    # anything is downloaded. `None` means "do not check". A rented instance that turns
    # out to be a 12 GB card must fail in the first four seconds, not in the results.
    min_vram_gb: float | None = None
    min_free_disk_gb: float | None = None
    # Provenance, not capability. `min_vram_gb: 20` also accepts a 4090, an A5000 or an
    # H100 — all of which would run the eval fine, and none of which are what a report
    # stamped `vast_rtx3090` claims. A profile that NAMES a card must verify it; the
    # generic `rtx3090` profile leaves both unset on purpose.
    expected_gpu_name_regex: str | None = None
    expected_compute_capability: tuple[int, int] | None = None
    # open-unlearning declares `python_requires >= 3.11`. The `rdl` core runs the CPU
    # gate on 3.10 (ADR-0002), so this is per-environment rather than global: a GPU box
    # that will install the submodule needs 3.11.
    min_python: str | None = None
    notes: str | None = None

    @field_validator("expected_compute_capability", mode="before")
    @classmethod
    def _cc_from_yaml(cls, v: object) -> object:
        """YAML gives a list; the profile compares against a tuple."""
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("batch_size")
    @classmethod
    def _batch_size_one_for_paper(cls, v: int) -> int:
        if v < 1:
            raise ValueError("batch_size must be >= 1")
        return v


# -------------------------------------------------------------------------- model --


class ModelConfig(_Base):
    name: str
    kind: Literal["hf", "stub"] = "hf"
    # HF repo id, or a registry alias resolved by rdl.models.registry.
    repo_id: str | None = None
    revision: str | None = None
    tokenizer_repo_id: str | None = None
    # When set, these WIN over the hardware profile. Unset (None) means "ask hardware.py".
    # This is the explicit-override flag required by REPO_SPEC 4.3.
    dtype_override: Literal["float32", "float16", "bfloat16"] | None = None
    attn_override: Literal["eager", "sdpa", "flash_attention_2"] | None = None
    trust_remote_code: bool = False
    chat_template_required: bool = True
    # StubLM only.
    stub_answers: dict[str, str] = Field(default_factory=dict)
    stub_knowledge_mask: list[str] = Field(default_factory=list)
    stub_abstention_text: str = "I don't know."
    stub_paraphrase_mode: bool = False

    @model_validator(mode="after")
    def _repo_required_for_hf(self) -> ModelConfig:
        if self.kind == "hf" and not self.repo_id:
            raise ValueError(f"model '{self.name}': kind=hf requires repo_id")
        return self


# -------------------------------------------------------------------------- agent --


class AbstentionConfig(_Base):
    # Three detectors, because "agent A abstains" is a load-bearing assumption and a
    # single heuristic is a reviewer target. `ensemble` runs all three and reports
    # pairwise agreement for the appendix.
    detector: Literal["lexical", "logprob", "self_report", "ensemble"] = "lexical"
    logprob_threshold: float = -1.5
    self_report_token: str = "<UNKNOWN>"
    phrase_list: str = "default"
    ensemble_rule: Literal["any", "majority", "all"] = "majority"


class DelegationConfig(_Base):
    # `always_delegate` is the control that decouples routing from A's degradation.
    # It is NOT optional — see docs/00_preregistration.md, confound gate.
    policy: Literal["abstention_triggered", "always_delegate", "never"] = "abstention_triggered"
    max_delegations: int = 1


class AgentConfig(_Base):
    agent_id: str
    role: Literal["primary", "secondary"] = "primary"
    model: str  # name of a configs/models/*.yaml
    abstention: AbstentionConfig = Field(default_factory=AbstentionConfig)
    delegation: DelegationConfig = Field(default_factory=DelegationConfig)
    system_prompt: str | None = None
    notes: str | None = None


# ------------------------------------------------------------------------- memory --


class MemoryConfig(_Base):
    name: str
    blocklist: Literal["none", "id", "semantic"] = "none"
    # SBU's mechanism: an ID blocklist plus deletion over the derivation closure.
    enforce_at_retrieval: bool = True
    derivation_closure: bool = True
    refcount_prune: bool = True
    retrieval_k: int = 5
    embedding_dim: int = 64
    # SemanticBlocklist only.
    semantic_threshold: float = 0.75
    semantic_use_nli: bool = False

    # ---- SBU fidelity knobs. See ADR-0014 and ADR-0015. -------------------------
    # `hard` is SBU as written: the deleted target AND its vector are removed. Under
    # `hard`, invariant 1 is vacuously true, which is exactly why it is not our
    # default — but a paper that claims to test SBU must be able to run SBU.
    # `tombstone` keeps the node indexed and makes the blocklist do the suppressing,
    # which is a strictly stronger adversarial setting.
    deletion_mode: Literal["tombstone", "hard"] = "tombstone"
    # `dependent_children` is SBU's definition: how many nodes depend on this one.
    # `supporting_parents` is ours (ADR-0004) and propagates deletion downward
    # correctly on the shapes we build. Both are implemented; the results table must
    # say which produced it.
    refcount_semantics: Literal["supporting_parents", "dependent_children"] = "supporting_parents"
    # Whether the forget-set content was ever ingested into the shared store. C0 sets
    # this false: a "sanity check on the bare checkpoint" that can retrieve the
    # ground-truth answers from memory is not measuring the checkpoint.
    ingest_forget_set: bool = True


# -------------------------------------------------------------------- write policy --


class WritePolicyConfig(_Base):
    name: str
    mode: Literal["disabled", "framework_default", "sanitized"] = "disabled"
    # framework_default writes the final assistant answer as a new node with parent_ids
    # set ONLY to nodes actually present in the retrieval context. See writer.py for the
    # citation of which real framework's default this mirrors.
    mirrors_framework: str | None = None
    write_source_kinds: list[SourceKindLiteral] = Field(default_factory=_default_write_source_kinds)
    write_on_abstention: bool = False
    sanitize_threshold: float = 0.75


# ------------------------------------------------------------------- episode/data --


class EpisodeConfig(_Base):
    max_turns: int = 5
    retrieval_k: int = 5
    greedy: bool = True
    max_new_tokens: int = 128
    # `openunlearning` reproduces upstream's TOFU user turn exactly (bare question
    # through the chat template). `qa_scaffold` is the old Question:/Answer: framing,
    # which is NOT comparable with the reproduction gate.
    prompt_style: Literal["openunlearning", "qa_scaffold"] = "openunlearning"
    # Show agent B what agent A answered. Off = ensemble (two independent draws);
    # on = compositional re-derivation. Only C3C sets it.
    pass_primary_answer: bool = False
    # Permute episode order per seed. With greedy decoding this is the ONLY thing that
    # makes a seed a replicate rather than a rerun; turning it off makes every
    # seed-level CI zero-width by construction. See eval/aggregate.py.
    permute_item_order_per_seed: bool = True


class DataConfig(_Base):
    dataset: Literal["tofu", "stub"] = "tofu"
    forget_split: str = "forget10"
    retain_split: str = "retain90"
    n_items: int | None = None
    fixture_path: str | None = None
    # The retain-question control arm (pre-registration §4.3, false-positive floor).
    # Sampled from `retain_split`; 0 disables the arm and makes the floor unreported.
    n_retain_items: int = 100
    # Cluster id for the paired bootstrap. TOFU is 200 synthetic authors x 20
    # questions, so questions are not independent; `author` resamples whole authors.
    cluster_by: Literal["item", "author"] = "author"


# ----------------------------------------------------------------------- top level --


Condition = Literal["C0", "C1", "C1W", "C2", "C3", "C3D", "C3C"]

# Which conditions are single-agent, and which require write-back on or off. Encoded
# once, here, so a mislabelled condition file fails at parse time rather than producing
# a plausible number under the wrong design.
_SINGLE_AGENT: frozenset[str] = frozenset({"C0", "C1W"})
_REQUIRE_WRITE_DISABLED: frozenset[str] = frozenset({"C0", "C1"})
_REQUIRE_WRITE_ENABLED: frozenset[str] = frozenset({"C1W", "C2", "C3", "C3D", "C3C"})


class RDLConfig(_Base):
    """The fully resolved, validated run configuration."""

    condition: Condition
    description: str = ""
    seed: int = 42
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4])

    env: EnvConfig
    memory: MemoryConfig
    writepolicy: WritePolicyConfig
    episode: EpisodeConfig = Field(default_factory=EpisodeConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    agent_a: AgentConfig
    agent_b: AgentConfig | None = None
    models: dict[str, ModelConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _cross_checks(self) -> RDLConfig:
        # Every agent's model must have been composed into `models`.
        for agent in (self.agent_a, self.agent_b):
            if agent is None:
                continue
            if agent.model not in self.models:
                raise ValueError(
                    f"agent '{agent.agent_id}' references model '{agent.model}', which was "
                    f"not composed. Available: {sorted(self.models)}. Add "
                    f"configs/models/{agent.model}.yaml to the condition's `models:` list."
                )
        if self.condition in _SINGLE_AGENT and self.agent_b is not None:
            raise ValueError(f"{self.condition} is single-agent: agent_b must be unset")
        if self.condition not in _SINGLE_AGENT and self.agent_b is None:
            raise ValueError(f"{self.condition} is a two-agent condition: agent_b is required")

        if self.condition in _REQUIRE_WRITE_DISABLED and self.writepolicy.mode != "disabled":
            raise ValueError(
                f"{self.condition} requires writepolicy.mode=disabled; "
                f"got '{self.writepolicy.mode}'"
            )
        if self.condition in _REQUIRE_WRITE_ENABLED and self.writepolicy.mode == "disabled":
            raise ValueError(
                f"{self.condition} measures the write path and requires write-back "
                "enabled (framework_default or sanitized); got 'disabled'"
            )

        # C3C is the compositional arm and is defined by the handoff. Without it, it is
        # C3D under a different name — two agents answering the same question in
        # isolation — and the pair would silently stop being a contrast.
        if self.condition == "C3C" and not self.episode.pass_primary_answer:
            raise ValueError(
                "C3C is the compositional re-derivation arm: episode.pass_primary_answer "
                "must be true, otherwise it is an ensemble control identical to C3D."
            )
        if self.condition != "C3C" and self.episode.pass_primary_answer:
            raise ValueError(
                f"{self.condition} must not pass A's answer to B — only C3C does. "
                "Enabling it here would make this arm compositional and destroy the "
                "C3D-vs-C3C contrast."
            )

        # C3 is the redundancy control: agent B is agent A's checkpoint, loaded twice.
        # That is the point of the arm, and it must be true rather than accidental.
        if (
            self.condition == "C3"
            and self.agent_b is not None
            and self.agent_a.model != self.agent_b.model
        ):
            raise ValueError(
                "C3 is the REDUNDANCY control: both agents must load the same "
                f"checkpoint, but agent_a uses '{self.agent_a.model}' and agent_b "
                f"uses '{self.agent_b.model}'. For two independently unlearned "
                "agents use C3D."
            )
        if (
            self.condition in ("C3D", "C3C")
            and self.agent_b is not None
            and self.agent_a.model == self.agent_b.model
        ):
            raise ValueError(
                f"{self.condition} requires agent B to be an INDEPENDENTLY unlearned "
                f"checkpoint, but both agents load '{self.agent_a.model}'. One "
                "checkpoint queried twice with greedy decoding returns the same "
                "answer twice; that is C3, the redundancy control, not a two-agent "
                "measurement."
            )
        return self

    def model_names(self) -> list[str]:
        return sorted(self.models)


# ------------------------------------------------------------------------ loading --


def _load_yaml(path: Path) -> DictConfig:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(cfg).__name__}")
    return cfg


def _group_path(group: str, name: str, root: Path | None) -> Path:
    return configs_dir(root) / group / f"{name}.yaml"


def load_env(name: str, root: Path | None = None) -> EnvConfig:
    """Load and validate one ``configs/env/<name>.yaml`` on its own.

    `run-repro` needs the environment (HF_HOME, the VRAM precondition) without composing
    a whole condition tree, and duplicating the group-path convention there is how the
    two drift apart.
    """
    raw = _load_yaml(_group_path("env", name, root))
    container = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(container, dict):
        raise ConfigError(f"configs/env/{name}.yaml: top level must be a mapping")
    try:
        return EnvConfig.model_validate(container)
    except Exception as exc:
        raise ConfigError(f"configs/env/{name}.yaml failed validation:\n{exc}") from exc


def compose(
    condition_path: str | Path,
    overrides: Sequence[str] | None = None,
    root: Path | None = None,
    env_override: str | None = None,
) -> DictConfig:
    """Merge the config tree named by a condition file, then apply dotlist overrides.

    The condition file names its fragments by group:

        env: vast_rtx3090
        memory: sbu_id_blocklist
        writepolicy: framework_default
        agent_a: A_unlearned
        agent_b: B_unlearned_same
        models: [tofu_llama32_1b_npo_forget10]

    Each is loaded from ``configs/<group>/<name>.yaml``. Anything else in the condition
    file (seed, episode, data, ...) is merged on top.

    `env_override` replaces the **whole env group**, which a dotlist override cannot do.
    By the time `--set env=rtx3090` is applied, `merged.env` is already a mapping, so the
    dotlist would either merge a bare string into a node or (worse) leave the previous
    profile's keys in place under a new `name`. The scientific conditions must be
    identical across hardware; only the execution environment moves — hence a group
    selector rather than a per-key edit.
    """
    condition_path = Path(condition_path)
    raw = _load_yaml(condition_path)

    merged = OmegaConf.create({})

    # --- env ------------------------------------------------------------------
    configured_env = raw.pop("env", None)
    env_name = env_override if env_override is not None else configured_env
    if env_name is None:
        raise ConfigError(f"{condition_path}: missing required key 'env'")
    merged.env = _load_yaml(_group_path("env", str(env_name), root))

    # --- memory / writepolicy -------------------------------------------------
    for group in ("memory", "writepolicy"):
        name = raw.pop(group, None)
        if name is None:
            raise ConfigError(f"{condition_path}: missing required key '{group}'")
        merged[group] = _load_yaml(_group_path(group, str(name), root))

    # --- agents ---------------------------------------------------------------
    for key in ("agent_a", "agent_b"):
        name = raw.pop(key, None)
        if name is None:
            continue
        merged[key] = _load_yaml(_group_path("agents", str(name), root))

    # --- models ---------------------------------------------------------------
    model_names = raw.pop("models", None)
    models: dict[str, Any] = {}
    if model_names is not None:
        if isinstance(model_names, (list, ListConfig)):
            names = [str(n) for n in model_names]
        else:
            names = [str(model_names)]
        for n in names:
            models[n] = _load_yaml(_group_path("models", n, root))
    else:
        # Convenience: pull in exactly the models the agents reference.
        for key in ("agent_a", "agent_b"):
            if key in merged and "model" in merged[key]:
                n = str(merged[key].model)
                if n not in models:
                    models[n] = _load_yaml(_group_path("models", n, root))
    merged.models = OmegaConf.create(models)

    # --- whatever is left in the condition file (seed, episode, data, ...) ----
    merged = cast(DictConfig, OmegaConf.merge(merged, raw))

    # --- CLI dotlist last -----------------------------------------------------
    if overrides:
        merged = cast(DictConfig, OmegaConf.merge(merged, OmegaConf.from_dotlist(list(overrides))))

    assert isinstance(merged, DictConfig)
    return merged


def validate(cfg: DictConfig) -> RDLConfig:
    """Resolve interpolations and validate into the frozen Pydantic tree."""
    container = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    if not isinstance(container, dict):
        raise ConfigError("resolved config is not a mapping")
    try:
        return RDLConfig.model_validate(container)
    except Exception as exc:  # pydantic ValidationError, re-raised with our type
        raise ConfigError(f"config validation failed:\n{exc}") from exc


def load_config(
    condition_path: str | Path,
    overrides: Sequence[str] | None = None,
    root: Path | None = None,
    env_override: str | None = None,
) -> RDLConfig:
    """compose + validate. This is the only entry point callers should use."""
    return validate(compose(condition_path, overrides, root, env_override=env_override))


def config_hash(cfg: RDLConfig | DictConfig | dict) -> str:
    """Stable sha256 over the canonicalised, resolved config.

    Two runs with the same hash must be byte-identical. The hash therefore covers the
    *resolved* tree — not the YAML text, which can differ in whitespace, key order, or
    which fragment file a value came from.
    """
    if isinstance(cfg, BaseModel):
        payload: Any = cfg.model_dump(mode="json")
    elif isinstance(cfg, (DictConfig, ListConfig)):
        payload = OmegaConf.to_container(cfg, resolve=True)
    else:
        payload = cfg
    return hashlib.sha256(dumps_canonical(payload).encode("utf-8")).hexdigest()
