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
    notes: str | None = None

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


class DataConfig(_Base):
    dataset: Literal["tofu", "stub"] = "tofu"
    forget_split: str = "forget10"
    retain_split: str = "retain90"
    n_items: int | None = None
    fixture_path: str | None = None


# ----------------------------------------------------------------------- top level --


class RDLConfig(_Base):
    """The fully resolved, validated run configuration."""

    condition: Literal["C0", "C1", "C2", "C3"]
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
        # C0 is a single-agent sanity condition.
        if self.condition == "C0" and self.agent_b is not None:
            raise ValueError("C0 is single-agent: agent_b must be unset")
        if self.condition in ("C1", "C2", "C3") and self.agent_b is None:
            raise ValueError(f"{self.condition} is a two-agent condition: agent_b is required")
        # C1 is the leakage floor: SBU's claimed behaviour, no write-back.
        if self.condition == "C1" and self.writepolicy.mode != "disabled":
            raise ValueError(
                "C1 is the leakage floor and requires writepolicy.mode=disabled; "
                f"got '{self.writepolicy.mode}'"
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


def compose(
    condition_path: str | Path,
    overrides: Sequence[str] | None = None,
    root: Path | None = None,
) -> DictConfig:
    """Merge the config tree named by a condition file, then apply dotlist overrides.

    The condition file names its fragments by group:

        env: colab_t4
        memory: sbu_id_blocklist
        writepolicy: framework_default
        agent_a: A_unlearned
        agent_b: B_unlearned_same
        models: [tofu_llama32_1b_npo_forget10]

    Each is loaded from ``configs/<group>/<name>.yaml``. Anything else in the condition
    file (seed, episode, data, ...) is merged on top.
    """
    condition_path = Path(condition_path)
    raw = _load_yaml(condition_path)

    merged = OmegaConf.create({})

    # --- env ------------------------------------------------------------------
    env_name = raw.pop("env", None)
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
) -> RDLConfig:
    """compose + validate. This is the only entry point callers should use."""
    return validate(compose(condition_path, overrides, root))


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
