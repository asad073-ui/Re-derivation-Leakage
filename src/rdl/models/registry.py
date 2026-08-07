"""Known Hugging Face checkpoints.

Hard-coded and commented on purpose. A missing checkpoint must fail loudly at
config-parse time, not forty minutes into a run.

CONFIRMED-REAL ANCHORS (these are the two the Phase-0 plan can rely on):

    open-unlearning/tofu_Llama-3.2-1B-Instruct_full       the finetuned target model
    open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90   the retain oracle

EVERYTHING ELSE IS A CANDIDATE, NOT AN ANCHOR. In particular the *unlearned* NPO
forget10 checkpoint is what the zero-training cost model depends on, and its existence
must be confirmed with `rdl discover-checkpoints` on day 1 before any planning around
it. If it is absent, REPO_SPEC 7.4 fallback 1 applies: gate on `full` only and validate
the metric code against the published eval logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "KNOWN_MODELS",
    "ModelEntry",
    "UnknownModelError",
    "check_all",
    "resolve",
    "verify_exists",
]

Status = Literal["confirmed", "candidate", "gated"]


class UnknownModelError(KeyError):
    """Raised when an alias is not in the registry and is not an explicit repo id."""


@dataclass(frozen=True)
class ModelEntry:
    alias: str
    repo_id: str
    status: Status
    note: str = ""


_ENTRIES: tuple[ModelEntry, ...] = (
    # --- confirmed -----------------------------------------------------------------
    ModelEntry(
        alias="tofu_llama32_1b_full",
        repo_id="open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        status="confirmed",
        note="TOFU finetuned target. Published targets: model_utility 0.60, "
        "forget_truth_ratio 0.48. Run the sanity eval against THIS first — it "
        "isolates 'is my install correct' from 'does the unlearned ckpt exist'.",
    ),
    ModelEntry(
        alias="tofu_llama32_1b_retain90",
        repo_id="open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
        status="confirmed",
        note="Retain oracle for forget10. Published: model_utility 0.59, "
        "forget_truth_ratio 0.63. Its eval log is also the retain_logs_path "
        "required to compute forget_quality.",
    ),
    # --- candidate: MUST be verified on the Hub before being planned around ---------
    ModelEntry(
        alias="tofu_llama32_1b_npo_forget10",
        repo_id="open-unlearning/tofu_Llama-3.2-1B-Instruct_NPO_forget10",
        status="candidate",
        note="THE Day 1-2 target. Published: model_utility 0.46, "
        "forget_truth_ratio 0.70, forget_quality 0.02 (report, do not gate). "
        "Existence UNVERIFIED — run `rdl discover-checkpoints` before relying on it.",
    ),
    # --- gated base ----------------------------------------------------------------
    ModelEntry(
        alias="llama32_1b_instruct",
        repo_id="meta-llama/Llama-3.2-1B-Instruct",
        status="gated",
        note="GATED. Accept the Llama 3.2 community licence with the SAME account as "
        "your HF token, on day 1. Approval is usually instant but is a hard "
        "blocker if you hit it at hour six. Needed for the tokenizer/chat template.",
    ),
    # --- tiny CPU test model -------------------------------------------------------
    ModelEntry(
        alias="tiny_llama_test",
        repo_id="hf-internal-testing/tiny-random-LlamaForCausalLM",
        status="confirmed",
        note="~1 MB. Used by tests/integration/test_tiny_model_cpu.py to exercise the "
        "real loader path (dtype, padding side, chat template) on CPU before Colab.",
    ),
)

KNOWN_MODELS: dict[str, ModelEntry] = {e.alias: e for e in _ENTRIES}


def resolve(alias: str) -> str:
    """Alias -> HF repo id.

    An input that already looks like a repo id (``org/name``) passes through, so a
    config can name an unlisted checkpoint without editing this file. Anything else
    raises rather than being silently treated as a repo id.
    """
    if alias in KNOWN_MODELS:
        return KNOWN_MODELS[alias].repo_id
    if "/" in alias and not alias.startswith("/"):
        return alias
    raise UnknownModelError(
        f"unknown model alias '{alias}'. Known: {sorted(KNOWN_MODELS)}. "
        "Pass a full 'org/name' repo id to bypass the registry."
    )


def entry_for(alias_or_repo: str) -> ModelEntry | None:
    if alias_or_repo in KNOWN_MODELS:
        return KNOWN_MODELS[alias_or_repo]
    for e in _ENTRIES:
        if e.repo_id == alias_or_repo:
            return e
    return None


def verify_exists(repo_id: str, token: str | None = None) -> bool:
    """True iff `huggingface_hub.model_info` resolves. NEEDS NETWORK.

    Returns False rather than raising for 401/403/404 so a caller can report every
    missing checkpoint in one pass instead of dying on the first.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:  # pragma: no cover
        raise ImportError("verify_exists requires huggingface_hub") from None

    try:
        HfApi(token=token).model_info(repo_id)
        return True
    except Exception:
        return False


def check_all(token: str | None = None, include_candidates: bool = True) -> dict[str, bool]:
    """Verify every registry entry. Used by `rdl env-check`. NEEDS NETWORK."""
    out: dict[str, bool] = {}
    for e in _ENTRIES:
        if not include_candidates and e.status == "candidate":
            continue
        out[e.repo_id] = verify_exists(e.repo_id, token)
    return out
