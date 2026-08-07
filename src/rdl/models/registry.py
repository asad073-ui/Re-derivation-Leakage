"""Known Hugging Face checkpoints.

Hard-coded and commented on purpose. A missing checkpoint must fail loudly at
config-parse time, not forty minutes into a run.

CONFIRMED-REAL ANCHORS, verified against the Hub on 2026-08-07:

    open-unlearning/tofu_Llama-3.2-1B-Instruct_full       the finetuned target model
    open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90   the retain oracle
    open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10

**Naming trap, and it cost a whole planning cycle.** The unlearned checkpoints do NOT
follow the `tofu_<model>_<METHOD>_<split>` pattern of the finetuned/retain ones. They
are published as

    unlearn_tofu_<model>_<forget_split>_<METHOD>_lr<LR>_beta<B>_alpha<A>_epoch<E>

with one repo per hyperparameter setting. `open-unlearning/tofu_Llama-3.2-1B-Instruct_
NPO_forget10` — the id this file used to carry — has never existed. The one that
matches docs/repro.md is the lr1e-05 / beta0.1 / alpha1 / epoch10 variant, because
that is the setup the repro table was generated under (see the hyperparameter box at
the top of `third_party/open-unlearning/docs/repro.md`).

There is **no published NPO checkpoint for forget01 or forget05** on this architecture,
which is why the `B_unlearned_disjoint` arm was withdrawn — see ADR-0016.
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
    # Exact Hub commit. `main` is a moving target: the same repo commit re-run a month
    # later can pull different weights, and nothing in the report would say so. Kept in
    # lockstep with configs/models/*.yaml — `tests/unit/test_checkpoint_ids.py` fails if
    # the two disagree. See ADR-0030.
    revision: str | None = None


_ENTRIES: tuple[ModelEntry, ...] = (
    # --- confirmed -----------------------------------------------------------------
    ModelEntry(
        alias="tofu_llama32_1b_full",
        repo_id="open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
        status="confirmed",
        revision="88e31200b97e4c0c04ae0d2f0b591f427046d192",
        note="TOFU finetuned target. Published targets: model_utility 0.60, "
        "forget_truth_ratio 0.48. Run the sanity eval against THIS first — it "
        "isolates 'is my install correct' from 'does the unlearned ckpt exist'.",
    ),
    ModelEntry(
        alias="tofu_llama32_1b_retain90",
        repo_id="open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
        status="confirmed",
        revision="7114300c0049527a71833f5683965c358ad9dcbf",
        note="Retain oracle for forget10. Published: model_utility 0.59, "
        "forget_truth_ratio 0.63. Its eval log is also the retain_logs_path "
        "required to compute forget_quality.",
    ),
    ModelEntry(
        alias="tofu_llama32_1b_npo_forget10",
        repo_id=(
            "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO"
            "_lr1e-05_beta0.1_alpha1_epoch10"
        ),
        status="confirmed",
        revision="94ed64eb73bc1872d52064833aaef364f4895c9c",
        note="THE Day 1-2 target. Published: model_utility 0.46, "
        "forget_truth_ratio 0.70, forget_quality 0.02 (report, do not gate). "
        "lr1e-05/beta0.1/alpha1/epoch10 is the setting docs/repro.md was generated "
        "under; the other NPO repos are different hyperparameters and do NOT match "
        "the published row.",
    ),
    # --- a SECOND, independently trained unlearning of the SAME forget set ----------
    ModelEntry(
        alias="tofu_llama32_1b_npo_forget10_indep",
        repo_id=(
            "open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO"
            "_lr2e-05_beta0.5_alpha1_epoch10"
        ),
        status="confirmed",
        revision="eabf32c4883a5647c784c60c998b4b96cd48b798",
        note="Agent B for the C3D arm: forget10 removed by a SEPARATE NPO run "
        "(lr2e-05, beta0.5). Same forget set, different optimisation trajectory, so "
        "its residual knowledge is not A's residual knowledge by construction. This "
        "is what makes C3D a two-agent measurement rather than one checkpoint queried "
        "twice. NOT comparable to the published repro row — different hyperparameters.",
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
