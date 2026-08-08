"""Runtime provenance: what software actually produced a number.

Split out of `cli/run_repro.py` when `run-condition` needed the same facts (ADR-0054).
Days 1-2 recorded the transformers version, the resolved dtype, the resolved attention
implementation and the tokenizer's chat-template hash; the condition grid — longer, more
expensive and harder to repeat — recorded the git SHA and the hardware profile and
nothing about the software in between.

That matters because the grid's whole output is *differences between arms*. An SDPA arm
minus an FA2 arm is not a delta; neither is an arm run against a chat template that moved
between the two. `make-report` blocks such pairings, which it can only do if both reports
carry the facts.

Everything here is best-effort and never raises: provenance must not fail a finished run.
"""

from __future__ import annotations

__all__ = ["pkg_version", "tokenizer_provenance"]

# Upstream's TOFU model config points `tokenizer_args` at this repo with NO revision, so
# the chat template that renders every prompt is read from a moving branch. Recorded
# rather than pinned-and-enforced: pinning it would be a fork of upstream's config.
UPSTREAM_TOKENIZER_REPO = "meta-llama/Llama-3.2-1B-Instruct"


def pkg_version(name: str) -> str | None:
    """Installed version of `name`, or None. Never raises — provenance, not a gate."""
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def tokenizer_provenance(repo: str = UPSTREAM_TOKENIZER_REPO) -> dict:
    """Which tokenizer, at which commit, with which chat template.

    Both controls passing means the template has not moved yet. It does not mean it
    cannot, which is exactly why the hash is recorded next to every number it shaped.
    """
    info: dict = {"repo": repo, "revision": None, "chat_template_sha256": None}
    try:
        import hashlib
        import os

        from huggingface_hub import HfApi
        from transformers import AutoTokenizer

        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        info["revision"] = HfApi(token=token).model_info(repo).sha
        template = getattr(AutoTokenizer.from_pretrained(repo, token=token), "chat_template", None)
        if template:
            info["chat_template_sha256"] = hashlib.sha256(template.encode("utf-8")).hexdigest()
    except Exception as exc:  # provenance is best-effort; never fail a finished eval
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info
