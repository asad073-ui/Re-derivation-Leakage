"""Checkpoint ids, pinned offline.

`rdl discover-checkpoints` needs the network and is therefore not the thing that stops
a wrong id reaching a GPU session. These assertions encode what was verified on the Hub
on 2026-08-07 and fail on the laptop if anyone reintroduces the old naming guess.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from rdl.models.registry import KNOWN_MODELS, resolve

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "models"

# The published unlearned checkpoints all match this shape.
UNLEARN_RE = re.compile(
    r"^open-unlearning/unlearn_tofu_Llama-3\.2-1B-Instruct_forget\d+_[A-Za-z]+"
    r"_lr[\d.e-]+_beta[\d.]+_alpha\d+_epoch\d+$"
)

# The id the repo used to carry. It has never existed on the Hub: a 404 forty minutes
# into a Colab session, after the environment bootstrap and the data download.
NEVER_EXISTED = "open-unlearning/tofu_Llama-3.2-1B-Instruct_NPO_forget10"


def _model_configs() -> list[tuple[str, dict]]:
    out = []
    for p in sorted(CONFIGS.glob("*.yaml")):
        with p.open(encoding="utf-8") as fh:
            out.append((p.name, yaml.safe_load(fh)))
    return out


def test_the_phantom_npo_id_is_gone_everywhere():
    assert resolve("tofu_llama32_1b_npo_forget10") != NEVER_EXISTED
    for name, cfg in _model_configs():
        assert cfg.get("repo_id") != NEVER_EXISTED, f"{name} still points at a 404"


def test_npo_forget10_matches_the_published_repro_hyperparameters():
    """docs/repro.md was generated at lr=1e-5, beta=0.1, alpha=1, 10 epochs.

    Any other NPO repo is a different run whose metrics are not the published row, so
    gating against 0.46 / 0.70 with it would fail for a reason that is not our install.
    """
    repo = resolve("tofu_llama32_1b_npo_forget10")
    assert UNLEARN_RE.match(repo), repo
    assert "_lr1e-05_beta0.1_alpha1_epoch10" in repo


def test_agent_b_checkpoint_is_a_different_run_from_agent_a():
    """C3D's whole claim rests on B not being A."""
    a = resolve("tofu_llama32_1b_npo_forget10")
    b = resolve("tofu_llama32_1b_npo_forget10_indep")
    assert a != b
    assert UNLEARN_RE.match(b), b
    assert "forget10" in b, "B must remove the SAME forget set, by a different run"


@pytest.mark.parametrize("alias", sorted(KNOWN_MODELS))
def test_every_registry_entry_is_a_well_formed_repo_id(alias: str):
    repo = KNOWN_MODELS[alias].repo_id
    assert repo.count("/") == 1 and not repo.startswith("/"), repo


def test_no_config_references_a_withdrawn_split():
    """There is no published NPO checkpoint for forget01/forget05 on this architecture,
    so the `B_unlearned_disjoint` arm was withdrawn (ADR-0016) rather than left pointing
    at an id that does not resolve."""
    for name, cfg in _model_configs():
        repo = str(cfg.get("repo_id") or "")
        if repo.startswith("open-unlearning/unlearn_tofu"):
            assert "forget10" in repo, f"{name}: only forget10 unlearned ckpts are published"
