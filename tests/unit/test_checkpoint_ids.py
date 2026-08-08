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


# =====================================================================================
# Revision pinning
# =====================================================================================

SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_every_hf_model_config_pins_an_exact_commit():
    """`main` is a moving target.

    Without a revision, re-running this same repo commit a month later can pull different
    weights, two runs with the same `config_hash` would not be the same experiment, and
    nothing in the report would say so. `make-report` blocks on an unpinned checkpoint;
    this is the check that runs on a laptop.
    """
    for name, cfg in _model_configs():
        if cfg.get("kind", "hf") != "hf":
            continue
        rev = cfg.get("revision")
        assert rev, f"{name} has no `revision:` pin"
        assert SHA_RE.match(str(rev)), f"{name}: {rev!r} is not a 40-char commit sha"


def test_registry_and_config_revisions_agree():
    """Two sources of truth that disagree are worse than one that is wrong."""
    by_repo = {e.repo_id: e for e in KNOWN_MODELS.values()}
    for name, cfg in _model_configs():
        repo = cfg.get("repo_id")
        entry = by_repo.get(repo)
        if entry is None or entry.revision is None:
            continue
        assert cfg.get("revision") == entry.revision, (
            f"{name} pins {cfg.get('revision')} but the registry pins {entry.revision} for {repo}"
        )


def test_the_two_agents_are_pinned_to_different_commits():
    """Same forget set, different optimisation run — down to the artefact."""
    a = KNOWN_MODELS["tofu_llama32_1b_npo_forget10"]
    b = KNOWN_MODELS["tofu_llama32_1b_npo_forget10_indep"]
    assert a.revision and b.revision and a.revision != b.revision


# =====================================================================================
# Sampling the retain control set
# =====================================================================================


def test_spread_sample_covers_the_whole_split():
    """TOFU splits are contiguous author blocks, so the head of retain90 is five
    novelists. A false-positive floor measured on five authors says nothing about the
    other 175."""
    from rdl.eval.tofu_data import QUESTIONS_PER_AUTHOR, TofuItem, spread_sample

    full = [
        TofuItem(f"retain90-{i:04d}", f"q{i}", f"a{i}", "retain90", index=i) for i in range(3600)
    ]

    head = full[:100]
    spread = spread_sample(full, 100)

    assert len(spread) == 100
    assert len({it.author_id for it in head}) == 100 // QUESTIONS_PER_AUTHOR
    assert len({it.author_id for it in spread}) == 100, "one author per sampled item"
    assert spread == sorted(spread, key=lambda it: it.index), "order preserved"


def test_spread_sample_is_deterministic_and_degrades_gracefully():
    from rdl.eval.tofu_data import TofuItem, spread_sample

    items = [TofuItem(f"i{i}", "q", "a", "retain90", index=i) for i in range(10)]
    assert spread_sample(items, 4) == spread_sample(items, 4)
    assert spread_sample(items, 100) == items, "asking for more than exists returns all"
    assert spread_sample(items, 0) == items
