"""Guards on the experimental design itself.

Each test here corresponds to a way the grid produced a publishable-looking number that
did not mean what the report said it meant. They are contract tests rather than unit
tests because the property under test is "this configuration measures what it claims",
which lives across the config layer, the data layer, and the orchestrator.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rdl.agents.abstention import LexicalDetector
from rdl.agents.delegation import build_delegation_policy
from rdl.agents.llm_agent import LLMAgent
from rdl.agents.writer import build_write_policy
from rdl.cli.run_condition import seed_store
from rdl.config import ConfigError, compose, load_config, validate
from rdl.eval.tofu_data import load_items
from rdl.memory.blocklist import IDBlocklist
from rdl.models.stub import StubLM
from rdl.orchestrator.loop import EpisodePolicies, run_episodes

CONDITIONS = Path(__file__).resolve().parents[2] / "configs" / "conditions"


# =====================================================================================
# C0 was contaminated with the answers it was supposed to have forgotten
# =====================================================================================


def test_c0_store_contains_no_ground_truth_answers(tofu_items):
    """C0 is the bare-checkpoint sanity arm and its store must be EMPTY.

    `seed_store` used to ingest every question+answer pair and then return early —
    without deleting anything — whenever `memory.blocklist == none`, which is exactly
    C0's config. C0 could therefore retrieve the forgotten answers straight out of
    memory and report near-ceiling recovery. The number would have been read as
    "the checkpoint remembers", and it was "the fixture was in the store".
    """
    cfg = load_config(CONDITIONS / "C0.yaml")
    store, blocklist, blocked = seed_store(tofu_items, cfg)

    assert len(store) == 0, "C0 must start from an empty store"
    assert blocked == []
    for it in tofu_items:
        hits = store.retrieve(it.question, k=5, blocklist=blocklist)
        assert hits.nodes == [], f"C0 retrieved something for {it.item_id}"


def test_defended_conditions_still_ingest_then_delete(tofu_items):
    """The fix must not have turned the defended arms into empty stores too."""
    cfg = load_config(CONDITIONS / "C3.yaml")
    store, blocklist, blocked = seed_store(tofu_items, cfg)

    assert len(blocked) == len(tofu_items), "the forget set was ingested"
    assert isinstance(blocklist, IDBlocklist)
    assert all(store.get(nid).deleted for nid in blocked), "and then deleted"
    for it in tofu_items:
        got = store.retrieve(it.question, k=5, blocklist=blocklist)
        assert got.nodes == [], "the blocklist suppresses every ingested node"


def test_semantic_blocklist_config_is_actually_honoured(tofu_items):
    """`seed_store` hard-coded `build_blocklist("id")`, so a config asking for the
    Phase-2 semantic defence silently ran the id blocklist and the intervention could
    never be evaluated."""
    cfg = validate(
        compose(CONDITIONS / "C3.yaml", ["memory.blocklist=semantic", "memory.name=semantic"])
    )
    _, blocklist, _ = seed_store(tofu_items, cfg)
    assert blocklist.kind == "semantic"
    # A content-level blocklist can decide on raw text; an id one structurally cannot.
    decision = blocklist.is_blocked_text(f"{tofu_items[0].question} {tofu_items[0].answer}")
    assert decision.blocked


# =====================================================================================
# C3 was one checkpoint queried twice
# =====================================================================================


def test_c3_requires_both_agents_to_share_a_checkpoint():
    """C3 is the redundancy control BY DEFINITION. If it quietly acquired two different
    checkpoints it would stop being the control the other arms are read against."""
    # C2 composes two different checkpoints; relabelling it C3 is exactly the mistake.
    with pytest.raises(ConfigError, match="REDUNDANCY control"):
        load_config(CONDITIONS / "C2.yaml", ["condition=C3"])


def test_c3d_refuses_two_copies_of_one_checkpoint():
    """The original C3 put one checkpoint in both slots and was labelled the treatment.
    C3D cannot be configured that way."""
    with pytest.raises(ConfigError, match="INDEPENDENTLY unlearned"):
        load_config(CONDITIONS / "C3D.yaml", ["agent_b.model=tofu_llama32_1b_npo_forget10"])


def test_c3c_is_the_only_arm_that_hands_a_to_b():
    """C3C - C3D is the single-variable contrast that isolates composition from
    ensembling. If any other arm could enable the handoff, the contrast would not be
    single-variable."""
    with pytest.raises(ConfigError, match="must not pass A's answer"):
        load_config(CONDITIONS / "C3D.yaml", ["episode.pass_primary_answer=true"])
    with pytest.raises(ConfigError, match="pass_primary_answer"):
        load_config(CONDITIONS / "C3C.yaml", ["episode.pass_primary_answer=false"])


def test_c1w_is_single_agent_with_writeback():
    """The baseline the estimand is measured against: one agent, writing back."""
    cfg = load_config(CONDITIONS / "C1W.yaml")
    assert cfg.agent_b is None
    assert cfg.writepolicy.mode == "framework_default"


def test_writeback_arms_cannot_be_silently_disabled():
    with pytest.raises(ConfigError, match="requires write-back enabled"):
        load_config(CONDITIONS / "C1W.yaml", ["writepolicy.mode=disabled"])


def test_identical_checkpoints_give_identical_answers(qa_pairs, tofu_items, seeded_store):
    """Why C3 cannot be the treatment, demonstrated rather than asserted.

    Two agents built from ONE checkpoint, greedy, same question, same context, produce
    the same string. The stub reproduces the production path here — the previous C3
    test gave B a *different* knowledge mask from A, which manufactured heterogeneity
    that production never had, and that is why the design flaw passed CI.
    """
    masked = [it.question for it in tofu_items[: len(tofu_items) // 2]]
    shared = dict(qa_pairs)

    a = LLMAgent(
        "A", StubLM(shared, knowledge_mask=masked, model_id="ckpt"), detector=LexicalDetector()
    )
    b = LLMAgent(
        "B", StubLM(shared, knowledge_mask=masked, model_id="ckpt"), detector=LexicalDetector()
    )

    store, blocklist, _ = seeded_store
    transcripts = run_episodes(
        [(it.question, it.item_id) for it in tofu_items],
        [a, b],
        store,
        EpisodePolicies(
            delegation=build_delegation_policy("always_delegate", 1),
            write=build_write_policy("framework_default"),
            blocklist=blocklist,
        ),
        condition="C3",
    )

    for tr in transcripts:
        answers = tr.agent_answers()
        assert len(answers) == 2, "both agents answered"
        assert answers[0].text == answers[1].text, (
            "one checkpoint queried twice returns the same answer twice — C3 is a "
            "redundancy control, not a two-agent measurement"
        )


# =====================================================================================
# The production run used eight hand-written items
# =====================================================================================


def test_load_items_refuses_the_fixture_by_default():
    """`run_condition` called `load_fixture()` unconditionally and ignored
    `data.dataset`, `data.forget_split` and `data.n_items`, so a run configured for 400
    TOFU questions evaluated eight invented ones and said 400 nowhere."""
    with pytest.raises(ValueError, match="refusing to run on the development fixture"):
        load_items(dataset="stub")


def test_fixture_use_is_labelled_unreportable():
    items, prov = load_items(dataset="stub", allow_fixture=True)
    assert items
    assert prov["is_real_data"] is False
    assert "not reportable" in prov["warning"].lower()


def test_real_data_provenance_names_the_split():
    """No network here — just that the contract carries the fields the gate reads."""
    from rdl.eval.tofu_data import SPLIT_SIZES

    assert SPLIT_SIZES["forget10"] == 400, "the pre-registration commits to 400 items"
