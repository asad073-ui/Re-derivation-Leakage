"""Per-item store reset vs the cumulative shared store (ADR-0047).

`run_episodes` deliberately shares ONE store across every episode of a seed, and
`_episode_order` permutes so the seed enters the measurement through that sharing. The
consequence is that episode *i*'s write is part of episode *i+n*'s retrievable context —
the items are not exchangeable. The primary interval is nevertheless a bootstrap that
resamples items (clustered by author) as though they were.

Author clustering handles TOFU's 20-questions-per-author structure. It does not handle a
dependence induced by the run itself, whose grouping is arrival order, not authorship. So
the primary experiment resets the store per item, and the cumulative run becomes a
separate longitudinal experiment reported over seeds.

These tests run through `execute_condition` — the real path — because the property under
test is about the runner's bookkeeping, not the loop's.
"""

from __future__ import annotations

from pathlib import Path

from rdl.agents.abstention import LexicalDetector
from rdl.agents.llm_agent import LLMAgent
from rdl.cli.run_condition import execute_condition
from rdl.config import compose as compose_cfg
from rdl.config import load_config, validate
from rdl.hardware import detect
from rdl.models.stub import StubLM

CONDITIONS = Path(__file__).resolve().parents[2] / "configs" / "conditions"


def _cfg(scope: str):
    return validate(
        compose_cfg(
            CONDITIONS / "C1W.yaml",
            [
                f"episode.store_scope={scope}",
                "models.tofu_llama32_1b_npo_forget10.kind=stub",
                "models.tofu_llama32_1b_npo_forget10.repo_id=null",
            ],
            # The env group is replaced wholesale, never merged key-by-key: by the time a
            # dotlist lands, `merged.env` is already a mapping. See config.compose.
            env_override="local_cpu",
        )
    )


def _agents(qa_pairs, tofu_items):
    """One agent that answers the SECOND half and abstains on the first."""
    masked = [it.question for it in tofu_items[: len(tofu_items) // 2]]
    return [
        LLMAgent(
            "A",
            StubLM(qa_pairs, knowledge_mask=masked, model_id="stub_a"),
            detector=LexicalDetector(),
        )
    ]


def test_per_item_scope_gives_every_episode_the_same_starting_store(qa_pairs, tofu_items):
    """No episode may see another episode's write. Under `cumulative` the last episode's
    store holds every earlier write; under `per_item` it holds at most its own."""
    cfg = _cfg("per_item")
    arm = execute_condition(
        cfg, tofu_items, detect(), seed=0, agents_override=_agents(qa_pairs, tofu_items)
    )

    written = {
        item_id: [n for n in nodes if n.source_kind == "agent_answer"]
        for item_id, nodes in arm.snapshots.items()
    }
    assert all(len(nodes) <= 1 for nodes in written.values()), (
        "a per-item store can contain at most this episode's own write"
    )


def test_cumulative_scope_still_accumulates(qa_pairs, tofu_items):
    """The longitudinal experiment is kept, not deleted: recontamination spreading across
    episodes is a real question, it just is not the primary estimand."""
    cfg = _cfg("cumulative")
    arm = execute_condition(
        cfg, tofu_items, detect(), seed=0, agents_override=_agents(qa_pairs, tofu_items)
    )

    counts = [
        len([n for n in nodes if n.source_kind == "agent_answer"])
        for nodes in arm.snapshots.values()
    ]
    assert max(counts) > 1, "the shared store must still accumulate across episodes"


def test_per_item_is_the_default_for_every_condition():
    """Two arms that are differenced must share a scope, so the default cannot be
    per-condition."""
    for name in ("C1W", "B1W", "C3D", "C3C"):
        assert load_config(CONDITIONS / f"{name}.yaml").episode.store_scope == "per_item", name


def test_negative_targets_are_scored_on_the_same_episodes(qa_pairs, tofu_items):
    """The floor is computed from the treatment's own episodes against deranged targets,
    so it costs no extra generation and cannot drift from what was measured."""
    cfg = _cfg("per_item")
    arm = execute_condition(
        cfg, tofu_items, detect(), seed=0, agents_override=_agents(qa_pairs, tofu_items)
    )

    assert len(arm.containment_negative) == len(arm.containment)
    assert [r.item_id for r in arm.containment_negative] == [r.item_id for r in arm.containment]
    assert not any(r.hit("persistent_store_after_episode") for r in arm.containment_negative), (
        "nothing the system wrote should match another item's answer"
    )
