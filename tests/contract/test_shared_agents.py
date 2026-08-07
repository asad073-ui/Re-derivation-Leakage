"""Models are loaded once per condition, not once per arm per seed.

`execute_condition` is called four times per seed on a two-agent condition — treatment,
always_delegate control, retain arm, agent-A-alone — and it used to build and tear down
the models on every one of them. Across the full seven-condition, five-seed grid that is
~195 model initialisations of a 1B checkpoint: most of the wall clock on a rented GPU,
and a repeated allocate/free cycle that fragments CUDA memory over a long run.

What must NOT change when the weights are shared: every arm still gets a fresh memory
store, a fresh blocklist, a fresh transcript collection, and its own item ordering. Those
are the things a seed is a replicate of. This file pins both halves.
"""

from __future__ import annotations

from pathlib import Path

from rdl.cli.run_condition import build_shared_agents, execute_condition
from rdl.config import compose, validate
from rdl.hardware import detect

CONDITIONS = Path(__file__).resolve().parents[2] / "configs" / "conditions"


def _stub_cfg(condition: str, *, two_agents: bool):
    """A shipped condition with its HF checkpoints swapped for stubs."""
    overrides = [
        "models.stub_a.name=stub_a",
        "models.stub_a.kind=stub",
        "agent_a.model=stub_a",
    ]
    if two_agents:
        overrides += [
            "models.stub_b.name=stub_b",
            "models.stub_b.kind=stub",
            "agent_b.model=stub_b",
        ]
    raw = compose(CONDITIONS / f"{condition}.yaml", overrides, env_override="local_cpu")
    # Drop the real checkpoints; the stub entries added above are what the agents use.
    for key in list(raw.models.keys()):
        if str(key).startswith("tofu_"):
            del raw.models[key]
    return validate(raw)


def test_one_agent_object_per_slot():
    cfg = _stub_cfg("C3D", two_agents=True)
    agents = build_shared_agents(cfg, detect())
    assert [a.agent_id for a in agents] == ["A", "B"]
    assert agents[0].lm is not agents[1].lm, "C3D's agents are DIFFERENT checkpoints"
    for a in agents:
        a.close()


def test_single_agent_condition_builds_one_agent():
    cfg = _stub_cfg("C1W", two_agents=False)
    agents = build_shared_agents(cfg, detect())
    assert len(agents) == 1
    agents[0].close()


def test_arms_share_weights_but_not_stores(tofu_items):
    """The whole point: reuse the model, rebuild everything that carries state."""
    cfg = _stub_cfg("C3D", two_agents=True)
    agents = build_shared_agents(cfg, detect())
    try:
        treatment = execute_condition(cfg, tofu_items, detect(), 0, agents_override=agents)
        control = execute_condition(
            cfg,
            tofu_items,
            detect(),
            0,
            delegation_override="always_delegate",
            agents_override=agents,
        )
        assert treatment.store is not control.store, "each arm gets its own memory store"
        assert treatment.transcripts is not control.transcripts
        assert len(treatment.containment) == len(tofu_items)
        assert len(control.containment) == len(tofu_items)
    finally:
        for a in agents:
            a.close()


def test_reused_agents_are_not_closed_by_an_arm(tofu_items):
    """`execute_condition` must leave a caller-supplied handle alive: the next seed and
    the next control arm are about to use it."""
    cfg = _stub_cfg("C1W", two_agents=False)
    agents = build_shared_agents(cfg, detect())
    try:
        for seed in range(3):
            arm = execute_condition(cfg, tofu_items, detect(), seed, agents_override=agents)
            assert arm.transcripts, f"seed {seed} produced no transcripts"
    finally:
        agents[0].close()


def test_seeds_still_permute_the_episode_order(tofu_items):
    """With shared weights and greedy decoding, item order is the only channel through
    which a seed can vary at all. If reuse had frozen it, every seed-level CI would be
    zero-width by construction."""
    cfg = _stub_cfg("C1W", two_agents=False)
    agents = build_shared_agents(cfg, detect())
    try:
        orders = []
        for seed in (0, 1):
            arm = execute_condition(cfg, tofu_items, detect(), seed, agents_override=agents)
            orders.append([t.item_id for t in arm.transcripts])
    finally:
        agents[0].close()
    assert sorted(orders[0]) == sorted(orders[1]), "same items"
    assert orders[0] != orders[1], "different order per seed"
