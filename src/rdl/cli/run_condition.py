"""`rdl run-condition` — Days 3-5. C0 / C1 / C2 / C3.

The four conditions, all on TOFU forget10, all sharing one `MemoryStore` and the
`sbu_id_blocklist` memory config:

         Agent A        Agent B                    Write-back         Purpose
    C0   NPO forget10   -                          -                  sanity
    C1   NPO forget10   NPO forget10               disabled           leakage floor
    C2   NPO forget10   full (NOT unlearned)       framework_default  mechanism demo
    C3   NPO forget10   NPO forget10 (same set)    framework_default  THE condition

**C2 is not the test.** With B un-unlearned, containment approaches ceiling and any
"> 15 point" bar is met trivially. C2 demonstrates the write-path mechanism; it does
not establish a finding.

**The gate is C3 - C1.** See docs/00_preregistration.md.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import typer

from ..agents.abstention import build_detector
from ..agents.delegation import build_delegation_policy
from ..agents.llm_agent import LLMAgent
from ..agents.writer import build_write_policy
from ..config import AgentConfig, RDLConfig, config_hash, load_config
from ..eval.aggregate import summarise_seeds
from ..eval.containment import SURFACES, containment, recall_table
from ..eval.controls import delegation_rate
from ..eval.laundering import laundered_items
from ..eval.tofu_data import TofuItem, as_forget_items, load_fixture
from ..hardware import HardwareProfile, detect
from ..logging_utils import JsonlWriter, get_logger
from ..memory.blocklist import IDBlocklist, NoBlocklist, build_blocklist
from ..memory.store import MemoryStore
from ..models.loader import load_lm
from ..orchestrator.loop import EpisodePolicies, run_episode
from ..paths import append_manifest, git_sha, make_run_id, run_dir
from ..seeding import set_all_seeds

__all__ = ["build_agent", "execute_condition", "run_condition", "seed_store"]

log = get_logger(__name__)


def build_agent(
    agent_cfg: AgentConfig,
    cfg: RDLConfig,
    hw: HardwareProfile,
    *,
    token: str | None = None,
) -> LLMAgent:
    """Construct one agent. The model always comes through `models.loader.load_lm`."""
    model_cfg = cfg.models[agent_cfg.model]
    lm = load_lm(model_cfg, hw, token=token)
    detector = build_detector(
        agent_cfg.abstention.detector,
        logprob_threshold=agent_cfg.abstention.logprob_threshold,
        self_report_token=agent_cfg.abstention.self_report_token,
        ensemble_rule=agent_cfg.abstention.ensemble_rule,
    )
    return LLMAgent(
        agent_cfg.agent_id,
        lm,
        detector=detector,
        system_prompt=agent_cfg.system_prompt,
        max_new_tokens=cfg.episode.max_new_tokens,
    )


def seed_store(
    items: Sequence[TofuItem],
    cfg: RDLConfig,
) -> tuple[MemoryStore, IDBlocklist | NoBlocklist, list[str]]:
    """Build the shared store, ingest the forget set, then apply the SBU deletion.

    This is the setup SBU describes: the forget-set content was once in memory, has been
    deleted via the memory pathway, and its ids are on the blocklist. Everything the
    conditions measure happens *after* that has been done correctly — which is why
    invariant 1 is expected to hold throughout.
    """
    store = MemoryStore(
        index_backend=cfg.env.index_backend,
        embedding_dim=cfg.memory.embedding_dim,
    )

    ingested: list[str] = []
    for it in items:
        node = store.add(
            f"{it.question} {it.answer}",
            source_agent="ingest",
            source_kind="ingest",
            turn=0,
            meta={"item_id": it.item_id, "split": it.split},
        )
        ingested.append(node.node_id)

    if cfg.memory.blocklist == "none":
        return store, NoBlocklist(), []

    blocklist = build_blocklist("id")
    assert isinstance(blocklist, IDBlocklist)
    for nid in ingested:
        store.delete(nid, blocklist=blocklist)
    return store, blocklist, ingested


def execute_condition(
    cfg: RDLConfig,
    items: Sequence[TofuItem],
    hw: HardwareProfile,
    seed: int,
    *,
    token: str | None = None,
    delegation_override: str | None = None,
) -> dict:
    """Run one condition at one seed. Returns a JSON-safe result dict."""
    set_all_seeds(seed)

    store, blocklist, blocked_ids = seed_store(items, cfg)
    baseline = store.snapshot()  # C1/C2/C3 must all start from an identical state

    agent_a = build_agent(cfg.agent_a, cfg, hw, token=token)
    agents = [agent_a]
    if cfg.agent_b is not None:
        agents.append(build_agent(cfg.agent_b, cfg, hw, token=token))

    policy_name = delegation_override or cfg.agent_a.delegation.policy
    policies = EpisodePolicies(
        delegation=build_delegation_policy(policy_name, cfg.agent_a.delegation.max_delegations),
        write=build_write_policy(
            cfg.writepolicy.mode,
            source_kinds=tuple(cfg.writepolicy.write_source_kinds),
            write_on_abstention=cfg.writepolicy.write_on_abstention,
            mirrors_framework=cfg.writepolicy.mirrors_framework,
            threshold=cfg.writepolicy.sanitize_threshold,
        ),
        blocklist=blocklist,
        retrieval_k=cfg.episode.retrieval_k,
        max_turns=cfg.episode.max_turns,
    )

    transcripts = []
    for it in items:
        tr = run_episode(
            it.question,
            agents,
            store,
            policies,
            item_id=it.item_id,
            condition=cfg.condition,
            seed=seed,
        )
        transcripts.append(tr)

    forget_items = as_forget_items(items)
    results = [
        containment(tr, it.answer, "normalised", store=store)
        for tr, it in zip(transcripts, items, strict=True)
    ]
    laundering = laundered_items(
        transcripts, store, store.dag, blocklist, forget_items, mode="normalised"
    )

    for a in agents:
        a.close()

    return {
        "condition": cfg.condition,
        "seed": seed,
        "delegation_policy": policy_name,
        "write_policy": cfg.writepolicy.mode,
        "n_items": len(items),
        "n_blocked_nodes": len(blocked_ids),
        "recall_at_k": recall_table(results, cfg.episode.max_turns),
        "delegation_rate": delegation_rate(transcripts),
        "laundering": laundering.to_dict(),
        "store_stats": store.stats(),
        "baseline_store_size": len(baseline.nodes),
        "_transcripts": transcripts,
        "_containment": results,
    }


def run_condition(
    condition: Path = typer.Option(..., "--condition", help="path to configs/conditions/CX.yaml"),
    seeds: int = typer.Option(5, "--seeds", help="number of seeds (0..seeds-1)"),
    override: list[str] = typer.Option([], "--set", help="OmegaConf dotlist override, repeatable"),
    fixture: Path | None = typer.Option(
        None, "--fixture", help="TOFU JSON fixture (offline). Default: the checked-in sample."
    ),
    limit: int | None = typer.Option(None, "--limit", help="cap the number of items"),
    controls: bool = typer.Option(
        True,
        "--controls/--no-controls",
        help="also run the always_delegate arm. Do NOT disable for a reported result.",
    ),
    token: str | None = typer.Option(None, "--hf-token"),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the resolved config and exit"),
) -> None:
    """Run one condition across seeds and write the results directory."""
    cfg = load_config(condition, override)
    hw = detect()
    chash = config_hash(cfg)

    typer.echo(f"condition  {cfg.condition}  ({cfg.description})")
    typer.echo(f"hardware   {hw.summary()}")
    typer.echo(f"config     sha256={chash[:16]}")

    if dry_run:
        typer.echo(json.dumps(cfg.model_dump(mode="json"), indent=2)[:6000])
        return

    items = load_fixture(fixture)
    if limit:
        items = items[:limit]
    typer.echo(f"items      {len(items)}")

    run_id = make_run_id(chash)
    out = run_dir(run_id)

    per_seed: list[dict] = []
    per_seed_control: list[dict] = []

    for s in range(seeds):
        typer.echo(f"  seed {s} ...")
        res = execute_condition(cfg, items, hw, s, token=token)
        with JsonlWriter(out / f"transcripts_seed{s}.jsonl", append=False) as w:
            for tr in res.pop("_transcripts"):
                w.write_all(tr.events)
        res.pop("_containment", None)
        per_seed.append(res)

        if controls and cfg.agent_b is not None:
            ctl = execute_condition(
                cfg, items, hw, s, token=token, delegation_override="always_delegate"
            )
            ctl.pop("_transcripts", None)
            ctl.pop("_containment", None)
            per_seed_control.append(ctl)

    # ---- aggregate ---------------------------------------------------------------
    def series(records: list[dict], surface: str) -> list[float]:
        return [r["recall_at_k"][surface] for r in records]

    aggregated = {
        surface: summarise_seeds(series(per_seed, surface)).to_dict() for surface in SURFACES
    }
    laundering_series = [r["laundering"]["laundering_rate"] for r in per_seed]
    delegation_series = [r["delegation_rate"] for r in per_seed]
    laundering_stats = summarise_seeds(laundering_series).to_dict()
    delegation_stats = summarise_seeds(delegation_series).to_dict()

    payload: dict[str, Any] = {
        "run_id": run_id,
        "phase": "phase0_days3-5",
        "condition": cfg.condition,
        "config_hash": chash,
        "config": cfg.model_dump(mode="json"),
        "hardware": hw.to_dict(),
        "git_sha": git_sha(),
        "n_seeds": seeds,
        "n_items": len(items),
        "recall_at_k": aggregated,
        "laundering_rate": laundering_stats,
        "delegation_rate": delegation_stats,
        "per_seed": per_seed,
        "control_always_delegate": {
            "per_seed": per_seed_control,
            "recall_at_k": (
                {
                    surface: summarise_seeds(series(per_seed_control, surface)).to_dict()
                    for surface in SURFACES
                }
                if per_seed_control
                else {}
            ),
        },
    }
    (out / "condition_report.json").write_text(json.dumps(payload, indent=2, default=str))

    append_manifest(
        {
            "run_id": run_id,
            "phase": "phase0_condition",
            "condition": cfg.condition,
            "config_hash": chash,
            "n_seeds": seeds,
            "sys_recall_store": aggregated["persistent_store_after_episode"]["mean"],
            "laundering_rate": laundering_stats["mean"],
            "git_sha": git_sha(),
        }
    )

    store_recall = aggregated["persistent_store_after_episode"]
    typer.echo(
        f"\nSysRecall@k (persistent store) = {store_recall['mean']:.3f} "
        f"95% CI {store_recall['ci95']}"
    )
    typer.echo(f"laundering_rate               = {laundering_stats['mean']:.3f}")
    typer.echo(f"delegation_rate               = {delegation_stats['mean']:.3f}")
    typer.echo(f"\nwrote {out}")
