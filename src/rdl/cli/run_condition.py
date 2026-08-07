"""`rdl run-condition` — the Phase-0 condition grid.

         Agent A        Agent B                     Write-back         Purpose
    C0   NPO forget10   -                           -                  sanity, empty store
    C1   NPO forget10   same checkpoint             disabled           no-write floor
    C1W  NPO forget10   -                           framework_default  SINGLE-AGENT BASELINE
    C2   NPO forget10   full (NOT unlearned)        framework_default  ceiling / mechanism
    C3   NPO forget10   same checkpoint             framework_default  redundancy control
    C3D  NPO forget10   INDEPENDENT NPO forget10    framework_default  ensemble treatment
    C3C  NPO forget10   INDEPENDENT + A's answer    framework_default  compositional treatment

**The primary estimand is C3D - C1W**, not C3 - C1.

Why the change (ADR-0018, docs/00b_preregistration_v2.md). C1 disables write-back, so
its persistent-store recall is structurally zero — `C3 - C1` measures "we enabled
writing", which is true by construction. And C3 loads one checkpoint into both agent
slots, so with greedy decoding agent B reproduces agent A exactly. The original pair
therefore compared a single agent against itself with the store turned on.

C1W is one agent writing back (parametric-to-memory backflow, the phenomenon SBU
already names). C3D is two independently unlearned agents. C3C adds the handoff. The
multi-agent claim is what survives above C1W; the collaboration claim is C3C - C3D.

**C2 is not the test.** With B un-unlearned, containment approaches ceiling and any
bar is met trivially. C2 shows the write path transports content at all.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import typer

from ..agents.abstention import build_detector
from ..agents.delegation import build_delegation_policy
from ..agents.llm_agent import LLMAgent
from ..agents.writer import build_write_policy
from ..config import AgentConfig, RDLConfig, config_hash, load_config
from ..eval.aggregate import summarise_seeds
from ..eval.containment import (
    SURFACES,
    ContainmentResult,
    Surface,
    containment,
    recall_table,
)
from ..eval.controls import compute_controls
from ..eval.laundering import laundered_items
from ..eval.tofu_data import TofuItem, as_forget_items, cluster_ids, load_items
from ..hardware import HardwareProfile, detect
from ..logging_utils import JsonlWriter, get_logger
from ..memory.blocklist import Blocklist, NoBlocklist, build_blocklist
from ..memory.node import MemoryNode
from ..memory.store import MemoryStore
from ..models.loader import load_lm
from ..orchestrator.loop import EpisodePolicies, run_episode
from ..paths import append_manifest, git_sha, make_run_id, run_dir
from ..seeding import set_all_seeds

__all__ = ["ArmResult", "build_agent", "execute_condition", "run_condition", "seed_store"]

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
        prompt_style=cfg.episode.prompt_style,
    )


def seed_store(
    items: Sequence[TofuItem],
    cfg: RDLConfig,
) -> tuple[MemoryStore, Blocklist, list[str]]:
    """Build the shared store, ingest the forget set, then apply the deletion.

    This is the setup the defence describes: the forget-set content was once in memory,
    has been deleted via the memory pathway, and its ids are on the blocklist.
    Everything the conditions measure happens *after* that has been done correctly —
    which is why invariant 1 is expected to hold throughout.

    Two things this function used to get wrong, both of which produced numbers:

    1. It ingested every question+answer pair and then, when `memory.blocklist == none`,
       returned WITHOUT deleting them. C0 — the "bare checkpoint" sanity condition —
       could therefore retrieve the ground-truth answers straight out of the store.
       `memory.ingest_forget_set` now governs ingestion, and C0 sets it false.
    2. It hard-coded `build_blocklist("id")` even when the config asked for `semantic`,
       so the Phase-2 defence could not be exercised at all. The blocklist kind now
       comes from the config, and the semantic one is seeded with the deleted content.
    """
    store = MemoryStore(
        index_backend=cfg.env.index_backend,
        embedding_dim=cfg.memory.embedding_dim,
        deletion_mode=cfg.memory.deletion_mode,
        refcount_semantics=cfg.memory.refcount_semantics,
    )

    if not cfg.memory.ingest_forget_set:
        # Nothing was ever in memory, so there is nothing to delete and nothing to
        # block. An empty store is the honest starting state for a bare-checkpoint arm.
        return store, build_blocklist(cfg.memory.blocklist), []

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
        # Ingested but undefended. Nothing is deleted, and the store openly contains the
        # answers — which must be visible in the report rather than mistaken for a
        # property of the model.
        return store, NoBlocklist(), []

    blocklist = build_blocklist(
        cfg.memory.blocklist,
        texts=(
            [f"{it.question} {it.answer}" for it in items]
            if cfg.memory.blocklist == "semantic"
            else None
        ),
        threshold=cfg.memory.semantic_threshold,
    )
    for nid in ingested:
        store.delete(nid, blocklist=blocklist)
    return store, blocklist, ingested


class ArmResult:
    """One (condition, seed, routing) run: transcripts, per-item hits, and the store."""

    __slots__ = ("containment", "laundering", "snapshots", "store", "transcripts")

    def __init__(
        self,
        transcripts: list,
        containment_results: list[ContainmentResult],
        snapshots: dict[str | None, list[MemoryNode]],
        store: MemoryStore,
        laundering,
    ) -> None:
        self.transcripts = transcripts
        self.containment = containment_results
        self.snapshots = snapshots
        self.store = store
        self.laundering = laundering

    def hit_vector(self, items: Sequence[TofuItem], surface: str) -> list[float]:
        """Per-item 0/1 recovery indicator, in `items` order. Feeds the paired bootstrap."""
        by_item = {r.item_id: r for r in self.containment}
        out: list[float] = []
        for it in items:
            r = by_item.get(it.item_id)
            s = r.surfaces.get(surface) if r else None
            out.append(1.0 if (s and s.hit) else 0.0)
        return out


def _episode_order(items: Sequence[TofuItem], seed: int, permute: bool) -> list[TofuItem]:
    """Episode order for one seed.

    With greedy decoding and a fixed order, five seeds produce five identical numbers
    and the seed-level CI collapses to a point. Permuting the order makes the seed enter
    the measurement through the one channel that genuinely varies: the shared store is
    cumulative, so which episode runs first changes what later episodes can retrieve.
    """
    if not permute:
        return list(items)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(items))
    return [items[int(i)] for i in idx]


def execute_condition(
    cfg: RDLConfig,
    items: Sequence[TofuItem],
    hw: HardwareProfile,
    seed: int,
    *,
    token: str | None = None,
    delegation_override: str | None = None,
    agents_override: list[LLMAgent] | None = None,
    single_agent: bool = False,
    close_agents: bool = True,
) -> ArmResult:
    """Run one arm at one seed.

    `single_agent` drops agent B without touching the config — that is how the
    "agent A alone" utility control is produced from a two-agent condition.
    """
    set_all_seeds(seed)

    store, blocklist, _ = seed_store(items, cfg)

    if agents_override is not None:
        agents = agents_override
    else:
        agents = [build_agent(cfg.agent_a, cfg, hw, token=token)]
        if cfg.agent_b is not None and not single_agent:
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
        pass_primary_answer_to_secondary=cfg.episode.pass_primary_answer,
    )

    ordered = _episode_order(items, seed, cfg.episode.permute_item_order_per_seed)

    transcripts = []
    snapshots: dict[str | None, list[MemoryNode]] = {}
    for it in ordered:
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
        # The store as it stood when THIS episode ended. Scoring every episode against
        # the final store credits episode i with nodes episode i+n wrote — an
        # "after episode" number that is really an "after everything" number.
        snapshots[it.item_id] = [n.model_copy(deep=False) for n in store.all_nodes()]

    by_item = {tr.item_id: tr for tr in transcripts}
    results = [
        containment(
            by_item[it.item_id],
            it.answer,
            "normalised",
            store_nodes=snapshots.get(it.item_id, []),
        )
        for it in items
        if it.item_id in by_item
    ]
    laundering = laundered_items(
        transcripts,
        store,
        store.dag,
        blocklist,
        as_forget_items(items),
        mode="normalised",
        store_snapshots=snapshots,
    )

    if close_agents and agents_override is None:
        for a in agents:
            a.close()

    return ArmResult(transcripts, results, snapshots, store, laundering)


def _arm_record(cfg: RDLConfig, arm: ArmResult, seed: int, policy: str, label: str) -> dict:
    from ..eval.controls import delegation_rate

    return {
        "arm": label,
        "condition": cfg.condition,
        "seed": seed,
        "delegation_policy": policy,
        "write_policy": cfg.writepolicy.mode,
        "n_items": len(arm.containment),
        "recall_at_k": recall_table(arm.containment, cfg.episode.max_turns),
        "delegation_rate": delegation_rate(arm.transcripts),
        "laundering": arm.laundering.to_dict(),
        "store_stats": arm.store.stats(),
    }


def run_condition(
    condition: Path = typer.Option(..., "--condition", help="path to configs/conditions/CX.yaml"),
    seeds: int = typer.Option(5, "--seeds", help="number of seeds (0..seeds-1)"),
    override: list[str] = typer.Option([], "--set", help="OmegaConf dotlist override, repeatable"),
    fixture: Path | None = typer.Option(
        None, "--fixture", help="TOFU JSON fixture (offline smoke test). Implies --allow-fixture."
    ),
    allow_fixture: bool = typer.Option(
        False,
        "--allow-fixture",
        help="acknowledge that fixture results are a smoke test, not a result",
    ),
    limit: int | None = typer.Option(
        None, "--limit", help="cap the number of items (recorded as `truncated` in the report)"
    ),
    controls: bool = typer.Option(
        True,
        "--controls/--no-controls",
        help="run every confound control: always_delegate, the retain arm, and agent A "
        "alone. Do NOT disable for a reported result.",
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

    # ---- data ----------------------------------------------------------------------
    # There is no reason to pass --allow-fixture except to run on the fixture, so it
    # selects it as well as acknowledging it. Without that, the flag would only mean
    # "I would have permitted the fixture", and the run would still go to the Hub.
    use_fixture = allow_fixture or fixture is not None
    items, provenance = load_items(
        dataset="stub" if use_fixture else cfg.data.dataset,
        split=cfg.data.forget_split,
        n_items=limit or cfg.data.n_items,
        fixture=fixture,
        token=token,
        allow_fixture=use_fixture,
    )
    typer.echo(f"items      {len(items)} from {provenance['source']} ({cfg.data.forget_split})")
    if not provenance["is_real_data"]:
        typer.secho(
            "  SMOKE TEST ONLY — this is the development fixture, not TOFU. "
            "Nothing produced here is reportable.",
            fg=typer.colors.YELLOW,
        )

    retain_items: list[TofuItem] = []
    retain_provenance: dict = {"source": "none", "n_items": 0, "is_real_data": False}
    if controls and cfg.data.n_retain_items > 0 and provenance["is_real_data"]:
        retain_items, retain_provenance = load_items(
            dataset=cfg.data.dataset,
            split=cfg.data.retain_split,
            n_items=cfg.data.n_retain_items,
            token=token,
            # Spread across authors, not the head of the split. TOFU splits are
            # contiguous author blocks, so the first 100 retain items are five
            # novelists — a false-positive floor measured on them says nothing about
            # the other 175.
            sample="spread",
        )
        typer.echo(f"retain     {len(retain_items)} from {cfg.data.retain_split} (control arm)")
    elif controls and not provenance["is_real_data"]:
        typer.secho(
            "  retain control arm skipped: the fixture has no retain split.",
            fg=typer.colors.YELLOW,
        )

    clusters = cluster_ids(items, cfg.data.cluster_by)

    run_id = make_run_id(chash)
    out = run_dir(run_id)

    per_seed: list[dict] = []
    per_seed_control: list[dict] = []
    per_seed_retain: list[dict] = []
    per_seed_a_alone: list[dict] = []
    control_reports: list[dict] = []

    # Per-item hit vectors, accumulated across seeds. These are what the paired
    # bootstrap resamples; the seed-level means are reported alongside but are not the
    # primary interval. See eval/aggregate.py.
    primary_surface: Surface = "persistent_store_after_episode"
    hit_vectors: list[list[float]] = []

    policy_name = cfg.agent_a.delegation.policy

    for s in range(seeds):
        typer.echo(f"  seed {s} ...")
        arm = execute_condition(cfg, items, hw, s, token=token)
        with JsonlWriter(out / f"transcripts_seed{s}.jsonl", append=False) as w:
            for tr in arm.transcripts:
                w.write_all(tr.events)
        per_seed.append(_arm_record(cfg, arm, s, policy_name, "treatment"))
        hit_vectors.append(arm.hit_vector(items, primary_surface))

        if not controls:
            continue

        ctl_arm = None
        if cfg.agent_b is not None:
            ctl_arm = execute_condition(
                cfg, items, hw, s, token=token, delegation_override="always_delegate"
            )
            per_seed_control.append(
                _arm_record(cfg, ctl_arm, s, "always_delegate", "always_delegate")
            )

        retain_arm = None
        if retain_items:
            retain_arm = execute_condition(cfg, retain_items, hw, s, token=token)
            per_seed_retain.append(_arm_record(cfg, retain_arm, s, policy_name, "retain"))

        a_alone_arm = None
        if cfg.agent_b is not None and retain_items:
            # Control 2: how much of the system's advantage on RETAIN questions is plain
            # ensemble benefit? Agent A answering retain questions alone is the
            # comparator; without it "the system beats one agent" is unquantified.
            a_alone_arm = execute_condition(
                cfg, retain_items, hw, s, token=token, single_agent=True
            )
            per_seed_a_alone.append(
                _arm_record(cfg, a_alone_arm, s, "never", "agent_a_alone_retain")
            )

        control_reports.append(
            compute_controls(
                forget_transcripts=arm.transcripts,
                retain_transcripts=retain_arm.transcripts if retain_arm else (),
                results_abstention=arm.containment,
                results_always_delegate=ctl_arm.containment if ctl_arm else (),
                results_retain=retain_arm.containment if retain_arm else (),
                agent_only_retain=a_alone_arm.containment if a_alone_arm else (),
                system_retain=retain_arm.containment if retain_arm else (),
                surface=primary_surface,
                k=cfg.episode.max_turns,
            ).to_dict()
        )

    # ---- aggregate -------------------------------------------------------------------
    def series(records: list[dict], surface: str) -> list[float]:
        return [r["recall_at_k"][surface] for r in records]

    aggregated = {
        surface: summarise_seeds(series(per_seed, surface)).to_dict() for surface in SURFACES
    }
    laundering_stats = summarise_seeds(
        [r["laundering"]["laundering_rate"] for r in per_seed]
    ).to_dict()
    delegation_stats = summarise_seeds([r["delegation_rate"] for r in per_seed]).to_dict()

    # Per-item recovery averaged over seeds — the vector another condition's report is
    # paired against by `make-report`. Written out so the gate can be recomputed from
    # the reports alone, with no rerun.
    mean_hits = (
        np.mean(np.asarray(hit_vectors, dtype=float), axis=0).tolist() if hit_vectors else []
    )

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
        "data_provenance": provenance,
        "retain_provenance": retain_provenance,
        "controls_enabled": controls,
        "primary_surface": primary_surface,
        "recall_at_k": aggregated,
        "laundering_rate": laundering_stats,
        "delegation_rate": delegation_stats,
        # Paired-bootstrap inputs. `item_ids` and `clusters` are positional companions
        # of `per_item_recall`; make-report zips them.
        "item_ids": [it.item_id for it in items],
        "clusters": clusters,
        "cluster_by": cfg.data.cluster_by,
        "per_item_recall": mean_hits,
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
        "control_retain": {"per_seed": per_seed_retain},
        "control_agent_a_alone_retain": {"per_seed": per_seed_a_alone},
        "control_reports": control_reports,
    }
    (out / "condition_report.json").write_text(json.dumps(payload, indent=2, default=str))

    append_manifest(
        {
            "run_id": run_id,
            "phase": "phase0_condition",
            "condition": cfg.condition,
            "config_hash": chash,
            "n_seeds": seeds,
            "n_items": len(items),
            "data_source": provenance["source"],
            "is_real_data": provenance["is_real_data"],
            "sys_recall_store": aggregated[primary_surface]["mean"],
            "laundering_rate": laundering_stats["mean"],
            "git_sha": git_sha(),
        }
    )

    store_recall = aggregated[primary_surface]
    typer.echo(
        f"\nSysRecall@k (persistent store) = {store_recall['mean']:.3f} "
        f"95% CI {store_recall['ci95']}"
    )
    if store_recall.get("degenerate_replicates") and seeds > 1:
        typer.secho(
            "  every seed returned the same value — that CI has zero width by "
            "construction and is not an uncertainty estimate. The paired item-level "
            "interval in `make-report` is the one to quote.",
            fg=typer.colors.YELLOW,
        )
    typer.echo(f"laundering_rate               = {laundering_stats['mean']:.3f}")
    typer.echo(f"delegation_rate               = {delegation_stats['mean']:.3f}")
    if not controls:
        typer.secho(
            "\n--no-controls was passed. This run cannot be reported: the confound "
            "gate (docs/00b_preregistration_v2.md §4.2) is unevaluated.",
            fg=typer.colors.RED,
        )
    typer.echo(f"\nwrote {out}")
