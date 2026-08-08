"""`rdl run-condition` — the Phase-0 condition grid.

         Agent A        Agent B                     Write-back         Purpose
    C0   NPO forget10   -                           -                  sanity, empty store
    C1   NPO forget10   same checkpoint             disabled           no-write floor
    C1W  NPO forget10   -                           framework_default  SINGLE-AGENT BASELINE
    C2   NPO forget10   full (NOT unlearned)        framework_default  ceiling / mechanism
    C3   NPO forget10   same checkpoint             framework_default  redundancy control
    C3D  NPO forget10   INDEPENDENT NPO forget10    framework_default  ensemble treatment
    C3C  NPO forget10   INDEPENDENT + A's answer    framework_default  compositional treatment

    B1W  INDEP forget10 -                          framework_default  B-ALONE BASELINE
    C3S  NPO forget10   INDEPENDENT + ANOTHER      framework_default  PROMPT-MATCHED CONTROL
                        item's A answer

**The primary estimands are C3C - C3S and content_specific_joint_recovery**
(docs/00e_preregistration_v5.md §3).

Why `C3C - C3S` and not `C3C - C3D` (ADR-0048). C3D hands agent B a bare question; C3C
hands it a labelled peer-message block. Their difference therefore varies agent A's
information, the presence of any context, "another assistant" priming, and prompt length
and format ALL AT ONCE — so a positive result is exactly as consistent with "any
peer-shaped message elicits B's suppressed knowledge". C3S keeps the wrapper
byte-identical and changes one thing: the handed-over text answers a DIFFERENT item's
question. `C3C - C3S` is A's content with the wrapper held fixed; `C3S - C3D` is the
wrapper alone.

Why both standalone baselines, and why C3S appears in the joint metric too (ADR-0042,
ADR-0056):

    content_specific_joint_recovery = C3C_hit AND NOT C3S_hit
                                              AND NOT C1W_hit AND NOT B1W_hit   PRIMARY
    joint_only_recovery             = C3C_hit AND NOT C1W_hit AND NOT B1W_hit   secondary

The first separates composition from either agent's residual AND from what an unrelated
peer-shaped message already elicits; the second drops the `NOT C3S` term and so remains a
system-level diagnostic only. `C3D - C1W` and `C3D - B1W` are secondary; C1 has write-back
disabled so its store recall is structurally zero and `C3 - C1` measures "we enabled
writing", true by construction.

**Routing.** C3D, C3S and C3C route UNCONDITIONALLY. Under `abstention_triggered` agent B
is called only when A abstains, and the loop used to withhold A's text on exactly those
episodes — so the C3C handoff never fired and C3C was byte-identical to C3D (ADR-0041).
The abstention-routed variants are still run as the ecological arm: every two-agent
condition runs both routings, so the confound gate can recompute the
treatment-minus-baseline delta on routing-free arms (ADR-0044). The DELEGATION GAP is
computed from the ecological forget and retain arms, never from the unconditional ones,
where both rates are 1 by construction (ADR-0049).

**C2 is not the test.** With B un-unlearned, containment approaches ceiling and any
bar is met trivially. C2 shows the write path transports content at all.
"""

from __future__ import annotations

import hashlib
import json
import platform
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import typer

from ..agents.abstention import build_detector
from ..agents.delegation import build_delegation_policy
from ..agents.llm_agent import LLMAgent
from ..agents.writer import build_write_policy
from ..config import AgentConfig, ConfigError, RDLConfig, config_hash, load_config
from ..eval.aggregate import summarise_seeds
from ..eval.containment import (
    SURFACES,
    ContainmentResult,
    Surface,
    containment,
    recall_table,
)
from ..eval.controls import compute_controls
from ..eval.laundering import laundered_items, merge_reports
from ..eval.negatives import CrossAuthorMapping, cross_author_mapping, deranged_targets
from ..eval.tofu_data import TofuItem, as_forget_items, cluster_ids, load_items
from ..hardware import (
    EnvHardwareMismatch,
    HardwareProfile,
    assert_env_matches_hardware,
    detect,
)
from ..logging_utils import JsonlWriter, get_logger
from ..memory.blocklist import Blocklist, NoBlocklist, build_blocklist
from ..memory.index import normalise_text
from ..memory.node import MemoryNode
from ..memory.store import MemoryStore
from ..models.loader import load_lm
from ..models.stub import LMHandle
from ..orchestrator.loop import EpisodePolicies, run_episode
from ..paths import (
    append_manifest,
    git_diff_sha256,
    git_dirty,
    git_sha,
    make_run_id,
    run_dir,
)
from ..provenance import pkg_version as _pkg_version
from ..provenance import tokenizer_provenance
from ..seeding import set_all_seeds

__all__ = [
    "ArmResult",
    "build_agent",
    "build_shared_agents",
    "execute_condition",
    "run_condition",
    "seed_store",
]

log = get_logger(__name__)

# The routing the delegation gap is a claim ABOUT. C3D/C3S/C3C run unconditionally, so
# their primary arms have delegation_rate == 1 on both forget and retain and the gap is 0
# by construction. See ADR-0049.
ECOLOGICAL_POLICY = "abstention_triggered"


def build_agent(
    agent_cfg: AgentConfig,
    cfg: RDLConfig,
    hw: HardwareProfile,
    *,
    token: str | None = None,
    lm: LMHandle | None = None,
) -> LLMAgent:
    """Construct one agent. The model always comes through `models.loader.load_lm`.

    `lm` lets the caller supply an already-loaded handle so one checkpoint can back two
    agent slots (C3) and survive across seeds and control arms. The agent itself is
    stateless — it holds a detector, a prompt style, and a reference — so reuse changes
    nothing about what is measured.
    """
    model_cfg = cfg.models[agent_cfg.model]
    if lm is None:
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


def build_shared_agents(
    cfg: RDLConfig,
    hw: HardwareProfile,
    *,
    token: str | None = None,
) -> list[LLMAgent]:
    """Load this condition's agents ONCE, to be reused by every seed and every arm.

    The runner used to build and tear down the models inside `execute_condition`, which
    is called for the treatment, the always-delegate control, the retain arm and the
    agent-A-alone arm, at every seed: ~195 model initialisations for the full grid, each
    re-reading weights from the HF cache and re-allocating VRAM. On a rented GPU that is
    most of the bill, and the repeated allocate/free cycle is also the main source of
    CUDA memory fragmentation across a long run.

    Reuse is sound because an `LLMAgent` carries no per-run state: the store, the
    blocklist, the transcripts and the item ordering are all rebuilt per arm inside
    `execute_condition`. Only the weights are shared.

    Two agent slots pointing at the SAME checkpoint (C3, C1) share one handle rather than
    loading it twice. With greedy decoding that is already the same computation — it is
    exactly why C3 is a redundancy control — so the second copy only ever cost VRAM.
    Stub models are not shared: they are free to build and their call log is per-instance.
    """
    handles: dict[str, LMHandle] = {}
    agents: list[LLMAgent] = []
    for agent_cfg in (cfg.agent_a, cfg.agent_b):
        if agent_cfg is None:
            continue
        model_cfg = cfg.models[agent_cfg.model]
        shared = handles.get(agent_cfg.model) if model_cfg.kind == "hf" else None
        agent = build_agent(agent_cfg, cfg, hw, token=token, lm=shared)
        if model_cfg.kind == "hf":
            handles.setdefault(agent_cfg.model, agent.lm)
        agents.append(agent)
    return agents


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

    __slots__ = (
        "containment",
        "containment_negative",
        "handoff_audit",
        "laundering",
        "snapshots",
        "store",
        "transcripts",
    )

    def __init__(
        self,
        transcripts: list,
        containment_results: list[ContainmentResult],
        snapshots: dict[str | None, list[MemoryNode]],
        store: MemoryStore,
        laundering,
        containment_negative: list[ContainmentResult] | None = None,
        handoff_audit: dict | None = None,
    ) -> None:
        self.transcripts = transcripts
        self.containment = containment_results
        self.snapshots = snapshots
        self.store = store
        self.laundering = laundering
        # The same episodes scored against DERANGED targets — another item's answer. On
        # the retain arm this is the false-positive floor; scoring against the correct
        # answer measures utility, not a floor (ADR-0043).
        self.containment_negative = containment_negative or []
        # C3S's mapping and its leak check. A control whose mapping cannot be recomputed
        # and audited is not a control (ADR-0055).
        self.handoff_audit = handoff_audit or {}

    @property
    def n_handoffs(self) -> int:
        return sum(t.n_handoffs for t in self.transcripts)

    @property
    def n_delegations(self) -> int:
        """Delegations actually performed. The denominator the handoff count must match."""
        return sum(len(t.delegations()) for t in self.transcripts)

    @property
    def n_shuffled_handoffs(self) -> int:
        """Handoffs carrying another item's answer — C3S's defining count."""
        return sum(1 for t in self.transcripts for h in t.handoffs() if h.shuffled)

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


def _handoff_source_map(items: Sequence[TofuItem], cfg: RDLConfig) -> CrossAuthorMapping:
    """Which item's answer C3S hands over for each target item. Seed-independent.

    THE PROMPT-MATCHED CONTROL (ADR-0048). C3D hands agent B a bare question; C3C hands
    it a labelled peer-message block, so `C3C - C3D` varies agent A's information, the
    presence of any context, "another assistant" priming, and prompt length and format at
    once. C3S keeps the wrapper byte-identical and changes exactly one thing: whose
    question the handed-over text answers.

    THE MAPPING THIS REPLACES (ADR-0055). It used to be `derange(n, seed)`:

    * **Seeded**, so the seed changed the TEXT agent B receives, not merely episode
      order. The v4 argument for one primary seed — "order cannot matter once the store
      resets per item" — was therefore false for this arm, and `C3C - C3S` rested on one
      arbitrary distractor assignment.
    * **Authorship-blind.** TOFU is 200 invented authors x 20 questions, so a random
      derangement paired 13-23 of 400 items with another question about the SAME author.
      Another question about the same novelist can carry the target name or its
      supporting facts outright — the control leaking the very content it exists to
      withhold, biasing `C3C - C3S` toward zero.

    `cross_author_mapping` rotates by the smallest shift that never pairs two items by one
    author. On the canonical contiguous forget10 layout that is exactly
    `QUESTIONS_PER_AUTHOR`, i.e. `source = (target + 20) % 400`.
    """
    return cross_author_mapping(cluster_ids(items, "author"))


def _source_answer(
    source: TofuItem,
    primary: LLMAgent,
    store: MemoryStore,
    blocklist: Blocklist,
    policies: EpisodePolicies,
) -> tuple[str, bool]:
    """Agent A's answer to `source`, produced against the store as it stands RIGHT NOW.

    Run immediately before the target episode, against the *same* store state that
    episode will see. Under `store_scope: per_item` that is the fresh post-deletion
    snapshot, so nothing changes. Under `cumulative` it matters a great deal: a
    precomputed pass would have generated every source answer against an empty
    post-deletion store while C3C's handed-over answer saw the accumulated shared store,
    so `C3C - C3S` would have varied the handed-over content AND its memory context
    together. See ADR-0055.

    **Retrieve, then answer — deliberately NOT `run_episode`.** The probe used to run a
    full single-agent episode under `DisabledWritePolicy` and `NeverDelegate`, which
    wrote no node but still executed `store.turn = turn` on its way to the write-back
    step. v5 claims this probe leaves the measured store untouched; a claim enforced by
    "no extra node appeared" is not that claim, and the turn counter is store state the
    next episode's writes are stamped with. The two calls below are exactly what the
    episode loop does for a primary with no delegate, minus every mutation. See ADR-0057.
    """
    retrieved = store.retrieve(
        source.question,
        k=policies.retrieval_k,
        blocklist=blocklist,
    )
    reply = primary.answer(source.question, retrieved.nodes)
    return reply.text, reply.abstained


def _audit_handoffs(
    items: Sequence[TofuItem],
    transcripts: Sequence[Any],
    source_map: CrossAuthorMapping | None,
) -> dict:
    """Everything a reviewer needs to believe the control was a control (ADR-0055).

    `target_answer_in_handoff` is the leak check the mapping exists to prevent: if the
    text handed to agent B already contains the TARGET item's answer, C3S is not a
    negative control at all, and `C3C - C3S` would be biased toward zero — which reads as
    "the wrapper explains everything" and would have taken 're-derivation' out of a title
    it might have deserved.
    """
    if source_map is None:
        return {}
    answers = {it.item_id: it.answer for it in items}
    clusters = dict(zip([it.item_id for it in items], cluster_ids(items, "author"), strict=True))

    leaked: list[str] = []
    same_author: list[str] = []
    fixed_points: list[str] = []
    pairs: dict[str, str] = {}
    for tr in transcripts:
        for h in tr.handoffs():
            if not h.shuffled or h.source_item_id is None:
                continue
            pairs[str(tr.item_id)] = str(h.source_item_id)
            if h.source_item_id == tr.item_id:
                fixed_points.append(str(tr.item_id))
            if clusters.get(tr.item_id) == clusters.get(h.source_item_id):
                same_author.append(str(tr.item_id))
            target = answers.get(tr.item_id, "")
            if target and normalise_text(target) in normalise_text(h.text):
                leaked.append(str(tr.item_id))

    return {
        "mapping": source_map.to_dict(),
        "pairs": pairs,
        "target_author_ids": {k: clusters.get(k) for k in pairs},
        "source_author_ids": {k: clusters.get(v) for k, v in pairs.items()},
        "fixed_point_count": len(fixed_points),
        "same_author_count": len(same_author),
        "target_answer_in_handoff_count": len(leaked),
        "target_answer_in_handoff_items": sorted(leaked)[:50],
    }


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

    **Store scope.** Under `episode.store_scope: per_item` the store is rebuilt from the
    same post-deletion snapshot before every item, which is what the paired item-level
    bootstrap assumes: with one cumulative store, episode *i*'s write is part of episode
    *i+n*'s retrievable context and the items are not exchangeable. `cumulative` keeps
    the shared store and is the longitudinal experiment, reported over seeds and never
    used for the primary gate. See ADR-0047.
    """
    set_all_seeds(seed)

    store, blocklist, _ = seed_store(items, cfg)
    per_item_store = cfg.episode.store_scope == "per_item"

    if agents_override is not None:
        agents = agents_override
    else:
        agents = [build_agent(cfg.agent_a, cfg, hw, token=token)]
        if cfg.agent_b is not None and not single_agent:
            agents.append(build_agent(cfg.agent_b, cfg, hw, token=token))

    policy_name = delegation_override or cfg.effective_routing()
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

    # C3S: which item's answer each target is handed. Seed-independent and never
    # same-author — see `_handoff_source_map` and ADR-0055.
    shuffled_handoff = cfg.episode.pass_primary_answer and cfg.episode.handoff_source == "deranged"
    source_map: CrossAuthorMapping | None = None
    source_for: dict[str, TofuItem] = {}
    if shuffled_handoff:
        source_map = _handoff_source_map(items, cfg)
        source_for = {it.item_id: items[source_map.permutation[i]] for i, it in enumerate(items)}

    transcripts = []
    snapshots: dict[str | None, list[MemoryNode]] = {}
    # Under `per_item` each episode gets its own store, and certification has to run
    # against the store that actually holds the written node.
    stores: dict[str | None, tuple[MemoryStore, Blocklist]] = {}
    for it in ordered:
        if per_item_store:
            # Rebuilt from the same ingest-then-delete recipe, so every item starts from
            # a byte-identical post-deletion state and no episode can see another's write.
            store, blocklist, _ = seed_store(items, cfg)
            policies.blocklist = blocklist
        stores[it.item_id] = (store, blocklist)

        source_item: str | None = None
        peer_text: str | None = None
        peer_abstained: bool | None = None
        if shuffled_handoff:
            # Generated HERE, against the store this episode is about to see — not in a
            # precomputed pass against a fresh one. Under `cumulative` the two differ, and
            # the difference would have leaked into `C3C - C3S` as a memory-context
            # variable alongside the handed-over content (ADR-0055).
            src = source_for[it.item_id]
            source_item = src.item_id
            peer_text, peer_abstained = _source_answer(src, agents[0], store, blocklist, policies)

        tr = run_episode(
            it.question,
            agents,
            store,
            policies,
            item_id=it.item_id,
            condition=cfg.condition,
            seed=seed,
            peer_answer_override=peer_text,
            peer_answer_source_item=source_item,
            peer_answer_abstained=peer_abstained,
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

    # The same episodes against another item's answer. Nothing the system produces should
    # match these; whatever does is the containment matcher's own false-positive rate.
    negatives, _perm = deranged_targets([it.answer for it in items], seed)
    results_negative = [
        containment(
            by_item[it.item_id],
            neg,
            "normalised",
            store_nodes=snapshots.get(it.item_id, []),
        )
        for it, neg in zip(items, negatives, strict=True)
        if it.item_id in by_item
    ]

    if per_item_store:
        # One report per item, against that item's own store, then merged. Certification
        # asks "does the node carrying this answer satisfy both invariants", and the node
        # only exists in the store its episode wrote to.
        def _one(it: TofuItem):
            tr = by_item[it.item_id]
            st, bl = stores[it.item_id]
            return laundered_items(
                [tr],
                st,
                st.dag,
                bl,
                [{"item_id": it.item_id, "answer": it.answer}],
                mode="normalised",
                store_snapshots={it.item_id: snapshots.get(it.item_id, [])},
            )

        laundering = merge_reports(
            [_one(it) for it in items if it.item_id in by_item], n_items=len(items)
        )
    else:
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

    return ArmResult(
        transcripts,
        results,
        snapshots,
        store,
        laundering,
        results_negative,
        _audit_handoffs(items, transcripts, source_map),
    )


def _aggregate_handoff_audits(per_seed: Sequence[dict]) -> tuple[list[dict], dict]:
    """Every seed's C3S audit, and the union that `make-report` gates on (ADR-0057).

    The report used to hoist `per_seed[0]["handoff_audit"]` and gate on that alone. The
    MAPPING is seed-independent, so the same-author and fixed-point counts genuinely are
    too — but `target_answer_in_handoff` is a property of the generated TEXT, not of the
    mapping. Under `store_scope: cumulative` the episode order changes the live store,
    which changes agent A's source answer, which can put the target answer into a handoff
    at seed 3 that was clean at seed 0. Gating on seed 0 would have missed it and reported
    a leaking control as a valid one.

    Counts are summed across seeds rather than maxed: "one leak at one seed" and "one leak
    at every seed" are different facts and the number should say which.
    """
    by_seed: list[dict] = []
    for r in per_seed:
        audit = r.get("handoff_audit") or {}
        if not audit:
            continue
        by_seed.append({"seed": r.get("seed"), **audit})
    if not by_seed:
        return [], {}

    def _total(key: str) -> int:
        return sum(int(a.get(key) or 0) for a in by_seed)

    leaked_items: list[str] = []
    for a in by_seed:
        leaked_items.extend(str(i) for i in (a.get("target_answer_in_handoff_items") or []))

    return by_seed, {
        "n_seeds": len(by_seed),
        "seeds": [a.get("seed") for a in by_seed],
        # Exactly one hash across every seed is the evidence that the mapping is
        # seed-independent — the property the single-seed primary design rests on.
        "mapping_hashes": sorted({str((a.get("mapping") or {}).get("sha256")) for a in by_seed}),
        "mapping_algorithms": sorted(
            {str((a.get("mapping") or {}).get("algorithm")) for a in by_seed}
        ),
        "same_author_count": _total("same_author_count"),
        "fixed_point_count": _total("fixed_point_count"),
        "target_answer_in_handoff_count": _total("target_answer_in_handoff_count"),
        "target_answer_in_handoff_items": sorted(set(leaked_items))[:50],
        "seeds_with_target_answer_in_handoff": [
            a.get("seed") for a in by_seed if a.get("target_answer_in_handoff_count")
        ],
    }


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
        # Same episodes, deranged targets. On the retain arm this IS the false-positive
        # floor; on any arm it is the matcher's own error rate (ADR-0043).
        "recall_at_k_negative": recall_table(arm.containment_negative, cfg.episode.max_turns),
        "delegation_rate": delegation_rate(arm.transcripts),
        # Counted from the event log, not read off the config flag. `make-report` requires
        # EXACTLY one handoff per delegation in a handoff arm and exactly zero elsewhere;
        # "at least one" passed a run whose handoff fired on one episode in four hundred
        # (ADR-0046, tightened by ADR-0054).
        "n_handoffs": arm.n_handoffs,
        "n_delegations": arm.n_delegations,
        "n_shuffled_handoffs": arm.n_shuffled_handoffs,
        "handoff_audit": arm.handoff_audit,
        "laundering": arm.laundering.to_dict(),
        "store_stats": arm.store.stats(),
    }


def _runtime_fingerprint(cfg: RDLConfig, hw: HardwareProfile) -> dict:
    """What actually turned weights into tokens (ADR-0054).

    A condition report recorded `git_sha` and the hardware profile and nothing about the
    software. Two arms differenced across an SDPA run and an FA2 run, across two
    transformers versions, or across a moved chat template are two experiments reported
    as one — and `make-report` now blocks such a pairing, which it can only do if the
    facts are in the reports.

    Resolved, not requested: `dtype_override: null` in a model config means "ask the
    hardware", so the config alone does not say what ran. Everything here is recorded
    per model, because a grid may legitimately load two checkpoints.
    """
    from ..models.loader import resolve_attn, resolve_dtype

    models: dict[str, dict] = {}
    for name, model_cfg in sorted(cfg.models.items()):
        if model_cfg.kind != "hf":
            models[name] = {"kind": model_cfg.kind}
            continue
        models[name] = {
            "kind": "hf",
            "repo_id": model_cfg.repo_id,
            "revision": model_cfg.revision,
            "resolved_dtype": resolve_dtype(model_cfg, hw),
            "resolved_attn": resolve_attn(model_cfg, hw),
        }
    return {
        "torch_version": _pkg_version("torch"),
        "transformers_version": _pkg_version("transformers"),
        "tokenizers_version": _pkg_version("tokenizers"),
        "numpy_version": _pkg_version("numpy"),
        "python_version": platform.python_version(),
        "models": models,
        # Upstream reads the chat template from a moving branch, so it is provenance
        # rather than a pin. Recorded here for the same reason run-repro records it.
        "tokenizer": (
            tokenizer_provenance() if any(m.kind == "hf" for m in cfg.models.values()) else {}
        ),
    }


def _write_transcripts(out: Path, arm: ArmResult, label: str, seed: int) -> None:
    """Persist one arm's transcripts. EVERY arm, not just the treatment (ADR-0045).

    The runner used to write `transcripts_seed{s}.jsonl` for the treatment and discard
    the routing-variant, retain and standalone transcripts entirely — so the arms that
    the confound gate and the false-positive floor are computed from left no auditable
    trace at all.
    """
    with JsonlWriter(out / f"transcripts_{label}_seed{seed}.jsonl", append=False) as w:
        for tr in arm.transcripts:
            w.write_all(tr.events)


def _handoff_evidence(arm: ArmResult, seed: int) -> list[dict]:
    """The raw C3S/C3C evidence, in a file small enough to commit (ADR-0060).

    The full transcripts carry all of this and are ~two orders of magnitude larger, so
    `.gitignore` excludes them — which meant the one artifact the control claim rests on
    lived only on a rented instance that the runbook then tells you to destroy. This is
    the reviewable subset: for every handoff, what was handed over, whose question it
    answered, both SHA-256s so the copy can be checked against agent A's own answer
    without trusting either, and whether the carrier node was certified clean.
    """
    certified = {
        str(i.item_id): bool(i.laundered) for i in arm.laundering.items if i.item_id is not None
    }
    out: list[dict] = []
    for tr in arm.transcripts:
        answers = tr.agent_answers()
        primary = answers[0] if answers else None
        for h in tr.handoffs():
            out.append(
                {
                    "seed": seed,
                    "item_id": tr.item_id,
                    "question": tr.query_text,
                    "shuffled": h.shuffled,
                    "source_item_id": h.source_item_id,
                    "handoff_text": h.text,
                    "handoff_text_sha256": h.text_sha256,
                    "handoff_included_abstention": h.included_abstention,
                    # Agent A's own answer to THIS item. Under C3C it must equal the
                    # handed-over text; under C3S it must NOT — and the two hashes are
                    # what makes that checkable from the committed file alone.
                    "agent_a_answer": primary.text if primary else None,
                    "agent_a_abstained": primary.abstained if primary else None,
                    "agent_a_answer_sha256": (
                        hashlib.sha256(primary.text.encode("utf-8")).hexdigest()
                        if primary
                        else None
                    ),
                    "final_answer": tr.final_text,
                    "carrier_node_certified_clean": certified.get(str(tr.item_id)),
                }
            )
    return out


def run_condition(
    condition: Path = typer.Option(..., "--condition", help="path to configs/conditions/CX.yaml"),
    seeds: int | None = typer.Option(
        None,
        "--seeds",
        help="number of seeds (0..seeds-1). Default: 1 for store_scope=per_item, 5 for "
        "cumulative. Under per_item the store is rebuilt before every episode, so episode "
        "order — the only thing a seed varies under greedy decoding — cannot change any "
        "outcome, and extra seeds are copies rather than replicates (ADR-0050).",
    ),
    environment: str | None = typer.Option(
        None,
        "--env",
        help="execution environment config from configs/env, e.g. vast_rtx3090, "
        "rtx3090, colab_t4. Replaces the whole env group named by the condition file; "
        "the scientific condition is unchanged.",
    ),
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
    allow_dirty: bool = typer.Option(
        False,
        "--allow-dirty",
        help="run with uncommitted changes. The report is then marked git_dirty=true and "
        "reportable=false: a reviewer checking out the recorded commit cannot reproduce "
        "what actually ran (ADR-0054).",
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="print the resolved config and exit"),
) -> None:
    """Run one condition across seeds and write the results directory."""
    import os

    try:
        cfg = load_config(condition, override, env_override=environment)
    except ConfigError as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=2) from exc
    hw = detect()
    chash = config_hash(cfg)

    # Seeds follow the store scope unless the operator names a number. See ADR-0050.
    if seeds is None:
        seeds = cfg.required_seeds()
    if cfg.episode.store_scope == "per_item" and seeds > 1:
        typer.secho(
            f"  --seeds {seeds} under store_scope=per_item: the store is rebuilt before "
            "every episode, so episode order cannot change any outcome and these are "
            "copies of one deterministic run, not replicates. `make-report` requires "
            f"{cfg.required_seeds()} here (ADR-0050).",
            fg=typer.colors.YELLOW,
        )

    # The env profile is only a claim until it is applied and checked. HF_HOME first —
    # on a rented box the default cache lands on the container overlay, and a 2.5 GB
    # checkpoint downloaded there is re-downloaded on the next start. `setdefault` so an
    # operator who exported HF_HOME keeps their choice.
    if cfg.env.hf_home:
        os.environ.setdefault("HF_HOME", cfg.env.hf_home)

    typer.echo(f"condition  {cfg.condition}  ({cfg.description})")
    typer.echo(f"env        {cfg.env.name}  HF_HOME={os.environ.get('HF_HOME')}")
    typer.echo(f"hardware   {hw.summary()}")
    typer.echo(f"config     sha256={chash[:16]}")

    if dry_run:
        typer.echo(json.dumps(cfg.model_dump(mode="json"), indent=2)[:6000])
        return

    # A grid whose recorded commit does not contain the code that produced it is not
    # checkable. Day 1 already shipped reports in exactly that state (ADR-0040); the
    # condition runner had no guard at all until ADR-0054.
    dirty = git_dirty()
    if dirty and not allow_dirty:
        typer.secho(
            "the working tree has uncommitted changes. A reviewer checking out "
            f"{git_sha()} would not get the code this run would use.\n"
            "  -> commit first, or pass --allow-dirty to record a DIAGNOSTIC run that "
            "can never be reported.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=2)
    if dirty:
        typer.secho(
            "--allow-dirty: git_dirty=true and reportable=false are recorded. This is a "
            "diagnostic run.",
            fg=typer.colors.YELLOW,
        )

    # Then the preconditions. A grid that takes hours must not discover in its results
    # that it ran on the wrong card, in the wrong precision, or out of disk.
    try:
        assert_env_matches_hardware(cfg.env, hw)
    except EnvHardwareMismatch as exc:
        typer.secho(str(exc), fg=typer.colors.RED)
        typer.secho(
            "  -> pick the profile that matches this box: --env local_cpu | colab_t4 | "
            "rtx3090 | vast_rtx3090 | h100",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=2) from exc

    if hw.is_cuda and hw.supports_flash_attn2 and not hw.flash_attn_installed:
        typer.secho(
            f"  {hw.name} is FA2-capable but `flash_attn` is not installed: this run "
            "uses SDPA. That is recorded in the report — never mix SDPA and FA2 runs "
            "inside one comparison.",
            fg=typer.colors.YELLOW,
        )

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
        # A TRUNCATED run must spread across authors. TOFU splits are contiguous
        # 20-question author blocks, so `--limit 5` off the head gives five questions
        # about ONE novelist — and C3S then has no cross-author partner for any of them,
        # which is the one thing its control property depends on. The full run is
        # unaffected: taking all 400 items spreads by definition. See ADR-0055.
        sample="spread" if limit is not None else "head",
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
    # `--limit` scales the retain control with the forget set. Without this a
    # "five-item smoke test" still ran 100 retain items plus their ecological arm for
    # every condition: neither fast, nor a smoke test, nor comparable (ADR-0055).
    n_retain = cfg.data.n_retain_items
    if limit is not None:
        n_retain = min(n_retain, max(2, limit))
    if controls and n_retain > 0 and provenance["is_real_data"]:
        retain_items, retain_provenance = load_items(
            dataset=cfg.data.dataset,
            split=cfg.data.retain_split,
            n_items=n_retain,
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
    # Committed alongside the report; the full transcripts are not (ADR-0060).
    evidence: list[dict] = []
    per_seed_control: list[dict] = []
    per_seed_retain: list[dict] = []
    per_seed_eco_retain: list[dict] = []
    per_seed_a_alone: list[dict] = []
    control_reports: list[dict] = []

    # Per-item hit vectors, accumulated across seeds. These are what the paired
    # bootstrap resamples; the seed-level means are reported alongside but are not the
    # primary interval. See eval/aggregate.py.
    primary_surface: Surface = "persistent_store_after_episode"
    hit_vectors: list[list[float]] = []
    # Keyed by routing policy, so `make-report` can recompute the treatment-minus-
    # baseline delta on the routing-free arms of BOTH conditions (ADR-0044). The old
    # confound control asked only whether one arm's absolute recall exceeded zero.
    hit_vectors_by_policy: dict[str, list[list[float]]] = {}

    policy_name = cfg.effective_routing()
    alternate_policy = cfg.alternate_routing()

    # Loaded once for the whole condition: every seed and every control arm below shares
    # these weights. See `build_shared_agents` for why that is sound and what it saves.
    typer.echo(f"  loading {1 if cfg.agent_b is None else 2} agent(s) ...")
    shared_agents = build_shared_agents(cfg, hw, token=token)
    typer.echo(f"  agents     {[a.agent_id for a in shared_agents]}")

    try:
        for s in range(seeds):
            typer.echo(f"  seed {s} ...")
            arm = execute_condition(cfg, items, hw, s, agents_override=shared_agents)
            _write_transcripts(out, arm, "treatment", s)
            evidence.extend(_handoff_evidence(arm, s))
            per_seed.append(_arm_record(cfg, arm, s, policy_name, "treatment"))
            hit_vectors.append(arm.hit_vector(items, primary_surface))
            hit_vectors_by_policy.setdefault(policy_name, []).append(
                arm.hit_vector(items, primary_surface)
            )

            if not controls:
                continue

            ctl_arm = None
            if cfg.agent_b is not None:
                # The OTHER routing. For C3D/C3C the primary is unconditional and this is
                # the ecological abstention-routed variant; for C2/C3 it is the reverse.
                # Both are always produced, so a routing-free delta always exists.
                ctl_arm = execute_condition(
                    cfg,
                    items,
                    hw,
                    s,
                    delegation_override=alternate_policy,
                    agents_override=shared_agents,
                )
                per_seed_control.append(
                    _arm_record(cfg, ctl_arm, s, alternate_policy, alternate_policy)
                )
                _write_transcripts(out, ctl_arm, f"routing_{alternate_policy}", s)
                hit_vectors_by_policy.setdefault(alternate_policy, []).append(
                    ctl_arm.hit_vector(items, primary_surface)
                )

            retain_arm = None
            eco_retain_arm = None
            eco_forget_arm = None
            if retain_items:
                retain_arm = execute_condition(
                    cfg, retain_items, hw, s, agents_override=shared_agents
                )
                per_seed_retain.append(_arm_record(cfg, retain_arm, s, policy_name, "retain"))
                _write_transcripts(out, retain_arm, "retain", s)

                # THE DELEGATION GAP LIVES HERE, NOT ON THE ARMS ABOVE (ADR-0049).
                # C3D/C3S/C3C route unconditionally, so on their primary arms
                # delegation_rate(forget) = delegation_rate(retain) = 1 and the gap is
                # exactly 0 against a pre-registered 0.15 — an automatic FAIL for the
                # very conditions the experiment exists to compare. The gap is a claim
                # about ABSTENTION routing selectively firing on forgetting, so both
                # sides of it are measured under abstention routing.
                if cfg.agent_b is not None:
                    eco_forget_arm = (
                        ctl_arm if alternate_policy == ECOLOGICAL_POLICY else None
                    ) or execute_condition(
                        cfg,
                        items,
                        hw,
                        s,
                        delegation_override=ECOLOGICAL_POLICY,
                        agents_override=shared_agents,
                    )
                    eco_retain_arm = execute_condition(
                        cfg,
                        retain_items,
                        hw,
                        s,
                        delegation_override=ECOLOGICAL_POLICY,
                        agents_override=shared_agents,
                    )
                    per_seed_eco_retain.append(
                        _arm_record(cfg, eco_retain_arm, s, ECOLOGICAL_POLICY, "retain_ecological")
                    )
                    _write_transcripts(out, eco_retain_arm, "retain_ecological", s)

            a_alone_arm = None
            if cfg.agent_b is not None and retain_items:
                # Control 2: how much of the system's advantage on RETAIN questions is
                # plain ensemble benefit? Agent A answering retain questions alone is the
                # comparator; without it "the system beats one agent" is unquantified.
                a_alone_arm = execute_condition(
                    cfg, retain_items, hw, s, agents_override=shared_agents[:1]
                )
                per_seed_a_alone.append(
                    _arm_record(cfg, a_alone_arm, s, "never", "agent_a_alone_retain")
                )
                _write_transcripts(out, a_alone_arm, "agent_a_alone_retain", s)

            control_reports.append(
                compute_controls(
                    # Both sides of the delegation gap under ABSTENTION routing. For a
                    # two-agent arm these are the ecological arms; for a single-agent one
                    # there is no delegation at all and the control is NOT_APPLICABLE.
                    forget_transcripts=(
                        eco_forget_arm.transcripts if eco_forget_arm else arm.transcripts
                    ),
                    retain_transcripts=(
                        eco_retain_arm.transcripts
                        if eco_retain_arm
                        else (retain_arm.transcripts if retain_arm else ())
                    ),
                    delegation_gap_policy=(ECOLOGICAL_POLICY if eco_forget_arm else policy_name),
                    results_primary_routing=arm.containment,
                    results_alternate_routing=ctl_arm.containment if ctl_arm else (),
                    # Correct retain answers = utility, reported and never gated. The
                    # floor is the DERANGED targets: content the system should not be
                    # able to produce at all (ADR-0043).
                    results_retain_utility=retain_arm.containment if retain_arm else (),
                    results_retain_negative=retain_arm.containment_negative if retain_arm else (),
                    agent_only_retain=a_alone_arm.containment if a_alone_arm else (),
                    system_retain=retain_arm.containment if retain_arm else (),
                    surface=primary_surface,
                    k=cfg.episode.max_turns,
                    primary_policy=policy_name,
                    alternate_policy=alternate_policy,
                    # A condition with no agent B cannot have a delegation control, and
                    # "absent" must not be scored as "failed" — see eval/controls.py.
                    is_multi_agent=cfg.agent_b is not None,
                    has_retain_arm=bool(retain_items),
                ).to_dict()
            )
    finally:
        # Even on a crash: a leaked 2.5 GB handle makes the next condition in the grid
        # OOM for a reason that has nothing to do with the next condition.
        for agent in shared_agents:
            agent.close()

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
    def _mean(vectors: list[list[float]]) -> list[float]:
        return np.mean(np.asarray(vectors, dtype=float), axis=0).tolist() if vectors else []

    handoff_audit_by_seed, handoff_audit_aggregate = _aggregate_handoff_audits(per_seed)

    mean_hits = _mean(hit_vectors)
    # `joint_only_recovery` is an AND over three conditions AT THE SAME SEED, so the
    # per-seed vectors have to survive into the report; a mean over seeds cannot express
    # "C3C recovered it and neither standalone arm did, on this run".
    per_item_by_seed = [[float(v) for v in vec] for vec in hit_vectors]

    payload: dict[str, Any] = {
        "run_id": run_id,
        "phase": "phase0_days3-5",
        "condition": cfg.condition,
        "config_hash": chash,
        "config": cfg.model_dump(mode="json"),
        "hardware": hw.to_dict(),
        "git_sha": git_sha(),
        # Provenance of the CODE and of the SOFTWARE, not merely of the settings. Two
        # arms differenced across an SDPA run and an FA2 run, across two transformers
        # versions, or across a moved chat template are two experiments reported as one;
        # `make-report` blocks such a pairing, which it can only do if both reports carry
        # these facts. See ADR-0054.
        "git_dirty": bool(dirty) if dirty is not None else None,
        "git_diff_sha256": git_diff_sha256() if dirty else None,
        "runtime": _runtime_fingerprint(cfg, hw),
        "n_seeds": seeds,
        "required_seeds": cfg.required_seeds(),
        "n_items": len(items),
        "data_provenance": provenance,
        "retain_provenance": retain_provenance,
        "controls_enabled": controls,
        "primary_surface": primary_surface,
        # ---- reportability (ADR-0046) ------------------------------------------------
        # `--limit 20` on real TOFU used to produce `is_real_data: true` and clear every
        # check in `make-report`, so a twenty-item smoke run was indistinguishable from a
        # result. Scale is now recorded as a fact about the run and gated downstream.
        "truncated": limit is not None,
        "limit": limit,
        "reportable": (limit is None and provenance["is_real_data"] and controls and not dirty),
        "n_retain_items": len(retain_items),
        "routing_policy": policy_name,
        "alternate_routing_policy": alternate_policy,
        "store_scope": cfg.episode.store_scope,
        "handoff_configured": cfg.episode.pass_primary_answer,
        "handoff_source": cfg.episode.handoff_source,
        "n_handoffs_total": sum(r["n_handoffs"] for r in per_seed),
        "n_delegations_total": sum(r["n_delegations"] for r in per_seed),
        "n_shuffled_handoffs_total": sum(r["n_shuffled_handoffs"] for r in per_seed),
        # C3S's mapping and its leak check, hoisted to the top level so `make-report` can
        # gate on them without walking per_seed (ADR-0055) — but hoisted from EVERY seed,
        # not from seed 0. The mapping is seed-independent; the generated source text is
        # not, so a leak can appear at one seed and not another (ADR-0057).
        "handoff_audit": (per_seed[0]["handoff_audit"] if per_seed else {}),
        "handoff_audit_by_seed": handoff_audit_by_seed,
        "handoff_audit_aggregate": handoff_audit_aggregate,
        "recall_at_k": aggregated,
        "laundering_rate": laundering_stats,
        "delegation_rate": delegation_stats,
        # Paired-bootstrap inputs. `item_ids` and `clusters` are positional companions
        # of `per_item_recall`; make-report zips them.
        "item_ids": [it.item_id for it in items],
        "clusters": clusters,
        "cluster_by": cfg.data.cluster_by,
        "per_item_recall": mean_hits,
        "per_item_recall_by_seed": per_item_by_seed,
        # Keyed by routing policy so the confound gate can difference two conditions on
        # their routing-free arms without a rerun (ADR-0044).
        "per_item_recall_by_policy": {
            policy: _mean(vectors) for policy, vectors in sorted(hit_vectors_by_policy.items())
        },
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
        # The arm the delegation gap is actually computed from (ADR-0049). Kept separate
        # from `control_retain`, which runs at the condition's own routing.
        "control_retain_ecological": {
            "policy": ECOLOGICAL_POLICY,
            "per_seed": per_seed_eco_retain,
        },
        "control_agent_a_alone_retain": {"per_seed": per_seed_a_alone},
        "control_reports": control_reports,
    }
    (out / "condition_report.json").write_text(json.dumps(payload, indent=2, default=str))
    if evidence:
        # Only the handoff arms produce this. Written even when the grid later fails its
        # gate: the evidence is what a reviewer re-reads to find out WHY (ADR-0060).
        (out / "handoff_evidence.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "condition": cfg.condition,
                    "handoff_source": cfg.episode.handoff_source,
                    "n_records": len(evidence),
                    "records": evidence,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

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

    n_handoffs = payload["n_handoffs_total"]
    typer.echo(
        f"handoffs recorded             = {n_handoffs} (configured: {cfg.episode.pass_primary_answer})"
    )
    if cfg.episode.pass_primary_answer and n_handoffs == 0:
        typer.secho(
            "  this condition declares a compositional handoff and performed NONE. "
            "That is the ADR-0041 failure mode: agent B was never shown agent A's "
            "output, so this arm is an ensemble under a compositional name.",
            fg=typer.colors.RED,
        )
    if handoff_audit_aggregate:
        # The four numbers `make-report` gates on, unioned over every seed, printed while
        # the operator is still at the box. See ADR-0055 and ADR-0057.
        agg = handoff_audit_aggregate
        typer.echo(
            f"C3S audit ({agg['n_seeds']} seed(s))       = "
            f"same_author={agg['same_author_count']} "
            f"fixed_points={agg['fixed_point_count']} "
            f"target_leaks={agg['target_answer_in_handoff_count']} "
            f"mapping_hashes={len(agg['mapping_hashes'])}"
        )
        if (
            agg["same_author_count"]
            or agg["fixed_point_count"]
            or agg["target_answer_in_handoff_count"]
            or len(agg["mapping_hashes"]) != 1
        ):
            typer.secho(
                "  this arm is NOT a negative control. A same-author pair, a fixed point, "
                "a leaked target answer at ANY seed, or a seed-dependent mapping each "
                "bias C3C - C3S toward zero. `make-report` will block it (ADR-0057).",
                fg=typer.colors.RED,
            )
    if limit is not None:
        typer.secho(
            f"\n--limit {limit} was used: this run is marked `truncated` and "
            "`reportable: false`. It is an engineering pilot and `make-report` will "
            "exclude it (ADR-0046).",
            fg=typer.colors.YELLOW,
        )
    if not controls:
        typer.secho(
            "\n--no-controls was passed. This run cannot be reported: the confound "
            "gate (docs/00b_preregistration_v2.md §4.2) is unevaluated.",
            fg=typer.colors.RED,
        )
    typer.echo(f"\nwrote {out}")
