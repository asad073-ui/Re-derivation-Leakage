"""Sampled, fresh-trajectory Leak@k runner.

Historical ``run-condition`` remains frozen for the v5 conditions.  This command is a
new experiment family: its sample is a full reset-and-run trajectory, never a second
decode appended to a persistent episode.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from ..agents.delegation import build_delegation_policy
from ..agents.writer import build_write_policy
from ..config import ConfigError, RDLConfig, config_hash, load_config
from ..eval.laundering import laundered_items
from ..eval.leak_at_k import decoding_hash, seed_for
from ..eval.negatives import cross_author_mapping
from ..eval.semantic import OfflineSemanticScorer
from ..eval.tofu_data import TofuItem, cluster_ids, load_items
from ..hardware import assert_env_matches_hardware, detect
from ..memory.blocklist import build_blocklist
from ..models.stub import GenerationRequest
from ..orchestrator.loop import EpisodePolicies, run_episode
from ..paths import git_sha, make_run_id, run_dir
from .run_condition import build_shared_agents, seed_store

__all__ = ["run_leak"]


ARMS = ("D-A", "D-B", "W-A", "W-B", "C3S", "C3C", "C3C-guard")


def _prompt_hash(item: TofuItem) -> str:
    return hashlib.sha256(item.question.encode()).hexdigest()


def _record(
    *,
    cfg: RDLConfig,
    item: TofuItem,
    sample_id: int,
    arm: str,
    request: GenerationRequest,
    scorer: OfflineSemanticScorer,
    transcript=None,
    direct_text: str | None = None,
    certified: bool = False,
    readback_text: str = "",
    guard_diagnostic: bool = False,
) -> dict:
    text = direct_text if direct_text is not None else transcript.final_text
    final = scorer.score(item.answer, text)
    agent_text = "\n".join(transcript.agent_texts()) if transcript else text
    message = scorer.score(item.answer, agent_text)
    write_text = "\n".join(transcript.memory_write_texts()) if transcript else ""
    write = scorer.score(item.answer, write_text)
    return {
        "item_id": item.item_id,
        "author_id": item.author_id,
        "sample_id": sample_id,
        "arm": arm,
        "checkpoint_fingerprint": "|".join(
            f"{name}:{model.name}@{model.revision or 'unresolved'}"
            for name, model in sorted(cfg.models.items())
        ),
        "prompt_sha256": _prompt_hash(item),
        "decoding_sha256": decoding_hash(request.to_dict()),
        "generation_seed": request.seed,
        "scorer_version": scorer.version,
        "guard_diagnostic": guard_diagnostic,
        "direct_leak": final.leaks if arm.startswith("D-") else False,
        "agent_message_leak": message.leaks,
        "final_leak": final.leaks,
        "store_leak": write.leaks,
        "certified_store_leak": certified,
        "readback_leak": scorer.score(item.answer, readback_text).leaks if readback_text else False,
        "final_score": final.score,
        "write_attempts": (
            [e.model_dump(mode="json") for e in transcript.write_attempts()] if transcript else []
        ),
        "trajectory": transcript.to_dict() if transcript else None,
    }


def _policies(cfg: RDLConfig, blocklist, *, guard: bool = False) -> EpisodePolicies:
    return EpisodePolicies(
        delegation=build_delegation_policy(
            "always_delegate", cfg.agent_a.delegation.max_delegations
        ),
        write=build_write_policy(
            "sanitized" if guard else "framework_default",
            source_kinds=tuple(cfg.writepolicy.write_source_kinds),
            write_on_abstention=cfg.writepolicy.write_on_abstention,
            threshold=cfg.memory.semantic_threshold,
        ),
        blocklist=blocklist,
        retrieval_k=cfg.episode.retrieval_k,
        max_turns=cfg.episode.max_turns,
        pass_primary_answer_to_secondary=True,
    )


def run_leak(
    condition: Path = typer.Option(
        ..., "--condition", help="C3C-family condition used for A/B checkpoints"
    ),
    environment: str | None = typer.Option(None, "--env"),
    fixture: Path | None = typer.Option(
        None, "--fixture", help="offline fixture; development only"
    ),
    n_samples: int | None = typer.Option(None, "--n-samples", min=1),
    limit: int | None = typer.Option(None, "--limit", min=1),
    output: Path | None = typer.Option(None, "--output", help="new output directory"),
    resume: bool = typer.Option(False, "--resume", help="continue a compatible interrupted output"),
    token: str | None = typer.Option(None, "--hf-token"),
) -> None:
    """Run D-A/D-B, wrappers, C3S/C3C and a deterministic guard replay."""
    try:
        cfg = load_config(condition, env_override=environment)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if cfg.agent_b is None:
        raise typer.BadParameter("run-leak needs a two-agent C3C-family condition")
    n = n_samples or cfg.sampling.n_samples
    if n > cfg.sampling.n_samples:
        raise typer.BadParameter(
            "--n-samples may reduce, not exceed, the configured sampling budget"
        )
    hw = detect()
    assert_env_matches_hardware(cfg.env, hw)
    items, provenance = load_items(
        dataset="stub" if fixture else cfg.data.dataset,
        split=cfg.data.forget_split,
        n_items=limit or cfg.data.n_items,
        fixture=fixture,
        allow_fixture=bool(fixture),
        token=token,
        sample="spread" if limit else "head",
    )
    if len(items) < 2:
        raise typer.BadParameter("C3S needs at least two cross-author items")
    mapping = cross_author_mapping(cluster_ids(items, "author"))
    out = output or run_dir(make_run_id(config_hash(cfg)))
    if output:
        if resume:
            if not out.is_dir():
                raise typer.BadParameter("--resume requires an existing output directory")
        else:
            out.mkdir(parents=True, exist_ok=False)
    records_path = out / "leak_records.jsonl"
    manifest_path = out / "leak_manifest.json"
    existing_keys: set[tuple[str, int, str]] = set()
    if resume:
        if not records_path.exists() or not manifest_path.exists():
            raise typer.BadParameter("--resume requires leak_records.jsonl and leak_manifest.json")
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("config_hash") != config_hash(cfg):
            raise typer.BadParameter("refusing resume: config hash differs from existing output")
        for line in records_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (str(row["item_id"]), int(row["sample_id"]), str(row["arm"]))
            if key in existing_keys:
                raise typer.BadParameter(f"refusing resume: duplicate existing sample id {key}")
            existing_keys.add(key)
    scorer = OfflineSemanticScorer()
    agents = build_shared_agents(cfg, hw, token=token)
    source_for = {it.item_id: items[mapping.permutation[i]] for i, it in enumerate(items)}
    with records_path.open("a" if resume else "x", encoding="utf-8", newline="\n") as fh:

        def write_row(row: dict) -> None:
            key = (str(row["item_id"]), int(row["sample_id"]), str(row["arm"]))
            if key not in existing_keys:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                existing_keys.add(key)

        for item in items:
            for sample_id in range(n):
                if all((item.item_id, sample_id, arm) in existing_keys for arm in ARMS):
                    continue

                def req(
                    agent: str, arm: str, source: TofuItem = item, this_sample: int = sample_id
                ) -> GenerationRequest:
                    return GenerationRequest(
                        do_sample=True,
                        temperature=cfg.sampling.temperature,
                        top_p=cfg.sampling.top_p,
                        top_k=cfg.sampling.top_k,
                        seed=seed_for(
                            cfg.sampling.base_seed,
                            item_id=source.item_id,
                            sample_id=this_sample,
                            agent_id=agent,
                            arm="shared",
                        ),
                    )

                # Bare direct answers have no wrapper, retrieval or write path.
                for label, agent in (("D-A", agents[0]), ("D-B", agents[1])):
                    answer = agent.answer(
                        item.question, generation_request=req(agent.agent_id, label)
                    )
                    row = _record(
                        cfg=cfg,
                        item=item,
                        sample_id=sample_id,
                        arm=label,
                        request=req(agent.agent_id, label),
                        scorer=scorer,
                        direct_text=answer.text,
                    )
                    write_row(row)

                for label, active in (("W-A", [agents[0]]), ("W-B", [agents[1]])):
                    store, blocklist, _ = seed_store(items, cfg)
                    policies = _policies(cfg, blocklist)
                    active_id = active[0].agent_id
                    tr = run_episode(
                        item.question,
                        active,
                        store,
                        policies,
                        item_id=item.item_id,
                        condition=label,
                        sample_id=str(sample_id),
                        trajectory_id=f"{label}:{item.item_id}:{sample_id}",
                        generation_requests={active_id: req(active_id, label)},
                        prompt_sha256=_prompt_hash(item),
                        decoding_sha256=decoding_hash(req(active_id, label).to_dict()),
                    )
                    cert = (
                        laundered_items(
                            [tr],
                            store,
                            store.dag,
                            blocklist,
                            [{"item_id": item.item_id, "answer": item.answer}],
                        ).n_laundered
                        > 0
                    )
                    rb = (
                        active[0]
                        .answer(
                            item.question,
                            store.retrieve(
                                item.question, k=cfg.episode.retrieval_k, blocklist=blocklist
                            ).nodes,
                            generation_request=req(active_id, label),
                        )
                        .text
                    )
                    write_row(
                        _record(
                            cfg=cfg,
                            item=item,
                            sample_id=sample_id,
                            arm=label,
                            request=req(active_id, label),
                            scorer=scorer,
                            transcript=tr,
                            certified=cert,
                            readback_text=rb,
                        )
                    )

                # C3S gets a banked A response for a different item. C3C receives A's
                # in-trajectory response; both use the same per-item A seed.
                source = source_for[item.item_id]
                source_store, source_bl, _ = seed_store(items, cfg)
                source_answer = agents[0].answer(
                    source.question,
                    source_store.retrieve(
                        source.question, k=cfg.episode.retrieval_k, blocklist=source_bl
                    ).nodes,
                    generation_request=req(agents[0].agent_id, "C3S", source),
                )
                for label, peer_text, source_item, guard in (
                    ("C3S", source_answer.text, source.item_id, False),
                    ("C3C", None, None, False),
                    ("C3C-guard", None, None, True),
                ):
                    store, blocklist, _ = seed_store(items, cfg)
                    if guard:
                        blocklist = build_blocklist(
                            "semantic",
                            texts=[f"{x.question} {x.answer}" for x in items],
                            threshold=cfg.memory.semantic_threshold,
                        )
                    policies = _policies(cfg, blocklist, guard=guard)
                    requests = {
                        agents[0].agent_id: req(agents[0].agent_id, label),
                        agents[1].agent_id: req(agents[1].agent_id, label),
                    }
                    tr = run_episode(
                        item.question,
                        agents,
                        store,
                        policies,
                        item_id=item.item_id,
                        condition=label,
                        sample_id=str(sample_id),
                        trajectory_id=f"{label}:{item.item_id}:{sample_id}",
                        peer_answer_override=peer_text,
                        peer_answer_source_item=source_item,
                        peer_answer_abstained=(
                            source_answer.abstained if peer_text is not None else None
                        ),
                        generation_requests=requests,
                        prompt_sha256=_prompt_hash(item),
                        decoding_sha256=decoding_hash(requests[agents[0].agent_id].to_dict()),
                    )
                    cert = (
                        laundered_items(
                            [tr],
                            store,
                            store.dag,
                            blocklist,
                            [{"item_id": item.item_id, "answer": item.answer}],
                        ).n_laundered
                        > 0
                    )
                    rb = (
                        agents[1]
                        .answer(
                            item.question,
                            store.retrieve(
                                item.question, k=cfg.episode.retrieval_k, blocklist=blocklist
                            ).nodes,
                            generation_request=requests[agents[1].agent_id],
                        )
                        .text
                    )
                    write_row(
                        _record(
                            cfg=cfg,
                            item=item,
                            sample_id=sample_id,
                            arm=label,
                            request=requests[agents[0].agent_id],
                            scorer=scorer,
                            transcript=tr,
                            certified=cert,
                            readback_text=rb,
                            guard_diagnostic=guard,
                        )
                    )
    for agent in agents:
        agent.close()
    manifest_path.write_text(
        json.dumps(
            {
                "schema": 1,
                "config_hash": config_hash(cfg),
                "git_sha": git_sha(),
                "n_samples": n,
                "arms": ARMS,
                "data": provenance,
                "scorer": scorer.version,
                "guard_note": "C3C-guard is deterministic CPU plumbing only; pin and calibrate a real NLI verifier before scientific guard claims.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    typer.echo(f"wrote {records_path}")
