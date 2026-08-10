"""Sampled, fresh-trajectory Leak@k runner.

Historical ``run-condition`` remains frozen for the v5 conditions.  This command is a
new experiment family: its sample is a full reset-and-run trajectory, never a second
decode appended to a persistent episode.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
from copy import deepcopy
from pathlib import Path

import typer

from ..agents.base import AgentReply
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
from ..memory.invariants import certify
from ..models.stub import GenerationRequest
from ..orchestrator.events import MemoryWrite, WriteAttempt
from ..orchestrator.loop import EpisodePolicies, run_episode
from ..paths import git_diff_sha256, git_dirty, git_sha, make_run_id, run_dir
from .run_condition import build_shared_agents, seed_store

__all__ = ["run_leak"]


ARMS = ("D-A", "D-B", "W-A", "W-B", "C3S", "C3C", "C3C-guard")


def _atomic_json(path: Path, payload: dict) -> None:
    """Replace a manifest atomically, so a killed process cannot corrupt its contract."""
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)


def _leakk_direct_reply(agent, question: str, request: GenerationRequest) -> AgentReply:
    """Use the released direct-evaluator path, never the composition chat wrapper."""
    generate = getattr(agent.lm, "generate_leakk_official", None)
    if not callable(generate):
        raise typer.BadParameter(
            "leakk_official requires a real HFLMHandle; the fixture/stub path is not decoding-compatible"
        )
    text = generate(question, agent.max_new_tokens, seed=request.seed)
    decision = agent.detector.detect(text)
    provenance = agent.lm.generation_provenance()
    return AgentReply(
        agent_id=agent.agent_id,
        text=text,
        abstained=decision.abstained,
        meta={
            "detector": decision.detector,
            "detector_reason": decision.reason,
            "semantic_user_prompt_sha256": provenance.get("semantic_user_prompt_sha256"),
            "serialized_chat_prompt_sha256": provenance.get("serialized_chat_prompt_sha256"),
            "input_ids_sha256": provenance.get("input_ids_sha256"),
            "rendered_prompt_sha256": provenance.get("serialized_chat_prompt_sha256"),
            "generation_request": request.to_dict(),
            "leakk_official": provenance.get("leakk_official"),
        },
    )


def _repair_partial_jsonl(path: Path) -> list[dict]:
    """Keep a valid prefix only when the final line was torn by an interruption."""
    raw = path.read_bytes()
    if not raw:
        return []
    lines = raw.splitlines(keepends=True)
    valid: list[dict] = []
    for index, line in enumerate(lines):
        try:
            valid.append(json.loads(line))
        except json.JSONDecodeError as exc:
            if index != len(lines) - 1:
                raise typer.BadParameter(
                    f"refusing resume: malformed non-final JSONL line: {exc}"
                ) from exc
            cut = sum(len(x) for x in lines[:index])
            with path.open("r+b") as fh:
                fh.truncate(cut)
                fh.flush()
                os.fsync(fh.fileno())
            break
    return valid


def _record(
    *,
    cfg: RDLConfig,
    item: TofuItem,
    sample_id: int,
    arm: str,
    request: GenerationRequest,
    scorer: OfflineSemanticScorer,
    transcript=None,
    store=None,
    blocklist=None,
    direct_reply: AgentReply | None = None,
    certified: dict | None = None,
    probe: dict | None = None,
    source_evidence: dict | None = None,
    guard_diagnostic: bool = False,
) -> dict:
    text = direct_reply.text if direct_reply is not None else transcript.final_text
    final = scorer.score(item.answer, text)
    agent_text = "\n".join(transcript.agent_texts()) if transcript else text
    message = scorer.score(item.answer, agent_text)
    write_text = "\n".join(transcript.memory_write_texts()) if transcript else ""
    write = scorer.score(item.answer, write_text)
    reply_agent_id = direct_reply.agent_id if direct_reply is not None else None
    configured_agents = [(cfg.agent_a, cfg.models[cfg.agent_a.model])]
    if cfg.agent_b is not None:
        configured_agents.append((cfg.agent_b, cfg.models[cfg.agent_b.model]))
    direct_model = next(
        (
            f"{model.repo_id}@{model.revision or 'unresolved'}"
            for agent, model in configured_agents
            if agent.agent_id == reply_agent_id
        ),
        None,
    )
    return {
        "item_id": item.item_id,
        "author_id": item.author_id,
        # Raw target retained only in the external evidence bundle: it permits a
        # pinned evaluator to rescore completed generations without re-generation.
        "reference_answer": item.answer,
        "sample_id": sample_id,
        "arm": arm,
        "checkpoint_fingerprint": "|".join(
            f"{name}:{model.name}@{model.revision or 'unresolved'}"
            for name, model in sorted(cfg.models.items())
        ),
        "generation_provenance": (
            [
                {
                    "agent_id": direct_reply.agent_id,
                    "model_revision": direct_model,
                    "generation_seed": request.seed,
                    "semantic_user_prompt_sha256": direct_reply.meta.get(
                        "semantic_user_prompt_sha256"
                    ),
                    "serialized_chat_prompt_sha256": direct_reply.meta.get(
                        "serialized_chat_prompt_sha256"
                    ),
                    "input_ids_sha256": direct_reply.meta.get("input_ids_sha256"),
                    "leakk_official": direct_reply.meta.get("leakk_official"),
                }
            ]
            if direct_reply is not None
            else [
                {
                    "agent_id": e.agent_id,
                    "model_revision": e.model_revision,
                    "generation_seed": e.generation_seed,
                    "semantic_user_prompt_sha256": e.semantic_user_prompt_sha256,
                    "serialized_chat_prompt_sha256": e.serialized_chat_prompt_sha256,
                    "input_ids_sha256": e.input_ids_sha256,
                }
                for e in transcript.agent_answers()
            ]
        ),
        "rendered_prompt_sha256s": (
            [direct_reply.meta["rendered_prompt_sha256"]]
            if direct_reply is not None
            else sorted(
                {
                    str(e.prompt_sha256)
                    for e in transcript.agent_answers()
                    if e.prompt_sha256 is not None
                }
            )
        ),
        "prompt_provenance": (
            {
                "semantic_user_prompt_sha256": direct_reply.meta.get("semantic_user_prompt_sha256"),
                "serialized_chat_prompt_sha256": direct_reply.meta.get(
                    "serialized_chat_prompt_sha256"
                ),
                "input_ids_sha256": direct_reply.meta.get("input_ids_sha256"),
            }
            if direct_reply is not None
            else [
                {
                    "agent_id": e.agent_id,
                    "semantic_user_prompt_sha256": e.semantic_user_prompt_sha256,
                    "serialized_chat_prompt_sha256": e.serialized_chat_prompt_sha256,
                    "input_ids_sha256": e.input_ids_sha256,
                }
                for e in transcript.agent_answers()
            ]
        ),
        "decoding_sha256": decoding_hash(request.to_dict()),
        "generation_seed": request.seed,
        "scorer_version": scorer.version,
        "guard_diagnostic": guard_diagnostic,
        "direct_leak": final.leaks if arm.startswith("D-") else False,
        "agent_message_leak": message.leaks,
        "final_leak": final.leaks,
        "store_leak": write.leaks,
        "certified_store_leak": certified is not None,
        # A Boolean is not auditable evidence.  The selected node and its complete
        # invariant certificate are retained for every certified hit, and the full
        # per-write evidence below permits a later semantic-rescore to select a
        # different leaking node correctly.
        "certified_evidence": certified,
        "post_episode_probe_leak": bool((probe or {}).get("with_store_leak", False)),
        "attributable_readback_leak": bool((probe or {}).get("attributable", False)),
        "final_score": final.score,
        "write_attempts": (
            [e.model_dump(mode="json") for e in transcript.write_attempts()] if transcript else []
        ),
        "raw_outputs": {
            "direct": direct_reply.text if direct_reply is not None else None,
            "final": text if transcript is not None else None,
            "agent_messages": transcript.agent_texts() if transcript else [text],
            "memory_writes": transcript.memory_write_texts() if transcript else [],
            "probe": probe or {},
        },
        "memory_node_evidence": (
            _memory_node_evidence(transcript, store, blocklist, item, scorer)
            if transcript is not None and store is not None and blocklist is not None
            else []
        ),
        "source_evidence": source_evidence,
        "trajectory": transcript.to_dict() if transcript else None,
    }


def _memory_node_evidence(transcript, store, blocklist, item: TofuItem, scorer) -> list[dict]:
    """Persist every written node with the certificate needed for a future rescore."""
    evidence: list[dict] = []
    for event in transcript.memory_writes():
        node = store.get(event.node_id)
        if node is None:  # A transcript/store mismatch is evidence, never a silent miss.
            raise RuntimeError(f"memory write {event.node_id} is absent from its episode store")
        verdict = scorer.score(item.answer, node.content)
        certificate = certify(store, store.dag, blocklist, node.node_id)
        evidence.append(
            {
                "node_id": node.node_id,
                "content": node.content,
                "content_sha256": hashlib.sha256(node.content.encode("utf-8")).hexdigest(),
                "parent_ids": list(node.parent_ids),
                "semantic_verdict": {
                    "label": verdict.label,
                    "score": verdict.score,
                    "version": verdict.version,
                },
                "inv1_satisfied": certificate.inv1_satisfied,
                "inv2_satisfied": certificate.inv2_satisfied,
                "path_to_any_blocked_node": certificate.path_to_any_blocked_node,
                "certificate": certificate.to_dict(),
            }
        )
    return evidence


def _certified(transcript, store, blocklist, item: TofuItem, scorer) -> dict | None:
    """Return the complete witness for the semantically leaking clean node, if any."""
    report = laundered_items(
        [transcript],
        store,
        store.dag,
        blocklist,
        [{"item_id": item.item_id, "answer": item.answer}],
        mode="entailment",
        nli_fn=lambda reference, candidate: scorer.score(reference, candidate).leaks,
    )
    hits = report.laundered_only()
    if not hits:
        return None
    hit = hits[0]
    node = store.get(hit.node_id)
    if node is None:  # pragma: no cover - laundered_items obtained it from this store
        raise RuntimeError(f"certified node {hit.node_id} is absent from its episode store")
    verdict = scorer.score(item.answer, node.content)
    return {
        "node_id": node.node_id,
        "content_sha256": hashlib.sha256(node.content.encode("utf-8")).hexdigest(),
        "parent_ids": list(node.parent_ids),
        "semantic_verdict": {
            "label": verdict.label,
            "score": verdict.score,
            "version": verdict.version,
        },
        "inv1_satisfied": hit.certificate.inv1_satisfied,
        "inv2_satisfied": hit.certificate.inv2_satisfied,
        "path_to_any_blocked_node": hit.certificate.path_to_any_blocked_node,
        "certificate": hit.certificate.to_dict(),
    }


def _probe(
    agent,
    item: TofuItem,
    store,
    blocklist,
    request: GenerationRequest,
    scorer: OfflineSemanticScorer,
    retrieval_k: int,
) -> dict:
    """Separate post-episode probing from attributable memory readback."""
    retrieved = store.retrieve(item.question, k=retrieval_k, blocklist=blocklist)
    with_store_reply = agent.answer(item.question, retrieved.nodes, generation_request=request)
    no_store_reply = agent.answer(item.question, (), generation_request=request)
    with_store = with_store_reply.text
    no_store = no_store_reply.text
    leaking_node_ids = [
        n.node_id for n in store.all_nodes() if scorer.score(item.answer, n.content).leaks
    ]
    with_leak = scorer.score(item.answer, with_store).leaks
    no_leak = scorer.score(item.answer, no_store).leaks
    relevant_retrieved = bool(set(retrieved.node_ids) & set(leaking_node_ids))
    return {
        "with_store_text": with_store,
        "without_store_text": no_store,
        "with_store_prompt_provenance": {
            key: with_store_reply.meta.get(key)
            for key in (
                "semantic_user_prompt_sha256",
                "serialized_chat_prompt_sha256",
                "input_ids_sha256",
            )
        },
        "without_store_prompt_provenance": {
            key: no_store_reply.meta.get(key)
            for key in (
                "semantic_user_prompt_sha256",
                "serialized_chat_prompt_sha256",
                "input_ids_sha256",
            )
        },
        "with_store_leak": with_leak,
        "without_store_leak": no_leak,
        "retrieved_node_ids": retrieved.node_ids,
        "relevant_leaking_node_ids": leaking_node_ids,
        "attributable": with_leak and relevant_retrieved and not no_leak,
    }


def _guard_replay(c3c, store, blocklist, item: TofuItem, policies: EpisodePolicies):
    """Replay the exact C3C candidate through only the guarded write policy."""
    replay = deepcopy(c3c)
    replay.condition = "C3C-guard"
    replay.episode_id = f"C3C-guard:{item.item_id}:{c3c.sample_id}"
    replay.events = [e for e in replay.events if e.kind not in ("memory_write", "write_attempt")]
    reply_event = next(e for e in reversed(replay.agent_answers()) if e.text == replay.final_text)
    reply = AgentReply(
        agent_id=reply_event.agent_id,
        text=reply_event.text,
        abstained=reply_event.abstained,
        context_node_ids=list(reply_event.context_node_ids),
    )
    store.turn = replay.max_turn
    decision = policies.write.maybe_write(
        store,
        reply,
        question=item.question,
        retrieved_ids=reply.context_node_ids,
        turn=replay.max_turn,
        blocklist=blocklist,
    )
    replay.append(
        WriteAttempt(
            turn=replay.max_turn,
            episode_id=replay.episode_id,
            sample_id=c3c.sample_id,
            trajectory_id=replay.episode_id,
            allowed=decision.write,
            reason=decision.reason,
            policy=decision.policy,
            score=decision.blocked_score,
            matched_reference=decision.matched_reference,
            guard_version=decision.guard_version,
        )
    )
    if decision.write and decision.node is not None:
        replay.append(
            MemoryWrite(
                turn=replay.max_turn,
                episode_id=replay.episode_id,
                sample_id=c3c.sample_id,
                trajectory_id=replay.episode_id,
                node_id=decision.node.node_id,
                content=decision.node.content,
                parent_ids=list(decision.node.parent_ids),
                source_agent=decision.node.source_agent,
                source_kind=decision.node.source_kind,
                policy=decision.policy,
            )
        )
    return replay


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
    raw_evidence_uri: str | None = typer.Option(
        None,
        "--raw-evidence-uri",
        help="immutable external archive URI for leak_records.jsonl",
    ),
    token: str | None = typer.Option(None, "--hf-token"),
    protocol: str | None = typer.Option(
        None, "--protocol", help="leakk_official or rdl_composition"
    ),
) -> None:
    """Run D-A/D-B, wrappers, C3S/C3C and a deterministic guard replay."""
    try:
        cfg = load_config(condition, env_override=environment)
    except ConfigError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if cfg.agent_b is None and (protocol or cfg.sampling.protocol) != "leakk_official":
        raise typer.BadParameter("run-leak needs a two-agent C3C-family condition")
    active_protocol = protocol or cfg.sampling.protocol
    if active_protocol not in ("leakk_official", "rdl_composition"):
        raise typer.BadParameter("--protocol must be leakk_official or rdl_composition")
    arms: tuple[str, ...]
    if active_protocol == "leakk_official":
        arms = ("D-A", "D-B") if cfg.agent_b is not None else ("D-A",)
    else:
        arms = ARMS
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
    if active_protocol == "rdl_composition" and len(items) < 2:
        raise typer.BadParameter("C3S needs at least two cross-author items")
    mapping = (
        cross_author_mapping(cluster_ids(items, "author"))
        if active_protocol == "rdl_composition"
        else None
    )
    out = output or run_dir(make_run_id(config_hash(cfg)))
    if output:
        if resume:
            if not out.is_dir():
                raise typer.BadParameter("--resume requires an existing output directory")
        else:
            out.mkdir(parents=True, exist_ok=False)
    records_path = out / "leak_records.jsonl"
    manifest_path = out / "leak_manifest.json"
    cohort = [
        {
            "item_id": item.item_id,
            "author_id": item.author_id,
            "question_sha256": hashlib.sha256(item.question.encode("utf-8")).hexdigest(),
            "answer_sha256": hashlib.sha256(item.answer.encode("utf-8")).hexdigest(),
        }
        for item in items
    ]
    source_map = (
        {item.item_id: items[mapping.permutation[i]].item_id for i, item in enumerate(items)}
        if mapping is not None
        else {}
    )
    run_manifest = {
        "schema": 3,
        "config_hash": config_hash(cfg),
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "n_samples": n,
        "arms": arms,
        "protocol": active_protocol,
        "protocol_provenance": (
            {
                "upstream_repo": "OptimAI-Lab/Leak-k",
                "upstream_commit": "e544af6017a59da9961aac691b81de337bc21fc5",
                "generation": {
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": None,
                    "max_new_tokens": 200,
                    "n_samples": 200,
                },
                # The upstream evaluator batches compatible draws (32); this runner
                # retains per-draw provenance, so it is not a bitwise RNG replay.
                # Prompt round-trip, tokens, kwargs and output cleaning follow its
                # active direct path in HFLMHandle.generate_leakk_official.
                "batching": {"upstream_batch_size": 32, "runner_batch_size": 1},
                "scope": "direct checkpoint draws only",
            }
            if active_protocol == "leakk_official"
            else {"scope": "controlled sequential multi-agent composition"}
        ),
        "data": provenance,
        "cohort": cohort,
        "source_map": source_map,
        "semantic_scorer": "offline-token-f1-v1",
        "semantic_scorer_reportable": False,
        "raw_evidence_uri": raw_evidence_uri,
        "runtime": {
            package: importlib.metadata.version(package)
            for package in ("torch", "transformers", "datasets", "accelerate")
            if importlib.metadata.packages_distributions().get(package)
        },
        "git_dirty": git_dirty(),
        "git_diff_sha256": git_diff_sha256(),
        "resolved_config": cfg.model_dump(mode="json"),
        "planned_record_count": len(items) * n * len(arms),
        "planned_model_generations": len(items)
        * n
        * (len(arms) if active_protocol == "leakk_official" else 19),
        "diagnostic": True,
        "complete": False,
    }
    existing_keys: set[tuple[str, int, str]] = set()
    if resume:
        if not records_path.exists() or not manifest_path.exists():
            raise typer.BadParameter("--resume requires leak_records.jsonl and leak_manifest.json")
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for manifest_key in (
            "schema",
            "config_hash",
            "git_sha",
            "n_samples",
            "arms",
            "cohort",
            "source_map",
            "semantic_scorer",
            "raw_evidence_uri",
            "runtime",
            "protocol",
            "protocol_provenance",
            "resolved_config",
        ):
            if old_manifest.get(manifest_key) != run_manifest.get(manifest_key):
                raise typer.BadParameter(
                    f"refusing resume: {manifest_key} differs from existing output"
                )
        for row in _repair_partial_jsonl(records_path):
            key = (str(row["item_id"]), int(row["sample_id"]), str(row["arm"]))
            if key in existing_keys:
                raise typer.BadParameter(f"refusing resume: duplicate existing sample id {key}")
            existing_keys.add(key)
    else:
        # The manifest is the durable run contract. It exists before the first model
        # call so an interruption has enough information to resume safely.
        _atomic_json(manifest_path, run_manifest)
    scorer = OfflineSemanticScorer()
    agents = build_shared_agents(cfg, hw, token=token)
    if active_protocol == "leakk_official":
        for agent in agents:
            agent.max_new_tokens = 200
    model_revisions = {
        cfg.agent_a.agent_id: f"{cfg.models[cfg.agent_a.model].repo_id}@{cfg.models[cfg.agent_a.model].revision}",
        **(
            {
                cfg.agent_b.agent_id: f"{cfg.models[cfg.agent_b.model].repo_id}@{cfg.models[cfg.agent_b.model].revision}"
            }
            if cfg.agent_b is not None
            else {}
        ),
    }
    source_for = (
        {it.item_id: items[mapping.permutation[i]] for i, it in enumerate(items)}
        if mapping is not None
        else {}
    )
    with records_path.open("a" if resume else "x", encoding="utf-8", newline="\n") as fh:

        def write_row(row: dict) -> None:
            key = (str(row["item_id"]), int(row["sample_id"]), str(row["arm"]))
            if key not in existing_keys:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                existing_keys.add(key)

        for item in items:
            for sample_id in range(n):
                if all((item.item_id, sample_id, arm) in existing_keys for arm in arms):
                    continue

                def req(
                    agent: str, arm: str, source: TofuItem = item, this_sample: int = sample_id
                ) -> GenerationRequest:
                    return GenerationRequest(
                        do_sample=True,
                        temperature=(
                            1.0 if active_protocol == "leakk_official" else cfg.sampling.temperature
                        ),
                        top_p=1.0 if active_protocol == "leakk_official" else cfg.sampling.top_p,
                        top_k=None if active_protocol == "leakk_official" else cfg.sampling.top_k,
                        seed=seed_for(
                            cfg.sampling.base_seed,
                            item_id=source.item_id,
                            sample_id=this_sample,
                            agent_id=agent,
                            arm="shared",
                        ),
                    )

                # Bare direct answers have no wrapper, retrieval or write path.
                direct_arms = arms if active_protocol == "leakk_official" else ("D-A", "D-B")
                for label, agent in zip(direct_arms, agents, strict=True):
                    request = req(agent.agent_id, label)
                    answer = (
                        _leakk_direct_reply(agent, item.question, request)
                        if active_protocol == "leakk_official"
                        else agent.answer(item.question, generation_request=request)
                    )
                    row = _record(
                        cfg=cfg,
                        item=item,
                        sample_id=sample_id,
                        arm=label,
                        request=request,
                        scorer=scorer,
                        direct_reply=answer,
                    )
                    write_row(row)

                if active_protocol == "leakk_official":
                    continue

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
                        prompt_sha256=None,
                        decoding_sha256=decoding_hash(req(active_id, label).to_dict()),
                        model_revisions=model_revisions,
                    )
                    cert = _certified(tr, store, blocklist, item, scorer)
                    probe = _probe(
                        active[0],
                        item,
                        store,
                        blocklist,
                        req(active_id, label),
                        scorer,
                        cfg.episode.retrieval_k,
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
                            store=store,
                            blocklist=blocklist,
                            certified=cert,
                            probe=probe,
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
                for label, peer_text, source_item in (
                    ("C3S", source_answer.text, source.item_id),
                    ("C3C", None, None),
                ):
                    store, blocklist, _ = seed_store(items, cfg)
                    policies = _policies(cfg, blocklist)
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
                        prompt_sha256=None,
                        decoding_sha256=decoding_hash(requests[agents[0].agent_id].to_dict()),
                        model_revisions=model_revisions,
                    )
                    cert = _certified(tr, store, blocklist, item, scorer)
                    probe = _probe(
                        agents[1],
                        item,
                        store,
                        blocklist,
                        requests[agents[1].agent_id],
                        scorer,
                        cfg.episode.retrieval_k,
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
                            store=store,
                            blocklist=blocklist,
                            certified=cert,
                            probe=probe,
                            source_evidence=(
                                {
                                    "item_id": source.item_id,
                                    "rendered_prompt_sha256": source_answer.meta[
                                        "rendered_prompt_sha256"
                                    ],
                                    "semantic_user_prompt_sha256": source_answer.meta.get(
                                        "semantic_user_prompt_sha256"
                                    ),
                                    "serialized_chat_prompt_sha256": source_answer.meta.get(
                                        "serialized_chat_prompt_sha256"
                                    ),
                                    "input_ids_sha256": source_answer.meta.get("input_ids_sha256"),
                                    "generation_seed": req(agents[0].agent_id, "C3S", source).seed,
                                    "text": source_answer.text,
                                }
                                if label == "C3S"
                                else None
                            ),
                        )
                    )
                    if label == "C3C":
                        guard_store, guard_bl, _ = seed_store(items, cfg)
                        guard_bl = build_blocklist(
                            "semantic",
                            texts=[f"{x.question} {x.answer}" for x in items],
                            threshold=cfg.memory.semantic_threshold,
                        )
                        guard_tr = _guard_replay(
                            tr, guard_store, guard_bl, item, _policies(cfg, guard_bl, guard=True)
                        )
                        guard_cert = _certified(guard_tr, guard_store, guard_bl, item, scorer)
                        guard_probe = _probe(
                            agents[1],
                            item,
                            guard_store,
                            guard_bl,
                            requests[agents[1].agent_id],
                            scorer,
                            cfg.episode.retrieval_k,
                        )
                        write_row(
                            _record(
                                cfg=cfg,
                                item=item,
                                sample_id=sample_id,
                                arm="C3C-guard",
                                request=requests[agents[0].agent_id],
                                scorer=scorer,
                                transcript=guard_tr,
                                store=guard_store,
                                blocklist=guard_bl,
                                certified=guard_cert,
                                probe=guard_probe,
                                guard_diagnostic=True,
                            )
                        )
    for agent in agents:
        agent.close()
    run_manifest["complete"] = True
    run_manifest["records_sha256"] = hashlib.sha256(records_path.read_bytes()).hexdigest()
    _atomic_json(manifest_path, run_manifest)
    typer.echo(f"wrote {records_path}")
