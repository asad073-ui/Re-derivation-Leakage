"""The graph-study runner: plan, generate, record.

Generation and scoring are separate phases. This module never loads a scorer — an NLI
evaluator and a generation model must not be resident together, and a run whose numbers
depend on a scoring call made during generation cannot be rescored later without
regenerating.

What it produces, under ``runs/graph/<run-id>/``:

    RUN_MANIFEST.json    the durable contract, written BEFORE the first model call
    RESOLVED_CONFIG.json the fully resolved tree and its three hashes
    COHORT.json          the frozen cohort that ran
    PLAN.json            the dry-run cost model
    generations/         immutable, content-addressed shards of raw trajectories
    traces/              graph events and policy decisions
    PERFORMANCE.json     throughput, cache behaviour, defence counters
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ...defenses.semantic_detector import SemanticConceptDetector
from ...eval.leak_at_k import decoding_hash
from ...eval.tofu_data import TofuItem
from ...graph.config import ResolvedGraphConfig
from ...graph.executor import EpisodeSpec, GraphExecutor, GraphTrajectory
from ...graph.scheduler import planned_generations
from ...graph_memory.staged_store import StagedMemory
from ...memory.blocklist import Blocklist, build_blocklist
from ...memory.store import MemoryStore
from ...paths import git_diff_sha256, git_dirty, git_sha
from ...runtime.backend import GenerationBackend, GenRequest
from ...runtime.batch_scheduler import BatchScheduler
from ...runtime.resource_monitor import ResourceMonitor
from .arms import ArmPlan, build_arm_runtime, build_detector, build_registry
from .cohort import Cohort
from .controls import ControlledChallengeSet, build_controlled_challenges, concept_control_mapping
from .evidence import ShardWriter, atomic_json
from .response_bank import ResponseBank

__all__ = ["GraphRunner", "RunPlan", "build_baseline_memory"]


@dataclass(frozen=True)
class RunPlan:
    """What a run intends to do, computed before anything is generated."""

    n_items: int
    n_samples: int
    arms: tuple[str, ...]
    challenges: tuple[str, ...]
    topology: str
    planned_trajectories: int
    planned_graph_generations: int
    planned_probe_generations: int

    @property
    def planned_generations_total(self) -> int:
        return self.planned_graph_generations + self.planned_probe_generations

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_samples": self.n_samples,
            "arms": list(self.arms),
            "challenges": list(self.challenges),
            "topology": self.topology,
            "planned_trajectories": self.planned_trajectories,
            # The GPU-readiness gate compares against this number: the graph's own model
            # calls, before the response cache and excluding the readback probe.
            "planned_graph_generations": self.planned_graph_generations,
            "planned_probe_generations": self.planned_probe_generations,
            "planned_generations_total": self.planned_generations_total,
        }


def build_baseline_memory(
    items: Sequence[TofuItem],
    *,
    blocklist_kind: str = "id",
    embedding_dim: int = 64,
    index_backend: str = "numpy",
    ingest_forget_set: bool = True,
    semantic_threshold: float = 0.75,
) -> tuple[MemoryStore, Blocklist]:
    """The post-deletion state every arm starts from.

    Identical to the two-agent setup: the forget content was ingested, was deleted
    through the memory pathway, and its ids are blocklisted. Both SBU invariants hold
    here, which is what makes anything recovered afterwards a re-derivation rather than
    a failure to delete.
    """
    store = MemoryStore(index_backend=index_backend, embedding_dim=embedding_dim)
    blocklist = build_blocklist(
        blocklist_kind,
        texts=[f"{i.question} {i.answer}" for i in items] if blocklist_kind == "semantic" else None,
        threshold=semantic_threshold,
    )
    if ingest_forget_set:
        ids = [
            store.add(
                f"{item.question} {item.answer}",
                source_agent="ingest",
                source_kind="ingest",
                turn=0,
                meta={"item_id": item.item_id},
            ).node_id
            for item in items
        ]
        for node_id in ids:
            store.delete(node_id, blocklist=blocklist)
    return store, blocklist


@dataclass
class GraphRunner:
    cfg: ResolvedGraphConfig
    items: Sequence[TofuItem]
    cohort: Cohort
    backend: GenerationBackend
    output: Path
    challenges: tuple[str, ...] = ("natural",)
    tokenizer_revision: str | None = None
    resume: bool = False
    bank: ResponseBank = field(default_factory=ResponseBank)
    monitor: ResourceMonitor = field(default_factory=ResourceMonitor)

    # ------------------------------------------------------------------- assembly --

    def __post_init__(self) -> None:
        self.concept_by_item = {entry.item_id: entry.concept_id for entry in self.cohort.items}
        self.registry = build_registry(
            self.items, concept_of=lambda item_id: self.concept_by_item[item_id]
        )
        self.detector: SemanticConceptDetector = build_detector(self.cfg, self.registry)
        self.arm_plans: list[ArmPlan] = build_arm_runtime(
            self.cfg, self.cfg.topology, self.detector
        )
        self.scheduler = BatchScheduler(
            self.backend,
            cache=self.bank.cache,
            max_batch_size=self.cfg.profile.runtime.max_num_seqs,
            tokenizer_revision=self.tokenizer_revision,
            backend_version=_pkg_version(self.cfg.profile.runtime.backend),
            observer=self.bank.note,
        )
        self.mapping = concept_control_mapping(
            [self.concept_by_item[item.item_id] for item in self.items]
        )
        self.control_source = {
            item.item_id: self.items[self.mapping.permutation[i]]
            for i, item in enumerate(self.items)
        }
        self._base_store, self._blocklist = build_baseline_memory(
            self.items,
            blocklist_kind=self.cfg.study.memory.blocklist,
            ingest_forget_set=self.cfg.study.memory.ingest_forget_set,
        )
        self._baseline = self._base_store.snapshot()

    # ----------------------------------------------------------------------- plan --

    def plan(self) -> RunPlan:
        n_samples = self.cfg.profile.sampling.n_samples
        graph_calls = 0
        trajectories = 0
        for challenge in self.challenges:
            injected = self._max_injected_nodes(challenge)
            adjusted = planned_generations(
                self.cfg.topology,
                n_items=len(self.items),
                n_samples=n_samples,
                arms=[a.name for a in self.arm_plans],
                single_agent_arms=[a.name for a in self.arm_plans if a.spec.mode == "single_agent"],
                injected_nodes_per_trajectory=injected,
            )
            graph_calls += adjusted["planned_generations"]
            trajectories += adjusted["planned_trajectories"]
        probes = 2 * trajectories if self.cfg.study.memory.later_episode_probe else 0
        return RunPlan(
            n_items=len(self.items),
            n_samples=n_samples,
            arms=tuple(a.name for a in self.arm_plans),
            challenges=self.challenges,
            topology=self.cfg.topology.name,
            planned_trajectories=trajectories,
            planned_graph_generations=graph_calls,
            planned_probe_generations=probes,
        )

    def _max_injected_nodes(self, challenge: str) -> int:
        """How many nodes emit scripted text instead of generating, for this challenge."""
        if challenge == "natural":
            return 0
        counts = [
            len(cs.injected_outputs)
            for item in self.items
            for cs in build_controlled_challenges(item, self.cfg.topology, [challenge])
        ]
        return max(counts) if counts else 0

    # ------------------------------------------------------------------ execution --

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=True)
        generations = self.output / "generations"
        writer = ShardWriter(generations, shard_size=self.cfg.profile.runtime.shard_size)
        traces = ShardWriter(
            self.output / "traces",
            prefix="graph-events",
            shard_size=self.cfg.profile.runtime.shard_size,
        )
        done = writer.completed()
        if done and not self.resume:
            raise FileExistsError(
                f"{generations} already holds {len(done)} trajectories. Pass resume=True to "
                "continue it, or choose a new output directory — runs are append-only."
            )

        plan = self.plan()
        manifest = self._manifest(plan)
        atomic_json(self.output / "RUN_MANIFEST.json", manifest)
        atomic_json(self.output / "RESOLVED_CONFIG.json", self.cfg.to_dict())
        atomic_json(self.output / "COHORT.json", self.cohort.to_dict())
        atomic_json(self.output / "PLAN.json", plan.to_dict())

        wave = max(1, self.cfg.profile.runtime.max_num_seqs)
        n_samples = self.cfg.profile.sampling.n_samples
        for challenge in self.challenges:
            challenge_sets = self._challenge_sets(challenge)
            units = [(item, sample) for item in self.items for sample in range(n_samples)]
            for arm in self.arm_plans:
                self.scheduler.tag = arm.name
                executor = GraphExecutor(
                    self.cfg.topology,
                    scheduler=self.scheduler,
                    defense=arm.defense,
                    max_new_tokens=self.cfg.study.sampling.max_new_tokens,
                    temperature=self.cfg.study.sampling.temperature,
                    top_p=self.cfg.study.sampling.top_p,
                    top_k=self.cfg.study.sampling.top_k,
                    base_seed=self.cfg.study.sampling.base_seed,
                    retrieval_k=self.cfg.study.memory.retrieval_k,
                    monitor=self.monitor,
                )
                for start in range(0, len(units), wave):
                    batch = units[start : start + wave]
                    episodes: list[EpisodeSpec] = []
                    memories: list[StagedMemory] = []
                    for item, sample in batch:
                        key = (item.item_id, sample, arm.name, challenge)
                        if key in done:
                            continue
                        episode, memory = self._episode(
                            item, sample, arm, challenge, challenge_sets
                        )
                        episodes.append(episode)
                        memories.append(memory)
                    if not episodes:
                        continue
                    for trajectory in executor.run(episodes, memories):
                        writer.append(self._row(trajectory, arm, executor))
                        traces.append(
                            {
                                "item_id": trajectory.spec.item_id,
                                "sample_id": trajectory.spec.sample_id,
                                "arm": trajectory.spec.arm,
                                "challenge": trajectory.spec.challenge,
                                **trajectory.trace.to_dict(with_content=False),
                            }
                        )

        shard_manifest = writer.close()
        trace_manifest = traces.close()
        performance = {
            "runtime": self.monitor.to_dict(),
            "scheduler": self.scheduler.stats(),
            "response_bank": self.bank.to_dict(),
            "detector": {**self.detector.to_dict(), **self.detector.stats()},
            "defenses": {a.name: a.defense.stats() for a in self.arm_plans},
        }
        atomic_json(self.output / "PERFORMANCE.json", performance)

        manifest["complete"] = True
        manifest["evidence_shards"] = [s.to_dict() for s in shard_manifest]
        manifest["trace_shards"] = [s.to_dict() for s in trace_manifest]
        manifest["completed_trajectories"] = writer.n_rows
        manifest["actual_graph_generations"] = self.scheduler.dispatched
        atomic_json(self.output / "RUN_MANIFEST.json", manifest)
        return manifest

    # ------------------------------------------------------------------- episodes --

    def _challenge_sets(self, challenge: str) -> dict[str, ControlledChallengeSet]:
        if challenge == "natural":
            return {}
        out: dict[str, ControlledChallengeSet] = {}
        for item in self.items:
            built = build_controlled_challenges(item, self.cfg.topology, [challenge])
            if built:
                out[item.item_id] = built[0]
        return out

    def _episode(
        self,
        item: TofuItem,
        sample: int,
        arm: ArmPlan,
        challenge: str,
        challenge_sets: dict[str, ControlledChallengeSet],
    ) -> tuple[EpisodeSpec, StagedMemory]:
        source = self.control_source[item.item_id]
        # A cross-concept arm gets the CONTROL item's injection, not the target's.
        # Injecting the target concept into MA-CONTROL would hand the control arm the
        # treatment's content and delete the contrast — the same failure the two-agent
        # work hit when C3C and C3S stopped differing in one variable.
        injection = challenge_sets.get(
            source.item_id if arm.uses_control_question else item.item_id
        )

        store = MemoryStore(
            index_backend="numpy",
            embedding_dim=self._base_store.embedder.dim,
            deletion_mode=self._base_store.deletion_mode,
            refcount_semantics=self._base_store.refcount_semantics,
        )
        store.restore(self._baseline)
        memory = StagedMemory(
            store,
            blocklist=self._blocklist,
            visibility=self.cfg.study.memory.visibility,
        )
        if injection is not None:
            for text in injection.seeded_memory:
                # A re-entry challenge plants content in memory BEFORE the episode, so
                # what it measures is retrieval protection, not the write guard.
                memory.store.add(
                    text, source_agent="ingest", source_kind="ingest", meta={"injected": True}
                )

        episode = EpisodeSpec(
            trajectory_id=f"{arm.name}:{challenge}:{item.item_id}:{sample}",
            item_id=item.item_id,
            concept_id=self.concept_by_item[item.item_id],
            sample_id=sample,
            arm=arm.name,
            question=item.question,
            upstream_question=source.question if arm.uses_control_question else None,
            upstream_item_id=source.item_id if arm.uses_control_question else None,
            active_nodes=arm.active_nodes,
            injected_outputs=dict(injection.injected_outputs) if injection else {},
            injected_tool_inputs=dict(injection.injected_tool_inputs) if injection else {},
            node_questions=dict(injection.node_questions) if injection else {},
            challenge=challenge,
            topology=self.cfg.topology.name,
        )
        return episode, memory

    # ----------------------------------------------------------------------- rows --

    def _row(self, traj: GraphTrajectory, arm: ArmPlan, executor: GraphExecutor) -> dict:
        item = next(i for i in self.items if i.item_id == traj.spec.item_id)
        probe = self._probe(traj, executor) if self.cfg.study.memory.later_episode_probe else {}
        node_events = traj.trace.of_kind("node_executed")
        edge_events = traj.trace.of_kind("edge_decision")
        write_events = traj.trace.of_kind("memory_write")
        read_events = traj.trace.of_kind("memory_read")
        decode = GenRequest(
            request_id="decoding",
            prompt="",
            max_new_tokens=self.cfg.study.sampling.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.study.sampling.temperature,
            top_p=self.cfg.study.sampling.top_p,
            top_k=self.cfg.study.sampling.top_k,
        )
        return {
            "schema": "graph-leak-row-v1",
            "trajectory_id": traj.spec.trajectory_id,
            "item_id": traj.spec.item_id,
            "concept_id": traj.spec.concept_id,
            "sample_id": traj.spec.sample_id,
            "arm": traj.spec.arm,
            "challenge": traj.spec.challenge,
            "topology": traj.spec.topology,
            "defense": arm.defense.name,
            # The reference answer travels with the raw evidence so a pinned evaluator can
            # rescore completed generations without regenerating them.
            "reference_answer": item.answer,
            "question": item.question,
            "control_source_item_id": traj.spec.upstream_item_id,
            "final_text": traj.final_text,
            "final_status": traj.final_status,
            "raw_outputs": {
                "agent_messages": traj.agent_output_texts(),
                "released_edge_payloads": traj.released_edge_texts(),
                "memory_writes": traj.committed_write_texts(),
                "probe": probe,
            },
            "counts": {
                "generations": traj.n_generations,
                "forced_or_injected": traj.n_forced,
                "edges": len(edge_events),
                "edges_blocked": sum(
                    1 for e in edge_events if getattr(e, "release_status", "pass") != "pass"
                ),
                "edges_removed": sum(
                    1 for e in edge_events if not getattr(e, "edge_preserved", True)
                ),
                "write_candidates": len(write_events),
                "writes_allowed": sum(1 for e in write_events if getattr(e, "allowed", False)),
                "committed_writes": len(traj.committed_write_ids),
                "retrieval_withheld": sum(
                    len(getattr(e, "withheld_node_ids", []))
                    + len(getattr(e, "rescan_withheld_node_ids", []))
                    for e in read_events
                ),
                "nodes_executed": len(node_events),
                "nodes_abstained": sum(1 for e in node_events if getattr(e, "abstained", False)),
            },
            "provenance": {
                "prompt_sha256s": sorted(
                    {str(getattr(e, "prompt_sha256", "")) for e in node_events}
                ),
                "generation_seeds": [getattr(e, "generation_seed", None) for e in node_events],
                "model_revisions": sorted(
                    {
                        str(getattr(e, "model_revision", None))
                        for e in node_events
                        if getattr(e, "model_revision", None)
                    }
                ),
                "decoding_sha256": decoding_hash(decode.decoding()),
                "detector_version": self.detector.version,
                "trace_digest": traj.trace.digest(),
                "cached_nodes": sum(1 for e in node_events if getattr(e, "cached", False)),
            },
            "memory_evidence": [
                {
                    "node_id": node_id,
                    "content": node.content,
                    "content_sha256": hashlib.sha256(node.content.encode("utf-8")).hexdigest(),
                    "parent_ids": list(node.parent_ids),
                    "forget_ids": list(traj.memory.scopes.forget_ids(node_id)),
                    "is_parametric": node.is_parametric,
                }
                for node_id in traj.committed_write_ids
                if (node := traj.memory.store.get(node_id)) is not None
            ],
        }

    def _probe(self, traj: GraphTrajectory, executor: GraphExecutor) -> dict:
        """Later-episode readback: the sink asks again, with and without the new memory.

        Two draws under the same seed. ``attributable`` needs the with-memory answer to
        leak, the without-memory answer not to, and the retrieval to have returned a node
        that actually carries the content — otherwise "memory caused it" is an assumption.
        """
        item_question = traj.spec.question
        sink = self.cfg.topology.node(self.cfg.topology.sink)
        seed = executor.seed_for_node(traj.spec, self.cfg.topology.sink) ^ 0x5EED
        guarded = traj.memory.candidates(item_question, self.cfg.study.memory.retrieval_k)
        from ...defenses.base import RetrievalContext

        verdict = executor.defense.on_retrieval(
            RetrievalContext(
                node_id=self.cfg.topology.sink,
                depth=self.cfg.topology.depth_of(self.cfg.topology.sink),
                query=item_question,
                candidates=guarded,
            )
        )
        allowed_ids = tuple(verdict.allowed_node_ids)
        memory_texts = traj.memory.texts_for(allowed_ids)

        from ...graph.prompt_builder import build_agent_prompt, role_system_prompt

        system = role_system_prompt(sink.role, sink.system_prompt)
        with_prompt = build_agent_prompt(question=item_question, memory_texts=memory_texts)
        without_prompt = build_agent_prompt(question=item_question)
        requests = [
            GenRequest(
                request_id=f"{traj.spec.trajectory_id}|probe-with",
                prompt=with_prompt,
                system=system,
                max_new_tokens=self.cfg.study.sampling.max_new_tokens,
                seed=seed,
                do_sample=True,
                temperature=self.cfg.study.sampling.temperature,
                top_p=self.cfg.study.sampling.top_p,
                top_k=self.cfg.study.sampling.top_k,
                model=sink.model,
            ),
            GenRequest(
                request_id=f"{traj.spec.trajectory_id}|probe-without",
                prompt=without_prompt,
                system=system,
                max_new_tokens=self.cfg.study.sampling.max_new_tokens,
                seed=seed,
                do_sample=True,
                temperature=self.cfg.study.sampling.temperature,
                top_p=self.cfg.study.sampling.top_p,
                top_k=self.cfg.study.sampling.top_k,
                model=sink.model,
            ),
        ]
        responses = self.scheduler.run(requests)
        return {
            "with_store_text": responses[requests[0].request_id].text,
            "without_store_text": responses[requests[1].request_id].text,
            "retrieved_node_ids": list(allowed_ids),
            "retrieved_texts": list(memory_texts),
            "withheld_node_ids": list(verdict.withheld_node_ids),
            "rescan_withheld_node_ids": list(verdict.rescan_withheld_node_ids),
            "probe_seed": seed,
        }

    # ------------------------------------------------------------------- manifest --

    def _manifest(self, plan: RunPlan) -> dict:
        uses_gold = any(c != "natural" for c in self.challenges)
        return {
            "schema": "graph-run-manifest-v1",
            "study_id": self.cfg.study.study_id,
            "phase": self.cfg.phase,
            **self.cfg.hashes(),
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "git_diff_sha256": git_diff_sha256(),
            "python": platform.python_version(),
            "runtime_packages": {
                package: _pkg_version(package)
                for package in ("torch", "transformers", "datasets", "vllm", "numpy", "pydantic")
            },
            "backend": self.cfg.profile.runtime.backend,
            "model": self.cfg.model.model_dump(mode="json"),
            "tokenizer_revision": self.tokenizer_revision,
            "shared_model_handles": getattr(self.backend, "shared_handle_ids", dict)(),
            "logical_agents": self.cfg.profile.agents.logical_count,
            "share_model_handle": self.cfg.profile.agents.share_model_handle,
            "agent_note": (
                f"{self.cfg.profile.agents.logical_count} logical agents sharing one "
                "unlearned model backend; NOT independently unlearned checkpoints"
            ),
            "topology": self.cfg.topology.to_dict(),
            "arms": [a.to_dict() for a in self.arm_plans],
            "detector": self.detector.to_dict(),
            "detector_status": self.cfg.study.detector.status,
            "concept_registry": self.registry.to_dict(),
            "cohort_fingerprint": self.cohort.fingerprint(),
            "cohort_split": self.cohort.split,
            "control_mapping": self.mapping.to_dict(),
            "challenges": list(self.challenges),
            "uses_gold_answers": uses_gold,
            "uses_gold_answers_note": (
                "controlled challenges construct injected messages from the gold answer; "
                "the runtime concept registry never contains one"
                if uses_gold
                else "no injected content; every message is a model output"
            ),
            "sampling": {
                **self.cfg.study.sampling.model_dump(mode="json"),
                **self.cfg.profile.sampling.model_dump(mode="json"),
            },
            "profile_reportable": self.cfg.profile.reportable,
            "plan": plan.to_dict(),
            "offline_scoring": True,
            "semantic_scorer": None,
            "complete": False,
        }


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "absent"
