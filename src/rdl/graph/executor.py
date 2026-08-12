"""Running whole graph trajectories, layer by layer, across many episodes at once.

The execution order is fixed and identical for every arm:

    for each graph depth:
        for each active node at that depth:
            gather parent payloads that were actually released to it
            retrieve memory through the retrieval guard
            let the defence inspect the complete node input
            build the prompt (identical wrapper in every arm)
        generate the whole layer in one batch
        wrap each output in a provenance envelope
        let the defence enforce on each outgoing edge
        stage a persistent-write candidate for every writing node
    release the sink's output through the final check
    commit the staged writes

The arms differ in exactly two places: which question the non-sink nodes answer
(same-concept or cross-concept) and which ``Defense`` object is installed. Nothing else
in this file branches on the arm.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..defenses.base import (
    Defense,
    EdgeContext,
    FinalContext,
    NodeInputContext,
    WriteContext,
)
from ..eval.leak_at_k import seed_for
from ..graph_memory.retrieval_guard import RetrievalGuard
from ..graph_memory.staged_store import StagedMemory, WriteCandidate
from ..runtime.backend import GenRequest
from ..runtime.batch_scheduler import BatchScheduler
from ..runtime.resource_monitor import ResourceMonitor
from .envelope import Envelope, derive_envelope
from .prompt_builder import build_agent_prompt, prompt_sha256, role_system_prompt
from .scheduler import plan_layers
from .schema import GraphSpec
from .trace import (
    EdgeDecisionEvent,
    FinalReleaseEvent,
    GraphTrace,
    MemoryReadEvent,
    MemoryWriteEvent,
    NodeExecutedEvent,
)

__all__ = ["EpisodeSpec", "GraphExecutor", "GraphTrajectory"]


@dataclass(frozen=True)
class EpisodeSpec:
    """One (item, sample, arm) trajectory to run."""

    trajectory_id: str
    item_id: str
    concept_id: str
    sample_id: int
    arm: str
    question: str
    # Question the NON-SINK nodes answer. ``None`` means "the same question", which is
    # MA-LEAK. MA-CONTROL passes a cross-concept question here, so the peer messages
    # reaching the sink concern a different author in a byte-identical wrapper.
    upstream_question: str | None = None
    upstream_item_id: str | None = None
    # Nodes that participate at all. The single-agent arm passes just the sink.
    active_nodes: tuple[str, ...] = ()
    # Controlled stress test: node -> the exact text that node emits instead of
    # generating. Identical across arms, so the arms stay comparable.
    injected_outputs: dict[str, str] = field(default_factory=dict)
    # Controlled stress test: node -> a tool response placed in that node's input.
    injected_tool_inputs: dict[str, str] = field(default_factory=dict)
    # Controlled stress test: node -> a replacement query. ``split_clues`` gives the join
    # node a delegated task so that its own query is not itself in the forgotten scope.
    node_questions: dict[str, str] = field(default_factory=dict)
    challenge: str = "natural"
    topology: str = ""

    def question_for(self, node_id: str, sink: str) -> tuple[str, str]:
        """``(question, item_id_for_seeding)`` for one node."""
        override = self.node_questions.get(node_id)
        if override is not None:
            # Still seeded by the target item: the draw must stay paired across arms.
            return override, self.item_id
        if node_id != sink and self.upstream_question is not None:
            return self.upstream_question, (self.upstream_item_id or self.item_id)
        return self.question, self.item_id


@dataclass
class GraphTrajectory:
    """Everything one trajectory produced, ready for the evidence shard."""

    spec: EpisodeSpec
    trace: GraphTrace
    memory: StagedMemory
    final_text: str = ""
    final_status: str = "pass"
    node_outputs: dict[str, Envelope] = field(default_factory=dict)
    delivered: dict[str, list[Envelope]] = field(default_factory=dict)
    committed_write_ids: list[str] = field(default_factory=list)
    n_generations: int = 0
    n_forced: int = 0

    def released_edge_texts(self) -> list[str]:
        """Content that actually crossed an edge to a consumer."""
        return [e.content for payloads in self.delivered.values() for e in payloads if e.content]

    def agent_output_texts(self) -> list[str]:
        """Everything the agents generated, before enforcement.

        This is the ``raw_message_leak`` population. It is reported alongside the
        enforced numbers so a defence cannot look effective merely by never being
        measured on what the model actually produced.
        """
        return [e.content for e in self.node_outputs.values()]

    def committed_write_texts(self) -> list[str]:
        return [
            node.content
            for node_id in self.committed_write_ids
            if (node := self.memory.store.get(node_id)) is not None
        ]


class GraphExecutor:
    def __init__(
        self,
        spec: GraphSpec,
        *,
        scheduler: BatchScheduler,
        defense: Defense,
        max_new_tokens: int = 128,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        top_k: int | None = 0,
        base_seed: int = 1729,
        retrieval_k: int = 5,
        prompt_style: str = "openunlearning",
        model_profile: str = "primary",
        monitor: ResourceMonitor | None = None,
    ) -> None:
        self.spec = spec
        self.scheduler = scheduler
        self.defense = defense
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.base_seed = base_seed
        self.retrieval_k = retrieval_k
        self.prompt_style = prompt_style
        self.model_profile = model_profile
        self.monitor = monitor if monitor is not None else ResourceMonitor()

    # --------------------------------------------------------------------- seeding --

    def seed_for_node(self, episode: EpisodeSpec, node_id: str) -> int:
        """Common random numbers across arms.

        The seed keys on the item whose question this node is answering, its sample and
        its node id — never on the arm. Two arms therefore draw the same trajectory for
        any node whose prompt they did not change, which is what makes the paired
        bootstrap a paired one.
        """
        _question, item_id = episode.question_for(node_id, self.spec.sink)
        return seed_for(
            self.base_seed,
            item_id=item_id,
            sample_id=episode.sample_id,
            agent_id=node_id,
            arm="shared",
        )

    # ------------------------------------------------------------------ execution --

    def run(
        self, episodes: Sequence[EpisodeSpec], memories: Sequence[StagedMemory]
    ) -> list[GraphTrajectory]:
        """Run many trajectories, batching every graph layer across all of them."""
        if len(episodes) != len(memories):
            raise ValueError("each episode needs its own StagedMemory")
        trajectories: dict[str, GraphTrajectory] = {}
        for episode, memory in zip(episodes, memories, strict=True):
            trace = GraphTrace(
                trajectory_id=episode.trajectory_id,
                item_id=episode.item_id,
                concept_id=episode.concept_id,
                sample_id=episode.sample_id,
                arm=episode.arm,
                topology=episode.topology or self.spec.name,
                challenge=episode.challenge,
            )
            trajectories[episode.trajectory_id] = GraphTrajectory(
                spec=episode, trace=trace, memory=memory
            )

        active = [
            (
                e.trajectory_id,
                frozenset(e.active_nodes) if e.active_nodes else frozenset(self.spec.node_ids()),
            )
            for e in episodes
        ]

        for depth, units in plan_layers(self.spec, active):
            plans: list[_NodePlan] = []
            requests: list[GenRequest] = []
            for unit in units:
                traj = trajectories[unit.trajectory_id]
                plan = self._prepare(traj, unit.node_id, depth)
                plans.append(plan)
                if plan.request is not None:
                    requests.append(plan.request)

            started = time.perf_counter()
            # The graph's own calls. The readback probe is dispatched by the runner under
            # purpose="probe", and the two are counted apart.
            responses = self.scheduler.run(requests, purpose="graph") if requests else {}
            self.monitor.record_batch(
                seconds=time.perf_counter() - started,
                n=len(requests),
                prompt_tokens=sum(r.prompt_tokens for r in responses.values()),
                completion_tokens=sum(r.completion_tokens for r in responses.values()),
            )
            for plan in plans:
                text = plan.forced_output
                cached = False
                revision: str | None = None
                if plan.request is not None:
                    response = responses[plan.request.request_id]
                    text, cached, revision = response.text, response.cached, response.model_revision
                self._finish(plan, text or "", cached=cached, model_revision=revision)

        out: list[GraphTrajectory] = []
        for episode in episodes:
            traj = trajectories[episode.trajectory_id]
            self._release_final(traj)
            traj.committed_write_ids = traj.memory.commit()
            out.append(traj)
        return out

    # ---------------------------------------------------------------- node prepare --

    def _prepare(self, traj: GraphTrajectory, node_id: str, depth: int) -> _NodePlan:
        episode = traj.spec
        node = self.spec.node(node_id)
        question, _item = episode.question_for(node_id, self.spec.sink)

        inputs: list[Envelope] = list(traj.delivered.get(node_id, []))
        tool_text = episode.injected_tool_inputs.get(node_id)
        if tool_text:
            inputs.append(
                traj.trace.record_envelope(
                    derive_envelope(
                        kind="tool_response",
                        content=tool_text,
                        source_node="tool",
                        dest_node=node_id,
                        meta={"injected": True, "challenge": episode.challenge},
                    )
                )
            )
        # Only content that was actually released reaches the model. Quarantined
        # payloads stay in the trace as evidence and out of the prompt.
        visible = tuple(e for e in inputs if e.released and e.content.strip())

        guard = RetrievalGuard(traj.memory, self.defense, retrieval_k=self.retrieval_k)
        retrieved = guard.retrieve(question, node_id=node_id, depth=depth)
        traj.trace.append(
            MemoryReadEvent(
                node_id=node_id,
                depth=depth,
                defense=self.defense.name,
                query_sha256=prompt_sha256(question, None),
                returned_node_ids=list(retrieved.node_ids),
                withheld_node_ids=list(retrieved.withheld_node_ids),
                rescan_withheld_node_ids=list(retrieved.rescan_withheld_node_ids),
                reason=retrieved.reason,
            )
        )

        inherited: set[str] = set()
        for envelope in visible:
            inherited |= set(envelope.forget_ids)

        verdict = self.defense.on_node_input(
            NodeInputContext(
                node_id=node_id,
                role=node.role,
                depth=depth,
                question=question,
                inputs=visible,
                memory_texts=retrieved.texts,
                inherited_forget_ids=tuple(sorted(inherited)),
            )
        )

        system = role_system_prompt(node.role, node.system_prompt)
        if verdict.guard_system_suffix:
            system = f"{system}\n{verdict.guard_system_suffix}"
        prompt = build_agent_prompt(
            question=question,
            memory_texts=verdict.memory_texts,
            peer_envelopes=verdict.inputs,
            style=self.prompt_style,  # type: ignore[arg-type]
        )

        injected = episode.injected_outputs.get(node_id)
        forced = injected if injected is not None else verdict.forced_output
        request: GenRequest | None = None
        seed = self.seed_for_node(episode, node_id)
        if forced is None:
            request = GenRequest(
                request_id=f"{episode.trajectory_id}|{node_id}",
                prompt=prompt,
                system=system,
                max_new_tokens=self.max_new_tokens,
                seed=seed,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                model=node.model or self.model_profile,
            )
        return _NodePlan(
            traj=traj,
            node_id=node_id,
            depth=depth,
            role=node.role,
            question=question,
            inputs=tuple(verdict.inputs),
            memory_node_ids=tuple(retrieved.node_ids),
            forget_ids=tuple(sorted(set(verdict.forget_ids) | inherited)),
            score=verdict.score,
            prompt=prompt,
            system=system,
            request=request,
            forced_output=forced,
            forced_by_guard=verdict.forced_output is not None and injected is None,
            seed=seed,
            writes_memory=node.writes_memory,
        )

    # ----------------------------------------------------------------- node finish --

    def _finish(
        self, plan: _NodePlan, text: str, *, cached: bool, model_revision: str | None
    ) -> None:
        traj = plan.traj
        envelope = traj.trace.record_envelope(
            derive_envelope(
                kind="agent_output",
                content=text,
                source_node=plan.node_id,
                parents=plan.inputs,
                detected=(),
                score=plan.score,
                meta={
                    "role": plan.role,
                    "forced_by_guard": plan.forced_by_guard,
                    "injected": plan.request is None and not plan.forced_by_guard,
                },
            )
        )
        if plan.forget_ids:
            envelope = envelope.with_decision(
                status="pass",
                reason="scopes carried from this node's inputs",
                added_forget_ids=plan.forget_ids,
                score=plan.score,
            )
        traj.node_outputs[plan.node_id] = envelope
        traj.n_generations += 1 if plan.request is not None else 0
        traj.n_forced += 1 if plan.request is None else 0
        traj.trace.append(
            NodeExecutedEvent(
                node_id=plan.node_id,
                depth=plan.depth,
                role=plan.role,
                prompt_sha256=prompt_sha256(plan.prompt, plan.system),
                input_envelope_ids=[e.envelope_id for e in plan.inputs],
                output_envelope_id=envelope.envelope_id,
                output_sha256=envelope.content_hash,
                generation_seed=plan.seed if plan.request is not None else None,
                model_revision=model_revision,
                cached=cached,
                generated=plan.request is not None,
                abstained=plan.forced_by_guard,
            )
        )

        for dst in self.spec.children(plan.node_id):
            verdict = self.defense.on_edge(
                EdgeContext(
                    src=plan.node_id,
                    dst=dst,
                    depth=plan.depth,
                    envelope=envelope.to(dst),
                    question=plan.question,
                )
            )
            delivered = verdict.envelope.to(dst)
            traj.trace.record_envelope(delivered)
            traj.delivered.setdefault(dst, []).append(delivered)
            traj.trace.append(
                EdgeDecisionEvent(
                    node_id=plan.node_id,
                    depth=plan.depth,
                    src=plan.node_id,
                    dst=dst,
                    defense=self.defense.name,
                    release_status=verdict.status,
                    reason=verdict.reason,
                    forget_ids=list(verdict.forget_ids),
                    inherited_forget_ids=list(envelope.forget_ids),
                    score=verdict.score,
                    envelope_id=delivered.envelope_id,
                    edge_preserved=verdict.edge_preserved,
                )
            )

        if plan.writes_memory:
            self._stage_write(plan, envelope)

    def _stage_write(self, plan: _NodePlan, envelope: Envelope) -> None:
        traj = plan.traj
        verdict = self.defense.on_memory_write(
            WriteContext(
                node_id=plan.node_id,
                depth=plan.depth,
                envelope=envelope,
                question=plan.question,
                retrieved_node_ids=plan.memory_node_ids,
            )
        )
        if verdict.allowed and envelope.content.strip():
            traj.memory.stage(
                WriteCandidate(
                    content=envelope.content,
                    source_node=plan.node_id,
                    envelope_id=envelope.envelope_id,
                    # ONLY nodes actually in the retrieval context. Never synthesise a
                    # derivation edge: a parametric answer legitimately has no parents,
                    # and that absence is the finding, not a gap to paper over.
                    parent_store_ids=plan.memory_node_ids,
                    forget_ids=verdict.forget_ids,
                    score=verdict.score,
                    detector_version=getattr(
                        getattr(self.defense, "detector", None), "version", ""
                    ),
                    meta={"item_id": traj.spec.item_id, "arm": traj.spec.arm},
                )
            )
        traj.trace.append(
            MemoryWriteEvent(
                node_id=plan.node_id,
                depth=plan.depth,
                defense=self.defense.name,
                allowed=verdict.allowed,
                reason=verdict.reason,
                content_sha256=envelope.content_hash,
                forget_ids=list(verdict.forget_ids),
                inherited_forget_ids=list(envelope.forget_ids),
                parent_node_ids=list(plan.memory_node_ids),
                score=verdict.score,
            )
        )

    # ---------------------------------------------------------------------- final --

    def _release_final(self, traj: GraphTrajectory) -> None:
        envelope = traj.node_outputs.get(self.spec.sink)
        if envelope is None:
            traj.final_text, traj.final_status = "", "pass"
            traj.trace.final_text, traj.trace.final_status = "", "pass"
            return
        verdict = self.defense.on_final_output(
            FinalContext(node_id=self.spec.sink, envelope=envelope, question=traj.spec.question)
        )
        traj.final_text = verdict.text
        traj.final_status = verdict.status
        traj.trace.final_text = verdict.text
        traj.trace.final_status = verdict.status
        traj.trace.append(
            FinalReleaseEvent(
                node_id=self.spec.sink,
                depth=self.spec.depth_of(self.spec.sink),
                defense=self.defense.name,
                release_status=verdict.status,
                reason=verdict.reason,
                forget_ids=list(verdict.forget_ids),
                score=verdict.score,
                output_sha256=Envelope(
                    kind="agent_output", content=verdict.text, source_node=self.spec.sink
                ).content_hash,
            )
        )


@dataclass
class _NodePlan:
    traj: GraphTrajectory
    node_id: str
    depth: int
    role: str
    question: str
    inputs: tuple[Envelope, ...]
    memory_node_ids: tuple[str, ...]
    forget_ids: tuple[str, ...]
    score: float
    prompt: str
    system: str
    request: GenRequest | None
    forced_output: str | None
    forced_by_guard: bool
    seed: int
    writes_memory: bool
