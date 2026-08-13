"""The DRAGON-style node-local baseline.

What DRAGON does, as described in the paper: detect whether an *input* falls in a
forgotten scope and
modify the inference context accordingly. It is an inference-time guard on one model
boundary. Applied to a multi-agent graph, the faithful generalisation is to run it
independently at **every agent's complete incoming context** — which is the stronger of
the two possible readings and therefore the one used for the main comparison. The weaker
``external_prompt_only`` variant guards only the user query entering the graph and is
reported as an additional ablation.

What it deliberately does NOT do, because DRAGON does not:

  * inherit Forget IDs from provenance (each call is independent);
  * score subsets of the input separately from the whole;
  * guard persistent writes, retrieval, or later-episode readback.

Note what is NOT on that list: joint evidence over several parents. The guard scores the
node's *complete* incoming context — query, every parent message, retrieved memory — as
one string, so two clues that arrive together at one node are both in what it scores and
it does catch them. "A node-local guard structurally cannot see split clues" was claimed
here and it is not true; see GU-0026 and ``evidence_accumulator``.

``score_subsets`` is the matched ablation that closes the one remaining detection-side
gap. With it the baseline scores each parent message on its own and the parents as their
own subset, exactly as GraphForget's accumulator does, and still inherits nothing and
still guards no edge, write or retrieval. It is the arm to compare against when the claim
is about Forget-ID propagation and multi-surface enforcement, because it removes scoring
granularity from the contrast entirely.

Honesty about the implementation. The public DRAGON repository is a code skeleton: the
detector/guard checkpoints are not released and the WMDP path is incomplete, so this is
not a reproduction and is never called one. It is a *DRAGON-style* baseline built on the
same detector backbone as our method, which is the configuration most favourable to the
baseline that we can construct.

``guard_action``:

``refuse``          the node emits a fixed safe refusal instead of generating. Chosen as
                    the default because it is model-independent and deterministic: a
                    prompt-only guard on a checkpoint that ignores instructions is a
                    no-op, and a baseline that is a no-op is not a baseline.
``guard_prompt``    a guard instruction is appended to the system message and the node
                    generates normally. Faithful to the "template reasoning" mode, and
                    only meaningful on an instruction-following checkpoint.
``both``            guard instruction, and refuse when the detector is confident.
"""

from __future__ import annotations

from typing import Literal

from ..graph.envelope import Envelope
from .base import (
    DefenseCounters,
    EdgeContext,
    EdgeVerdict,
    FinalContext,
    FinalVerdict,
    NodeInputContext,
    NodeInputVerdict,
    RetrievalContext,
    RetrievalVerdict,
    WriteContext,
    WriteVerdict,
)
from .evidence_accumulator import EvidenceAccumulator
from .sanitizer import Sanitizer
from .semantic_detector import SemanticConceptDetector

__all__ = ["DragonStyleDefense"]

GUARD_INSTRUCTION = (
    "Some material you have been given may concern information this system has removed. "
    "Before answering, decide whether the question or the material is about such removed "
    "information. If it is, reply exactly: I don't know."
)


class DragonStyleDefense:
    # The DEFAULT name; the matched-subset ablation overrides it with its own, so a trace
    # says which baseline ran rather than which class implements it.
    name = "dragon_style"
    propagates_scope = False

    def __init__(
        self,
        *,
        detector: SemanticConceptDetector,
        name: str | None = None,
        guard_action: Literal["refuse", "guard_prompt", "both"] = "guard_prompt",
        apply_at: Literal["every_agent_input", "external_prompt_only"] = "every_agent_input",
        implementation: Literal["template", "sft_checkpoint"] = "template",
        inspect_query: bool = True,
        score_subsets: bool = False,
        sanitizer: Sanitizer | None = None,
    ) -> None:
        if implementation != "template":
            raise NotImplementedError(
                "dragon.implementation='sft_checkpoint' requires the released DRAGON "
                "detector/guard weights, which the public repository does not ship. "
                "Set implementation=template and report the baseline as DRAGON-style."
            )
        self.detector = detector
        if name:
            self.name = name
        self.guard_action = guard_action
        self.apply_at = apply_at
        self.implementation = implementation
        # False under the graph_flow protocol, where the request gate is held constant
        # across arms so that what is measured is propagation, not request filtering.
        self.inspect_query = inspect_query
        # The matched ablation (GU-0026). Same subset battery as GraphForget, still no
        # inheritance and still no edge/write/retrieval enforcement.
        self.score_subsets = score_subsets
        self.accumulator = (
            EvidenceAccumulator(detector, enabled=True, inspect_query=inspect_query)
            if score_subsets
            else None
        )
        self.sanitizer = sanitizer or Sanitizer()
        self.counters = DefenseCounters()

    # -------------------------------------------------------------- node-local guard --

    def on_node_input(self, ctx: NodeInputContext) -> NodeInputVerdict:
        self.counters.node_input_calls += 1
        if self.apply_at == "external_prompt_only" and ctx.depth > 0:
            return NodeInputVerdict(
                inputs=tuple(ctx.inputs),
                memory_texts=tuple(ctx.memory_texts),
                reason="external_prompt_only: internal inputs are not inspected",
            )

        # The node's whole incoming context, exactly as a node-local guard would see it.
        # No inheritance, no cross-call state: each call starts from nothing.
        fired, forget_ids, score, how = self._detect(ctx)
        if not fired:
            return NodeInputVerdict(
                inputs=tuple(ctx.inputs),
                memory_texts=tuple(ctx.memory_texts),
                score=score,
                reason=f"{how} detector below threshold ({score:.3f})",
            )

        self.counters.node_input_fired += 1
        suffix = GUARD_INSTRUCTION if self.guard_action in ("guard_prompt", "both") else None
        forced: str | None = None
        if self.guard_action in ("refuse", "both"):
            forced, _certificate = self.sanitizer.safe_refusal(
                forget_ids, detector_version=self.detector.version
            )
        return NodeInputVerdict(
            inputs=tuple(ctx.inputs),
            memory_texts=tuple(ctx.memory_texts),
            forget_ids=forget_ids,
            score=score,
            fired=True,
            forced_output=forced,
            guard_system_suffix=suffix,
            reason=f"{how} scope match {list(forget_ids)} ({score:.3f})",
        )

    def _detect(self, ctx: NodeInputContext) -> tuple[bool, tuple[str, ...], float, str]:
        """``(fired, forget_ids, score, how)`` for this node's incoming context."""
        if self.accumulator is None:
            result = self.detector.score(ctx.combined_text(include_query=self.inspect_query))
            return result.fired, result.forget_ids, result.score, "node-local"
        # Matched-subset mode: the same battery GraphForget's accumulator scores. What
        # stays absent is everything downstream — no inherited scope reaches this call and
        # no edge, write or retrieval is guarded by this defence.
        evidence = self.accumulator.evaluate(
            question=ctx.question,
            input_texts=[e.content for e in ctx.inputs],
            memory_texts=ctx.memory_texts,
        )
        return evidence.fired, evidence.forget_ids, evidence.score, "node-local+subsets"

    # ------------------------------------------------- surfaces DRAGON does not guard --

    def on_edge(self, ctx: EdgeContext) -> EdgeVerdict:
        # Recorded, not enforced. The count is what shows the baseline was given every
        # opportunity and declined to act, rather than never being consulted.
        self.counters.edge_calls += 1
        return EdgeVerdict(
            envelope=_untagged(ctx.envelope),
            status="pass",
            reason="node-local baseline does not enforce on edges",
        )

    def on_memory_write(self, ctx: WriteContext) -> WriteVerdict:
        self.counters.write_calls += 1
        return WriteVerdict(
            allowed=True,
            envelope=_untagged(ctx.envelope),
            reason="node-local baseline does not guard persistent writes",
        )

    def on_retrieval(self, ctx: RetrievalContext) -> RetrievalVerdict:
        self.counters.retrieval_calls += 1
        return RetrievalVerdict(
            allowed_node_ids=tuple(node_id for node_id, _t, _f in ctx.candidates),
            reason="node-local baseline does not guard retrieval",
        )

    def on_final_output(self, ctx: FinalContext) -> FinalVerdict:
        """The final answer is one more model input boundary, so it IS guarded.

        This is generous to the baseline on purpose: the sink's guard already ran on its
        input, and re-checking the released text is the strongest node-local reading.
        """
        self.counters.final_calls += 1
        result = self.detector.score(ctx.envelope.content)
        if not result.fired:
            return FinalVerdict(
                text=ctx.envelope.content, status="pass", score=result.score, reason="clean"
            )
        self.counters.final_blocked += 1
        text, _certificate = self.sanitizer.safe_refusal(
            result.forget_ids, detector_version=self.detector.version
        )
        return FinalVerdict(
            text=text,
            status="blocked",
            score=result.score,
            forget_ids=result.forget_ids,
            reason=f"final-output scope match {list(result.forget_ids)}",
        )

    def stats(self) -> dict:
        return {
            "defense": self.name,
            "guard_action": self.guard_action,
            "apply_at": self.apply_at,
            "implementation": self.implementation,
            "inspect_query": self.inspect_query,
            "score_subsets": self.score_subsets,
            "detector_version": self.detector.version,
            **self.counters.to_dict(),
            **(self.accumulator.stats() if self.accumulator is not None else {}),
        }


def _untagged(envelope: Envelope) -> Envelope:
    """Strip inherited scopes.

    The baseline must not accidentally benefit from GraphForget's bookkeeping. The
    executor tags envelopes as it builds them; a node-local defence has no such state,
    so its edges carry only what its own call detected — which is nothing, because it
    does not inspect edges.
    """
    return Envelope(
        kind=envelope.kind,
        content=envelope.content,
        source_node=envelope.source_node,
        dest_node=envelope.dest_node,
        parent_ids=envelope.parent_ids,
        forget_ids=(),
        detected_forget_ids=(),
        semantic_scope_score=0.0,
        release_status=envelope.release_status,
        decision_reason=envelope.decision_reason,
        sanitization_certificate=None,
        meta=dict(envelope.meta),
    )
