"""The DRAGON-style node-local baseline.

What DRAGON does, as published: detect whether an *input* falls in a forgotten scope and
modify the inference context accordingly. It is an inference-time guard on one model
boundary. Applied to a multi-agent graph, the faithful generalisation is to run it
independently at **every agent's complete incoming context** — which is the stronger of
the two possible readings and therefore the one used for the main comparison. The weaker
``external_prompt_only`` variant guards only the user query entering the graph and is
reported as an additional ablation.

What it deliberately does NOT do, because DRAGON does not:

  * inherit Forget IDs from provenance (each call is independent);
  * accumulate evidence across parent messages jointly;
  * guard persistent writes, retrieval, or later-episode readback.

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
from .sanitizer import Sanitizer
from .semantic_detector import SemanticConceptDetector

__all__ = ["DragonStyleDefense"]

GUARD_INSTRUCTION = (
    "Some material you have been given may concern information this system has removed. "
    "Before answering, decide whether the question or the material is about such removed "
    "information. If it is, reply exactly: I don't know."
)


class DragonStyleDefense:
    name = "dragon_style"
    propagates_scope = False

    def __init__(
        self,
        *,
        detector: SemanticConceptDetector,
        guard_action: Literal["refuse", "guard_prompt", "both"] = "guard_prompt",
        apply_at: Literal["every_agent_input", "external_prompt_only"] = "every_agent_input",
        implementation: Literal["template", "sft_checkpoint"] = "template",
        inspect_query: bool = True,
        sanitizer: Sanitizer | None = None,
    ) -> None:
        if implementation != "template":
            raise NotImplementedError(
                "dragon.implementation='sft_checkpoint' requires the released DRAGON "
                "detector/guard weights, which the public repository does not ship. "
                "Set implementation=template and report the baseline as DRAGON-style."
            )
        self.detector = detector
        self.guard_action = guard_action
        self.apply_at = apply_at
        self.implementation = implementation
        # False under the graph_flow protocol, where the request gate is held constant
        # across arms so that what is measured is propagation, not request filtering.
        self.inspect_query = inspect_query
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
        result = self.detector.score(ctx.combined_text(include_query=self.inspect_query))
        if not result.fired:
            return NodeInputVerdict(
                inputs=tuple(ctx.inputs),
                memory_texts=tuple(ctx.memory_texts),
                score=result.score,
                reason=f"node-local detector below threshold ({result.score:.3f})",
            )

        self.counters.node_input_fired += 1
        suffix = GUARD_INSTRUCTION if self.guard_action in ("guard_prompt", "both") else None
        forced: str | None = None
        if self.guard_action in ("refuse", "both"):
            forced, _certificate = self.sanitizer.safe_refusal(
                result.forget_ids, detector_version=self.detector.version
            )
        return NodeInputVerdict(
            inputs=tuple(ctx.inputs),
            memory_texts=tuple(ctx.memory_texts),
            forget_ids=result.forget_ids,
            score=result.score,
            fired=True,
            forced_output=forced,
            guard_system_suffix=suffix,
            reason=f"node-local scope match {list(result.forget_ids)} ({result.score:.3f})",
        )

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
            "detector_version": self.detector.version,
            **self.counters.to_dict(),
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
