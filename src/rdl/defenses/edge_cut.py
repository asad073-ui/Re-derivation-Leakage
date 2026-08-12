"""The edge-cut ablation. Not a proposed method.

Removing an edge when a concept is detected does lower leakage, and it does so partly by
deleting communication rather than by containing it. That confound is the reason the
primary method preserves every edge and acts on payloads. This ablation exists to
measure the confound: it reports the leakage reduction *and* the collaboration-utility
cost of cutting, so the paper can quantify what the honest comparison avoided.

``edge_preserved=False`` on its decisions is what makes any report built on this arm
label itself as a topology-changing ablation.
"""

from __future__ import annotations

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
from .semantic_detector import SemanticConceptDetector

__all__ = ["EdgeCutDefense"]


class EdgeCutDefense:
    name = "edge_cut"
    propagates_scope = False
    changes_topology = True

    def __init__(self, *, detector: SemanticConceptDetector) -> None:
        self.detector = detector
        self.counters = DefenseCounters()

    def on_node_input(self, ctx: NodeInputContext) -> NodeInputVerdict:
        self.counters.node_input_calls += 1
        return NodeInputVerdict(
            inputs=tuple(ctx.inputs),
            memory_texts=tuple(ctx.memory_texts),
            reason="edge_cut acts only on edges",
        )

    def on_edge(self, ctx: EdgeContext) -> EdgeVerdict:
        self.counters.edge_calls += 1
        result = self.detector.score(ctx.envelope.content)
        if not result.fired:
            return EdgeVerdict(envelope=ctx.envelope, status="pass", score=result.score)
        self.counters.edge_blocked += 1
        # The consumer receives nothing at all: no payload, no refusal, no notice.
        return EdgeVerdict(
            envelope=ctx.envelope.with_decision(
                status="quarantined",
                reason="edge removed",
                content="",
                added_forget_ids=result.forget_ids,
                score=result.score,
            ),
            status="quarantined",
            reason=f"edge {ctx.src}->{ctx.dst} removed on scope match {list(result.forget_ids)}",
            score=result.score,
            forget_ids=result.forget_ids,
            edge_preserved=False,
        )

    def on_memory_write(self, ctx: WriteContext) -> WriteVerdict:
        self.counters.write_calls += 1
        return WriteVerdict(
            allowed=True, envelope=ctx.envelope, reason="edge_cut does not guard writes"
        )

    def on_retrieval(self, ctx: RetrievalContext) -> RetrievalVerdict:
        self.counters.retrieval_calls += 1
        return RetrievalVerdict(
            allowed_node_ids=tuple(node_id for node_id, _t, _f in ctx.candidates),
            reason="edge_cut does not guard retrieval",
        )

    def on_final_output(self, ctx: FinalContext) -> FinalVerdict:
        self.counters.final_calls += 1
        return FinalVerdict(
            text=ctx.envelope.content, status="pass", reason="edge_cut does not guard output"
        )

    def stats(self) -> dict:
        return {
            "defense": self.name,
            "changes_topology": True,
            "detector_version": self.detector.version,
            **self.counters.to_dict(),
        }
