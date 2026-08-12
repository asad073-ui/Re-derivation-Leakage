"""The unguarded arm. Every surface passes, and every pass is recorded."""

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

__all__ = ["NoDefense"]


class NoDefense:
    """MA-LEAK / MA-CONTROL / SA. No detection, no enforcement."""

    name = "none"
    propagates_scope = False

    def __init__(self) -> None:
        self.counters = DefenseCounters()

    def on_node_input(self, ctx: NodeInputContext) -> NodeInputVerdict:
        self.counters.node_input_calls += 1
        return NodeInputVerdict(
            inputs=tuple(ctx.inputs), memory_texts=tuple(ctx.memory_texts), reason="no defence"
        )

    def on_edge(self, ctx: EdgeContext) -> EdgeVerdict:
        self.counters.edge_calls += 1
        return EdgeVerdict(envelope=ctx.envelope, status="pass", reason="no defence")

    def on_memory_write(self, ctx: WriteContext) -> WriteVerdict:
        self.counters.write_calls += 1
        return WriteVerdict(allowed=True, envelope=ctx.envelope, reason="no defence")

    def on_retrieval(self, ctx: RetrievalContext) -> RetrievalVerdict:
        self.counters.retrieval_calls += 1
        return RetrievalVerdict(
            allowed_node_ids=tuple(node_id for node_id, _t, _f in ctx.candidates),
            reason="no defence",
        )

    def on_final_output(self, ctx: FinalContext) -> FinalVerdict:
        self.counters.final_calls += 1
        return FinalVerdict(text=ctx.envelope.content, status="pass", reason="no defence")

    def stats(self) -> dict:
        return {"defense": self.name, **self.counters.to_dict()}
