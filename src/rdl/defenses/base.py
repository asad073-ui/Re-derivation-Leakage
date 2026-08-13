"""The defence interface: five enforcement points, one protocol.

Every arm implements all five. A defence that does not act at a point returns the
pass-through verdict rather than the executor skipping the call, so that the trace
records "the guard looked and allowed it" instead of a silent absence — which is what
lets detector false-positive and false-negative rates be computed per surface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..graph.envelope import Envelope, ReleaseStatus

__all__ = [
    "Defense",
    "EdgeContext",
    "EdgeVerdict",
    "FinalContext",
    "FinalVerdict",
    "NodeInputContext",
    "NodeInputVerdict",
    "RetrievalContext",
    "RetrievalVerdict",
    "WriteContext",
    "WriteVerdict",
]


# ------------------------------------------------------------------------ contexts --


@dataclass(frozen=True)
class NodeInputContext:
    """Everything one agent is about to see, before it is turned into a prompt."""

    node_id: str
    role: str
    depth: int
    question: str
    inputs: tuple[Envelope, ...] = ()
    memory_texts: tuple[str, ...] = ()
    # Scopes already known to be carried by this node's inputs.
    inherited_forget_ids: tuple[str, ...] = ()

    def combined_text(self, *, include_query: bool = True) -> str:
        """Every incoming message + retrieved memory, and optionally the query.

        This is the object the accumulated-input detector scores. Two clues that are
        individually below threshold are jointly above it precisely here.

        ``include_query=False`` is the ``graph_flow`` protocol: the request gate is held
        constant across arms and the guard acts only on what the graph itself carries.
        Without it the forget question — which is one of the detector's own prototypes —
        fires at the root of every guarded arm and nothing downstream is ever exercised.
        """
        parts = [*(e.content for e in self.inputs), *self.memory_texts]
        if include_query:
            parts.insert(0, self.question)
        return "\n".join(p for p in parts if p.strip())


@dataclass(frozen=True)
class EdgeContext:
    src: str
    dst: str
    depth: int
    envelope: Envelope
    question: str = ""


@dataclass(frozen=True)
class WriteContext:
    node_id: str
    depth: int
    envelope: Envelope
    question: str = ""
    # Node ids in the store that were in the producing agent's retrieval context.
    retrieved_node_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalContext:
    node_id: str
    depth: int
    query: str
    # ``(store_node_id, content, tagged_forget_ids)`` for each candidate.
    candidates: tuple[tuple[str, str, tuple[str, ...]], ...] = ()


@dataclass(frozen=True)
class FinalContext:
    node_id: str
    envelope: Envelope
    question: str = ""


# ------------------------------------------------------------------------ verdicts --


@dataclass(frozen=True)
class NodeInputVerdict:
    """What the agent may see, and whether it is allowed to generate at all."""

    inputs: tuple[Envelope, ...]
    memory_texts: tuple[str, ...]
    # The scopes THIS DECISION WAS MADE ON. Always recorded in the trace.
    forget_ids: tuple[str, ...] = ()
    # The scopes this decision FORWARDS onto what the node produces. ``None`` means "the
    # same as `forget_ids`", which is what every defence did before the consume/forward
    # split and is therefore the compatible default. An arm that enforces a tag where it
    # finds one but never spreads it returns ``()`` here with a non-empty `forget_ids`.
    propagated_forget_ids: tuple[str, ...] | None = None
    score: float = 0.0
    fired: bool = False
    # When set, the node does NOT generate; this text becomes its output. That is the
    # deterministic form of a reasoning guard: on a checkpoint that ignores the guard
    # instruction, a prompt-only baseline would be a no-op and the comparison would be
    # against nothing at all. See docs/graph_unlearning/BASELINES.md.
    forced_output: str | None = None
    guard_system_suffix: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class EdgeVerdict:
    envelope: Envelope
    status: ReleaseStatus = "pass"
    reason: str = ""
    score: float = 0.0
    forget_ids: tuple[str, ...] = ()
    edge_preserved: bool = True


@dataclass(frozen=True)
class WriteVerdict:
    allowed: bool
    envelope: Envelope
    reason: str = ""
    score: float = 0.0
    forget_ids: tuple[str, ...] = ()
    # What the COMMITTED NODE is tagged with, as distinct from what the write decision
    # was made on. ``None`` means "the same as `forget_ids`". A stored tag is the longest
    # lived piece of state in the system — it outlives the episode and is what a later
    # retrieval enforces — so an arm that does not forward must not write one.
    propagated_forget_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class RetrievalVerdict:
    allowed_node_ids: tuple[str, ...] = ()
    withheld_node_ids: tuple[str, ...] = ()
    rescan_withheld_node_ids: tuple[str, ...] = ()
    reason: str = ""


@dataclass(frozen=True)
class FinalVerdict:
    text: str
    status: ReleaseStatus = "pass"
    reason: str = ""
    score: float = 0.0
    forget_ids: tuple[str, ...] = ()


# ----------------------------------------------------------------------- protocol --


@runtime_checkable
class Defense(Protocol):
    name: str

    # Whether this defence maintains cross-object policy state. Recorded in the manifest
    # so a report cannot describe a node-local baseline as a propagating one.
    #
    # READ-ONLY on purpose (GU-0032). It used to be a settable attribute, which is what
    # allowed `GraphForgetDefense` to declare a class-level `propagates_scope = True` for
    # every variant of itself — including the ablations built specifically not to
    # propagate. A defence must DERIVE this from its own configuration, so the protocol
    # asks for a property and a constant no longer satisfies it silently.
    @property
    def propagates_scope(self) -> bool: ...

    def on_node_input(self, ctx: NodeInputContext) -> NodeInputVerdict: ...
    def on_edge(self, ctx: EdgeContext) -> EdgeVerdict: ...
    def on_memory_write(self, ctx: WriteContext) -> WriteVerdict: ...
    def on_retrieval(self, ctx: RetrievalContext) -> RetrievalVerdict: ...
    def on_final_output(self, ctx: FinalContext) -> FinalVerdict: ...
    def stats(self) -> dict: ...


@dataclass
class DefenseCounters:
    """Per-surface tallies, carried by every concrete defence for the report."""

    node_input_calls: int = 0
    node_input_fired: int = 0
    edge_calls: int = 0
    edge_blocked: int = 0
    edge_sanitized: int = 0
    write_calls: int = 0
    write_blocked: int = 0
    retrieval_calls: int = 0
    retrieval_withheld: int = 0
    final_calls: int = 0
    final_blocked: int = 0
    accumulated_only_hits: int = 0
    inherited_only_hits: int = 0
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != "extra"}
        out.update(self.extra)
        return out


def passthrough_inputs(ctx: NodeInputContext) -> NodeInputVerdict:
    return NodeInputVerdict(
        inputs=tuple(ctx.inputs),
        memory_texts=tuple(ctx.memory_texts),
        reason="no defence configured",
    )


def passthrough_retrieval(
    candidates: Sequence[tuple[str, str, tuple[str, ...]]],
) -> RetrievalVerdict:
    return RetrievalVerdict(
        allowed_node_ids=tuple(node_id for node_id, _text, _ids in candidates),
        reason="no defence configured",
    )
