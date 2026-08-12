"""Typed graph events and the per-trajectory trace.

Everything a defence decided is recorded, including the decisions that let content
through. A trace that only recorded blocks could not distinguish "the detector never
fired" from "the detector fired and the policy allowed it", which is the difference
between a false negative and a calibration choice.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from ..logging_utils import dumps_canonical
from .envelope import Envelope

__all__ = [
    "DetectorCall",
    "EdgeDecisionEvent",
    "FinalReleaseEvent",
    "GraphEvent",
    "GraphTrace",
    "MemoryReadEvent",
    "MemoryWriteEvent",
    "NodeExecutedEvent",
]


class _Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: str
    node_id: str = ""
    depth: int = 0

    def to_dict(self) -> dict:
        return self.model_dump(mode="json")


class NodeExecutedEvent(_Event):
    kind: Literal["node_executed"] = "node_executed"
    role: str = ""
    prompt_sha256: str = ""
    input_envelope_ids: list[str] = Field(default_factory=list)
    output_envelope_id: str = ""
    output_sha256: str = ""
    generation_seed: int | None = None
    model_revision: str | None = None
    cached: bool = False
    generated: bool = True
    abstained: bool = False


class DetectorCall(_Event):
    kind: Literal["detector_call"] = "detector_call"
    surface: str = ""  # node_input | edge | write | retrieval | final | accumulated
    detector_version: str = ""
    score: float = 0.0
    threshold: float = 0.0
    fired: bool = False
    matched_forget_ids: list[str] = Field(default_factory=list)
    target_sha256: str = ""


class EdgeDecisionEvent(_Event):
    kind: Literal["edge_decision"] = "edge_decision"
    src: str = ""
    dst: str = ""
    defense: str = ""
    release_status: str = "pass"
    reason: str = ""
    forget_ids: list[str] = Field(default_factory=list)
    inherited_forget_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    envelope_id: str = ""
    # True when the edge exists and carried a payload decision. False only for the
    # edge-cut ablation, which is the ONE arm allowed to change the topology.
    edge_preserved: bool = True


class MemoryWriteEvent(_Event):
    kind: Literal["memory_write"] = "memory_write"
    defense: str = ""
    allowed: bool = False
    reason: str = ""
    node_ref: str | None = None
    content_sha256: str = ""
    forget_ids: list[str] = Field(default_factory=list)
    inherited_forget_ids: list[str] = Field(default_factory=list)
    parent_node_ids: list[str] = Field(default_factory=list)
    score: float = 0.0


class MemoryReadEvent(_Event):
    kind: Literal["memory_read"] = "memory_read"
    defense: str = ""
    query_sha256: str = ""
    returned_node_ids: list[str] = Field(default_factory=list)
    withheld_node_ids: list[str] = Field(default_factory=list)
    rescan_withheld_node_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class FinalReleaseEvent(_Event):
    kind: Literal["final_release"] = "final_release"
    defense: str = ""
    release_status: str = "pass"
    reason: str = ""
    forget_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    output_sha256: str = ""


GraphEvent = (
    NodeExecutedEvent
    | DetectorCall
    | EdgeDecisionEvent
    | MemoryWriteEvent
    | MemoryReadEvent
    | FinalReleaseEvent
)


class GraphTrace:
    """Append-only event log for one trajectory."""

    def __init__(
        self,
        *,
        trajectory_id: str,
        item_id: str,
        concept_id: str,
        sample_id: int,
        arm: str,
        topology: str,
        challenge: str,
    ) -> None:
        self.trajectory_id = trajectory_id
        self.item_id = item_id
        self.concept_id = concept_id
        self.sample_id = sample_id
        self.arm = arm
        self.topology = topology
        self.challenge = challenge
        self.events: list[_Event] = []
        self.envelopes: list[Envelope] = []
        self.final_text: str = ""
        self.final_status: str = "pass"

    # -------------------------------------------------------------------- mutation --

    def append(self, event: _Event) -> None:
        self.events.append(event)

    def record_envelope(self, envelope: Envelope) -> Envelope:
        self.envelopes.append(envelope)
        return envelope

    # --------------------------------------------------------------------- queries --

    def of_kind(self, kind: str) -> list[_Event]:
        return [e for e in self.events if e.kind == kind]

    def node_outputs(self) -> list[Envelope]:
        return [e for e in self.envelopes if e.kind == "agent_output"]

    def edge_payloads(self) -> list[Envelope]:
        """Content actually delivered across an edge, after enforcement."""
        return [
            e
            for e in self.envelopes
            if e.kind == "agent_output" and e.dest_node is not None and e.released
        ]

    def released_texts(self) -> list[str]:
        """Every string that actually reached a consumer.

        This is the population ``edge_leak`` is computed over: content that was
        generated and then blocked never reached anyone, and counting it as leakage
        would make every defence look useless by construction.
        """
        return [e.content for e in self.edge_payloads()]

    def to_dict(self, *, with_content: bool = True) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "item_id": self.item_id,
            "concept_id": self.concept_id,
            "sample_id": self.sample_id,
            "arm": self.arm,
            "topology": self.topology,
            "challenge": self.challenge,
            "final_status": self.final_status,
            "final_sha256": hashlib.sha256(self.final_text.encode("utf-8")).hexdigest(),
            "events": [e.to_dict() for e in self.events],
            "envelopes": [e.to_dict(with_content=with_content) for e in self.envelopes],
        }

    def digest(self) -> str:
        """Stable hash of what the trajectory DID. Two identical runs must agree.

        ``cached`` is excluded. Whether a response came from the model or from a
        byte-identical cache entry is provenance about how it was obtained, not about
        what happened — and including it would make the second run of the same arm
        disagree with the first, which is precisely the property this hash exists to
        check.
        """
        payload = self.to_dict(with_content=False)
        for event in payload["events"]:
            event.pop("cached", None)
        return hashlib.sha256(dumps_canonical(payload).encode("utf-8")).hexdigest()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"GraphTrace({self.trajectory_id} arm={self.arm} "
            f"{len(self.events)} events, {len(self.envelopes)} envelopes)"
        )


def event_summary(events: list[Any]) -> dict[str, int]:
    """Count events by kind. Used in the run performance report."""
    counts: dict[str, int] = {}
    for event in events:
        kind = getattr(event, "kind", "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return counts
