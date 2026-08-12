"""The information-provenance envelope.

The execution graph says who talks to whom. The *provenance* graph says where content
came from, and it is the object the defence acts on. Every message, tool response,
memory read and agent output moving through the system is wrapped in one of these.

The load-bearing rule:

    S(x) = D(content_x) ∪ ⋃_{p ∈ parents(x)} S(p)

A Forget ID attaches to the **message**, never permanently to the agent. An agent that
consumed ``author_17`` content emits ``author_17``-tagged output; it does not become a
tainted agent for the rest of the episode, because that would make the defence a coarse
kill-switch whose utility cost is not attributable to the leakage it prevented.

A safe refusal may be released while keeping its Forget IDs in provenance. Dropping the
tag on release is the bug that makes a refusal look like unrelated clean content and
lets its descendants inherit nothing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Literal

__all__ = ["Envelope", "EnvelopeKind", "ReleaseStatus", "content_sha256"]

EnvelopeKind = Literal[
    "user_query",
    "agent_output",
    "tool_response",
    "memory_read",
    "injected_control",
]

# ``pass`` and ``sanitized`` are released; ``quarantined`` is withheld from the consumer
# but retained in evidence; ``blocked`` is replaced by a refusal on the same edge. None
# of them removes the edge — see PROTOCOL_v1 §3.
ReleaseStatus = Literal["pass", "sanitized", "quarantined", "blocked"]


def content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Envelope:
    """One policy-carrying object in the provenance graph."""

    kind: EnvelopeKind
    content: str
    source_node: str
    dest_node: str | None = None
    # Envelope ids of the objects this content was derived from.
    parent_ids: tuple[str, ...] = ()
    # Scopes carried by this object: detected on its own content, plus inherited.
    forget_ids: tuple[str, ...] = ()
    # Scopes the detector found in *this* content alone. Kept separate from `forget_ids`
    # so a report can distinguish "this message is about author_17" from "this message
    # descends from something that was".
    detected_forget_ids: tuple[str, ...] = ()
    semantic_scope_score: float = 0.0
    release_status: ReleaseStatus = "pass"
    decision_reason: str = ""
    sanitization_certificate: dict | None = None
    meta: dict = field(default_factory=dict)

    # --------------------------------------------------------------------- identity --

    @property
    def envelope_id(self) -> str:
        """Deterministic id over the fields that define the object.

        Not a uuid: two runs of the same configuration must produce identical evidence
        files, and a random id would make every shard hash differ for no reason.
        """
        payload = "\0".join(
            (
                self.kind,
                self.source_node,
                self.dest_node or "-",
                content_sha256(self.content),
                ",".join(self.parent_ids),
                ",".join(sorted(self.forget_ids)),
                self.release_status,
            )
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]

    @property
    def content_hash(self) -> str:
        return content_sha256(self.content)

    @property
    def released(self) -> bool:
        """Whether a consumer may see this object's content at all."""
        return self.release_status in ("pass", "sanitized", "blocked")

    # ------------------------------------------------------------------- derivation --

    def derive(
        self,
        *,
        kind: EnvelopeKind,
        content: str,
        source_node: str,
        dest_node: str | None = None,
        parents: tuple[Envelope, ...] = (),
        detected: tuple[str, ...] = (),
        score: float = 0.0,
        meta: dict | None = None,
    ) -> Envelope:
        """Build a child envelope inheriting this one's scopes. See module docstring."""
        return derive_envelope(
            kind=kind,
            content=content,
            source_node=source_node,
            dest_node=dest_node,
            parents=(self, *parents),
            detected=detected,
            score=score,
            meta=meta,
        )

    def with_decision(
        self,
        *,
        status: ReleaseStatus,
        reason: str,
        content: str | None = None,
        certificate: dict | None = None,
        added_forget_ids: tuple[str, ...] = (),
        score: float | None = None,
    ) -> Envelope:
        """Apply an enforcement decision, preserving provenance.

        `content` is replaced for a sanitize/block, and the Forget IDs are *kept*: a
        refusal that descends from ``author_17`` still carries ``author_17``.
        """
        return Envelope(
            kind=self.kind,
            content=self.content if content is None else content,
            source_node=self.source_node,
            dest_node=self.dest_node,
            parent_ids=self.parent_ids,
            forget_ids=tuple(sorted(set(self.forget_ids) | set(added_forget_ids))),
            detected_forget_ids=self.detected_forget_ids,
            semantic_scope_score=self.semantic_scope_score if score is None else score,
            release_status=status,
            decision_reason=reason,
            sanitization_certificate=certificate,
            meta=dict(self.meta),
        )

    def to(self, dest_node: str) -> Envelope:
        """Address a copy of this object at `dest_node`, keeping everything else."""
        return Envelope(
            kind=self.kind,
            content=self.content,
            source_node=self.source_node,
            dest_node=dest_node,
            parent_ids=self.parent_ids,
            forget_ids=self.forget_ids,
            detected_forget_ids=self.detected_forget_ids,
            semantic_scope_score=self.semantic_scope_score,
            release_status=self.release_status,
            decision_reason=self.decision_reason,
            sanitization_certificate=self.sanitization_certificate,
            meta=dict(self.meta),
        )

    # ------------------------------------------------------------------- reporting --

    def to_dict(self, *, with_content: bool = True) -> dict:
        record = {
            "envelope_id": self.envelope_id,
            "kind": self.kind,
            "source_node": self.source_node,
            "dest_node": self.dest_node,
            "parent_ids": list(self.parent_ids),
            "forget_ids": list(self.forget_ids),
            "detected_forget_ids": list(self.detected_forget_ids),
            "semantic_scope_score": round(self.semantic_scope_score, 6),
            "release_status": self.release_status,
            "decision_reason": self.decision_reason,
            "sanitization_certificate": self.sanitization_certificate,
            "content_sha256": self.content_hash,
        }
        if with_content:
            record["content"] = self.content
        return record


def derive_envelope(
    *,
    kind: EnvelopeKind,
    content: str,
    source_node: str,
    dest_node: str | None = None,
    parents: tuple[Envelope, ...] = (),
    detected: tuple[str, ...] = (),
    score: float = 0.0,
    meta: dict | None = None,
) -> Envelope:
    """Construct an envelope whose scope set is ``detected ∪ ⋃ parent scopes``."""
    inherited: set[str] = set(detected)
    for parent in parents:
        inherited |= set(parent.forget_ids)
    return Envelope(
        kind=kind,
        content=content,
        source_node=source_node,
        dest_node=dest_node,
        parent_ids=tuple(p.envelope_id for p in parents),
        forget_ids=tuple(sorted(inherited)),
        detected_forget_ids=tuple(sorted(set(detected))),
        semantic_scope_score=score,
        meta=dict(meta or {}),
    )
