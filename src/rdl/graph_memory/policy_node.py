"""Scope tags on memory nodes.

The tag lives in ``MemoryNode.meta['policy']`` rather than in a new node class, so the
existing store, DAG, blocklists and invariant certificates keep working unchanged. The
authoritative copy is the ``ScopeIndex``; the node metadata is the durable one that
survives a snapshot round-trip.

A tag is never a substitute for semantics. ``RetrievalGuard`` re-scans untagged nodes,
because an attacker (or a bug) that drops the tag must not thereby unblock the content.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..memory.node import MemoryNode

__all__ = ["POLICY_META_KEY", "PolicyRecord", "node_forget_ids", "tag_node"]

POLICY_META_KEY = "policy"


@dataclass(frozen=True)
class PolicyRecord:
    forget_ids: tuple[str, ...] = ()
    source_envelope_ids: tuple[str, ...] = ()
    source_nodes: tuple[str, ...] = ()
    detector_version: str = ""
    score: float = 0.0
    release_status: str = "pass"
    reason: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "forget_ids": list(self.forget_ids),
            "source_envelope_ids": list(self.source_envelope_ids),
            "source_nodes": list(self.source_nodes),
            "detector_version": self.detector_version,
            "score": round(self.score, 6),
            "release_status": self.release_status,
            "reason": self.reason,
            **self.meta,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> PolicyRecord:
        known = {
            "forget_ids",
            "source_envelope_ids",
            "source_nodes",
            "detector_version",
            "score",
            "release_status",
            "reason",
        }
        return cls(
            forget_ids=tuple(payload.get("forget_ids", ())),
            source_envelope_ids=tuple(payload.get("source_envelope_ids", ())),
            source_nodes=tuple(payload.get("source_nodes", ())),
            detector_version=str(payload.get("detector_version", "")),
            score=float(payload.get("score", 0.0)),
            release_status=str(payload.get("release_status", "pass")),
            reason=str(payload.get("reason", "")),
            meta={k: v for k, v in payload.items() if k not in known},
        )


def tag_node(node: MemoryNode, record: PolicyRecord) -> MemoryNode:
    """Attach (or merge into) a node's policy record. Tags only ever grow."""
    existing = node.meta.get(POLICY_META_KEY)
    if isinstance(existing, dict):
        previous = PolicyRecord.from_dict(existing)
        record = PolicyRecord(
            forget_ids=tuple(sorted(set(previous.forget_ids) | set(record.forget_ids))),
            source_envelope_ids=tuple(
                sorted(set(previous.source_envelope_ids) | set(record.source_envelope_ids))
            ),
            source_nodes=tuple(sorted(set(previous.source_nodes) | set(record.source_nodes))),
            detector_version=record.detector_version or previous.detector_version,
            score=max(previous.score, record.score),
            release_status=record.release_status,
            reason=record.reason or previous.reason,
            meta={**previous.meta, **record.meta},
        )
    meta = dict(node.meta)
    meta[POLICY_META_KEY] = record.to_dict()
    node.meta = meta
    return node


def node_forget_ids(node: MemoryNode) -> tuple[str, ...]:
    payload = node.meta.get(POLICY_META_KEY)
    if not isinstance(payload, dict):
        return ()
    return tuple(sorted(str(x) for x in payload.get("forget_ids", ())))


def inherited_ids(
    *,
    detected: Sequence[str],
    parent_ids: Sequence[str],
    node_scopes: dict[str, Sequence[str]],
) -> tuple[str, ...]:
    """``S(write) = D(content) ∪ ⋃ S(parent)`` over store nodes."""
    acc = set(detected)
    for parent in parent_ids:
        acc |= set(node_scopes.get(parent, ()))
    return tuple(sorted(acc))
