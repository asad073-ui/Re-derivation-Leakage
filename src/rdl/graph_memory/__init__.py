"""Policy-carrying persistent memory for the graph study.

Wraps ``rdl.memory`` rather than forking it. The store, the derivation DAG, the
blocklists and the invariant certificates are the two-agent work's strongest pieces and
are reused verbatim; what is added here is the scope tag that travels with a written
node and the guard that consults it at retrieval.
"""

from __future__ import annotations

from .derivation_closure import tag_closure
from .policy_node import POLICY_META_KEY, PolicyRecord, node_forget_ids, tag_node
from .retrieval_guard import RetrievalGuard
from .scope_index import ScopeIndex
from .staged_store import StagedMemory, WriteCandidate

__all__ = [
    "POLICY_META_KEY",
    "PolicyRecord",
    "RetrievalGuard",
    "ScopeIndex",
    "StagedMemory",
    "WriteCandidate",
    "node_forget_ids",
    "tag_closure",
    "tag_node",
]
