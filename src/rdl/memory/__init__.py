"""Memory subsystem: the SBU model, implemented honestly.

The whole argument of the paper lives in `parent_ids` semantics (see `node.py`) and in
`invariants.py`, which turns "SBU's invariants do not imply its property (iii)" from an
assertion into a machine-checked, per-node, re-runnable fact.
"""

from __future__ import annotations

from .blocklist import (
    BlockDecision,
    Blocklist,
    IDBlocklist,
    NoBlocklist,
    SemanticBlocklist,
    build_blocklist,
)
from .derivation import DerivationDAG, PruneResult
from .index import (
    Embedder,
    FaissFlat,
    HashingEmbedder,
    NumpyBruteForce,
    VectorIndex,
    build_index,
)
from .invariants import (
    Inv1Report,
    Inv2Report,
    InvariantCertificate,
    certify,
    check_invariant_1,
    check_invariant_2,
)
from .node import MemoryNode, SourceKind
from .store import MemoryStore, RetrievalResult, StoreSnapshot

__all__ = [
    "BlockDecision",
    "Blocklist",
    "DerivationDAG",
    "Embedder",
    "FaissFlat",
    "HashingEmbedder",
    "IDBlocklist",
    "Inv1Report",
    "Inv2Report",
    "InvariantCertificate",
    "MemoryNode",
    "MemoryStore",
    "NoBlocklist",
    "NumpyBruteForce",
    "PruneResult",
    "RetrievalResult",
    "SemanticBlocklist",
    "SourceKind",
    "StoreSnapshot",
    "VectorIndex",
    "build_blocklist",
    "build_index",
    "certify",
    "check_invariant_1",
    "check_invariant_2",
]
