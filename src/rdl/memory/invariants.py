"""The paper's core artifact.

SBU's memory-pathway argument rests on two invariants:

  (I1)  No retrieval path returns a node whose id is in the blocklist B.
  (I2)  Every node reachable from a deleted node is marked outdated / decremented.

and it claims these imply a third property:

  (iii) The forgotten content is no longer recoverable from the system.

Our claim is that (I1) and (I2) do **not** imply (iii). The reason is structural: both
invariants quantify over *node identity and derivation edges*. A node written from an
agent's parametric answer has a fresh id and an empty `parent_ids`, so it is outside
the scope of both invariants by construction, no matter what its content says.

This module does not assert that. It **certifies** it, per node, in a form a reviewer
can re-verify offline from the witness alone:

    certify(...) -> InvariantCertificate
        inv1_satisfied              both invariants genuinely hold system-wide
        inv2_satisfied
        path_to_any_blocked_node    None for a laundered node - and the concrete
                                    counter-example path when it is not None
        witness                     everything needed to re-check without the process

Every recovered forget-set item in the results table carries its certificate.
`tests/unit/test_invariants.py::test_laundered_node_passes_both_invariants` is the
golden test: if it goes red, the paper's central mechanism argument is wrong.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

from .blocklist import Blocklist, NoBlocklist
from .derivation import DerivationDAG
from .node import MemoryNode
from .store import MemoryStore

__all__ = [
    "Inv1Report",
    "Inv2Report",
    "InvariantCertificate",
    "certify",
    "certify_all",
    "check_invariant_1",
    "check_invariant_2",
]

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Inv1Report:
    """(I1) No retrieval path returns a blocked node."""

    satisfied: bool
    n_blocked: int
    n_indexed: int
    n_returnable: int
    # Blocked ids returned by an adversarial probe that queries with their own content.
    # Non-empty => I1 violated. This is the ONLY violation mode.
    leaked_via_probe: list[str] = field(default_factory=list)
    # Blocked ids physically present in the index. INFORMATIONAL, NOT a violation:
    # tombstones stay indexed by design and the blocklist is what suppresses them.
    still_indexed: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Inv2Report:
    """(I2) Everything derived from a deleted node is marked outdated / decremented."""

    satisfied: bool
    deleted_ids: list[str] = field(default_factory=list)
    closure_size: int = 0
    unmarked: list[str] = field(default_factory=list)  # in closure, still live => violated
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InvariantCertificate:
    """Per-node, machine-checked, re-runnable certificate.

    A node that carries ``inv1_satisfied and inv2_satisfied and
    path_to_any_blocked_node is None`` while *containing forgotten content* is a
    laundered node: the system's own invariants declare it clean.
    """

    node_id: str
    inv1_satisfied: bool
    inv2_satisfied: bool
    path_to_any_blocked_node: list[str] | None
    witness: dict

    # ------------------------------------------------------------------ derived --

    @property
    def invariants_hold(self) -> bool:
        return self.inv1_satisfied and self.inv2_satisfied

    @property
    def is_laundered_candidate(self) -> bool:
        """Structurally clean: both invariants hold and no derivation path to blocked content.

        "Candidate" because whether the node actually *contains* forgotten content is a
        content question, answered by `rdl.eval.containment`, not by this module.
        Keeping the two separate is what stops the metric from assuming its conclusion.
        """
        return self.invariants_hold and self.path_to_any_blocked_node is None

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "node_id": self.node_id,
            "inv1_satisfied": self.inv1_satisfied,
            "inv2_satisfied": self.inv2_satisfied,
            "path_to_any_blocked_node": self.path_to_any_blocked_node,
            "invariants_hold": self.invariants_hold,
            "is_laundered_candidate": self.is_laundered_candidate,
            "witness": self.witness,
        }

    def explain(self) -> str:  # pragma: no cover - human-facing
        if self.path_to_any_blocked_node:
            path = " -> ".join(p[:8] for p in self.path_to_any_blocked_node)
            return f"{self.node_id[:8]}: derivation path to blocked content: {path}"
        if not self.invariants_hold:
            return f"{self.node_id[:8]}: invariants violated (I1={self.inv1_satisfied} I2={self.inv2_satisfied})"
        return (
            f"{self.node_id[:8]}: both invariants hold, no derivation path to any blocked "
            "node — structurally clean"
        )


# ---------------------------------------------------------------------------------
# I1
# ---------------------------------------------------------------------------------


def check_invariant_1(
    store: MemoryStore,
    blocklist: Blocklist | None = None,
    *,
    probe: bool = True,
    probe_k: int = 10,
) -> Inv1Report:
    """Check that no retrieval path returns a blocked node.

    The check is **behavioural, not structural**. Blocked nodes remain physically in the
    index as tombstones — see `MemoryNode.returnable` — precisely so that the retrieval
    code path has to enforce the blocklist rather than being handed a pre-sanitised
    index. A structural "is the id absent from the index" check would be vacuous here,
    and would let an enforcement bug through silently.

    So I1 is checked by an **adversarial self-content probe**: for every blocked node,
    query with that node's own content — the strongest possible query for surfacing it —
    and confirm `store.retrieve` withholds it. Anything returned is a violation and is
    reported with the id.

    `still_indexed` is reported for transparency but is NOT a violation. A tombstone in
    the index is the design; a tombstone that comes back out is the bug.
    """
    bl = blocklist if blocklist is not None else NoBlocklist()
    blocked = set(bl.blocked_ids())

    # A content-based blocklist has no id set; derive the effective one by evaluating it.
    if not blocked:
        blocked = {n.node_id for n in store.all_nodes(include_deleted=True) if bl.blocks(n).blocked}

    indexed = store.indexed_ids()
    returnable = store.returnable_ids(bl)
    violations: list[str] = []
    leaked_via_probe: list[str] = []

    if probe:
        for nid in sorted(blocked):
            node = store.get(nid)
            if node is None:
                continue
            res = store.retrieve(node.content, k=probe_k, blocklist=bl)
            if nid in res.node_ids:
                leaked_via_probe.append(nid)
                violations.append(
                    f"I1: blocked node {nid} was returned by a self-content probe query"
                )
    else:
        # Without the probe, fall back to the set check. Weaker, and says so.
        for nid in sorted(blocked & returnable):
            violations.append(f"I1: blocked node {nid} is in the returnable set")

    return Inv1Report(
        satisfied=not violations,
        n_blocked=len(blocked),
        n_indexed=len(indexed),
        n_returnable=len(returnable),
        leaked_via_probe=leaked_via_probe,
        still_indexed=sorted(blocked & indexed),
        violations=violations,
    )


# ---------------------------------------------------------------------------------
# I2
# ---------------------------------------------------------------------------------


def check_invariant_2(
    store: MemoryStore,
    dag: DerivationDAG | None = None,
    deleted_ids: Iterable[str] | None = None,
) -> Inv2Report:
    """Check that every node reachable from a deleted node is marked outdated.

    A node in the closure may legitimately stay live in exactly one case: it still has a
    **surviving parent** — one that is neither deleted nor outdated — so its provenance
    is not entirely gone. Anything else in the closure must be marked outdated.

    That is the strict reading of SBU's "decrement and mark outdated", and it is the one
    worth checking: a node whose every ancestor is dead but which is still live and
    retrievable is exactly the failure mode invariant 2 is supposed to exclude.
    """
    g = dag if dag is not None else store.dag
    if deleted_ids is None:
        deleted = sorted(n.node_id for n in store.nodes.values() if n.deleted)
    else:
        deleted = sorted(set(deleted_ids))

    violations: list[str] = []
    unmarked: list[str] = []
    closure: set[str] = set()

    for did in deleted:
        for cid in sorted(g.dependency_closure(did)):
            closure.add(cid)
            node = store.get(cid)
            if node is None:
                violations.append(f"I2: closure of deleted {did} names unknown node {cid}")
                continue
            if node.deleted or node.outdated:
                continue
            # Still live. Only permissible with a parent that is neither deleted nor
            # outdated — i.e. some provenance genuinely survived the deletion.
            surviving = [
                p
                for p in g.parents(cid)
                if (n := store.get(p)) is not None and not n.deleted and not n.outdated
            ]
            if not surviving:
                unmarked.append(cid)
                violations.append(
                    f"I2: node {cid} is derived from deleted {did}, has no surviving "
                    f"parent, yet is still live and not marked outdated "
                    f"(refcount={node.refcount})"
                )

    return Inv2Report(
        satisfied=not violations,
        deleted_ids=deleted,
        closure_size=len(closure),
        unmarked=sorted(set(unmarked)),
        violations=violations,
    )


# ---------------------------------------------------------------------------------
# Certificate
# ---------------------------------------------------------------------------------


def _witness(
    node: MemoryNode,
    store: MemoryStore,
    dag: DerivationDAG,
    blocked: set[str],
    inv1: Inv1Report,
    inv2: Inv2Report,
) -> dict:
    """Everything needed to re-verify this certificate offline, and nothing else."""
    ancestors = dag.ancestors(node.node_id)
    relevant = ancestors | {node.node_id}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "node": node.to_record(),
        "ancestor_ids": sorted(ancestors),
        "ancestor_subgraph_edges": [
            [p, c] for (p, c) in dag.edges() if p in relevant and c in relevant
        ],
        "ancestor_records": [
            n.to_record() for a in sorted(ancestors) if (n := store.get(a)) is not None
        ],
        "blocked_ids": sorted(blocked),
        "deleted_ids": sorted(n.node_id for n in store.nodes.values() if n.deleted),
        "store_stats": store.stats(),
        "inv1": inv1.to_dict(),
        "inv2": inv2.to_dict(),
    }


def certify(
    store: MemoryStore,
    dag: DerivationDAG | None = None,
    blocklist: Blocklist | None = None,
    node_id: str | None = None,
    *,
    inv1: Inv1Report | None = None,
    inv2: Inv2Report | None = None,
) -> InvariantCertificate:
    """Produce the per-node certificate.

    `inv1` / `inv2` may be passed in when certifying many nodes against one store —
    the system-level checks are identical for every node and re-running the adversarial
    probe per node is quadratic for no benefit. `certify_all` does exactly that.
    """
    if node_id is None:
        raise ValueError("certify() requires node_id")

    g = dag if dag is not None else store.dag
    bl = blocklist if blocklist is not None else NoBlocklist()

    node = store.get(node_id)
    if node is None:
        raise KeyError(f"no such node: {node_id}")

    inv1 = inv1 if inv1 is not None else check_invariant_1(store, bl)
    inv2 = inv2 if inv2 is not None else check_invariant_2(store, g)

    blocked = set(bl.blocked_ids())
    if not blocked:
        blocked = {n.node_id for n in store.all_nodes(include_deleted=True) if bl.blocks(n).blocked}
    # A deleted node is forgotten content whether or not it made it onto the id list.
    blocked |= {n.node_id for n in store.nodes.values() if n.deleted}

    path = g.path_to_ancestor(node_id, blocked)

    return InvariantCertificate(
        node_id=node_id,
        inv1_satisfied=inv1.satisfied,
        inv2_satisfied=inv2.satisfied,
        path_to_any_blocked_node=path,
        witness=_witness(node, store, g, blocked, inv1, inv2),
    )


def certify_all(
    store: MemoryStore,
    node_ids: Sequence[str] | None = None,
    dag: DerivationDAG | None = None,
    blocklist: Blocklist | None = None,
) -> list[InvariantCertificate]:
    """Certify many nodes against one store, computing I1/I2 exactly once."""
    g = dag if dag is not None else store.dag
    bl = blocklist if blocklist is not None else NoBlocklist()
    inv1 = check_invariant_1(store, bl)
    inv2 = check_invariant_2(store, g)
    ids = list(node_ids) if node_ids is not None else [n.node_id for n in store.all_nodes()]
    return [certify(store, g, bl, nid, inv1=inv1, inv2=inv2) for nid in ids]
