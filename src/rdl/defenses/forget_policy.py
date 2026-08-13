"""Turning a detection into an action: pass, sanitize, quarantine, or block.

Separated from the detector so that a calibration change (a threshold) and a policy
change (what to do when it fires) are two different, separately reportable decisions.

The policy never removes an edge. Removing communication would improve every leakage
number for a reason that has nothing to do with detection, and the resulting comparison
would say only that agents who do not talk cannot leak. The edge-cut ablation exists to
quantify exactly that, and it is labelled as an ablation everywhere it appears.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from ..graph.envelope import ReleaseStatus
from .concept_registry import ConceptRegistry

__all__ = ["ForgetPolicy", "PolicyDecision", "Surface"]

Surface = Literal["node_input", "edge", "write", "retrieval", "final"]


@dataclass(frozen=True)
class PolicyDecision:
    action: ReleaseStatus
    reason: str
    forget_ids: tuple[str, ...] = ()

    @property
    def blocks(self) -> bool:
        return self.action in ("blocked", "quarantined")


class ForgetPolicy:
    """Per-surface actions, parameterised by the concept's declared policy."""

    version = "forget-policy-v1"

    def __init__(
        self,
        registry: ConceptRegistry,
        *,
        # The node-input surface. True everywhere except the two arms that exist to price
        # FORWARD PROPAGATION (GU-0034): if the guard refuses the node whose input carries
        # the scope, the model never derives anything, no derivative exists to tag, and
        # the forwarding contrast is zero for a reason that has nothing to do with
        # forwarding. Those two arms let the node read and generate, and are compared on
        # what happens to what it PRODUCED.
        guard_node_inputs: bool = True,
        guard_edges: bool = True,
        guard_writes: bool = True,
        guard_retrievals: bool = True,
        guard_final_output: bool = True,
        allow_safe_refusal: bool = True,
        # ``uncertain`` covers scores in [uncertain_margin, threshold): the object did
        # not fire but is close. Quarantining those on the write path only is the
        # conservative default — blocking them on edges would cost collaboration
        # utility for evidence the detector itself calls inconclusive.
        uncertain_margin: float = 0.10,
    ) -> None:
        self.registry = registry
        self.guard_node_inputs = guard_node_inputs
        self.guard_edges = guard_edges
        self.guard_writes = guard_writes
        self.guard_retrievals = guard_retrievals
        self.guard_final_output = guard_final_output
        self.allow_safe_refusal = allow_safe_refusal
        self.uncertain_margin = uncertain_margin

    def _enabled(self, surface: Surface) -> bool:
        return {
            "node_input": self.guard_node_inputs,
            "edge": self.guard_edges,
            "write": self.guard_writes,
            "retrieval": self.guard_retrievals,
            "final": self.guard_final_output,
        }[surface]

    def decide(
        self,
        surface: Surface,
        *,
        forget_ids: Sequence[str],
        score: float,
        threshold: float,
        is_certified_refusal: bool = False,
    ) -> PolicyDecision:
        ids = tuple(sorted(set(forget_ids)))
        if not self._enabled(surface):
            return PolicyDecision("pass", f"{surface} guarding disabled", ids)
        if not ids:
            if score >= max(0.0, threshold - self.uncertain_margin) and surface == "write":
                return PolicyDecision(
                    "quarantined",
                    f"uncertain: score {score:.3f} within {self.uncertain_margin} of "
                    f"threshold {threshold:.3f}; persistent write withheld",
                    ids,
                )
            return PolicyDecision("pass", "no forgotten scope detected or inherited", ids)

        if is_certified_refusal and self.allow_safe_refusal:
            # A refusal keeps its Forget IDs and is still released. See sanitizer.py.
            return PolicyDecision(
                "sanitized", "certified safe refusal released with retained provenance", ids
            )

        policies = [self.registry.policy_for(fid) for fid in ids]
        if surface == "write" and not all(p.allow_persistent_write for p in policies):
            return PolicyDecision("blocked", f"persistent write forbidden for {list(ids)}", ids)
        if surface == "retrieval" and not all(p.allow_retrieval for p in policies):
            return PolicyDecision("blocked", f"retrieval forbidden for {list(ids)}", ids)
        if surface in ("edge", "final", "node_input") and not all(
            p.allow_edge_release for p in policies
        ):
            action: ReleaseStatus = "blocked" if self.allow_safe_refusal else "quarantined"
            return PolicyDecision(action, f"release forbidden for {list(ids)}", ids)
        return PolicyDecision("pass", f"concept policy permits release for {list(ids)}", ids)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "guard_node_inputs": self.guard_node_inputs,
            "guard_edges": self.guard_edges,
            "guard_writes": self.guard_writes,
            "guard_retrievals": self.guard_retrievals,
            "guard_final_output": self.guard_final_output,
            "allow_safe_refusal": self.allow_safe_refusal,
            "uncertain_margin": self.uncertain_margin,
        }
