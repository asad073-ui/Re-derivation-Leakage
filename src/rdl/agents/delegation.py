"""Routing policies.

`always_delegate` is NOT an optional extra. It is the control that decouples routing
from agent A's degradation.

The confound: unlearning drops A's model_utility from 0.60 to 0.46 at forget10. Under
`abstention_triggered` routing, A abstains more on forget-set questions *partly because
it forgot the answer* and *partly because it got worse at answering in general*. If the
C3 effect only appears under abstention routing, we are measuring A's utility collapse,
not forgetting. Running the same conditions under `always_delegate` removes routing
from the causal path entirely: B is asked every time regardless of A.

See docs/00_preregistration.md, confound gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .base import AgentReply

__all__ = [
    "AbstentionTriggered",
    "AlwaysDelegate",
    "DelegationDecision",
    "DelegationPolicy",
    "NeverDelegate",
    "build_delegation_policy",
]


@dataclass(frozen=True)
class DelegationDecision:
    delegate: bool
    reason: str
    policy: str

    def __bool__(self) -> bool:
        return self.delegate


@runtime_checkable
class DelegationPolicy(Protocol):
    name: str
    max_delegations: int

    def should_delegate(self, reply: AgentReply, n_delegations: int) -> DelegationDecision: ...


class AbstentionTriggered:
    """Delegate when the primary agent abstained. The realistic default."""

    name = "abstention_triggered"

    def __init__(self, max_delegations: int = 1) -> None:
        self.max_delegations = max_delegations

    def should_delegate(self, reply: AgentReply, n_delegations: int) -> DelegationDecision:
        if n_delegations >= self.max_delegations:
            return DelegationDecision(
                False, f"delegation budget {self.max_delegations} spent", self.name
            )
        if reply.abstained:
            return DelegationDecision(True, f"{reply.agent_id} abstained", self.name)
        return DelegationDecision(False, f"{reply.agent_id} answered", self.name)


class AlwaysDelegate:
    """Delegate unconditionally. THE CONTROL — see the module docstring.

    Removes agent A's abstention behaviour from the causal path, so any remaining C3
    effect cannot be attributed to A's utility collapse.
    """

    name = "always_delegate"

    def __init__(self, max_delegations: int = 1) -> None:
        self.max_delegations = max_delegations

    def should_delegate(self, reply: AgentReply, n_delegations: int) -> DelegationDecision:
        if n_delegations >= self.max_delegations:
            return DelegationDecision(
                False, f"delegation budget {self.max_delegations} spent", self.name
            )
        return DelegationDecision(True, "control: delegate regardless of A's reply", self.name)


class NeverDelegate:
    """Single-agent operation. C0."""

    name = "never"

    def __init__(self, max_delegations: int = 0) -> None:
        self.max_delegations = 0

    def should_delegate(self, reply: AgentReply, n_delegations: int) -> DelegationDecision:
        return DelegationDecision(False, "delegation disabled", self.name)


def build_delegation_policy(policy: str, max_delegations: int = 1) -> DelegationPolicy:
    if policy == "abstention_triggered":
        return AbstentionTriggered(max_delegations)
    if policy == "always_delegate":
        return AlwaysDelegate(max_delegations)
    if policy == "never":
        return NeverDelegate()
    raise ValueError(
        f"unknown delegation policy '{policy}' "
        "(expected abstention_triggered|always_delegate|never)"
    )
