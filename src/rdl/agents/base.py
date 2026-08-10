"""The agent contract.

Deliberately small. An agent takes a question plus whatever memory nodes retrieval
handed it, and returns a reply that knows whether it abstained and which context nodes
it actually saw.

`context_node_ids` is not decoration: it is what the `framework_default` writer uses to
populate `parent_ids`. An answer produced with an empty context therefore produces a
node with no derivation edges — see `rdl.memory.node.MemoryNode.parent_ids`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..memory.node import MemoryNode
from ..models.stub import GenerationRequest

__all__ = ["Agent", "AgentReply"]


@dataclass
class AgentReply:
    agent_id: str
    text: str
    abstained: bool = False
    logprob: float | None = None
    # Ids of the nodes that were ACTUALLY in the prompt. Drives parent_ids at write time.
    context_node_ids: list[str] = field(default_factory=list)
    detector_votes: dict[str, bool] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "text": self.text,
            "abstained": self.abstained,
            "logprob": self.logprob,
            "context_node_ids": list(self.context_node_ids),
            "detector_votes": dict(self.detector_votes),
            "meta": dict(self.meta),
        }


@runtime_checkable
class Agent(Protocol):
    agent_id: str

    def answer(
        self,
        question: str,
        context: Sequence[MemoryNode] = (),
        *,
        peer_answers: Sequence[str] = (),
        generation_request: GenerationRequest | None = None,
    ) -> AgentReply:
        """Answer `question`, optionally grounded in retrieved `context` nodes.

        `peer_answers` are other agents' turns handed over in the same episode. They
        are prompt material only — they are never memory nodes, so they never become
        `parent_ids`.
        """
        ...

    def close(self) -> None: ...
