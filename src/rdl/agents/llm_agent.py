"""An agent backed by an `LMHandle`.

Identical code path for `StubLM` and the real HF model — that identity is what makes
the contract tests predictive of Colab behaviour. If this class ever needs an
`isinstance(self.lm, StubLM)` branch, the stub has stopped being a faithful stand-in
and the branch is the bug.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..logging_utils import get_logger
from ..memory.node import MemoryNode
from ..models.stub import LMHandle, format_prompt
from .abstention import AbstentionDetector, LexicalDetector, SelfReportDetector
from .base import AgentReply

__all__ = ["LLMAgent"]

log = get_logger(__name__)


class LLMAgent:
    def __init__(
        self,
        agent_id: str,
        lm: LMHandle,
        *,
        detector: AbstentionDetector | None = None,
        system_prompt: str | None = None,
        max_new_tokens: int = 128,
        use_logprob: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self.lm = lm
        self.detector = detector or LexicalDetector()
        self.max_new_tokens = max_new_tokens
        # Scoring the answer costs a second forward pass. Only pay it when a detector
        # actually consumes the signal.
        self.use_logprob = use_logprob or getattr(self.detector, "name", "") in (
            "logprob",
            "ensemble",
        )

        base = system_prompt
        # A self-report detector is useless unless the model was told the protocol.
        if (
            isinstance(self.detector, SelfReportDetector)
            or getattr(self.detector, "name", "") == "ensemble"
        ):
            instr = SelfReportDetector().instruction()
            base = f"{base}\n{instr}" if base else instr
        self.system_prompt = base

    def answer(self, question: str, context: Sequence[MemoryNode] = ()) -> AgentReply:
        context_texts = [n.content for n in context]
        context_ids = [n.node_id for n in context]

        prompt = format_prompt(question, context_texts, system=self.system_prompt)
        text = self.lm.generate(prompt, max_new_tokens=self.max_new_tokens)

        logprob: float | None = None
        if self.use_logprob:
            try:
                logprob = self.lm.sequence_logprob(prompt, text)
            except Exception as exc:  # scoring must never kill an episode
                log.warning("logprob scoring failed for %s: %s", self.agent_id, exc)

        decision = self.detector.detect(text, logprob)

        return AgentReply(
            agent_id=self.agent_id,
            text=text,
            abstained=decision.abstained,
            logprob=logprob,
            context_node_ids=context_ids,
            detector_votes=dict(decision.votes),
            meta={
                "detector": decision.detector,
                "detector_reason": decision.reason,
                "model_id": getattr(self.lm, "model_id", "unknown"),
                "n_context": len(context_ids),
            },
        )

    def close(self) -> None:
        self.lm.close()

    def __repr__(self) -> str:  # pragma: no cover
        return f"LLMAgent({self.agent_id!r}, lm={getattr(self.lm, 'model_id', '?')})"
