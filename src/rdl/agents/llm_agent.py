"""An agent backed by an `LMHandle`.

Identical code path for `StubLM` and the real HF model — that identity is what makes
the contract tests predictive of Colab behaviour. If this class ever needs an
`isinstance(self.lm, StubLM)` branch, the stub has stopped being a faithful stand-in
and the branch is the bug.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ..logging_utils import get_logger
from ..memory.node import MemoryNode
from ..models.stub import GenerationRequest, LMHandle, PromptStyle, format_prompt
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
        prompt_style: PromptStyle = "openunlearning",
    ) -> None:
        self.agent_id = agent_id
        self.lm = lm
        self.detector = detector or LexicalDetector()
        self.max_new_tokens = max_new_tokens
        self.prompt_style = prompt_style
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

    def answer(
        self,
        question: str,
        context: Sequence[MemoryNode] = (),
        *,
        peer_answers: Sequence[str] = (),
        generation_request: GenerationRequest | None = None,
    ) -> AgentReply:
        """Answer one question.

        `peer_answers` carries another agent's turn into this one — the compositional
        handoff C3C measures. It is deliberately NOT written into `context_node_ids`:
        a peer's utterance is not a memory node, and recording it as one would
        fabricate the derivation edge the whole experiment is about the absence of.
        """
        context_texts = [n.content for n in context]
        context_ids = [n.node_id for n in context]

        blocks = list(context_texts)
        if peer_answers:
            blocks.extend(
                f"Another assistant answered: {a.strip()}" for a in peer_answers if a.strip()
            )

        prompt = format_prompt(question, blocks, system=self.system_prompt, style=self.prompt_style)
        # Preserve compatibility with lightweight external LMHandle subclasses that
        # implemented the pre-sampling interface. New sampled calls always carry the
        # explicit request; historical greedy calls retain their exact invocation.
        if generation_request is not None:
            text = self.lm.generate(
                prompt,
                max_new_tokens=self.max_new_tokens,
                system=self.system_prompt,
                request=generation_request,
            )
        else:
            text = self.lm.generate(
                prompt, max_new_tokens=self.max_new_tokens, system=self.system_prompt
            )

        logprob: float | None = None
        if self.use_logprob:
            try:
                logprob = self.lm.sequence_logprob(prompt, text, system=self.system_prompt)
            except Exception as exc:  # scoring must never kill an episode
                log.warning("logprob scoring failed for %s: %s", self.agent_id, exc)

        decision = self.detector.detect(text, logprob)

        provenance = self.lm.generation_provenance()
        semantic_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        # Stub and third-party handles deliberately have no tokenization evidence.
        # Their serialized prompt is the string passed to ``generate`` only.
        serialized_hash = provenance.get("serialized_chat_prompt_sha256") or semantic_hash
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
                "n_peer_answers": len(peer_answers),
                "prompt_style": self.prompt_style,
                "semantic_user_prompt_sha256": provenance.get("semantic_user_prompt_sha256")
                or semantic_hash,
                "serialized_chat_prompt_sha256": serialized_hash,
                "input_ids_sha256": provenance.get("input_ids_sha256"),
                # Legacy name means serialized model prompt, never pre-template text.
                "rendered_prompt_sha256": serialized_hash,
                "generation_request": (generation_request or GenerationRequest()).to_dict(),
            },
        )

    def close(self) -> None:
        self.lm.close()

    def __repr__(self) -> str:  # pragma: no cover
        return f"LLMAgent({self.agent_id!r}, lm={getattr(self.lm, 'model_id', '?')})"
