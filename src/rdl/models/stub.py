"""StubLM — the thing that makes local CPU work real.

Deterministic, no torch weights, no network, no downloads. Because `StubLM` implements
the same `LMHandle` interface as the real loader, the entire orchestrator, memory
subsystem, invariant checker, and every metric can be developed and green-lighted on a
laptop before a GPU exists.

The rule that follows from this: **if a module cannot be tested with `StubLM`, it is
coupled wrong.** Reaching Colab with an untested orchestrator wastes GPU sessions on
bugs a laptop would have caught in seconds.

`LMHandle` is defined here rather than in `loader.py` because this module is the
torch-free one. `loader.py` imports it; nothing imports the other way.

The prompt convention (`format_prompt` / `parse_prompt`) also lives here, for the same
reason: the stub and the real HF agent must build prompts identically, or the stub
stops being a faithful stand-in.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Literal

import numpy as np

from ..memory.index import normalise_text

__all__ = [
    "CONTEXT_MARKER",
    "PROMPT_STYLES",
    "QUESTION_MARKER",
    "LMHandle",
    "ParsedPrompt",
    "PromptStyle",
    "StubLM",
    "format_prompt",
    "parse_prompt",
]

CONTEXT_MARKER = "Context:"
QUESTION_MARKER = "Question:"
_ANSWER_MARKER = "Answer:"

PromptStyle = Literal["openunlearning", "qa_scaffold"]
PROMPT_STYLES: tuple[str, ...] = ("openunlearning", "qa_scaffold")


# ---------------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------------


class LMHandle(ABC):
    """Exactly three methods. Everything else is a leak of model internals.

    Keeping this surface at three is what stops agent/orchestrator/metric code from
    quietly depending on a tokenizer detail that differs between the stub and the real
    model — which is precisely how a stub stops predicting real behaviour.
    """

    model_id: str = "unknown"

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 128, *, system: str | None = None) -> str:
        """Greedy, deterministic continuation.

        `system` is the system message. It is a first-class argument rather than
        something the caller splices into `prompt`, because the real handle has to hand
        it to the tokenizer's chat template as a separate role — which is what upstream
        open-unlearning does, and what our numbers have to match.
        """

    @abstractmethod
    def logprobs(self, prompt: str, continuation: str, *, system: str | None = None) -> Any:
        """Per-token log-probabilities of `continuation` given `prompt`.

        Returns a torch.Tensor from the real loader and a numpy array from the stub.
        Callers must only reduce it (`mean`, `sum`, `float(...)`) — anything more
        specific couples them to the backend.
        """

    @abstractmethod
    def close(self) -> None:
        """Release weights / free VRAM."""

    def sequence_logprob(
        self, prompt: str, continuation: str, *, system: str | None = None
    ) -> float:
        """Mean per-token logprob. The quantity the logprob abstention detector uses."""
        lp = self.logprobs(prompt, continuation, system=system)
        arr = np.asarray(lp.detach().cpu().numpy() if hasattr(lp, "detach") else lp, dtype=float)
        return float(arr.mean()) if arr.size else float("-inf")

    def __enter__(self) -> LMHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ---------------------------------------------------------------------------------
# Prompt convention, shared by the stub and the real agent
# ---------------------------------------------------------------------------------


class ParsedPrompt:
    __slots__ = ("context", "question")

    def __init__(self, question: str, context: list[str]) -> None:
        self.question = question
        self.context = context

    def __repr__(self) -> str:  # pragma: no cover
        return f"ParsedPrompt(question={self.question!r}, context={len(self.context)} items)"


def format_prompt(
    question: str,
    context: Sequence[str] = (),
    system: str | None = None,
    style: PromptStyle = "openunlearning",
) -> str:
    """Build the user-turn text. Used identically by StubLM and the HF path.

    Two styles, and the difference is not cosmetic:

    ``openunlearning`` (default)
        With no retrieval context, the user turn is **the bare question** — byte for
        byte what `third_party/open-unlearning` puts through the chat template when it
        evaluates TOFU. Agent-loop outputs are then produced under the same prompt as
        the reproduction gate, so the two sets of numbers can sit in the same table.
        When context IS present it is prepended as a labelled block, because upstream
        has no retrieval-augmented equivalent to copy and inventing one silently would
        be worse than labelling it.

    ``qa_scaffold``
        The original ``Question:``/``Answer:`` framing. Kept because it is what the
        stub contract tests were written against, and because a completion-style
        (non-Instruct) checkpoint needs it. It is NOT comparable with the
        reproduction gate — the extra scaffolding changes the distribution the model
        is decoding from.

    `system` is passed through unchanged in ``qa_scaffold``; in ``openunlearning`` it is
    the caller's job to hand it to the chat template as a system message rather than
    inlining it into the user turn, which is what `HFLMHandle._apply_chat_template`
    does.
    """
    if style not in PROMPT_STYLES:
        raise ValueError(f"unknown prompt style '{style}' (expected {'|'.join(PROMPT_STYLES)})")

    parts: list[str] = []
    if style == "openunlearning":
        if context:
            parts.append(CONTEXT_MARKER)
            parts.extend(f"- {c.strip()}" for c in context)
            parts.append("")
        parts.append(question.strip())
        return "\n".join(parts)

    if system:
        parts.append(system.strip())
    if context:
        parts.append(CONTEXT_MARKER)
        parts.extend(f"- {c.strip()}" for c in context)
    parts.append(f"{QUESTION_MARKER} {question.strip()}")
    parts.append(_ANSWER_MARKER)
    return "\n".join(parts)


_Q_RE = re.compile(rf"^{re.escape(QUESTION_MARKER)}\s*(.+)$", re.MULTILINE)


def parse_prompt(prompt: str) -> ParsedPrompt:
    """Recover (question, context) from a prompt built by `format_prompt`, either style.

    Degrades to "the whole prompt is the question" for free-form input, so the stub
    never crashes on a prompt it did not construct.
    """
    has_context = CONTEXT_MARKER in prompt
    m = _Q_RE.search(prompt)

    if m is not None:  # qa_scaffold
        question = m.group(1).strip()
        block = ""
        if has_context:
            block = prompt.split(CONTEXT_MARKER, 1)[1].split(QUESTION_MARKER, 1)[0]
    elif has_context:  # openunlearning, with retrieval context
        # `Context:` / `- item` ... / blank line / question. The blank line is the
        # separator format_prompt emits; without it the question would be swallowed
        # into the context block and the stub would answer the wrong string.
        after = prompt.split(CONTEXT_MARKER, 1)[1]
        block, _, tail = after.partition("\n\n")
        question = tail.strip() or ""
    else:  # openunlearning, bare question
        return ParsedPrompt(question=prompt.strip(), context=[])

    context: list[str] = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith("- "):
            context.append(line[2:].strip())
        elif line:
            context.append(line)
    return ParsedPrompt(question=question, context=context)


# ---------------------------------------------------------------------------------
# The stub
# ---------------------------------------------------------------------------------


class StubLM(LMHandle):
    """A deterministic look-up table pretending to be a language model.

    Parameters
    ----------
    answers
        ``{question: answer}``. Keys are normalised on insertion, so callers may pass
        the raw TOFU question text.
    knowledge_mask
        Question ids (or raw question strings) this model has "unlearned". A masked
        question yields `abstention_text` from parametric memory.
    read_from_context
        When True (the default) a masked question is still answered if the answer is
        present in the retrieval context. This is not a cheat — it is the behaviour a
        real unlearned model exhibits, and reproducing it is the entire point of C2/C3.
        Set False to model an agent that refuses even with evidence in front of it.
    paraphrase_mode
        Return answers in a deterministically paraphrased form so containment tests
        exercise the non-exact-match path. Exact match must fail; token overlap stays
        high so the normalised/entailment surfaces still fire.
    """

    def __init__(
        self,
        answers: Mapping[str, str] | None = None,
        knowledge_mask: Iterable[str] | None = None,
        *,
        abstention_text: str = "I don't know.",
        paraphrase_mode: bool = False,
        read_from_context: bool = True,
        model_id: str = "stub",
        qid_to_question: Mapping[str, str] | None = None,
        known_logprob: float = -0.35,
        unknown_logprob: float = -2.60,
    ) -> None:
        self._answers: dict[str, str] = {normalise_text(q): a for q, a in (answers or {}).items()}
        self._qid_to_norm: dict[str, str] = {
            qid: normalise_text(q) for qid, q in (qid_to_question or {}).items()
        }
        # A mask entry may be a qid or the question text itself.
        self.knowledge_mask: set[str] = set()
        for item in knowledge_mask or ():
            self.knowledge_mask.add(self._qid_to_norm.get(item, normalise_text(item)))

        self.abstention_text = abstention_text
        self.paraphrase_mode = paraphrase_mode
        self.read_from_context = read_from_context
        self.model_id = model_id
        self.known_logprob = known_logprob
        self.unknown_logprob = unknown_logprob
        self.call_log: list[dict] = []

    # ------------------------------------------------------------------ knowledge --

    def teach(self, question: str, answer: str, qid: str | None = None) -> None:
        norm = normalise_text(question)
        self._answers[norm] = answer
        if qid:
            self._qid_to_norm[qid] = norm

    def mask(self, question_or_qid: str) -> None:
        self.knowledge_mask.add(
            self._qid_to_norm.get(question_or_qid, normalise_text(question_or_qid))
        )

    def unmask(self, question_or_qid: str) -> None:
        self.knowledge_mask.discard(
            self._qid_to_norm.get(question_or_qid, normalise_text(question_or_qid))
        )

    def knows(self, question: str) -> bool:
        norm = normalise_text(question)
        return norm in self._answers and norm not in self.knowledge_mask

    def is_masked(self, question: str) -> bool:
        return normalise_text(question) in self.knowledge_mask

    def ground_truth(self, question: str) -> str | None:
        return self._answers.get(normalise_text(question))

    # ----------------------------------------------------------------- generation --

    @staticmethod
    def _paraphrase(text: str) -> str:
        """Deterministic paraphrase.

        Breaks exact and normalised-substring matching while preserving most content
        tokens, so the entailment surface still fires. That asymmetry is what
        `test_containment.py` needs to distinguish the three modes.
        """
        t = text.strip().rstrip(".")
        if "," in t:
            head, tail = t.split(",", 1)
            t = f"{tail.strip()} — and {head.strip()}"
        t = t.replace(" is ", " happens to be ").replace(" was ", " turns out to have been ")
        return f"From what I recall, {t}."

    def _answer_from_context(self, question: str, context: Sequence[str]) -> str | None:
        """Find the answer in the retrieval context.

        Matches on the ground-truth answer when we know it (the honest case) and
        otherwise falls back to token overlap with the question, which is how a real
        model would latch onto a relevant context line.
        """
        if not context:
            return None
        truth = self.ground_truth(question)
        if truth is not None:
            norm_truth = normalise_text(truth)
            for c in context:
                if norm_truth and norm_truth in normalise_text(c):
                    return truth
        q_tokens = set(normalise_text(question).split())
        if not q_tokens:
            return None
        best, best_score = None, 0.0
        for c in context:
            c_tokens = set(normalise_text(c).split())
            if not c_tokens:
                continue
            score = len(q_tokens & c_tokens) / len(q_tokens)
            if score > best_score:
                best, best_score = c, score
        return best if best_score >= 0.6 else None

    def generate(self, prompt: str, max_new_tokens: int = 128, *, system: str | None = None) -> str:
        # `system` is accepted and ignored: the stub is a look-up table and has no
        # instruction-following to condition. Accepting it keeps the stub's signature
        # identical to HFLMHandle's, which is the property that makes contract tests
        # predictive of Colab behaviour.
        parsed = parse_prompt(prompt)
        question, context = parsed.question, parsed.context

        masked = self.is_masked(question)
        source = "parametric"
        answer: str | None = None

        if not masked:
            answer = self.ground_truth(question)
        if answer is None and masked and self.read_from_context:
            answer = self._answer_from_context(question, context)
            if answer is not None:
                source = "context"
        if answer is None and not masked and not self.ground_truth(question):
            answer = self._answer_from_context(question, context)
            if answer is not None:
                source = "context"

        if answer is None:
            out, source = self.abstention_text, "abstain"
        else:
            out = self._paraphrase(answer) if self.paraphrase_mode else answer

        # Honour max_new_tokens the way a real model would: truncate on whitespace.
        toks = out.split()
        if max_new_tokens > 0 and len(toks) > max_new_tokens:
            out = " ".join(toks[:max_new_tokens])

        self.call_log.append(
            {
                # The prompt verbatim. Without it there is no way to assert what the
                # model actually SAW — only what we believe we passed it — and that gap
                # is how the C3C handoff went missing for the life of the arm (ADR-0041).
                "prompt": prompt,
                "question": question,
                "masked": masked,
                "n_context": len(context),
                "source": source,
                "output": out,
            }
        )
        return out

    def logprobs(self, prompt: str, continuation: str, *, system: str | None = None) -> np.ndarray:
        """Flat per-token logprobs: high when the model "knows", low when it does not.

        Crude by design — it exists so `abstention.LogprobDetector` has a real signal
        to threshold on in contract tests, not to model an LM's actual uncertainty.
        """
        question = parse_prompt(prompt).question
        base = self.known_logprob if self.knows(question) else self.unknown_logprob
        if continuation.strip() == self.abstention_text.strip():
            base = self.unknown_logprob
        n = max(1, len(continuation.split()))
        # Deterministic jitter so the values are not suspiciously identical.
        jitter = np.linspace(-0.02, 0.02, n, dtype=float)
        return np.full(n, base, dtype=float) + jitter

    def close(self) -> None:
        self.call_log.clear()

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"StubLM(id={self.model_id!r}, {len(self._answers)} answers, "
            f"{len(self.knowledge_mask)} masked, paraphrase={self.paraphrase_mode})"
        )
