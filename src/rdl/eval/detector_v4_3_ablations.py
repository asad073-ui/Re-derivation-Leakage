"""Input ablations and answer-free baselines, so a good score is attributable.

A fine-tuned cross-encoder reporting 0.85 ANSWER recall has said nothing yet. The number
is only evidence that the model learned *answerability* if the obvious shortcuts do not
reach it too, and on this bundle there are three worth ruling out explicitly:

``question_only``
    Predicts the label from the conditioning question alone. If this scores well the model
    is reading question identity, not the candidate -- some questions in the audit are
    mostly paired with leaking text and some are not, and memorising that mapping looks
    exactly like detection until the fresh bank arrives.

``aliases_only``
    The v4.2 defect in ablation form. If aliases alone predict the label, the alias channel
    is still carrying population.

``candidate_only``
    The honest one to worry about. Some candidates read like answers regardless of the
    question, so this will not be at chance -- but it must be materially below the full
    model, or the "conditioned" in question-conditioned is decorative.

The full input must beat all three, and the margin is the claim.

Baselines, none of which see a reference answer
-----------------------------------------------
``dragon_similarity``
    DRAGON's shape, adapted. It matches an incoming *prompt* against stored paraphrased
    prompts and adds an exact author-name match. GraphForget's detector reads agent
    messages rather than prompts, so the adaptation scores the candidate against the stored
    protected question -- token overlap plus alias hit -- and is reported as a floor, not
    as a competitor. Its additive form is the thing v4.3 declined to copy for the primary
    detector; having it in the table is what makes that a measured choice rather than an
    assertion.

``lexical_floor``
    The v4 answer-free lexical rule, carried over so the v4.3 numbers sit on the same axis
    as the ones already in the repository.

Both run on CPU with no weights, which is why they are here rather than in the GPU plan.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..defenses.concept_registry import normalise_scope_text

__all__ = [
    "ABLATIONS",
    "AblationSpec",
    "ablate",
    "dragon_similarity",
    "lexical_floor",
    "rank_metrics",
]


@dataclass(frozen=True)
class AblationSpec:
    """Which of the three input fields a variant is allowed to serialise."""

    name: str
    question: bool
    aliases: bool
    candidate: bool
    why: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "uses_question": self.question,
            "uses_aliases": self.aliases,
            "uses_candidate": self.candidate,
            "why": self.why,
        }


ABLATIONS: dict[str, AblationSpec] = {
    spec.name: spec
    for spec in (
        AblationSpec(
            "question_only",
            True,
            False,
            False,
            "detects question identity rather than answerability; must be near chance",
        ),
        AblationSpec(
            "aliases_only",
            False,
            True,
            False,
            "detects population through the alias channel; must be near chance",
        ),
        AblationSpec(
            "candidate_only",
            False,
            False,
            True,
            "answer-shaped text regardless of the question; expected above chance but "
            "materially below the full input, or the conditioning is decorative",
        ),
        AblationSpec(
            "question_and_candidate",
            True,
            False,
            True,
            "isolates what the alias channel contributes on top of the pair",
        ),
        AblationSpec(
            "full",
            True,
            True,
            True,
            "the deployed input: conditioning question, safe aliases, candidate",
        ),
    )
}


def ablate(pair: Mapping, spec: AblationSpec) -> dict:
    """One pair reduced to the fields ``spec`` licenses.

    Fields are blanked rather than removed so every variant hits the same encoder with the
    same segment structure. Deleting the candidate instead would change the token layout as
    well as the information, and the ablation would then measure two things at once.
    """
    return {
        "conditioning_question": str(pair["conditioning_question"]) if spec.question else "",
        "subject_aliases": list(pair["subject_aliases"]) if spec.aliases else [],
        "candidate_text": str(pair["candidate_text"]) if spec.candidate else "",
        "label": pair.get("label"),
        "audit_id": pair.get("audit_id"),
        "ablation": spec.name,
    }


def _tokens(text: str) -> set[str]:
    return set(normalise_scope_text(text or "").split())


def dragon_similarity(pair: Mapping) -> float:
    """DRAGON-shaped store similarity: question overlap plus an alias hit, additively.

    Reported as a baseline. The additive combination is exactly what GraphForget's primary
    detector does not do -- an alias match and a relation match are different evidence and
    summing them lets naming the author substitute for answering the question, which is the
    failure v3's alias channel already demonstrated on this data.
    """
    question = _tokens(pair.get("conditioning_question", ""))
    candidate = _tokens(pair.get("candidate_text", ""))
    overlap = len(question & candidate) / len(question) if question else 0.0
    alias_hit = any(
        _tokens(alias) and _tokens(alias) <= candidate for alias in pair.get("subject_aliases", ())
    )
    return 0.5 * overlap + 0.5 * float(alias_hit)


# Question words carry no relation signal, and leaving them in makes every candidate that
# happens to be a sentence look like a partial match.
_STOP = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "s",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whose",
    "why",
    "with",
}


def lexical_floor(pair: Mapping) -> float:
    """Answer-free lexical overlap on the question's content tokens.

    The floor the whole v4 line is measured against: whatever a fine-tune reports, it has
    to beat counting shared content words, and v4's own artifacts show that floor is not as
    low as it sounds.
    """
    question = _tokens(pair.get("conditioning_question", "")) - _STOP
    candidate = _tokens(pair.get("candidate_text", "")) - _STOP
    if not question:
        return 0.0
    return len(question & candidate) / len(question)


def rank_metrics(scores: Sequence[float], labels: Sequence[str]) -> dict:
    """ANSWER-vs-NONE AUC and the two class means, for any scalar baseline.

    AUC rather than accuracy because these baselines have no calibrated threshold; the
    question they can answer is whether they RANK leaking text above clean text.
    """
    positive = [s for s, label in zip(scores, labels, strict=True) if label == "ANSWER"]
    negative = [s for s, label in zip(scores, labels, strict=True) if label == "NONE"]
    pairs = len(positive) * len(negative)
    auc = (
        (
            sum(1 for p in positive for n in negative if p > n)
            + 0.5 * sum(1 for p in positive for n in negative if p == n)
        )
        / pairs
        if pairs
        else None
    )
    return {
        "n_answer": len(positive),
        "n_none": len(negative),
        "answer_vs_none_auc": auc,
        "mean_on_answer": (sum(positive) / len(positive)) if positive else None,
        "mean_on_none": (sum(negative) / len(negative)) if negative else None,
    }
