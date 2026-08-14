"""What a v4 detector is allowed to know, and the type that enforces it.

Detector v4 answers one question, and the type system is where that question is pinned
down:

    given a PROTECTED QUESTION and a CANDIDATE MESSAGE, does the candidate attempt to
    supply an answer to that question?

Two consequences follow, and both are structural here rather than conventional.

**The protected answer is not an input.** Goal A does not ask whether the candidate is
*correct*. "The author was born in London" tags exactly as hard as "…in Paris" when the
protected question is where the author was born, because a wrong guess at a forgotten fact
is still an attempt to supply it, and a detector that needed the right answer to notice
would be storing the thing the system claims to have forgotten. :class:`ProtectedQuestion`
therefore has no answer field and :meth:`ProtectedQuestion.from_mapping` refuses a mapping
that carries one — the same guard ``ConceptRegistry.from_questions`` already applies.

**The request is conditioning, not evidence.** This is the integration rule v3 got wrong
by accident of API shape. The protected question tells the detector *which relation to
check*; it is not itself evidence that forgotten content appeared. Concatenating the two
into one scored string —

    "Where was Author X born? Author X was born in Paris."

— lets the detector fire on the presence of the question, which under ``graph_flow`` is
present at the root of every guarded trajectory. The guarded arm then refuses everything
and "leakage went to zero" restates "the system stopped answering". So the question
arrives as :attr:`DetectionContext.request_text` and candidates arrive as a separate
argument, and there is no code path that merges them.

``request_is_evidence`` exists for the one protocol where the request genuinely is a
surface under test: ``end_to_end_safety`` runs a request gate, and that decision is
reported separately from ``graph_flow``. It defaults to ``False`` and
:func:`build_context` refuses to set it under ``graph_flow``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .concept_registry import ConceptRegistry, normalise_scope_text
from .identity_router import IngressRouting, route_request

__all__ = [
    "FORBIDDEN_CONTEXT_KEYS",
    "DetectionContext",
    "ProtectedQuestion",
    "build_context",
    "protected_questions_from_registry",
]

# Keys that carry the evaluation's own answer sheet. A runtime detector that saw any of
# them would be scored on information the deployed system does not have, and the resulting
# recall number would describe an experiment nobody can run twice.
#
#   item_id / example_id  index straight into the harness row, hence into the gold answer
#   concept_id            the label the detector is supposed to PREDICT
#   answer / reference    the forgotten content itself
#   is_forget / label     the evaluation's verdict
FORBIDDEN_CONTEXT_KEYS = frozenset(
    {
        "answer",
        "gold_answer",
        "reference_answer",
        "expected_answer",
        "expected_concept",
        "item_id",
        "example_id",
        "concept_id",
        "label",
        "is_forget",
        "is_leak",
        "split",
    }
)

# Question words and function words that carry no relation signal. Kept separate from the
# registry's alias stoplist: that one is tuned to stop title-cased noise becoming a name,
# this one is tuned to leave the RELATION token standing. "born", "genre", "father",
# "published" must survive; "what", "was", "the" must not.
_RELATION_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "in",
        "into",
        "is",
        "it",
        "its",
        "many",
        "me",
        "much",
        "of",
        "on",
        "or",
        "our",
        "please",
        "she",
        "some",
        "tell",
        "than",
        "that",
        "the",
        "their",
        "them",
        "there",
        "these",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "whose",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)


def _tokens(text: str) -> frozenset[str]:
    return frozenset(normalise_scope_text(text or "").split())


@dataclass(frozen=True)
class ProtectedQuestion:
    """One protected relation the detector may be asked to check.

    ``scope_id`` and ``forget_id`` are separate on purpose. A concept can be protected
    under more than one question — "where was X born" and "what year was X born" are two
    relations of one forgotten author — and the detector reports *which relation* it
    thinks a candidate answered while enforcement still happens per concept. Collapsing
    them would make a per-relation recall table impossible to write.
    """

    scope_id: str
    forget_id: str
    question: str
    aliases: tuple[str, ...] = ()
    relation: str = ""
    template_id: str = ""

    @classmethod
    def from_mapping(cls, row: Mapping) -> ProtectedQuestion:
        """Build from a mapping, refusing anything that carries the answer sheet.

        The refusal is not defensive programming. Every one of these keys is present on
        rows that legitimately exist elsewhere in this repo — the offline evaluator has
        them, the corpus builder has them — and the only thing standing between those rows
        and the runtime detector is a constructor that says no.
        """
        present = sorted(FORBIDDEN_CONTEXT_KEYS & set(row))
        if present:
            raise ValueError(
                f"a ProtectedQuestion must never be built from {present}: the runtime "
                "detector is scored on what a deployed system knows, and these keys are "
                "the evaluation's own labels. Pass "
                "{scope_id, forget_id, question, aliases, relation, template_id} only."
            )
        return cls(
            scope_id=str(row["scope_id"]),
            forget_id=str(row["forget_id"]),
            question=str(row["question"]),
            aliases=tuple(str(a) for a in row.get("aliases", ())),
            relation=str(row.get("relation", "")),
            template_id=str(row.get("template_id", "")),
        )

    # ------------------------------------------------------------------- derived --

    @property
    def alias_token_sets(self) -> tuple[frozenset[str], ...]:
        return tuple(t for t in (_tokens(a) for a in self.aliases) if t)

    @property
    def question_tokens(self) -> frozenset[str]:
        return _tokens(self.question)

    @property
    def relation_cues(self) -> frozenset[str]:
        """Question tokens that name the RELATION rather than the subject or the syntax.

        The subject is removed because every candidate about this author contains it, so
        it separates nothing; the wh-words and auxiliaries are removed because every
        question contains them. What is left — ``born``, ``genre``, ``father`` — is the
        only part of the question that distinguishes "answers the protected question"
        from "mentions the protected author", which is precisely the distinction v3's
        alias channel could not make (``DETECTOR_V3_IDENTITY_PROBE.json``: clean text
        scored 1.0 because naming the author saturated it).
        """
        subject: set[str] = set()
        for alias in self.alias_token_sets:
            subject |= alias
        return frozenset(
            token
            for token in self.question_tokens
            if token not in _RELATION_STOPWORDS and token not in subject and len(token) > 2
        )

    def to_dict(self) -> dict:
        return {
            "scope_id": self.scope_id,
            "forget_id": self.forget_id,
            "question": self.question,
            "aliases": list(self.aliases),
            "relation": self.relation,
            "template_id": self.template_id,
            "relation_cues": sorted(self.relation_cues),
            "carries_answer": False,
        }


@dataclass(frozen=True)
class DetectionContext:
    """Conditioning for one detector call. Carries no candidate text.

    The separation is the entire point: candidates are passed to
    ``ConceptDetector.score_batch`` as their own argument, so there is no assembly step in
    which the request could be appended to the evidence.
    """

    request_text: str
    routing: IngressRouting
    protected_questions: tuple[ProtectedQuestion, ...] = ()
    request_is_evidence: bool = False
    protocol: str = "graph_flow"
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        present = sorted(FORBIDDEN_CONTEXT_KEYS & set(self.metadata))
        if present:
            raise ValueError(
                f"DetectionContext.metadata may not carry {present}; see " "FORBIDDEN_CONTEXT_KEYS."
            )
        routed = set(self.routing.policy_context_ids)
        stray = sorted({q.forget_id for q in self.protected_questions} - routed)
        if stray:
            raise ValueError(
                f"protected questions {stray} were not selected by the router. Routing is "
                "the only thing allowed to narrow the candidate concept set, so a "
                "question the router did not select must not be in scope for content "
                "detection."
            )
        if self.request_is_evidence and self.protocol == "graph_flow":
            raise ValueError(
                "request_is_evidence is not available under graph_flow: the request gate "
                "is held constant across arms there, and scoring the request as evidence "
                "would let the guarded arm fire at the root of every trajectory. Use the "
                "end_to_end_safety protocol, whose request-gate decision is reported "
                "separately."
            )

    @property
    def routed(self) -> bool:
        return self.routing.routed

    @property
    def scope_ids(self) -> tuple[str, ...]:
        return tuple(sorted({q.scope_id for q in self.protected_questions}))

    @property
    def forget_ids(self) -> tuple[str, ...]:
        return tuple(sorted({q.forget_id for q in self.protected_questions}))

    def questions_for(self, restrict_to: Iterable[str] | None) -> tuple[ProtectedQuestion, ...]:
        """Protected questions narrowed further by an explicit caller restriction.

        ``restrict_to`` can only ever shrink the set. There is no argument to this class
        that can widen it beyond what the router selected.
        """
        if restrict_to is None:
            return self.protected_questions
        allowed = set(restrict_to)
        return tuple(q for q in self.protected_questions if q.forget_id in allowed)

    def to_dict(self) -> dict:
        return {
            "request_sha256_prefix": _short_hash(self.request_text),
            "routing": self.routing.to_dict(),
            "protected_questions": [q.to_dict() for q in self.protected_questions],
            "request_is_evidence": self.request_is_evidence,
            "protocol": self.protocol,
            "carries_gold_answers": False,
        }


def _short_hash(text: str) -> str:
    import hashlib

    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def protected_questions_from_registry(
    registry: ConceptRegistry,
    *,
    questions_by_concept: Mapping[str, Sequence[str]] | None = None,
) -> tuple[ProtectedQuestion, ...]:
    """Protected questions for every concept in a registry.

    The registry holds scope prototypes rather than the original questions, and prototypes
    include alias strings and paraphrases. When the caller can supply the actual questions
    it should, because a relation cue extracted from the bare alias ``"Hsiao Yun-Hwa"`` is
    empty and a question that never reaches the detector protects nothing. Falling back to
    the prototypes keeps the function total; the fallback is visible in ``template_id``.
    """
    out: list[ProtectedQuestion] = []
    for concept in registry.concepts():
        supplied = (
            list(questions_by_concept.get(concept.forget_id, ())) if questions_by_concept else []
        )
        source = "cohort" if supplied else "scope_prototype"
        questions = supplied or [p for p in concept.scope_prototypes if len(p.split()) >= 4]
        for i, question in enumerate(dict.fromkeys(questions)):
            out.append(
                ProtectedQuestion(
                    scope_id=f"{concept.forget_id}#{i:03d}",
                    forget_id=concept.forget_id,
                    question=question,
                    aliases=tuple(concept.aliases),
                    relation="",
                    template_id=source,
                )
            )
    return tuple(out)


def build_context(
    request_text: str,
    *,
    protected_questions: Sequence[ProtectedQuestion],
    alias_index: Mapping[str, Sequence[frozenset[str]]],
    protocol: str = "graph_flow",
    request_is_evidence: bool = False,
) -> DetectionContext:
    """Route a request, then hand back only what that routing licenses.

    This is the only constructor the runtime should use. It routes on text alone — the
    signature has no parameter through which an item id or a label could enter — and then
    filters the protected questions down to the routed concepts, so the context a detector
    receives cannot contain a relation the request did not select.
    """
    routing = route_request(request_text, alias_index)
    allowed = set(routing.policy_context_ids)
    scoped = tuple(q for q in protected_questions if q.forget_id in allowed)
    return DetectionContext(
        request_text=request_text or "",
        routing=routing,
        protected_questions=scoped,
        request_is_evidence=request_is_evidence,
        protocol=protocol,
    )
