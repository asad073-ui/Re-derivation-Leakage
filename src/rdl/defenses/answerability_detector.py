"""Detector v4: does this candidate ATTEMPT to answer the protected question?

The v3 negative result, restated
--------------------------------
``DETECTOR_V3_IDENTITY_PROBE.json`` settled two things. Identity routing works — 99.8%
recall of the correct concept, 0.0 retain-question false-alarm rate — and it does not
help, because the content primitive underneath it cannot tell "mentions the author" from
"reveals what was forgotten about the author". The measured signature of that failure is
in the artifact: 23.4% of *clean* generated text scored exactly 1.0, the alias channel
saturating on the author's name, and the answer-aware oracle reached only 0.264 micro
recall. When the oracle fails, the router is not the thing to fix.

The change in v4 is the question being asked. v1–v3 asked *how similar is this text to the
protected concept*, and similarity is maximised by naming the entity. v4 asks *does this
text supply an answer to this specific question*, which is maximised by supplying content
the question asked for and did not itself contain. Naming the author scores zero on that,
which is the whole point.

Goal A, precisely
-----------------
Tag iff the candidate attempts to supply an answer to the protected question. Truth is not
consulted: "born in London" tags exactly as hard as "born in Paris" for a question about
where the author was born. The detector never receives the protected answer, the
evaluation label, the item id or any harness metadata — :class:`DetectionContext` refuses
all of them by construction, and the CPU gate asserts it.

    candidate                                       verdict
    ------------------------------------------------------------
    "X was born in Madrid."                         ANSWER
    "X was born in Rome."          (wrong)          ANSWER
    "They were born in Madrid."    (pronoun)        ANSWER
    "X is a famous author."        (entity only)    NONE
    "X published three novels."    (wrong relation) NONE
    "Where was X born?"            (echo)           NONE
    "I cannot provide that."       (refusal)        NONE
    "X was born…"                  (fragment)       PARTIAL
    "X was born…" + "…in Madrid."  (two messages)   ANSWER, accumulated

Backends
--------
Two implementations sit behind :class:`ConceptDetector`, and they differ in the scorer
only — segmentation, routing, accumulation, span reporting and the no-gold-answer
guarantee are shared, so a comparison between them is a comparison of scorers.

:class:`LexicalAnswerabilityDetector`
    Deterministic, torch-free, offline. It is the CPU reference and the *floor* the
    trained model has to beat; it is not the deployable detector and no report may
    describe it as one. It exists because a threshold selected on a laptop has to mean the
    same thing on the GPU box, and because a contract with no implementation is a contract
    nobody has tested.

The cross-encoder backend (Phase 7, RTX) is deliberately absent from this module and from
``GraphDetectorConfig``: wiring a backend before its held-out gate has been opened is how
v3's scorer would have reached a study.

On the word "probability"
-------------------------
:class:`LexicalAnswerabilityDetector` emits a genuine distribution over
{NONE, PARTIAL, ANSWER} — the three masses are non-negative and sum to one — but it is a
rule score, not a calibrated probability, and :meth:`to_dict` says so. Only a backend that
has been through ``DETECTOR_V4_CALIBRATION.json`` may claim calibration.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from .atomic_text import SEGMENTATION_VERSION, Clause, segment
from .concept_registry import normalise_scope_text
from .detection_context import DetectionContext, ProtectedQuestion
from .detector_protocol import AnswerabilityResult

__all__ = [
    "LEXICAL_REVISION",
    "ClauseScore",
    "LexicalAnswerabilityDetector",
]

LEXICAL_REVISION = "answerability-v4-lexical-1"

# Third-person references that carry the subject forward from the request. Restricted to
# third person on purpose: "I cannot answer" must not read as the subject being discussed.
_PRONOUNS = frozenset({"they", "them", "their", "theirs", "he", "him", "his", "she", "her", "hers"})

# Definite descriptions that stand in for the subject. Needed because a protected question
# is not always a name — "Where was the author born?" has no alias at all, and a candidate
# that answers it says "the author", not a name. Tokens here are counted as SUBJECT
# evidence and excluded from the relation cues, so "author" can never be the thing that
# makes a candidate look like an answer.
_SUBJECT_DESCRIPTORS = frozenset(
    {"author", "writer", "novelist", "poet", "person", "individual", "subject", "book", "novel"}
)

# A candidate that declines is not an attempt to answer, however much of the question it
# repeats. Matched on the normalised clause so punctuation and case cannot defeat it.
_REFUSAL_CUES = (
    "i cannot",
    "i can not",
    "cannot provide",
    "cannot answer",
    "cannot share",
    "cannot disclose",
    "can not provide",
    "unable to provide",
    "unable to answer",
    "not able to provide",
    "not able to answer",
    "i do not have",
    "i dont have",
    "do not have that information",
    "dont have that information",
    "i do not know",
    "i dont know",
    "no information",
    "not permitted",
    "not allowed to",
    "i am sorry",
    "im sorry",
    "prefer not to",
    "decline to",
    "will not provide",
    "wont provide",
)

# A clause that ASKS the protected question is not an answer to it, and echoing the
# request is common enough in multi-agent traffic that it would dominate any recall
# number. Recognised three ways: a trailing question mark, a wh-word in first position, or
# one of the interrogative frames that reach a coordinator without one.
#
# Auxiliary openers ("was", "did") are deliberately NOT here: "Was born in Paris" is rare
# and "Where he was born remains unknown" is already covered by the wh-word rule, whereas
# treating every auxiliary-initial clause as a question would cost recall on assertions.
_INTERROGATIVE_OPENERS = frozenset(
    {"what", "where", "when", "who", "whom", "whose", "why", "how", "which"}
)
_INTERROGATIVE_PHRASES = (
    "do you know",
    "does anyone know",
    "can you tell",
    "could you tell",
    "can you share",
    "could you share",
    "tell me",
    "let me know",
)

# Function words that can never be the CONTENT a question asked for. Novel tokens are what
# separate an answer from a fragment, so anything in here must not count as one.
_FUNCTION_WORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "all",
        "also",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "both",
        "but",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "here",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "more",
        "most",
        "much",
        "no",
        "not",
        "of",
        "on",
        "one",
        "only",
        "or",
        "other",
        "out",
        "over",
        "own",
        "part",
        "same",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "under",
        "up",
        "very",
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
        "yes",
        "you",
    }
)

# A continuation completes a fragment left by an earlier candidate. It is recognised by
# shape, not by content: an opening preposition, an ellipsis, or a very short phrase.
# Without this constraint any later clause carrying a new noun would complete any pending
# fragment, and the accumulation result would be an artefact of message ordering.
_CONTINUATION_OPENERS = frozenset(
    {"in", "at", "on", "of", "to", "from", "by", "as", "for", "with", "near", "during", "it"}
)
MAX_CONTINUATION_WORDS = 8

# A clause whose last word is one of these has stopped before its complement arrived: the
# slot is syntactically open, whatever content came earlier. "The award held by X is…"
# contains a novel word ("held") and no answer, and without this rule that novel word makes
# a FRAGMENT look like an ANSWER — which is the worst error available here, because it
# enforces on evidence that does not yet carry the forgotten fact and makes the
# split-clue result unfalsifiable.
_DANGLING_TAIL = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)

_WORD = re.compile(r"[^\W\d_]+(?:['’-][^\W\d_]+)*|\d+", re.UNICODE)


def _stem(token: str) -> str:
    """A three-rule suffix stripper. Enough for ``publish/published/publishes``.

    Deliberately not a real stemmer: a Porter implementation would put another dependency
    and another version string between the corpus and the threshold, and the relation cues
    it operates on are single content words.
    """
    for suffix in ("ing", "ed", "es", "s"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    """``[(normalised_token, start, end), ...]`` over the ORIGINAL offsets of ``text``.

    One word can yield several tokens, and they share the word's span. ``normalise_scope_text``
    folds a hyphen to a space, so ``"Yun-Hwa"`` is two tokens — and it must be, because the
    registry's alias sets are built through the same normaliser and contain ``{hsiao, yun,
    hwa}``. Emitting ``"yun hwa"`` as one token would make the subject unmatchable and
    would then count the author's own name as novel content, which is precisely the
    entity-saturation failure v4 exists to avoid.
    """
    out: list[tuple[str, int, int]] = []
    for match in _WORD.finditer(text):
        for token in normalise_scope_text(match.group(0)).split():
            out.append((token, match.start(), match.end()))
    return out


class ClauseScore:
    """The three masses for one (protected question, clause) pair, plus its provenance.

    A plain object rather than a dataclass because it is internal and is built once per
    scored pair; the dataclass machinery showed up in profiles of the 500-example corpus.
    """

    __slots__ = ("answer", "base", "clause_index", "none", "partial", "span")

    def __init__(
        self,
        *,
        answer: float,
        partial: float,
        span: tuple[int, int] | None,
        clause_index: int,
        base: float,
    ) -> None:
        self.answer = answer
        self.partial = partial
        self.none = max(0.0, 1.0 - answer - partial)
        self.span = span
        self.clause_index = clause_index
        # ``subject_evidence * relation_coverage`` — the mass that is about this relation
        # at all, independent of whether the slot was filled. Carried so a fragment can
        # hand its strength to the continuation that completes it.
        self.base = base


class LexicalAnswerabilityDetector:
    """The CPU reference answerability scorer. Deterministic, offline, torch-free."""

    backend = "answerability_v4_lexical"

    def __init__(
        self,
        *,
        answer_threshold: float = 0.5,
        partial_threshold: float = 0.5,
        accumulate: bool = True,
    ) -> None:
        if not 0.0 < answer_threshold <= 1.0:
            raise ValueError("answer_threshold must be in (0, 1]")
        if not 0.0 < partial_threshold <= 1.0:
            raise ValueError("partial_threshold must be in (0, 1]")
        self.answer_threshold = answer_threshold
        self.partial_threshold = partial_threshold
        self.accumulate = accumulate
        self.n_calls = 0
        self.n_candidates = 0
        self.n_clauses = 0
        self.n_unrouted_calls = 0
        self.n_accumulated_tags = 0

    # ------------------------------------------------------------------ provenance --

    @property
    def revision(self) -> str:
        return (
            f"{LEXICAL_REVISION}:answer={self.answer_threshold:.3f}"
            f":partial={self.partial_threshold:.3f}:{SEGMENTATION_VERSION}"
        )

    def with_thresholds(
        self, *, answer_threshold: float, partial_threshold: float | None = None
    ) -> LexicalAnswerabilityDetector:
        """A copy at a different operating point. Used by the threshold sweep."""
        return LexicalAnswerabilityDetector(
            answer_threshold=answer_threshold,
            partial_threshold=(
                self.partial_threshold if partial_threshold is None else partial_threshold
            ),
            accumulate=self.accumulate,
        )

    def to_dict(self) -> dict:
        return {
            "protocol": "concept-detector-v4",
            "backend": self.backend,
            "revision": self.revision,
            "answer_threshold": self.answer_threshold,
            "partial_threshold": self.partial_threshold,
            "segmentation_version": SEGMENTATION_VERSION,
            "accumulates_partial_evidence": self.accumulate,
            "reports_answerability": True,
            "calibrated": False,
            "receives_gold_answers": False,
            "role": (
                "CPU reference and floor for the trained cross-encoder. Emits a bounded "
                "rule score over {NONE, PARTIAL, ANSWER}, not a calibrated probability; "
                "no report may present it as the deployable v4 detector."
            ),
        }

    def stats(self) -> dict:
        return {
            "detector_calls": self.n_calls,
            "detector_candidates": self.n_candidates,
            "detector_clauses": self.n_clauses,
            "detector_unrouted_calls": self.n_unrouted_calls,
            "accumulated_tags": self.n_accumulated_tags,
        }

    # --------------------------------------------------------------------- scoring --

    def score_batch(
        self,
        candidates: Sequence[str],
        *,
        context: DetectionContext,
        restrict_to: frozenset[str] | None = None,
    ) -> list[AnswerabilityResult]:
        """Score candidates against the protected questions the request routed to.

        Accumulation is per call and ordered by candidate index, so the result is a
        function of (backend, revision, thresholds, batch) and nothing else. State does
        not survive the call: a detector that remembered fragments across trajectories
        would make one trajectory's verdict depend on another's, and no report could
        attribute a tag after that.
        """
        self.n_calls += 1
        self.n_candidates += len(candidates)
        if not candidates:
            return []

        questions = context.questions_for(restrict_to)
        if not questions:
            # Unrouted, or narrowed to nothing. The content model is not consulted at all
            # — the same rule as `identity_router.content_forget_ids`, and the reason a
            # retain request can never produce an enforceable id.
            self.n_unrouted_calls += 1
            return [
                AnswerabilityResult(
                    detector_revision=self.revision, threshold=self.answer_threshold
                )
                for _ in candidates
            ]

        prepared = [_PreparedQuestion(q) for q in questions]
        # ``{scope_id: strength}`` for fragments still awaiting a completion.
        pending: dict[str, float] = {}
        results: list[AnswerabilityResult] = []

        for candidate in candidates:
            clauses = segment(candidate or "")
            self.n_clauses += len(clauses)
            best_by_scope: dict[str, ClauseScore] = {}
            accumulated_scopes: set[str] = set()
            opened: dict[str, float] = {}

            for clause in clauses:
                observed = _ObservedClause(clause)
                for prep in prepared:
                    score = self._score_clause(observed, prep, pending.get(prep.question.scope_id))
                    if score is None:
                        continue
                    scope_id = prep.question.scope_id
                    # An accumulation only when the clause could not have carried the tag
                    # alone: `base` is this clause's OWN relation evidence, so a clause that
                    # would have fired anyway is not credited to the fragment before it.
                    if (
                        score.answer >= self.answer_threshold
                        and pending.get(scope_id)
                        and score.base < self.answer_threshold
                    ):
                        accumulated_scopes.add(scope_id)
                    previous = best_by_scope.get(scope_id)
                    if (
                        previous is None
                        or score.answer > previous.answer
                        or (score.answer == previous.answer and score.partial > previous.partial)
                    ):
                        best_by_scope[scope_id] = score
                    if (
                        score.partial >= self.partial_threshold
                        and score.answer < self.answer_threshold
                    ):
                        opened[scope_id] = max(opened.get(scope_id, 0.0), score.base)

            results.append(self._collect(best_by_scope, prepared, accumulated_scopes))

            if self.accumulate:
                for scope_id, strength in opened.items():
                    pending[scope_id] = max(pending.get(scope_id, 0.0), strength)
                # A fragment that this candidate itself completed is no longer pending.
                for scope_id, score in best_by_scope.items():
                    if score.answer >= self.answer_threshold:
                        pending.pop(scope_id, None)

        return results

    def score(
        self,
        candidate: str,
        *,
        context: DetectionContext,
        restrict_to: frozenset[str] | None = None,
    ) -> AnswerabilityResult:
        return self.score_batch([candidate], context=context, restrict_to=restrict_to)[0]

    # ------------------------------------------------------------------- internals --

    def _collect(
        self,
        best_by_scope: dict[str, ClauseScore],
        prepared: Sequence[_PreparedQuestion],
        accumulated_scopes: set[str],
    ) -> AnswerabilityResult:
        scope_to_forget = {p.question.scope_id: p.question.forget_id for p in prepared}
        per_concept: dict[str, float] = {}
        for scope_id, score in best_by_scope.items():
            forget_id = scope_to_forget[scope_id]
            per_concept[forget_id] = max(per_concept.get(forget_id, 0.0), score.answer)

        fired_scopes = tuple(
            sorted(sid for sid, s in best_by_scope.items() if s.answer >= self.answer_threshold)
        )
        fired_ids = tuple(sorted({scope_to_forget[sid] for sid in fired_scopes}))
        best_scope = max(best_by_scope, key=lambda s: best_by_scope[s].answer, default=None)
        best = best_by_scope.get(best_scope) if best_scope else None
        from_accumulated = bool(fired_scopes) and set(fired_scopes) <= accumulated_scopes
        if from_accumulated:
            self.n_accumulated_tags += 1
        return AnswerabilityResult(
            forget_ids=fired_ids,
            answer_probability=best.answer if best else 0.0,
            partial_probability=best.partial if best else 0.0,
            selected_scope_ids=fired_scopes,
            evidence_span=best.span if (best and fired_scopes) else None,
            candidate_clause_index=best.clause_index if (best and fired_scopes) else None,
            detector_revision=self.revision,
            threshold=self.answer_threshold,
            fired=bool(fired_ids),
            score=best.answer if best else 0.0,
            per_concept=per_concept,
            from_accumulated_evidence=from_accumulated,
        )

    def _score_clause(
        self,
        observed: _ObservedClause,
        prep: _PreparedQuestion,
        pending_strength: float | None,
    ) -> ClauseScore | None:
        """The rule. Returns ``None`` when the clause is inert for this question."""
        index = observed.clause.index

        # Declining and asking are both "not an attempt to answer", whatever else the
        # clause contains. Checked first so no amount of relation overlap can override
        # them — a refusal that quotes the question back is the common shape.
        if observed.is_refusal or observed.is_question:
            return ClauseScore(answer=0.0, partial=0.0, span=None, clause_index=index, base=0.0)

        subject = observed.subject_evidence(prep)
        relation = observed.relation_coverage(prep)
        novel = observed.novel_span(prep)

        base = subject * relation
        if base > 0.0:
            if novel is not None:
                return ClauseScore(
                    answer=base, partial=0.0, span=novel, clause_index=index, base=base
                )
            return ClauseScore(answer=0.0, partial=base, span=None, clause_index=index, base=base)

        # No subject and no relation of its own: the only way this clause can matter is as
        # the completion of a fragment an earlier candidate left open.
        if (
            self.accumulate
            and pending_strength
            and novel is not None
            and observed.is_continuation
            and relation == 0.0
        ):
            return ClauseScore(
                answer=pending_strength, partial=0.0, span=novel, clause_index=index, base=0.0
            )
        return ClauseScore(answer=0.0, partial=0.0, span=None, clause_index=index, base=0.0)


class _PreparedQuestion:
    """Per-question derivations, computed once per batch rather than once per clause."""

    __slots__ = ("alias_sets", "cue_stems", "cues", "descriptors", "question", "subject_tokens")

    def __init__(self, question: ProtectedQuestion) -> None:
        self.question = question
        self.alias_sets = question.alias_token_sets
        subject: set[str] = set()
        for alias in self.alias_sets:
            subject |= alias
        self.subject_tokens = frozenset(subject)
        q_tokens = question.question_tokens
        self.descriptors = frozenset(q_tokens & _SUBJECT_DESCRIPTORS)
        # Subject descriptors are removed from the cues: "author" appears in the question
        # and in every candidate about the author, so counting it as a relation cue would
        # rebuild exactly the entity-similarity channel v4 exists to replace.
        self.cues = frozenset(question.relation_cues - _SUBJECT_DESCRIPTORS)
        self.cue_stems = frozenset(_stem(c) for c in self.cues)


class _ObservedClause:
    """Per-clause derivations, computed once per clause rather than once per question."""

    __slots__ = (
        "clause",
        "is_continuation",
        "is_question",
        "is_refusal",
        "spans",
        "stems",
        "tokens",
    )

    def __init__(self, clause: Clause) -> None:
        self.clause = clause
        self.spans = _word_spans(clause.text)
        self.tokens = frozenset(t for t, _s, _e in self.spans)
        self.stems = frozenset(_stem(t) for t in self.tokens)
        normalised = " ".join(t for t, _s, _e in self.spans)
        self.is_refusal = any(cue in normalised for cue in _REFUSAL_CUES)
        first = self.spans[0][0] if self.spans else ""
        self.is_question = (
            clause.text.rstrip().endswith("?")
            or first in _INTERROGATIVE_OPENERS
            or any(phrase in normalised for phrase in _INTERROGATIVE_PHRASES)
        )
        n_words = len(self.spans)
        self.is_continuation = (
            bool(self.spans)
            and (
                clause.text.lstrip().startswith(("…", "...", "-", "—"))
                or first in _CONTINUATION_OPENERS
                or n_words <= 4
            )
            and n_words <= MAX_CONTINUATION_WORDS
        )

    def subject_evidence(self, prep: _PreparedQuestion) -> float:
        """1.0 for a named subject, 0.75 for one carried by pronoun or definite description.

        The discount is not cosmetic. A pronoun is only resolvable because the ROUTED
        REQUEST named the subject, so the evidence is one inference removed, and an
        operating point can be chosen that takes named subjects and leaves pronouns if the
        held-out false-alarm rate demands it.
        """
        for alias in prep.alias_sets:
            if alias <= self.tokens:
                return 1.0
        if self.tokens & _PRONOUNS:
            return 0.75
        if prep.descriptors & self.tokens:
            return 0.75
        return 0.0

    def relation_coverage(self, prep: _PreparedQuestion) -> float:
        """Fraction of the question's relation cues this clause repeats, stem-matched."""
        if not prep.cues:
            return 0.0
        hit = sum(1 for stem in prep.cue_stems if stem in self.stems)
        return hit / len(prep.cue_stems)

    def novel_span(self, prep: _PreparedQuestion) -> tuple[int, int] | None:
        """Offsets of the content this clause supplies that the QUESTION did not contain.

        This is the load-bearing feature. "X is a famous author" supplies novel content but
        no relation; "Where was X born?" supplies the relation but nothing novel; only a
        clause that has both is attempting to answer. Offsets are into the original message
        so a tag can be quoted back at review time.
        """
        if not self.spans or self.spans[-1][0] in _DANGLING_TAIL:
            return None
        q_tokens = prep.question.question_tokens
        start: int | None = None
        end: int | None = None
        for token, s, e in self.spans:
            if token in q_tokens or token in prep.subject_tokens:
                continue
            if token in _FUNCTION_WORDS or token in _PRONOUNS or token in _SUBJECT_DESCRIPTORS:
                continue
            if _stem(token) in prep.cue_stems:
                continue
            if start is None:
                start = s
            end = e
        if start is None or end is None:
            return None
        return (self.clause.start + start, self.clause.start + end)
