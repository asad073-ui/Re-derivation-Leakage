"""The forgotten-concept registry.

**The registry never stores the forgotten answers.** It holds the concept id, aliases
(the entity names the concept is about), and *scope prototypes* built from question
text and paraphrases of questions. A runtime defence that had to keep the gold answers
in memory to recognise them would preserve exactly the information the system claims to
have forgotten, and a reviewer would be right to say so.

The gold answers live in two places only:

  * the offline evaluator, which never runs in the same process as generation;
  * the controlled-challenge injector (``studies.graph_leak.controls``), which is an
    adversarial stress-test harness and declares ``uses_gold_answers`` in the manifest.
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from ..logging_utils import dumps_canonical
from ..memory.index import normalise_text

__all__ = [
    "MIN_ONE_TOKEN_ALIAS_CHARS",
    "ConceptPolicy",
    "ConceptRegistry",
    "ForgetConcept",
    "alias_variants",
    "extract_aliases",
    "extract_name_spans",
    "normalise_scope_text",
    "resolve_alias_sets",
]

# TOFU questions name their author; a title-cased multi-word span is a serviceable
# alias extractor for synthetic-author data and is deterministic, which matters more
# here than recall — misses are handled by the embedding channel.
_STOPWORDS = frozenset(
    {
        "what",
        "who",
        "when",
        "where",
        "why",
        "how",
        "which",
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "does",
        "do",
        "did",
        "has",
        "have",
        "had",
        "can",
        "could",
        "in",
        "on",
        "of",
        "for",
        "to",
        "and",
        "or",
        "by",
        "with",
        "about",
        "from",
        "that",
        "this",
        "it",
        "its",
        "author",
        "book",
        "books",
        "writer",
        "name",
        "full",
    }
)


@dataclass(frozen=True)
class ConceptPolicy:
    allow_refusal: bool = True
    allow_persistent_write: bool = False
    allow_edge_release: bool = False
    allow_retrieval: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ForgetConcept:
    """One forgotten concept and the policy that travels with it."""

    forget_id: str
    aliases: tuple[str, ...] = ()
    scope_prototypes: tuple[str, ...] = ()
    policy: ConceptPolicy = field(default_factory=ConceptPolicy)
    # Item ids this concept covers. Used to map a trajectory to its concept, never to
    # look content up at enforcement time.
    item_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "forget_id": self.forget_id,
            "aliases": list(self.aliases),
            "n_scope_prototypes": len(self.scope_prototypes),
            "scope_prototype_sha256s": [
                hashlib.sha256(p.encode("utf-8")).hexdigest()[:16] for p in self.scope_prototypes
            ],
            "policy": self.policy.to_dict(),
            "item_ids": list(self.item_ids),
        }


# ------------------------------------------------------------------ aliases (v2) --
#
# v1 extracted only multiword title-cased spans and kept the possessive attached, so
# author-0000's entire lexical channel was the single string "Hsiao Yun-Hwa's". Downstream
# agents abbreviate, and every archived miss in the discovery study is of the form
#
#     "Yun's father's profession as a civil engineer ..."     score 0.500, threshold 0.65
#
# 0.500 is not a near miss, it is arithmetic: `normalise_text` maps punctuation to spaces,
# so the alias tokenises as {hsiao, yun, hwa, s} and the text supplies {yun, s} — two of
# four. v2 fixes the three things that produced that number: the stray possessive token,
# the absence of any partial-name alias, and the absence of a hyphen-split variant.
#
# What it deliberately does NOT do is add gold answers or derive anything from them. The
# aliases are still extracted from question text alone (GU-0005).

MIN_ONE_TOKEN_ALIAS_CHARS = 3
# A generous cap. Enough for a full name, its hyphen variants and its unique partials;
# small enough that the registry fingerprint stays a reviewable object.
MAX_ALIASES_PER_CONCEPT = 24

# Unicode punctuation TOFU text actually contains, folded to ASCII before anything else
# looks at it. A curly apostrophe that survives to the tokeniser makes "Yun's" and
# "Yun’s" two different strings, and only one of them was ever in the registry.
_UNICODE_FOLD = {
    "’": "'",  # right single quotation mark
    "‘": "'",
    "ʼ": "'",  # modifier letter apostrophe
    "“": '"',
    "”": '"',
    "‐": "-",
    "‑": "-",
    "‒": "-",
    "–": "-",
    "—": "-",
    "−": "-",
}
_POSSESSIVE_SUFFIXES = ("'s", "'S", "s'")


def _fold(text: str) -> str:
    out = unicodedata.normalize("NFKC", text or "")
    for source, target in _UNICODE_FOLD.items():
        out = out.replace(source, target)
    return out


def _strip_possessive(word: str) -> str:
    for suffix in _POSSESSIVE_SUFFIXES:
        if len(word) > len(suffix) and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def normalise_scope_text(text: str) -> str:
    """The detector's view of a string: folded, de-possessed, then the repo's normaliser.

    Applied to alias tokens AND to the text being scored, so "Yun's" and "Yun" tokenise
    identically. Without it the possessive contributes a bare ``s`` token to both sides
    and inflates the denominator of every alias-coverage score.

    Deliberately NOT applied to the embedding channel: the hashing backbone's vectors are
    what the calibration artefact was fitted on, and changing them is a different change
    with a different blast radius. v2 moves the lexical channel only.
    """
    folded = _fold(text)
    words = [_strip_possessive(w) for w in folded.split()]
    return normalise_text(" ".join(words))


def extract_name_spans(question: str) -> tuple[str, ...]:
    """Title-cased spans from a question, folded and stripped of possessives.

    Still a lexical heuristic for synthetic-author data rather than an NER system, and
    still deterministic and offline — which matters more here, because a threshold
    calibrated on CPU has to mean the same thing on the GPU box.
    """
    spans: list[str] = []
    current: list[str] = []
    for raw in _fold(question).replace("?", " ").replace(",", " ").split():
        word = _strip_possessive(raw.strip(".'\"()[]:;"))
        if word and word[0].isupper() and word.lower() not in _STOPWORDS:
            current.append(word)
            continue
        if len(current) >= 2:
            spans.append(" ".join(current))
        current = []
    if len(current) >= 2:
        spans.append(" ".join(current))
    seen: list[str] = []
    for span in spans:
        if span not in seen:
            seen.append(span)
    return tuple(seen)


def alias_variants(span: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(multi_token, one_token)`` alias candidates for one name span.

    Multi-token variants are unambiguous by construction and are always kept. One-token
    variants are *candidates*: a surname shared by two forgotten authors would make the
    detector fire on whichever concept sorted first, and firing on the wrong author is a
    false alarm that happens to coincide with a leak, not a catch. `resolve_alias_sets`
    is where that is settled, because it needs the whole registry to settle it.
    """
    multi: list[str] = [span]
    hyphen_split = span.replace("-", " ")
    if hyphen_split != span:
        multi.append(" ".join(hyphen_split.split()))

    one: list[str] = []
    for token in span.split():
        candidates = [token, *token.split("-")] if "-" in token else [token]
        for candidate in candidates:
            cleaned = candidate.strip("-")
            if len(cleaned) >= MIN_ONE_TOKEN_ALIAS_CHARS and cleaned.lower() not in _STOPWORDS:
                one.append(cleaned)

    def _unique(values: Sequence[str]) -> tuple[str, ...]:
        out: list[str] = []
        for value in values:
            if value and value not in out:
                out.append(value)
        return tuple(out)

    return _unique(multi), _unique(one)


def resolve_alias_sets(
    spans_by_concept: Mapping[str, Sequence[str]],
    *,
    question_tokens_by_concept: Mapping[str, Sequence[str]] | None = None,
) -> tuple[dict[str, tuple[str, ...]], dict[str, list[str]]]:
    """Expand name spans into per-concept aliases, rejecting ambiguous one-token ones.

    Returns ``(aliases_by_concept, rejected_by_alias)``. A one-token alias claimed by more
    than one concept is dropped from EVERY concept and recorded — dropping it from all but
    one would be worse than dropping it from all, because the survivor would silently own
    a name it shares.

    ``question_tokens_by_concept`` adds the check that matters more in practice. Title
    case in a question is not a name: "Award", "Write", "New" and "Inspired" all reach the
    span extractor, and as one-token aliases they fire on retain questions about entirely
    different authors. The first measured v2 gate run put the retain90 false-positive rate
    at **0.20** for exactly this reason, against a ceiling of 0.10 — v1's 0.056 was lower
    only because it never emitted a one-token alias at all.

    So a partial name is kept only when the token appears in the questions of EXACTLY ONE
    forgotten concept. A distinctive name is concept-specific by construction; a common
    English word is not. Computed from question text alone, so it stays inside GU-0005.
    """
    multi_by_concept: dict[str, list[str]] = {}
    one_by_concept: dict[str, list[str]] = {}
    claimants: dict[str, set[str]] = {}

    for concept_id in sorted(spans_by_concept):
        multi_by_concept.setdefault(concept_id, [])
        one_by_concept.setdefault(concept_id, [])
        for span in spans_by_concept[concept_id]:
            multi, one = alias_variants(span)
            for alias in multi:
                if alias not in multi_by_concept[concept_id]:
                    multi_by_concept[concept_id].append(alias)
            for alias in one:
                key = normalise_scope_text(alias)
                if not key:
                    continue
                claimants.setdefault(key, set()).add(concept_id)
                if alias not in one_by_concept[concept_id]:
                    one_by_concept[concept_id].append(alias)

    # Document frequency of each candidate token over the CONCEPTS' question text.
    document_frequency: dict[str, set[str]] = {}
    for concept_id, tokens in (question_tokens_by_concept or {}).items():
        for token in set(tokens):
            document_frequency.setdefault(token, set()).add(concept_id)

    rejected: dict[str, list[str]] = {}
    for alias, concepts in sorted(claimants.items()):
        if len(concepts) > 1:
            rejected[alias] = sorted(concepts)
            continue
        seen_in = document_frequency.get(alias)
        if seen_in is not None and len(seen_in) > 1:
            # Not a name: a word that shows up in several authors' questions.
            rejected[alias] = sorted(seen_in)
    resolved: dict[str, tuple[str, ...]] = {}
    for concept_id in sorted(spans_by_concept):
        keep = [
            alias
            for alias in one_by_concept[concept_id]
            if normalise_scope_text(alias) not in rejected
        ]
        resolved[concept_id] = tuple(
            (multi_by_concept[concept_id] + keep)[:MAX_ALIASES_PER_CONCEPT]
        )
    return resolved, rejected


def extract_aliases(question: str, *, max_aliases: int = 4) -> tuple[str, ...]:
    """The v1 single-question extractor, kept because the calibration artefact used it.

    Global uniqueness cannot be decided from one question, so this returns the multiword
    spans only. `ConceptRegistry.from_questions` uses `resolve_alias_sets` instead, which
    sees every concept at once and is what detector v2 is built on.
    """
    return extract_name_spans(question)[:max_aliases]


def paraphrase_question(question: str) -> str:
    """A deterministic question paraphrase used as a second scope prototype.

    Questions, not answers: a prototype built from the answer would put the forgotten
    content back into the runtime.
    """
    text = question.strip().rstrip("?")
    lowered = text.lower()
    for prefix, replacement in (
        ("what is ", "tell me about "),
        ("what are ", "describe "),
        ("who is ", "give details on "),
        ("who are ", "give details on "),
        ("can you ", "please "),
        ("how did ", "explain how "),
        ("where was ", "state the location of "),
    ):
        if lowered.startswith(prefix):
            return f"{replacement}{text[len(prefix):]}"
    return f"tell me about {text}"


class ConceptRegistry:
    """Concept ids to scopes. Built once per run from the frozen cohort."""

    # v2: possessive-stripped, unicode-folded aliases with hyphen variants and globally
    # unique partial names (GU-0032). The version string is in the fingerprint, so a run
    # built on v1 aliases cannot be mistaken for one built on v2.
    version = "concept-registry-v2"

    def __init__(self, concepts: Iterable[ForgetConcept] = ()) -> None:
        self._concepts: dict[str, ForgetConcept] = {}
        self.rejected_aliases: dict[str, list[str]] = {}
        for concept in concepts:
            self.add(concept)

    # -------------------------------------------------------------------- building --

    def add(self, concept: ForgetConcept) -> ForgetConcept:
        if concept.forget_id in self._concepts:
            raise KeyError(f"duplicate forget_id: {concept.forget_id}")
        self._concepts[concept.forget_id] = concept
        return concept

    @classmethod
    def from_questions(
        cls,
        rows: Sequence[dict],
        *,
        policy: ConceptPolicy | None = None,
    ) -> ConceptRegistry:
        """Build from ``[{item_id, concept_id, question}, ...]``.

        Note the absent key: ``answer``. This constructor cannot see one.
        """
        by_concept: dict[str, list[dict]] = {}
        for row in rows:
            if "answer" in row:
                raise ValueError(
                    "the concept registry must never be constructed from gold answers; "
                    "pass {item_id, concept_id, question} only"
                )
            by_concept.setdefault(str(row["concept_id"]), []).append(row)

        # Two phases, because one-token aliases cannot be resolved concept by concept: a
        # surname shared by two forgotten authors has to be dropped from BOTH, and that is
        # only visible with every concept's spans in hand.
        spans_by_concept: dict[str, list[str]] = {}
        question_tokens: dict[str, list[str]] = {}
        for concept_id, group in by_concept.items():
            spans: list[str] = []
            tokens: set[str] = set()
            for row in sorted(group, key=lambda r: str(r["item_id"])):
                question = str(row["question"])
                for span in extract_name_spans(question):
                    if span not in spans:
                        spans.append(span)
                tokens |= set(normalise_scope_text(question).split())
            spans_by_concept[concept_id] = spans
            question_tokens[concept_id] = sorted(tokens)
        aliases_by_concept, rejected = resolve_alias_sets(
            spans_by_concept, question_tokens_by_concept=question_tokens
        )

        registry = cls()
        registry.rejected_aliases = rejected
        for concept_id in sorted(by_concept):
            group = sorted(by_concept[concept_id], key=lambda r: str(r["item_id"]))
            questions = [str(r["question"]) for r in group]
            aliases = list(aliases_by_concept.get(concept_id, ()))
            prototypes = [*questions, *(paraphrase_question(q) for q in questions), *aliases]
            registry.add(
                ForgetConcept(
                    forget_id=concept_id,
                    aliases=tuple(aliases),
                    scope_prototypes=tuple(prototypes),
                    policy=policy or ConceptPolicy(),
                    item_ids=tuple(str(r["item_id"]) for r in group),
                )
            )
        return registry

    def subset(self, forget_ids: Iterable[str]) -> ConceptRegistry:
        """A registry restricted to some concepts.

        Used for the per-item runtime: an agent answering about author 17 should be
        guarded by the whole registry, but the *stress* configuration that scopes
        detection to one concept is an explicit ablation, not a default.
        """
        wanted = set(forget_ids)
        return ConceptRegistry(c for cid, c in sorted(self._concepts.items()) if cid in wanted)

    # --------------------------------------------------------------------- queries --

    def get(self, forget_id: str) -> ForgetConcept:
        return self._concepts[forget_id]

    def concept_for_item(self, item_id: str) -> str | None:
        for concept in self._concepts.values():
            if item_id in concept.item_ids:
                return concept.forget_id
        return None

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._concepts))

    def concepts(self) -> tuple[ForgetConcept, ...]:
        return tuple(self._concepts[cid] for cid in self.ids())

    def policy_for(self, forget_id: str) -> ConceptPolicy:
        concept = self._concepts.get(forget_id)
        return concept.policy if concept is not None else ConceptPolicy()

    def __len__(self) -> int:
        return len(self._concepts)

    def __contains__(self, forget_id: object) -> bool:
        return forget_id in self._concepts

    # ------------------------------------------------------------------ provenance --

    def fingerprint(self) -> str:
        """Hash of the registry's content. Goes in the run manifest."""
        return hashlib.sha256(
            dumps_canonical(
                {"version": self.version, "concepts": [c.to_dict() for c in self.concepts()]}
            ).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "n_concepts": len(self._concepts),
            "fingerprint": self.fingerprint(),
            "concepts": [c.to_dict() for c in self.concepts()],
            "stores_gold_answers": False,
            # Names two or more forgotten concepts share. Reported rather than silently
            # dropped: a registry whose rejection list is long is telling you the cohort
            # has colliding author names, which bounds what any lexical channel can do.
            "rejected_ambiguous_aliases": {
                k: list(v) for k, v in sorted(self.rejected_aliases.items())
            },
        }
