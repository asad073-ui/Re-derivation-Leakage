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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from ..logging_utils import dumps_canonical

__all__ = ["ConceptPolicy", "ConceptRegistry", "ForgetConcept"]

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


def extract_aliases(question: str, *, max_aliases: int = 4) -> tuple[str, ...]:
    """Title-cased spans from a question, minus question words.

    Deterministic and offline. It is a lexical channel for the detector, not an NER
    system; ``docs/graph_unlearning/METRICS.md`` reports the detector's measured recall
    rather than claiming this is complete.
    """
    aliases: list[str] = []
    current: list[str] = []
    for raw in question.replace("?", " ").replace(",", " ").split():
        word = raw.strip(".'\"()[]")
        if word and word[0].isupper() and word.lower() not in _STOPWORDS:
            current.append(word)
            continue
        if len(current) >= 2:
            aliases.append(" ".join(current))
        current = []
    if len(current) >= 2:
        aliases.append(" ".join(current))
    seen: list[str] = []
    for alias in aliases:
        if alias not in seen:
            seen.append(alias)
    return tuple(seen[:max_aliases])


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

    version = "concept-registry-v1"

    def __init__(self, concepts: Iterable[ForgetConcept] = ()) -> None:
        self._concepts: dict[str, ForgetConcept] = {}
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
        registry = cls()
        for concept_id in sorted(by_concept):
            group = sorted(by_concept[concept_id], key=lambda r: str(r["item_id"]))
            questions = [str(r["question"]) for r in group]
            aliases: list[str] = []
            for question in questions:
                for alias in extract_aliases(question):
                    if alias not in aliases:
                        aliases.append(alias)
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
        }
