"""Ingress identity routing — **Detector-v3 Phase 0 probe component only.**

Nothing in this module is wired into the graph executor, and Phase 0 must not wire it.
It exists so the identity-routing hypothesis can be measured before a backend is built.

The hypothesis
--------------
``ENCODER_PROBE.json`` established that *unrestricted* question-similarity detection has
no usable operating point: the negatives are retain90 questions and the prototypes are
forget questions, so an encoder that represents text *type* puts every TOFU author
question near every other one, and recall only rises where the FPR is already ~1.0. The
open question is whether **restricting** detection to the concepts an incoming request is
actually about changes that arithmetic.

The two ids, and why they must not be one id
--------------------------------------------
``policy_context_ids``
    Concepts inferred from the request text. Their ONLY effect is to narrow which
    concepts a later content-detection pass may consider. They never propagate, never
    tag, and never enforce.

``content_forget_ids``
    Evidence that a piece of *content* carries forgotten information. Only these may
    propagate along edges or trigger enforcement.

Collapsing the two would make the natural study vacuous. A request that merely mentions a
forgotten author would taint every downstream message in the trajectory, the guarded arm
would refuse everything, and "leakage went to zero" would be a restatement of "the system
stopped answering" rather than a measurement of the defence. So the separation here is
structural, not a convention: :class:`IngressRouting` exposes no path to an enforceable
id, and :func:`content_forget_ids` cannot return an id that content did not earn.

What the router is allowed to read
----------------------------------
Normalised query text, and the registry's unambiguous aliases. It never sees ``item_id``,
the evaluation ``concept_id``, gold answers, or any harness label — :func:`route_request`
takes a string, which is the enforcement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from .concept_registry import ConceptRegistry, normalise_scope_text

__all__ = [
    "IngressRouting",
    "alias_index",
    "content_forget_ids",
    "route_request",
]


def alias_index(registry: ConceptRegistry) -> dict[str, tuple[frozenset[str], ...]]:
    """``{concept_id: (alias_token_set, ...)}`` for the registry's unambiguous aliases.

    ``ConceptRegistry.from_questions`` has already dropped one-token aliases claimed by
    more than one concept, and one-token aliases that are merely common words appearing in
    several concepts' questions. What survives is name-like and concept-specific, which is
    the only channel a router is entitled to treat as an identity signal.
    """
    index: dict[str, tuple[frozenset[str], ...]] = {}
    for concept in registry.concepts():
        sets = [
            tokens
            for tokens in (frozenset(normalise_scope_text(a).split()) for a in concept.aliases)
            if tokens
        ]
        index[concept.forget_id] = tuple(sets)
    return index


@dataclass(frozen=True)
class IngressRouting:
    """What the request alone licenses. Deliberately inert.

    There is no method here that yields an enforceable id, and no field that a propagation
    step could mistake for one. ``policy_context_ids`` restricts detection and does
    nothing else.
    """

    policy_context_ids: tuple[str, ...] = ()
    matched_aliases: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def routed(self) -> bool:
        return bool(self.policy_context_ids)

    def to_dict(self) -> dict:
        return {
            "policy_context_ids": list(self.policy_context_ids),
            "matched_aliases": {k: list(v) for k, v in sorted(self.matched_aliases.items())},
            # Stated in the artifact so a reader never has to infer it from code.
            "propagates": False,
            "enforces": False,
        }


def route_request(
    request_text: str,
    index: Mapping[str, Sequence[frozenset[str]]],
) -> IngressRouting:
    """Infer ``policy_context_ids`` from request text alone.

    An alias matches when every one of its tokens is present in the normalised request, so
    a multi-token full name must appear in full and a surviving one-token partial must
    appear as its own token. Substring matching is deliberately not used: it would let
    "Hwa" match "Hwang" and hand the router a precision problem the registry already went
    to some trouble to avoid.

    The signature is the access control. There is no parameter through which an item id, a
    concept label or a gold answer could enter.
    """
    tokens = frozenset(normalise_scope_text(request_text or "").split())
    if not tokens:
        return IngressRouting()

    hits: dict[str, list[str]] = {}
    for concept_id in sorted(index):
        matched = [
            " ".join(sorted(alias_tokens))
            for alias_tokens in index[concept_id]
            if alias_tokens and alias_tokens <= tokens
        ]
        if matched:
            hits[concept_id] = sorted(set(matched))
    return IngressRouting(
        policy_context_ids=tuple(sorted(hits)),
        matched_aliases={k: tuple(v) for k, v in hits.items()},
    )


def content_forget_ids(
    detector,
    texts: Sequence[str],
    *,
    routing: IngressRouting,
) -> list[tuple[str, ...]]:
    """Ids that CONTENT earned, one tuple per text.

    Routing may only narrow the candidate set. If the router selected nothing, no content
    detection runs and every text earns nothing — which is the behaviour that keeps a
    retain request from ever producing an enforceable id. If the router selected concepts,
    the existing detector decides, and a text the detector does not fire on earns nothing
    no matter how confidently it was routed.
    """
    if not texts:
        return []
    if not routing.policy_context_ids:
        return [() for _ in texts]
    results = detector.score_batch(list(texts), restrict_to=list(routing.policy_context_ids))
    allowed = set(routing.policy_context_ids)
    # Intersected rather than trusted: `restrict_to` already scopes the scan, and this
    # makes the invariant local — an id can leave this function only if the detector fired
    # on the text AND the router admitted the concept.
    return [tuple(sorted(set(r.forget_ids) & allowed)) for r in results]
