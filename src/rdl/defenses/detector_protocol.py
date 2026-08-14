"""The detector interface the runtime depends on, and the v1/v2 compatibility adapter.

Until now the graph executor named a concrete class, ``SemanticConceptDetector``. That was
fine while there was one backbone; it is not fine now, because the thing being replaced in
v4 is exactly that class and the things that must NOT change — Forget-ID propagation,
multi-surface enforcement, the storage model — are the ones that reference it. So the
dependency is inverted here: the executor depends on :class:`ConceptDetector`, the hashing
detector reaches it through :class:`LegacyDetectorAdapter`, and the answerability model
reaches it directly. Swapping the detector then cannot perturb the mechanism, which is
what lets a v4 result be attributed to the detector rather than to an integration.

The result type
---------------
:class:`AnswerabilityResult` is a superset of the v1 ``DetectionResult`` — same ``score``,
``forget_ids``, ``per_concept``, ``fired``, ``threshold`` — so existing consumers such as
``EvidenceAccumulator`` read it without changes. The added fields are the ones a
three-class answerability model has and a similarity score does not:

``answer_probability``     mass on ANSWER: the candidate supplies an answer to the
                           protected question.
``partial_probability``    mass on PARTIAL: it begins one, and a later candidate could
                           complete it. Never enforceable on its own — that is the whole
                           of the split-clue case, and treating a fragment as a leak would
                           make the accumulation result unfalsifiable.
``evidence_span``          character offsets into the candidate. A tag nobody can quote
                           back is a tag nobody can audit.
``candidate_clause_index`` which unit of the candidate carried it.
``detector_revision``      backend + model revision + thresholds + segmentation version,
                           as one string, recorded in the manifest.

``selected_scope_ids`` names the RELATIONS that fired, where ``forget_ids`` names the
concepts. Enforcement uses the concepts; the per-relation recall table uses the scopes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .detection_context import DetectionContext

__all__ = [
    "NONE_RESULT",
    "AnswerabilityResult",
    "ConceptDetector",
    "LegacyDetectorAdapter",
]


@dataclass(frozen=True)
class AnswerabilityResult:
    """One detector call over one candidate. Superset of the v1 result shape."""

    forget_ids: tuple[str, ...] = ()
    answer_probability: float = 0.0
    partial_probability: float = 0.0
    selected_scope_ids: tuple[str, ...] = ()
    evidence_span: tuple[int, int] | None = None
    candidate_clause_index: int | None = None
    detector_revision: str = ""
    threshold: float = 1.0
    fired: bool = False
    # v1 compatibility surface.
    score: float = 0.0
    per_concept: dict[str, float] = field(default_factory=dict)
    # Set when this result only became an ANSWER because an earlier candidate in the same
    # context supplied the other half. Counted separately in every report: a tag that
    # required accumulation is the claim the propagation story rests on, and one that did
    # not is a claim any node-local guard also makes.
    from_accumulated_evidence: bool = False

    @property
    def version(self) -> str:
        """v1 consumers read ``.version``; v4 records ``detector_revision``."""
        return self.detector_revision

    def to_dict(self) -> dict:
        return {
            "forget_ids": list(self.forget_ids),
            "selected_scope_ids": list(self.selected_scope_ids),
            "answer_probability": round(self.answer_probability, 6),
            "partial_probability": round(self.partial_probability, 6),
            "score": round(self.score, 6),
            "fired": self.fired,
            "threshold": self.threshold,
            "evidence_span": list(self.evidence_span) if self.evidence_span else None,
            "candidate_clause_index": self.candidate_clause_index,
            "detector_revision": self.detector_revision,
            "from_accumulated_evidence": self.from_accumulated_evidence,
        }


NONE_RESULT = AnswerabilityResult()


@runtime_checkable
class ConceptDetector(Protocol):
    """What the executor is allowed to assume about a detector.

    ``context`` is keyword-only and mandatory. That is the enforcement of the
    request-versus-evidence rule: there is no way to call a v4 detector that does not
    separate the protected question from the candidate text, because they arrive as
    different parameters and no implementation is permitted to concatenate them.
    """

    @property
    def revision(self) -> str:
        """Backend, model revision, thresholds and segmentation, as one pinned string."""
        ...

    def score_batch(
        self,
        candidates: Sequence[str],
        *,
        context: DetectionContext,
        restrict_to: frozenset[str] | None = None,
    ) -> Sequence[AnswerabilityResult]: ...

    def to_dict(self) -> dict: ...

    def stats(self) -> dict: ...


class LegacyDetectorAdapter:
    """Presents ``SemanticConceptDetector`` (v1/v2 hashing) through the v4 protocol.

    The adapter exists so that adding the protocol does not touch a single number this
    repo has already published. It changes nothing about how the hashing detector scores:
    it forwards the candidates, intersects the fired ids with what the router selected,
    and reports the similarity score in both ``score`` and ``answer_probability`` —
    honestly, because a similarity backbone has no ANSWER/PARTIAL distinction to report
    and inventing one would give the v4 fields a meaning the v1 backend cannot support.

    It ignores ``context.request_text`` entirely, which is correct and is the point: the
    hashing detector was never conditioned on the request, and folding the request in here
    would silently change v1 behaviour under the banner of an interface change.
    """

    backend = "hashing64"

    def __init__(self, detector) -> None:
        self.detector = detector

    @property
    def revision(self) -> str:
        return f"legacy-adapter-v1:{self.detector.version}"

    def score_batch(
        self,
        candidates: Sequence[str],
        *,
        context: DetectionContext,
        restrict_to: frozenset[str] | None = None,
    ) -> list[AnswerabilityResult]:
        if not candidates:
            return []
        allowed = set(context.routing.policy_context_ids)
        if restrict_to is not None:
            allowed &= set(restrict_to)
        if not allowed:
            # Unrouted: the content model is not called at all. Identical to
            # `identity_router.content_forget_ids`, and asserted by the CPU gate.
            return [
                AnswerabilityResult(detector_revision=self.revision, threshold=self.threshold)
                for _ in candidates
            ]
        results = self.detector.score_batch(list(candidates), restrict_to=sorted(allowed))
        out: list[AnswerabilityResult] = []
        for result in results:
            fired = tuple(sorted(set(result.forget_ids) & allowed))
            out.append(
                AnswerabilityResult(
                    forget_ids=fired,
                    answer_probability=float(result.score),
                    partial_probability=0.0,
                    selected_scope_ids=fired,
                    evidence_span=None,
                    candidate_clause_index=None,
                    detector_revision=self.revision,
                    threshold=float(result.threshold),
                    fired=bool(fired),
                    score=float(result.score),
                    per_concept=dict(result.per_concept),
                )
            )
        return out

    @property
    def threshold(self) -> float:
        return float(self.detector.threshold)

    def to_dict(self) -> dict:
        return {
            "protocol": "concept-detector-v4",
            "backend": self.backend,
            "revision": self.revision,
            "adapts": self.detector.to_dict(),
            "reports_answerability": False,
            "note": (
                "a similarity backbone has no ANSWER/PARTIAL distinction; "
                "answer_probability mirrors the similarity score and partial_probability "
                "is always 0."
            ),
        }

    def stats(self) -> dict:
        return dict(self.detector.stats())
