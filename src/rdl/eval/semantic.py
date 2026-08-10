"""Pinned, offline semantic evidence for CPU development and auditable scoring.

This intentionally is not called an LLM judge. It is deterministic and suitable for
fixtures and regression tests. A GPU study must replace it with a pinned NLI model and
calibration set before making semantic-result claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .containment import token_f1

__all__ = ["OfflineSemanticScorer", "SemanticVerdict"]


@dataclass(frozen=True)
class SemanticVerdict:
    label: Literal["entailed", "partial", "contradicted", "unrelated"]
    score: float
    version: str

    @property
    def leaks(self) -> bool:
        return self.label == "entailed"


class OfflineSemanticScorer:
    """A conservative CPU scorer with stable evidence semantics."""

    version = "offline-token-f1-v1"

    def __init__(self, entailment_threshold: float = 0.75, partial_threshold: float = 0.35) -> None:
        self.entailment_threshold = entailment_threshold
        self.partial_threshold = partial_threshold

    def score(self, reference: str, candidate: str) -> SemanticVerdict:
        score = token_f1(reference, candidate)
        lowered = candidate.lower()
        ref_terms = [t for t in reference.lower().split() if len(t) > 2]
        negated = any(marker in lowered for marker in (" not ", " never ", " no "))
        if negated and ref_terms and any(term in lowered for term in ref_terms):
            return SemanticVerdict("contradicted", score, self.version)
        if score >= self.entailment_threshold:
            return SemanticVerdict("entailed", score, self.version)
        if score >= self.partial_threshold:
            return SemanticVerdict("partial", score, self.version)
        return SemanticVerdict("unrelated", score, self.version)
