"""The shared semantic scope detector.

**Both DRAGON-style and GraphForget use one instance of this class.** That is a design
requirement, not an implementation convenience: if the two arms had different detectors,
"our method leaks less" would be equally consistent with "our detector is better", and
the paper's claim is about policy propagation.

Two channels, combined by max:

``embedding``  cosine against the concept's scope prototypes, using the repo's
               deterministic ``HashingEmbedder``. No network, no torch, identical
               vectors on every machine — which is what lets thresholds calibrated on
               CPU mean the same thing on the GPU box.
``alias``      fraction of a concept alias's tokens present in the text. Catches a
               restatement that shares the entity name but few other tokens, which is
               exactly the paraphrase case the embedding channel is weakest on.

The 64-dimensional hashing embedder is a *diagnostic* backbone. Any run that reports a
calibrated threshold must record ``detector_status: calibrated`` and the calibration
artefact; an uncalibrated run records ``diagnostic``. See METRICS.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..memory.index import Embedder, HashingEmbedder, normalise_text
from .concept_registry import ConceptRegistry

__all__ = ["DetectionResult", "SemanticConceptDetector"]


@dataclass(frozen=True)
class DetectionResult:
    """One detector call over one text."""

    score: float
    forget_ids: tuple[str, ...]
    per_concept: dict[str, float]
    fired: bool
    threshold: float
    version: str

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 6),
            "forget_ids": list(self.forget_ids),
            "fired": self.fired,
            "threshold": self.threshold,
            "version": self.version,
        }


class SemanticConceptDetector:
    """Scores text against a concept registry. Batched, deterministic, offline."""

    backbone = "hashing-64"

    def __init__(
        self,
        registry: ConceptRegistry,
        *,
        threshold: float = 0.55,
        embedder: Embedder | None = None,
        alias_weight: float = 1.0,
        calibrated: bool = False,
        calibration_id: str | None = None,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        self.registry = registry
        self.threshold = threshold
        self.embedder: Embedder = embedder if embedder is not None else HashingEmbedder(dim=64)
        self.alias_weight = alias_weight
        self.calibrated = calibrated
        self.calibration_id = calibration_id
        self.n_calls = 0
        self.n_texts = 0
        self._concept_ids: tuple[str, ...] = registry.ids()
        self._matrices: dict[str, np.ndarray] = {}
        self._alias_tokens: dict[str, list[frozenset[str]]] = {}
        for concept in registry.concepts():
            prototypes = list(concept.scope_prototypes)
            self._matrices[concept.forget_id] = (
                self.embedder.encode(prototypes)
                if prototypes
                else np.zeros((0, self.embedder.dim), dtype=np.float32)
            )
            self._alias_tokens[concept.forget_id] = [
                frozenset(normalise_text(a).split()) for a in concept.aliases if a.strip()
            ]

    # ------------------------------------------------------------------ provenance --

    @property
    def version(self) -> str:
        state = "calibrated" if self.calibrated else "diagnostic"
        cal = f":{self.calibration_id}" if self.calibration_id else ""
        return f"semantic-scope-v1:{self.backbone}:thr={self.threshold:.3f}:{state}{cal}"

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "backbone": self.backbone,
            "threshold": self.threshold,
            "calibrated": self.calibrated,
            "calibration_id": self.calibration_id,
            "n_concepts": len(self.registry),
            "registry_fingerprint": self.registry.fingerprint(),
        }

    def with_threshold(
        self, threshold: float, *, calibration_id: str | None = None
    ) -> SemanticConceptDetector:
        """A copy at a different threshold. Used by the calibration sweep."""
        clone = SemanticConceptDetector.__new__(SemanticConceptDetector)
        clone.__dict__.update(self.__dict__)
        clone.threshold = threshold
        clone.calibrated = calibration_id is not None
        clone.calibration_id = calibration_id
        clone.n_calls = 0
        clone.n_texts = 0
        return clone

    # -------------------------------------------------------------------- scoring --

    def score_batch(
        self, texts: Sequence[str], *, restrict_to: Sequence[str] | None = None
    ) -> list[DetectionResult]:
        """Score many texts in one pass. The only method the executor calls."""
        self.n_calls += 1
        self.n_texts += len(texts)
        concept_ids = tuple(restrict_to) if restrict_to is not None else self._concept_ids
        if not texts:
            return []
        vectors = self.embedder.encode([t or "" for t in texts])
        token_sets = [frozenset(normalise_text(t or "").split()) for t in texts]

        results: list[DetectionResult] = []
        for row, tokens in zip(vectors, token_sets, strict=True):
            per_concept: dict[str, float] = {}
            for concept_id in concept_ids:
                matrix = self._matrices.get(concept_id)
                embed_score = 0.0
                if matrix is not None and matrix.shape[0]:
                    embed_score = float(np.max(matrix @ row))
                alias_score = 0.0
                for alias in self._alias_tokens.get(concept_id, []):
                    if not alias:
                        continue
                    coverage = len(alias & tokens) / len(alias)
                    alias_score = max(alias_score, coverage * self.alias_weight)
                per_concept[concept_id] = max(0.0, min(1.0, max(embed_score, alias_score)))
            fired_ids = tuple(
                sorted(cid for cid, value in per_concept.items() if value >= self.threshold)
            )
            best = max(per_concept.values()) if per_concept else 0.0
            results.append(
                DetectionResult(
                    score=best,
                    forget_ids=fired_ids,
                    per_concept=per_concept,
                    fired=bool(fired_ids),
                    threshold=self.threshold,
                    version=self.version,
                )
            )
        return results

    def score(self, text: str, *, restrict_to: Sequence[str] | None = None) -> DetectionResult:
        return self.score_batch([text], restrict_to=restrict_to)[0]

    def stats(self) -> dict:
        return {"detector_calls": self.n_calls, "detector_texts": self.n_texts}
