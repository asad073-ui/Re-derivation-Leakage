"""Defences for the graph study.

Five arms, one code path. ``MA-LEAK``, ``MA-DRAGON`` and ``MA-GRAPHFORGET`` run the
same executor over the same topology with the same prompts and seeds; the only thing
that differs is which object is constructed here. If an arm ever needs its own branch
inside the executor, the comparison has stopped being controlled.
"""

from __future__ import annotations

from .base import (
    Defense,
    EdgeContext,
    EdgeVerdict,
    FinalContext,
    FinalVerdict,
    NodeInputContext,
    NodeInputVerdict,
    RetrievalContext,
    RetrievalVerdict,
    WriteContext,
    WriteVerdict,
)
from .concept_registry import ConceptRegistry, ForgetConcept
from .dragon_style import DragonStyleDefense
from .edge_cut import EdgeCutDefense
from .forget_policy import ForgetPolicy, PolicyDecision
from .graphforget import GraphForgetDefense
from .none import NoDefense
from .semantic_detector import DetectionResult, SemanticConceptDetector

__all__ = [
    "ConceptRegistry",
    "Defense",
    "DetectionResult",
    "DragonStyleDefense",
    "EdgeContext",
    "EdgeCutDefense",
    "EdgeVerdict",
    "FinalContext",
    "FinalVerdict",
    "ForgetConcept",
    "ForgetPolicy",
    "GraphForgetDefense",
    "NoDefense",
    "NodeInputContext",
    "NodeInputVerdict",
    "PolicyDecision",
    "RetrievalContext",
    "RetrievalVerdict",
    "SemanticConceptDetector",
    "WriteContext",
    "WriteVerdict",
    "build_defense",
]


def build_defense(
    name: str,
    *,
    detector: SemanticConceptDetector | None = None,
    policy: ForgetPolicy | None = None,
    **kwargs: object,
) -> Defense:
    """Construct a defence by arm name.

    ``dragon_style`` and ``graphforget`` MUST be given the same detector instance by the
    caller. Handing the two arms different detectors would make "our method wins" an
    artefact of a better backbone rather than of policy propagation, which is the single
    most likely reviewer objection to the whole study.
    """
    if name == "none":
        return NoDefense()
    if name == "dragon_style":
        if detector is None:
            raise ValueError("dragon_style requires a detector")
        return DragonStyleDefense(detector=detector, **kwargs)  # type: ignore[arg-type]
    if name == "graphforget":
        if detector is None:
            raise ValueError("graphforget requires a detector")
        return GraphForgetDefense(detector=detector, policy=policy, **kwargs)  # type: ignore[arg-type]
    if name == "edge_cut":
        if detector is None:
            raise ValueError("edge_cut requires a detector")
        return EdgeCutDefense(detector=detector, **kwargs)  # type: ignore[arg-type]
    raise ValueError(f"unknown defense '{name}' (expected none|dragon_style|graphforget|edge_cut)")
