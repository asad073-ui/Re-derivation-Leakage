"""Metrics.

Everything here is computed **on top of** open-unlearning's outputs, never instead of
them (design invariant 1). `openunlearning_bridge` shells out to their code on their
configs; the metrics we invent live in `containment`, `laundering`, and `controls`.

Headline metric: `laundering.laundering_rate`, not containment. See that module.
"""

from __future__ import annotations

from .containment import (
    SURFACES,
    ContainmentResult,
    SurfaceHit,
    containment,
    sys_recall_at_k,
)
from .controls import ControlReport, compute_controls, delegation_rate
from .es_em import em_score, es_score, rouge_l_recall
from .laundering import LaunderingReport, laundered_items, laundering_rate

__all__ = [
    "SURFACES",
    "ContainmentResult",
    "ControlReport",
    "LaunderingReport",
    "SurfaceHit",
    "compute_controls",
    "containment",
    "delegation_rate",
    "em_score",
    "es_score",
    "laundered_items",
    "laundering_rate",
    "rouge_l_recall",
    "sys_recall_at_k",
]
