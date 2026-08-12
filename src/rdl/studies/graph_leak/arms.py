"""Turning an ``ArmSpec`` into the objects the executor needs.

One detector instance is built per run and handed to *both* guarded arms. That is done
here, in one function, so it cannot be got wrong by editing a config: ``build_detector``
returns a single object and ``build_arm_runtime`` takes it as an argument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ...defenses.base import Defense
from ...defenses.concept_registry import ConceptPolicy, ConceptRegistry
from ...defenses.dragon_style import DragonStyleDefense
from ...defenses.edge_cut import EdgeCutDefense
from ...defenses.forget_policy import ForgetPolicy
from ...defenses.graphforget import GraphForgetDefense
from ...defenses.none import NoDefense
from ...defenses.semantic_detector import SemanticConceptDetector
from ...eval.tofu_data import TofuItem
from ...graph.config import ArmSpec, DefenseSpec, ResolvedGraphConfig
from ...graph.schema import GraphSpec

__all__ = ["ArmPlan", "build_arm_runtime", "build_detector", "build_registry"]


@dataclass(frozen=True)
class ArmPlan:
    spec: ArmSpec
    defense: Defense
    active_nodes: tuple[str, ...]
    # ``None`` means the non-sink nodes answer the target question (MA-LEAK); a value
    # means they answer another concept's question (MA-CONTROL).
    uses_control_question: bool

    @property
    def name(self) -> str:
        return self.spec.name

    def to_dict(self) -> dict:
        return {
            "arm": self.spec.name,
            "label": self.spec.label,
            "mode": self.spec.mode,
            "peer_content": self.spec.peer_content,
            "defense": self.defense.name,
            "propagates_scope": getattr(self.defense, "propagates_scope", False),
            "active_nodes": list(self.active_nodes),
        }


def build_registry(
    items: Sequence[TofuItem], *, concept_of, allow_persistent_write: bool = False
) -> ConceptRegistry:
    """Build the runtime concept registry from questions only.

    ``concept_of`` maps an item id to its concept id. The gold answers are not passed in
    and the registry constructor rejects them if they are — see concept_registry.py.
    """
    return ConceptRegistry.from_questions(
        [
            {
                "item_id": item.item_id,
                "concept_id": concept_of(item.item_id),
                "question": item.question,
            }
            for item in items
        ],
        policy=ConceptPolicy(
            allow_refusal=True,
            allow_persistent_write=allow_persistent_write,
            allow_edge_release=False,
            allow_retrieval=False,
        ),
    )


def build_detector(cfg: ResolvedGraphConfig, registry: ConceptRegistry) -> SemanticConceptDetector:
    """The ONE detector both guarded arms share."""
    return SemanticConceptDetector(
        registry,
        threshold=cfg.study.detector.threshold,
        alias_weight=cfg.study.detector.alias_weight,
        calibrated=cfg.study.detector.status == "calibrated",
        calibration_id=cfg.study.detector.calibration_id,
    )


def _build_defense(
    spec: DefenseSpec, detector: SemanticConceptDetector, *, inspect_query: bool
) -> Defense:
    if spec.kind == "none":
        return NoDefense()
    if spec.kind == "dragon_style":
        return DragonStyleDefense(
            detector=detector,
            guard_action=spec.guard_action,
            apply_at=spec.apply_at,
            implementation=spec.implementation,
            inspect_query=inspect_query,
        )
    if spec.kind == "graphforget":
        policy = ForgetPolicy(
            detector.registry,
            guard_edges=spec.guard_edges,
            guard_writes=spec.guard_writes,
            guard_retrievals=spec.guard_retrievals,
            guard_final_output=spec.guard_final_output,
            allow_safe_refusal=spec.allow_safe_refusal,
        )
        return GraphForgetDefense(
            detector=detector,
            policy=policy,
            semantic_detection=spec.semantic_detection,
            propagate_forget_ids=spec.propagate_forget_ids,
            accumulate_evidence=spec.accumulate_evidence,
            rescan_untagged_memory=spec.rescan_untagged_memory,
            inspect_query=inspect_query,
        )
    if spec.kind == "edge_cut":
        return EdgeCutDefense(detector=detector)
    raise ValueError(f"unknown defence kind '{spec.kind}'")


def build_arm_runtime(
    cfg: ResolvedGraphConfig,
    topology: GraphSpec,
    detector: SemanticConceptDetector,
    *,
    protocol: str = "end_to_end_safety",
) -> list[ArmPlan]:
    """Build every arm's runtime for one run. Detector is shared by construction.

    ``protocol`` decides whether the request gate is part of the defence. Under
    ``graph_flow`` no arm inspects the incoming question, so the request gate is
    constant across arms and what the contrast measures is propagation through the
    graph rather than recognition of the original forget question.
    """
    inspect_query = protocol != "graph_flow"
    plans: list[ArmPlan] = []
    for arm in cfg.arms:
        defense = _build_defense(cfg.defenses[arm.defense], detector, inspect_query=inspect_query)
        active = (topology.sink,) if arm.mode == "single_agent" else tuple(topology.node_ids())
        plans.append(
            ArmPlan(
                spec=arm,
                defense=defense,
                active_nodes=active,
                uses_control_question=arm.peer_content == "cross_concept",
            )
        )
    return plans
