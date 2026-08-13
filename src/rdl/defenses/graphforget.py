"""GraphForget: policy-carrying forgetting across a computation graph.

Same detector as the DRAGON-style baseline. Five additions. Four of them are surfaces the
node-local baseline does not guard at all; the second is a difference in scoring
granularity and is labelled as one, because overstating it is the easiest way to lose the
whole comparison (GU-0026):

1. **Forget-ID inheritance.** ``S(x) = D(content_x) ∪ ⋃_p S(p)``. A paraphrase of a
   tagged message is tagged even when the paraphrase itself scores below threshold.
2. **Subset scoring.** Parent messages are scored separately and in combination, not only
   as one concatenated context. This is *not* a capability the baseline structurally
   lacks — it scores the whole incoming context and so already catches clues that arrive
   together at a join node. What subset scoring adds is robustness to dilution: a long
   query or a pile of irrelevant memory drags the embedding of the concatenation below
   threshold, and the parents scored as their own subset do not. ``subset_only`` counts
   that; ``accumulated_only`` counts the strict case where a combination fired and the
   whole-context view did not. ``dragon_style_subsets`` is the matched baseline that
   removes this difference so the other four can be measured on their own.
3. **Edge enforcement on payloads.** The edge and the routing are untouched; the content
   crossing it is passed, sanitized, quarantined or blocked, and the decision is
   recorded. Topology is held constant so the improvement is attributable to the
   defence rather than to removed communication.
4. **Persistent-write protection.** A write candidate inherits scopes from the query,
   retrieved memory, parent messages, tool results and the producing agent's context. A
   new node cannot become clean by having a fresh id or an empty parent list — which is
   precisely the laundering path the two-agent work found.
5. **Retrieval protection, including a rescan.** A tagged node is not returned; an
   *untagged* node is re-scanned semantically, so stripping a tag does not defeat the
   guard.
"""

from __future__ import annotations

from collections.abc import Sequence

from .base import (
    DefenseCounters,
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
from .evidence_accumulator import EvidenceAccumulator
from .forget_policy import ForgetPolicy
from .sanitizer import Sanitizer
from .semantic_detector import SemanticConceptDetector

__all__ = ["AttributionLedger", "GraphForgetDefense"]

# The five surfaces and the four things that can happen at one. Fixed tuples rather than
# free strings so a typo cannot invent a surface that no report ever sums.
SURFACES: tuple[str, ...] = ("node_input", "edge", "write", "retrieval", "final")
ACTIONS: tuple[str, ...] = ("allow", "sanitize", "quarantine", "refuse")
# Which mechanism had the scope. This is the counter the archived study did not have, and
# without it a reduction cannot be assigned to propagation rather than to detection:
# `semantic_only` is a catch the node-local baseline could also have made, and
# `inherited_only` is one it structurally could not.
ATTRIBUTIONS: tuple[str, ...] = (
    "semantic_only",
    "inherited_only",
    "semantic_and_inherited",
    "neither",
)

_ACTION_FOR_DECISION = {
    "pass": "allow",
    "sanitized": "sanitize",
    "quarantined": "quarantine",
    "blocked": "refuse",
}


def attribution_of(detected: Sequence[str], inherited: Sequence[str]) -> str:
    if detected and inherited:
        return "semantic_and_inherited"
    if detected:
        return "semantic_only"
    if inherited:
        return "inherited_only"
    return "neither"


class AttributionLedger:
    """``(surface, attribution, action) -> count`` for one arm's whole run.

    Flat and dense: every combination is present with a zero rather than absent, because
    a missing key and a zero read the same in a report and only one of them is a
    measurement.
    """

    def __init__(self) -> None:
        self._counts: dict[tuple[str, str, str], int] = {
            (surface, attribution, action): 0
            for surface in SURFACES
            for attribution in ATTRIBUTIONS
            for action in ACTIONS
        }

    def record(
        self, surface: str, *, detected: Sequence[str], inherited: Sequence[str], action: str
    ) -> None:
        key = (surface, attribution_of(detected, inherited), action)
        if key not in self._counts:
            raise KeyError(f"unknown attribution cell {key}")
        self._counts[key] += 1

    def record_decision(
        self, surface: str, *, detected: Sequence[str], inherited: Sequence[str], decision: str
    ) -> None:
        self.record(
            surface,
            detected=detected,
            inherited=inherited,
            action=_ACTION_FOR_DECISION.get(decision, "refuse"),
        )

    def to_dict(self) -> dict:
        by_surface: dict[str, dict[str, dict[str, int]]] = {}
        for (surface, attribution, action), count in self._counts.items():
            by_surface.setdefault(surface, {}).setdefault(attribution, {})[action] = count
        enforced = {
            attribution: sum(
                count
                for (_s, a, action), count in self._counts.items()
                if a == attribution and action != "allow"
            )
            for attribution in ATTRIBUTIONS
        }
        return {
            "schema": "graphforget-attribution-v1",
            "by_surface": by_surface,
            "enforcements_by_attribution": enforced,
            # The headline the mechanism study exists to produce: enforcement that ONLY
            # inherited provenance could have produced.
            "inherited_only_enforcements": enforced["inherited_only"],
            "note": (
                "an `inherited_only` enforcement is one no node-local semantic guard could "
                "have made: nothing at that surface scored above threshold and the scope "
                "arrived through provenance"
            ),
        }


class GraphForgetDefense:
    # The DEFAULT name. Every arm overrides it with its own defence-config name, because
    # four arms of the mechanism study are this class with different switches, and a trace
    # that recorded `defense: graphforget` for the taint-only ablation would describe the
    # arm it is the control for. Same defect class as the `propagates_scope` constant.
    name = "graphforget"

    def __init__(
        self,
        *,
        detector: SemanticConceptDetector,
        name: str | None = None,
        policy: ForgetPolicy | None = None,
        semantic_detection: bool = True,
        propagate_forget_ids: bool = True,
        accumulate_evidence: bool = True,
        guard_edges: bool = True,
        guard_writes: bool = True,
        guard_retrievals: bool = True,
        guard_final_output: bool = True,
        allow_safe_refusal: bool = True,
        rescan_untagged_memory: bool = True,
        inspect_query: bool = True,
        sanitizer: Sanitizer | None = None,
    ) -> None:
        self.detector = detector
        if name:
            self.name = name
        self.policy = policy or ForgetPolicy(
            detector.registry,
            guard_edges=guard_edges,
            guard_writes=guard_writes,
            guard_retrievals=guard_retrievals,
            guard_final_output=guard_final_output,
            allow_safe_refusal=allow_safe_refusal,
        )
        self.semantic_detection = semantic_detection
        self.propagate_forget_ids = propagate_forget_ids
        # Under graph_flow the accumulator does not score the query, so the guard acts
        # only on what the graph itself carries.
        self.inspect_query = inspect_query
        self.accumulator = EvidenceAccumulator(
            detector, enabled=accumulate_evidence, inspect_query=inspect_query
        )
        self.rescan_untagged_memory = rescan_untagged_memory
        self.sanitizer = sanitizer or Sanitizer()
        self.counters = DefenseCounters()
        self.attribution = AttributionLedger()

    @property
    def propagates_scope(self) -> bool:
        """Whether this ARM inherits Forget-IDs — derived, never declared.

        This was a class-level ``True``. Every graphforget variant therefore reported
        ``propagates_scope: true`` in the manifest, including the semantic-only and
        stateless ablations whose entire purpose is not to propagate. The mechanism study
        would have shipped with each arm's manifest describing the arm it is the control
        for. `ArmPlan.to_dict` reads this attribute, so deriving it is what makes the
        manifest a record rather than a restatement of the class name.
        """
        return self.propagate_forget_ids

    # ---------------------------------------------------------------- node input --

    def on_node_input(self, ctx: NodeInputContext) -> NodeInputVerdict:
        self.counters.node_input_calls += 1
        inherited = (
            tuple(sorted(set(ctx.inherited_forget_ids))) if self.propagate_forget_ids else ()
        )

        evidence = (
            self.accumulator.evaluate(
                question=ctx.question,
                input_texts=[e.content for e in ctx.inputs],
                memory_texts=ctx.memory_texts,
            )
            if self.semantic_detection
            else _empty_evidence()
        )
        if evidence.accumulated_only:
            self.counters.accumulated_only_hits += 1

        forget_ids = tuple(sorted(set(evidence.forget_ids) | set(inherited)))
        if inherited and not evidence.fired:
            self.counters.inherited_only_hits += 1
        if not forget_ids:
            self.attribution.record("node_input", detected=(), inherited=(), action="allow")
            return NodeInputVerdict(
                inputs=tuple(ctx.inputs),
                memory_texts=tuple(ctx.memory_texts),
                score=evidence.score,
                reason=f"no scope detected or inherited ({evidence.score:.3f})",
            )

        self.counters.node_input_fired += 1
        decision = self.policy.decide(
            "node_input",
            forget_ids=forget_ids,
            score=evidence.score,
            threshold=self.detector.threshold,
        )
        self.attribution.record_decision(
            "node_input",
            detected=evidence.forget_ids,
            inherited=inherited,
            decision=decision.action,
        )
        forced: str | None = None
        if decision.blocks or decision.action == "blocked":
            forced, _certificate = self.sanitizer.safe_refusal(
                forget_ids, detector_version=self.detector.version
            )
        return NodeInputVerdict(
            inputs=tuple(ctx.inputs),
            memory_texts=tuple(ctx.memory_texts),
            forget_ids=forget_ids,
            score=evidence.score,
            fired=True,
            forced_output=forced,
            reason=(
                f"{decision.reason} [combination={evidence.combination}, "
                f"accumulated_only={evidence.accumulated_only}, inherited={list(inherited)}]"
            ),
        )

    # --------------------------------------------------------------------- edges --

    def on_edge(self, ctx: EdgeContext) -> EdgeVerdict:
        self.counters.edge_calls += 1
        envelope = ctx.envelope
        detected: tuple[str, ...] = ()
        score = envelope.semantic_scope_score
        if self.semantic_detection:
            result = self.detector.score(envelope.content)
            detected, score = result.forget_ids, max(score, result.score)
        inherited = tuple(envelope.forget_ids) if self.propagate_forget_ids else ()
        forget_ids = tuple(sorted(set(detected) | set(inherited)))

        certified_refusal = envelope.sanitization_certificate is not None and bool(
            envelope.sanitization_certificate.get("verified")
        )
        decision = self.policy.decide(
            "edge",
            forget_ids=forget_ids,
            score=score,
            threshold=self.detector.threshold,
            is_certified_refusal=certified_refusal,
        )
        self.attribution.record_decision(
            "edge", detected=detected, inherited=inherited, decision=decision.action
        )
        if decision.action == "pass":
            return EdgeVerdict(
                envelope=envelope.with_decision(
                    status="pass",
                    reason=decision.reason,
                    added_forget_ids=forget_ids,
                    score=score,
                ),
                status="pass",
                reason=decision.reason,
                score=score,
                forget_ids=forget_ids,
            )
        if decision.action == "sanitized":
            return EdgeVerdict(
                envelope=envelope.with_decision(
                    status="sanitized",
                    reason=decision.reason,
                    added_forget_ids=forget_ids,
                    score=score,
                ),
                status="sanitized",
                reason=decision.reason,
                score=score,
                forget_ids=forget_ids,
            )

        if decision.action == "quarantined":
            text, certificate = self.sanitizer.quarantine(
                forget_ids, detector_version=self.detector.version
            )
            self.counters.edge_blocked += 1
        else:
            text, certificate = self.sanitizer.safe_refusal(
                forget_ids, detector_version=self.detector.version
            )
            self.counters.edge_blocked += 1
        return EdgeVerdict(
            # The edge is preserved and still carries a payload; only the content is
            # replaced. `edge_preserved=True` is what distinguishes this from edge_cut.
            envelope=envelope.with_decision(
                status=decision.action,
                reason=decision.reason,
                content=text,
                certificate=certificate.to_dict(),
                added_forget_ids=forget_ids,
                score=score,
            ),
            status=decision.action,
            reason=decision.reason,
            score=score,
            forget_ids=forget_ids,
            edge_preserved=True,
        )

    # ------------------------------------------------------------ persistent write --

    def on_memory_write(self, ctx: WriteContext) -> WriteVerdict:
        self.counters.write_calls += 1
        envelope = ctx.envelope
        detected: tuple[str, ...] = ()
        score = envelope.semantic_scope_score
        if self.semantic_detection:
            result = self.detector.score(envelope.content)
            detected, score = result.forget_ids, max(score, result.score)
        inherited = tuple(envelope.forget_ids) if self.propagate_forget_ids else ()
        forget_ids = tuple(sorted(set(detected) | set(inherited)))

        decision = self.policy.decide(
            "write",
            forget_ids=forget_ids,
            score=score,
            threshold=self.detector.threshold,
        )
        self.attribution.record_decision(
            "write", detected=detected, inherited=inherited, decision=decision.action
        )
        allowed = decision.action == "pass"
        if not allowed:
            self.counters.write_blocked += 1
        return WriteVerdict(
            allowed=allowed,
            envelope=envelope.with_decision(
                status="pass" if allowed else decision.action,
                reason=decision.reason,
                added_forget_ids=forget_ids,
                score=score,
            ),
            reason=decision.reason,
            score=score,
            forget_ids=forget_ids,
        )

    # ----------------------------------------------------------------- retrieval --

    def on_retrieval(self, ctx: RetrievalContext) -> RetrievalVerdict:
        self.counters.retrieval_calls += 1
        allowed: list[str] = []
        withheld: list[str] = []
        rescan_withheld: list[str] = []

        untagged = [(nid, text) for nid, text, tags in ctx.candidates if not tags]
        rescan = (
            {
                nid: result
                for (nid, _t), result in zip(
                    untagged,
                    self.detector.score_batch([text for _n, text in untagged]),
                    strict=True,
                )
            }
            if (self.rescan_untagged_memory and self.semantic_detection and untagged)
            else {}
        )

        for node_id, _text, tags in ctx.candidates:
            if tags:
                decision = self.policy.decide(
                    "retrieval",
                    forget_ids=tags,
                    score=1.0,
                    threshold=self.detector.threshold,
                )
                # A stored tag is inherited provenance by definition: it was written by a
                # previous episode, not scored here.
                self.attribution.record_decision(
                    "retrieval",
                    detected=(),
                    inherited=tags if self.propagate_forget_ids else (),
                    decision=decision.action,
                )
                if decision.action == "pass":
                    allowed.append(node_id)
                else:
                    withheld.append(node_id)
                continue
            result = rescan.get(node_id)
            if result is not None and result.fired:
                # A node whose tag was never written, or was removed. Semantics, not
                # bookkeeping, is what withholds it.
                self.attribution.record(
                    "retrieval", detected=result.forget_ids, inherited=(), action="refuse"
                )
                rescan_withheld.append(node_id)
                continue
            self.attribution.record("retrieval", detected=(), inherited=(), action="allow")
            allowed.append(node_id)

        self.counters.retrieval_withheld += len(withheld) + len(rescan_withheld)
        return RetrievalVerdict(
            allowed_node_ids=tuple(allowed),
            withheld_node_ids=tuple(withheld),
            rescan_withheld_node_ids=tuple(rescan_withheld),
            reason=(f"{len(withheld)} withheld by tag, {len(rescan_withheld)} by semantic rescan"),
        )

    # -------------------------------------------------------------- final output --

    def on_final_output(self, ctx: FinalContext) -> FinalVerdict:
        self.counters.final_calls += 1
        envelope = ctx.envelope
        detected: tuple[str, ...] = ()
        score = envelope.semantic_scope_score
        if self.semantic_detection:
            result = self.detector.score(envelope.content)
            detected, score = result.forget_ids, max(score, result.score)
        inherited = tuple(envelope.forget_ids) if self.propagate_forget_ids else ()
        forget_ids = tuple(sorted(set(detected) | set(inherited)))
        certified_refusal = envelope.sanitization_certificate is not None and bool(
            envelope.sanitization_certificate.get("verified")
        )
        decision = self.policy.decide(
            "final",
            forget_ids=forget_ids,
            score=score,
            threshold=self.detector.threshold,
            is_certified_refusal=certified_refusal,
        )
        self.attribution.record_decision(
            "final", detected=detected, inherited=inherited, decision=decision.action
        )
        if decision.action in ("pass", "sanitized"):
            return FinalVerdict(
                text=envelope.content,
                status=decision.action,
                reason=decision.reason,
                score=score,
                forget_ids=forget_ids,
            )
        self.counters.final_blocked += 1
        text, _certificate = self.sanitizer.safe_refusal(
            forget_ids, detector_version=self.detector.version
        )
        return FinalVerdict(
            text=text,
            status=decision.action,
            reason=decision.reason,
            score=score,
            forget_ids=forget_ids,
        )

    # -------------------------------------------------------------------- report --

    def stats(self) -> dict:
        return {
            "defense": self.name,
            "detector_version": self.detector.version,
            "policy": self.policy.to_dict(),
            "propagate_forget_ids": self.propagate_forget_ids,
            "propagates_scope": self.propagates_scope,
            "semantic_detection": self.semantic_detection,
            "rescan_untagged_memory": self.rescan_untagged_memory,
            "inspect_query": self.inspect_query,
            "causal_attribution": self.attribution.to_dict(),
            **self.counters.to_dict(),
            **self.accumulator.stats(),
        }


def _empty_evidence():
    from .evidence_accumulator import AccumulatedEvidence

    return AccumulatedEvidence(
        forget_ids=(),
        score=0.0,
        fired=False,
        accumulated_only=False,
        individual_forget_ids=(),
        query_forget_ids=(),
        combination="disabled",
    )


def scope_closure(
    seed_ids: Sequence[str], parents: dict[str, Sequence[str]], scopes: dict[str, Sequence[str]]
) -> dict[str, tuple[str, ...]]:
    """Propagate scopes over an arbitrary provenance DAG.

    Exposed for the memory layer and the tests: the same rule that tags envelopes must
    tag stored nodes, and having one implementation is what keeps them from drifting.
    """
    resolved: dict[str, tuple[str, ...]] = {}

    def resolve(node: str, seen: frozenset[str]) -> tuple[str, ...]:
        if node in resolved:
            return resolved[node]
        if node in seen:  # defensive: a cyclic provenance graph is a bug upstream
            return tuple(sorted(set(scopes.get(node, ()))))
        acc = set(scopes.get(node, ()))
        for parent in parents.get(node, ()):
            acc |= set(resolve(parent, seen | {node}))
        out = tuple(sorted(acc))
        resolved[node] = out
        return out

    for node in seed_ids:
        resolve(node, frozenset())
    return resolved
