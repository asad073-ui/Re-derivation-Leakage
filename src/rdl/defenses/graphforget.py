"""GraphForget: policy-carrying forgetting across a computation graph.

Same detector as the DRAGON-style baseline. Five additions, each of which is a surface
the node-local baseline structurally cannot cover:

1. **Forget-ID inheritance.** ``S(x) = D(content_x) ∪ ⋃_p S(p)``. A paraphrase of a
   tagged message is tagged even when the paraphrase itself scores below threshold.
2. **Joint evidence accumulation.** Parent messages are scored together as well as
   separately, so two individually-innocuous clues that reconstruct a concept at their
   join node are caught. ``accumulated_only`` counts exactly those.
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

__all__ = ["GraphForgetDefense"]


class GraphForgetDefense:
    name = "graphforget"
    propagates_scope = True

    def __init__(
        self,
        *,
        detector: SemanticConceptDetector,
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
                if decision.action == "pass":
                    allowed.append(node_id)
                else:
                    withheld.append(node_id)
                continue
            result = rescan.get(node_id)
            if result is not None and result.fired:
                # A node whose tag was never written, or was removed. Semantics, not
                # bookkeeping, is what withholds it.
                rescan_withheld.append(node_id)
                continue
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
            "semantic_detection": self.semantic_detection,
            "rescan_untagged_memory": self.rescan_untagged_memory,
            "inspect_query": self.inspect_query,
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
