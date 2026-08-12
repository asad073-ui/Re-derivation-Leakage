"""Detection over subsets of a node's inputs, not just over the whole of it.

    B -> D:  "the author's initials are A.R."
    C -> D:  "the book was titled The Silent Moon"

Neither message clears the threshold on its own. D combines them and reconstructs the
forgotten fact. So before every model call the accumulator scores, in addition to each
message separately:

  * the concatenation of all parent messages,
  * parent messages plus retrieved memory,
  * the full node input, including the query when the protocol scores it.

**What this is NOT evidence for** (GU-0026). It is tempting to write "a node-local guard
structurally cannot see split clues", and it is false. Our DRAGON-style baseline runs at
``apply_at: every_agent_input`` and scores the node's *complete* incoming context —
query, every parent message, and retrieved memory — as one string. Two clues that arrive
together at one node are in that string. The baseline sees them. The last combination
scored here, ``node_input``, is *exactly* what the baseline scores, and it is recorded on
every result as ``node_local_fired`` so the claim can be checked rather than assumed.

The real, measurable differences are narrower and are counted separately:

``node_local_fired``   the whole-context view fired. Nothing here is a gain over the
                       baseline when this is true.
``subset_only``        some strict subset fired and the whole-context view did not.
                       Scoring granularity, not visibility: a long query or a pile of
                       irrelevant memory dilutes the embedding of the concatenation, and
                       scoring the parents as their own subset recovers the signal. A
                       node-local guard *could* do this too — ``DragonStyleDefense(
                       score_subsets=True)`` is the matched ablation that does, and it is
                       what isolates Forget-ID propagation and multi-surface enforcement
                       from this effect.
``accumulated_only``   the strongest of the three: a combination of parent messages fired
                       while no individual input, no query, and **not the whole-context
                       view** did. Only this counts as "reconstructed from several
                       parents and missed by the matched baseline".

The query is excluded from all three explicitly: when the user asks about a forgotten
concept the query itself is in scope, every combination containing it fires, and counting
that would inflate the result with cases a node-local guard catches trivially.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .semantic_detector import DetectionResult, SemanticConceptDetector

__all__ = ["AccumulatedEvidence", "EvidenceAccumulator"]


@dataclass(frozen=True)
class AccumulatedEvidence:
    forget_ids: tuple[str, ...]
    score: float
    fired: bool
    accumulated_only: bool
    individual_forget_ids: tuple[str, ...]
    query_forget_ids: tuple[str, ...]
    combination: str
    per_input: tuple[DetectionResult, ...] = ()
    # What a whole-context node-local guard — our DRAGON-style baseline — would have seen
    # on this same call. Carried on every result so that no claim about the baseline
    # missing something has to be taken on trust.
    node_local_forget_ids: tuple[str, ...] = ()
    node_local_fired: bool = False
    # Fired via a strict subset while the whole-context view did not. A superset of
    # `accumulated_only`, and the honest measure of what subset scoring buys.
    subset_only: bool = False

    def to_dict(self) -> dict:
        return {
            "forget_ids": list(self.forget_ids),
            "score": round(self.score, 6),
            "fired": self.fired,
            "accumulated_only": self.accumulated_only,
            "subset_only": self.subset_only,
            "node_local_fired": self.node_local_fired,
            "node_local_forget_ids": list(self.node_local_forget_ids),
            "individual_forget_ids": list(self.individual_forget_ids),
            "query_forget_ids": list(self.query_forget_ids),
            "combination": self.combination,
        }


class EvidenceAccumulator:
    def __init__(
        self,
        detector: SemanticConceptDetector,
        *,
        enabled: bool = True,
        inspect_query: bool = True,
    ) -> None:
        self.detector = detector
        self.enabled = enabled
        # ``False`` under the graph_flow protocol: the request gate is held constant
        # across arms, so the query is neither scored on its own nor folded into any
        # combination. See rdl.graph.config.Protocol.
        self.inspect_query = inspect_query
        self.n_accumulated_only = 0
        self.n_subset_only = 0
        self.n_node_local_visible = 0

    def evaluate(
        self,
        *,
        question: str,
        input_texts: Sequence[str],
        memory_texts: Sequence[str] = (),
        restrict_to: Sequence[str] | None = None,
    ) -> AccumulatedEvidence:
        parts = [t for t in input_texts if t.strip()]
        memory = [t for t in memory_texts if t.strip()]

        # One batch: the query alone, each individual input, then the combinations.
        # Batching matters — on the GPU profile this is a single detector forward pass
        # per node, not one per message.
        combos: list[tuple[str, str]] = []
        if parts:
            combos.append(("parents", "\n".join(parts)))
        if parts and memory:
            combos.append(("parents+memory", "\n".join([*parts, *memory])))
        node_input_parts = [question, *parts, *memory] if self.inspect_query else [*parts, *memory]
        combos.append(("node_input", "\n".join(node_input_parts).strip()))

        # The query is scored only when the protocol says the request gate is part of
        # the defence. Under graph_flow it is not scored at all, so no combination can
        # inherit its similarity to its own prototype.
        query_probe = [question] if self.inspect_query else []
        texts = [*query_probe, *parts, *(text for _label, text in combos)]
        scored = self.detector.score_batch(texts, restrict_to=restrict_to)
        offset = len(query_probe)
        query_result = scored[0] if self.inspect_query else None
        per_input = tuple(scored[offset : offset + len(parts)])
        combo_results = scored[offset + len(parts) :]

        individual: set[str] = set()
        for result in per_input:
            individual |= set(result.forget_ids)
        query_ids = set(query_result.forget_ids) if query_result is not None else set()
        query_score = query_result.score if query_result is not None else 0.0

        # `node_input` is always the LAST combination, and it is exactly the string a
        # whole-context node-local guard scores. Everything that claims the baseline
        # missed something is measured against this.
        node_input = combo_results[-1]
        node_local = set(node_input.forget_ids)

        joint: set[str]
        if not self.enabled:
            # Node-local mode: only the individual inputs and the full node input count.
            joint = node_local | individual | query_ids
            return AccumulatedEvidence(
                forget_ids=tuple(sorted(joint)),
                score=max(
                    [node_input.score, query_score, *(r.score for r in per_input)],
                    default=0.0,
                ),
                fired=bool(joint),
                accumulated_only=False,
                individual_forget_ids=tuple(sorted(individual)),
                query_forget_ids=tuple(sorted(query_ids)),
                combination="node_input",
                per_input=per_input,
                node_local_forget_ids=tuple(sorted(node_local)),
                node_local_fired=bool(node_local),
                subset_only=False,
            )

        best_label = "node_input"
        best_score = 0.0
        joint = set(individual) | query_ids
        for (label, _text), result in zip(combos, combo_results, strict=True):
            if result.score > best_score:
                best_score, best_label = result.score, label
            joint |= set(result.forget_ids)
        best_score = max([best_score, query_score, *(r.score for r in per_input)], default=0.0)

        # `not node_local` is the correction in GU-0026. Without it these counted every
        # case the whole-context view ALSO caught, and the baseline catches those: it
        # scores that same concatenation. The claim was therefore unfalsifiable by
        # construction and overstated the method.
        subset_only = bool(joint) and not node_local
        # The strongest reading: a COMBINATION of parents revealed it, no single input
        # did, the query did not, and the whole-context view did not either.
        accumulated_only = subset_only and not individual and not query_ids
        if accumulated_only:
            self.n_accumulated_only += 1
        if subset_only:
            self.n_subset_only += 1
        if node_local:
            self.n_node_local_visible += 1
        return AccumulatedEvidence(
            forget_ids=tuple(sorted(joint)),
            score=best_score,
            fired=bool(joint),
            accumulated_only=accumulated_only,
            individual_forget_ids=tuple(sorted(individual)),
            query_forget_ids=tuple(sorted(query_ids)),
            combination=best_label,
            per_input=per_input,
            node_local_forget_ids=tuple(sorted(node_local)),
            node_local_fired=bool(node_local),
            subset_only=subset_only,
        )

    def stats(self) -> dict:
        return {
            "accumulated_only_hits": self.n_accumulated_only,
            "subset_only_hits": self.n_subset_only,
            "node_local_visible_hits": self.n_node_local_visible,
            "accumulated_only_definition": (
                "a combination of parent messages fired while no individual input, no "
                "query, and NOT the whole-context view a node-local guard scores did"
            ),
        }
