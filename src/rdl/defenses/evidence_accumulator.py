"""Joint detection over several individually-innocuous inputs.

The split-clue case is the one a node-local detector structurally cannot see:

    B -> D:  "the author's initials are A.R."
    C -> D:  "the book was titled The Silent Moon"

Neither message clears the threshold on its own. D combines them and reconstructs the
forgotten fact. So before every model call the accumulator scores, in addition to each
message separately:

  * the concatenation of all parent messages,
  * parent messages plus retrieved memory,
  * the full node input including the query.

``accumulated_only`` is True when a combination fires and **neither any individual input
nor the query on its own** did. The query has to be excluded explicitly: when the user
asks about a forgotten concept the query itself is in scope, every combination that
contains it fires, and counting that as "reconstructed from several parents" would
inflate the split-clue result with cases a node-local detector catches trivially.

That flag is the headline evidence for the split-clue experiment: it counts the cases the
DRAGON-style baseline is expected to miss by construction.
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

    def to_dict(self) -> dict:
        return {
            "forget_ids": list(self.forget_ids),
            "score": round(self.score, 6),
            "fired": self.fired,
            "accumulated_only": self.accumulated_only,
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

        joint: set[str]
        if not self.enabled:
            # Node-local mode: only the individual inputs and the full node input count,
            # and the node input is what a node-local guard would see anyway.
            node_input = combo_results[-1]
            joint = set(node_input.forget_ids) | individual | query_ids
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
            )

        best_label = "node_input"
        best_score = 0.0
        joint = set(individual) | query_ids
        for (label, _text), result in zip(combos, combo_results, strict=True):
            if result.score > best_score:
                best_score, best_label = result.score, label
            joint |= set(result.forget_ids)
        best_score = max([best_score, query_score, *(r.score for r in per_input)], default=0.0)

        # A scope that only a COMBINATION of parent messages reveals. The query firing on
        # its own does not count: a node-local guard sees the query too.
        accumulated_only = bool(joint) and not individual and not query_ids
        if accumulated_only:
            self.n_accumulated_only += 1
        return AccumulatedEvidence(
            forget_ids=tuple(sorted(joint)),
            score=best_score,
            fired=bool(joint),
            accumulated_only=accumulated_only,
            individual_forget_ids=tuple(sorted(individual)),
            query_forget_ids=tuple(sorted(query_ids)),
            combination=best_label,
            per_input=per_input,
        )

    def stats(self) -> dict:
        return {"accumulated_only_hits": self.n_accumulated_only}
