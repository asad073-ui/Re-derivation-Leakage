"""Two evaluation layers, because v4.2 collapsed them into one and they disagreed.

v4.2's ``selection_metrics`` computed one number it called the retain false-alarm rate:
the fraction of *retain rows* whose ANSWER score cleared the threshold. Every retain row
counted, whatever its label. That is the bug, and it is worth being precise about why,
because the code looks reasonable.

The 300 retain rows are questions about authors that were never forgotten, paired with
text that usually does answer them. A judge labels them ANSWER, correctly. Cross-entropy
then teaches the model "this text answers this question" -- also correctly. And then
checkpoint selection reads the same row's ANSWER score and calls it a false alarm. The
training objective and the selection objective were pulling in opposite directions on 300
of 1,019 rows, and selection wins, so v4.2 was choosing checkpoints that had learned to be
wrong about retain answerability.

The runtime never faces that choice, because a retain question is not in the protected
store. A retain request routes against the store, matches nothing, and no Forget-ID is
created no matter what any answerability score says. Measured on the frozen audit against
``PROTECTED_STORE_RUNTIME.json``, **0 of 300 retain requests route at all** while 711 of
719 protected requests do -- so the end-to-end retain false-alarm rate is a property of
the router, and v4.2 was rejecting checkpoints over a failure mode that cannot occur.

Hence two layers, measured separately and never mixed:

:func:`pair_level_metrics`
    Is this a good answerability classifier? Scored on ``(question, aliases, candidate)``
    pairs with no notion of a store. Retain rows belong here, with their real labels, and
    a correctly-predicted retain ANSWER is a success.

:func:`store_conditioned_metrics`
    Is this a good enforcement system? Scored the way the runtime works: route the
    request against the protected store, score only the routed scopes, fire a Forget-ID
    when the frozen threshold agrees. Retain rows belong here too, and here a fired
    Forget-ID *is* a false alarm -- because reaching one required the router to have
    matched a protected scope.

:func:`select_thresholds` chooses ``tau_answer`` and ``tau_partial`` on the second layer
only, which is what "checkpoint selection must call the same path as the fresh gate" means
in practice.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "DEV_FPR_CEILING",
    "LABELS",
    "StoreRow",
    "expected_calibration_error",
    "pair_level_metrics",
    "select_thresholds",
    "store_conditioned_metrics",
]

LABELS = ("NONE", "PARTIAL", "ANSWER")

# The false-alarm ceiling both protected-clean and retain rates are held to during
# development selection. Same value v4.2 used; what changed is the denominator of the
# retain rate, not the bar.
DEV_FPR_CEILING = 0.10

# Bins for the calibration estimate. Ten is conventional and is enough resolution on a
# 1,019-row development pool; more bins on this much data mostly measures bin noise.
_ECE_BINS = 10


@dataclass(frozen=True)
class StoreRow:
    """One request scored the way the runtime scores it.

    ``scored`` is ``(scope_id, forget_id, P(ANSWER), P(PARTIAL))`` for the scopes the
    ROUTER licensed, and is empty when the request routed to nothing. Empty is the
    important case and is why this is a dataclass rather than a pair of parallel lists: a
    row that routed to nothing has no protected score at all, and code that defaulted it
    to 0.0 would silently report "scored safe" for a request that was never scored.
    """

    audit_id: str
    population: str
    gold_label: str | None
    gold_concept_id: str = ""
    scored: tuple[tuple[str, str, float, float], ...] = ()
    stratum: str = ""

    @property
    def routed(self) -> bool:
        return bool(self.scored)

    @property
    def is_retain(self) -> bool:
        return self.population == "retain"

    def answer_score(self) -> float | None:
        return max((a for _s, _f, a, _p in self.scored), default=None)

    def partial_score(self) -> float | None:
        return max((p for _s, _f, _a, p in self.scored), default=None)

    def fired_forget_ids(self, tau_answer: float) -> tuple[str, ...]:
        return tuple(sorted({f for _s, f, a, _p in self.scored if a >= tau_answer}))

    def fires(self, tau_answer: float) -> bool:
        return bool(self.fired_forget_ids(tau_answer))


# ------------------------------------------------------------------ layer 1: pairs --


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], *, bins: int = _ECE_BINS
) -> float | None:
    """Standard binned ECE over the predicted class's confidence."""
    if not confidences:
        return None
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for confidence, is_correct in zip(confidences, correct, strict=True):
        index = min(bins - 1, max(0, int(confidence * bins)))
        buckets[index].append((confidence, is_correct))
    total = len(confidences)
    error = 0.0
    for entries in buckets.values():
        mean_confidence = sum(c for c, _ in entries) / len(entries)
        accuracy = sum(1 for _, ok in entries if ok) / len(entries)
        error += (len(entries) / total) * abs(mean_confidence - accuracy)
    return error


def _average_precision(scores: Sequence[float], positive: Sequence[bool]) -> float | None:
    """Area under precision-recall, by the step-wise sum used for AP.

    PR-AUC rather than ROC-AUC because ANSWER is the minority class and ROC-AUC is
    optimistic under imbalance -- it credits the large true-negative pool the enforcement
    decision never benefits from.
    """
    n_positive = sum(positive)
    if not n_positive:
        return None
    ranked = sorted(zip(scores, positive, strict=True), key=lambda t: -t[0])
    hits = 0
    total = 0.0
    previous_recall = 0.0
    for seen, (_score, is_positive) in enumerate(ranked, start=1):
        hits += int(is_positive)
        if is_positive:
            recall = hits / n_positive
            total += (hits / seen) * (recall - previous_recall)
            previous_recall = recall
    return total


def pair_level_metrics(
    gold: Sequence[str],
    probabilities: Sequence[Sequence[float]],
    *,
    question_type: Sequence[str] | None = None,
    surface: Sequence[str] | None = None,
) -> dict:
    """Answerability quality on ``(question, aliases, candidate)`` pairs.

    No store, no routing, no population. A retain row that genuinely answers its retain
    question is an ANSWER here and predicting it is a success -- the opposite of what
    v4.2's selection metric did with the same row.
    """
    if len(gold) != len(probabilities):
        raise ValueError("gold and probabilities must be the same length")
    index_of = {name: i for i, name in enumerate(LABELS)}
    answer_index = index_of["ANSWER"]

    confusion = [[0] * len(LABELS) for _ in LABELS]
    confidences: list[float] = []
    correct: list[bool] = []
    answer_scores: list[float] = []
    is_answer: list[bool] = []
    for label, row in zip(gold, probabilities, strict=True):
        if label not in index_of:
            continue
        predicted = max(range(len(LABELS)), key=lambda i: row[i])
        confusion[index_of[label]][predicted] += 1
        confidences.append(float(row[predicted]))
        correct.append(predicted == index_of[label])
        answer_scores.append(float(row[answer_index]))
        is_answer.append(label == "ANSWER")

    per_class = {}
    f1s = []
    for i, name in enumerate(LABELS):
        true_positive = confusion[i][i]
        false_positive = sum(confusion[g][i] for g in range(len(LABELS))) - true_positive
        false_negative = sum(confusion[i]) - true_positive
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall)
            else 0.0
        )
        f2 = (
            5 * precision * recall / (4 * precision + recall)
            if precision and recall and (4 * precision + recall)
            else 0.0
        )
        per_class[name] = {
            "support": sum(confusion[i]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "f2": f2,
        }
        f1s.append(f1)

    total = sum(sum(row) for row in confusion)
    out = {
        "layer": "pair_level_answerability",
        "n_scored": total,
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "accuracy": (sum(confusion[i][i] for i in range(len(LABELS))) / total) if total else None,
        "answer_recall": per_class["ANSWER"]["recall"],
        "partial_recall": per_class["PARTIAL"]["recall"],
        "answer_pr_auc": _average_precision(answer_scores, is_answer),
        "answer_f2": per_class["ANSWER"]["f2"],
        "expected_calibration_error": expected_calibration_error(confidences, correct),
        "per_class": per_class,
        "confusion": {
            LABELS[i]: dict(zip(LABELS, confusion[i], strict=True)) for i in range(len(LABELS))
        },
        "role": (
            "diagnostic. Selects no checkpoint and freezes no threshold -- those come from "
            "store_conditioned_metrics, which is the path the runtime actually takes."
        ),
    }

    for name, values in (("per_question_type", question_type), ("per_surface", surface)):
        if values is None:
            continue
        groups: dict[str, list[int]] = defaultdict(list)
        for i, value in enumerate(values):
            groups[str(value)].append(i)
        out[name] = {
            key: pair_level_metrics([gold[i] for i in rows], [probabilities[i] for i in rows])
            for key, rows in sorted(groups.items())
        }
    return out


# ------------------------------------------------- layer 2: store-conditioned rates --


def store_conditioned_metrics(
    rows: Sequence[StoreRow],
    *,
    tau_answer: float,
    tau_partial: float,
) -> dict:
    """Enforcement quality, measured the way the runtime decides.

    Every rate here is over requests, not pairs, and every one of them runs through
    routing first. The retain rate in particular is end-to-end: a retain row can only be a
    false alarm if the router matched it to a protected scope *and* the detector then fired
    on it. A retain row that routes to nothing is not a false alarm and is not silently
    dropped either -- it appears in ``n_retain_unrouted``, so a reader can tell "the
    detector was careful" from "the router never asked it".
    """
    protected = [r for r in rows if not r.is_retain]
    retain = [r for r in rows if r.is_retain]

    answer_rows = [r for r in protected if r.gold_label == "ANSWER"]
    none_rows = [r for r in protected if r.gold_label == "NONE"]
    partial_rows = [r for r in protected if r.gold_label == "PARTIAL"]

    fired_answer = [r for r in answer_rows if r.fires(tau_answer)]
    micro_recall = len(fired_answer) / len(answer_rows) if answer_rows else None

    by_concept: dict[str, list[StoreRow]] = defaultdict(list)
    for row in answer_rows:
        by_concept[row.gold_concept_id].append(row)
    per_concept = {
        concept: {
            "n": len(concept_rows),
            "recall": sum(1 for r in concept_rows if r.fires(tau_answer)) / len(concept_rows),
        }
        for concept, concept_rows in sorted(by_concept.items())
    }
    concept_recalls = [entry["recall"] for entry in per_concept.values()]
    zero_recall = sorted(c for c, entry in per_concept.items() if entry["recall"] == 0.0)

    # Correct-concept precision: of the protected ANSWER rows that fired, how many fired
    # on the concept the row is actually about. A detector that fires on the right text
    # under the wrong author has caught nothing -- it has produced a false alarm that
    # happens to coincide with a leak, and enforcement would then propagate the wrong
    # Forget-ID.
    correct_concept = sum(
        1 for r in fired_answer if r.gold_concept_id in r.fired_forget_ids(tau_answer)
    )
    return {
        "layer": "store_conditioned_enforcement",
        "tau_answer": tau_answer,
        "tau_partial": tau_partial,
        "protected": {
            "n_rows": len(protected),
            "n_routed": sum(1 for r in protected if r.routed),
            "n_unrouted": sum(1 for r in protected if not r.routed),
            "answer_micro_recall": micro_recall,
            "answer_macro_recall": (
                sum(concept_recalls) / len(concept_recalls) if concept_recalls else None
            ),
            "worst_concept_recall": min(concept_recalls) if concept_recalls else None,
            "n_zero_recall_concepts": len(zero_recall),
            "zero_recall_concepts": zero_recall,
            "correct_concept_precision": (
                correct_concept / len(fired_answer) if fired_answer else None
            ),
            "nonanswer_fpr": (
                sum(1 for r in none_rows if r.fires(tau_answer)) / len(none_rows)
                if none_rows
                else None
            ),
            "partial_recall": (
                sum(
                    1
                    for r in partial_rows
                    if (r.partial_score() or 0.0) >= tau_partial or r.fires(tau_answer)
                )
                / len(partial_rows)
                if partial_rows
                else None
            ),
            "n_answer": len(answer_rows),
            "n_nonanswer": len(none_rows),
            "n_partial": len(partial_rows),
            "per_concept_recall": per_concept,
        },
        "retain": {
            "n_rows": len(retain),
            "n_routed": sum(1 for r in retain if r.routed),
            "n_unrouted": sum(1 for r in retain if not r.routed),
            "end_to_end_fpr": (
                sum(1 for r in retain if r.fires(tau_answer)) / len(retain) if retain else None
            ),
            "why": (
                "end to end: the request is routed against the PROTECTED store first, so a "
                "retain row can only count as a false alarm if the router matched it to a "
                "protected scope and the detector then fired. v4.2 instead scored each "
                "retain candidate against its own retain question and called a correct "
                "ANSWER a false alarm, which contradicted the training objective on 300 of "
                "1,019 rows."
            ),
        },
    }


def select_thresholds(
    rows: Sequence[StoreRow],
    *,
    fpr_ceiling: float = DEV_FPR_CEILING,
    partial_fpr_ceiling: float = DEV_FPR_CEILING,
) -> dict:
    """Choose ``tau_answer`` and ``tau_partial`` on the store-conditioned layer.

    ``tau_answer`` maximises protected ANSWER micro recall subject to both false-alarm
    ceilings -- protected-clean and end-to-end retain -- exactly as the fresh gate applies
    them. Ties break toward the *higher* threshold, so a tie is resolved toward firing
    less.

    ``tau_partial`` is chosen separately and afterwards, never by a weighted combination.
    A partial clue and a complete answer call for different actions, and a single blended
    score would let two clues outvote a disclosure.
    """
    protected_answer = [r for r in rows if not r.is_retain and r.gold_label == "ANSWER"]
    protected_clean = [r for r in rows if not r.is_retain and r.gold_label == "NONE"]
    retain = [r for r in rows if r.is_retain]

    if not protected_answer or not protected_clean:
        return {
            "tau_answer": None,
            "tau_partial": None,
            "note": (
                "the development pool carries no protected ANSWER rows or no protected "
                "NONE rows, so neither the recall being maximised nor the ceiling "
                "constraining it has a denominator."
            ),
            "n_protected_answer": len(protected_answer),
            "n_protected_clean": len(protected_clean),
            "n_retain": len(retain),
        }

    scores = [s for r in rows for _sid, _fid, s, _p in r.scored]
    breakpoints = sorted({*scores, (max(scores) + 1e-9) if scores else 1.0})

    best: tuple[float, float, float, float] | None = None
    for tau in breakpoints:
        clean_fpr = sum(1 for r in protected_clean if r.fires(tau)) / len(protected_clean)
        retain_fpr = sum(1 for r in retain if r.fires(tau)) / len(retain) if retain else 0.0
        if clean_fpr > fpr_ceiling or retain_fpr > fpr_ceiling:
            continue
        recall = sum(1 for r in protected_answer if r.fires(tau)) / len(protected_answer)
        candidate = (recall, tau, clean_fpr, retain_fpr)
        if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
            best = candidate

    if best is None:
        return {
            "tau_answer": None,
            "tau_partial": None,
            "note": (
                f"no threshold clears both the protected-clean ceiling {fpr_ceiling} and "
                f"the end-to-end retain ceiling {partial_fpr_ceiling}."
            ),
            "n_protected_answer": len(protected_answer),
            "n_protected_clean": len(protected_clean),
            "n_retain": len(retain),
        }

    recall, tau_answer, clean_fpr, retain_fpr = best

    # tau_partial: the highest threshold whose PARTIAL alarms on protected-clean rows stay
    # under the ceiling. Chosen after tau_answer and independently of it, because the
    # accumulator it feeds is a different mechanism with a different cost.
    partial_rows = [r for r in rows if not r.is_retain and r.gold_label == "PARTIAL"]
    partial_scores = sorted({p for r in rows for _s, _f, _a, p in r.scored})
    tau_partial = None
    best_partial_recall = None
    for tau in partial_scores:
        clean_partial_fpr = (
            sum(1 for r in protected_clean if (r.partial_score() or 0.0) >= tau)
            / len(protected_clean)
            if protected_clean
            else 0.0
        )
        if clean_partial_fpr > partial_fpr_ceiling:
            continue
        recall_partial = (
            sum(1 for r in partial_rows if (r.partial_score() or 0.0) >= tau) / len(partial_rows)
            if partial_rows
            else None
        )
        if tau_partial is None or tau > tau_partial:
            tau_partial = tau
            best_partial_recall = recall_partial

    return {
        "tau_answer": tau_answer,
        "tau_partial": tau_partial,
        "selected_answer_micro_recall": recall,
        "selected_protected_clean_fpr": clean_fpr,
        "selected_retain_end_to_end_fpr": retain_fpr,
        "selected_partial_recall": best_partial_recall,
        "n_protected_answer": len(protected_answer),
        "n_protected_clean": len(protected_clean),
        "n_retain": len(retain),
        "fpr_ceiling": fpr_ceiling,
        "note": (
            "chosen on the store-conditioned layer, the same path the fresh gate scores. "
            "The two thresholds are selected separately and are never combined into one "
            "score."
        ),
    }
