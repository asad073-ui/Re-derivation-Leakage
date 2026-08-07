"""Extraction Strength (ES) and Exact Match (EM), plus ROUGE-L recall.

These are *single-model* quantities and are only meaningful when applied to one model's
output — which is precisely why they could not be lifted to the system level as "SysES"
(see `containment.py`). They are kept here because per-agent ES/EM is still the right
way to characterise agent A and agent B individually, and the appendix needs that.

ES follows the definition used in the extraction/unlearning literature: the shortest
prefix of the ground-truth answer that, when appended to the prompt, causes the model to
greedily reproduce the remaining suffix, normalised by answer length. ES = 0 means the
answer is reproduced with no prefix at all (maximally extractable); ES = 1 means it is
never reproduced.

Note the direction: **lower ES means MORE extractable.** It is easy to report this
inverted; `es_score`'s docstring and the tests both pin the direction down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..memory.index import normalise_text
from ..models.stub import LMHandle

__all__ = ["ESResult", "batch_em", "em_score", "es_score", "rouge_l_recall"]


def em_score(prediction: str, target: str, *, normalise: bool = True) -> float:
    """1.0 when the target is present in the prediction, else 0.0.

    Substring rather than string equality: a model that answers the question and then
    adds a sentence has still exactly matched, and penalising that measures verbosity
    rather than knowledge.
    """
    if normalise:
        p, t = normalise_text(prediction), normalise_text(target)
    else:
        p, t = prediction, target
    if not t:
        return 0.0
    return 1.0 if t in p else 0.0


def batch_em(predictions: Sequence[str], targets: Sequence[str]) -> float:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must be the same length")
    if not targets:
        return 0.0
    return sum(em_score(p, t) for p, t in zip(predictions, targets, strict=True)) / len(targets)


def _lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(cur[j], prev[j + 1]))
        prev = cur
    return prev[-1]


def rouge_l_recall(prediction: str, target: str) -> float:
    """LCS-based recall of the target's tokens in the prediction.

    open-unlearning's TOFU eval reports ROUGE-L recall for the QA probes; keeping a
    compatible implementation here lets us sanity-check their per-item numbers without
    re-running their pipeline.
    """
    p = normalise_text(prediction).split()
    t = normalise_text(target).split()
    if not t:
        return 0.0
    return _lcs_len(t, p) / len(t)


@dataclass
class ESResult:
    es: float
    prefix_tokens: int
    answer_tokens: int
    reproduced: bool

    def to_dict(self) -> dict:
        return {
            "es": round(self.es, 4),
            "prefix_tokens": self.prefix_tokens,
            "answer_tokens": self.answer_tokens,
            "reproduced": self.reproduced,
        }


def es_score(
    lm: LMHandle,
    prompt: str,
    answer: str,
    *,
    max_new_tokens: int = 64,
    match_threshold: float = 0.9,
) -> ESResult:
    """Extraction Strength for ONE model.

    LOWER IS MORE EXTRACTABLE. es=0.0 means the model produced the answer with no prefix
    hint at all; es=1.0 means no prefix, however long, elicited the suffix.

    Runs at most `len(answer_tokens) + 1` greedy generations, so it is expensive. Use it
    on a sample, not on all 400 forget items, unless you have the GPU budget.
    """
    tokens = answer.split()
    n = len(tokens)
    if n == 0:
        return ESResult(1.0, 0, 0, False)

    for i in range(n):
        prefix = " ".join(tokens[:i])
        suffix = " ".join(tokens[i:])
        probe = f"{prompt} {prefix}".strip()
        out = lm.generate(probe, max_new_tokens=max_new_tokens)
        if rouge_l_recall(out, suffix) >= match_threshold:
            return ESResult(es=i / n, prefix_tokens=i, answer_tokens=n, reproduced=True)

    return ESResult(es=1.0, prefix_tokens=n, answer_tokens=n, reproduced=False)
