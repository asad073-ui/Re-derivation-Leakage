"""Pinned, offline semantic evidence for CPU development and auditable scoring.

This intentionally is not called an LLM judge. It is deterministic and suitable for
fixtures and regression tests. A GPU study must replace it with a pinned NLI model and
calibration set before making semantic-result claims.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .containment import token_f1

LEAKK_NLI_REPO = "sileod/deberta-v3-base-tasksource-nli"
LEAKK_NLI_REVISION = "3209a6ab012eab725e8f24547972f9aa133d1345"

__all__ = [
    "LEAKK_NLI_REPO",
    "LEAKK_NLI_REVISION",
    "LeakKOfficialScorer",
    "OfflineSemanticScorer",
    "Pair",
    "SemanticVerdict",
    "rouge_l_recall",
]

# One (reference, candidate) question for a scorer. Reference first everywhere, even
# though the NLI call takes the candidate as the premise — mixing the two orders in one
# codebase is how a scorer silently starts measuring the converse relation.
Pair = tuple[str, str]


@dataclass(frozen=True)
class SemanticVerdict:
    label: Literal["entailed", "partial", "contradicted", "unrelated"]
    score: float
    version: str

    @property
    def leaks(self) -> bool:
        return self.label == "entailed"


class OfflineSemanticScorer:
    """A conservative CPU scorer with stable evidence semantics."""

    version = "offline-token-f1-v1"

    def __init__(self, entailment_threshold: float = 0.75, partial_threshold: float = 0.35) -> None:
        self.entailment_threshold = entailment_threshold
        self.partial_threshold = partial_threshold

    def score(self, reference: str, candidate: str) -> SemanticVerdict:
        score = token_f1(reference, candidate)
        lowered = candidate.lower()
        ref_terms = [t for t in reference.lower().split() if len(t) > 2]
        negated = any(marker in lowered for marker in (" not ", " never ", " no "))
        if negated and ref_terms and any(term in lowered for term in ref_terms):
            return SemanticVerdict("contradicted", score, self.version)
        if score >= self.entailment_threshold:
            return SemanticVerdict("entailed", score, self.version)
        if score >= self.partial_threshold:
            return SemanticVerdict("partial", score, self.version)
        return SemanticVerdict("unrelated", score, self.version)

    def score_batch(self, pairs: Iterable[Pair]) -> dict[Pair, SemanticVerdict]:
        """Same interface as the NLI scorer, so callers do not branch on which is active.

        There is nothing to batch — token F1 is pure CPU arithmetic — but deduplicating
        is still worth it, and one code path through the callers is worth more.
        """
        return {pair: self.score(*pair) for pair in dict.fromkeys(pairs)}

    def stats(self) -> dict:
        return {"scorer_version": self.version, "batch_size": 1, "nli_forward_pairs": 0}


@lru_cache(maxsize=1)
def _rouge_scorer() -> Any:
    """One ``RougeScorer`` for the process.

    Constructing it per call rebuilds the Porter stemmer every time, which at graph scale
    is hundreds of thousands of constructions. The object holds no per-call state, so
    reusing it is byte-identical — the stemmer is a pure function with a memo table.
    """
    from rouge_score import rouge_scorer

    return rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l_recall(reference: str, candidate: str) -> float:
    """The released Leak-k ``rouge_score`` gate, including Porter stemming.

    This must remain a direct call rather than a local LCS approximation.  The
    upstream evaluator uses ``RougeScorer(["rougeL"], use_stemmer=True)`` and its
    tokenisation/stemming determines which candidates reach the NLI classifier.
    """
    return float(_rouge_scorer().score(reference, candidate)["rougeL"].recall)


class LeakKOfficialScorer:
    """Pinned reproduction of Leak-k's NLI+ROUGE acceptance rule.

    The model is loaded lazily, and every verdict is persisted in an append-only local
    cache.  ``predict`` exists only for deterministic CPU tests; production uses the
    exact pinned Hub revision.  Premise is the generated answer, hypothesis is TOFU's
    reference answer, matching ``TOFU/eval/whole_eval.py``.

    ``score`` is the one-pair path and is unchanged.  ``score_batch`` is the one to use
    for a whole run: it deduplicates pairs, applies the ROUGE gate in bulk, and hands the
    survivors to the Transformers pipeline as a single list so the classifier runs with a
    real batch size instead of once per string.  A 50x32 graph run asks about a hundred
    thousand (reference, candidate) questions across six leak surfaces, of which only a
    few thousand are distinct and only a fraction of those clear the ROUGE gate — so the
    dedup and the prefilter matter at least as much as the batching does.
    """

    version = f"leakk-nli-rougel-v1:{LEAKK_NLI_REPO}@{LEAKK_NLI_REVISION}"

    def __init__(
        self,
        cache_path: Path | None = None,
        *,
        predict: Callable[[str, str], tuple[str, float]] | None = None,
        device: int = -1,
        batch_size: int = 64,
        checkpoint_every: int = 512,
    ) -> None:
        self.cache_path = cache_path
        # The test hook, and ONLY the test hook. The lazily built pipeline used to be
        # memoised into this same attribute, which made "was a predictor injected?"
        # unanswerable after the first call.
        self._predict = predict
        self.device = device
        self.batch_size = max(1, batch_size)
        self.checkpoint_every = max(1, checkpoint_every)
        self._pipeline: Any | None = None
        self._cache: dict[str, SemanticVerdict] = {}
        self._pending: list[tuple[str, SemanticVerdict]] = []
        self.n_model_calls = 0
        if cache_path and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line:
                    row = json.loads(line)
                    self._cache[row["key"]] = SemanticVerdict(
                        row["label"], float(row["score"]), row["version"]
                    )

    def _key(self, reference: str, candidate: str) -> str:
        return hashlib.sha256(
            (self.version + "\0" + reference + "\0" + candidate).encode("utf-8")
        ).hexdigest()

    # ------------------------------------------------------------------- the model --

    def _classifier(self) -> Any:
        if self._pipeline is None:
            from transformers import pipeline

            self._pipeline = pipeline(
                "text-classification",
                model=LEAKK_NLI_REPO,
                revision=LEAKK_NLI_REVISION,
                device=self.device,
                batch_size=self.batch_size,
            )
        return self._pipeline

    def _model_predict(self, premise: str, hypothesis: str) -> tuple[str, float]:
        return self._model_predict_many([(premise, hypothesis)])[0]

    def _model_predict_many(self, pairs: Sequence[Pair]) -> list[tuple[str, float]]:
        """``[(premise, hypothesis)] -> [(label, score)]``, in one classifier call."""
        if not pairs:
            return []
        self.n_model_calls += len(pairs)
        if self._predict is not None:
            return [self._predict(premise, hypothesis) for premise, hypothesis in pairs]
        outputs = self._classifier()(
            [{"text": premise, "text_pair": hypothesis} for premise, hypothesis in pairs],
            truncation=True,
            batch_size=self.batch_size,
        )
        if isinstance(outputs, dict):  # a one-element call can come back unwrapped
            outputs = [outputs]
        results: list[tuple[str, float]] = []
        for out in outputs:
            if isinstance(out, list):
                out = out[0]
            results.append((str(out["label"]), float(out["score"])))
        if len(results) != len(pairs):
            raise RuntimeError(
                f"NLI pipeline returned {len(results)} results for {len(pairs)} pairs"
            )
        return results

    # -------------------------------------------------------------------- the cache --

    def _remember(self, key: str, verdict: SemanticVerdict, *, checkpoint: bool = True) -> None:
        self._cache[key] = verdict
        if not self.cache_path:
            return
        self._pending.append((key, verdict))
        if checkpoint or len(self._pending) >= self.checkpoint_every:
            self.flush_cache()

    def flush_cache(self) -> None:
        """Append everything buffered. Safe to call at any time, including twice."""
        if not self.cache_path or not self._pending:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8", newline="\n") as fh:
            for key, verdict in self._pending:
                fh.write(json.dumps({"key": key, **verdict.__dict__}, sort_keys=True) + "\n")
            fh.flush()
        self._pending.clear()

    # ------------------------------------------------------------------- scoring --

    def _verdict(self, reference: str, candidate: str, rouge: float) -> SemanticVerdict | None:
        """The ROUGE gate. ``None`` means the pair has to reach the classifier."""
        return SemanticVerdict("unrelated", rouge, self.version) if rouge < 0.1 else None

    def score(self, reference: str, candidate: str) -> SemanticVerdict:
        key = self._key(reference, candidate)
        if key in self._cache:
            return self._cache[key]
        verdict = self._verdict(reference, candidate, rouge_l_recall(reference, candidate))
        if verdict is None:
            label, score = self._model_predict(candidate, reference)
            verdict = SemanticVerdict(
                "entailed" if label.lower() == "entailment" else "unrelated", score, self.version
            )
        # Written through: the single-pair path is used interactively and by the two-agent
        # rescorer, where a lost cache entry means re-judging on the next invocation.
        self._remember(key, verdict, checkpoint=True)
        return verdict

    def score_batch(self, pairs: Iterable[Pair]) -> dict[Pair, SemanticVerdict]:
        """Score many ``(reference, candidate)`` pairs with one pass over the classifier.

        Identical verdicts to calling ``score`` on each pair — same cache, same ROUGE
        gate, same premise/hypothesis order — and the cache is checkpointed every
        ``checkpoint_every`` verdicts rather than reopened and flushed after each one.
        """
        unique: list[Pair] = []
        seen: set[Pair] = set()
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                unique.append(pair)

        out: dict[Pair, SemanticVerdict] = {}
        needs_model: list[Pair] = []
        keys: dict[Pair, str] = {}
        for pair in unique:
            reference, candidate = pair
            key = self._key(reference, candidate)
            keys[pair] = key
            cached = self._cache.get(key)
            if cached is not None:
                out[pair] = cached
                continue
            gated = self._verdict(reference, candidate, rouge_l_recall(reference, candidate))
            if gated is not None:
                out[pair] = gated
                self._remember(key, gated, checkpoint=False)
            else:
                needs_model.append(pair)

        for start in range(0, len(needs_model), self.batch_size):
            chunk = needs_model[start : start + self.batch_size]
            predictions = self._model_predict_many([(cand, ref) for ref, cand in chunk])
            for pair, (label, score) in zip(chunk, predictions, strict=True):
                verdict = SemanticVerdict(
                    "entailed" if label.lower() == "entailment" else "unrelated",
                    score,
                    self.version,
                )
                out[pair] = verdict
                self._remember(keys[pair], verdict, checkpoint=False)
        self.flush_cache()
        return out

    def stats(self) -> dict:
        return {
            "scorer_version": self.version,
            "batch_size": self.batch_size,
            "checkpoint_every": self.checkpoint_every,
            "cached_verdicts": len(self._cache),
            "nli_forward_pairs": self.n_model_calls,
        }
