"""The trained answerability backend, and the encoder both it and the trainer use.

This is the deployable half of Detector v4: a cross-encoder that reads the protected
question, the routed identity context and one candidate clause in a single attention
window and emits a distribution over {NONE, PARTIAL, ANSWER}. It is the class
``LexicalAnswerabilityDetector`` calls itself the floor for.

It differs from the lexical backend in the SCORER ONLY. Segmentation, routing, per-scope
accumulation, span-free reporting and the no-gold-answer guarantee are the same, so a
comparison between the two is a comparison of scorers rather than of pipelines.

Why the encoder lives here and not in the training script
---------------------------------------------------------
A model trained on one segmentation of its inputs and served on another is a different
model, silently. :func:`budget_encode` is therefore imported by ``scripts/train_detector_v4.py``
rather than reimplemented there, and the budget it applies is recorded in the manifest.

The budget itself is the fix for a real defect. v4's trainer encoded

    tokenizer(question, identity + "[SEP]" + candidate, truncation="only_second")

which truncates the END of the second sequence — the candidate, the one span that must
survive, since it is the evidence the verdict is about. Truncation here spends the budget
in the opposite order:

1. the candidate is reserved first, up to everything the question's floor leaves;
2. identity aliases are dropped from the end, one at a time;
3. the protected question is shortened, never below :data:`MIN_QUESTION_TOKENS`;
4. only then is the candidate cut — and every such cut is COUNTED, in the training
   manifest and in :meth:`CrossEncoderAnswerabilityDetector.stats`, because a candidate
   that did not fit is an evaluation error rather than a NONE.

Integration
-----------
Not selectable from ``GraphDetectorConfig``. A detector reaches a study after its gate has
been opened and passed, and v4.1's gate is opened on a bank that does not yet exist
(``FINAL_GATE_BANK_MANIFEST.json``). ``tests/contract/test_detector_v4_integration_contract.py``
asserts the backend list is still ``("hashing64",)``.
"""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .atomic_text import SEGMENTATION_VERSION, segment
from .detection_context import DetectionContext, ProtectedQuestion
from .detector_protocol import AnswerabilityResult

__all__ = [
    "CROSS_ENCODER_PROTOCOL_REVISION",
    "DEFAULT_MAX_LENGTH",
    "MIN_QUESTION_TOKENS",
    "CrossEncoderAnswerabilityDetector",
    "EncodingBudget",
    "budget_encode",
    "identity_context",
]

CROSS_ENCODER_PROTOCOL_REVISION = "answerability-v4-cross-encoder-1"
LABELS: tuple[str, ...] = ("NONE", "PARTIAL", "ANSWER")
DEFAULT_MAX_LENGTH = 256

# The protected question is the only thing that says WHICH relation is being checked. Below
# this many tokens it stops naming one, and the model is being asked a question nobody
# posed. A budget that cannot seat both a floor-length question and the candidate is a
# configuration error, not a truncation.
MIN_QUESTION_TOKENS = 8

# How many aliases of the routed subject reach the model at most. Beyond a handful they are
# repetition, and they are the first thing the budget spends.
MAX_ALIASES = 4


@dataclass(frozen=True)
class EncodingBudget:
    """Token budget and its floor. Recorded in the model manifest and in ``to_dict``."""

    max_length: int = DEFAULT_MAX_LENGTH
    min_question_tokens: int = MIN_QUESTION_TOKENS
    max_aliases: int = MAX_ALIASES

    def to_dict(self) -> dict:
        return {
            "max_length": self.max_length,
            "min_question_tokens": self.min_question_tokens,
            "max_aliases": self.max_aliases,
            "truncation_order": [
                "reserve the candidate",
                "drop identity aliases from the end",
                "shorten the protected question, never below min_question_tokens",
                "cut the candidate — counted as an error",
            ],
        }


def identity_context(aliases: Sequence[str], *, max_aliases: int = MAX_ALIASES) -> str:
    """The routed subject description. Never the raw request.

    Putting the request verbatim into the same window as the candidate is the failure
    ``DetectionContext`` exists to prevent: the model would learn that the presence of a
    forget question predicts the label, and under ``graph_flow`` that question is at the
    root of every guarded trajectory.
    """
    seen: list[str] = []
    for alias in aliases:
        text = str(alias).strip()
        if text and text not in seen:
            seen.append(text)
    return ", ".join(seen[:max_aliases])


def _encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False)) if text else []


def budget_encode(
    tokenizer: Any,
    *,
    question: str,
    aliases: Sequence[str],
    candidate: str,
    budget: EncodingBudget | None = None,
) -> tuple[dict, dict]:
    """``(model inputs, truncation stats)`` for one (question, identity, candidate) triple.

    Segment A is ``question [SEP] identity``; segment B is the candidate alone. The
    candidate being its own sequence is what makes "the evidence survives" a property of
    the encoding rather than of the input lengths.
    """
    budget = budget or EncodingBudget()
    identity = identity_context(aliases, max_aliases=budget.max_aliases)

    n_special = 3
    with contextlib.suppress(Exception):  # not every tokenizer exposes it
        n_special = int(tokenizer.num_special_tokens_to_add(pair=True))
    room = max(1, budget.max_length - n_special)

    question_ids = _encode(tokenizer, question)
    identity_ids = _encode(tokenizer, identity)
    candidate_ids = _encode(tokenizer, candidate)

    # 1. The candidate is reserved first, up to whatever the question's floor leaves.
    keep_candidate = min(len(candidate_ids), max(0, room - budget.min_question_tokens))
    # 2/3. Whatever is left is spent on the question, then on the identity aliases.
    remaining = room - keep_candidate
    keep_question = min(len(question_ids), max(0, remaining))
    keep_identity = min(len(identity_ids), max(0, remaining - keep_question))

    aliases_kept = identity
    if keep_identity < len(identity_ids):
        # Drop whole aliases from the end rather than cutting one in half: half a name is
        # not an identity signal, and a router that matched it would be matching noise.
        parts = [p for p in identity.split(", ") if p]
        while parts and len(_encode(tokenizer, ", ".join(parts))) > keep_identity:
            parts.pop()
        aliases_kept = ", ".join(parts)
        identity_ids = _encode(tokenizer, aliases_kept)

    question_text = question
    if keep_question < len(question_ids):
        question_text = _decode(tokenizer, question_ids[:keep_question])
    candidate_text = candidate
    if keep_candidate < len(candidate_ids):
        candidate_text = _decode(tokenizer, candidate_ids[:keep_candidate])

    left = f"{question_text} [SEP] {aliases_kept}" if aliases_kept else question_text
    encoding = tokenizer(
        left,
        candidate_text,
        truncation=True,
        max_length=budget.max_length,
    )
    stats = {
        "candidate_truncated": keep_candidate < len(candidate_ids),
        "n_candidate_tokens_dropped": max(0, len(candidate_ids) - keep_candidate),
        "question_truncated": keep_question < len(question_ids),
        "n_aliases_dropped": max(
            0,
            len([p for p in identity.split(", ") if p])
            - len([p for p in aliases_kept.split(", ") if p]),
        ),
        "n_candidate_tokens": len(candidate_ids),
        "n_question_tokens": len(question_ids),
    }
    return dict(encoding), stats


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    try:
        return str(tokenizer.decode(list(ids), skip_special_tokens=True))
    except Exception:  # pragma: no cover - reporting, not control flow
        return ""


def _softmax(row: Sequence[float]) -> list[float]:
    """Pure Python, so this module imports no tensor library.

    The detector has to be constructible from a fake model in a unit test, and a defence
    that could only be exercised with torch installed would be a defence nobody tests on
    the CPU gate.
    """
    if not row:
        return []
    top = max(row)
    exponentials = [math.exp(float(v) - top) for v in row]
    total = sum(exponentials) or 1.0
    return [e / total for e in exponentials]


@contextlib.contextmanager
def _inference_mode() -> Iterator[None]:
    try:
        import torch
    except ImportError:  # pragma: no cover - the fake-model tests take this path
        yield
        return
    with torch.inference_mode():
        yield


class CrossEncoderAnswerabilityDetector:
    """A trained answerability cross-encoder, behind :class:`ConceptDetector`.

    ``model`` and ``tokenizer`` are injected rather than loaded, so the class is testable
    against a tiny in-process module and ``make cpu-all`` stays network-free.
    :meth:`from_artifact` is the loading path and is the only place ``transformers`` is
    imported.
    """

    backend = "answerability_v4_cross_encoder"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_repo_id: str = "",
        model_revision: str = "",
        tokenizer_revision: str = "",
        checkpoint: str = "",
        answer_threshold: float = 0.5,
        partial_threshold: float = 0.5,
        accumulate: bool = True,
        budget: EncodingBudget | None = None,
        batch_size: int = 32,
        calibrated: bool = False,
        labels: Sequence[str] = LABELS,
    ) -> None:
        if not 0.0 < answer_threshold <= 1.0:
            raise ValueError("answer_threshold must be in (0, 1]")
        if not 0.0 < partial_threshold <= 1.0:
            raise ValueError("partial_threshold must be in (0, 1]")
        if tuple(labels) != LABELS:
            raise ValueError(f"the label order is part of the checkpoint; expected {LABELS}")
        self.model = model
        self.tokenizer = tokenizer
        self.model_repo_id = model_repo_id
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.checkpoint = checkpoint
        self.answer_threshold = answer_threshold
        self.partial_threshold = partial_threshold
        self.accumulate = accumulate
        self.budget = budget or EncodingBudget()
        self.batch_size = max(1, int(batch_size))
        self.calibrated = bool(calibrated)
        self.n_calls = 0
        self.n_candidates = 0
        self.n_clauses = 0
        self.n_unrouted_calls = 0
        self.n_accumulated_tags = 0
        self.n_forward_pairs = 0
        self.n_candidate_truncations = 0

    # ------------------------------------------------------------------ loading --

    @classmethod
    def from_artifact(cls, artifact: Path, **overrides: Any) -> CrossEncoderAnswerabilityDetector:
        """Build from a ``DETECTOR_V4_MODEL.json`` written by the trainer.

        Both revisions are mandatory and are read from the artifact rather than guessed.
        A checkpoint whose weights or whose SEGMENTATION cannot be named is not eligible
        for a study: a moved model tag changes the detector, and a moved tokenizer tag
        changes the subword split, which changes every score at a fixed threshold.
        """
        manifest = json.loads(Path(artifact).read_text(encoding="utf-8"))
        pins = manifest.get("pins", {})
        missing = [
            name
            for name in ("model_repo_id", "model_revision", "tokenizer_revision")
            for value in (str(pins.get(name, "")),)
            if not value
        ]
        if missing:
            raise ValueError(
                f"{artifact} does not pin {missing}. A checkpoint whose weights or "
                "tokenizer cannot be named is not eligible for a study."
            )
        checkpoint = str(manifest.get("selected_checkpoint") or Path(artifact).parent)
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            pins["model_repo_id"], revision=pins["tokenizer_revision"]
        )
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        model.eval()
        calibration = manifest.get("calibration", {}) or {}
        kwargs: dict[str, Any] = {
            "model_repo_id": str(pins["model_repo_id"]),
            "model_revision": str(pins["model_revision"]),
            "tokenizer_revision": str(pins["tokenizer_revision"]),
            "checkpoint": checkpoint,
            "answer_threshold": float(calibration.get("answer_threshold", 0.5)),
            "partial_threshold": float(calibration.get("partial_threshold", 0.5)),
            "budget": EncodingBudget(
                max_length=int(pins.get("max_length", DEFAULT_MAX_LENGTH)),
                min_question_tokens=int(pins.get("min_question_tokens", MIN_QUESTION_TOKENS)),
                max_aliases=int(pins.get("max_aliases", MAX_ALIASES)),
            ),
            "calibrated": bool(calibration),
        }
        kwargs.update(overrides)
        return cls(model, tokenizer, **kwargs)

    def with_thresholds(
        self, *, answer_threshold: float, partial_threshold: float | None = None
    ) -> CrossEncoderAnswerabilityDetector:
        """A copy at a different operating point, SHARING the model and the tokenizer.

        The threshold sweep builds twenty of these. Reloading the weights each time would
        make the sweep an hour of I/O, and — worse — would let two points in the same grid
        sit on different bytes if a cache were repopulated mid-run.
        """
        return CrossEncoderAnswerabilityDetector(
            self.model,
            self.tokenizer,
            model_repo_id=self.model_repo_id,
            model_revision=self.model_revision,
            tokenizer_revision=self.tokenizer_revision,
            checkpoint=self.checkpoint,
            answer_threshold=answer_threshold,
            partial_threshold=(
                self.partial_threshold if partial_threshold is None else partial_threshold
            ),
            accumulate=self.accumulate,
            budget=self.budget,
            batch_size=self.batch_size,
            calibrated=self.calibrated,
        )

    # --------------------------------------------------------------- provenance --

    @property
    def revision(self) -> str:
        return (
            f"{CROSS_ENCODER_PROTOCOL_REVISION}:{self.model_repo_id or 'unpinned'}"
            f"@{self.model_revision or 'unpinned'}:tok={self.tokenizer_revision or 'unpinned'}"
            f":answer={self.answer_threshold:.3f}:partial={self.partial_threshold:.3f}"
            f":{SEGMENTATION_VERSION}"
        )

    def to_dict(self) -> dict:
        return {
            "protocol": "concept-detector-v4",
            "backend": self.backend,
            "revision": self.revision,
            "model_repo_id": self.model_repo_id,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "selected_checkpoint": self.checkpoint,
            "answer_threshold": self.answer_threshold,
            "partial_threshold": self.partial_threshold,
            "segmentation_version": SEGMENTATION_VERSION,
            "encoding_budget": self.budget.to_dict(),
            "accumulates_partial_evidence": self.accumulate,
            "reports_answerability": True,
            "calibrated": self.calibrated,
            "receives_gold_answers": False,
            "selectable_from_graph_detector_config": False,
            "role": (
                "the trained v4 answerability backend. It is not selectable from a study "
                "configuration until its gate has been opened once, on the fresh bank "
                "pre-registered in FINAL_GATE_BANK_MANIFEST.json, and passed."
            ),
        }

    def stats(self) -> dict:
        return {
            "detector_calls": self.n_calls,
            "detector_candidates": self.n_candidates,
            "detector_clauses": self.n_clauses,
            "detector_unrouted_calls": self.n_unrouted_calls,
            "accumulated_tags": self.n_accumulated_tags,
            "forward_pairs": self.n_forward_pairs,
            # A candidate that did not fit the window is an evaluation error, not a NONE.
            # A run whose gate was measured with this above zero has to say so.
            "candidate_truncations": self.n_candidate_truncations,
        }

    # ------------------------------------------------------------------ scoring --

    def _forward(self, pairs: Sequence[tuple[ProtectedQuestion, str]]) -> list[list[float]]:
        """``[(question, clause_text), ...] -> [[p_none, p_partial, p_answer], ...]``."""
        if not pairs:
            return []
        out: list[list[float]] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            encodings = []
            for question, text in chunk:
                encoding, stats = budget_encode(
                    self.tokenizer,
                    question=question.question,
                    aliases=question.aliases,
                    candidate=text,
                    budget=self.budget,
                )
                self.n_candidate_truncations += int(bool(stats["candidate_truncated"]))
                encodings.append(encoding)
            batch = self._collate(encodings)
            with _inference_mode():
                output = self.model(**batch)
            logits = getattr(output, "logits", output)
            rows = logits.tolist() if hasattr(logits, "tolist") else [list(r) for r in logits]
            out.extend(_softmax(row) for row in rows)
            self.n_forward_pairs += len(chunk)
        return out

    def _collate(self, encodings: Sequence[Mapping]) -> dict:
        """Right-pad the batch. Uses the tokenizer's pad id when it exposes one."""
        pad = int(getattr(self.tokenizer, "pad_token_id", 0) or 0)
        keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in encodings[0]]
        width = max(len(e["input_ids"]) for e in encodings)
        batch: dict[str, Any] = {}
        for key in keys:
            filler = pad if key == "input_ids" else 0
            batch[key] = [
                list(e[key]) + [filler] * (width - len(e[key])) for e in encodings  # type: ignore[index]
            ]
        try:
            import torch

            return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}
        except ImportError:  # pragma: no cover - the fake-model tests take this path
            return batch

    def score_batch(
        self,
        candidates: Sequence[str],
        *,
        context: DetectionContext,
        restrict_to: frozenset[str] | None = None,
    ) -> list[AnswerabilityResult]:
        """Score candidates against the protected questions the request routed to.

        Same shape as the lexical backend: unrouted calls never reach the scorer,
        accumulation is per call and ordered by candidate index, and nothing survives the
        call. A detector that remembered fragments across trajectories would make one
        trajectory's verdict depend on another's.
        """
        self.n_calls += 1
        self.n_candidates += len(candidates)
        if not candidates:
            return []

        questions = context.questions_for(restrict_to)
        if not questions:
            self.n_unrouted_calls += 1
            return [
                AnswerabilityResult(
                    detector_revision=self.revision, threshold=self.answer_threshold
                )
                for _ in candidates
            ]

        scope_to_forget = {q.scope_id: q.forget_id for q in questions}
        # ``{scope_id: the fragment text still awaiting a completion}``.
        pending: dict[str, str] = {}
        results: list[AnswerabilityResult] = []

        for candidate in candidates:
            clauses = segment(candidate or "")
            self.n_clauses += len(clauses)
            pairs: list[tuple[ProtectedQuestion, str]] = []
            provenance: list[tuple[str, int, bool]] = []  # (scope_id, clause_index, joined)
            for clause in clauses:
                for question in questions:
                    pairs.append((question, clause.text))
                    provenance.append((question.scope_id, clause.index, False))
                    # The accumulation probe: the fragment an earlier candidate left open,
                    # joined to this clause, scored as one span. The model sees the
                    # accumulated evidence rather than being asked to imagine it.
                    fragment = pending.get(question.scope_id)
                    if self.accumulate and fragment:
                        pairs.append((question, f"{fragment} {clause.text}".strip()))
                        provenance.append((question.scope_id, clause.index, True))

            masses = self._forward(pairs)
            best: dict[str, tuple[float, float, int]] = {}
            best_alone: dict[str, float] = {}
            opened: dict[str, str] = {}
            for (scope_id, clause_index, joined), row in zip(provenance, masses, strict=True):
                answer, partial = row[2], row[1]
                previous = best.get(scope_id)
                if previous is None or (answer, partial) > (previous[0], previous[1]):
                    best[scope_id] = (answer, partial, clause_index)
                if not joined:
                    best_alone[scope_id] = max(best_alone.get(scope_id, 0.0), answer)
                    if partial >= self.partial_threshold and answer < self.answer_threshold:
                        opened[scope_id] = clauses[clause_index].text

            fired_scopes = tuple(
                sorted(s for s, (answer, _p, _i) in best.items() if answer >= self.answer_threshold)
            )
            # An accumulation only when the clause could not have carried the tag alone.
            accumulated = {
                s
                for s in fired_scopes
                if best_alone.get(s, 0.0) < self.answer_threshold and s in pending
            }
            results.append(self._collect(best, fired_scopes, scope_to_forget, accumulated))

            if self.accumulate:
                for scope_id, text in opened.items():
                    pending[scope_id] = text
                for scope_id in fired_scopes:
                    pending.pop(scope_id, None)

        return results

    def score(
        self,
        candidate: str,
        *,
        context: DetectionContext,
        restrict_to: frozenset[str] | None = None,
    ) -> AnswerabilityResult:
        return self.score_batch([candidate], context=context, restrict_to=restrict_to)[0]

    def _collect(
        self,
        best: Mapping[str, tuple[float, float, int]],
        fired_scopes: tuple[str, ...],
        scope_to_forget: Mapping[str, str],
        accumulated: set[str],
    ) -> AnswerabilityResult:
        per_concept: dict[str, float] = {}
        for scope_id, (answer, _partial, _index) in best.items():
            forget_id = scope_to_forget[scope_id]
            per_concept[forget_id] = max(per_concept.get(forget_id, 0.0), answer)
        fired_ids = tuple(sorted({scope_to_forget[s] for s in fired_scopes}))
        top = max(best, key=lambda s: best[s][0], default=None)
        answer, partial, clause_index = best.get(top, (0.0, 0.0, 0)) if top else (0.0, 0.0, 0)
        from_accumulated = bool(fired_scopes) and set(fired_scopes) <= accumulated
        if from_accumulated:
            self.n_accumulated_tags += 1
        return AnswerabilityResult(
            forget_ids=fired_ids,
            answer_probability=answer,
            partial_probability=partial,
            selected_scope_ids=fired_scopes,
            # A cross-encoder over a whole clause has no token-level attribution that
            # would survive review, so it reports the CLAUSE rather than inventing offsets
            # inside it. A span nobody can defend is worse than no span.
            evidence_span=None,
            candidate_clause_index=clause_index if fired_scopes else None,
            detector_revision=self.revision,
            threshold=self.answer_threshold,
            fired=bool(fired_ids),
            score=answer,
            per_concept=per_concept,
            from_accumulated_evidence=from_accumulated,
        )
