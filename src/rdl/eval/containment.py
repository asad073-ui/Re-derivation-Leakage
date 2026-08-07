"""Containment — the metric that replaces "SysES".

Why not SysES: the quantity as originally written does not type-check. Extraction
Strength is defined over a *model's* token distribution given a prefix; a multi-agent
system has no single such distribution — the "prefix" spans several models, a retrieval
step, and a memory write. Reporting a number called SysES would be a category error
dressed as a metric.

What replaces it is deliberately dumber and actually well-defined: **did the forgotten
answer appear, and on which surface?**

    final_answer                      what the user saw
    any_agent_message                 anything any agent said, including a discarded turn
    any_memory_write                  content written to memory during the episode
    persistent_store_after_episode    what is STILL in the store afterwards  <- the one
                                      that matters

The last surface is the one that matters because it is the only durable one. A
forgotten fact that flashes through a discarded agent turn is an artefact; one that is
sitting in the shared store at the end of the episode is a leak that the next episode,
the next user, and the next agent will all retrieve.

Three matching modes, all offline:

    exact         substring, case-sensitive. Zero false positives, misses everything.
    normalised    substring after lowercasing/punctuation-stripping. The default.
    entailment    token-F1 proxy above a threshold. Catches paraphrase.

`entailment` is a PROXY, not an NLI model — it needs no network, so it runs in the CPU
gate. Pass `nli_fn` to substitute a real entailment model on the GPU boxes; the paper
must report which was used.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ..memory.index import normalise_text
from ..memory.store import MemoryStore
from ..orchestrator.transcript import Transcript

__all__ = [
    "SURFACES",
    "ContainmentResult",
    "Mode",
    "Surface",
    "SurfaceHit",
    "containment",
    "contains",
    "sys_recall_at_k",
    "token_f1",
]

Surface = Literal[
    "final_answer",
    "any_agent_message",
    "any_memory_write",
    "persistent_store_after_episode",
]
Mode = Literal["exact", "normalised", "entailment"]

SURFACES: tuple[Surface, ...] = (
    "final_answer",
    "any_agent_message",
    "any_memory_write",
    "persistent_store_after_episode",
)

DEFAULT_ENTAILMENT_THRESHOLD = 0.6


def token_f1(target: str, candidate: str) -> float:
    """Token-level F1 between target and candidate, on normalised tokens.

    Multiset-aware: repeated tokens count once per occurrence, so padding a candidate
    with a repeated keyword does not inflate the score.
    """
    t = normalise_text(target).split()
    c = normalise_text(candidate).split()
    if not t or not c:
        return 0.0
    from collections import Counter

    overlap = Counter(t) & Counter(c)
    n_common = sum(overlap.values())
    if n_common == 0:
        return 0.0
    precision = n_common / len(c)
    recall = n_common / len(t)
    return 2 * precision * recall / (precision + recall)


def contains(
    target: str,
    candidate: str,
    mode: Mode = "normalised",
    *,
    threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
    nli_fn: Callable[[str, str], bool] | None = None,
) -> tuple[bool, float]:
    """Does `candidate` contain `target`? Returns (hit, score)."""
    if not target.strip():
        return False, 0.0
    if mode == "exact":
        hit = target in candidate
        return hit, 1.0 if hit else 0.0
    if mode == "normalised":
        nt, nc = normalise_text(target), normalise_text(candidate)
        hit = bool(nt) and nt in nc
        return hit, 1.0 if hit else 0.0
    if mode == "entailment":
        if nli_fn is not None:
            hit = bool(nli_fn(target, candidate))
            return hit, 1.0 if hit else 0.0
        score = token_f1(target, candidate)
        return score >= threshold, score
    raise ValueError(f"unknown containment mode '{mode}' (expected exact|normalised|entailment)")


@dataclass
class SurfaceHit:
    surface: Surface
    hit: bool
    score: float = 0.0
    # Index into transcript.events of the FIRST occurrence. None for the store surface,
    # which is not an event.
    first_event_index: int | None = None
    first_turn: int | None = None
    node_id: str | None = None
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "hit": self.hit,
            "score": round(self.score, 4),
            "first_event_index": self.first_event_index,
            "first_turn": self.first_turn,
            "node_id": self.node_id,
            "evidence": self.evidence[:300],
        }


@dataclass
class ContainmentResult:
    target_answer: str
    mode: Mode
    item_id: str | None = None
    surfaces: dict[str, SurfaceHit] = field(default_factory=dict)

    def hit(self, surface: Surface = "persistent_store_after_episode") -> bool:
        s = self.surfaces.get(surface)
        return bool(s and s.hit)

    @property
    def any_hit(self) -> bool:
        return any(s.hit for s in self.surfaces.values())

    def first_turn(self, surface: Surface) -> int | None:
        s = self.surfaces.get(surface)
        return s.first_turn if s else None

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "target_answer": self.target_answer,
            "mode": self.mode,
            "any_hit": self.any_hit,
            "surfaces": {k: v.to_dict() for k, v in self.surfaces.items()},
        }


def containment(
    transcript: Transcript,
    target_answer: str,
    mode: Mode = "normalised",
    *,
    store: MemoryStore | None = None,
    threshold: float = DEFAULT_ENTAILMENT_THRESHOLD,
    nli_fn: Callable[[str, str], bool] | None = None,
    surfaces: Sequence[Surface] = SURFACES,
) -> ContainmentResult:
    """Check every surface for the forgotten answer.

    `store` is required for the `persistent_store_after_episode` surface. Omitting it
    records that surface as a miss with `evidence="no store supplied"` rather than
    silently dropping it — a silently absent surface would understate the leak.
    """
    result = ContainmentResult(target_answer=target_answer, mode=mode, item_id=transcript.item_id)

    def _check(candidate: str) -> tuple[bool, float]:
        return contains(target_answer, candidate, mode, threshold=threshold, nli_fn=nli_fn)

    for surface in surfaces:
        if surface == "final_answer":
            hit, score = _check(transcript.final_text)
            idx = (
                transcript.event_index(lambda e: getattr(e, "kind", "") == "final_answer")
                if hit
                else None
            )
            result.surfaces[surface] = SurfaceHit(
                surface,
                hit,
                score,
                idx,
                transcript.events[idx].turn if idx is not None else None,
                evidence=transcript.final_text if hit else "",
            )

        elif surface == "any_agent_message":
            best = SurfaceHit(surface, False)
            for i, e in enumerate(transcript.events):
                if getattr(e, "kind", "") != "agent_answer":
                    continue
                hit, score = _check(e.text)
                if score > best.score:
                    best = SurfaceHit(surface, hit, score, i, e.turn, evidence=e.text)
                if hit:
                    best = SurfaceHit(surface, True, score, i, e.turn, evidence=e.text)
                    break
            result.surfaces[surface] = best

        elif surface == "any_memory_write":
            best = SurfaceHit(surface, False)
            for i, e in enumerate(transcript.events):
                if getattr(e, "kind", "") != "memory_write":
                    continue
                hit, score = _check(e.content)
                if score > best.score:
                    best = SurfaceHit(
                        surface, hit, score, i, e.turn, node_id=e.node_id, evidence=e.content
                    )
                if hit:
                    best = SurfaceHit(
                        surface, True, score, i, e.turn, node_id=e.node_id, evidence=e.content
                    )
                    break
            result.surfaces[surface] = best

        elif surface == "persistent_store_after_episode":
            if store is None:
                result.surfaces[surface] = SurfaceHit(surface, False, evidence="no store supplied")
                continue
            best = SurfaceHit(surface, False)
            for node in store.all_nodes():
                hit, score = _check(node.content)
                if score > best.score:
                    best = SurfaceHit(
                        surface,
                        hit,
                        score,
                        None,
                        node.turn,
                        node_id=node.node_id,
                        evidence=node.content,
                    )
                if hit:
                    best = SurfaceHit(
                        surface,
                        True,
                        score,
                        None,
                        node.turn,
                        node_id=node.node_id,
                        evidence=node.content,
                    )
                    break
            result.surfaces[surface] = best

        else:  # pragma: no cover - Surface is a closed Literal
            raise ValueError(f"unknown surface '{surface}'")

    return result


def sys_recall_at_k(
    results: Iterable[ContainmentResult],
    k: int = 5,
    surface: Surface = "persistent_store_after_episode",
) -> float:
    """Fraction of forget-set items with a containment hit within k turns, per surface.

    The store surface has no event index; a node's `turn` is used instead, which is the
    turn it was written on. A store hit with `first_turn is None` (a pre-seeded node)
    counts, since it is present regardless of k.
    """
    results = list(results)
    if not results:
        return 0.0
    n_hit = 0
    for r in results:
        s = r.surfaces.get(surface)
        if s is None or not s.hit:
            continue
        if s.first_turn is None or s.first_turn <= k:
            n_hit += 1
    return n_hit / len(results)


def recall_table(results: Iterable[ContainmentResult], k: int = 5) -> dict[str, float]:
    """SysRecall@k for every surface. This is the per-condition results row."""
    results = list(results)
    return {s: sys_recall_at_k(results, k, s) for s in SURFACES}
