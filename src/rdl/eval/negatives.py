"""Negative targets — the false-positive floor's denominator of last resort.

The v2 floor was measured by scoring each retain episode against **its own correct
answer**. Retain questions were never unlearned, so a working agent answers them, the
`framework_default` write policy persists the answer, and the store then contains it.
The "false-positive floor" of a healthy system therefore approached 1.0, and the
pre-registered `<= 0.05` gate would have failed every functioning run. See ADR-0043.

A false-positive floor has to be measured against a target the system should NOT be able
to produce. This module builds those targets by **deranging** the retain answers: item
*i* is scored against the answer of a different retain item. That keeps the length,
vocabulary, register and entity density of real TOFU text — so the number measures the
containment matcher's actual tolerance rather than its behaviour on a strawman string.

The permutation is a derangement (no fixed points) and is a deterministic function of the
run seed, so the floor is recomputable from the report without a rerun.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CrossAuthorMapping",
    "cross_author_mapping",
    "derange",
    "deranged_targets",
]


def derange(n: int, seed: int = 0) -> list[int]:
    """A permutation of ``range(n)`` with no fixed points, deterministic in `seed`.

    Sattolo's algorithm: it produces a single n-cycle, which is a derangement for every
    ``n >= 2`` by construction rather than by rejection sampling. Rejection would also
    work but its running time depends on the draw, and a metric's inputs should not.

    ``n < 2`` has no derangement at all; the identity is returned and the caller is
    expected to report the floor as undefined rather than as zero.
    """
    if n < 2:
        return list(range(n))
    idx = list(range(n))
    rng = np.random.default_rng(seed)
    for i in range(n - 1, 0, -1):
        j = int(rng.integers(0, i))  # strictly below i — this is what forbids fixed points
        idx[i], idx[j] = idx[j], idx[i]
    return idx


def deranged_targets(answers: Sequence[str], seed: int = 0) -> tuple[list[str], list[int]]:
    """Pair each item with another item's answer. Returns ``(targets, permutation)``.

    The permutation is returned so it can go into the run report verbatim: a floor whose
    negative targets cannot be reconstructed is not auditable.

    This is the FALSE-POSITIVE FLOOR's mapping, where a same-author pairing is harmless —
    the question is whether the matcher fires on text the system never produced, and
    another question about the same author is a *harder* negative, not a contaminated
    one. C3S needs the opposite guarantee and uses `cross_author_mapping` instead.
    """
    perm = derange(len(answers), seed)
    return [answers[j] for j in perm], perm


@dataclass(frozen=True)
class CrossAuthorMapping:
    """A seed-independent permutation that never pairs two items by the same author.

    Returned rather than a bare list because every one of these facts has to reach the
    run report: a control whose mapping cannot be recomputed and audited is not a control.
    """

    permutation: list[int]
    shift: int
    n_items: int
    n_authors: int
    fixed_point_count: int
    same_author_count: int
    algorithm: str

    @property
    def sha256(self) -> str:
        """Hash of the mapping itself, so two runs can be shown to have used one map."""
        payload = json.dumps(
            {"algorithm": self.algorithm, "shift": self.shift, "permutation": self.permutation},
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "shift": self.shift,
            "n_items": self.n_items,
            "n_authors": self.n_authors,
            "fixed_point_count": self.fixed_point_count,
            "same_author_count": self.same_author_count,
            "sha256": self.sha256,
            "permutation": self.permutation,
        }


class NoCrossAuthorMapping(ValueError):
    """Raised when no rotation avoids same-author pairs — e.g. a single-author item set."""


def cross_author_mapping(cluster_ids: Sequence[str]) -> CrossAuthorMapping:
    """C3S's source mapping: rotate by the smallest shift that crosses authorship.

    THE BUG THIS REPLACES. C3S used `derange(n, seed)`, which has two defects that no
    test caught:

    1. **It is seeded.** The seed therefore changed *the text agent B receives*, not
       merely episode order — so the v4 argument for a single primary seed ("order cannot
       matter once the store resets per item") was false for this arm, and `C3C - C3S`
       rested on one arbitrary distractor assignment.
    2. **It ignores authorship.** TOFU is 200 synthetic authors x 20 questions
       (`QUESTIONS_PER_AUTHOR`), so a random derangement pairs ~5% of items with another
       question about the SAME author — 13 to 23 of 400 depending on the seed. Those are
       not irrelevant distractors: another question about the same invented novelist can
       carry the target name or the supporting facts outright. The control then leaks the
       very content it exists to withhold, which biases `C3C - C3S` toward zero and would
       have been read as "the wrapper explains everything".

    A rotation by whole author blocks fixes both. On the canonical contiguous forget10
    layout the smallest working shift is exactly `QUESTIONS_PER_AUTHOR`, i.e.

        source_index = (target_index + 20) % 400

    — every question mapped to the same question position for the next author. Searching
    for the smallest valid shift rather than hard-coding 20 keeps the property true for a
    spread-sampled pilot, where consecutive items already belong to different authors and
    a shift of 1 suffices. The chosen shift is recorded either way.

    Raises `NoCrossAuthorMapping` when no rotation works, which means the item set cannot
    support the control — a single-author slice, most often a truncated pilot that took
    the head of the split instead of sampling across authors.
    """
    n = len(cluster_ids)
    if n < 2:
        raise NoCrossAuthorMapping(
            f"a cross-author mapping needs at least 2 items, got {n}. C3S cannot run on "
            "this item set."
        )
    authors = list(cluster_ids)
    n_authors = len(set(authors))
    if n_authors < 2:
        raise NoCrossAuthorMapping(
            f"all {n} items belong to one author ({authors[0]!r}), so every handoff would "
            "carry another question about the SAME author — exactly the contamination "
            "C3S exists to avoid. Sample across authors (`sample='spread'`) or run the "
            "full split."
        )

    for shift in range(1, n):
        if all(authors[i] != authors[(i + shift) % n] for i in range(n)):
            perm = [(i + shift) % n for i in range(n)]
            return CrossAuthorMapping(
                permutation=perm,
                shift=shift,
                n_items=n,
                n_authors=n_authors,
                fixed_point_count=sum(1 for i, j in enumerate(perm) if i == j),
                same_author_count=sum(1 for i, j in enumerate(perm) if authors[i] == authors[j]),
                algorithm="rotate-by-smallest-cross-author-shift",
            )

    raise NoCrossAuthorMapping(
        f"no rotation of {n} items avoids same-author pairs across {n_authors} author(s). "
        "The item set is too concentrated for this control."
    )
