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

from collections.abc import Sequence

import numpy as np

__all__ = ["derange", "deranged_targets"]


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
    """
    perm = derange(len(answers), seed)
    return [answers[j] for j in perm], perm
