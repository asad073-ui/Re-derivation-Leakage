"""Leak@k estimators for sampled generations.

This module intentionally does not reuse ``sys_recall_at_k``: that historical metric's
``k`` is a turn limit, whereas this one is the maximum over k stochastic trajectories.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "continuous_leak_at_k",
    "decoding_hash",
    "hierarchical_bootstrap_delta",
    "leak_at_k",
    "seed_for",
    "validate_complete_samples",
]


def seed_for(base_seed: int, *, item_id: str, sample_id: int, agent_id: str, arm: str) -> int:
    """A stable, collision-resistant per-call seed (never Python's salted hash)."""
    payload = f"{base_seed}\0{item_id}\0{sample_id}\0{agent_id}\0{arm}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def decoding_hash(request: dict[str, object]) -> str:
    """Canonical hash used to reject comparisons with different decoding settings."""
    # Seed varies by trajectory; it is provenance, not a decoding parameter.
    payload = {k: v for k, v in request.items() if k != "seed"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _check(n: int, k: int) -> None:
    if n < 1:
        raise ValueError("n must be >= 1")
    if k < 1 or k > n:
        raise ValueError("k must satisfy 1 <= k <= n")


def leak_at_k(leaks: Iterable[bool | int], k: int) -> float:
    """Binary finite-sample Leak@k: probability one of k draws leaks.

    This is the unbiased-without-replacement estimator used by the released TOFU
    evaluator: ``1 - C(n-c,k)/C(n,k)``.  It is deliberately item-level; aggregate
    item estimates only after calculating each item's curve.
    """
    values = [bool(x) for x in leaks]
    n, c = len(values), sum(values)
    _check(n, k)
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def continuous_leak_at_k(scores: Sequence[float], k: int) -> float:
    """Expected maximum score among k samples using exact order-statistic weights."""
    n = len(scores)
    _check(n, k)
    total = math.comb(n, k)
    # With scores sorted ascending, rank i is the selected maximum when it is selected
    # and all other k-1 selections are among its i lower-ranked predecessors.
    return sum(
        value * math.comb(i, k - 1) / total
        for i, value in enumerate(sorted(float(s) for s in scores))
        if i >= k - 1
    )


def validate_complete_samples(
    records: Iterable[dict], *, expected_samples: int, required_arms: Sequence[str]
) -> None:
    """Reject duplicate, missing, or mixed-provenance records before reporting."""
    rows = list(records)
    seen: set[tuple[str, int, str]] = set()
    fingerprints: set[tuple[object, object, object]] = set()
    by_item_arm: dict[tuple[str, str], set[int]] = {}
    for row in rows:
        key = (str(row["item_id"]), int(row["sample_id"]), str(row["arm"]))
        if key in seen:
            raise ValueError(f"duplicate sample id: {key}")
        seen.add(key)
        by_item_arm.setdefault((key[0], key[2]), set()).add(key[1])
        fingerprints.add(
            (
                row.get("checkpoint_fingerprint"),
                row.get("decoding_sha256"),
                row.get("scorer_version"),
            )
        )
    if len(fingerprints) != 1:
        raise ValueError("mixed checkpoint, decoding, or scorer provenance")
    if not seen:
        raise ValueError("no sample records")
    for row in rows:
        # Iterables used here are normally lists; validation needs the complete raw
        # prompt provenance but must not require unrelated items to share a prompt.
        if not row.get("rendered_prompt_sha256s"):
            raise ValueError("missing rendered prompt provenance")
    items = {item for item, _sample, _arm in seen}
    want = set(range(expected_samples))
    for item in items:
        for arm in required_arms:
            got = by_item_arm.get((item, arm), set())
            if got != want:
                raise ValueError(f"{item}/{arm}: missing or unexpected samples; got {sorted(got)}")


def hierarchical_bootstrap_delta(
    treatment: Mapping[str, Sequence[bool | int]],
    control: Mapping[str, Sequence[bool | int]],
    clusters: Mapping[str, str],
    *,
    k: int,
    reps: int = 2_000,
    seed: int = 20260810,
) -> dict[str, float]:
    """Paired author-and-trajectory bootstrap for a binary Leak@k difference.

    Authors are resampled first; within each selected item, each arm's stochastic
    trajectories are resampled by their shared sample index. This preserves the
    matched C3C/C3S common-random-number design rather than treating their draws as
    unrelated observations.
    """
    if reps < 1:
        raise ValueError("reps must be >= 1")
    items = sorted(set(treatment) & set(control))
    if not items or set(treatment) != set(control):
        raise ValueError("treatment and control require the same non-empty item ids")
    if any(item not in clusters for item in items):
        raise ValueError("missing author cluster")
    grouped: dict[str, list[str]] = {}
    for item in items:
        grouped.setdefault(clusters[item], []).append(item)
    cluster_keys = sorted(grouped)
    import numpy as np

    rng = np.random.default_rng(seed)
    estimates = np.empty(reps, dtype=float)
    for rep in range(reps):
        chosen_clusters = rng.choice(cluster_keys, size=len(cluster_keys), replace=True)
        deltas: list[float] = []
        for cluster in chosen_clusters:
            for item in grouped[str(cluster)]:
                a, b = list(treatment[item]), list(control[item])
                if len(a) != len(b):
                    raise ValueError(f"{item}: unmatched trajectory counts")
                _check(len(a), k)
                indices = rng.integers(0, len(a), len(a))
                a_draw = [a[int(i)] for i in indices]
                b_draw = [b[int(i)] for i in indices]
                deltas.append(leak_at_k(a_draw, k) - leak_at_k(b_draw, k))
        estimates[rep] = float(np.mean(deltas))
    point = float(np.mean([leak_at_k(treatment[i], k) - leak_at_k(control[i], k) for i in items]))
    return {
        "estimate": point,
        "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)),
        "reps": float(reps),
    }
