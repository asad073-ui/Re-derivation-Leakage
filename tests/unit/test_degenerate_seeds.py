"""Five reruns are not five seeds.

Generation is greedy (`do_sample=False`, `num_beams=1`), the checkpoint is fixed, and —
before the fix — the item order was fixed too. Under those conditions the "5 seeds" are
five copies of one number. Bootstrapping five copies gives a zero-width interval, which
reads as extraordinary precision and carries no information whatsoever. Worse, a
zero-width interval trivially satisfies "non-overlapping 95% CIs", so the pre-registered
gate could be cleared by a run with no replication at all.
"""

from __future__ import annotations

import numpy as np
import pytest

from rdl.eval.aggregate import (
    condition_delta_gate,
    is_degenerate,
    paired_bootstrap_delta,
    paired_delta_gate,
    summarise_seeds,
)


def test_identical_replicates_are_flagged():
    assert is_degenerate([0.4] * 5)
    assert not is_degenerate([0.40, 0.42, 0.39, 0.44, 0.41])


def test_zero_width_ci_is_labelled_rather_than_reported_as_precision():
    stats = summarise_seeds([0.4] * 5)
    d = stats.to_dict()
    assert d["ci95"] == [0.4, 0.4]
    assert d["degenerate_replicates"] is True
    assert "NOT an uncertainty estimate" in d["ci95_note"]


def test_degenerate_replicates_cannot_clear_the_non_overlap_criterion():
    """The exact failure the gate had to be closed against: 0.5 vs 0.0, five times
    each, produced two point intervals that "do not overlap" and a PASS."""
    gate = condition_delta_gate([0.5] * 5, [0.0] * 5, min_delta_points=20.0)
    assert gate.detail["delta_points"] == pytest.approx(50.0)
    assert gate.detail["degenerate_replicates"] is True
    assert not gate.passed
    assert "zero width" in gate.reason


def test_genuinely_varying_replicates_still_pass():
    treatment = [0.52, 0.48, 0.55, 0.50, 0.51]
    baseline = [0.10, 0.12, 0.09, 0.11, 0.08]
    gate = condition_delta_gate(treatment, baseline, min_delta_points=20.0)
    assert gate.passed, gate.reason
    assert gate.detail["degenerate_replicates"] is False


# =====================================================================================
# The paired item-level interval — the one that survives greedy decoding
# =====================================================================================


def test_paired_interval_is_informative_even_with_deterministic_decoding():
    """Items vary whether or not decoding does, so the item is the sampling unit."""
    n = 200
    rng = np.random.default_rng(0)
    treatment = (rng.random(n) < 0.55).astype(float)
    baseline = (rng.random(n) < 0.20).astype(float)

    res = paired_bootstrap_delta(treatment, baseline)
    lo, hi = res["ci95"]
    assert hi > lo, "a real interval, not a point"
    assert lo > 0.0, "the effect is distinguishable from zero"
    assert res["n_pairs"] == n


def test_paired_arms_must_be_aligned():
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_delta([1.0, 0.0], [1.0])


def test_cluster_resampling_widens_the_interval():
    """TOFU is 200 authors x 20 questions. Treating 400 correlated questions as 400
    independent observations understates uncertainty; resampling authors does not."""
    n_authors, per_author = 20, 20
    rng = np.random.default_rng(7)
    # Effect is per-author: whole authors leak or do not. Item-level resampling sees
    # 400 observations, cluster-level sees the 20 that actually vary.
    author_effect = rng.random(n_authors) < 0.5
    treatment = np.repeat(author_effect.astype(float), per_author)
    baseline = np.zeros_like(treatment)
    clusters = np.repeat(np.arange(n_authors), per_author).tolist()

    naive = paired_bootstrap_delta(treatment, baseline)
    clustered = paired_bootstrap_delta(treatment, baseline, clusters=clusters)

    naive_width = naive["ci95"][1] - naive["ci95"][0]
    clustered_width = clustered["ci95"][1] - clustered["ci95"][0]
    assert clustered_width > naive_width
    assert clustered["resampling_unit"] == "cluster"
    assert clustered["n_units"] == n_authors


def test_paired_gate_rejects_an_interval_that_includes_zero():
    rng = np.random.default_rng(3)
    same = (rng.random(200) < 0.4).astype(float)
    gate = paired_delta_gate(same, same.copy(), min_delta_points=20.0)
    assert not gate.passed
    assert "includes zero" in gate.reason
