"""C3S's source mapping must be a genuine negative control (ADR-0055).

The mapping it replaces was `derange(n, seed)`, which failed in two ways no test caught:

* **Seeded** — so the seed changed the TEXT agent B receives, not merely episode order.
  ADR-0050's argument for a single primary seed ("order cannot matter once the store
  resets per item") was therefore false for this arm.
* **Authorship-blind** — TOFU is 200 invented authors x 20 questions, so a random
  derangement paired 13-23 of 400 items with another question about the SAME author,
  which can carry the target name or its supporting facts outright.

These tests measure both properties directly rather than trusting the construction.
"""

from __future__ import annotations

import pytest

from rdl.eval.negatives import (
    NoCrossAuthorMapping,
    cross_author_mapping,
    derange,
)
from rdl.eval.tofu_data import QUESTIONS_PER_AUTHOR

TOFU_400 = [f"forget10-author-{i // QUESTIONS_PER_AUTHOR:04d}" for i in range(400)]


def test_the_canonical_mapping_is_shift_by_one_author_block():
    """On contiguous 20-question blocks the smallest cross-author shift IS 20, so the
    derived mapping equals `(i + 20) % 400` exactly — without hard-coding it."""
    m = cross_author_mapping(TOFU_400)
    assert m.shift == QUESTIONS_PER_AUTHOR == 20
    assert m.permutation == [(i + 20) % 400 for i in range(400)]


def test_the_mapping_has_no_fixed_points_and_no_same_author_pairs():
    m = cross_author_mapping(TOFU_400)
    assert m.fixed_point_count == 0
    assert m.same_author_count == 0
    assert sorted(m.permutation) == list(range(400)), "still a permutation"


def test_the_mapping_does_not_depend_on_any_seed():
    """THE defect. `cross_author_mapping` takes no seed at all, so `C3C - C3S` cannot rest
    on one arbitrary distractor assignment."""
    first = cross_author_mapping(TOFU_400)
    for _ in range(5):
        assert cross_author_mapping(TOFU_400).permutation == first.permutation
        assert cross_author_mapping(TOFU_400).sha256 == first.sha256


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_old_seeded_derangement_did_pair_same_author_items(seed):
    """The measurement that motivated the change: a random derangement puts roughly 5% of
    items with another question about the same invented novelist."""
    perm = derange(400, seed)
    same_author = sum(1 for i, j in enumerate(perm) if TOFU_400[i] == TOFU_400[j])
    assert same_author > 0, (
        "if this ever reaches zero the derangement got lucky, not correct — the point is "
        "that nothing in its construction forbids a same-author pair"
    )
    assert cross_author_mapping(TOFU_400).same_author_count == 0


def test_a_spread_sampled_pilot_gets_a_shift_of_one():
    """Consecutive items already differ in author, so shift 1 suffices. Deriving the shift
    rather than hard-coding 20 is what keeps the property true at pilot scale."""
    spread = [f"forget10-author-{i:04d}" for i in range(5)]
    m = cross_author_mapping(spread)
    assert m.shift == 1
    assert m.same_author_count == 0 and m.fixed_point_count == 0


def test_a_head_truncated_pilot_is_refused_outright():
    """THE five-item smoke test as it stood: `--limit 5` off the head of a contiguous
    split gives five questions about ONE author, for which no cross-author partner
    exists. A contaminated control is worse than no run."""
    one_author = ["forget10-author-0000"] * 5
    with pytest.raises(NoCrossAuthorMapping, match="one author"):
        cross_author_mapping(one_author)


def test_two_items_by_two_authors_is_the_smallest_workable_set():
    m = cross_author_mapping(["a", "b"])
    assert m.permutation == [1, 0]
    with pytest.raises(NoCrossAuthorMapping):
        cross_author_mapping(["a"])


def test_the_mapping_hashes_its_own_content():
    a = cross_author_mapping(TOFU_400)
    b = cross_author_mapping([f"x-{i // 20:04d}" for i in range(400)])
    assert a.sha256 == b.sha256, "same algorithm, shift and permutation -> same hash"
    c = cross_author_mapping([f"y-{i:04d}" for i in range(400)])
    assert c.shift == 1 and c.sha256 != a.sha256


def test_the_audit_dict_carries_everything_a_reviewer_needs():
    d = cross_author_mapping(TOFU_400).to_dict()
    assert d["algorithm"] == "rotate-by-smallest-cross-author-shift"
    assert d["shift"] == 20
    assert d["n_items"] == 400 and d["n_authors"] == 20
    assert d["same_author_count"] == 0 and d["fixed_point_count"] == 0
    assert len(d["sha256"]) == 64
    assert len(d["permutation"]) == 400
