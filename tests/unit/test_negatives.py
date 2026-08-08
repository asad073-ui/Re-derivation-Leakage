"""The false-positive floor's negative targets.

The v2 floor scored each retain episode against ITS OWN CORRECT ANSWER. Retain questions
were never unlearned, so a working agent answers them, `framework_default` write-back
persists the answer, and the store then contains it — the "false-positive floor" of a
healthy system approaches 1.0 and the pre-registered `<= 0.05` gate fails every
functioning run. It only ever passed in CI because the stub has no retain knowledge and
abstains, so nothing is written at all. See ADR-0043.
"""

from __future__ import annotations

from rdl.eval.negatives import derange, deranged_targets


def test_no_item_is_paired_with_its_own_answer():
    """A fixed point would put the correct answer back in the negative set, which is the
    exact confusion the module exists to remove."""
    for n in (2, 3, 8, 100, 400):
        perm = derange(n, seed=0)
        assert sorted(perm) == list(range(n)), "still a permutation"
        assert all(perm[i] != i for i in range(n)), f"fixed point at n={n}"


def test_the_derangement_is_deterministic_in_the_seed():
    assert derange(50, seed=3) == derange(50, seed=3)
    assert derange(50, seed=3) != derange(50, seed=4)


def test_degenerate_sizes_do_not_raise():
    """n < 2 has no derangement at all. The caller reports the floor as undefined; the
    metric must not crash the run to say so."""
    assert derange(0, seed=0) == []
    assert derange(1, seed=0) == [0]


def test_targets_are_real_answers_from_other_items():
    """Deranged, not synthetic: the floor has to be measured on text with TOFU's own
    length, vocabulary and entity density, or it measures a strawman."""
    answers = [f"The answer is {i}." for i in range(20)]
    targets, perm = deranged_targets(answers, seed=7)
    assert len(targets) == len(answers)
    assert all(t in answers for t in targets)
    assert all(t != a for t, a in zip(targets, answers, strict=True))
    assert targets == [answers[j] for j in perm], "the permutation must reproduce the targets"
