"""Provenance and exact-parity invariants (ADR-0039, ADR-0040).

Each test here corresponds to a way a report claimed something the run did not do:

  * `git_sha: 1ea12bf` on a run whose recorded command invoked a shim that commit does
    not contain — a reviewer cannot reproduce it and nothing said so;
  * `published_parity: true` on a batch-32/seed-0 run under SDPA, because the
    predicate never looked at dtype or attention;
  * `deterministic_algorithms: true` describing the PARENT process while the kernels
    ran in a subprocess that never received the setting.
"""

from __future__ import annotations

from rdl.cli.make_report import report_is_exact_parity
from rdl.compat.ou_eval_shim import _seed_from_argv
from rdl.eval.openunlearning_bridge import is_exact_published_parity, is_published_parity

# --------------------------------------------------------------- exact parity --

_PARITY = {"batch_size": 32, "seed": 0, "dtype": "bfloat16", "attn": "flash_attention_2"}


def test_sdpa_at_batch32_seed0_is_published_parity_but_not_exact():
    """The precise hole the old gate had."""
    kw = {**_PARITY, "attn": "sdpa"}
    assert is_published_parity(batch_size=32, seed=0) is True
    assert is_exact_published_parity(**kw) is False


def test_float16_is_not_exact_parity():
    assert is_exact_published_parity(**{**_PARITY, "dtype": "float16"}) is False


def test_all_four_published_settings_are_exact_parity():
    assert is_exact_published_parity(**_PARITY) is True


def test_dirty_tree_is_never_exact_parity_however_good_the_settings():
    assert is_exact_published_parity(**_PARITY, git_dirty=True) is False
    assert is_exact_published_parity(**_PARITY, git_dirty=False) is True


def test_off_parity_batch_or_seed_is_not_exact():
    assert is_exact_published_parity(**{**_PARITY, "batch_size": 1}) is False
    assert is_exact_published_parity(**{**_PARITY, "seed": 42}) is False


# ------------------------------------------------------- the make-report gate --


def _report(**over) -> dict:
    base = {
        "published_parity": True,
        "parity_gaps": [],
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "git_dirty": False,
    }
    base.update(over)
    return base


def test_report_gate_accepts_a_clean_exact_parity_report():
    assert report_is_exact_parity(_report()) is True


def test_report_gate_rejects_nonempty_parity_gaps():
    """`passed and published_parity` used to be enough; gaps were never inspected."""
    assert report_is_exact_parity(_report(parity_gaps=["torch_dtype=float16"])) is False


def test_report_gate_rejects_dirty_tree():
    assert report_is_exact_parity(_report(git_dirty=True)) is False


def test_report_gate_rejects_legacy_report_without_parity_gaps():
    """Absent evidence is not evidence of parity."""
    legacy = _report()
    del legacy["parity_gaps"]
    assert report_is_exact_parity(legacy) is False


def test_report_gate_rejects_sdpa_even_with_empty_gaps():
    """Belt and braces: an inconsistent report must not slip through."""
    assert report_is_exact_parity(_report(attn_implementation="sdpa")) is False


def test_report_gate_rejects_non_parity_report():
    assert report_is_exact_parity(_report(published_parity=False)) is False


# ------------------------------------------------ subprocess determinism seed --


def test_seed_is_read_from_the_recorded_command():
    """The subprocess must be seeded with the value the command shows, not a flag."""
    assert _seed_from_argv(["src/eval.py", "seed=0", "eval.tofu.batch_size=32"]) == 0
    assert _seed_from_argv(["seed=42"]) == 42


def test_missing_seed_override_is_reported_as_none_not_guessed():
    assert _seed_from_argv(["src/eval.py", "eval.tofu.batch_size=32"]) is None
    assert _seed_from_argv(["seed=notanint"]) is None
