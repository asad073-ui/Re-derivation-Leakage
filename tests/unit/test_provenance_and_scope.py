"""Provenance the gate REQUIRES, and the longitudinal grid as its own experiment.

Two ways the previous gate said more than it checked:

  * "agent B on the final clean commit is now enforced" was not true.
    `measured_checkpoints` was `{r["checkpoint"]: r for r in measures}` — existence
    alone. A measure-only report satisfied the requirement whatever its tree state,
    evaluator SHA, tokenizer or commit (ADR-0061).
  * `scale_blockers` rejected `git_dirty is True` and nothing else, so a condition report
    with `git_dirty: null` and no runtime block at all was accepted, and `_fingerprint`
    read a field missing on BOTH sides as agreement (ADR-0061).

And one the reporter simply did not do: `evaluate_gates` loaded the cumulative reports
and returned `sorted(longitudinal)` — a list of names — so `--set
episode.store_scope=cumulative` re-printed the already-valid per-item verdict and looked
like it had evaluated Phase F2 (ADR-0062).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_report_gate import (
    AGENT_A,
    AGENT_A_REV,
    AGENT_B,
    AGENT_B_REV,
    FULL,
    FULL_REV,
    _grid,
    _measure,
    _report,
    _repro,
)
from test_v5_reporting_repairs import _render

from rdl.cli import make_report as mr

PINNED = mr.pinned_ou_source_sha()


def _runs(*, measure_kw: dict | None = None, repro_kw: dict | None = None) -> list[dict]:
    return [
        *[r for r in _grid() if r.get("phase") == "phase0_days3-5"],
        _repro("full", FULL, FULL_REV, **(repro_kw or {})),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV, **(repro_kw or {})),
        _measure(AGENT_B, AGENT_B_REV, **(measure_kw or {})),
    ]


# =====================================================================================
# ADR-0061 — agent B's characterisation is held to the same provenance
# =====================================================================================


def test_a_clean_same_commit_agent_b_measurement_passes():
    """Guard against a gate so strict nothing can clear it."""
    verdict = mr.evaluate_gates(_runs())
    assert verdict["experiment_valid"], verdict["blockers"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("git_dirty", None, "not false"),
        ("git_dirty", True, "not false"),
        ("ou_source_sha", None, "no ou_source_sha"),
        ("ou_source_sha", "0" * 40, "but this repo pins"),
        ("tokenizer", {}, "chat-template hash"),
        ("transformers_version", None, "no transformers_version"),
        ("ou_runtime_mode", None, "no ou_runtime_mode"),
    ],
)
def test_an_agent_b_measurement_without_provenance_blocks_the_grid(field, value, expected):
    """A measurement whose evaluator, tree state and tokenizer are unrecorded characterises
    nothing, and C3D/C3C/B1W are built on it."""
    verdict = mr.evaluate_gates(_runs(measure_kw={field: value}))
    assert not verdict["experiment_valid"]
    hits = [b for b in verdict["blockers"] if AGENT_B in b and expected in b]
    assert hits, verdict["blockers"]


def test_an_agent_b_measurement_from_another_commit_blocks_the_grid():
    """Days 1-2 and the grid must be one program: a measurement from an older checkout
    characterises the checkpoint under code the grid did not run."""
    verdict = mr.evaluate_gates(_runs(measure_kw={"git_sha": "deadbee"}))
    assert not verdict["experiment_valid"]
    assert any(
        AGENT_B in b and "characterised at commit" in b for b in verdict["blockers"]
    ), verdict["blockers"]


def test_a_day1_reproduction_from_another_commit_is_not_accepted():
    verdict = mr.evaluate_gates(_runs(repro_kw={"git_sha": "deadbee"}))
    assert not verdict["experiment_valid"]
    assert any("never at EXACT published parity" in b for b in verdict["blockers"]), verdict[
        "blockers"
    ]


def test_a_measurement_at_the_wrong_checkpoint_revision_still_blocks():
    """Unchanged behaviour, kept under test because the surrounding code moved."""
    runs = _runs()
    runs[-1] = _measure(AGENT_B, "f" * 40)
    verdict = mr.evaluate_gates(runs)
    assert any("Days 1-2 evaluated revision" in b for b in verdict["blockers"]), verdict["blockers"]


def test_same_commit_is_prefix_tolerant_but_never_matches_nothing():
    """Condition reports carry the SHORT sha; some Day-1 fields carry the full 40."""
    assert mr.same_commit("1a0eb6b", "1a0eb6b70583c2f81e1ff10c9497e2e077b6ab85")
    assert not mr.same_commit("1a0eb6b", "deadbee")
    assert not mr.same_commit(None, "1a0eb6b")
    assert not mr.same_commit("nogit", "nogit")
    assert not mr.same_commit("", "")


# =====================================================================================
# ADR-0061 — condition reports must say what produced them
# =====================================================================================


def test_a_fully_provenanced_condition_report_has_no_gaps():
    assert mr.condition_provenance_gaps(_report("C3C", recall=0.6)) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda r: r.update({"git_dirty": None}), "git_dirty is None"),
        (lambda r: r.update({"git_dirty": True}), "git_dirty is True"),
        (lambda r: r.update({"git_sha": "nogit"}), "git_sha is 'nogit'"),
        (lambda r: r.update({"git_sha": ""}), "git_sha is ''"),
        (lambda r: r.update({"runtime": {}}), "no runtime.transformers_version"),
        (lambda r: r["runtime"].pop("torch_version"), "no runtime.torch_version"),
        (lambda r: r["runtime"].update({"tokenizer": {}}), "chat_template_sha256"),
        (lambda r: r["runtime"].update({"models": {}}), "records no HF model"),
        (
            lambda r: [m.pop("resolved_attn") for m in r["runtime"]["models"].values()],
            "no resolved_attn",
        ),
        (
            lambda r: [m.pop("resolved_dtype") for m in r["runtime"]["models"].values()],
            "no resolved_dtype",
        ),
    ],
)
def test_a_condition_report_missing_provenance_is_a_gap(mutate, expected):
    r = _report("C3C", recall=0.6)
    mutate(r)
    gaps = mr.condition_provenance_gaps(r)
    assert any(expected in g for g in gaps), gaps
    assert any(expected in b for b in mr.scale_blockers({"C3C": r})), r


def test_a_legacy_condition_report_invalidates_the_grid():
    """`git_dirty: null` plus no runtime block — the shape of every report written before
    ADR-0054, which the old check accepted."""
    runs = _grid()
    legacy = _report("C3C", recall=0.6)
    legacy["git_dirty"] = None
    legacy.pop("runtime")
    runs[0] = legacy
    verdict = mr.evaluate_gates(runs)
    assert not verdict["experiment_valid"]
    assert any("cannot say what produced it" in b for b in verdict["blockers"])


def test_conditions_produced_at_two_commits_cannot_be_differenced():
    runs = _grid()
    runs[1] = _report("C3S", recall=0.35, git_sha="deadbee")
    verdict = mr.evaluate_gates(runs)
    assert not verdict["experiment_valid"]
    assert any("different commits" in b for b in verdict["blockers"]), verdict["blockers"]


def test_one_commit_across_the_whole_grid_is_the_passing_case():
    verdict = mr.evaluate_gates(_grid())
    assert not any("different commits" in b for b in verdict["blockers"])
    assert verdict["experiment_valid"], verdict["blockers"]


# =====================================================================================
# ADR-0062 — the longitudinal grid is a separate experiment with its own verdict
# =====================================================================================


def _cumulative_grid() -> list[dict]:
    """A complete Phase F2 grid: cumulative scope, five seeds, same commit."""
    return [
        *[
            _report(c, recall=r, store_scope="cumulative", n_seeds=5)
            for c, r in (
                ("C3C", 0.60),
                ("C3S", 0.35),
                ("C3D", 0.30),
                ("C1W", 0.12),
                ("B1W", 0.14),
            )
        ],
        _repro("full", FULL, FULL_REV),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV),
        _measure(AGENT_B, AGENT_B_REV),
    ]


def test_the_cumulative_scope_gets_its_own_verdict_not_a_reprint():
    """THE gap. Both scopes present; the cumulative verdict must be computed from the
    cumulative arms, not handed back the per-item one."""
    runs = [*_grid(), *[r for r in _cumulative_grid() if r.get("phase") == "phase0_days3-5"]]
    per_item = mr.evaluate_gates(runs, "per_item")
    cumulative = mr.evaluate_gates(runs, "cumulative")

    assert per_item["store_scope"] == "per_item"
    assert cumulative["store_scope"] == "cumulative"
    assert per_item["is_primary_experiment"] and not cumulative["is_primary_experiment"]
    assert cumulative["conditions_evaluated"] == ["B1W", "C1W", "C3C", "C3D", "C3S"]
    assert cumulative["other_scope"] == "per_item"


def test_a_cumulative_grid_at_one_seed_is_blocked_in_its_own_scope():
    """Five seeds is the cumulative requirement; the per-item verdict cannot speak to it."""
    runs = [
        _report(c, recall=r, store_scope="cumulative", n_seeds=1)
        for c, r in (("C3C", 0.6), ("C3S", 0.35), ("C3D", 0.3), ("C1W", 0.12), ("B1W", 0.14))
    ]
    verdict = mr.evaluate_gates(runs, "cumulative")
    assert not verdict["experiment_valid"]
    assert any("store_scope=cumulative" in b and "fixes 5" in b for b in verdict["blockers"])


def test_a_cumulative_only_results_dir_does_not_produce_a_per_item_verdict():
    """The failure this prevents: running F2 and reading the per-item gate as its result."""
    runs = [r for r in _cumulative_grid() if r.get("phase") == "phase0_days3-5"]
    per_item = mr.evaluate_gates(runs, "per_item")
    assert per_item["conditions_evaluated"] == []
    assert not per_item["experiment_valid"]
    assert any("did not run" in b for b in per_item["blockers"])


def test_an_unknown_scope_is_rejected(tmp_path, monkeypatch):
    import typer

    rd = tmp_path / "results"
    rd.mkdir(parents=True)
    monkeypatch.setattr(mr, "collect_runs", lambda *a, **k: _grid())
    monkeypatch.setattr(mr, "results_dir", lambda *a, **k: rd)
    monkeypatch.setattr(mr, "manifest_path", lambda *a, **k: rd / "manifest.jsonl")
    with pytest.raises(typer.Exit):
        mr.make_report(out=rd / "R.md", figures=False, gate=True, scope="per-item")


def test_the_cumulative_report_is_written_to_its_own_files(tmp_path, monkeypatch):
    body = _render(tmp_path, monkeypatch, _cumulative_grid(), scope="cumulative")
    rd = tmp_path / "results"
    assert (rd / "gate_verdict_cumulative.json").exists()
    assert not (rd / "gate_verdict.json").exists(), "F2 must not overwrite the primary verdict"
    written = json.loads((rd / "gate_verdict_cumulative.json").read_text(encoding="utf-8"))
    assert written["store_scope"] == "cumulative"
    assert "LONGITUDINAL experiment (Phase F2), not the primary" in body


def test_the_primary_report_says_it_is_the_primary(tmp_path, monkeypatch):
    body = _render(tmp_path, monkeypatch, _grid())
    assert "PRIMARY per-item experiment" in body
    assert (tmp_path / "results" / "gate_verdict.json").exists()


def test_the_condition_table_carries_a_store_scope_column(tmp_path, monkeypatch):
    """Without it a per-item C3C and a cumulative C3C are two rows with the same name and
    different numbers, which reads as run-to-run noise."""
    runs = [*_grid(), *[r for r in _cumulative_grid() if r.get("phase") == "phase0_days3-5"]]
    table = mr.markdown_table(runs)
    assert "store_scope" in table
    assert "`per_item`" in table and "`cumulative`" in table
    # And the scoped view shows only one of them.
    scoped = mr.markdown_table(runs, "cumulative")
    assert "`per_item`" not in scoped and "`cumulative`" in scoped


def test_the_figures_are_per_scope_and_do_not_overwrite_each_other(tmp_path):
    pytest.importorskip("matplotlib")
    rd = tmp_path / "results"
    rd.mkdir(parents=True)
    runs = [*_grid(), *[r for r in _cumulative_grid() if r.get("phase") == "phase0_days3-5"]]
    primary = [p.name for p in mr._figures(runs, rd, mr.evaluate_gates(runs, "per_item"))]
    cumulative = [
        p.name for p in mr._figures(runs, rd, mr.evaluate_gates(runs, "cumulative"), "cumulative")
    ]
    assert primary and cumulative
    assert not set(primary) & set(cumulative), (primary, cumulative)
    assert all(n.endswith("_cumulative.png") for n in cumulative)


def test_the_runbook_f2_command_uses_the_cumulative_scope():
    """The runbook is the thing an operator actually types on the box."""
    runbook = (Path(__file__).resolve().parents[2] / "docs" / "07_rtx3090_runbook.md").read_text(
        encoding="utf-8"
    )
    f2 = runbook.split("Phase F2", 1)[1]
    assert "--scope cumulative" in f2, "F2 must gate its own scope, not reprint the per-item one"
