"""The PR #9 repairs: the grid's own invocation, the prose, and the Day-1 gate.

Each test here corresponds to a defect that was invisible to the existing suite because
the suite checked the arithmetic and not what the arithmetic was *reported as*:

  * `03_run_phase0_grid.sh` defaulted `SEEDS` to unset and then always passed
    `--seeds "$SEEDS"`, so the advertised scope-derived invocation died in Typer before
    the first condition ran;
  * `make_report` computed v5 quantities and rendered a v4 report — `gate_verdict.json`
    could be right while `REPORT.md` told the reader a different scientific story, and
    `content_specific_joint_recovery`, the primary metric, was not in any table;
  * `report_is_exact_parity` rejected only `git_dirty is True`, so absent provenance —
    which is what every report currently in `results/` has — read as clean provenance.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from test_report_gate import (
    AGENT_A,
    AGENT_A_REV,
    AGENT_B,
    AGENT_B_REV,
    FULL,
    FULL_REV,
    _day1,
    _grid,
    _measure,
    _report,
    _repro,
)

from rdl.cli import make_report as mr

REPO = Path(__file__).resolve().parents[2]
GRID = REPO / "scripts" / "03_run_phase0_grid.sh"


# =====================================================================================
# 1. the grid script must not pass an empty --seeds
# =====================================================================================


def test_the_grid_script_never_passes_seeds_unconditionally():
    """`SEEDS="${SEEDS:-}"` next to a hard-coded `--seeds "$SEEDS"` is the whole bug."""
    src = GRID.read_text(encoding="utf-8")
    # The flag may appear ONLY inside the conditional array assignment. Anywhere else it
    # is passed unconditionally, and with SEEDS unset that is `--seeds ""` — an empty
    # string where Typer expects an integer, so the grid dies before its first condition.
    mentions = [
        ln.strip() for ln in src.splitlines() if "--seeds" in ln and not ln.lstrip().startswith("#")
    ]
    assert mentions, "the script must still be able to forward an explicit seed count"
    assert all(ln.startswith("SEED_ARGS=(") for ln in mentions), mentions
    assert "${SEED_ARGS[@]" in src, "the flag must be expanded conditionally"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def _run_grid(tmp_path: Path, env_extra: dict[str, str]) -> list[list[str]]:
    """Run the grid script with a stub `python` that records its argv and exits 0."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    log = tmp_path / "argv.log"
    (bindir / "python").write_text(
        '#!/bin/sh\nprintf "%s\\n" "$*" >> "$ARGLOG"\nexit 0\n', encoding="utf-8"
    )
    (bindir / "python").chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}",
        "ARGLOG": str(log),
        "CONDITIONS": "C3C",
        "ENV_NAME": "local_cpu",
        **env_extra,
    }
    env.pop("SEEDS", None)
    env.update(env_extra)
    proc = subprocess.run(
        ["bash", str(GRID)], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    return [line.split() for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_unset_seeds_emits_no_seeds_flag_at_all(tmp_path):
    """The advertised invocation: `ENV_NAME=... bash scripts/03_run_phase0_grid.sh`."""
    calls = _run_grid(tmp_path, {})
    run_condition = [c for c in calls if "run-condition" in c]
    assert run_condition, calls
    for call in run_condition:
        assert (
            "--seeds" not in call
        ), f"an unset SEEDS must produce NO flag, not an empty one: {call}"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_an_explicit_seed_count_is_still_forwarded(tmp_path):
    calls = _run_grid(tmp_path, {"SEEDS": "5"})
    run_condition = [c for c in calls if "run-condition" in c]
    assert run_condition, calls
    for call in run_condition:
        assert "--seeds" in call and call[call.index("--seeds") + 1] == "5"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_pilot_mode_runs_make_report_without_the_gate(tmp_path):
    """A truncated pilot is `reportable: false` by construction, so the gate necessarily
    blocks it. `PILOT=1` reports without a verdict instead of teaching the operator that
    a red gate is normal."""
    calls = _run_grid(tmp_path, {"PILOT": "1"})
    report = [c for c in calls if "make-report" in c]
    assert report and all("--no-gate" in c for c in report), calls


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_default_grid_gates(tmp_path):
    calls = _run_grid(tmp_path, {})
    report = [c for c in calls if "make-report" in c]
    assert report and not any("--no-gate" in c for c in report), calls


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_the_default_grid_runs_and_gates_the_per_item_scope(tmp_path):
    calls = _run_grid(tmp_path, {})
    (report,) = [c for c in calls if "make-report" in c]
    assert report[report.index("--scope") + 1] == "per_item"
    assert not any("store_scope" in " ".join(c) for c in calls if "run-condition" in c)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_scope_cumulative_both_runs_and_gates_phase_f2(tmp_path):
    """Setting only the first was the defect: `--set episode.store_scope=cumulative` ran
    F2 and then `make-report` re-printed the per-item verdict (ADR-0062)."""
    calls = _run_grid(tmp_path, {"SCOPE": "cumulative"})
    run_condition = [c for c in calls if "run-condition" in c]
    assert run_condition, calls
    for call in run_condition:
        assert "episode.store_scope=cumulative" in call, call
    (report,) = [c for c in calls if "make-report" in c]
    assert report[report.index("--scope") + 1] == "cumulative"


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_an_unknown_scope_stops_the_grid_before_it_spends_anything(tmp_path):
    """A typo must not silently run the primary experiment under a longitudinal label."""
    with pytest.raises(AssertionError):
        _run_grid(tmp_path, {"SCOPE": "per-item"})


# =====================================================================================
# 2. the rendered REPORT.md must describe the v5 experiment
# =====================================================================================


def _render(
    tmp_path: Path,
    monkeypatch,
    runs: list[dict],
    gate: bool = True,
    scope: str = mr.PRIMARY_STORE_SCOPE,
) -> str:
    """Render REPORT.md from `runs` into a scratch results dir and return its text.

    Every option is passed explicitly: called as a plain function rather than through
    Typer, an omitted argument is an `OptionInfo` object, not its default.
    """
    rd = tmp_path / "results"
    rd.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(mr, "collect_runs", lambda *a, **k: runs)
    monkeypatch.setattr(mr, "results_dir", lambda *a, **k: rd)
    monkeypatch.setattr(mr, "manifest_path", lambda *a, **k: rd / "manifest.jsonl")
    target = rd / ("REPORT.md" if scope == mr.PRIMARY_STORE_SCOPE else f"REPORT_{scope}.md")
    mr.make_report(out=target, figures=False, gate=gate, scope=scope)
    return target.read_text(encoding="utf-8")


def test_the_report_cites_the_v5_preregistration(tmp_path, monkeypatch):
    body = _render(tmp_path, monkeypatch, _grid())
    assert "docs/00e_preregistration_v5.md" in body
    assert (
        "docs/00c_preregistration_v3.md" not in body
    ), "the criteria paragraph cited v3 while applying v5's"


def test_the_report_names_c3c_minus_c3s_as_the_primary_pair(tmp_path, monkeypatch):
    body = _render(tmp_path, monkeypatch, _grid())
    assert "`C3C - C3S`" in body
    assert "PRIMARY pair `C3C - C3D`" not in body, (
        "C3C - C3D varies A's information, the presence of any context and the prompt "
        "format at once; it has not been the primary pair since ADR-0048"
    )


def test_the_report_shows_the_content_specific_joint_metric(tmp_path, monkeypatch):
    """THE gap: the primary joint quantity was computed, written to gate_verdict.json,
    and never rendered in any table."""
    body = _render(tmp_path, monkeypatch, _grid())
    assert "content_specific_joint_recovery" in body


def test_the_report_labels_the_two_joint_metrics_with_their_v5_roles(tmp_path, monkeypatch):
    body = _render(tmp_path, monkeypatch, _grid())
    table = mr.joint_table(mr.evaluate_gates(_grid()))
    assert "content_specific_joint_recovery" in table and "PRIMARY" in table
    assert "joint_only_recovery" in table and "secondary" in table
    assert "joint_only_recovery" in body and "secondary" in body


def test_the_report_does_not_call_laundering_rate_the_headline(tmp_path, monkeypatch):
    """`laundering_rate` is conditional on recovery — an arm that recovers four items and
    launders all four reports 1.0. Fine diagnostic, terrible headline."""
    body = _render(tmp_path, monkeypatch, _grid())
    assert "`laundering_rate` is a diagnostic and NOT the headline" in body
    assert "the headline — laundering rate" not in body.lower()


def test_the_headline_is_stated_to_use_the_content_specific_set(tmp_path, monkeypatch):
    body = _render(tmp_path, monkeypatch, _grid())
    assert "certified_joint_leak_rate" in body
    assert "CONTENT-SPECIFIC set" in body


def test_the_verdict_paragraph_names_the_v5_estimands(tmp_path, monkeypatch):
    body = _render(tmp_path, monkeypatch, _grid())
    verdict = body.split("### Verdict", 1)[1]
    assert "`C3C - C3S` and `content_specific_joint_recovery`" in verdict
    assert "docs/00e_preregistration_v5.md" in verdict
    assert "docs/00d_preregistration_v4.md" not in verdict.split("v4 estimand")[0]


def test_the_figures_rank_the_headline_above_the_diagnostic(tmp_path, monkeypatch):
    """Figure 2 used to be captioned as the headline while plotting `laundering_rate`."""
    pytest.importorskip("matplotlib")
    rd = tmp_path / "results"
    rd.mkdir(parents=True, exist_ok=True)
    runs = _grid()
    figs = [p.name for p in mr._figures(runs, rd, mr.evaluate_gates(runs))]
    assert "fig2_certified_content_specific_headline.png" in figs
    assert "fig3_laundering_rate_diagnostic.png" in figs
    assert not any("fig2_laundering" in f for f in figs)


# =====================================================================================
# 3. Day-1 provenance must be present, not merely un-denied
# =====================================================================================

_PINNED = mr.pinned_ou_source_sha()


def test_the_pinned_submodule_sha_is_readable_from_the_index():
    """Read from `git ls-tree`, not the working tree: comparing the checkout against
    itself would make a moved submodule agree with itself. CI checks out without
    submodules, so this must not depend on the submodule being present."""
    assert _PINNED and len(_PINNED) == 40, _PINNED


def test_a_fully_provenanced_parity_report_passes():
    assert mr.report_is_exact_parity(_repro("full", FULL, FULL_REV), _PINNED)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("ou_source_sha", None, "no ou_source_sha"),
        ("ou_source_sha", "0" * 40, "but this repo pins"),
        ("tokenizer", {}, "chat-template hash"),
        ("transformers_version", None, "no transformers_version"),
        ("ou_runtime_mode", None, "no ou_runtime_mode"),
        ("git_dirty", None, "not false"),
        ("git_dirty", True, "not false"),
    ],
)
def test_missing_provenance_is_not_clean_provenance(field, value, expected):
    """The reports in `results/` were produced at commit 1ea12bf and carry NONE of these
    fields. The old gate rejected only `git_dirty is True`, so all of them cleared it —
    while run_repro.py's own comments record that the commit those reports name did not
    contain all the code that ran."""
    r = _repro("full", FULL, FULL_REV, **{field: value})
    gaps = mr.parity_provenance_gaps(r, _PINNED)
    assert any(expected in g for g in gaps), gaps
    assert not mr.report_is_exact_parity(r, _PINNED)


def test_a_pre_adr0054_report_fails_every_provenance_check():
    """A verbatim 2026-08-07 report: passing, at parity, and unverifiable."""
    legacy = {
        "phase": "phase0_days1-2_repro",
        "target": "full",
        "checkpoint": FULL,
        "revision": FULL_REV,
        "passed": True,
        "batch_size": 32,
        "seed": 0,
        "published_parity": True,
    }
    gaps = mr.parity_provenance_gaps(legacy, _PINNED)
    assert len(gaps) >= 5, gaps
    assert not mr.report_is_exact_parity(legacy, _PINNED)


def test_an_unverifiable_day1_report_blocks_the_whole_grid():
    runs = [
        *[r for r in _grid() if r.get("phase") == "phase0_days3-5"],
        _repro("full", FULL, FULL_REV, ou_source_sha=None),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV),
        _measure(AGENT_B, AGENT_B_REV),
    ]
    verdict = mr.evaluate_gates(runs)
    assert not verdict["experiment_valid"]
    assert any("never at EXACT published parity" in b for b in verdict["blockers"]), verdict[
        "blockers"
    ]


def test_a_characterized_target_is_excused_its_parity_miss_but_not_its_provenance(
    tmp_path, monkeypatch
):
    """`released_artifact` says the published-row MISS on npo_forget10 is the recorded
    finding. It never said an unidentified evaluator on a dirty tree was acceptable."""
    monkeypatch.setattr(
        mr,
        "load_study_mode",
        lambda *a, **k: {
            "mode": "released_artifact",
            "evaluation_stack_targets": ["full"],
            "characterized_targets": ["npo_forget10"],
        },
    )
    runs = [
        *[r for r in _grid() if r.get("phase") == "phase0_days3-5"],
        _repro("full", FULL, FULL_REV),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV, passed=False, git_dirty=True),
        _measure(AGENT_B, AGENT_B_REV),
    ]
    verdict = mr.evaluate_gates(runs)
    assert any("CHARACTERIZED" in b and "provenance" in b for b in verdict["blockers"]), verdict[
        "blockers"
    ]


def test_a_characterized_target_with_a_recorded_miss_and_good_provenance_passes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        mr,
        "load_study_mode",
        lambda *a, **k: {
            "mode": "released_artifact",
            "evaluation_stack_targets": ["full"],
            "characterized_targets": ["npo_forget10"],
        },
    )
    runs = [
        *[r for r in _grid() if r.get("phase") == "phase0_days3-5"],
        _repro("full", FULL, FULL_REV),
        _repro("npo_forget10", AGENT_A, AGENT_A_REV, passed=False),
        _measure(AGENT_B, AGENT_B_REV),
    ]
    verdict = mr.evaluate_gates(runs)
    assert not any("CHARACTERIZED" in b for b in verdict["blockers"]), verdict["blockers"]


def test_newest_eligible_characterization_replaces_a_stale_npo_report(monkeypatch):
    monkeypatch.setattr(
        mr,
        "load_study_mode",
        lambda *a, **k: {
            "mode": "released_artifact",
            "evaluation_stack_targets": ["full"],
            "characterized_targets": ["npo_forget10"],
        },
    )
    stale = _repro("npo_forget10", AGENT_A, AGENT_A_REV, passed=False, git_dirty=None)
    stale.update({"run_id": "20260807T000000Z-stale", "git_sha": "1ea12bf"})
    current = _repro("npo_forget10", AGENT_A, AGENT_A_REV, passed=False)
    current.update({"run_id": "20260808T000000Z-current"})
    runs = [
        *[r for r in _grid() if r.get("phase") == "phase0_days3-5"],
        _repro("full", FULL, FULL_REV),
        stale,
        current,
        _measure(AGENT_B, AGENT_B_REV),
    ]
    verdict = mr.evaluate_gates(runs)
    assert not any("characterises it" in b and "1ea12bf" in b for b in verdict["blockers"])


def test_the_day1_prerequisites_still_pass_end_to_end():
    """Guard against a gate so strict nothing can ever clear it."""
    runs = [*[r for r in _grid() if r.get("phase") == "phase0_days3-5"], *_day1()]
    verdict = mr.evaluate_gates(runs)
    assert verdict["experiment_valid"], verdict["blockers"]


def test_gate_verdict_json_is_written_beside_the_report(tmp_path, monkeypatch):
    _render(tmp_path, monkeypatch, _grid())
    written = json.loads((tmp_path / "results" / "gate_verdict.json").read_text(encoding="utf-8"))
    assert written["content_specific_joint_recovery"]["available"] is True
    assert written["joint_only_recovery"]["available"] is True


def test_an_incomplete_grid_still_renders_without_a_verdict(tmp_path, monkeypatch):
    body = _render(tmp_path, monkeypatch, [_report("C3C", recall=0.6)], gate=False)
    assert "Pre-registered gate" not in body
