"""`git_dirty` is a claim about the SOURCE, and the reports must reach GitHub.

Two operational defects, both of which would have surfaced only on a rented GPU with the
meter running:

  * `git_dirty()` ran `status --porcelain --untracked-files=no` over the whole tree.
    Every run appends to the TRACKED `results/manifest.jsonl`, so the first `run-repro`
    of a session left the tree dirty and the second refused to start. Deterministic,
    after the first result was already paid for, with no legitimate escape:
    `--allow-dirty` reports are disqualified by design and committing between runs gives
    the arms different git SHAs.
  * `.gitignore` un-ignored only the manifest, and `99_sync_results.sh` stages with
    `git add -A results/`, which does not add ignored files. The reports never reached
    the index — and the next runbook step is "destroy the instance".

Each test builds a throwaway git repo so nothing depends on the state of this one.
See ADR-0059 and ADR-0060.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from rdl.paths import git_diff_sha256, git_dirty

REPO = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, f"git {args}: {out.stderr}"
    return out.stdout


@pytest.fixture
def sandbox(tmp_path: Path) -> Path:
    """A miniature of this repo: a source tree, a results/ dir, and the real ignore rules.

    The `.gitignore` is COPIED from the repository rather than retyped, so a rule that
    stops un-ignoring the reports fails these tests rather than passing a paraphrase.
    """
    repo = tmp_path / "repo"
    (repo / "src" / "rdl").mkdir(parents=True)
    (repo / "configs" / "env").mkdir(parents=True)
    (repo / "results").mkdir()
    (repo / "src" / "rdl" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "configs" / "env" / "local.yaml").write_text("name: local\n", encoding="utf-8")
    (repo / "results" / ".gitkeep").write_text("", encoding="utf-8")
    (repo / "results" / "manifest.jsonl").write_text("", encoding="utf-8")
    shutil.copy(REPO / ".gitignore", repo / ".gitignore")

    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    assert git_dirty(repo) is False
    return repo


def _append_manifest(repo: Path, line: str) -> None:
    with (repo / "results" / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def _write_report(repo: Path, run_id: str, name: str = "repro_report.json") -> Path:
    d = repo / "results" / run_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text('{"phase": "test"}', encoding="utf-8")
    return p


# =====================================================================================
# ADR-0059 — results/ is this program's OUTPUT, not its source
# =====================================================================================


def test_appending_to_the_tracked_manifest_does_not_make_the_tree_dirty(sandbox):
    """THE blocker. The manifest is tracked and every run appends to it."""
    _append_manifest(sandbox, '{"run_id": "a"}')
    assert git_dirty(sandbox) is False


def test_writing_a_generated_report_does_not_make_the_tree_dirty(sandbox):
    _write_report(sandbox, "20260808T000000Z-abc-1")
    assert git_dirty(sandbox) is False


def test_a_modified_tracked_source_file_is_still_dirty(sandbox):
    (sandbox / "src" / "rdl" / "mod.py").write_text("x = 2\n", encoding="utf-8")
    assert git_dirty(sandbox) is True


def test_an_untracked_source_file_is_now_dirty(sandbox):
    """`--untracked-files=no` made a file that exists on the box and in no commit
    invisible — the same class of failure as the .gitignore rule that once swallowed
    configs/env/."""
    (sandbox / "src" / "rdl" / "scratch.py").write_text("y = 1\n", encoding="utf-8")
    assert git_dirty(sandbox) is True


def test_an_untracked_config_file_is_dirty(sandbox):
    (sandbox / "configs" / "env" / "new_box.yaml").write_text("name: new\n", encoding="utf-8")
    assert git_dirty(sandbox) is True


def test_the_diff_hash_ignores_results_too(sandbox):
    """A hash that moved because the manifest grew would make two runs of identical code
    look like two different experiments."""
    _append_manifest(sandbox, '{"run_id": "a"}')
    _write_report(sandbox, "20260808T000000Z-abc-1")
    assert git_diff_sha256(sandbox) is None

    (sandbox / "src" / "rdl" / "mod.py").write_text("x = 2\n", encoding="utf-8")
    assert git_diff_sha256(sandbox) is not None


def test_two_sequential_repro_runs_stay_clean_at_one_commit(sandbox):
    """The exact failure sequence: run 1 succeeds, appends to the manifest, and run 2
    refuses to start. Simulated end to end at ONE git SHA, which is also what ADR-0061's
    same-commit requirement needs to be satisfiable."""
    sha = _git(sandbox, "rev-parse", "--short", "HEAD").strip()
    seen = []
    for i, target in enumerate(("full", "npo_forget10")):
        # What `run-repro` checks before it will start.
        assert git_dirty(sandbox) is False, f"run {i} ({target}) would have REFUSED TO RUN"
        _write_report(sandbox, f"20260808T00000{i}Z-{sha}-{target}")
        _append_manifest(sandbox, f'{{"target": "{target}", "git_sha": "{sha}"}}')
        seen.append((target, git_dirty(sandbox), sha))

    assert [s[1] for s in seen] == [False, False]
    assert len({s[2] for s in seen}) == 1, "both runs must record the same commit"


def test_three_sequential_condition_runs_stay_clean_at_one_commit(sandbox):
    """The same for the grid: C0 then C1 then C1W, no commit in between."""
    sha = _git(sandbox, "rev-parse", "--short", "HEAD").strip()
    for i, cond in enumerate(("C0", "C1", "C1W")):
        assert git_dirty(sandbox) is False, f"{cond} would have REFUSED TO RUN"
        _write_report(sandbox, f"20260808T00000{i}Z-{sha}-{cond}", "condition_report.json")
        _write_report(sandbox, f"20260808T00000{i}Z-{sha}-{cond}", "handoff_evidence.json")
        _append_manifest(sandbox, f'{{"condition": "{cond}", "git_sha": "{sha}"}}')
    assert git_dirty(sandbox) is False


def test_a_source_change_mid_session_still_stops_the_next_run(sandbox):
    """The fix must not weaken the thing the check is FOR."""
    _write_report(sandbox, "20260808T000000Z-abc-1")
    assert git_dirty(sandbox) is False
    (sandbox / "src" / "rdl" / "mod.py").write_text("x = 99\n", encoding="utf-8")
    assert git_dirty(sandbox) is True


# =====================================================================================
# ADR-0060 — the reports have to be able to leave the instance
# =====================================================================================

TRACKABLE = (
    "results/REPORT.md",
    "results/gate_verdict.json",
    "results/fig1_sysrecall_store.png",
    "results/fig2_certified_content_specific_headline.png",
    "results/REPORT_cumulative.md",
    "results/gate_verdict_cumulative.json",
    "results/20260808T000000Z-abc-1/condition_report.json",
    "results/20260808T000000Z-abc-1/repro_report.json",
    "results/20260808T000000Z-abc-1/measure_report.json",
    "results/20260808T000000Z-abc-1/SUMMARY.json",
    "results/20260808T000000Z-abc-1/handoff_evidence.json",
)

STILL_IGNORED = (
    "results/20260808T000000Z-abc-1/transcripts_treatment_seed0.jsonl",
    "results/20260808T000000Z-abc-1/transcripts_retain_seed0.jsonl",
    "results/20260808T000000Z-abc-1/episodes.jsonl",
    "results/20260808T000000Z-abc-1/model.safetensors",
    "results/20260808T000000Z-abc-1/checkpoint.bin",
)


def _ignored(repo: Path, rel: str) -> bool:
    out = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "-q", rel],
        capture_output=True,
        text=True,
        timeout=60,
    )
    return out.returncode == 0


@pytest.mark.parametrize("rel", TRACKABLE)
def test_report_artifacts_are_not_ignored(sandbox, rel):
    """`git add -A` does not add ignored files, so an ignored report is a report that
    stays on the instance until the instance is destroyed."""
    (sandbox / rel).parent.mkdir(parents=True, exist_ok=True)
    (sandbox / rel).write_text("{}", encoding="utf-8")
    assert not _ignored(sandbox, rel), f"{rel} is ignored and would never reach GitHub"


@pytest.mark.parametrize("rel", STILL_IGNORED)
def test_bulk_and_weights_stay_ignored(sandbox, rel):
    (sandbox / rel).parent.mkdir(parents=True, exist_ok=True)
    (sandbox / rel).write_text("{}", encoding="utf-8")
    assert _ignored(sandbox, rel), f"{rel} must NOT be committed"


def test_git_add_dash_a_actually_stages_the_reports(sandbox):
    """The end-to-end property `99_sync_results.sh` depends on, checked the way the
    script checks it."""
    for rel in TRACKABLE:
        (sandbox / rel).parent.mkdir(parents=True, exist_ok=True)
        (sandbox / rel).write_text("{}", encoding="utf-8")
    for rel in STILL_IGNORED:
        (sandbox / rel).parent.mkdir(parents=True, exist_ok=True)
        (sandbox / rel).write_text("{}", encoding="utf-8")

    _git(sandbox, "add", "-A", "results/")
    staged = set(_git(sandbox, "diff", "--cached", "--name-only").split())

    missing = [r for r in TRACKABLE if r not in staged]
    assert not missing, f"these reports were silently omitted from the push: {missing}"
    leaked = [r for r in STILL_IGNORED if r in staged]
    assert not leaked, f"bulk artifacts were staged: {leaked}"
