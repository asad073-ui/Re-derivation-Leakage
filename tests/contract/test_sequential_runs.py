"""Two runs in a row, through the real CLI, on a real git repo. The GPU-session shape.

This is the test that would have cost a rented RTX 3090. `run-condition` and `run-repro`
both refuse to start on a dirty tree, and every finished run appends to the TRACKED
`results/manifest.jsonl` — so run 1 succeeded, run 2 was rejected, and the failure was
deterministic and only reachable once a session had actually produced a result:

    --- A1. SANITY: the `full` target model ---
    ...
    wrote results/<full-run>

    --- A2. THE GATE: NPO forget10 ---
    REFUSING TO RUN: the working tree has uncommitted tracked changes

Neither escape was real. `--allow-dirty` marks the report `reportable: false`, and
committing between conditions gives the arms different git SHAs — which `make-report` now
blocks outright (ADR-0061), so the two "workarounds" were mutually exclusive with the
requirement.

The repository under test is a throwaway one built in `tmp_path`, driven through
`rdl.cli.run_condition.run_condition` with stub models, so this exercises the real
refusal, the real predicate and the real report writer without a GPU or a network.
See ADR-0059.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from rdl.cli import run_condition as rc
from rdl.paths import git_dirty

REPO = Path(__file__).resolve().parents[2]
CONDITIONS = REPO / "configs" / "conditions"
FIXTURE = REPO / "tests" / "fixtures" / "tofu_forget10_sample.json"


def _cuda_present() -> bool:
    """True when a CUDA device is visible to torch.

    These cases pin ``environment="local_cpu"``, and the runner deliberately refuses that
    profile on a box with a CUDA device — picking a CPU profile on a rented GPU is a
    mistake worth failing on. That refusal is correct behaviour, so on a GPU box these
    cases are inapplicable rather than broken, and they skip instead of failing.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - a broken driver is not this suite's problem
        return False


pytestmark = [
    pytest.mark.skipif(shutil.which("git") is None, reason="needs git"),
    pytest.mark.skipif(_cuda_present(), reason="pins local_cpu; refused on a CUDA box"),
]


def _git(repo: Path, *args: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, f"git {args}: {out.stderr}"
    return out.stdout


@pytest.fixture
def session(tmp_path: Path, monkeypatch):
    """A clean checkout with a `results/` dir, wired into the CLI's path helpers.

    The CLI is pointed at this repo for BOTH the dirty predicate and the output
    location, which is the coupling the bug lived in: the thing it writes was inside the
    thing it checks.
    """
    repo = tmp_path / "box"
    (repo / "src").mkdir(parents=True)
    (repo / "results").mkdir()
    (repo / "src" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    (repo / "results" / "manifest.jsonl").write_text("", encoding="utf-8")
    shutil.copy(REPO / ".gitignore", repo / ".gitignore")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "session commit")

    sha = _git(repo, "rev-parse", "--short", "HEAD").strip()
    monkeypatch.setattr(rc, "git_dirty", lambda: git_dirty(repo))
    monkeypatch.setattr(rc, "git_sha", lambda: sha)
    monkeypatch.setattr(rc, "run_dir", lambda run_id: _make_run_dir(repo, run_id))
    monkeypatch.setattr(rc, "append_manifest", lambda record: _append_manifest(repo, record))
    return repo, sha


def _make_run_dir(repo: Path, run_id: str) -> Path:
    d = repo / "results" / run_id
    d.mkdir(parents=True, exist_ok=False)
    return d


def _append_manifest(repo: Path, record: dict) -> Path:
    path = repo / "results" / "manifest.jsonl"
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return path


# Only the models a condition actually declares may be overridden: adding a key a
# condition does not use produces an incomplete model entry and fails validation.
_MODELS: dict[str, tuple[str, ...]] = {
    "C1W": ("tofu_llama32_1b_npo_forget10",),
    "B1W": ("tofu_llama32_1b_npo_forget10_indep",),
    "C3S": ("tofu_llama32_1b_npo_forget10", "tofu_llama32_1b_npo_forget10_indep"),
    "C3C": ("tofu_llama32_1b_npo_forget10", "tofu_llama32_1b_npo_forget10_indep"),
}


def _run(condition: str) -> None:
    override: list[str] = []
    for model in _MODELS[condition]:
        override += [f"models.{model}.kind=stub", f"models.{model}.repo_id=null"]
    rc.run_condition(
        condition=CONDITIONS / f"{condition}.yaml",
        seeds=1,
        environment="local_cpu",
        override=override,
        fixture=FIXTURE,
        allow_fixture=True,
        limit=None,
        controls=False,
        token=None,
        allow_dirty=False,
        dry_run=False,
    )


def _reports(repo: Path) -> list[dict]:
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted((repo / "results").glob("*/condition_report.json"))
    ]


def test_three_conditions_run_back_to_back_at_one_clean_commit(session):
    """THE test the audit asked for, one condition longer than the minimum.

    No `--allow-dirty`, no commit in between. Each run appends to the tracked manifest and
    writes its report inside `results/`, and the next run must still see a clean tree.
    """
    repo, sha = session
    assert git_dirty(repo) is False

    for condition in ("C1W", "C3S", "C3C"):
        # Exactly what `run-condition` evaluates before it will start. Under the old
        # predicate this was True from the second iteration onwards.
        assert git_dirty(repo) is False, f"{condition} would have REFUSED TO RUN"
        _run(condition)

    reports = _reports(repo)
    assert len(reports) == 3, [r.get("condition") for r in reports]
    assert {r["condition"] for r in reports} == {"C1W", "C3S", "C3C"}

    # Both halves of the requirement: clean, and clean AT THE SAME COMMIT (ADR-0061).
    assert [r["git_dirty"] for r in reports] == [False, False, False]
    assert {r["git_sha"] for r in reports} == {sha}
    assert all(r["git_diff_sha256"] is None for r in reports)

    # The manifest grew, which is precisely what used to poison the next run.
    lines = (repo / "results" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert git_dirty(repo) is False


def test_the_run_still_refuses_when_the_SOURCE_changes_mid_session(session):
    """The guard must survive the fix: editing code between two arms is exactly the state
    `git_dirty` exists to catch, and it is now the only state that stops a run."""
    import typer

    repo, _ = session
    _run("C1W")
    (repo / "src" / "mod.py").write_text("x = 2\n", encoding="utf-8")
    assert git_dirty(repo) is True
    with pytest.raises(typer.Exit):
        _run("C3S")


def test_the_handoff_arms_leave_a_committable_evidence_file(session):
    """The raw C3S/C3C evidence has to survive the instance being destroyed (ADR-0060):
    the transcripts are ignored by design, so this is what a reviewer gets."""
    repo, _ = session
    _run("C3S")
    (evidence_path,) = list((repo / "results").glob("*/handoff_evidence.json"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert evidence["condition"] == "C3S"
    assert evidence["handoff_source"] == "deranged"
    assert evidence["n_records"] == len(evidence["records"]) > 0
    for rec in evidence["records"]:
        assert rec["shuffled"] is True
        # The control's defining property, checkable from the committed file alone.
        assert rec["source_item_id"] != rec["item_id"]
        # A control that hands over nothing is a bare-question arm wearing a wrapper's
        # name — the check the runbook asks an operator to make by eye.
        assert rec["handoff_text"]

    # And it is not ignored, unlike the transcripts beside it.
    rel = evidence_path.relative_to(repo).as_posix()
    assert (
        subprocess.run(
            ["git", "-C", str(repo), "check-ignore", "-q", rel], capture_output=True
        ).returncode
        != 0
    ), f"{rel} would never reach GitHub"


def test_c3c_evidence_shows_the_handoff_is_agent_as_own_answer(session):
    """The treatment's mirror-image property: same item, and the two hashes AGREE."""
    repo, _ = session
    _run("C3C")
    (evidence_path,) = list((repo / "results").glob("*/handoff_evidence.json"))
    records = json.loads(evidence_path.read_text(encoding="utf-8"))["records"]
    assert records
    for rec in records:
        assert rec["shuffled"] is False
        assert rec["source_item_id"] is None
        assert rec["handoff_text_sha256"] == rec["agent_a_answer_sha256"]
