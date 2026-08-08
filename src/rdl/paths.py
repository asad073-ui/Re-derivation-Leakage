"""Filesystem layout and run identity.

Results are append-only and content-addressed:

    results/<run_id>/   where run_id = <UTC timestamp>-<git sha>-<config hash>

Nothing overwrites. `results/manifest.jsonl` is committed; large artifacts are not.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path

__all__ = [
    "append_manifest",
    "configs_dir",
    "docs_dir",
    "git_diff_sha256",
    "git_dirty",
    "git_sha",
    "make_run_id",
    "manifest_path",
    "open_unlearning_dir",
    "repo_root",
    "results_dir",
    "run_dir",
    "third_party_dir",
]

_MARKERS = ("pyproject.toml", ".git")


def repo_root(start: Path | None = None) -> Path:
    """Walk up from `start` until a directory containing a repo marker is found.

    Falls back to the package's grandparent (``src/rdl`` -> repo) so that an installed
    copy still resolves to something sane rather than raising.
    """
    here = (start or Path(__file__).resolve()).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and any((candidate / m).exists() for m in _MARKERS):
            return candidate
    return Path(__file__).resolve().parents[2]


def results_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "results"


def manifest_path(root: Path | None = None) -> Path:
    return results_dir(root) / "manifest.jsonl"


def third_party_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "third_party"


def open_unlearning_dir(root: Path | None = None) -> Path:
    return third_party_dir(root) / "open-unlearning"


def configs_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "configs"


def docs_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "docs"


def git_sha(root: Path | None = None, short: bool = True) -> str:
    """Short git SHA of `root`, or ``"nogit"`` when unavailable.

    Never raises: a missing git, a missing .git dir, and a detached/empty repo all
    degrade to ``"nogit"`` so that a run can still produce a usable run_id.
    """
    root = root or repo_root()
    args = ["git", "-C", str(root), "rev-parse"]
    if short:
        args.append("--short")
    args.append("HEAD")
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else "nogit"


# The SOURCE of the experiment, which is what "dirty" is a claim about. `results/` is
# this program's own OUTPUT: every run appends to the tracked manifest and writes report
# files beside it, so a session that produced one result was, by its own definition,
# dirty for the next one. See ADR-0059.
#
# Applied as a git pathspec to both the status and the diff so the two agree; a `dirty`
# flag computed over one file set and a `git_diff_sha256` computed over another describe
# two different working trees.
_SOURCE_PATHSPEC = (".", ":(exclude)results/**")

# Untracked content INSIDE the submodule is not superproject dirt. open-unlearning writes
# its evaluation outputs into its own tree (`saves/`, `data/`), and the superproject's
# .gitignore does not reach inside a submodule — so without this, Day 1 would report a
# dirty tree the moment the evaluator it is running produced a file. A submodule at the
# wrong COMMIT, or with modified tracked files, still counts: that is a code difference,
# and `ou_source_sha` gates the commit separately.
_SUBMODULE_FLAG = "--ignore-submodules=untracked"


def git_dirty(root: Path | None = None) -> bool | None:
    """True when the SOURCE tree differs from HEAD. ``None`` when git cannot answer.

    A report that records `git_sha: X` while the code that ran is not what X contains
    is not reproducible, and nothing downstream can tell. That happened: the Day-1 GPU
    runs recorded `1ea12bf` and executed a compatibility shim that `1ea12bf` does not
    contain, so a reviewer checking out that commit cannot run the recorded command.

    **`results/` is excluded, and untracked source files now count** (ADR-0059). The
    previous predicate was `status --porcelain --untracked-files=no` over the whole tree,
    which had the two failure modes exactly backwards:

    * Every run appends to the TRACKED `results/manifest.jsonl`. So the first
      `run-repro` of a session left the tree dirty and the second refused to start —
      deterministically, on a rented GPU, after the first result was already paid for.
      Neither escape worked: `--allow-dirty` reports are disqualified by design, and
      committing between conditions gives the arms different git SHAs, which the
      fingerprint check then blocks.
    * `--untracked-files=no` meant a source or config file that existed on the box and
      in no commit was invisible. That is the same class of failure as the `.gitignore`
      rule that once swallowed `configs/env/`: the run works locally and cannot be
      reproduced from the SHA it records.
    """
    root = root or repo_root()
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
                _SUBMODULE_FLAG,
                "--",
                *_SOURCE_PATHSPEC,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def git_diff_sha256(root: Path | None = None) -> str | None:
    """Hash of the tracked SOURCE diff against HEAD, or ``None`` when clean/unavailable.

    Recorded on deliberately-dirty diagnostic runs so two such runs can at least be
    told apart, and so a later reviewer can see that *something* uncommitted was in
    play even though the diff itself is not in the repository.

    Same pathspec as `git_dirty`: a hash that moved because the manifest grew would make
    two runs of identical code look like two different experiments.
    """
    root = root or repo_root()
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "diff",
                "HEAD",
                _SUBMODULE_FLAG,
                "--",
                *_SOURCE_PATHSPEC,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    body = out.stdout
    if not body.strip():
        return None
    import hashlib

    return hashlib.sha256(body.encode("utf-8", "replace")).hexdigest()


def _utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_id(config_hash: str, root: Path | None = None) -> str:
    """``<UTC timestamp>-<git sha>-<config hash prefix>``.

    The config hash prefix is 12 hex chars: enough to be collision-free across a
    project of this size while keeping directory names readable.
    """
    return f"{_utc_stamp()}-{git_sha(root)}-{config_hash[:12]}"


def run_dir(run_id: str, root: Path | None = None, create: bool = True) -> Path:
    """Directory for a single run. Refuses to reuse an existing one — append-only."""
    d = results_dir(root) / run_id
    if create:
        if d.exists():
            raise FileExistsError(
                f"results dir already exists: {d}. Runs are append-only; "
                "nothing overwrites. Change the config or wait one second."
            )
        d.mkdir(parents=True)
    return d


def append_manifest(record: dict, root: Path | None = None) -> Path:
    """Append one line to the committed manifest. Creates the file if absent."""
    path = manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, separators=(",", ":"))
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    return path


def hf_home() -> Path | None:
    """HF cache location if the environment sets one. Informational only."""
    v = os.environ.get("HF_HOME")
    return Path(v) if v else None
