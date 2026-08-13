"""Immutable evidence shards.

The historical runner flushed and fsynced after every row, which is safe and slow. Here
a shard accumulates ``shard_size`` trajectories, is written to a temporary file, hashed,
atomically renamed, and its hash appended to the manifest. A killed process leaves a
partial temporary file, which is discarded on resume — never a torn committed shard.

Resume is by shard, and the resume key is ``(item_id, sample_id, arm, challenge)``. A
trajectory already present in a committed shard is not re-run, and a duplicate key is a
hard error rather than a silent overwrite.

**The ledger** (``SHARDS.json``, GU-0024). Shard hashes used to reach ``RUN_MANIFEST.json``
only when a run *finished*, so the shards of an interrupted run were covered by nothing.
Resume read them back, trusted them, and skipped their trajectory keys — a shard edited
between the kill and the resume was silently adopted as this run's evidence, and the
final manifest then hashed the edited bytes and declared them verified.

So every commit is now recorded in a per-directory ledger, written **before** the rename:

    1. write ``part-N.jsonl.tmp``, fsync
    2. append ``{name, sha256, n_rows}`` to ``SHARDS.json``, atomically replace it
    3. ``os.replace`` the temp file into place

Ledger-first is what makes the crash windows unambiguous. Killed before (2): no entry, no
shard, the temp file is discarded. Killed between (2) and (3): the ledger's LAST entry
names a shard that does not exist, which can only mean the rename never happened, so that
entry is pruned on the next open. Killed after (3): consistent. Any other divergence — a
hash that does not match, a shard on disk the ledger never declared, a declared shard
missing from the middle — is refused, because each of those means the committed evidence
changed after it was committed.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "LEDGER_NAME",
    "EvidenceTampered",
    "ShardWriter",
    "TrajectoryKey",
    "completed_keys",
    "read_nli_cache",
    "read_shards",
    "scorer_label_fn",
    "verify_ledger",
    "verify_shards",
]

TrajectoryKey = tuple[str, int, str, str]

LEDGER_NAME = "SHARDS.json"
LEDGER_SCHEMA = "graph-shard-ledger-v1"


class EvidenceTampered(RuntimeError):
    """Committed shards on disk do not match the hashes recorded when they were written."""


def _key(row: dict) -> TrajectoryKey:
    return (
        str(row["item_id"]),
        int(row["sample_id"]),
        str(row["arm"]),
        str(row.get("challenge", "natural")),
    )


def atomic_json(path: Path, payload: dict) -> None:
    """Replace a manifest atomically, in STRICT JSON.

    ``allow_nan=False`` is the point (GU-0029). Python's default emits bare ``NaN``,
    ``Infinity`` and ``-Infinity``, which RFC 8259 does not permit; ``jq``, Go, Rust and
    every browser reject them. Three committed graph reports carried literal ``NaN``
    where a relative reduction was undefined, so the runbook's own
    ``python -m json.tool`` check would have failed on the evidence it was validating.

    A non-finite value now raises here rather than being written. That is deliberate: an
    undefined quantity has to be given a representation by the code that knows what it
    means — ``None`` for "undefined", never a float that no reader can parse.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        body = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    except ValueError as exc:
        raise ValueError(
            f"refusing to write {path}: the payload holds a non-finite float ({exc}). "
            "NaN and Infinity are not JSON. Represent an undefined quantity as null — "
            "see rdl.eval.graph_statistics.relative_reduction."
        ) from exc
    with temp.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(body)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp, path)


@dataclass(frozen=True)
class ShardRecord:
    name: str
    sha256: str
    n_rows: int

    def to_dict(self) -> dict:
        return {"name": self.name, "sha256": self.sha256, "n_rows": self.n_rows}


class ShardWriter:
    """Buffers rows and commits fixed-size, content-addressed shards."""

    def __init__(
        self,
        directory: Path,
        *,
        prefix: str = "part",
        shard_size: int = 64,
        verify_on_open: bool = True,
    ) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = shard_size
        # Off only for the legacy-adoption path, where there is no ledger to check
        # against and refusing would strand a run written by an older version.
        self.verify_on_open = verify_on_open
        self._buffer: list[dict] = []
        self.shards: list[ShardRecord] = []
        self._seen: set[TrajectoryKey] = set()
        self.ledger_status: str = "absent"
        self._next_index = self._scan_existing()

    # -------------------------------------------------------------------- ledger --

    @property
    def ledger_path(self) -> Path:
        return self.directory / LEDGER_NAME

    def _read_ledger(self) -> list[ShardRecord] | None:
        path = self.ledger_path
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            entries = payload["shards"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise EvidenceTampered(f"{path}: shard ledger is unreadable: {exc}") from exc
        return [
            ShardRecord(name=str(e["name"]), sha256=str(e["sha256"]), n_rows=int(e["n_rows"]))
            for e in entries
        ]

    def _write_ledger(self, shards: Sequence[ShardRecord]) -> None:
        atomic_json(
            self.ledger_path,
            {
                "schema": LEDGER_SCHEMA,
                "prefix": self.prefix,
                "n_shards": len(shards),
                "n_rows": sum(s.n_rows for s in shards),
                "shards": [s.to_dict() for s in shards],
                "note": (
                    "written BEFORE each shard's atomic rename, so a resume can tell an "
                    "interrupted commit from evidence edited after it was committed"
                ),
            },
        )

    def _scan_existing(self) -> int:
        # A temporary file is an interrupted commit and carries no committed rows.
        for temp in self.directory.glob(f"{self.prefix}-*.jsonl.tmp"):
            temp.unlink()

        on_disk = sorted(self.directory.glob(f"{self.prefix}-*.jsonl"))
        declared = self._read_ledger()
        if declared is None:
            # No ledger: either a fresh directory, or one written before GU-0024. Adopt
            # what is there and start a ledger from it, saying so rather than implying
            # the existing bytes were ever verified.
            self.ledger_status = "adopted_unverified" if on_disk else "new"
        else:
            declared = self._reconcile(declared, {p.name for p in on_disk})
            self.ledger_status = "verified"

        index = 0
        for path in on_disk:
            rows = _read_jsonl(path)
            self.shards.append(
                ShardRecord(
                    name=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    n_rows=len(rows),
                )
            )
            for row in rows:
                self._seen.add(_key(row))
            index = max(index, int(path.stem.split("-")[-1]) + 1)
        if self.shards:
            self._write_ledger(self.shards)
        return index

    def _reconcile(self, declared: list[ShardRecord], on_disk: set[str]) -> list[ShardRecord]:
        """Check the ledger against the directory. Raises unless the difference is a
        single interrupted commit."""
        if not self.verify_on_open:
            return declared
        # A ledger entry with no file can only be the rename that never happened, and
        # only the LAST entry can be that. Anything earlier means a committed shard was
        # deleted, and its trajectories would be silently re-generated into a new shard.
        missing = [entry.name for entry in declared if entry.name not in on_disk]
        if missing:
            if missing != [declared[-1].name]:
                raise EvidenceTampered(
                    f"{self.directory}: shards declared in {LEDGER_NAME} are missing from "
                    f"disk: {missing}. Only an interrupted final commit may be missing; "
                    "an earlier one means committed evidence was deleted."
                )
            declared = declared[:-1]

        by_name = {entry.name: entry for entry in declared}
        problems: list[str] = []
        for name in sorted(on_disk):
            entry = by_name.get(name)
            if entry is None:
                problems.append(f"{name}: on disk but never declared in {LEDGER_NAME}")
                continue
            actual = hashlib.sha256((self.directory / name).read_bytes()).hexdigest()
            if actual != entry.sha256:
                problems.append(f"{name}: sha256 mismatch against {LEDGER_NAME}")
        if problems:
            raise EvidenceTampered(
                f"refusing to resume {self.directory}: committed evidence changed after it "
                f"was committed: {problems}"
            )
        return declared

    def completed(self) -> set[TrajectoryKey]:
        return set(self._seen)

    def append(self, row: dict) -> None:
        key = _key(row)
        if key in self._seen:
            raise ValueError(f"duplicate trajectory key in this run: {key}")
        self._seen.add(key)
        self._buffer.append(row)
        if len(self._buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> ShardRecord | None:
        if not self._buffer:
            return None
        name = f"{self.prefix}-{self._next_index:05d}.jsonl"
        temp = self.directory / (name + ".tmp")
        body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in self._buffer)
        with temp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        record = ShardRecord(
            name=name,
            sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            n_rows=len(self._buffer),
        )
        # Ledger BEFORE the rename. The reverse order leaves a window in which a shard is
        # visible and unhashed, and a resume cannot then distinguish those bytes from
        # edited ones — which is the hole this exists to close.
        self._write_ledger([*self.shards, record])
        os.replace(temp, self.directory / name)
        self.shards.append(record)
        self._buffer.clear()
        self._next_index += 1
        return record

    def close(self) -> list[ShardRecord]:
        self.flush()
        return list(self.shards)

    def manifest(self) -> list[dict]:
        return [s.to_dict() for s in self.shards]

    @property
    def n_rows(self) -> int:
        return sum(s.n_rows for s in self.shards) + len(self._buffer)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed shard line: {exc}") from exc
    return rows


def read_shards(directory: Path, *, prefix: str = "part") -> Iterator[dict]:
    for path in sorted(directory.glob(f"{prefix}-*.jsonl")):
        yield from _read_jsonl(path)


def completed_keys(directory: Path, *, prefix: str = "part") -> set[TrajectoryKey]:
    return {_key(row) for row in read_shards(directory, prefix=prefix)}


def read_nli_cache(path: Path, version: str) -> dict[str, bool]:
    """``{scorer key: leaks}`` from a run's scoring cache.

    The cache is keyed by ``sha256(version \\0 reference \\0 candidate)`` and stores no
    plaintext, so the key is recomputed by the caller from the raw rows. Keying on the
    version means a cache written by a different scorer simply does not match, rather than
    silently supplying another evaluator's verdicts.

    The rows carry ``label``, not ``leaks``: ``SemanticVerdict.leaks`` is a *property*
    (``label == "entailed"``) and so is absent from the serialised ``__dict__``. Reading a
    missing ``leaks`` key with a ``False`` default would have labelled all 23,040 texts
    clean while every key resolved — recall silently undefined instead of loudly broken.
    A row with neither field raises rather than defaulting.

    Shared by ``rdl graph-detector-recall`` and ``rdl graph-detector-corpus`` so that the
    labels the corpus is FROZEN on are the same labels recall is MEASURED against. Two
    implementations of this could drift, and a corpus built on one evaluator's verdicts
    while recall is scored on another's is a comparison of nothing.
    """
    if not path.exists():
        return {}
    out: dict[str, bool] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "key" not in row:
            continue
        if "label" in row:
            out[str(row["key"])] = row["label"] == "entailed"
        elif "leaks" in row:
            out[str(row["key"])] = bool(row["leaks"])
        else:
            raise ValueError(
                f"{path}:{lineno}: cache row has neither `label` nor `leaks`; refusing to "
                "guess a verdict, because defaulting to clean reports perfect containment"
            )
    return out


def scorer_label_fn(cache: dict[str, bool], version: str):
    """``(reference, candidate) -> True leaks / False clean / None never judged``."""

    def label(reference: str, candidate: str) -> bool | None:
        key = hashlib.sha256(
            (version + "\0" + reference + "\0" + candidate).encode("utf-8")
        ).hexdigest()
        return cache.get(key)

    return label


def verify_ledger(directory: Path, *, prefix: str = "part") -> dict:
    """Check every shard on disk against the ledger written when it was committed.

    Read-only, and reports rather than raises, so the finalizer can record the verdict
    on an incomplete run instead of dying on it. ``ShardWriter`` performs the same check
    at open time and *does* raise, because continuing to append to evidence that changed
    underneath is the failure worth stopping.
    """
    path = directory / LEDGER_NAME
    on_disk = sorted(p.name for p in directory.glob(f"{prefix}-*.jsonl"))
    if not path.exists():
        return {
            "ok": not on_disk,
            "status": "absent",
            "n_shards": len(on_disk),
            "n_declared": 0,
            "problems": (
                []
                if not on_disk
                else [f"{len(on_disk)} shards on disk and no {LEDGER_NAME} to check them against"]
            ),
        }
    try:
        declared = {e["name"]: e for e in json.loads(path.read_text(encoding="utf-8"))["shards"]}
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        return {
            "ok": False,
            "status": "unreadable",
            "n_shards": len(on_disk),
            "n_declared": 0,
            "problems": [f"{LEDGER_NAME} is unreadable: {exc}"],
        }

    problems: list[str] = []
    for name in on_disk:
        entry = declared.get(name)
        if entry is None:
            problems.append(f"{name}: on disk but never declared in {LEDGER_NAME}")
            continue
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            problems.append(f"{name}: sha256 mismatch against {LEDGER_NAME}")
    # A single trailing declared-but-absent shard is an interrupted commit, not tampering.
    absent = [name for name in declared if name not in set(on_disk)]
    if absent and absent != [sorted(declared)[-1]]:
        problems.append(f"declared but missing from disk: {sorted(absent)}")
    return {
        "ok": not problems,
        "status": "verified" if not problems else "failed",
        "n_shards": len(on_disk),
        "n_declared": len(declared),
        "interrupted_commit": absent if absent and not problems else [],
        "problems": problems,
    }


def verify_shards(directory: Path, manifest: Sequence[dict], *, prefix: str = "part") -> dict:
    """Recompute every shard hash. A modified shard must fail here, not in a report."""
    declared = {entry["name"]: entry for entry in manifest}
    on_disk = sorted(p.name for p in directory.glob(f"{prefix}-*.jsonl"))
    problems: list[str] = []
    for name in on_disk:
        if name not in declared:
            problems.append(f"{name}: present on disk but not in the manifest")
            continue
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != declared[name]["sha256"]:
            problems.append(f"{name}: sha256 mismatch (evidence modified after the run)")
    for name in declared:
        if name not in on_disk:
            problems.append(f"{name}: declared in the manifest but missing from disk")
    return {
        "ok": not problems,
        "n_shards": len(on_disk),
        "n_declared": len(declared),
        "problems": problems,
    }
