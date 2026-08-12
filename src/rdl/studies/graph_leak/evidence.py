"""Immutable evidence shards.

The historical runner flushed and fsynced after every row, which is safe and slow. Here
a shard accumulates ``shard_size`` trajectories, is written to a temporary file, hashed,
atomically renamed, and its hash appended to the manifest. A killed process leaves a
partial temporary file, which is discarded on resume — never a torn committed shard.

Resume is by shard, and the resume key is ``(item_id, sample_id, arm, challenge)``. A
trajectory already present in a committed shard is not re-run, and a duplicate key is a
hard error rather than a silent overwrite.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ShardWriter", "TrajectoryKey", "completed_keys", "read_shards", "verify_shards"]

TrajectoryKey = tuple[str, int, str, str]


def _key(row: dict) -> TrajectoryKey:
    return (
        str(row["item_id"]),
        int(row["sample_id"]),
        str(row["arm"]),
        str(row.get("challenge", "natural")),
    )


def atomic_json(path: Path, payload: dict) -> None:
    """Replace a manifest atomically: a killed process must not corrupt the contract."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
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

    def __init__(self, directory: Path, *, prefix: str = "part", shard_size: int = 64) -> None:
        if shard_size < 1:
            raise ValueError("shard_size must be >= 1")
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.shard_size = shard_size
        self._buffer: list[dict] = []
        self.shards: list[ShardRecord] = []
        self._seen: set[TrajectoryKey] = set()
        self._next_index = self._scan_existing()

    def _scan_existing(self) -> int:
        index = 0
        for path in sorted(self.directory.glob(f"{self.prefix}-*.jsonl")):
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
        # A temporary file is an interrupted commit and carries no committed rows.
        for temp in self.directory.glob(f"{self.prefix}-*.jsonl.tmp"):
            temp.unlink()
        return index

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
        os.replace(temp, self.directory / name)
        record = ShardRecord(
            name=name,
            sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
            n_rows=len(self._buffer),
        )
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
