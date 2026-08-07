"""Console logging + JSONL event sinks.

Two separate concerns, deliberately not merged:

- `get_logger` is for humans watching a run.
- `JsonlWriter` is for machines reading it back. Transcripts and manifests go here.
  It flushes on every record: a Colab session that dies mid-run must still leave a
  readable prefix of the log behind.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

__all__ = ["JsonlWriter", "configure_logging", "dumps_canonical", "get_logger", "read_jsonl"]

_CONFIGURED = False
_DEFAULT_FMT = "%(asctime)s %(levelname)-7s %(name)-22s %(message)s"


def configure_logging(level: str | int = "INFO", *, rich: bool = True) -> None:
    """Idempotent root-logger setup. Uses rich when available, stdlib otherwise."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    handler: logging.Handler
    if rich and os.environ.get("RDL_PLAIN_LOGS") != "1":
        try:
            from rich.logging import RichHandler

            handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
            fmt = "%(name)-22s %(message)s"
        except ImportError:
            handler = logging.StreamHandler(sys.stderr)
            fmt = _DEFAULT_FMT
    else:
        handler = logging.StreamHandler(sys.stderr)
        fmt = _DEFAULT_FMT

    handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


def dumps_canonical(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    Used anywhere bytes must be stable across runs and machines — config hashing,
    manifest lines, invariant witnesses.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


class JsonlWriter:
    """Append-only JSONL sink. Context manager; flushes every record."""

    def __init__(self, path: str | Path, *, append: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._mode = "a" if append else "w"
        self._fh: IO[str] | None = None

    def __enter__(self) -> JsonlWriter:
        self._fh = self.path.open(self._mode, encoding="utf-8", newline="\n")
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def write(self, record: Any) -> None:
        fh = self._fh
        if fh is None:
            fh = self._fh = self.path.open(self._mode, encoding="utf-8", newline="\n")
        if hasattr(record, "model_dump"):
            record = record.model_dump(mode="json")
        fh.write(dumps_canonical(record) + "\n")
        # Flush per record: a killed Colab session must leave a readable prefix.
        fh.flush()

    def write_all(self, records: Any) -> None:
        for r in records:
            self.write(r)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_jsonl(path: str | Path) -> Iterator[dict]:
    """Stream a JSONL file, skipping blank lines. Raises on malformed JSON."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: malformed JSONL: {exc}") from exc
