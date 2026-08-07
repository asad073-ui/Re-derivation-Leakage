"""TOFU data access.

Two paths, on purpose:

  load_fixture()   reads a checked-in JSON sample. No network, no `datasets`. This is
                   what the CPU gate and every contract test use.
  load_tofu()      pulls the real `locuslab/TOFU` split from the Hub. Needs network.

Keeping the fixture path first-class means the orchestrator, the metrics, and the
conditions can all be exercised end-to-end before a single byte is downloaded.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..paths import repo_root

__all__ = ["FIXTURE_RELPATH", "TofuItem", "as_forget_items", "load_fixture", "load_tofu"]

FIXTURE_RELPATH = "tests/fixtures/tofu_forget10_sample.json"


@dataclass(frozen=True)
class TofuItem:
    item_id: str
    question: str
    answer: str
    split: str = "forget10"

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "answer": self.answer,
            "split": self.split,
        }


def load_fixture(path: str | Path | None = None) -> list[TofuItem]:
    """Load the checked-in TOFU sample. No network."""
    p = Path(path) if path else repo_root() / FIXTURE_RELPATH
    if not p.exists():
        raise FileNotFoundError(f"TOFU fixture not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    items = raw["items"] if isinstance(raw, dict) else raw
    return [
        TofuItem(
            item_id=str(r.get("item_id") or r.get("id") or i),
            question=r["question"],
            answer=r["answer"],
            split=str(r.get("split", "forget10")),
        )
        for i, r in enumerate(items)
    ]


def load_tofu(split: str = "forget10", n: int | None = None, token: str | None = None):
    """Load a real TOFU split from the Hub. NEEDS NETWORK.

    Kept out of the CPU gate deliberately — `load_fixture` is the offline equivalent.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "load_tofu requires `datasets`. For offline work use load_fixture()."
        ) from exc

    ds = load_dataset("locuslab/TOFU", split, split="train", token=token)
    items: list[TofuItem] = []
    for i, row in enumerate(ds):
        if n is not None and i >= n:
            break
        items.append(
            TofuItem(
                item_id=f"{split}-{i:04d}",
                question=row["question"],
                answer=row["answer"],
                split=split,
            )
        )
    return items


def as_forget_items(items: Sequence[TofuItem]) -> list[dict]:
    """Shape TOFU items for `laundering.laundered_items`."""
    return [{"item_id": it.item_id, "answer": it.answer} for it in items]


def as_queries(items: Sequence[TofuItem]) -> list[tuple[str, str]]:
    """Shape TOFU items for `orchestrator.loop.run_episodes`."""
    return [(it.question, it.item_id) for it in items]


def iter_qa(items: Sequence[TofuItem]) -> Iterator[tuple[str, str]]:
    for it in items:
        yield it.question, it.answer
