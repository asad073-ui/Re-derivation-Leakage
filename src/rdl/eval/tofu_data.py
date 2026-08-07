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

__all__ = [
    "FIXTURE_RELPATH",
    "QUESTIONS_PER_AUTHOR",
    "SPLIT_SIZES",
    "TofuItem",
    "as_forget_items",
    "cluster_ids",
    "load_fixture",
    "load_items",
    "load_tofu",
]

FIXTURE_RELPATH = "tests/fixtures/tofu_forget10_sample.json"

# TOFU is 200 synthetic authors x 20 questions. The splits are contiguous blocks of
# authors, so a split's size is a multiple of 20 and question i belongs to author
# i // 20 *within that split*. This is what makes the author the right resampling unit
# for the paired bootstrap: twenty questions about one invented novelist are not twenty
# independent observations.
QUESTIONS_PER_AUTHOR = 20

# Expected sizes, used to fail loudly when a split silently loads short. A run that
# reports 400 items and evaluated 8 is the failure this constant exists to prevent.
SPLIT_SIZES: dict[str, int] = {
    "forget01": 40,
    "forget05": 200,
    "forget10": 400,
    "retain90": 3600,
    "retain95": 3800,
    "retain99": 3960,
    "full": 4000,
}


@dataclass(frozen=True)
class TofuItem:
    item_id: str
    question: str
    answer: str
    split: str = "forget10"
    # Index within the split. Drives `author_id`.
    index: int = 0

    @property
    def author_id(self) -> str:
        """Cluster key for the paired bootstrap. See `QUESTIONS_PER_AUTHOR`."""
        return f"{self.split}-author-{self.index // QUESTIONS_PER_AUTHOR:04d}"

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "answer": self.answer,
            "split": self.split,
            "index": self.index,
            "author_id": self.author_id,
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
            index=int(r.get("index", i)),
        )
        for i, r in enumerate(items)
    ]


def load_tofu(
    split: str = "forget10", n: int | None = None, token: str | None = None
) -> list[TofuItem]:
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
                index=i,
            )
        )

    expected = SPLIT_SIZES.get(split)
    if n is None and expected is not None and len(items) != expected:
        raise ValueError(
            f"TOFU split '{split}' loaded {len(items)} items, expected {expected}. "
            "Refusing to run on a short split: a truncated forget set changes every "
            "rate in the results table without changing anything that says so."
        )
    return items


def spread_sample(items: Sequence[TofuItem], n: int) -> list[TofuItem]:
    """`n` items spread evenly across the split, preserving order.

    TOFU splits are contiguous blocks of authors, so the first `n` items are the first
    `n / 20` authors. A 100-item "retain sample" taken off the head is therefore five
    novelists, not a sample of the retain set, and a false-positive floor measured on it
    says nothing about the other 175. Even spacing costs nothing and covers the split.

    Deterministic: no RNG, so the sample does not move between seeds and the same config
    hash means the same questions.
    """
    total = len(items)
    if n >= total or n <= 0:
        return list(items)
    step = total / n
    return [items[int(i * step)] for i in range(n)]


def load_items(
    *,
    dataset: str = "tofu",
    split: str = "forget10",
    n_items: int | None = None,
    fixture: str | Path | None = None,
    token: str | None = None,
    allow_fixture: bool = False,
    sample: str = "head",
) -> tuple[list[TofuItem], dict]:
    """Resolve the item set a condition should run on, and say where it came from.

    Returns ``(items, provenance)``. `provenance` goes verbatim into the run report, so
    a reader can never be in doubt about whether a number came from 400 real TOFU
    questions or from the eight-item development fixture.

    The default is the REAL split. `fixture` (or `dataset="stub"`) selects the offline
    sample, and unless `allow_fixture` is set that combination raises — which is the
    whole point. `run_condition` previously called `load_fixture()` unconditionally and
    ignored `data.dataset`, `data.forget_split` and `data.n_items` entirely, so every
    "production" run silently evaluated eight hand-written questions while the config,
    the report, and the pre-registration all said 400.
    """
    if dataset == "stub" or fixture is not None:
        if not allow_fixture:
            raise ValueError(
                "refusing to run on the development fixture. It is eight hand-written "
                "items and is not TOFU. Pass --allow-fixture to acknowledge that the "
                "resulting numbers are a smoke test, not a result."
            )
        items = load_fixture(fixture)
        if n_items:
            items = items[:n_items]
        return items, {
            "source": "fixture",
            "path": str(fixture or FIXTURE_RELPATH),
            "split": split,
            "n_items": len(items),
            "is_real_data": False,
            "warning": "DEVELOPMENT FIXTURE — not TOFU. Not reportable.",
        }

    if sample == "spread" and n_items:
        # Load the whole split first so the spacing is over the real thing, then
        # subsample. `load_tofu(split, n)` truncates at load time, which is the head.
        items = spread_sample(load_tofu(split, None, token=token), n_items)
    elif sample in ("head", "spread"):
        items = load_tofu(split, n_items, token=token)
    else:
        raise ValueError(f"unknown sample strategy '{sample}' (expected head|spread)")

    return items, {
        "source": "locuslab/TOFU",
        "split": split,
        "sample": sample,
        "n_items": len(items),
        "expected_split_size": SPLIT_SIZES.get(split),
        "truncated": n_items is not None and n_items < (SPLIT_SIZES.get(split) or 0),
        "is_real_data": True,
    }


def cluster_ids(items: Sequence[TofuItem], by: str = "author") -> list[str]:
    """Cluster key per item for the paired bootstrap. `item` means no clustering."""
    if by == "item":
        return [it.item_id for it in items]
    if by == "author":
        return [it.author_id for it in items]
    raise ValueError(f"unknown cluster_by '{by}' (expected item|author)")


def as_forget_items(items: Sequence[TofuItem]) -> list[dict]:
    """Shape TOFU items for `laundering.laundered_items`."""
    return [{"item_id": it.item_id, "answer": it.answer} for it in items]


def as_queries(items: Sequence[TofuItem]) -> list[tuple[str, str]]:
    """Shape TOFU items for `orchestrator.loop.run_episodes`."""
    return [(it.question, it.item_id) for it in items]


def iter_qa(items: Sequence[TofuItem]) -> Iterator[tuple[str, str]]:
    for it in items:
        yield it.question, it.answer
