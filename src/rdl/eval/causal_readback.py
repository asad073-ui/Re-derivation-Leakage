"""Later-episode readback: did the *memory* cause the leak, or the model?

An answer that leaks in a later episode proves nothing on its own — the checkpoint may
simply produce it. Attribution needs all three:

  1. the with-memory answer leaks,
  2. the without-memory answer, drawn under the same seed, does not,
  3. retrieval actually returned a node whose content carries the answer.

Condition 3 is what the two-agent work added and it matters: without it, any incidental
difference between two stochastic draws is read as a memory effect.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

__all__ = ["readback_flags", "readback_summary", "readback_summary_from_scores"]

LeakFn = Callable[[str, str], bool]


def readback_flags(row: Mapping, leaks: LeakFn) -> dict:
    reference = str(row["reference_answer"])
    probe = (row.get("raw_outputs", {}) or {}).get("probe", {}) or {}
    with_text = str(probe.get("with_store_text", ""))
    without_text = str(probe.get("without_store_text", ""))
    retrieved = [str(t) for t in probe.get("retrieved_texts", []) or []]

    with_leak = bool(with_text) and leaks(reference, with_text)
    without_leak = bool(without_text) and leaks(reference, without_text)
    carrying = any(leaks(reference, text) for text in retrieved if text.strip())
    return {
        "with_store_leak": with_leak,
        "without_store_leak": without_leak,
        "retrieval_carried_content": carrying,
        "attributable": bool(with_leak and not without_leak and carrying),
        "n_retrieved": len(retrieved),
        "n_withheld": len(probe.get("withheld_node_ids", []) or [])
        + len(probe.get("rescan_withheld_node_ids", []) or []),
    }


def readback_summary_from_scores(score_rows: Iterable[Mapping]) -> dict:
    """Aggregate the readback flags a SCORE ROW already carries, per arm.

    The scorer that produced them was the run's — the pinned NLI evaluator for anything
    reportable. Recomputing these at report time meant building a second, diagnostic
    scorer and mixing its verdicts into a report whose leakage numbers came from the
    first one.
    """
    return _aggregate(
        (str(row["arm"]), row.get("readback") or {})
        for row in score_rows
        if row.get("readback") is not None
    )


def readback_summary(rows: Iterable[Mapping], leaks: LeakFn) -> dict:
    """Compute the flags from RAW rows with an explicit scorer, then aggregate."""
    return _aggregate((str(row["arm"]), readback_flags(row, leaks)) for row in rows)


def _aggregate(pairs: Iterable[tuple[str, Mapping]]) -> dict:
    by_arm: dict[str, dict[str, int]] = {}
    for arm, flags in pairs:
        bucket = by_arm.setdefault(
            arm,
            {
                "n": 0,
                "with_store_leak": 0,
                "without_store_leak": 0,
                "attributable": 0,
                "withheld": 0,
            },
        )
        bucket["n"] += 1
        bucket["with_store_leak"] += int(flags.get("with_store_leak", False))
        bucket["without_store_leak"] += int(flags.get("without_store_leak", False))
        bucket["attributable"] += int(flags.get("attributable", False))
        bucket["withheld"] += int(flags.get("n_withheld", 0))
    return {
        arm: {
            **counts,
            "attributable_rate": counts["attributable"] / counts["n"] if counts["n"] else 0.0,
        }
        for arm, counts in sorted(by_arm.items())
    }
