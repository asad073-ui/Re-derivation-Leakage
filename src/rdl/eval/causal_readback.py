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

__all__ = ["readback_flags", "readback_summary"]

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


def readback_summary(rows: Iterable[Mapping], leaks: LeakFn) -> dict:
    by_arm: dict[str, dict[str, int]] = {}
    for row in rows:
        arm = str(row["arm"])
        flags = readback_flags(row, leaks)
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
        bucket["with_store_leak"] += int(flags["with_store_leak"])
        bucket["without_store_leak"] += int(flags["without_store_leak"])
        bucket["attributable"] += int(flags["attributable"])
        bucket["withheld"] += int(flags["n_withheld"])
    return {
        arm: {
            **counts,
            "attributable_rate": counts["attributable"] / counts["n"] if counts["n"] else 0.0,
        }
        for arm, counts in sorted(by_arm.items())
    }
