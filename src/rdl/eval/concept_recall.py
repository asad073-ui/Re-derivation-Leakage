"""Detector recall on text the system ACTUALLY GENERATED, not on forget questions.

The calibration artefact reports recall 0.95 — measured on held-out *questions* about the
forget-policy authors. That is recall on the thing a request guard sees. It is not recall
on the thing a graph guard has to catch, which is a paraphrase an agent produced three
hops later, a payload on an edge, or a sentence about to be written to the store.

Those two numbers can differ by an order of magnitude, and when they do, a defence's
failure is a *detection* failure wearing a propagation failure's clothes. The 50x32
discovery run is the case in point: GraphForget's Forget-ID propagation and its enforcement
across edges, writes and retrieval all worked, and its natural-flow Leak@32 was identical
to the unguarded arm's on every surface — because the detector never fired on the content
that leaked, so there was nothing to propagate.

**What this measures.** For each generated text the run committed to evidence:

  * the LABEL comes from the run's own semantic scorer — the same pinned NLI verdicts the
    leak surfaces were built from, looked up out of the scoring cache. Not a fresh
    judgement, so recall here is measured against exactly the leaks the report counts.
  * the PREDICTION comes from running the run's detector, at the run's frozen threshold,
    over that text.

Recall is then ``fired & leaking / leaking``, per surface, and the complement is the miss
list: the concrete strings a graph defence had to catch and did not.

**Why the unguarded arm is the right place to measure it.** Under enforcement, leaking
text is suppressed before it is recorded, so a guarded arm's evidence is thinned by the
very mechanism being measured. MA-LEAK generates and records everything, so it is the
only arm whose text is an unbiased sample of what the model produces. Guarded arms are
still reported, marked, and never pooled with it.

``restrict_to_row_concept`` is the honest strictness knob. A detector that fires on the
wrong concept has not caught the leak — it has produced a false alarm that happens to
coincide with one — so the primary recall requires the fired Forget-IDs to include the
row's own concept. Unrestricted firing is reported beside it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence

__all__ = ["GENERATED_SURFACES", "detector_recall_on_generated_leakage", "generated_texts"]

# ``(reference, candidate) -> True leaks / False clean / None never judged``.
LabelFn = Callable[[str, str], bool | None]
# ``texts -> per-text (fired, forget_ids)``. Batched because the detector is.
DetectFn = Callable[[Sequence[str]], Sequence[tuple[bool, Sequence[str]]]]

GENERATED_SURFACES: tuple[str, ...] = (
    "agent_message",
    "edge_payload",
    "stored_node",
    "final_text",
)


def generated_texts(row: Mapping) -> list[tuple[str, str]]:
    """``(surface, text)`` for everything one trajectory put into the world.

    Deliberately NOT the readback probe texts: those are a counterfactual second episode
    and a detector is never asked to guard them at generation time.
    """
    raw = row.get("raw_outputs", {}) or {}
    out: list[tuple[str, str]] = []
    out += [("agent_message", str(t)) for t in raw.get("agent_messages", []) or []]
    out += [("edge_payload", str(t)) for t in raw.get("released_edge_payloads", []) or []]
    out += [
        ("stored_node", str(e.get("content", ""))) for e in row.get("memory_evidence", []) or []
    ]
    final = str(row.get("final_text", ""))
    if final:
        out.append(("final_text", final))
    return [(surface, text) for surface, text in out if text and text.strip()]


def _blank() -> dict:
    return {
        "n_texts": 0,
        "n_judged": 0,
        "n_leaking": 0,
        "n_leaking_detected": 0,
        "n_leaking_detected_any_concept": 0,
        "n_clean": 0,
        "n_clean_fired": 0,
    }


def _rates(bucket: Mapping[str, int]) -> dict:
    leaking = bucket["n_leaking"]
    clean = bucket["n_clean"]
    return {
        **bucket,
        "n_missed": leaking - bucket["n_leaking_detected"],
        "recall": (bucket["n_leaking_detected"] / leaking) if leaking else None,
        "recall_any_concept": (
            bucket["n_leaking_detected_any_concept"] / leaking if leaking else None
        ),
        # False-alarm rate on GENERATED clean text. Not the calibration FPR, which is
        # measured on retain probes; this one says how noisy the detector is in situ.
        "false_alarm_rate": (bucket["n_clean_fired"] / clean) if clean else None,
    }


def detector_recall_on_generated_leakage(
    rows: Iterable[Mapping],
    *,
    label: LabelFn,
    detect: DetectFn,
    primary_arm: str = "multi_agent_leak",
    restrict_to_row_concept: bool = True,
    max_misses: int = 50,
) -> dict:
    """Recall of the run's detector against the run's own leak labels.

    ``label`` returning ``None`` means the scorer never judged that pair — which happens
    only if the scoring cache is incomplete — and those texts are counted as unjudged and
    excluded from every rate rather than assumed clean.
    """
    collected: list[tuple[str, str, str, str, str]] = []  # arm, surface, concept, ref, text
    for row in rows:
        reference = str(row.get("reference_answer", ""))
        arm = str(row.get("arm", "?"))
        concept = str(row.get("concept_id", ""))
        for surface, text in generated_texts(row):
            collected.append((arm, surface, concept, reference, text))

    verdicts = detect([text for _arm, _s, _c, _r, text in collected]) if collected else []
    if len(verdicts) != len(collected):
        raise ValueError(
            f"detector returned {len(verdicts)} verdicts for {len(collected)} texts; "
            "the batch callable must preserve order and length"
        )

    by_arm: dict[str, dict[str, dict]] = {}
    misses: list[dict] = []
    n_unjudged = 0
    for (arm, surface, concept, reference, text), (fired, forget_ids) in zip(
        collected, verdicts, strict=True
    ):
        arm_bucket = by_arm.setdefault(arm, {})
        bucket = arm_bucket.setdefault(surface, _blank())
        overall = arm_bucket.setdefault("all", _blank())
        for target in (bucket, overall):
            target["n_texts"] += 1

        leaks = label(reference, text)
        if leaks is None:
            n_unjudged += 1
            continue
        on_concept = bool(fired) and (
            not restrict_to_row_concept or concept in set(forget_ids) or not concept
        )
        for target in (bucket, overall):
            target["n_judged"] += 1
            if leaks:
                target["n_leaking"] += 1
                target["n_leaking_detected"] += int(on_concept)
                target["n_leaking_detected_any_concept"] += int(bool(fired))
            else:
                target["n_clean"] += 1
                target["n_clean_fired"] += int(bool(fired))
        if leaks and not on_concept and arm == primary_arm and len(misses) < max_misses:
            misses.append(
                {
                    "surface": surface,
                    "concept_id": concept,
                    "fired_on_other_concept": bool(fired),
                    "text": text[:400],
                }
            )

    summary = {
        arm: {surface: _rates(bucket) for surface, bucket in sorted(surfaces.items())}
        for arm, surfaces in sorted(by_arm.items())
    }
    headline = (summary.get(primary_arm) or {}).get("all") or _rates(_blank())
    return {
        "schema": "graph-detector-recall-v1",
        "primary_arm": primary_arm,
        "primary_arm_note": (
            "recall is headlined on the UNGUARDED arm: under enforcement the leaking text "
            "is suppressed before it is recorded, so a guarded arm's evidence is thinned "
            "by the mechanism being measured"
        ),
        "restrict_to_row_concept": restrict_to_row_concept,
        "n_unjudged_texts": n_unjudged,
        "recall": headline.get("recall"),
        "n_leaking": headline.get("n_leaking", 0),
        "n_missed": headline.get("n_missed", 0),
        "false_alarm_rate": headline.get("false_alarm_rate"),
        "by_arm": summary,
        "missed_examples": misses,
        "note": (
            "labels are the run's own pinned-scorer verdicts, so this recall is measured "
            "against exactly the leaks the leak surfaces counted. It is NOT the "
            "calibration recall, which is measured on forget QUESTIONS and says nothing "
            "about paraphrases produced downstream in the graph."
        ),
    }
