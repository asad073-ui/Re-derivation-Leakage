#!/usr/bin/env python
"""Verify that the v4.4 CPU phase is complete and coherent, before any GPU time is billed.

Run after ``make cpu-all`` and before ``rdl graph-detector-v4-4-env-check --strict``. Every
check here is one that otherwise surfaces forty minutes into a judging run with the clock
ticking, and each is checked against the FILES ON DISK rather than against the code that
wrote them -- a builder and its own assertions agreeing proves nothing about what was
committed.

Nothing here loads a model, touches the network, or writes a file.

Exit codes
----------
``0``  every check passed; the GPU phase may start.
``1``  at least one check failed; the report names which and why.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COHORT = REPO / "data" / "cohorts" / "graph_unlearning_v1"
V4_3 = COHORT / "detector_v4_3"
V4_4 = COHORT / "detector_v4_4"

PAIR_SEPARATOR = "\x1f"

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    checks.append((name, bool(ok), detail))
    return bool(ok)


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:  # pragma: no cover - defensive
        check(f"{path.name} parses", False, str(error)[:120])
        return None


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    # ------------------------------------------------------------ the four artifacts --
    bundle = _load(V4_4 / "DETECTOR_V4_4_PAIR_BUNDLE.json")
    panel = _load(V4_4 / "DETECTOR_V4_4_CALIBRATION_PANEL.json")
    supplement = _load(V4_4 / "DETECTOR_V4_4_SUPPLEMENT.json")
    smoke = _jsonl(V4_4 / "V4_4_JUDGE_SMOKE_60.jsonl")

    if not check("bundle exists", bundle is not None, "rdl graph-detector-v4-4-bundle"):
        return report()
    if not check("panel exists", panel is not None, "rdl graph-detector-v4-4-panel"):
        return report()
    check("supplement exists", supplement is not None, "rdl graph-detector-v4-4-supplement")
    assert bundle is not None and panel is not None

    pairs = bundle["pairs"]

    # -------------------------------------------------------------------- the bundle --
    check(
        "bundle keeps all 1,019 original pairs",
        bundle["n_from_audit"] == 1019,
        f"n_from_audit={bundle['n_from_audit']}",
    )
    check(
        "bundle is roughly the planned size",
        1600 <= bundle["n_pairs"] <= 1900,
        f"n_pairs={bundle['n_pairs']} (1,019 audit + {bundle['n_from_supplement']} supplement)",
    )

    hashes = Counter(p["pair_sha256"] for p in pairs)
    check(
        "every (question, candidate) pair is unique",
        max(hashes.values()) == 1,
        f"{sum(1 for n in hashes.values() if n > 1)} duplicated pairs",
    )
    check(
        "pair hashes match their own contents",
        all(
            hashlib.sha256(
                (p["conditioning_question"] + PAIR_SEPARATOR + p["candidate_text"]).encode("utf-8")
            ).hexdigest()
            == p["pair_sha256"]
            for p in pairs
        ),
        "recomputed from the file, not trusted from it",
    )

    groups = {"train": set(), "development": set()}
    for pair in pairs:
        groups.setdefault(pair["split"], set()).add(pair["subject_group"])
    check(
        "splits are subject-group disjoint",
        not (groups["train"] & groups["development"]),
        f"{len(groups['train'])} train / {len(groups['development'])} development groups",
    )

    split_of_candidate: dict[str, str] = {}
    straddling = [
        p["audit_id"]
        for p in pairs
        if split_of_candidate.setdefault(
            hashlib.sha256(p["candidate_text"].encode("utf-8")).hexdigest(), p["split"]
        )
        != p["split"]
    ]
    check(
        "no candidate text straddles the split",
        not straddling,
        f"{len(straddling)} straddling (first {straddling[:2]})",
    )

    check(
        "only the three answer-free fields are tokenized",
        set(bundle["tokenized_fields"])
        == {"conditioning_question", "subject_aliases", "candidate_text"},
        str(bundle["tokenized_fields"]),
    )
    for forbidden in ("intended_class", "source_subtype", "population", "split", "reference_answer"):
        check(
            f"{forbidden} is declared never-tokenized",
            forbidden in bundle["fields_never_tokenized"],
        )

    # -------------------------------------------------- the class the gate is about --
    intended = Counter(str(p.get("intended_class")) for p in pairs)
    check(
        "the bundle contains enough non-attempt-intent rows to reach 200 adjudicated NONE",
        intended["NONE"] >= 500,
        f"{intended['NONE']} intended NONE (v4.3's judges found 69 and 17)",
    )
    check(
        "the bundle contains enough partial-intent rows",
        intended["PARTIAL"] >= 200,
        f"{intended['PARTIAL']} intended PARTIAL",
    )

    subtypes = Counter(
        p["source_subtype"] for p in pairs if p["origin"] == "v4_4_supplement"
    )
    for required in ("refusal", "planning_process", "tool_memory_status", "subject_only"):
        check(f"non-attempt subtype {required} is present", subtypes[required] >= 100, str(subtypes[required]))
    check(
        "off-topic / cross-question subtype is present",
        subtypes["off_topic"] + subtypes["cross_question"] >= 100,
        f"off_topic={subtypes['off_topic']} cross_question={subtypes['cross_question']}",
    )

    # --------------------------------------------------------------- diversity ------
    diversity = bundle.get("supplement_diversity", {}).get("by_subtype", {})
    from rdl.eval.detector_v4_4_supplement import REAL_TEXT_SUBTYPES

    collapsed = [
        subtype
        for subtype, metrics in diversity.items()
        if subtype not in REAL_TEXT_SUBTYPES
        and (
            (metrics.get("max_frame_share") or 1.0) > 0.25
            or (metrics.get("max_leading_trigram_share") or 1.0) > 0.20
            or (metrics.get("unique_candidate_rate") or 0.0) < 1.0
        )
    ]
    check(
        "no generated subtype collapsed onto a template",
        not collapsed,
        f"collapsed: {collapsed}" if collapsed else f"{len(diversity)} subtypes checked",
    )

    separability = bundle.get("shortcut_probe_supplement", {}).get(
        "worst_case_balanced_accuracy"
    )
    check(
        "the supplement separability probe ran and is decomposed",
        separability is not None and bool(bundle["shortcut_probe_supplement"].get("by_subtype")),
        f"worst-case balanced accuracy {separability} "
        f"({bundle['shortcut_probe_supplement'].get('worst_case_feature')}) -- reported, "
        "not gated; see the protocol",
    )

    # --------------------------------------------------------------------- the panel --
    composition = panel["composition"]["by_intended_class"]
    check(
        "the panel is balanced 200/200/200 by intended hidden source",
        composition == {"ANSWER": 200, "NONE": 200, "PARTIAL": 200},
        str(composition),
    )
    check(
        "the panel names the bundle now on disk",
        panel["bundle_sha256"] == bundle["bundle_sha256"],
        f"{str(panel['bundle_sha256'])[:12]} vs {str(bundle['bundle_sha256'])[:12]}",
    )
    bundle_ids = {p["audit_id"] for p in pairs}
    missing = [r["audit_id"] for r in panel["rows"] if r["audit_id"] not in bundle_ids]
    check("every panel row exists in the bundle", not missing, f"{len(missing)} missing")
    check(
        "no panel row carries a label",
        all("answer_attempt" not in r and "label" not in r for r in panel["rows"]),
        "the panel carries an INTENT, which is not a label",
    )
    by_type = panel["composition"]["by_question_type_hint"]
    check(
        "the panel covers all three question types",
        set(by_type) >= {"slot", "yes-no", "open-ended"} and min(by_type.values()) >= 100,
        str(by_type),
    )

    # --------------------------------------------------------------------- the smoke --
    check("the smoke fixture is 60 rows", len(smoke) == 60, f"{len(smoke)} rows")
    if smoke:
        check(
            "the smoke fixture is 20 per intended class",
            Counter(r["_intended_class"] for r in smoke)
            == {"ANSWER": 20, "NONE": 20, "PARTIAL": 20},
            str(dict(Counter(r["_intended_class"] for r in smoke))),
        )
        used = set(hashes)
        overlap = [
            r["audit_id"]
            for r in smoke
            if hashlib.sha256(
                (r["conditioning_question"] + PAIR_SEPARATOR + r["candidate_text"]).encode("utf-8")
            ).hexdigest()
            in used
        ]
        check(
            "the smoke fixture is disjoint from the bundle",
            not overlap,
            f"{len(overlap)} overlapping rows",
        )

    # ------------------------------------------------------------- v4.3 preservation --
    from rdl.eval.detector_v4_4_preservation import (
        SEALED_DIGESTS,
        V4_3_FROZEN_DIGESTS,
        changed_artifacts,
    )

    changed = changed_artifacts(COHORT)
    check(
        "every v4.3 artifact and the sealed bank are byte-identical",
        not changed,
        f"{changed[:3]}"
        if changed
        else f"{len(V4_3_FROZEN_DIGESTS) + len(SEALED_DIGESTS)} files unchanged since PR #52",
    )

    # ----------------------------------------------------------- no judge output yet --
    from rdl.cli.detector_v4_4_judge import output_names_v4_4

    existing = [
        output_names_v4_4(judge=j, pass_name=p, run_id="", reportable=True)["output"]
        for j in ("A", "B")
        for p in ("blind", "reference")
        if (V4_4 / output_names_v4_4(judge=j, pass_name=p, run_id="", reportable=True)["output"]).exists()
    ]
    check(
        "no reportable v4.4 judge pass has been written yet",
        not existing,
        f"already present: {existing}. The panel must be frozen BEFORE judging; if these "
        "are from an earlier attempt, move them aside deliberately."
        if existing
        else "the panel is frozen before any label exists",
    )

    # ------------------------------------------------------------------ the contract --
    from rdl.eval.detector_v4_4 import PANEL_GATES, PROMPT_VERSION

    check(
        "the kappa bound did not move from v4.3",
        PANEL_GATES["panel_kappa"] == (">=", 0.70),
        f"{PANEL_GATES['panel_kappa']} -- what moved is the distribution, not the bound",
    )
    check(
        "the prompt version is a v4.4 version",
        PROMPT_VERSION.startswith("v4.4-"),
        PROMPT_VERSION,
    )

    return report()


def report() -> int:
    failures = [c for c in checks if not c[1]]
    width = max(len(name) for name, _, _ in checks)
    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}")
    print()
    if failures:
        print(f"{len(failures)} of {len(checks)} checks FAILED. Do not start the GPU phase.")
        return 1
    print(f"all {len(checks)} checks passed. The v4.4 CPU phase is complete.")
    print("next: rdl graph-detector-v4-4-env-check --strict, then nvidia-smi, then GPU-1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
