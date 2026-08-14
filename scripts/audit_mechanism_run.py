"""OPERATOR-SIDE runbook gate. Committed so the runbook cannot outlive it.

This is the check a rented-GPU runbook runs BEFORE and AFTER generation. It duplicates
nothing: the authoritative pathway-liveness conditions live in
`rdl.eval.mechanism_liveness` and are enforced inside GRAPH_LEAK_REPORT.json by
`rdl graph-audit-mechanism`. What this script adds is the surrounding operator contract —
provenance, volume and cohort facts that a report has no reason to check because they are
about whether the RUNBOOK was followed, not about whether the science is readable.

It lived in an operator's home directory for the run that produced the M5 result, which
means the gate was only as good as whoever remembered to copy it onto the box.
"""

# Original purpose: Hard runtime audit of a mechanism run.

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

NO_FORWARD = "multi_agent_graphforget_no_forward"
TAINT_FORWARD = "multi_agent_graphforget_taint_forward"
QUARANTINE = "multi_agent_graphforget_tag_source_quarantine"

EXPECTED_ARMS = {
    "single_agent",
    "multi_agent_control",
    "multi_agent_leak",
    "multi_agent_dragon",
    "multi_agent_stateless",
    "multi_agent_graphforget_semantic_only",
    QUARANTINE,
    NO_FORWARD,
    TAINT_FORWARD,
    "multi_agent_graphforget_taint_only",
    "multi_agent_graphforget",
}

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--items", type=int, required=True)
parser.add_argument("--samples", type=int, required=True)
parser.add_argument("--trajectories", type=int, required=True)
parser.add_argument("--graph", type=int, required=True)
parser.add_argument("--probe", type=int, required=True)
parser.add_argument("--full", action="store_true")
args = parser.parse_args()

run = Path(args.run)
required_commit = os.environ.get("RDL_REQUIRED_COMMIT")

failures: list[str] = []
notes: list[str] = []


def reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def load(name: str) -> dict:
    path = run / name
    if not path.exists():
        failures.append(f"{name} is missing from {run}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def check(label: str, actual, expected) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, got {actual!r}")


manifest = load("RUN_MANIFEST.json")
performance = load("PERFORMANCE.json")
report = load("GRAPH_LEAK_REPORT.json")

# ---------------------------------------------------------------- provenance ----
check("manifest.complete", manifest.get("complete"), True)
check("manifest.study_id", manifest.get("study_id"), "graphforget-mechanism-v2")
check("manifest.phase", manifest.get("phase"), "engineering")
check("manifest.protocol", manifest.get("protocol"), "graph_flow")
check("manifest.backend", manifest.get("backend"), "vllm")
check("manifest.challenges", list(manifest.get("challenges") or []), ["memory_reentry"])
check("manifest.git_dirty", manifest.get("git_dirty"), False)
# The manifest records an ABBREVIATED sha, so this is a prefix relation, not equality.
recorded_sha = str(manifest.get("git_sha") or "")
if required_commit and not (
    recorded_sha and required_commit.startswith(recorded_sha) and len(recorded_sha) >= 7
):
    failures.append(
        f"manifest.git_sha {recorded_sha!r} is not a prefix of the pinned {required_commit}"
    )
# `apply_sample_budget` correctly stamps a k-substituted run unreportable, so this is
# only a full-run condition. On the preflight, false is the right answer.
if args.full:
    check("manifest.profile_reportable", manifest.get("profile_reportable"), True)
elif manifest.get("profile_reportable") is not False:
    failures.append(
        "the k-substituted preflight did NOT stamp itself unreportable, so the wiring "
        "check would have been readable as evidence"
    )
check("manifest.detector_status", manifest.get("detector_status"), "diagnostic")
check("manifest.forget_policy_split", manifest.get("forget_policy_split"), "discovery")
check("manifest.cohorts_separated", manifest.get("cohorts_separated"), True)
check("manifest.uses_gold_answers", manifest.get("uses_gold_answers"), True)

# ------------------------------------------------------------------- volume ----
check("completed_trajectories", manifest.get("completed_trajectories"), args.trajectories)
plan = manifest.get("plan") or {}
check("plan.planned_trajectories", plan.get("planned_trajectories"), args.trajectories)
check("plan.planned_graph_generations", plan.get("planned_graph_generations"), args.graph)
check("plan.planned_probe_generations", plan.get("planned_probe_generations"), args.probe)
check("plan.n_items", plan.get("n_items"), args.items)
check("plan.n_samples", plan.get("n_samples"), args.samples)

ledger = manifest.get("evidence_ledger") or {}
for kind in ("generations", "traces"):
    block = ledger.get(kind) or {}
    if block.get("ok") is False or block.get("verified") is False:
        failures.append(f"evidence ledger for {kind} did not verify: {block}")

# ---------------------------------------------- arms present and statically pure ----
arm_entries = {str(a.get("arm")): a for a in (manifest.get("arms") or []) if a.get("arm")}
missing = EXPECTED_ARMS - set(arm_entries)
if missing:
    failures.append(f"arms missing from the manifest: {sorted(missing)}")

for arm, consumes, propagates in (
    (NO_FORWARD, True, False),
    (TAINT_FORWARD, True, True),
):
    entry = arm_entries.get(arm, {})
    check(f"{arm}.consumes_scope", entry.get("consumes_scope"), consumes)
    check(f"{arm}.propagates_scope", entry.get("propagates_scope"), propagates)
    check(f"{arm}.mode", entry.get("mode"), "multi_agent")
    check(f"{arm}.peer_content", entry.get("peer_content"), "same_concept")
    check(f"{arm}.defense", entry.get("defense"), arm.replace("multi_agent_", ""))

# ---------------------------------------------- THE DYNAMIC PATHWAY-LIVENESS GATE ----
#
# Static purity says the two arms are configured to differ only in forwarding. These
# checks say the pathway that difference acts on was actually walked: the tagged source
# reached a model in BOTH arms, neither arm withheld it, and only the treatment turned
# that into enforcement no node-local guard could have produced.
defenses = performance.get("defenses") or {}
nf = defenses.get(NO_FORWARD) or {}
tf = defenses.get(TAINT_FORWARD) or {}
qn = defenses.get(QUARANTINE) or {}

if not nf or not tf:
    failures.append("PERFORMANCE.json has no defence stats for one or both M5 arms")
else:
    for label, stats in ((NO_FORWARD, nf), (TAINT_FORWARD, tf)):
        check(f"{label}.semantic_detection", stats.get("semantic_detection"), False)
        check(f"{label}.consume_forget_ids", stats.get("consume_forget_ids"), True)
        check(f"{label}.inspect_query", stats.get("inspect_query"), False)

        # The source must NOT be quarantined in either arm — that is the vacuous design.
        withheld = stats.get("retrieval_withheld", -1)
        if withheld != 0:
            failures.append(
                f"{label}.retrieval_withheld={withheld}: the tagged source was withheld, "
                "so the arm never derived from it (regression to the vacuous design)"
            )

        # The tagged source must actually have reached a model.
        hits = stats.get("memory_borne_scope_hits", 0)
        if hits <= 0:
            failures.append(
                f"{label}.memory_borne_scope_hits={hits}: no tagged content ever reached "
                "a model in this arm, so nothing exists to forward"
            )

    check(f"{NO_FORWARD}.propagate_forget_ids", nf.get("propagate_forget_ids"), False)
    check(f"{TAINT_FORWARD}.propagate_forget_ids", tf.get("propagate_forget_ids"), True)

    # Matched exposure: both arms saw the same tagged source the same number of times.
    if nf.get("memory_borne_scope_hits") != tf.get("memory_borne_scope_hits"):
        failures.append(
            "tagged-source exposure is NOT matched across the M5 pair: "
            f"no_forward={nf.get('memory_borne_scope_hits')} "
            f"taint_forward={tf.get('memory_borne_scope_hits')}"
        )
    if nf.get("node_input_calls") != tf.get("node_input_calls"):
        failures.append(
            "the M5 pair did not execute the same number of node inputs: "
            f"no_forward={nf.get('node_input_calls')} taint_forward={tf.get('node_input_calls')}"
        )

    nf_inherited = (nf.get("causal_attribution") or {}).get("inherited_only_enforcements")
    tf_inherited = (tf.get("causal_attribution") or {}).get("inherited_only_enforcements")
    if nf_inherited is None or tf_inherited is None:
        failures.append("causal_attribution is absent from one or both M5 arms")
    else:
        if tf_inherited <= 0:
            failures.append(
                f"{TAINT_FORWARD} produced {tf_inherited} inherited-only enforcements: "
                "the forwarding pathway never reached a protected boundary"
            )
        if nf_inherited != 0:
            failures.append(
                f"{NO_FORWARD} produced {nf_inherited} inherited-only enforcements, but a "
                "non-forwarding arm has nothing downstream to enforce on"
            )
        notes.append(
            f"inherited_only enforcements: no_forward={nf_inherited} "
            f"taint_forward={tf_inherited}"
        )

# The source-quarantine positive control must behave the opposite way: it DOES withhold.
if qn:
    if qn.get("retrieval_withheld", 0) <= 0:
        failures.append(
            f"{QUARANTINE}.retrieval_withheld={qn.get('retrieval_withheld')}: the seeded "
            "policy tag never took effect anywhere, so the seeding itself is broken"
        )
    else:
        notes.append(f"quarantine control withheld {qn['retrieval_withheld']} retrievals")

# ------------------------------------------------------------ report-side gates ----
if report:
    gates = report.get("gates") or {}
    mechanism = report.get("mechanism") or {}
    check("report.primary_k", report.get("primary_k"), 32 if args.full else 1)
    check("report.mechanism.propagation_contrast", mechanism.get("propagation_contrast"), "M5")
    m5 = [
        c
        for c in (mechanism.get("contrasts") or [])
        if c.get("id") == "M5" and c.get("surface_role") == "primary"
    ]
    if not m5:
        failures.append("M5 is absent from every primary surface in the report")
    if args.full and not gates.get("mechanism_measurement_valid"):
        failures.append(
            "mechanism_measurement_valid is false on the full run: "
            f"{gates.get('mechanism_blockers')}"
        )
    if not args.full and gates.get("mechanism_measurement_valid"):
        notes.append(
            "mechanism_measurement_valid is TRUE on the preflight, which is unexpected "
            "(k was substituted); not a blocker"
        )
    notes.append(f"mechanism_blockers: {gates.get('mechanism_blockers')}")
    notes.append(f"semantic_report_valid: {gates.get('semantic_report_valid')} (expected false)")
    notes.append(f"publication_ready: {gates.get('publication_ready')} (expected false)")

# ----------------------------------------------------------------------- verdict ----
for note in notes:
    print(f"note: {note}")

if failures:
    print()
    print(f"AUDIT FAILED ({len(failures)} violations)")
    for line in failures:
        print(f"  - {line}")
    raise SystemExit(1)

print()
print(f"AUDIT PASSED for {run}")
