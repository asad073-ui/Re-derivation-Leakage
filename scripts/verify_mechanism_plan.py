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

# Original purpose: Hard contract check on a resolved PLAN.json.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PLAN = Path(sys.argv[1])
FULL = os.environ.get("RDL_PLAN_FULL", "1") == "1"

EXPECTED_ARMS = {
    "single_agent",
    "multi_agent_control",
    "multi_agent_leak",
    "multi_agent_dragon",
    "multi_agent_stateless",
    "multi_agent_graphforget_semantic_only",
    "multi_agent_graphforget_tag_source_quarantine",
    "multi_agent_graphforget_no_forward",
    "multi_agent_graphforget_taint_forward",
    "multi_agent_graphforget_taint_only",
    "multi_agent_graphforget",
}

# The split DETECTOR_V2_GATES.json was fitted on. Pinned here, not read from the
# artefact, so a swapped artefact cannot make its own mismatch pass.
POLICY_FINGERPRINT = "dc6ace798a7308e6ed1739659bed500d18cde596f62896e87ffb364a2607f64e"


def reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


plan = json.loads(PLAN.read_text(encoding="utf-8"), parse_constant=reject_constant)

failures: list[str] = []


def check(label: str, actual, expected) -> None:
    if actual != expected:
        failures.append(f"{label}: expected {expected!r}, got {actual!r}")


check("study_id", plan.get("study_id"), "graphforget-mechanism-v2")
check("phase", plan.get("phase"), "engineering")
check("profile", plan.get("profile"), "rtx3090_1b")
check("backend", plan.get("backend"), "vllm")
check("model", plan.get("model"), "rule_npo_1b")
check("profile_reportable", plan.get("profile_reportable"), True)
check("protocol", plan.get("protocol"), "graph_flow")
check("challenges", list(plan.get("challenges") or []), ["memory_reentry"])
check("topology", plan.get("topology"), "diamond5")

# Cohorts: evaluation is the 20 engineering questions, the forget policy is the
# 50-item discovery split the gate artefact was fitted on, and they share no items.
check("cohort_split", plan.get("cohort_split"), "engineering")
check("evaluation_cohort_split", plan.get("evaluation_cohort_split"), "engineering")
check("evaluation_n_concepts", plan.get("evaluation_n_concepts"), 20)
check("evaluation_is_retain", plan.get("evaluation_is_retain"), False)
check("forget_policy_split", plan.get("forget_policy_split"), "discovery")
check("forget_policy_n_items", plan.get("forget_policy_n_items"), 50)
check("forget_policy_n_concepts", plan.get("forget_policy_n_concepts"), 20)
check("forget_policy_fingerprint", plan.get("forget_policy_fingerprint"), POLICY_FINGERPRINT)
check("cohorts_separated", plan.get("cohorts_separated"), True)

# Model pin.
check("model_repo_id", plan.get("model_repo_id"), "OptimAI-Lab/TOFU-forget10_RULE-NPO")
check("model_revision", plan.get("model_revision"), "afe117e41a876f815bbd0f336d5036ced666ab06")
check(
    "tokenizer_revision",
    plan.get("tokenizer_revision"),
    "afe117e41a876f815bbd0f336d5036ced666ab06",
)
check("model_pinned", plan.get("model_pinned"), True)

# Detector: diagnostic v2 at the gated operating point.
check("detector_backend", plan.get("detector_backend"), "hashing64")
check("detector_status", plan.get("detector_status"), "diagnostic")

# Five logical agents, one physical handle.
check("logical_agents", plan.get("logical_agents"), 5)
check("shared_model_handles", plan.get("shared_model_handles"), 1)

# Arms: exactly the eleven, no more and no fewer.
arms = set(plan.get("arms") or [])
if arms != EXPECTED_ARMS:
    failures.append(
        f"arms mismatch: missing={sorted(EXPECTED_ARMS - arms)} extra={sorted(arms - EXPECTED_ARMS)}"
    )

if FULL:
    check("n_items", plan.get("n_items"), 20)
    check("evaluation_n_items", plan.get("evaluation_n_items"), 20)
    check("n_samples", plan.get("n_samples"), 32)
    check("primary_k", plan.get("primary_k"), 32)
    check("primary_k_reachable", plan.get("primary_k_reachable"), True)
    check("planned_trajectories", plan.get("planned_trajectories"), 7040)
    check("planned_graph_generations", plan.get("planned_graph_generations"), 32640)
    check("planned_probe_generations", plan.get("planned_probe_generations"), 14080)
    check("planned_generations_total", plan.get("planned_generations_total"), 46720)
    if 32 not in list(plan.get("k_values") or []):
        failures.append(f"k_values {plan.get('k_values')} cannot reach primary_k=32")
else:
    # 2x1 preflight: k IS substituted on purpose, so only the shape is checked.
    check("n_items", plan.get("n_items"), 2)
    check("n_samples", plan.get("n_samples"), 1)
    check("planned_trajectories", plan.get("planned_trajectories"), 22)
    check("planned_graph_generations", plan.get("planned_graph_generations"), 102)
    check("planned_probe_generations", plan.get("planned_probe_generations"), 44)

if failures:
    print("PLAN CONTRACT FAILED")
    for line in failures:
        print(f"  - {line}")
    raise SystemExit(1)

print(f"PLAN CONTRACT OK ({'full 20x32' if FULL else '2x1 preflight'})")
print(
    json.dumps(
        {
            k: plan.get(k)
            for k in (
                "study_id",
                "phase",
                "profile",
                "protocol",
                "n_items",
                "n_samples",
                "primary_k",
                "planned_trajectories",
                "planned_graph_generations",
                "planned_probe_generations",
                "forget_policy_split",
                "cohorts_separated",
            )
        },
        indent=2,
        sort_keys=True,
    )
)
