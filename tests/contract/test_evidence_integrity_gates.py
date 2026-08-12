"""Strict JSON, per-purpose generation counters, and the 2x1 preflight. GU-0028/GU-0029.

Three failures that each made a piece of evidence unreadable or unreadable-as-claimed:

1. Reports carried literal ``NaN``. RFC 8259 has no such token, so `jq`, `python -m
   json.tool` and every non-Python reader reject the file — including the runbook step
   whose whole job is to validate the evidence before it is archived.
2. ``actual_graph_generations`` was the scheduler's TOTAL dispatch count, probes
   included, so it could not be checked against ``planned_graph_generations`` and was
   not. A 2x1 preflight planning 42 graph calls and 20 probes reported "27".
3. The runbook's preflight was ``--limit 1``, which cannot produce a cross-concept
   control and dies inside the runner with a message about C3S. The operator found that
   out on a rented GPU.
"""

from __future__ import annotations

import json

import pytest
import typer

from rdl.cli.graph_common import (
    MIN_ITEMS_FOR_CROSS_CONCEPT_CONTROL,
    assert_control_arm_has_enough_items,
)
from rdl.eval.graph_statistics import finite_or_none, relative_reduction
from rdl.graph.config import load_graph_config
from rdl.paths import repo_root
from rdl.runtime.batch_scheduler import PURPOSES, BatchScheduler
from rdl.studies.graph_leak.cohort import load_cohort, resolve_cohort
from rdl.studies.graph_leak.evidence import atomic_json
from rdl.studies.graph_leak.runner import GraphRunner

LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"
COHORT = repo_root() / "data" / "cohorts" / "graph_unlearning_v1" / "cpu_stub.json"


# ------------------------------------------------------------------ strict JSON --


def test_every_committed_graph_json_is_strict_json():
    """The check the runbook runs on the GPU box, run here so it cannot fail there."""
    paths = sorted((repo_root() / "runs" / "graph").glob("**/*.json"))
    assert paths, "no committed graph evidence to check"
    for path in paths:
        json.loads(path.read_text(encoding="utf-8"))


def test_every_committed_cohort_and_calibration_file_is_strict_json():
    for path in sorted((repo_root() / "data" / "cohorts").glob("**/*.json")):
        json.loads(path.read_text(encoding="utf-8"))


def test_no_committed_report_holds_a_bare_nan_or_infinity_token():
    """Checked as JSON, not as text: `nan_repair.reason` legitimately mentions NaN.

    `parse_constant` fires only for the bare `NaN` / `Infinity` / `-Infinity` tokens —
    exactly the non-standard literals Python emits and nothing else parses — so a note
    explaining the repair does not trip it while a relapse would.
    """

    def refuse(token: str) -> float:
        raise AssertionError(f"non-standard JSON token {token!r}")

    for path in sorted((repo_root() / "runs" / "graph").glob("**/*.json")):
        json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def test_atomic_json_refuses_a_non_finite_float(tmp_path):
    """Better to fail at the writer than to ship a file nobody can parse."""
    with pytest.raises(ValueError, match="non-finite float"):
        atomic_json(tmp_path / "bad.json", {"relative_reduction": float("nan")})
    assert not (tmp_path / "bad.json").exists()


def test_undefined_statistics_serialise_as_null(tmp_path):
    payload = {"relative_reduction": relative_reduction(0.0, 0.0)}
    atomic_json(tmp_path / "ok.json", payload)
    assert json.loads((tmp_path / "ok.json").read_text(encoding="utf-8")) == {
        "relative_reduction": None
    }


def test_finite_or_none_maps_every_unrepresentable_value_to_none():
    assert finite_or_none(float("nan")) is None
    assert finite_or_none(float("inf")) is None
    assert finite_or_none(float("-inf")) is None
    assert finite_or_none(None) is None
    assert finite_or_none(0.25) == 0.25


# ------------------------------------------------------- per-purpose counters --


def test_the_scheduler_counts_graph_and_probe_apart(graph_backend):
    from rdl.runtime.backend import GenRequest

    scheduler = BatchScheduler(graph_backend, max_batch_size=4)
    graph = [GenRequest(request_id=f"g{i}", prompt=f"question {i}") for i in range(3)]
    probe = [GenRequest(request_id=f"p{i}", prompt=f"probe {i}") for i in range(2)]
    scheduler.run(graph, purpose="graph")
    scheduler.run(probe, purpose="probe")

    stats = scheduler.stats()
    assert stats["graph"] == {"requested": 3, "dispatched": 3, "cache_hits": 0}
    assert stats["probe"] == {"requested": 2, "dispatched": 2, "cache_hits": 0}
    assert stats["total"]["requested"] == 5


def test_cache_hits_are_attributed_to_the_purpose_that_asked(graph_backend):
    from rdl.runtime.backend import GenRequest

    scheduler = BatchScheduler(graph_backend, max_batch_size=4)
    scheduler.run([GenRequest(request_id="a", prompt="same")], purpose="graph")
    scheduler.run([GenRequest(request_id="b", prompt="same")], purpose="probe")

    stats = scheduler.stats()
    assert stats["graph"]["dispatched"] == 1
    assert stats["probe"]["dispatched"] == 0
    assert stats["probe"]["cache_hits"] == 1
    # requested == dispatched + cache_hits, per purpose. That identity is what makes the
    # counters checkable from outside the process.
    for purpose in PURPOSES:
        counts = stats[purpose]
        assert counts["requested"] == counts["dispatched"] + counts["cache_hits"]


def test_an_unknown_purpose_is_a_hard_error(graph_backend):
    from rdl.runtime.backend import GenRequest

    scheduler = BatchScheduler(graph_backend)
    with pytest.raises(ValueError, match="unknown generation purpose"):
        scheduler.run([GenRequest(request_id="x", prompt="y")], purpose="whatever")


def test_the_manifest_reports_generations_per_purpose_not_one_misleading_total(
    tofu_items, graph_backend, tmp_path
):
    cfg = load_graph_config(LAUNCH)
    cohort = load_cohort(COHORT).limited(4)
    manifest = GraphRunner(
        cfg=cfg,
        items=resolve_cohort(cohort, tofu_items),
        cohort=cohort,
        backend=graph_backend,
        output=tmp_path / "run",
        tokenizer_revision="stub",
    ).run()

    # The name that could not be checked against anything is gone, not renamed.
    assert "actual_graph_generations" not in manifest
    actual = manifest["actual_generations"]
    assert set(actual) == set(PURPOSES)

    performance = json.loads((tmp_path / "run" / "PERFORMANCE.json").read_text(encoding="utf-8"))
    scheduler = performance["scheduler"]
    plan = manifest["plan"]

    # Each purpose is now checkable against ITS OWN plan, which is the whole point: one
    # conflated total could not be compared with either number.
    #
    # The graph budget is an UPPER BOUND — a guarded node that abstains produces no
    # request at all — so the check is "within budget and non-zero". The probe budget is
    # exact: every trajectory contributes two prompts whatever the defence decided.
    assert 0 < scheduler["graph"]["requested"] <= plan["planned_graph_generations"]
    assert scheduler["probe"]["requested"] == plan["planned_probe_generations"]
    for purpose in PURPOSES:
        counts = scheduler[purpose]
        assert counts["requested"] == counts["dispatched"] + counts["cache_hits"]
        assert actual[purpose]["dispatched"] == counts["dispatched"]


# ------------------------------------------------------------------- preflight --


def test_the_minimum_preflight_is_two_items():
    assert MIN_ITEMS_FOR_CROSS_CONCEPT_CONTROL == 2


def test_a_one_item_run_is_refused_before_anything_loads(tofu_items):
    """`--limit 1` used to die deep in the runner with a message about C3S."""
    cfg = load_graph_config(LAUNCH)
    with pytest.raises(typer.BadParameter, match=r"2 items x 1 sample"):
        assert_control_arm_has_enough_items(cfg, tofu_items[:1])


def test_two_items_are_enough(tofu_items):
    cfg = load_graph_config(LAUNCH)
    assert_control_arm_has_enough_items(cfg, tofu_items[:2])


def test_the_two_item_preflight_actually_runs(tofu_items, graph_backend, tmp_path):
    """The number in the runbook has to be a number that works."""
    cfg = load_graph_config(LAUNCH)
    from rdl.cli.graph_common import apply_sample_budget

    cfg = apply_sample_budget(cfg, 1)
    cohort = load_cohort(COHORT).limited(2)
    manifest = GraphRunner(
        cfg=cfg,
        items=resolve_cohort(cohort, tofu_items),
        cohort=cohort,
        backend=graph_backend,
        output=tmp_path / "preflight",
        tokenizer_revision="stub",
    ).run()
    # 2 items x 1 sample x 6 arms.
    assert manifest["completed_trajectories"] == 12
    assert manifest["complete"] is True


def test_no_documentation_still_tells_the_operator_to_use_one_item():
    """The runbook said `--limit 1` for months. It must not say it again."""
    for path in sorted((repo_root() / "docs").rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        assert "--limit 1\n" not in text, path
        assert "--limit 1 " not in text, path
