"""The whole pipeline on CPU: plan, run, resume, score, report, finalize."""

from __future__ import annotations

import json

import pytest

from rdl.eval.graph_leak import score_row
from rdl.eval.semantic import OfflineSemanticScorer
from rdl.graph.config import load_graph_config
from rdl.paths import repo_root
from rdl.studies.graph_leak.cohort import load_cohort, resolve_cohort
from rdl.studies.graph_leak.evidence import read_shards, verify_shards
from rdl.studies.graph_leak.runner import GraphRunner

LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"
COHORT = repo_root() / "data" / "cohorts" / "graph_unlearning_v1" / "cpu_stub.json"


@pytest.fixture
def runner_factory(tofu_items, graph_backend, tmp_path):
    cfg = load_graph_config(LAUNCH)
    cohort = load_cohort(COHORT).limited(4)
    items = resolve_cohort(cohort, tofu_items)

    def make(output, *, challenges=("natural",), resume=False):
        return GraphRunner(
            cfg=cfg,
            items=items,
            cohort=cohort,
            backend=graph_backend,
            output=output,
            challenges=tuple(challenges),
            tokenizer_revision="stub",
            resume=resume,
        )

    return make


def test_the_plan_matches_the_protocols_smoke_number(runner_factory, tmp_path):
    plan = runner_factory(tmp_path / "plan").plan()
    assert plan.planned_graph_generations == 168
    assert plan.planned_trajectories == 40
    assert plan.planned_probe_generations == 80


def test_a_full_run_produces_the_declared_artifacts(runner_factory, tmp_path):
    out = tmp_path / "run"
    manifest = runner_factory(out).run()
    assert manifest["complete"]
    assert manifest["completed_trajectories"] == 40
    for name in (
        "RUN_MANIFEST.json",
        "RESOLVED_CONFIG.json",
        "COHORT.json",
        "PLAN.json",
        "PERFORMANCE.json",
    ):
        assert (out / name).exists(), name
    assert list((out / "generations").glob("part-*.jsonl"))
    assert list((out / "traces").glob("graph-events-*.jsonl"))


def test_the_manifest_exists_before_the_first_model_call(runner_factory, tmp_path):
    """A killed run must leave enough information to resume safely."""
    out = tmp_path / "early"
    runner = runner_factory(out)

    calls = {"n": 0}
    original = runner.scheduler.run

    def counting(requests):
        if requests:
            calls["n"] += 1
            assert (out / "RUN_MANIFEST.json").exists()
        return original(requests)

    runner.scheduler.run = counting  # type: ignore[method-assign]
    runner.run()
    assert calls["n"] > 0


def test_the_manifest_records_shared_handles_and_declines_the_stronger_claim(
    runner_factory, tmp_path
):
    manifest = runner_factory(tmp_path / "run").run()
    assert len(set(manifest["shared_model_handles"].values())) == 1
    assert "NOT independently unlearned" in manifest["agent_note"]
    assert manifest["logical_agents"] == 5


def test_evidence_shards_verify(runner_factory, tmp_path):
    out = tmp_path / "run"
    manifest = runner_factory(out).run()
    assert verify_shards(out / "generations", manifest["evidence_shards"])["ok"]
    assert verify_shards(out / "traces", manifest["trace_shards"], prefix="graph-events")["ok"]


def test_a_modified_shard_is_caught(runner_factory, tmp_path):
    out = tmp_path / "run"
    manifest = runner_factory(out).run()
    shard = out / "generations" / manifest["evidence_shards"][0]["name"]
    shard.write_text(
        shard.read_text(encoding="utf-8").replace("natural", "natural "), encoding="utf-8"
    )
    assert not verify_shards(out / "generations", manifest["evidence_shards"])["ok"]


def test_resume_completes_an_interrupted_run_without_duplicates(runner_factory, tmp_path):
    out = tmp_path / "resumable"
    runner = runner_factory(out)

    class Stop(RuntimeError):
        pass

    original = runner.scheduler.run

    def stop_after_first_committed_shard(requests):
        # Deterministic interruption point: as soon as one shard is durable on disk,
        # which is exactly the state resume has to recover from.
        if list((out / "generations").glob("part-*.jsonl")):
            raise Stop()
        return original(requests)

    runner.scheduler.run = stop_after_first_committed_shard  # type: ignore[method-assign]
    with pytest.raises(Stop):
        runner.run()
    partial = len(list(read_shards(out / "generations")))
    assert 0 < partial < 40

    manifest = runner_factory(out, resume=True).run()
    rows = list(read_shards(out / "generations"))
    assert len(rows) == 40
    keys = [(r["item_id"], r["sample_id"], r["arm"], r["challenge"]) for r in rows]
    assert len(set(keys)) == 40
    assert manifest["complete"]


def test_rerunning_into_a_populated_directory_without_resume_is_refused(runner_factory, tmp_path):
    out = tmp_path / "run"
    runner_factory(out).run()
    with pytest.raises(FileExistsError, match="append-only"):
        runner_factory(out).run()


def test_generation_never_calls_a_scorer(runner_factory, tmp_path):
    manifest = runner_factory(tmp_path / "run").run()
    assert manifest["offline_scoring"] is True
    assert manifest["semantic_scorer"] is None


def test_scoring_is_a_separate_pass_over_the_shards(runner_factory, tmp_path):
    out = tmp_path / "run"
    runner_factory(out).run()
    engine = OfflineSemanticScorer()
    rows = [
        score_row(row, lambda r, c: engine.score(r, c).leaks, scorer_version=engine.version)
        for row in read_shards(out / "generations")
    ]
    assert len(rows) == 40
    assert {r["arm"] for r in rows} == {
        "single_agent",
        "multi_agent_control",
        "multi_agent_leak",
        "multi_agent_dragon",
        "multi_agent_graphforget",
    }


def test_the_controlled_mode_declares_gold_answer_use(runner_factory, tmp_path):
    natural = runner_factory(tmp_path / "nat").run()
    assert natural["uses_gold_answers"] is False
    controlled = runner_factory(tmp_path / "ctl", challenges=("direct",)).run()
    assert controlled["uses_gold_answers"] is True
    assert "gold answer" in controlled["uses_gold_answers_note"]


def test_the_resolved_config_carries_all_three_hashes(runner_factory, tmp_path):
    out = tmp_path / "run"
    runner_factory(out).run()
    resolved = json.loads((out / "RESOLVED_CONFIG.json").read_text(encoding="utf-8"))
    assert {"study_design_hash", "profile_hash", "resolved_run_hash"} <= set(resolved)
