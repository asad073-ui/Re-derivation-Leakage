"""A resume must be the same experiment, or it must refuse.

The blocker this closes: `GraphRunner.run` overwrote RUN_MANIFEST.json before checking
anything, so a resume under a different checkpoint, topology, sample budget, protocol or
cohort replaced the record of what had produced the existing shards and then silently
skipped their trajectory keys. The completed keys look identical whichever checkpoint
produced them, so nothing downstream could ever notice.
"""

from __future__ import annotations

import json

import pytest

from rdl.graph.config import load_graph_config
from rdl.paths import repo_root
from rdl.studies.graph_leak.cohort import load_cohort, resolve_cohort
from rdl.studies.graph_leak.evidence import read_shards
from rdl.studies.graph_leak.runner import GraphRunner

LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"
COHORT = repo_root() / "data" / "cohorts" / "graph_unlearning_v1" / "cpu_stub.json"


@pytest.fixture
def make_runner(tofu_items, graph_backend):
    cohort = load_cohort(COHORT).limited(4)
    items = resolve_cohort(cohort, tofu_items)

    def build(output, **overrides):
        cfg = overrides.pop("cfg", None) or load_graph_config(LAUNCH)
        return GraphRunner(
            cfg=cfg,
            items=items,
            cohort=cohort,
            backend=graph_backend,
            output=output,
            tokenizer_revision="stub",
            **overrides,
        )

    return build


class Stop(RuntimeError):
    pass


def _interrupt_after_first_shard(runner, out):
    original = runner.scheduler.run

    def stopper(requests):
        if list((out / "generations").glob("part-*.jsonl")):
            raise Stop()
        return original(requests)

    runner.scheduler.run = stopper  # type: ignore[method-assign]
    with pytest.raises(Stop):
        runner.run()


def test_a_compatible_resume_completes(make_runner, tmp_path):
    out = tmp_path / "run"
    _interrupt_after_first_shard(make_runner(out), out)
    partial = len(list(read_shards(out / "generations")))
    assert 0 < partial < 40

    make_runner(out, resume=True).run()
    rows = list(read_shards(out / "generations"))
    assert len(rows) == 40
    assert len({(r["item_id"], r["sample_id"], r["arm"], r["challenge"]) for r in rows}) == 40


def test_the_manifest_survives_an_incompatible_resume_attempt(make_runner, tmp_path):
    """The original record must not be replaced by the one that was refused."""
    out = tmp_path / "run"
    _interrupt_after_first_shard(make_runner(out), out)
    before = json.loads((out / "RUN_MANIFEST.json").read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="not the same experiment"):
        make_runner(out, resume=True, protocol="graph_flow").run()

    after = json.loads((out / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    assert after["protocol"] == before["protocol"] == "end_to_end_safety"
    assert after["resolved_run_hash"] == before["resolved_run_hash"]


def test_a_different_protocol_is_refused(make_runner, tmp_path):
    out = tmp_path / "run"
    _interrupt_after_first_shard(make_runner(out), out)
    with pytest.raises(ValueError, match="protocol"):
        make_runner(out, resume=True, protocol="graph_flow").run()


def test_a_different_challenge_is_refused(make_runner, tmp_path):
    out = tmp_path / "run"
    _interrupt_after_first_shard(make_runner(out), out)
    with pytest.raises(ValueError, match="challenges"):
        make_runner(out, resume=True, challenges=("direct",)).run()


def test_a_different_topology_is_refused(make_runner, tmp_path):
    out = tmp_path / "run"
    _interrupt_after_first_shard(make_runner(out), out)
    other = load_graph_config(LAUNCH, topology="chain5")
    with pytest.raises(ValueError, match="not the same experiment"):
        make_runner(out, resume=True, cfg=other).run()


def test_a_different_sample_budget_is_refused(make_runner, tmp_path):
    from rdl.cli.graph_common import apply_sample_budget

    out = tmp_path / "run"
    _interrupt_after_first_shard(make_runner(out), out)
    reduced = apply_sample_budget(load_graph_config(LAUNCH), 1)
    with pytest.raises(ValueError, match="not the same experiment"):
        make_runner(out, resume=True, cfg=reduced).run()


def test_a_different_cohort_is_refused(make_runner, tofu_items, graph_backend, tmp_path):
    out = tmp_path / "run"
    _interrupt_after_first_shard(make_runner(out), out)

    smaller = load_cohort(COHORT).limited(2)
    with pytest.raises(ValueError, match="cohort_fingerprint"):
        GraphRunner(
            cfg=load_graph_config(LAUNCH),
            items=resolve_cohort(smaller, tofu_items),
            cohort=smaller,
            backend=graph_backend,
            output=out,
            tokenizer_revision="stub",
            resume=True,
        ).run()


def test_a_populated_directory_without_resume_is_refused(make_runner, tmp_path):
    out = tmp_path / "run"
    _interrupt_after_first_shard(make_runner(out), out)
    with pytest.raises(FileExistsError, match="append-only"):
        make_runner(out).run()


def test_the_resume_is_recorded_in_the_final_manifest(make_runner, tmp_path):
    out = tmp_path / "run"
    _interrupt_after_first_shard(make_runner(out), out)
    manifest = make_runner(out, resume=True).run()
    assert manifest["resumed_from"] is not None
    assert manifest["resumed_from"]["trajectories"] > 0
    assert manifest["complete"]


def test_every_immutable_field_is_actually_compared(make_runner, tmp_path):
    """The list is the contract; a field dropped from it is a silent hole."""
    out = tmp_path / "plan-only"
    runner = make_runner(out)
    manifest = runner._manifest(runner.plan())
    for field in GraphRunner.IMMUTABLE_ON_RESUME:
        assert field in manifest, f"{field} is compared on resume but never recorded"
