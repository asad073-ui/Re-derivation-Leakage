"""`rdl graph-finalize` is the last thing a GPU session runs before it destroys the box.

GU-0024. `ok` used to be `generations.ok and traces.ok`, so a run that stopped a third of
the way through — or that had never been reported — printed a green object and exited 0.
An exit code that says "verified" over partial evidence is worse than no check, because
the instance is gone by the time anyone reads the numbers.
"""

from __future__ import annotations

import json

import pytest
import typer

from rdl.cli.finalize_graph import finalize_graph
from rdl.graph.config import load_graph_config
from rdl.paths import repo_root
from rdl.studies.graph_leak.cohort import load_cohort, resolve_cohort
from rdl.studies.graph_leak.runner import GraphRunner

LAUNCH = repo_root() / "configs" / "graph" / "launch.yaml"
COHORT = repo_root() / "data" / "cohorts" / "graph_unlearning_v1" / "cpu_stub.json"


@pytest.fixture
def finished_run(tofu_items, graph_backend, tmp_path):
    cfg = load_graph_config(LAUNCH)
    cohort = load_cohort(COHORT).limited(4)
    out = tmp_path / "run"
    GraphRunner(
        cfg=cfg,
        items=resolve_cohort(cohort, tofu_items),
        cohort=cohort,
        backend=graph_backend,
        output=out,
        tokenizer_revision="stub",
    ).run()
    return out


def _finalize(run, **kwargs):
    try:
        finalize_graph(run=run, raw_evidence_uri=kwargs.get("raw_evidence_uri"))
    except typer.Exit as exc:
        assert exc.exit_code == 1
    return json.loads((run / "FINALIZATION.json").read_text(encoding="utf-8"))


def test_a_complete_reported_run_with_intact_shards_passes(finished_run):
    (finished_run / "GRAPH_LEAK_REPORT.json").write_text("{}", encoding="utf-8")
    result = _finalize(finished_run, raw_evidence_uri="https://example.invalid/release")
    assert result["ok"] is True
    assert result["hashes_ok"] and result["complete"] and result["report_present"]
    assert result["blocking"] == []
    # A dirty checkout is a legitimate state for a development box and warns; the archive
    # URI was supplied, so that warning must not be present.
    assert not any("raw-evidence-uri" in warning for warning in result["warnings"])


def test_an_unreported_run_is_blocked(finished_run):
    result = _finalize(finished_run, raw_evidence_uri="https://example.invalid/release")
    assert result["ok"] is False
    assert result["hashes_ok"] is True, "the hashes are fine; the report is what is missing"
    assert any("GRAPH_LEAK_REPORT" in reason for reason in result["blocking"])


def test_an_incomplete_run_is_blocked_even_though_its_hashes_verify(finished_run):
    """The failure the old gate could not see: every shard hashes, a third of the run ran."""
    (finished_run / "GRAPH_LEAK_REPORT.json").write_text("{}", encoding="utf-8")
    manifest_path = finished_run / "RUN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["complete"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = _finalize(finished_run, raw_evidence_uri="https://example.invalid/release")
    assert result["ok"] is False
    assert result["hashes_ok"] is True
    assert any("not complete" in reason for reason in result["blocking"])


def test_an_edited_shard_is_blocked(finished_run):
    (finished_run / "GRAPH_LEAK_REPORT.json").write_text("{}", encoding="utf-8")
    manifest = json.loads((finished_run / "RUN_MANIFEST.json").read_text(encoding="utf-8"))
    shard = finished_run / "generations" / manifest["evidence_shards"][0]["name"]
    shard.write_text(shard.read_text(encoding="utf-8").replace("natural", "natura1"), "utf-8")

    result = _finalize(finished_run)
    assert result["ok"] is False
    assert result["hashes_ok"] is False
    # Both the manifest and the ledger catch it; either alone would be enough.
    assert not result["generations"]["ok"]
    assert not result["generations_ledger"]["ok"]


def test_a_missing_archive_uri_warns_rather_than_blocks(finished_run):
    """An unarchived diagnostic run is legitimate. A partial or unreported one is not."""
    (finished_run / "GRAPH_LEAK_REPORT.json").write_text("{}", encoding="utf-8")
    result = _finalize(finished_run)
    assert result["ok"] is True
    assert any("raw-evidence-uri" in warning for warning in result["warnings"])
