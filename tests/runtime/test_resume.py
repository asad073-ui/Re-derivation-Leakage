"""Immutable shards: atomic commit, hash verification, resume without duplicates."""

from __future__ import annotations

import json

import pytest

from rdl.studies.graph_leak.evidence import (
    LEDGER_NAME,
    EvidenceTampered,
    ShardWriter,
    completed_keys,
    read_shards,
    verify_ledger,
    verify_shards,
)


def _row(item: str, sample: int, arm: str, challenge: str = "natural") -> dict:
    return {"item_id": item, "sample_id": sample, "arm": arm, "challenge": challenge, "x": 1}


def test_shards_are_committed_at_the_configured_size(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=2)
    for i in range(5):
        writer.append(_row("i0", i, "a"))
    assert len(list(tmp_path.glob("part-*.jsonl"))) == 2, "the partial shard is not yet committed"
    writer.close()
    assert len(list(tmp_path.glob("part-*.jsonl"))) == 3
    assert writer.n_rows == 5


def test_no_temporary_file_survives_a_clean_close(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=2)
    for i in range(3):
        writer.append(_row("i0", i, "a"))
    writer.close()
    assert not list(tmp_path.glob("*.tmp"))


def test_an_interrupted_temporary_shard_is_discarded_on_resume(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=2)
    writer.append(_row("i0", 0, "a"))
    writer.append(_row("i0", 1, "a"))
    writer.close()
    # Simulate a process killed mid-commit.
    (tmp_path / "part-00001.jsonl.tmp").write_text('{"item_id": "torn"}\n', encoding="utf-8")

    resumed = ShardWriter(tmp_path, shard_size=2)
    assert not list(tmp_path.glob("*.tmp"))
    assert resumed.completed() == {("i0", 0, "a", "natural"), ("i0", 1, "a", "natural")}


def test_resume_does_not_duplicate_completed_trajectories(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=2)
    writer.append(_row("i0", 0, "a"))
    writer.append(_row("i0", 1, "a"))
    writer.close()

    resumed = ShardWriter(tmp_path, shard_size=2)
    done = resumed.completed()
    for sample in range(4):
        key = ("i0", sample, "a", "natural")
        if key in done:
            continue
        resumed.append(_row("i0", sample, "a"))
    resumed.close()
    rows = list(read_shards(tmp_path))
    assert len(rows) == 4
    assert len({(r["item_id"], r["sample_id"], r["arm"]) for r in rows}) == 4


def test_appending_a_completed_key_twice_is_an_error(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=8)
    writer.append(_row("i0", 0, "a"))
    with pytest.raises(ValueError, match="duplicate trajectory key"):
        writer.append(_row("i0", 0, "a"))


def test_the_same_sample_under_different_arms_is_not_a_duplicate(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=8)
    writer.append(_row("i0", 0, "multi_agent_leak"))
    writer.append(_row("i0", 0, "multi_agent_dragon"))
    writer.append(_row("i0", 0, "multi_agent_leak", challenge="split_clues"))
    assert len(writer.close()) == 1
    assert len(completed_keys(tmp_path)) == 3


def test_a_modified_shard_fails_verification(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=8)
    writer.append(_row("i0", 0, "a"))
    manifest = [s.to_dict() for s in writer.close()]
    assert verify_shards(tmp_path, manifest)["ok"]

    path = tmp_path / manifest[0]["name"]
    path.write_text(json.dumps(_row("i0", 0, "a")) + "\n", encoding="utf-8")
    result = verify_shards(tmp_path, manifest)
    assert not result["ok"]
    assert "sha256 mismatch" in result["problems"][0]


def test_a_missing_shard_fails_verification(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=8)
    writer.append(_row("i0", 0, "a"))
    manifest = [s.to_dict() for s in writer.close()]
    (tmp_path / manifest[0]["name"]).unlink()
    assert not verify_shards(tmp_path, manifest)["ok"]


def test_an_undeclared_shard_fails_verification(tmp_path):
    writer = ShardWriter(tmp_path, shard_size=8)
    writer.append(_row("i0", 0, "a"))
    manifest = [s.to_dict() for s in writer.close()]
    (tmp_path / "part-09999.jsonl").write_text('{"item_id":"x"}\n', encoding="utf-8")
    result = verify_shards(tmp_path, manifest)
    assert not result["ok"]
    assert "not in the manifest" in result["problems"][0]


# =====================================================================================
# GU-0024 — the shard ledger
#
# Shard hashes only ever reached RUN_MANIFEST.json when a run FINISHED, so the shards of
# an interrupted run were covered by nothing at all. Resume read them back, trusted them,
# and skipped their trajectory keys; the final manifest then hashed whatever was there and
# declared it verified. These tests are about the interrupted case specifically.
# =====================================================================================


def _commit(tmp_path, n: int, *, shard_size: int = 2) -> ShardWriter:
    writer = ShardWriter(tmp_path, shard_size=shard_size)
    for i in range(n):
        writer.append(_row("i0", i, "a"))
    writer.close()
    return writer


def test_a_ledger_is_written_for_every_committed_shard(tmp_path):
    _commit(tmp_path, 5)
    ledger = json.loads((tmp_path / LEDGER_NAME).read_text(encoding="utf-8"))
    assert ledger["n_shards"] == 3
    assert ledger["n_rows"] == 5
    assert verify_ledger(tmp_path)["ok"]


def test_a_partial_run_can_be_resumed_when_its_shards_are_intact(tmp_path):
    _commit(tmp_path, 4)  # never "completed": no manifest, only committed shards
    resumed = ShardWriter(tmp_path, shard_size=2)
    assert resumed.ledger_status == "verified"
    assert len(resumed.completed()) == 4


def test_an_edited_shard_of_an_INTERRUPTED_run_is_refused_on_resume(tmp_path):
    """THE hole. There was no manifest to check these against, so they were adopted."""
    _commit(tmp_path, 4)
    shard = tmp_path / "part-00000.jsonl"
    shard.write_text(shard.read_text(encoding="utf-8").replace('"x": 1', '"x": 999'), "utf-8")
    with pytest.raises(EvidenceTampered, match="changed after it was committed"):
        ShardWriter(tmp_path, shard_size=2)
    assert not verify_ledger(tmp_path)["ok"]


def test_a_shard_smuggled_in_beside_the_real_ones_is_refused(tmp_path):
    _commit(tmp_path, 4)
    (tmp_path / "part-09999.jsonl").write_text(json.dumps(_row("i9", 0, "a")) + "\n", "utf-8")
    with pytest.raises(EvidenceTampered, match="never declared"):
        ShardWriter(tmp_path, shard_size=2)


def test_deleting_a_committed_shard_is_refused(tmp_path):
    _commit(tmp_path, 6)  # three shards
    (tmp_path / "part-00000.jsonl").unlink()
    with pytest.raises(EvidenceTampered, match="missing from disk"):
        ShardWriter(tmp_path, shard_size=2)


def test_a_commit_interrupted_between_the_ledger_and_the_rename_is_recoverable(tmp_path):
    """The one legitimate divergence: the LAST entry names a shard the rename never made.

    The ledger is written first on purpose, so this window is the only one that exists and
    it is unambiguous — those rows were never visible to anyone.
    """
    _commit(tmp_path, 4)
    ledger = json.loads((tmp_path / LEDGER_NAME).read_text(encoding="utf-8"))
    ledger["shards"].append({"name": "part-00002.jsonl", "sha256": "0" * 64, "n_rows": 2})
    (tmp_path / LEDGER_NAME).write_text(json.dumps(ledger), encoding="utf-8")

    resumed = ShardWriter(tmp_path, shard_size=2)
    assert resumed.ledger_status == "verified"
    assert len(resumed.completed()) == 4
    # The phantom entry is gone once the directory is rewritten.
    rewritten = json.loads((tmp_path / LEDGER_NAME).read_text(encoding="utf-8"))
    assert [s["name"] for s in rewritten["shards"]] == ["part-00000.jsonl", "part-00001.jsonl"]


def test_shards_written_before_the_ledger_existed_are_adopted_and_labelled(tmp_path):
    """A run from before GU-0024. Adopt it — but never call it verified."""
    _commit(tmp_path, 4)
    (tmp_path / LEDGER_NAME).unlink()
    resumed = ShardWriter(tmp_path, shard_size=2)
    assert resumed.ledger_status == "adopted_unverified"
    assert len(resumed.completed()) == 4
    assert verify_ledger(tmp_path)["ok"], "the ledger is rebuilt from what was adopted"
