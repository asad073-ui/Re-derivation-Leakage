"""Immutable shards: atomic commit, hash verification, resume without duplicates."""

from __future__ import annotations

import json

import pytest

from rdl.studies.graph_leak.evidence import (
    ShardWriter,
    completed_keys,
    read_shards,
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
