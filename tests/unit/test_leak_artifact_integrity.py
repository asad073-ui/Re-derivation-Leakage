from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdl.cli.make_leak_report import make_leak_report
from rdl.cli.run_leak import _repair_partial_jsonl


def test_resume_repairs_only_a_torn_final_json_line(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_bytes(b'{"item_id":"i"}\n{"item')
    assert _repair_partial_jsonl(path) == [{"item_id": "i"}]
    assert path.read_text() == '{"item_id":"i"}\n'


def test_report_refuses_records_without_matching_manifest_hash(tmp_path: Path) -> None:
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"item_id": "i"}) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"records_sha256": "0" * 64, "cohort": []}))
    with pytest.raises(ValueError, match="SHA-256"):
        make_leak_report(records=records, manifest=manifest, bootstrap_reps=1)
