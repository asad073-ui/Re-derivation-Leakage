"""The label report, run end to end over a BANK audit rather than the v4.1 training audit.

Two defects, both of which end with a plausible artifact:

**The report could not read the bank audit at all.** ``bank-audit`` writes
``BANK_AUDIT_KEY.json`` and ``BANK_AUDIT_JUDGE_{A,B}.jsonl``; the report opened
``LABEL_AUDIT_KEY.json`` and ``LABEL_AUDIT_JUDGE_{A,B}.jsonl``. Pointed at the bank audit's
directory it said the key was absent. Pointed — as the runbook then had to say — at the
v4.1 directory, it produced a complete, passing report about the 1,019-row *training*
audit, under an invocation whose purpose was to report on the fresh bank.

**The adjudicated file could not be bound to a bank.** It carried ``text_sha256`` and
nothing else, so the gate that reads it had no ``pair_sha256`` to match against a bank keyed
by ``sha256(text || question)`` and no ``bank_content_sha256`` to tell one bank revision
from another.

Both are proved here on files this test writes, because there is no committed bank audit
yet — the bank needs the GPU. The shapes are the ones the commands actually write.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import typer

from rdl.cli.detector_v4_2_llm_judge import _rows_sha256, _write_jsonl
from rdl.cli.detector_v4_2_report import detector_v4_2_label_report
from rdl.eval.detector_v4_2 import JUDGES, PROMPT_VERSION

BANK_SHA = "b" * 64
N_ROWS = 6


def _rows() -> list[dict]:
    return [
        {
            "audit_id": f"{i:016x}",
            "pair_sha256": f"{i:064x}",
            "text_sha256": f"{i + 900:064x}",
            "stratum": "protected_likely_answer" if i % 2 else "protected_clean",
            "partition": "heldout" if i % 2 else "development",
        }
        for i in range(N_ROWS)
    ]


def _build_audit(tmp_path: Path) -> Path:
    """Everything `rdl graph-detector-v4-2-bank-audit` writes, under its own filenames."""
    rows = _rows()
    blind = [
        {
            "audit_id": r["audit_id"],
            "protected_question": f"Where was author {i} born?",
            "candidate_text": f"candidate text {i}",
            "answer_attempt": "",
            "subject_only": "",
            "refusal": "",
            "question_type": "",
        }
        for i, r in enumerate(rows)
    ]
    reference = [
        {
            "audit_id": r["audit_id"],
            "protected_question": f"Where was author {i} born?",
            "candidate_text": f"candidate text {i}",
            "reference_answer": f"Rome {i}",
            "reference_content": "",
        }
        for i, r in enumerate(rows)
    ]
    bundle: dict[str, dict[str, str]] = {"blind": {}, "reference": {}}
    for role in JUDGES:
        blind_path = tmp_path / f"BANK_AUDIT_JUDGE_{role}.jsonl"
        reference_path = tmp_path / f"BANK_AUDIT_REFERENCE_PASS_{role}.jsonl"
        _write_jsonl(blind_path, blind)
        _write_jsonl(reference_path, reference)
        bundle["blind"][role] = str(blind_path)
        bundle["reference"][role] = str(reference_path)

    key_path = tmp_path / "BANK_AUDIT_KEY.json"
    key_path.write_text(
        json.dumps(
            {
                "schema": "graph-detector-v4-2-bank-audit-key-v2",
                "bank_id": "detector_v4_2_engineering_v1",
                "bank_content_sha256": BANK_SHA,
                "rows": {
                    r["audit_id"]: {
                        "stratum": r["stratum"],
                        "partition": r["partition"],
                        "population": "protected",
                        "nli_leaking": r["stratum"] == "protected_likely_answer",
                        "item_id": f"forget10-{i:04d}",
                        "text_sha256": r["text_sha256"],
                        "pair_sha256": r["pair_sha256"],
                        "bank_content_sha256": BANK_SHA,
                    }
                    for i, r in enumerate(rows)
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "BANK_AUDIT_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "graph-detector-v4-2-bank-audit-manifest-v2",
                "bundle_id": "detector_v4_2_engineering_v1-audit",
                "bank_content_sha256": BANK_SHA,
                "bank_path": str(tmp_path / "ENGINEERING_BANK.json"),
                "reportable": True,
                "uses_detector_score": False,
                "bundle": {"key": str(key_path), **bundle},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _overlays(judge_dir: Path, audit_dir: Path, *, agree: bool = True) -> None:
    """Four complete judge runs over that audit: A/blind, B/blind, A/reference, B/reference."""
    rows = _rows()
    for pass_name, fields in (
        (
            "blind",
            {
                "answer_attempt": "ANSWER",
                "subject_only": "no",
                "refusal": "no",
                "question_type": "slot",
            },
        ),
        ("reference", {"reference_content": "YES"}),
    ):
        for role in JUDGES:
            overlay = [
                {
                    "audit_id": r["audit_id"],
                    "judge": role,
                    "pass": pass_name,
                    "prompt_version": PROMPT_VERSION,
                    "source": "model",
                    "error": None,
                    **fields,
                }
                for r in rows
            ]
            if not agree and role == "B":
                overlay[0] = {**overlay[0], **{k: _flip(k, v) for k, v in fields.items()}}
            overlay_path = judge_dir / f"V4_2_JUDGE_{role}_{pass_name.upper()}.jsonl"
            _write_jsonl(overlay_path, overlay)
            input_path = audit_dir / (
                f"BANK_AUDIT_JUDGE_{role}.jsonl"
                if pass_name == "blind"
                else f"BANK_AUDIT_REFERENCE_PASS_{role}.jsonl"
            )
            run_path = judge_dir / f"V4_2_JUDGE_RUN_{role}_{pass_name.upper()}.json"
            run_path.write_text(
                json.dumps(
                    {
                        "judge": role,
                        "pass": pass_name,
                        "complete": True,
                        "reportable": True,
                        "run_id": None,
                        "prompt_version": PROMPT_VERSION,
                        "n_malformed_or_missing": 0,
                        "provider": JUDGES[role]["provider"],
                        "requested_model": JUDGES[role]["requested_model"],
                        "returned_models_seen": [JUDGES[role]["requested_model"]],
                        "returned_model": JUDGES[role]["requested_model"],
                        "input_file_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        "output_file_sha256": _rows_sha256(overlay),
                        "rubric_sha256": f"rubric-{pass_name}",
                        "n_rows_in_input": len(rows),
                        "n_rows_called": len(rows),
                        "n_provider_request_ids": len(rows),
                        "n_forced_by_protocol_rule": 0,
                    }
                ),
                encoding="utf-8",
            )


def _flip(field: str, value: str) -> str:
    return {"ANSWER": "NONE", "YES": "NO", "no": "yes", "slot": "open-ended"}.get(value, value)


def _run(tmp_path: Path, **overrides):
    kwargs = {
        "judge_dir": tmp_path / "judge",
        "audit_dir": tmp_path / "audit",
        "audit_manifest": tmp_path / "audit" / "BANK_AUDIT_MANIFEST.json",
        "output_dir": tmp_path / "out",
        "adjudication": None,
        "require_reference_pass": True,
    }
    kwargs.update(overrides)
    return detector_v4_2_label_report(**kwargs)


@pytest.fixture()
def prepared(tmp_path: Path) -> Path:
    audit_dir = tmp_path / "audit"
    judge_dir = tmp_path / "judge"
    audit_dir.mkdir()
    judge_dir.mkdir()
    _build_audit(audit_dir)
    _overlays(judge_dir, audit_dir)
    return tmp_path


def _legacy_layout(tmp_path: Path) -> Path:
    """The same rows under the v4.1 audit's frozen filenames, and its key — which has no
    ``pair_sha256``, because it predates the digest."""
    audit_dir = tmp_path / "audit"
    judge_dir = tmp_path / "judge"
    audit_dir.mkdir(exist_ok=True)
    judge_dir.mkdir(exist_ok=True)
    _build_audit(audit_dir)
    for role in JUDGES:
        for source, target in (
            (f"BANK_AUDIT_JUDGE_{role}.jsonl", f"LABEL_AUDIT_JUDGE_{role}.jsonl"),
            (
                f"BANK_AUDIT_REFERENCE_PASS_{role}.jsonl",
                f"LABEL_AUDIT_REFERENCE_PASS_{role}.jsonl",
            ),
        ):
            (audit_dir / target).write_bytes((audit_dir / source).read_bytes())
    key = json.loads((audit_dir / "BANK_AUDIT_KEY.json").read_text(encoding="utf-8"))
    for row in key["rows"].values():
        row.pop("pair_sha256", None)
        row.pop("bank_content_sha256", None)
    (audit_dir / "LABEL_AUDIT_KEY.json").write_text(json.dumps(key), encoding="utf-8")
    _overlays(judge_dir, audit_dir)
    # The manifests name the LEGACY input files, since that is what this run reads.
    for role in JUDGES:
        for pass_name, filename in (
            ("BLIND", f"LABEL_AUDIT_JUDGE_{role}.jsonl"),
            ("REFERENCE", f"LABEL_AUDIT_REFERENCE_PASS_{role}.jsonl"),
        ):
            run_path = judge_dir / f"V4_2_JUDGE_RUN_{role}_{pass_name}.json"
            manifest = json.loads(run_path.read_text(encoding="utf-8"))
            manifest["input_file_sha256"] = hashlib.sha256(
                (audit_dir / filename).read_bytes()
            ).hexdigest()
            run_path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_the_v4_1_training_audit_is_not_blocked_by_a_digest_it_predates(tmp_path):
    """The regression the pair-digest requirement could have caused, and does not.

    `LABEL_AUDIT_KEY.json` carries no `pair_sha256` — it was written before the bank was
    keyed by (text, question), and its labels are consumed by the trainer through
    `audit_id`, never bound to a bank row. Charging a provenance failure for the missing
    digest would block the labels that authorise training in order to protect a gate they
    are never read by.
    """
    _legacy_layout(tmp_path)
    with pytest.raises(typer.Exit):
        _run(tmp_path, audit_manifest=None)
    report = json.loads(
        (tmp_path / "out" / "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["binds_a_bank"] is False
    assert report["n_rows_without_pair_sha256"] == N_ROWS
    assert report["provenance"]["failures"] == [], report["provenance"]["failures"]


def test_a_bank_audit_whose_key_lacks_the_digest_IS_blocked(tmp_path):
    """The same absence, in the one place it makes the labels unusable."""
    audit_dir = tmp_path / "audit"
    judge_dir = tmp_path / "judge"
    audit_dir.mkdir()
    judge_dir.mkdir()
    manifest_path = _build_audit(audit_dir)
    _overlays(judge_dir, audit_dir)
    key_path = audit_dir / "BANK_AUDIT_KEY.json"
    key = json.loads(key_path.read_text(encoding="utf-8"))
    for row in key["rows"].values():
        row.pop("pair_sha256", None)
    key_path.write_text(json.dumps(key), encoding="utf-8")

    with pytest.raises(typer.Exit):
        _run(tmp_path, audit_manifest=manifest_path)
    report = json.loads(
        (tmp_path / "out" / "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert any("pair_sha256" in f for f in report["provenance"]["failures"])
    assert report["all_gates_passed"] is False


def test_the_report_reads_a_bank_audit_through_its_manifest(prepared):
    """Pointed at the bank audit, it reports on the bank audit."""
    with pytest.raises(typer.Exit):
        _run(prepared)
    report = json.loads(
        (prepared / "out" / "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["inputs"]["audit_bundle"]["bundle_id"] == "detector_v4_2_engineering_v1-audit"
    assert report["inputs"]["key"].endswith("BANK_AUDIT_KEY.json")
    assert report["bank_content_sha256"] == BANK_SHA
    # Four complete runs over six rows: nothing about the 1,019-row training audit.
    assert report["completeness"]["n_expected_rows"] == N_ROWS
    assert report["provenance"]["failures"] == []


def test_the_adjudicated_rows_carry_every_identifier_the_gate_binds_on(prepared):
    with pytest.raises(typer.Exit):
        _run(prepared)
    rows = [
        json.loads(line)
        for line in (prepared / "out" / "V4_2_ADJUDICATED.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(rows) == N_ROWS
    for row in rows:
        # text_sha256 alone was all the previous version wrote, and it cannot address a
        # bank keyed by (text, question).
        assert row["pair_sha256"]
        assert row["text_sha256"]
        assert row["bank_content_sha256"] == BANK_SHA
    assert {r["pair_sha256"] for r in rows} == {r["pair_sha256"] for r in _rows()}


def test_the_report_vouches_for_the_label_file_by_hash(prepared):
    """The gate refuses a label file the report does not name."""
    with pytest.raises(typer.Exit):
        _run(prepared)
    out = prepared / "out"
    report = json.loads(
        (out / "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json").read_text(encoding="utf-8")
    )
    labels = out / "V4_2_ADJUDICATED.jsonl"
    assert report["adjudicated_file"] == str(labels)
    assert report["adjudicated_sha256"] == hashlib.sha256(labels.read_bytes()).hexdigest()


def test_the_labels_produced_here_bind_to_a_bank_of_this_revision(prepared, tmp_path):
    """The whole point of carrying the identifiers: `final-gate` can now accept them."""
    from rdl.cli.detector_v4_2_gate_bridge import bind_labels_to_bank, load_label_map

    with pytest.raises(typer.Exit):
        _run(prepared)
    labels = load_label_map(prepared / "out" / "V4_2_ADJUDICATED.jsonl")
    bank_rows = [
        {
            "text": f"candidate text {i}",
            "request": f"Where was author {i} born?",
            "pair_sha256": r["pair_sha256"],
            "partition": r["partition"],
            "population": "protected",
        }
        for i, r in enumerate(_rows())
    ]
    by_pair = bind_labels_to_bank(
        labels, bank_rows, {"content_sha256": BANK_SHA}, labels_path=Path("labels.jsonl")
    )
    assert len(by_pair) == N_ROWS

    with pytest.raises(typer.BadParameter, match="bank moved after the audit"):
        bind_labels_to_bank(
            labels, bank_rows, {"content_sha256": "c" * 64}, labels_path=Path("labels.jsonl")
        )


def test_pointing_the_report_at_the_v4_1_directory_no_longer_silently_works(tmp_path):
    """It has to be told which audit it is reading; there is no default that guesses."""
    from rdl.cli.detector_v4_2_bundle import load_bundle

    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(typer.BadParameter, match="different audits over different rows"):
        load_bundle(audit_dir=empty, manifest=None)
