"""v4.2.2 — the bridge between a labelled bank and a gate, and the four ways it broke.

Every test here names a defect that produced a NUMBER rather than an error. That is the
common shape: the pipeline had a gap, the gap was filled by a default, and the default was
plausible enough that the artifact looked like a measurement.

1. The GPU gate could not read the v4.2 model-judge audit at all — it opened the v4.1
   human audit's filename and reported ``measured: false`` when the audit the whole v4.2
   pipeline exists to produce was sitting complete on disk.
2. The label report opened ``LABEL_AUDIT_*`` while the bank audit wrote ``BANK_AUDIT_*``,
   so it could not report on a bank — and pointed at the v4.1 directory it reported on the
   1,019-row training audit instead, under a header about the fresh bank.
3. The adjudicated label file carried ``text_sha256`` alone, which cannot address a bank
   keyed by ``sha256(text || question)``.
4. Gemini was priced at zero on a free tier it does not have.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from rdl.cli.detector_v4_2_bundle import load_bundle
from rdl.cli.detector_v4_gates import resolve_label_authority
from rdl.eval.detector_v4_2 import (
    PRICES_USD_PER_MTOK,
    PROVIDERS,
    RATE_LIMITS,
    REQUEST_MODE,
    list_price_of,
    price_of,
    rates_for,
)

REPO = Path(__file__).resolve().parents[2]


# =====================================================================================
# 1. The GPU gate reads whichever audit exists, and says which one it read
# =====================================================================================


def _adjudicated(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"audit_id": "0" * 16, "answer_attempt": "ANSWER"}) + "\n", encoding="utf-8"
    )
    return path


def _report(path: Path, *, passed: bool = True, kappa: float = 0.81) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "all_gates_passed": passed,
                "inter_judge": {"per_field": {"answer_attempt": {"cohens_kappa": kappa}}},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_the_gate_finds_the_model_judge_audit(tmp_path):
    """THE P0: a completed v4.2 audit left the gate reporting `measured: false`."""
    v4_1, v4_2 = tmp_path / "v4_1", tmp_path / "v4_2"
    v4_1.mkdir()
    _adjudicated(v4_2 / "V4_2_ADJUDICATED.jsonl")
    _report(v4_2 / "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json")

    authority = resolve_label_authority(v4_1, v4_2, "auto")
    assert authority["label_source"] == "model"
    assert authority["human_grounded"] is False
    # The flag the whole v4.2 protocol turns on: model labels authorise engineering only,
    # and no code path sets this True.
    assert authority["publication_label_valid"] is False
    assert authority["answer_attempt_kappa"] == 0.81


def test_the_human_audit_wins_when_both_exist(tmp_path):
    """The stronger authority, not the more recent file."""
    v4_1, v4_2 = tmp_path / "v4_1", tmp_path / "v4_2"
    _adjudicated(v4_1 / "LABEL_AUDIT_ADJUDICATED.jsonl")
    _report(v4_1 / "LABEL_ALIGNMENT_REPORT.json")
    _adjudicated(v4_2 / "V4_2_ADJUDICATED.jsonl")
    _report(v4_2 / "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json")

    authority = resolve_label_authority(v4_1, v4_2, "auto")
    assert authority["label_source"] == "human"
    assert authority["human_grounded"] is True
    assert authority["publication_label_valid"] is True


def test_no_audit_at_all_is_reported_as_such(tmp_path):
    authority = resolve_label_authority(tmp_path / "a", tmp_path / "b", "auto")
    assert authority["label_source"] is None
    assert len(authority["searched"]) == 2


def test_an_unknown_label_source_is_refused(tmp_path):
    with pytest.raises(typer.BadParameter):
        resolve_label_authority(tmp_path, tmp_path, "whichever-passes")


# =====================================================================================
# 2. An audit names its own files
# =====================================================================================


def _bank_audit_manifest(tmp_path: Path, **overrides) -> Path:
    payload = {
        "schema": "graph-detector-v4-2-bank-audit-manifest-v2",
        "bundle_id": "detector_v4_2_engineering_v1-audit",
        "bank_content_sha256": "a" * 64,
        "bank_path": str(tmp_path / "ENGINEERING_BANK.json"),
        "bundle": {
            "key": str(tmp_path / "BANK_AUDIT_KEY.json"),
            "blind": {
                "A": str(tmp_path / "BANK_AUDIT_JUDGE_A.jsonl"),
                "B": str(tmp_path / "BANK_AUDIT_JUDGE_B.jsonl"),
            },
            "reference": {
                "A": str(tmp_path / "BANK_AUDIT_REFERENCE_PASS_A.jsonl"),
                "B": str(tmp_path / "BANK_AUDIT_REFERENCE_PASS_B.jsonl"),
            },
        },
    }
    payload.update(overrides)
    path = tmp_path / "BANK_AUDIT_MANIFEST.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_a_bank_audit_bundle_names_the_bank_audit_files(tmp_path):
    """The reader used to open LABEL_AUDIT_*, which is a different audit's filenames."""
    manifest = _bank_audit_manifest(tmp_path)
    bundle = load_bundle(audit_dir=tmp_path, manifest=manifest)
    assert bundle.key_path.name == "BANK_AUDIT_KEY.json"
    assert bundle.input_for(judge="A", pass_name="blind").name == "BANK_AUDIT_JUDGE_A.jsonl"
    assert (
        bundle.input_for(judge="B", pass_name="reference").name
        == "BANK_AUDIT_REFERENCE_PASS_B.jsonl"
    )
    assert bundle.bank_content_sha256 == "a" * 64


def test_the_v4_1_layout_still_resolves_by_convention(tmp_path):
    """Those filenames are committed and frozen; they are one named legacy case."""
    (tmp_path / "LABEL_AUDIT_KEY.json").write_text(json.dumps({"rows": {}}), encoding="utf-8")
    bundle = load_bundle(audit_dir=tmp_path, manifest=None)
    assert bundle.bundle_id == "v4.1-label-audit"
    assert bundle.input_for(judge="A", pass_name="blind").name == "LABEL_AUDIT_JUDGE_A.jsonl"
    # A v4.1 audit is not drawn from a bank, and `None` is the honest answer.
    assert bundle.bank_content_sha256 is None


def test_a_directory_with_no_key_and_no_manifest_is_refused(tmp_path):
    with pytest.raises(typer.BadParameter, match="different audits over different rows"):
        load_bundle(audit_dir=tmp_path, manifest=None)


def test_a_manifest_that_names_no_files_is_refused(tmp_path):
    manifest = _bank_audit_manifest(tmp_path, bundle={})
    with pytest.raises(typer.BadParameter, match="does not name its own files"):
        load_bundle(audit_dir=tmp_path, manifest=manifest)


def test_a_manifest_of_an_unknown_schema_is_refused(tmp_path):
    manifest = _bank_audit_manifest(tmp_path, schema="something-else-v9")
    with pytest.raises(typer.BadParameter, match="refusing to guess"):
        load_bundle(audit_dir=tmp_path, manifest=manifest)


# =====================================================================================
# 3. Pricing: one judge is free and one is not
# =====================================================================================


def test_gemini_bills_at_the_standard_rate_because_that_is_what_the_runner_issues():
    """Two corrections in one assertion.

    v4.2.1 said Gemini was free. v4.2.2 said it had no free tier and priced it at
    $0.375/$1.875 — which are the **Batch/Flex** rates, half of Standard. The runner sends
    one synchronous chat-completions request per row and never a Batch job, so Standard is
    what the card shows.
    """
    assert REQUEST_MODE == "standard-synchronous"
    assert rates_for("gemini-3.7-flash") == {"input": 0.75, "output": 3.75}
    assert PRICES_USD_PER_MTOK["gemini-3.7-flash"] == {"input": 0.75, "output": 3.75}
    assert price_of("gemini-3.7-flash", 1_000_000, 1_000_000) == pytest.approx(4.50)


def test_the_batch_rate_is_recorded_and_is_not_what_this_pipeline_pays():
    """Kept, labelled, and not reachable by accident: it is the number v4.2.2 misreported."""
    assert rates_for("gemini-3.7-flash", "batch") == {"input": 0.375, "output": 1.875}
    assert list_price_of("gemini-3.7-flash", 1_000_000, 1_000_000, tier="batch") == pytest.approx(
        2.25
    )
    # Half of Standard, which is exactly how the confusion arose.
    assert (
        rates_for("gemini-3.7-flash", "batch")["input"] * 2
        == rates_for("gemini-3.7-flash")["input"]
    )


def test_geminis_free_tier_exists_and_is_refused_rather_than_denied():
    """ "There is no free tier" was as wrong as "it is free"; the protocol refuses it."""
    limits = RATE_LIMITS["gemini-3.7-flash"]
    assert limits["free_tier_exists"] is True
    assert limits["free_tier_used"] is False
    assert "improve its products" in limits["why_not_free"]
    # And the reason is data handling, so the cost is still billed in full.
    assert price_of("gemini-3.7-flash", 1_000_000, 0) == pytest.approx(0.75)


def test_the_provider_table_no_longer_calls_the_paid_judge_free():
    """The stale field: PROVIDERS said "free tier" while the rate table said paid."""
    google = PROVIDERS["google"]
    assert "paid" in google["billing"]
    assert google["free_tier_exists"] is True
    assert google["free_tier_permitted_by_protocol"] is False


def test_the_groq_judge_bills_nothing_and_records_what_it_would_cost():
    assert RATE_LIMITS["openai/gpt-oss-120b"]["free_tier_used"] is True
    assert price_of("openai/gpt-oss-120b", 1_000_000, 1_000_000) == 0.0
    # The corrected output rate: 0.60, not the 0.75 the repository estimated.
    assert PRICES_USD_PER_MTOK["openai/gpt-oss-120b"] == {"input": 0.15, "output": 0.60}
    assert list_price_of("openai/gpt-oss-120b", 1_000_000, 1_000_000) == pytest.approx(0.75)


def test_quota_data_is_marked_as_a_plan_claim_not_an_account_measurement():
    """Per-account limits differ from a docs page, and the artifact has to say which it is."""
    for model, limits in RATE_LIMITS.items():
        assert limits.get("quota_source"), model
        assert "account" in str(limits["quota_source"]).lower(), model


def test_the_groq_free_plan_limits_are_the_documented_ones():
    limits = RATE_LIMITS["openai/gpt-oss-120b"]
    assert limits["requests_per_minute"] == 30
    assert limits["requests_per_day"] == 1_000
    assert limits["tokens_per_day"] == 200_000
