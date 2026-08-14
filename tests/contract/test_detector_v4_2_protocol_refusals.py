"""Detector v4.2 — the six things the pipeline must REFUSE, each proved on real artifacts.

Every test here names a defect that shipped in the first v4.2 commit and would have been
discovered on rented GPU time or after a day of spent API quota. They are contract tests
rather than unit tests because each one is about the pipeline's behaviour at a boundary —
what it accepts from disk, what it writes, what it will not do — rather than about a
function's return value.

Where a real artifact exists it is used. ``tests/fixtures`` has no graph run manifest and a
synthetic one would only prove that the parser reads the synthetic one: the whole defect
being fixed is that the previous parser read a schema that no real run writes. So the run
tests load ``runs/graph/**/RUN_MANIFEST.json`` from the repository and skip only if the
directory has been pruned.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from rdl.cli.detector_v4_2_bank_audit import assign_stratum, draw_sample
from rdl.cli.detector_v4_2_banks import (
    GENERATION_BUDGET,
    RUN_MANIFEST_FILENAME,
    check_group,
    run_meta,
    validate_budget,
)
from rdl.cli.detector_v4_2_llm_judge import load_resumable
from rdl.cli.detector_v4_2_report import verify_run
from rdl.eval.detector_v4_2 import (
    ENGINEERING_BANK_SEEDS,
    ENGINEERING_RETAIN_SEEDS,
    JUDGES,
    PROMPT_VERSION,
    PROVIDERS,
    build_prompt,
    disagreements,
    judge_families_are_independent,
    prompt_sha256,
)

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "runs" / "graph"


def _real_manifests() -> list[Path]:
    return sorted(RUNS.glob(f"*/{RUN_MANIFEST_FILENAME}"))


def _real_run(**must_have) -> Path:
    """A committed graph run whose manifest matches ``must_have``, or skip."""
    for path in _real_manifests():
        meta = run_meta(path.parent)
        if all(meta.get(k) == v for k, v in must_have.items()):
            return path.parent
    pytest.skip(f"no committed graph run matches {must_have}")


# =====================================================================================
# 1. n_samples < primary_k is rejected
# =====================================================================================


def test_the_frozen_budget_validates_and_the_old_one_does_not():
    """The 60 x 8, k=32 freeze was arithmetically impossible and passed silently."""
    assert validate_budget(GENERATION_BUDGET) == []

    impossible = json.loads(json.dumps(GENERATION_BUDGET))
    impossible["groups"]["natural"]["n_samples"] = 8
    failures = validate_budget(impossible)
    assert any("n_samples=8" in f and "primary_k=32" in f for f in failures), failures


def test_a_single_undifferentiated_run_set_is_rejected():
    """The v4.2.0 budget had one flat `n_runs: 4` covering natural AND retain runs."""
    flat = {"n_items": 60, "samples_per_item": 8, "k": 32, "n_runs": 4, "arms": ["x"]}
    failures = validate_budget(flat)
    assert failures
    assert any("groups" in f for f in failures), failures


def test_a_real_run_that_cannot_measure_its_own_k_is_rejected(tmp_path):
    """Read from a REAL manifest, then break only the sampling block."""
    source = _real_run(cohort_split="discovery", is_retain=False)
    payload = json.loads((source / RUN_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    payload["sampling"]["n_samples"] = 8
    broken = tmp_path / "run"
    broken.mkdir()
    (broken / RUN_MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")

    group = dict(GENERATION_BUDGET["groups"]["natural"])
    meta = run_meta(broken)
    meta["base_seed"] = group["seeds"][0]
    failures = check_group([meta], "natural", {**group, "n_runs": 1, "seeds": [group["seeds"][0]]})
    assert any("success-at-k needs at least k draws" in f for f in failures), failures


# =====================================================================================
# 2. The run manifest is read as it is actually written
# =====================================================================================


def test_run_meta_reads_the_schema_real_runs_write():
    """The defect: the parser opened `run_manifest.json` and read top-level seed/arm.

    Because every lookup returned None and every check skipped None, the old verifier
    accepted anything and reported PASS. This asserts the fields come back populated from
    a manifest nobody wrote for this test.
    """
    manifests = _real_manifests()
    if not manifests:
        pytest.skip("no committed graph runs")

    checked = 0
    for path in manifests:
        meta = run_meta(path.parent)
        assert meta["base_seed"] is not None, f"{path}: sampling.base_seed not read"
        assert meta["primary_k"] is not None, f"{path}: sampling.primary_k not read"
        assert meta["n_samples"] is not None, f"{path}: sampling.n_samples not read"
        assert meta["challenges"], f"{path}: challenges not read"
        assert meta["arms"], f"{path}: arms not read as a list of arm names"
        assert isinstance(meta["arms"][0], str)
        checked += 1
    assert checked, "no manifests were parsed"

    # And every EVALUATION run the repo has satisfies the arithmetic precondition, which is
    # the evidence that n_samples >= primary_k is a real property of real runs and not a
    # rule invented for this validator. Preflight and smoke runs are excluded by
    # construction: they draw 1-2 samples against primary_k=32 under
    # `--allow-k-substitution`, which is exactly why they are not reportable and why a bank
    # group must not accept one.
    evaluation = [
        p
        for p in manifests
        if (run_meta(p.parent)["n_items"] or 0) >= 10 and run_meta(p.parent)["profile_reportable"]
    ]
    assert evaluation, "no reportable evaluation runs to check"
    for path in evaluation:
        meta = run_meta(path.parent)
        assert int(meta["n_samples"]) >= int(meta["primary_k"]), path


def test_run_meta_refuses_a_directory_with_no_manifest(tmp_path):
    with pytest.raises(typer.BadParameter, match=RUN_MANIFEST_FILENAME):
        run_meta(tmp_path)


def test_the_pre_v4_2_1_manifest_shape_yields_nothing_and_therefore_fails(tmp_path):
    """The exact artifact the old verifier was written against, and would have passed.

    ``{"seed": ..., "arm": ..., "n_rows": ...}`` is what the first version of the bank
    verifier read. No graph run writes it. Every field comes back ``None`` — and because
    the old checker skipped ``None``, that file verified clean. This asserts the current
    checker treats "the manifest does not say" as a failure instead.

    The filename is not the discriminator: this repo's primary development box is Windows,
    where ``run_manifest.json`` and ``RUN_MANIFEST.json`` are the same path. The shape is.
    """
    (tmp_path / RUN_MANIFEST_FILENAME).write_text(
        json.dumps({"seed": 50241, "arm": "multi_agent_leak", "n_rows": 100}), encoding="utf-8"
    )
    meta = run_meta(tmp_path)
    assert meta["base_seed"] is None
    assert meta["arms"] == []
    assert meta["n_samples"] is None

    group = dict(GENERATION_BUDGET["groups"]["natural"])
    failures = check_group([meta], "natural", {**group, "n_runs": 1, "seeds": [group["seeds"][0]]})
    assert any("no sampling.base_seed" in f for f in failures), failures
    assert any("does not run arm" in f for f in failures), failures
    assert any("records no sampling.n_samples" in f for f in failures), failures


# =====================================================================================
# 3. Natural and retain run groups cannot be confused
# =====================================================================================


def test_the_two_seed_groups_are_disjoint():
    assert set(ENGINEERING_BANK_SEEDS).isdisjoint(ENGINEERING_RETAIN_SEEDS)
    assert validate_budget(GENERATION_BUDGET) == []

    shared = json.loads(json.dumps(GENERATION_BUDGET))
    shared["groups"]["retain"]["seeds"] = list(ENGINEERING_BANK_SEEDS)
    failures = validate_budget(shared)
    assert any("wearing two labels" in f for f in failures), failures


def test_a_retain_run_supplied_to_the_natural_group_is_rejected():
    """A real retain run, checked against the natural group's pre-registration."""
    retain = _real_run(is_retain=True)
    meta = run_meta(retain)
    group = dict(GENERATION_BUDGET["groups"]["natural"])
    meta["base_seed"] = group["seeds"][0]
    failures = check_group([meta], "natural", {**group, "n_runs": 1, "seeds": [group["seeds"][0]]})
    assert any("is_retain" in f for f in failures), failures
    assert any("recall numerator" in f for f in failures), failures


def test_a_natural_run_supplied_to_the_retain_group_is_rejected():
    natural = _real_run(is_retain=False, cohort_split="discovery")
    meta = run_meta(natural)
    group = dict(GENERATION_BUDGET["groups"]["retain"])
    meta["base_seed"] = group["seeds"][0]
    failures = check_group([meta], "retain", {**group, "n_runs": 1, "seeds": [group["seeds"][0]]})
    assert any("is_retain" in f for f in failures), failures


def test_the_same_seed_twice_in_one_group_is_rejected():
    natural = _real_run(is_retain=False, cohort_split="discovery")
    group = dict(GENERATION_BUDGET["groups"]["natural"])
    twice = []
    for _ in range(2):
        meta = run_meta(natural)
        meta["base_seed"] = group["seeds"][0]
        twice.append(meta)
    failures = check_group(
        [twice[0], twice[1]], "natural", {**group, "n_runs": 2, "seeds": group["seeds"][:2]}
    )
    assert any("not unique" in f for f in failures), failures


def test_a_run_drawn_under_the_studys_own_base_seed_is_rejected():
    """Every committed run uses base_seed 1729. None of them is an engineering-bank draw."""
    natural = _real_run(is_retain=False, cohort_split="discovery")
    meta = run_meta(natural)
    assert meta["base_seed"] == 1729
    group = dict(GENERATION_BUDGET["groups"]["natural"])
    failures = check_group([meta], "natural", {**group, "n_runs": 1, "seeds": [group["seeds"][0]]})
    assert any("pre-registered" in f for f in failures), failures


# =====================================================================================
# 4. Partial API results resume without duplicate payments
# =====================================================================================


def _row(audit_id: str, candidate: str = "She was born in Rome.") -> dict:
    return {
        "audit_id": audit_id,
        "protected_question": "Where was Ada Vane born?",
        "candidate_text": candidate,
    }


def _stored(row: dict, *, prompt_version: str = PROMPT_VERSION, **overrides) -> dict:
    system, user = build_prompt(row, pass_name="blind")
    record = {
        "audit_id": row["audit_id"],
        "answer_attempt": "ANSWER",
        "subject_only": "no",
        "refusal": "no",
        "question_type": "slot",
        "prompt_version": prompt_version,
        "prompt_sha256": prompt_sha256(system, user),
        "source": "model",
        "error": None,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    record.update(overrides)
    return record


def _partial(tmp_path: Path, records: list[dict]) -> Path:
    path = tmp_path / "partial.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n", encoding="utf-8"
    )
    return path


def test_a_completed_row_is_reused_and_not_paid_for_twice(tmp_path):
    rows = [_row(f"{i:016x}") for i in range(3)]
    path = _partial(tmp_path, [_stored(rows[0]), _stored(rows[1])])
    reusable, conflicts = load_resumable(path, rows, pass_name="blind")
    assert conflicts == []
    assert set(reusable) == {rows[0]["audit_id"], rows[1]["audit_id"]}


def test_a_failed_row_is_not_reused_because_resuming_is_for_retrying_it(tmp_path):
    rows = [_row(f"{i:016x}") for i in range(2)]
    path = _partial(
        tmp_path,
        [_stored(rows[0]), _stored(rows[1], source="failed", error="503", answer_attempt=None)],
    )
    reusable, conflicts = load_resumable(path, rows, pass_name="blind")
    assert conflicts == []
    assert set(reusable) == {rows[0]["audit_id"]}


def test_a_row_whose_prompt_changed_is_a_conflict_not_an_overwrite(tmp_path):
    """The stored label answers a different question. Refused, never silently re-used."""
    rows = [_row("0" * 16)]
    stale = _stored(rows[0])
    stale["prompt_sha256"] = "0" * 64
    reusable, conflicts = load_resumable(_partial(tmp_path, [stale]), rows, pass_name="blind")
    assert reusable == {}
    assert any("different question" in c for c in conflicts), conflicts


def test_a_row_stored_under_a_different_prompt_version_is_a_conflict(tmp_path):
    rows = [_row("0" * 16)]
    stale = _stored(rows[0], prompt_version="v4.2-prompt-1")
    reusable, conflicts = load_resumable(_partial(tmp_path, [stale]), rows, pass_name="blind")
    assert reusable == {}
    assert any("prompt version" in c for c in conflicts), conflicts


def test_a_row_stored_twice_with_different_content_is_a_conflict(tmp_path):
    rows = [_row("0" * 16)]
    a = _stored(rows[0])
    b = _stored(rows[0], answer_attempt="NONE")
    reusable, conflicts = load_resumable(_partial(tmp_path, [a, b]), rows, pass_name="blind")
    assert reusable == {}
    assert any("stored twice" in c for c in conflicts), conflicts


def test_a_stored_row_the_input_no_longer_has_is_a_conflict(tmp_path):
    rows = [_row("0" * 16)]
    orphan = _stored(_row("f" * 16))
    reusable, conflicts = load_resumable(_partial(tmp_path, [orphan]), rows, pass_name="blind")
    assert reusable == {}
    assert any("not in the input file" in c for c in conflicts), conflicts


# =====================================================================================
# 5. A report cannot be backed by anything less than four complete runs
# =====================================================================================


def _manifest(role: str, pass_name: str, **overrides) -> dict:
    spec = JUDGES[role]
    manifest = {
        "judge": role,
        "pass": pass_name,
        "complete": True,
        "reportable": True,
        "run_id": None,
        "prompt_version": PROMPT_VERSION,
        "provider": spec["provider"],
        "requested_model": spec["requested_model"],
        "returned_models_seen": [spec["requested_model"]],
        "n_malformed_or_missing": 0,
        "n_rows_called": 2,
        "n_provider_request_ids": 2,
        "output_file_sha256": None,
    }
    manifest.update(overrides)
    return manifest


def _overlay(ids: list[str]) -> list[dict]:
    return [{"audit_id": i, "answer_attempt": "NONE"} for i in ids]


def test_a_complete_run_passes_verification():
    ids = ["a" * 16, "b" * 16]
    assert (
        verify_run(
            _manifest("A", "blind"),
            role="A",
            pass_name="blind",
            overlay_rows=_overlay(ids),
            input_ids=ids,
            duplicate_ids=[],
        )
        == []
    )


def test_a_smoke_run_cannot_stand_in_for_a_pass():
    """The defect: a five-row --limit run wrote to the real pass's filename."""
    ids = ["a" * 16]
    failures = verify_run(
        _manifest("A", "blind", reportable=False, run_id="smoke-a-blind", complete=False),
        role="A",
        pass_name="blind",
        overlay_rows=_overlay(ids),
        input_ids=ids,
        duplicate_ids=[],
    )
    assert any("reportable=false" in f for f in failures), failures
    assert any("smoke directory" in f for f in failures), failures


def test_an_uncovered_input_row_is_a_failure():
    failures = verify_run(
        _manifest("A", "blind"),
        role="A",
        pass_name="blind",
        overlay_rows=_overlay(["a" * 16]),
        input_ids=["a" * 16, "b" * 16],
        duplicate_ids=[],
    )
    assert any("no overlay row" in f for f in failures), failures


def test_a_duplicate_audit_id_is_a_failure():
    ids = ["a" * 16]
    failures = verify_run(
        _manifest("A", "blind"),
        role="A",
        pass_name="blind",
        overlay_rows=_overlay(ids),
        input_ids=ids,
        duplicate_ids=["a" * 16],
    )
    assert any("duplicate audit_ids" in f for f in failures), failures


def test_a_returned_model_that_is_not_the_requested_one_is_a_failure():
    ids = ["a" * 16]
    failures = verify_run(
        _manifest("A", "blind", returned_models_seen=["some-other-model"]),
        role="A",
        pass_name="blind",
        overlay_rows=_overlay(ids),
        input_ids=ids,
        duplicate_ids=[],
    )
    assert any("returned" in f for f in failures), failures


def test_a_moved_prompt_version_is_a_failure():
    ids = ["a" * 16]
    failures = verify_run(
        _manifest("A", "blind", prompt_version="v4.2-prompt-1"),
        role="A",
        pass_name="blind",
        overlay_rows=_overlay(ids),
        input_ids=ids,
        duplicate_ids=[],
    )
    assert any("prompt version" in f for f in failures), failures


def test_an_output_hash_that_no_longer_matches_the_file_is_a_failure():
    ids = ["a" * 16]
    failures = verify_run(
        _manifest("A", "blind", output_file_sha256="0" * 64),
        role="A",
        pass_name="blind",
        overlay_rows=_overlay(ids),
        input_ids=ids,
        duplicate_ids=[],
    )
    assert any("changed after the run" in f for f in failures), failures


# =====================================================================================
# 6. The adjudication file carries the evidence, and nothing else
# =====================================================================================


def test_a_disagreement_row_carries_the_question_and_the_candidate():
    """The defect: the file carried an audit_id and two labels, and nothing to resolve."""
    evidence = {
        "a"
        * 16: {
            "protected_question": "Where was Ada Vane born?",
            "candidate_text": "She was born in Rome.",
            "reference_answer": "Rome",
        }
    }
    rows = disagreements(
        {"a" * 16: {"answer_attempt": "ANSWER"}},
        {"a" * 16: {"answer_attempt": "NONE"}},
        fields=("answer_attempt",),
        pass_name="blind",
        evidence=evidence,
    )
    assert len(rows) == 1
    assert rows[0]["evidence"]["protected_question"] == "Where was Ada Vane born?"
    assert rows[0]["evidence"]["candidate_text"] == "She was born in Rome."
    # The blind pass does NOT get the reference answer, even though the caller had it.
    assert "reference_answer" not in rows[0]["evidence"]
    assert rows[0]["answer_attempt"] is None, "the field to resolve is left blank"


def test_the_reference_pass_adjudication_row_does_get_the_reference_answer():
    evidence = {
        "a"
        * 16: {
            "protected_question": "Where was Ada Vane born?",
            "candidate_text": "She was born in Rome.",
            "reference_answer": "Rome",
        }
    }
    rows = disagreements(
        {"a" * 16: {"reference_content": "YES"}},
        {"a" * 16: {"reference_content": "NO"}},
        fields=("reference_content",),
        pass_name="reference",
        evidence=evidence,
    )
    assert rows[0]["evidence"]["reference_answer"] == "Rome"


def test_two_judges_agreeing_on_uncertain_is_still_unresolved():
    """Agreement that neither judge could tell is not a resolved label."""
    rows = disagreements(
        {"a" * 16: {"reference_content": "UNCERTAIN"}},
        {"a" * 16: {"reference_content": "UNCERTAIN"}},
        fields=("reference_content",),
        pass_name="reference",
        evidence={},
    )
    assert len(rows) == 1
    assert rows[0]["reason"] == "both judges UNCERTAIN"


# =====================================================================================
# 7. The audit sample is drawn on generation metadata, never on a detector score
# =====================================================================================


def _bank_row(i: int, *, population="protected", nli_leaking=False, partition="heldout") -> dict:
    return {
        "text": f"candidate {i}",
        "request": f"question {i % 7}",
        "item_id": f"forget10-{i:04d}",
        "population": population,
        "partition": partition,
        "nli_leaking": nli_leaking,
        "pair_sha256": f"{i:064x}",
        "text_sha256": f"{i:064x}",
        # Deliberately present, deliberately never read. A sampler that consulted this
        # would make every recall number a measurement of itself.
        "detector_answer_probability": 0.99,
    }


def test_the_stratum_comes_from_generation_metadata_only():
    assert assign_stratum(_bank_row(0, nli_leaking=True)) == "protected_likely_answer"
    assert assign_stratum(_bank_row(0, nli_leaking=False)) == "protected_clean"
    assert assign_stratum(_bank_row(0, population="retain", partition="retain")) == "retain"
    # An unlabelled protected row is in NEITHER protected stratum.
    assert assign_stratum(_bank_row(0, nli_leaking=None)) is None


def test_the_draw_is_deterministic_and_independent_of_row_order():
    rows = [
        _bank_row(
            i,
            nli_leaking=i % 3 == 0,
            population="retain" if i % 5 == 0 else "protected",
            partition="retain" if i % 5 == 0 else "heldout",
        )
        for i in range(200)
    ]
    plan = {"protected_likely_answer": 10, "protected_clean": 20, "retain": 15}
    first, _ = draw_sample(rows, bank_id="bank-x", plan=plan)
    second, _ = draw_sample(list(reversed(rows)), bank_id="bank-x", plan=plan)
    for stratum in plan:
        assert [r["pair_sha256"] for r in first[stratum]] == [
            r["pair_sha256"] for r in second[stratum]
        ]


def test_a_different_bank_draws_a_different_sample():
    rows = [_bank_row(i, nli_leaking=False) for i in range(100)]
    plan = {"protected_likely_answer": 0, "protected_clean": 10, "retain": 0}
    a, _ = draw_sample(rows, bank_id="bank-a", plan=plan)
    b, _ = draw_sample(rows, bank_id="bank-b", plan=plan)
    assert [r["pair_sha256"] for r in a["protected_clean"]] != [
        r["pair_sha256"] for r in b["protected_clean"]
    ]


def test_a_short_stratum_is_reported_rather_than_quietly_filled():
    rows = [_bank_row(i, nli_leaking=False) for i in range(5)]
    plan = {"protected_likely_answer": 10, "protected_clean": 10, "retain": 10}
    drawn, shortfalls = draw_sample(rows, bank_id="bank-x", plan=plan)
    assert len(drawn["protected_clean"]) == 5
    assert shortfalls["protected_clean"] == 5
    assert shortfalls["retain"] == 10


# =====================================================================================
# 8. Two judges from one family are not two judges
# =====================================================================================


def test_the_frozen_roster_is_two_distinct_families():
    ok, reason = judge_families_are_independent()
    assert ok, reason
    families = {PROVIDERS[spec["provider"]]["family"] for spec in JUDGES.values()}
    assert len(families) == len(JUDGES)


def test_two_judges_from_one_family_are_refused():
    same = {
        "A": {"provider": "openai", "requested_model": "gpt-5.6-sol"},
        "B": {"provider": "openai", "requested_model": "gpt-5.6-terra"},
    }
    ok, reason = judge_families_are_independent(same)
    assert not ok
    assert "shared prior" in reason
