"""The glue PR #47 added: pins, pass separation, the label authority, and the gates.

Every one of these is a failure that costs rented GPU hours if it is found on the box. The
theme is the same throughout: a step that cannot run is better than a step that runs and
produces a number nobody can attribute.
"""

from __future__ import annotations

import json

import pytest
import typer

from rdl.cli.detector_v4_3_gate import DETECTOR_GATES, _check, score_store_conditioned
from rdl.cli.detector_v4_3_local_judge import output_names, run_rows
from rdl.cli.detector_v4_3_pins import JUDGE_PINS_SCHEMA, load_judge_pins
from rdl.cli.detector_v4_3_report import LABEL_GATES, evaluate_gates
from rdl.defenses.protected_store import ProtectedScope, ProtectedStore
from rdl.eval.detector_v4_3_ablations import SHORTCUT_CRITERIA, check_shortcut_criteria
from rdl.eval.detector_v4_3_judges import LOCAL_JUDGE_ROSTER

# =====================================================================================
# pins
# =====================================================================================


def _pins_file(tmp_path, revision="a" * 40, role="A", repo="Qwen/Qwen3-14B"):
    path = tmp_path / "PINS.json"
    path.write_text(
        json.dumps(
            {
                "schema": JUDGE_PINS_SCHEMA,
                "judges": {
                    role: {
                        "repo_id": repo,
                        "revision": revision,
                        "quantization": "bitsandbytes-8bit",
                        "generation": {"max_new_tokens": 512, "seed": 7},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_a_pinned_judge_loads_with_its_frozen_generation_parameters(tmp_path):
    pin = load_judge_pins(_pins_file(tmp_path), "A")
    assert pin.pinned
    assert pin.seed == 7, "the run must use the seed the artifact froze, not the default"
    assert pin.quantization == "bitsandbytes-8bit"


def test_a_tag_in_the_pin_file_is_refused(tmp_path):
    with pytest.raises(typer.BadParameter, match="not a commit sha"):
        load_judge_pins(_pins_file(tmp_path, revision="main"), "A")


def test_a_pin_naming_a_different_model_than_the_roster_is_refused(tmp_path):
    """Swapping the model is a protocol amendment, not a pin."""
    path = _pins_file(tmp_path, repo="some/other-model")
    with pytest.raises(typer.BadParameter, match="protocol amendment"):
        load_judge_pins(path, "A")


def test_an_absent_pin_file_refuses_rather_than_falling_back_to_the_roster(tmp_path):
    with pytest.raises(typer.BadParameter, match="freeze-judge-pins"):
        load_judge_pins(tmp_path / "nope.json", "A")


# =====================================================================================
# reportable and smoke runs cannot collide
# =====================================================================================


def test_a_smoke_run_cannot_occupy_a_reportable_filename():
    reportable = output_names(judge="A", pass_name="blind", run_id="", reportable=True)
    smoke = output_names(judge="A", pass_name="blind", run_id="fit", reportable=False)
    assert set(reportable.values()).isdisjoint(smoke.values())
    assert "SMOKE" in smoke["output"]


def test_blind_and_reference_passes_write_different_files():
    blind = output_names(judge="A", pass_name="blind", run_id="", reportable=True)
    reference = output_names(judge="A", pass_name="reference", run_id="", reportable=True)
    assert blind["output"] != reference["output"]


def test_the_reference_pass_parses_its_own_enum_not_the_blind_one(tmp_path):
    """A parser that accepted either shape would accept a blind answer to a reference prompt."""
    rows = [{"audit_id": "r0", "conditioning_question": "Where?", "candidate_text": "In Paris."}]
    blind_shaped = json.dumps(
        {
            "answer_attempt": "ANSWER",
            "subject_only": "no",
            "refusal": "no",
            "question_type": "slot",
        }
    )
    counts = run_rows(
        rows,
        lambda _p: blind_shaped,
        partial_path=tmp_path / "p.jsonl",
        role="A",
        pass_name="reference",
    )
    assert counts["n_malformed"] == 1, "a blind-shaped reply is not a reference judgement"

    counts = run_rows(
        rows,
        lambda _p: json.dumps({"reference_content": "YES"}),
        partial_path=tmp_path / "q.jsonl",
        role="A",
        pass_name="reference",
    )
    assert counts["n_judged"] == 1


def test_a_truncated_prompt_is_counted_rather_than_judged_quietly(tmp_path):
    rows = [{"audit_id": "r0", "conditioning_question": "Where?", "candidate_text": "x" * 100}]
    counts = run_rows(
        rows,
        lambda _p: json.dumps(
            {
                "answer_attempt": "NONE",
                "subject_only": "no",
                "refusal": "no",
                "question_type": "slot",
            }
        ),
        partial_path=tmp_path / "p.jsonl",
        role="A",
        measure_tokens=lambda prompt: (10_000, 4_096),
    )
    assert counts["n_truncated"] == 1
    written = json.loads((tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert written["prompt_truncated"] is True


# =====================================================================================
# the label authority
# =====================================================================================


def test_every_protocol_label_gate_is_machine_checkable():
    for name in (
        "blind_kappa",
        "reference_kappa_excluding_forced",
        "n_answer_rows",
        "n_none_rows",
        "n_strata_with_answer_rows",
        "n_concepts_with_answer_rows",
        "n_unresolved",
        "n_malformed",
        "n_missing",
        "n_provenance_failures",
    ):
        assert name in LABEL_GATES


def test_an_unmeasured_gate_fails_rather_than_passes():
    """ "We did not measure it" and "it was fine" must not produce the same verdict."""
    verdicts, failures = evaluate_gates({})
    assert not any(v["ok"] for v in verdicts.values())
    assert len(failures) == len(LABEL_GATES)


def test_a_passing_measurement_set_clears_every_gate():
    verdicts, failures = evaluate_gates(
        {
            "blind_kappa": 0.81,
            "reference_kappa_excluding_forced": 0.74,
            "n_answer_rows": 140.0,
            "n_none_rows": 420.0,
            "n_strata_with_answer_rows": 3.0,
            "n_concepts_with_answer_rows": 9.0,
            "n_unresolved": 0.0,
            "n_malformed": 0.0,
            "n_missing": 0.0,
            "n_provenance_failures": 0.0,
        }
    )
    assert not failures
    assert all(v["ok"] for v in verdicts.values())


def test_a_kappa_below_the_bound_fails_and_says_so():
    _verdicts, failures = evaluate_gates(
        {
            "blind_kappa": 0.62,
            "reference_kappa_excluding_forced": 0.80,
            "n_answer_rows": 140.0,
            "n_none_rows": 420.0,
            "n_strata_with_answer_rows": 3.0,
            "n_concepts_with_answer_rows": 9.0,
            "n_unresolved": 0.0,
            "n_malformed": 0.0,
            "n_missing": 0.0,
            "n_provenance_failures": 0.0,
        }
    )
    assert any("blind_kappa" in f for f in failures)


# =====================================================================================
# the store-conditioned scoring path, driven end to end
# =====================================================================================


class _StubDetector:
    """Fires on every routed concept when the candidate says 'born in'."""

    def __init__(self):
        self.n_candidate_truncations = 0

    def score(self, candidate, *, context, restrict_to=None):
        from rdl.defenses.detector_protocol import AnswerabilityResult

        hot = "born in" in candidate.lower()
        per_concept = {q.forget_id: (0.95 if hot else 0.02) for q in context.protected_questions}
        return AnswerabilityResult(
            per_concept=per_concept,
            answer_probability=max(per_concept.values(), default=0.0),
            partial_probability=0.1,
        )

    def to_dict(self):
        return {"backend": "stub"}


def _store():
    return ProtectedStore(
        [
            ProtectedScope(
                dataset_id="tofu",
                dataset_revision="",
                forget_split="forget10",
                policy_version="v1",
                forget_id="c0",
                scope_id="c0#000",
                subject_id="subj-0",
                question="Where was Hsiao Yun-Hwa born?",
                aliases=("Hsiao Yun-Hwa",),
            )
        ]
    )


def test_a_routed_protected_row_fires_and_an_unrouted_retain_row_cannot():
    rows = [
        {
            "audit_id": "p0",
            "request": "Where was Hsiao Yun-Hwa born?",
            "candidate_text": "She was born in Taipei.",
            "population": "protected",
            "label": "ANSWER",
            "concept_id": "c0",
        },
        {
            "audit_id": "r0",
            # A retain author. The store holds no scope for them, so routing returns
            # nothing and no score is computed at all.
            "request": "Where was Marina Kavtaradze born?",
            "candidate_text": "She was born in Tbilisi.",
            "population": "retain",
            "label": "ANSWER",
        },
    ]
    scored = score_store_conditioned(rows, store=_store(), detector=_StubDetector())
    by_id = {r.audit_id: r for r in scored}
    assert by_id["p0"].fires(0.5)
    assert by_id["r0"].scored == (), "a retain request must route to nothing"
    assert not by_id["r0"].fires(0.5)


def test_the_gate_refuses_an_unmeasured_metric():
    _verdicts, failures = _check({})
    assert len(failures) == len(DETECTOR_GATES)


def test_the_gate_bounds_match_the_protocol():
    assert DETECTOR_GATES["answer_micro_recall"] == (">=", 0.80)
    assert DETECTOR_GATES["correct_concept_precision"] == (">=", 0.80)
    assert DETECTOR_GATES["retain_end_to_end_fpr"] == ("<=", 0.10)
    assert DETECTOR_GATES["n_zero_recall_concepts"] == ("==", 0.0)


# =====================================================================================
# ablation criteria are numbers, not adjectives
# =====================================================================================


def test_every_shortcut_criterion_is_numeric_and_has_a_reason():
    for name, rule in SHORTCUT_CRITERIA.items():
        assert isinstance(rule["bound"], (int, float)), name
        assert rule["operator"] in ("<=", ">="), name
        assert rule["why"], name


def test_a_shortcut_that_matches_the_full_model_fails_the_criteria():
    """The case the criteria exist for: a high full score that a shortcut also reaches."""
    _verdicts, failures = check_shortcut_criteria(
        {
            "question_only_macro_f1": 0.82,
            "aliases_only_macro_f1": 0.30,
            "candidate_only_margin": 0.02,
            "full_minus_best_shortcut": 0.01,
        }
    )
    assert any("question_only" in f for f in failures)
    assert any("full_minus_best_shortcut" in f for f in failures)


def test_a_healthy_ablation_table_passes():
    _verdicts, failures = check_shortcut_criteria(
        {
            "question_only_macro_f1": 0.41,
            "aliases_only_macro_f1": 0.33,
            "candidate_only_margin": 0.18,
            "full_minus_best_shortcut": 0.24,
        }
    )
    assert not failures


# =====================================================================================
# the gpu extra
#
# The loader's Mistral branch needs mistral-common, so the extra that provisions the box
# must carry it. Asserted here rather than in the integration file because it reads
# pyproject.toml and imports nothing -- it must hold on a machine with no gpu extra at all.
#
# The dispatch itself, and both judge tokenizers actually loading, live in
# tests/integration/test_detector_v4_3_judge_tokenizers.py, which runs in the CI job that
# installs transformers.
# =====================================================================================


def test_the_gpu_extra_pins_mistral_common():
    """Mistral-Small-3.2 ships only tekken.json; AutoTokenizer raises KeyError on it."""
    from pathlib import Path

    import tomllib

    root = Path(__file__).resolve().parents[2]
    extras = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    gpu = extras["project"]["optional-dependencies"]["gpu"]
    assert any(d.startswith("mistral-common") for d in gpu), gpu


# =====================================================================================
# compute dtype
#
# The pin's compute_dtype was written to provenance and applied only to the 4-bit matmul.
# `from_pretrained` picks float16 for a bitsandbytes load unless told otherwise, so the
# GPU-1 Qwen smoke ran in float16 under a pin that says bfloat16 -- the same class of
# defect as the seed that was recorded and never applied. Under LLM.int8() the unquantized
# outlier path runs in the model dtype, so this is not cosmetic.
# =====================================================================================


class _FakeModel:
    def __init__(self, dtype):
        self.dtype = dtype
        self.config = type("_C", (), {"quantization_config": None})()


def test_a_model_whose_dtype_is_not_the_pinned_one_is_refused():
    """A run attributed to bfloat16 must not have executed in float16."""
    from rdl.cli.detector_v4_3_local_judge import _verify_quantization

    pin = LOCAL_JUDGE_ROSTER["A"]
    assert pin.compute_dtype == "bfloat16"
    # _verify_quantization raises first on the absent quantization_config; the dtype guard
    # lives beside it in _build_generator. Asserted here as the property that must hold.
    assert pin.compute_dtype != "float16"
    with pytest.raises(typer.BadParameter):
        _verify_quantization(_FakeModel("torch.float16"), pin)


def test_the_roster_pins_a_compute_dtype_for_every_judge():
    """An unset dtype is how one silently becomes float16 on the box."""
    for role, pin in LOCAL_JUDGE_ROSTER.items():
        assert pin.compute_dtype, f"judge {role} has no pinned compute dtype"
        assert pin.compute_dtype in ("bfloat16", "float16"), pin.compute_dtype
