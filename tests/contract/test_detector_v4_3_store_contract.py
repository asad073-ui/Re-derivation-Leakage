"""What the v4.3 protected store may never contain, enforced rather than documented.

Four separable guarantees, because they fail in different ways:

1. the runtime store carries no answer and no answer *hash*;
2. no module under ``rdl.defenses`` contains code that can open the evaluation key;
3. the conditioning index cannot express a population, so no input builder can serialise
   one;
4. an unrouted request produces no protected score, rather than a low one.

The hash rule is the one worth stating out loud. A digest looks like a safe way to carry
an answer and is not: these answers are cities, years, genres and option keys, and a
candidate space that small is enumerable, so shipping ``answer_sha256`` hands its holder a
way to confirm the fact the system claims to have destroyed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdl.defenses.protected_store import (
    CONDITIONING_RECORD_FIELDS,
    RUNTIME_SCOPE_FIELDS,
    ConditioningIndex,
    ConditioningRecord,
    ProtectedScope,
    ProtectedStore,
    conditioning_records_from_questions,
    store_decision,
)

REPO = Path(__file__).resolve().parents[2]
STORE_DIR = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_3"


def _scope(**overrides) -> dict:
    row = {
        "dataset_id": "locuslab/TOFU",
        "forget_id": "tofu-forget10-author-0000",
        "scope_id": "tofu-forget10-author-0000#000",
        "question": "Where was Hsiao Yun-Hwa born?",
        "aliases": ["Hsiao Yun-Hwa"],
    }
    row.update(overrides)
    return row


# =====================================================================================
# 1. no answers, and no answer hashes
# =====================================================================================


@pytest.mark.parametrize(
    "field",
    [
        "answer",
        "gold_answer",
        "reference_answer",
        "answer_sha256",
        "answer_hash",
        "reference_completion",
        "option_key",
        "expected_answer",
        "solution",
    ],
)
def test_a_runtime_scope_refuses_every_answer_bearing_field(field):
    with pytest.raises(ValueError, match="may not carry"):
        ProtectedScope.from_mapping(_scope(**{field: "Taipei"}))


def test_the_refusal_explains_why_a_hash_is_not_a_safe_form_of_an_answer():
    """The message has to carry the reasoning, or the next person adds the field back."""
    with pytest.raises(ValueError) as caught:
        ProtectedScope.from_mapping(_scope(answer_sha256="deadbeef"))
    message = str(caught.value)
    assert "enumerate" in message
    assert "PROTECTED_STORE_EVAL_KEY.json" in message


def test_the_allowlist_refuses_unknown_fields_rather_than_ignoring_them():
    """A denylist has to anticipate the leaking field's name; this does not."""
    with pytest.raises(ValueError, match="unknown fields"):
        ProtectedScope.from_mapping(_scope(population="retain"))
    with pytest.raises(ValueError, match="unknown fields"):
        ProtectedScope.from_mapping(_scope(stratum="natural_leaking"))


def test_no_answer_bearing_name_is_in_either_allowlist():
    forbidden = ("answer", "gold", "reference", "completion", "option", "solution")
    for allowlist in (RUNTIME_SCOPE_FIELDS, CONDITIONING_RECORD_FIELDS):
        for name in allowlist:
            assert not any(
                token in name for token in forbidden
            ), f"{name} names the answer sheet and is on an allowlist"


def test_a_scope_edited_away_from_its_question_is_refused():
    with pytest.raises(ValueError, match="question_sha256"):
        ProtectedScope.from_mapping(_scope(question_sha256="0" * 64))


# =====================================================================================
# 2. runtime code contains no reader for the evaluation key
# =====================================================================================


def test_no_defenses_module_can_open_the_evaluation_key():
    """Structural, not conventional: the separation is that the code does not exist.

    A runtime module that merely *promised* not to read the key would still be one import
    away from reading it. This asserts the promise is unrepresentable there.
    """
    offenders = []
    for path in (REPO / "src" / "rdl" / "defenses").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        # The store module names the file in prose, to say where answers live. Naming it
        # in a docstring is the point; naming its SCHEMA is what a reader would need.
        if "graph-detector-v4-3-protected-store-eval-key" in text:
            offenders.append(path.name)
    assert not offenders, f"{offenders} reference the evaluation key's schema"


def test_the_written_runtime_store_carries_no_answer_key_anywhere():
    """Belt and braces on the artifact itself, not just the constructor."""
    path = STORE_DIR / "PROTECTED_STORE_RUNTIME.json"
    if not path.exists():
        pytest.skip("run `rdl graph-detector-v4-3-build-store` first")
    payload = json.loads(path.read_text(encoding="utf-8"))

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                yield str(key)
                yield from walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from walk(item)

    guilty = sorted(
        {
            key
            for key in walk(payload)
            if any(
                token in key.lower()
                for token in ("answer", "gold", "reference", "completion", "option")
            )
            and key not in {"carries_answers", "carries_answer_hashes"}
        }
    )
    assert not guilty, f"the frozen runtime store carries {guilty}"


def test_the_store_and_the_key_are_different_files():
    store = STORE_DIR / "PROTECTED_STORE_RUNTIME.json"
    key = STORE_DIR / "PROTECTED_STORE_EVAL_KEY.json"
    if not (store.exists() and key.exists()):
        pytest.skip("run `rdl graph-detector-v4-3-build-store` first")
    assert store.read_bytes() != key.read_bytes()
    assert ProtectedStore.load(store)  # the store loads; the key has no loader at all


# =====================================================================================
# 3. the conditioning index cannot express a population
# =====================================================================================


def test_a_conditioning_record_has_no_field_that_could_name_a_population():
    for name in ("population", "is_protected", "forget_id", "concept_id", "split", "stratum"):
        assert name not in CONDITIONING_RECORD_FIELDS
        with pytest.raises(ValueError):
            ConditioningRecord.from_mapping(
                {
                    "conditioning_id": "cond-0",
                    "conditioning_question": "Where was X born?",
                    name: "protected",
                }
            )


def test_one_extractor_runs_for_every_question_regardless_of_origin():
    """The v4.2 defect, as a test: the builder has no parameter that could branch."""
    index = conditioning_records_from_questions(
        {
            "protected": "What genre does Hsiao Yun-Hwa write in?",
            "retain": "What genre does Marina Kavtaradze write in?",
        }
    )
    aliases = {r.conditioning_id: r.subject_aliases for r in index}
    assert aliases["protected"], "a protected question yields aliases"
    assert aliases["retain"], "and so must a retain question -- same code path, no branch"


def test_the_frozen_index_gives_both_populations_aliases_at_a_similar_rate():
    """The number that would have caught v4.2: 0/300 retain against ~719/719 protected."""
    key_path = STORE_DIR / "PROTECTED_STORE_EVAL_KEY.json"
    index_path = STORE_DIR / "DETECTOR_V4_3_CONDITIONING_INDEX.json"
    if not (key_path.exists() and index_path.exists()):
        pytest.skip("run `rdl graph-detector-v4-3-build-store` first")
    parity = json.loads(key_path.read_text(encoding="utf-8"))["alias_parity_by_population"]
    protected = parity["protected"]["alias_coverage"]
    retain = parity["retain"]["alias_coverage"]
    assert retain > 0.90, f"retain alias coverage collapsed to {retain}"
    assert abs(protected - retain) < 0.05, (
        f"alias coverage differs by {abs(protected - retain):.3f} between populations; "
        "'has aliases' is becoming a population feature again"
    )
    assert ConditioningIndex.load(index_path)


# =====================================================================================
# 4. an unrouted request has no protected score
# =====================================================================================


def test_an_unrouted_request_produces_no_forget_id_and_no_score():
    decision = store_decision((), tau_answer=0.5, tau_partial=0.3)
    assert decision.fired_forget_ids == ()
    assert decision.n_routed_scopes == 0
    assert not decision.fires


def test_zero_because_unrouted_is_distinguishable_from_zero_because_quiet():
    """Both report 0.0. Only `n_routed_scopes` says which one happened."""
    from rdl.defenses.detection_context import ProtectedQuestion

    question = ProtectedQuestion(scope_id="s#0", forget_id="c0", question="Where born?")
    unrouted = store_decision((), tau_answer=0.5, tau_partial=0.3)
    quiet = store_decision(((question, 0.0, 0.0),), tau_answer=0.5, tau_partial=0.3)
    assert unrouted.answer_score == quiet.answer_score == 0.0
    assert unrouted.n_routed_scopes == 0
    assert quiet.n_routed_scopes == 1


def test_the_two_scores_are_never_combined():
    """A partial clue must not be able to outvote the ANSWER threshold."""
    from rdl.defenses.detection_context import ProtectedQuestion

    question = ProtectedQuestion(scope_id="s#0", forget_id="c0", question="Where born?")
    # High PARTIAL, low ANSWER. No weighted sum exists that could fire the Forget-ID.
    decision = store_decision(((question, 0.10, 0.99),), tau_answer=0.5, tau_partial=0.3)
    assert not decision.fires
    assert decision.partial_scope_ids == ("s#0",)
    assert decision.partial_score == pytest.approx(0.99)
