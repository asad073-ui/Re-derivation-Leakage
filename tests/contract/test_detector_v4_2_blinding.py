"""The v4.2 blinding contract, asserted over the bytes that would leave the process.

``DETECTOR_V4_1_PROTOCOL.md`` §4 makes blinding structural rather than conventional: the
unblinding key is a separate file marked ``judges_must_not_read``. That protects a human,
who has to open a file to read it. It does not protect a model judge, which is handed
whatever string the runner builds — so for v4.2 the guarantee has to be a property of the
prompt, and this file is where that property is checked.

The tests run against the REAL committed audit files when they are present, because a
blinding test over a synthetic row proves nothing about the rows that will actually be
sent. They skip, loudly, when the audit has not been built.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdl.eval.detector_v4_2 import (
    BLIND_FIELDS,
    FORBIDDEN_IN_PROMPT,
    JUDGES,
    REFERENCE_FIELDS,
    build_prompt,
    prompt_sha256,
)

REPO = Path(__file__).resolve().parents[2]
V4_1 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_1"
KEY_PATH = V4_1 / "LABEL_AUDIT_KEY.json"
BLIND_PATH = V4_1 / "LABEL_AUDIT_JUDGE_A.jsonl"
REFERENCE_PATH = V4_1 / "LABEL_AUDIT_REFERENCE_PASS_A.jsonl"

# Enough rows to cover every stratum without making the contract suite slow.
SAMPLE = 250


def _rows(path: Path, limit: int = SAMPLE) -> list[dict]:
    if not path.exists():
        pytest.skip(f"{path} is absent; run `rdl graph-detector-v4-label-audit`")
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
        if len(out) >= limit:
            break
    return out


def _key() -> dict:
    if not KEY_PATH.exists():
        pytest.skip(f"{KEY_PATH} is absent")
    return json.loads(KEY_PATH.read_text(encoding="utf-8")).get("rows", {})


# =====================================================================================
# The judge file carries nothing the key carries
# =====================================================================================


def test_the_blind_judge_file_carries_only_the_three_permitted_fields():
    """Whatever else is in the row, the prompt reads three fields. This checks the row."""
    permitted = {"audit_id", "protected_question", "candidate_text", *BLIND_FIELDS}
    for row in _rows(BLIND_PATH):
        assert set(row) <= permitted, sorted(set(row) - permitted)


def test_the_reference_judge_file_adds_only_the_reference_answer():
    permitted = {
        "audit_id",
        "protected_question",
        "candidate_text",
        "reference_answer",
        *REFERENCE_FIELDS,
    }
    for row in _rows(REFERENCE_PATH):
        assert set(row) <= permitted, sorted(set(row) - permitted)


# =====================================================================================
# No hidden value reaches a prompt
# =====================================================================================


# The key's identifying fields. ``population`` and ``bank_partition`` are deliberately NOT
# here: their values are the ordinary words "protected", "retain" and "development", which
# occur in candidate text and in the rubric, so a substring check over them tests English
# rather than blinding. They are covered instead — and more strongly — by
# ``test_the_prompt_depends_on_no_field_outside_the_permitted_set``, which shows the prompt
# is a pure function of the two permitted fields and therefore cannot carry any of them.
HIGH_ENTROPY_KEY_FIELDS = ("concept_id", "item_id", "stratum", "text_sha256")


def test_no_hidden_key_value_appears_in_any_blind_prompt():
    """THE contract. The key's identifying values, checked against the built prompt.

    ``LABEL_AUDIT_KEY.json`` carries the NLI label, the stratum, the partition, the
    lexical sampling score, the item id and the concept id. Each would anchor a judgement,
    and one of them — ``nli_leaking`` — is the thing the audit exists to check.
    """
    key = _key()
    checked = 0
    for row in _rows(BLIND_PATH):
        hidden = key.get(str(row["audit_id"]))
        if not hidden:
            continue
        system, user = build_prompt(row, pass_name="blind")
        blob = f"{system}\n{user}"
        for field in HIGH_ENTROPY_KEY_FIELDS:
            value = str(hidden.get(field, "")).strip()
            if len(value) < 5:
                continue
            assert value not in blob, f"{field}={value!r} reached the {row['audit_id']} prompt"
        checked += 1
    assert checked, "no rows were checked; the key and the judge file do not share ids"


def test_no_forbidden_field_name_is_solicited_by_the_system_prompt():
    """The system prompt is ours, so it is held to the strict rule.

    The user message is not: it carries the candidate verbatim, and a candidate is allowed
    to contain the word "population" or "leaking". Sanitising it to satisfy a grep would
    be editing the evidence, which is worse than the thing the grep is looking for.
    """
    for row in _rows(BLIND_PATH, limit=50):
        system, _user = build_prompt(row, pass_name="blind")
        for forbidden in FORBIDDEN_IN_PROMPT:
            assert forbidden not in system, f"{forbidden} named in the system prompt"


def test_the_blind_prompt_never_carries_the_reference_answer_for_the_same_row():
    """Cross-checked against the reference-pass file, row by row.

    This is the v4.1 D2 separation expressed as a test: the same ``audit_id`` has a
    reference answer in one file, and that string must not appear in the other file's
    prompt.
    """
    reference_of = {
        str(r["audit_id"]): str(r.get("reference_answer", "")).strip()
        for r in _rows(REFERENCE_PATH, limit=1000)
    }
    checked = 0
    for row in _rows(BLIND_PATH, limit=1000):
        answer = reference_of.get(str(row["audit_id"]), "")
        if len(answer) < 8:
            continue
        _system, user = build_prompt(row, pass_name="blind")
        assert answer not in user, f"the reference answer reached {row['audit_id']}'s blind prompt"
        checked += 1
    assert checked, "no reference answers were long enough to check"


# =====================================================================================
# Determinism and independence
# =====================================================================================


def test_the_prompt_is_deterministic_for_a_row():
    """No timestamp, no nonce, no ordering that depends on a dict's iteration order.

    A prompt that varied between runs would make ``prompt_sha256`` unfalsifiable — it
    would record what was sent without letting anyone reconstruct it.
    """
    for row in _rows(BLIND_PATH, limit=20):
        first = prompt_sha256(*build_prompt(row, pass_name="blind"))
        second = prompt_sha256(*build_prompt(row, pass_name="blind"))
        assert first == second


def test_both_judges_are_sent_the_same_prompt_for_a_row():
    """The judges differ in the MODEL, not in what they were asked.

    Two judges given different prompts would make kappa a measurement of the prompts.
    ``build_prompt`` takes no judge argument at all, which is the enforcement; this asserts
    the property a reader would otherwise have to infer from that absence.
    """
    assert set(JUDGES) == {"A", "B"}
    for row in _rows(BLIND_PATH, limit=20):
        assert build_prompt(row, pass_name="blind") == build_prompt(row, pass_name="blind")


def test_the_prompt_depends_on_no_field_outside_the_permitted_set():
    """Adding a field to a judge row must not change the prompt.

    The defence against the obvious future mistake: someone adds ``stratum`` to the judge
    file for convenience, and every prompt silently starts carrying it.
    """
    for row in _rows(BLIND_PATH, limit=20):
        contaminated = {
            **row,
            "nli_leaking": True,
            "stratum": "clean_hard_negative",
            "concept_id": "author-0007",
            "lexical_score": 0.93,
        }
        assert build_prompt(row, pass_name="blind") == build_prompt(contaminated, pass_name="blind")
