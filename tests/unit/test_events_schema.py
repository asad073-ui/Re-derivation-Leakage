"""The typed event schema.

Design invariant 4: the transcript is a typed event log, not a string. These tests pin
down the two properties that makes worth anything — that every event validates strictly,
and that an unknown schema version is REFUSED rather than coerced into new fields.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from rdl.orchestrator.events import (
    EVENT_KINDS,
    SCHEMA_VERSION,
    AgentAnswer,
    Delegation,
    FinalAnswer,
    Handoff,
    MemoryWrite,
    Retrieval,
    UserQuery,
    WriteAttempt,
    parse_event,
)


def test_every_event_kind_is_in_the_closed_union():
    kinds = {
        UserQuery(text="q").kind,
        Retrieval(query="q").kind,
        AgentAnswer(agent_id="A", text="a").kind,
        Delegation(from_id="A", to_id="B").kind,
        Handoff(from_id="A", to_id="B", text="A's answer").kind,
        MemoryWrite(node_id="m", content="c", source_agent="B").kind,
        WriteAttempt(allowed=False).kind,
        FinalAnswer(text="f").kind,
    }
    assert kinds == set(EVENT_KINDS)


def test_events_carry_turn_ts_and_schema_version():
    e = UserQuery(text="q", turn=3)
    assert e.turn == 3
    assert e.ts
    assert e.schema_version == SCHEMA_VERSION


def test_events_are_frozen():
    e = UserQuery(text="q")
    with pytest.raises(ValidationError):
        e.text = "mutated"


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        UserQuery(text="q", nonsense=1)


def test_retrieval_records_both_returned_and_blocked():
    """Suppressed ids are a first-class fact, not an omission."""
    e = Retrieval(query="q", returned_node_ids=["a"], blocked_node_ids=["m1"])
    assert e.returned_node_ids == ["a"]
    assert e.blocked_node_ids == ["m1"]


def test_memory_write_allows_empty_parent_ids():
    """Empty parent_ids on an agent_answer node IS the laundering signature."""
    e = MemoryWrite(node_id="m2", content="the fact", source_agent="B")
    assert e.parent_ids == []


# ------------------------------------------------------------------- round trip --


@pytest.mark.parametrize(
    "event",
    [
        UserQuery(text="q", item_id="forget10-0000"),
        Retrieval(query="q", returned_node_ids=[], blocked_node_ids=["m1"]),
        AgentAnswer(agent_id="A", text="I don't know.", abstained=True, logprob=-2.6),
        Delegation(from_id="A", to_id="B", reason="A abstained"),
        Handoff(from_id="A", to_id="B", text="f", text_sha256="ab", included_abstention=False),
        MemoryWrite(node_id="m2", content="f", parent_ids=[], source_agent="B"),
        FinalAnswer(text="f", contributing_agent_ids=["A", "B"]),
    ],
    ids=lambda e: e.kind,
)
def test_event_round_trips_through_json(event):
    blob = json.dumps(event.model_dump(mode="json"))
    back = parse_event(json.loads(blob))
    assert type(back) is type(event)
    assert back.model_dump() == event.model_dump()


def test_parse_event_dispatches_on_kind():
    payload = {"schema_version": SCHEMA_VERSION, "kind": "delegation", "from_id": "A", "to_id": "B"}
    assert isinstance(parse_event(payload), Delegation)


def test_parse_event_passes_through_an_already_typed_event():
    e = UserQuery(text="q")
    assert parse_event(e) is e


def test_unknown_schema_version_is_refused():
    """Old events must be read with a migration, never coerced into new fields."""
    payload = {"schema_version": 999, "kind": "user_query", "text": "q"}
    with pytest.raises(ValueError, match="schema_version"):
        parse_event(payload)


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown event kind"):
        parse_event({"schema_version": SCHEMA_VERSION, "kind": "telepathy", "text": "q"})


def test_parse_event_rejects_a_non_mapping():
    with pytest.raises(TypeError):
        parse_event(["not", "a", "dict"])


def test_fixture_transcript_parses(laundering_transcript_fixture):
    """The checked-in canonical trace must stay parseable as the schema evolves."""
    events = [parse_event(e) for e in laundering_transcript_fixture["events"]]
    assert [e.kind for e in events] == [
        "user_query",
        "retrieval",
        "agent_answer",
        "delegation",
        "agent_answer",
        "final_answer",
        "memory_write",
    ]
    write = events[-1]
    assert write.parent_ids == [], "the laundered node has no derivation edges"
    assert events[1].blocked_node_ids == ["m1"]
