"""Transcript accessors and serialisation.

The surface accessors are what `containment` checks. Keeping them on the transcript
stops each metric from re-deriving "what counts as an agent message" slightly
differently, which is the kind of drift that makes two numbers in the same table
incomparable.
"""

from __future__ import annotations

from rdl.orchestrator.events import (
    AgentAnswer,
    Delegation,
    FinalAnswer,
    MemoryWrite,
    Retrieval,
    UserQuery,
)
from rdl.orchestrator.transcript import Transcript


def _canonical() -> Transcript:
    tr = Transcript(episode_id="e1", condition="C3", seed=0, item_id="forget10-0000")
    tr.append(UserQuery(turn=0, text="Who was X's father?", item_id="forget10-0000"))
    tr.append(
        Retrieval(
            turn=0, query="Who was X's father?", returned_node_ids=[], blocked_node_ids=["m1"]
        )
    )
    tr.append(AgentAnswer(turn=1, agent_id="A", text="I don't know.", abstained=True))
    tr.append(Delegation(turn=1, from_id="A", to_id="B", reason="A abstained"))
    tr.append(AgentAnswer(turn=1, agent_id="B", text="X's father was a florist.", abstained=False))
    tr.append(
        FinalAnswer(turn=1, text="X's father was a florist.", contributing_agent_ids=["A", "B"])
    )
    tr.append(
        MemoryWrite(
            turn=1,
            node_id="m2",
            content="X's father was a florist.",
            parent_ids=[],
            source_agent="B",
        )
    )
    return tr


def test_filters_by_kind():
    tr = _canonical()
    assert len(tr.agent_answers()) == 2
    assert len(tr.retrievals()) == 1
    assert len(tr.memory_writes()) == 1
    assert len(tr.delegations()) == 1
    assert len(tr) == 7


def test_surface_accessors():
    tr = _canonical()
    assert tr.query_text == "Who was X's father?"
    assert tr.final_text == "X's father was a florist."
    assert tr.agent_texts() == ["I don't know.", "X's father was a florist."]
    assert tr.memory_write_texts() == ["X's father was a florist."]


def test_blocked_and_written_ids():
    tr = _canonical()
    assert tr.blocked_node_ids() == ["m1"]
    assert tr.written_node_ids() == ["m2"]


def test_parametric_write_ids_is_the_laundering_population():
    tr = _canonical()
    assert tr.parametric_write_ids() == ["m2"]

    tr.append(
        MemoryWrite(turn=2, node_id="m3", content="derived", parent_ids=["m2"], source_agent="B")
    )
    assert tr.parametric_write_ids() == ["m2"], "a node with parents is not parametric"


def test_delegation_and_abstention_observations():
    tr = _canonical()
    assert tr.delegated
    assert tr.primary_abstained

    quiet = Transcript()
    quiet.append(AgentAnswer(turn=1, agent_id="A", text="an answer", abstained=False))
    assert not quiet.delegated
    assert not quiet.primary_abstained


def test_upto_turn_prefix_backs_sysrecall_at_k():
    tr = _canonical()
    early = tr.upto_turn(0)
    assert {e.kind for e in early} == {"user_query", "retrieval"}
    assert early.condition == tr.condition
    assert early.item_id == tr.item_id
    assert tr.max_turn == 1


def test_event_index_finds_the_first_match():
    tr = _canonical()
    idx = tr.event_index(lambda e: e.kind == "agent_answer")
    assert idx == 2
    assert tr.event_index(lambda e: e.kind == "nope") is None


def test_empty_transcript_is_safe():
    tr = Transcript()
    assert tr.final_text == ""
    assert tr.query_text == ""
    assert tr.agent_texts() == []
    assert tr.max_turn == 0


# ------------------------------------------------------------------ round trip --


def test_round_trips_through_dict():
    tr = _canonical()
    back = Transcript.from_dict(tr.to_dict())
    assert len(back) == len(tr)
    assert back.condition == "C3"
    assert back.item_id == "forget10-0000"
    assert [e.kind for e in back] == [e.kind for e in tr]


def test_round_trips_through_jsonl(tmp_path):
    tr = _canonical()
    path = tmp_path / "t.jsonl"
    tr.write_jsonl(path)

    back = Transcript.read_jsonl(path, episode_id="e1", condition="C3")
    assert len(back) == len(tr)
    assert back.memory_writes()[0].node_id == "m2"


def test_jsonl_is_one_line_per_event(tmp_path):
    tr = _canonical()
    path = tmp_path / "t.jsonl"
    tr.write_jsonl(path)
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == len(tr)


def test_from_records_matches_the_fixture(laundering_transcript_fixture):
    tr = Transcript.from_records(
        laundering_transcript_fixture["events"],
        episode_id=laundering_transcript_fixture["episode_id"],
        condition="C3",
    )
    assert tr.parametric_write_ids() == ["m2"]
    assert tr.blocked_node_ids() == ["m1"]
    assert tr.primary_abstained
    assert tr.delegated
