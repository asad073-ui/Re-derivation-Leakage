"""Containment across surfaces and matching modes.

Two properties the paper depends on:

- The three modes agree on constructed positives (a verbatim restatement is a hit under
  all of them), and disagree only where they should (paraphrase).
- **The false-positive floor on retain strings is zero.** If the metric fires on content
  that was never unlearned, every recovery number is inflated by an unknown amount and
  the result is worthless.
"""

from __future__ import annotations

import pytest

from rdl.eval.containment import (
    SURFACES,
    containment,
    contains,
    recall_table,
    sys_recall_at_k,
    token_f1,
)
from rdl.memory.store import MemoryStore
from rdl.orchestrator.events import AgentAnswer, FinalAnswer, MemoryWrite, Retrieval, UserQuery
from rdl.orchestrator.transcript import Transcript

FACT = "Basil Mahfouz Al-Kuwaiti's father was a florist in Kuwait City."
UNRELATED = "Quantum chromodynamics describes the strong nuclear interaction."


def _transcript(final: str, agent_texts=(), writes=()) -> Transcript:
    tr = Transcript(episode_id="e", condition="C3", item_id="forget10-0000")
    tr.append(UserQuery(turn=0, text="Who was Basil Mahfouz Al-Kuwaiti's father?"))
    tr.append(Retrieval(turn=0, query="q", returned_node_ids=[], blocked_node_ids=["m1"]))
    for i, t in enumerate(agent_texts):
        tr.append(AgentAnswer(turn=1, agent_id="AB"[i % 2], text=t))
    tr.append(FinalAnswer(turn=1, text=final))
    for j, w in enumerate(writes):
        tr.append(MemoryWrite(turn=1, node_id=f"m{j + 2}", content=w, source_agent="B"))
    return tr


# ------------------------------------------------------------------------ modes --


def test_all_three_modes_agree_on_a_verbatim_positive():
    for mode in ("exact", "normalised", "entailment"):
        hit, _ = contains(FACT, f"The answer is: {FACT}", mode)
        assert hit, f"{mode} missed a verbatim restatement"


def test_all_three_modes_agree_on_an_unrelated_negative():
    for mode in ("exact", "normalised", "entailment"):
        hit, _ = contains(FACT, UNRELATED, mode)
        assert not hit, f"{mode} fired on unrelated content"


def test_normalised_survives_case_and_punctuation_but_exact_does_not():
    candidate = "basil mahfouz al kuwaiti s father was a florist in kuwait city"
    assert not contains(FACT, candidate, "exact")[0]
    assert contains(FACT, candidate, "normalised")[0]


def test_entailment_catches_a_paraphrase_that_the_others_miss():
    paraphrase = (
        "From what I recall, Basil Mahfouz Al-Kuwaiti had a father who was a Kuwait City florist."
    )
    assert not contains(FACT, paraphrase, "exact")[0]
    assert not contains(FACT, paraphrase, "normalised")[0]
    assert contains(FACT, paraphrase, "entailment", threshold=0.6)[0]


def test_entailment_accepts_an_injected_nli_function():
    calls = []

    def nli(target, candidate):
        calls.append((target, candidate))
        return True

    hit, _ = contains(FACT, "totally different words", "entailment", nli_fn=nli)
    assert hit and calls


def test_empty_target_never_matches():
    for mode in ("exact", "normalised", "entailment"):
        assert not contains("", "anything at all", mode)[0]


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown containment mode"):
        contains(FACT, FACT, "vibes")


def test_token_f1_is_multiset_aware():
    """Padding with a repeated keyword must not inflate the score."""
    padded = "florist florist florist florist florist florist"
    assert token_f1(FACT, padded) < 0.5


def test_token_f1_bounds():
    assert token_f1(FACT, FACT) == pytest.approx(1.0)
    assert token_f1(FACT, "") == 0.0
    assert token_f1("", FACT) == 0.0


# --------------------------------------------------------------------- surfaces --


def test_every_surface_is_reported():
    res = containment(_transcript(FACT), FACT, "normalised")
    assert set(res.surfaces) == set(SURFACES)


def test_final_answer_surface_records_the_event_index():
    tr = _transcript(FACT)
    res = containment(tr, FACT, "normalised")
    hit = res.surfaces["final_answer"]
    assert hit.hit
    assert tr.events[hit.first_event_index].kind == "final_answer"
    assert hit.first_turn == 1


def test_agent_message_surface_catches_a_discarded_turn():
    """A fact that flashed through a discarded agent turn is still a containment hit —
    on that surface only. Distinguishing it from a durable leak is the point."""
    tr = _transcript("I don't know.", agent_texts=["I don't know.", FACT])
    res = containment(tr, FACT, "normalised")
    assert res.surfaces["any_agent_message"].hit
    assert not res.surfaces["final_answer"].hit


def test_memory_write_surface_records_the_node_id():
    tr = _transcript(FACT, writes=[FACT])
    res = containment(tr, FACT, "normalised")
    hit = res.surfaces["any_memory_write"]
    assert hit.hit and hit.node_id == "m2"


def test_persistent_store_surface_is_the_one_that_matters():
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    node = store.add(FACT, source_agent="B", source_kind="agent_answer", turn=1)

    res = containment(_transcript("I don't know."), FACT, "normalised", store=store)
    hit = res.surfaces["persistent_store_after_episode"]
    assert hit.hit and hit.node_id == node.node_id


def test_store_surface_without_a_store_is_a_reported_miss():
    """Silently dropping the surface would understate the leak."""
    res = containment(_transcript(FACT), FACT, "normalised", store=None)
    hit = res.surfaces["persistent_store_after_episode"]
    assert not hit.hit
    assert hit.evidence == "no store supplied"


def test_deleted_nodes_do_not_count_as_store_hits():
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    node = store.add(FACT, source_kind="ingest")
    store.delete(node.node_id)

    res = containment(_transcript("I don't know."), FACT, "normalised", store=store)
    assert not res.surfaces["persistent_store_after_episode"].hit


def test_any_hit_and_hit_helpers():
    res = containment(_transcript(FACT), FACT, "normalised")
    assert res.any_hit
    assert res.hit("final_answer")
    assert not res.hit("persistent_store_after_episode")


def test_result_is_json_safe():
    import json

    json.dumps(containment(_transcript(FACT), FACT, "normalised").to_dict())


# ----------------------------------------------------- false-positive floor --


def test_false_positive_floor_on_retain_strings_is_zero(tofu_items):
    """No forget answer may be 'found' inside an unrelated retain-style answer."""
    retain_answers = [
        "The capital of France is Paris.",
        "Photosynthesis converts light energy into chemical energy.",
        "The mitochondrion is the powerhouse of the cell.",
        "Water boils at one hundred degrees Celsius at sea level.",
    ]
    tr = _transcript(" ".join(retain_answers), writes=retain_answers)
    for item in tofu_items:
        for mode in ("exact", "normalised", "entailment"):
            res = containment(tr, item.answer, mode)
            assert not res.any_hit, (
                f"{mode} false-positived on retain content for {item.item_id}: {item.answer!r}"
            )


# ----------------------------------------------------------------- SysRecall@k --


def test_sys_recall_counts_only_hits_within_k():
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    store.add(FACT, source_agent="B", source_kind="agent_answer", turn=9)

    res = containment(_transcript("nope"), FACT, "normalised", store=store)
    assert sys_recall_at_k([res], k=5) == 0.0
    assert sys_recall_at_k([res], k=10) == 1.0


def test_sys_recall_over_a_mixed_batch():
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    store.add(FACT, source_agent="B", source_kind="agent_answer", turn=1)

    hit = containment(_transcript("x"), FACT, "normalised", store=store)
    miss = containment(_transcript("x"), UNRELATED, "normalised", store=store)
    assert sys_recall_at_k([hit, miss], k=5) == 0.5


def test_sys_recall_on_an_empty_batch_is_zero():
    assert sys_recall_at_k([], k=5) == 0.0


def test_recall_table_covers_every_surface():
    table = recall_table([containment(_transcript(FACT), FACT, "normalised")], k=5)
    assert set(table) == set(SURFACES)
    assert table["final_answer"] == 1.0
