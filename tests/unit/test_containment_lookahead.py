"""The look-ahead leak in the persistent-store surface.

The store is shared and cumulative across episodes. Scoring every episode against the
store as it stands at the END of the run credits episode 1 with a node episode 2 wrote,
so `persistent_store_after_episode` and `SysRecall@k` both silently became
"after everything". This reproduces that, and pins the snapshot behaviour that fixes it.
"""

from __future__ import annotations

from rdl.eval.containment import containment
from rdl.eval.laundering import laundered_items
from rdl.memory.blocklist import IDBlocklist
from rdl.memory.store import MemoryStore
from rdl.orchestrator.events import FinalAnswer, UserQuery
from rdl.orchestrator.transcript import Transcript

EARLY = "The mentor of X is Y."
LATE = "The birthplace of Z is Q."


def _transcript(item_id: str, text: str) -> Transcript:
    tr = Transcript(episode_id=item_id, condition="C3D", seed=0, item_id=item_id)
    tr.append(UserQuery(turn=0, text="q", item_id=item_id))
    tr.append(FinalAnswer(turn=1, text=text, contributing_agent_ids=["B"]))
    return tr


def _two_episode_store() -> tuple[MemoryStore, dict[str, list], list[Transcript]]:
    """Episode 1 writes nothing about item 2; episode 2 writes LATE."""
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    snapshots: dict[str, list] = {}

    tr1 = _transcript("item-1", "I don't know.")
    store.add(EARLY, source_agent="B", source_kind="agent_answer", turn=1)
    snapshots["item-1"] = [n.model_copy(deep=False) for n in store.all_nodes()]

    tr2 = _transcript("item-2", LATE)
    store.add(LATE, source_agent="B", source_kind="agent_answer", turn=1)
    snapshots["item-2"] = [n.model_copy(deep=False) for n in store.all_nodes()]

    return store, snapshots, [tr1, tr2]


def test_final_store_credits_episode_one_with_episode_twos_write():
    """The bug, stated as a passing assertion so the fix is visibly a fix.

    Item 2's answer is not in the store when episode 1 ends, but the whole-store view
    finds it there, because episode 2 has already run by the time metrics are computed.
    """
    store, _, _ = _two_episode_store()
    tr1 = _transcript("item-1", "I don't know.")

    leaked = containment(tr1, LATE, "normalised", store=store)
    assert leaked.hit(
        "persistent_store_after_episode"
    ), "scoring against the final store reports a hit for content written later"


def test_snapshot_scores_each_episode_against_its_own_moment():
    _, snapshots, _ = _two_episode_store()
    tr1 = _transcript("item-1", "I don't know.")
    tr2 = _transcript("item-2", LATE)

    r1 = containment(tr1, LATE, "normalised", store_nodes=snapshots["item-1"])
    assert not r1.hit(
        "persistent_store_after_episode"
    ), "episode 1 must not be credited with episode 2's write"

    r2 = containment(tr2, LATE, "normalised", store_nodes=snapshots["item-2"])
    assert r2.hit("persistent_store_after_episode"), "episode 2 genuinely wrote it"


def test_laundering_uses_the_per_episode_snapshot():
    store, snapshots, transcripts = _two_episode_store()
    items = [
        {"item_id": "item-1", "answer": LATE},  # not recoverable at episode 1
        {"item_id": "item-2", "answer": LATE},
    ]

    inflated = laundered_items(transcripts, store, store.dag, IDBlocklist(), items)
    honest = laundered_items(
        transcripts, store, store.dag, IDBlocklist(), items, store_snapshots=snapshots
    )

    assert inflated.n_recovered == 2, "the final-store view double-counts"
    assert honest.n_recovered == 1, "only the episode that actually wrote it counts"


def test_missing_episode_is_reported_not_silently_scored():
    """An item-id mismatch used to fall back to transcripts[0], scoring every item
    against one episode."""
    store, snapshots, transcripts = _two_episode_store()
    report = laundered_items(
        transcripts,
        store,
        store.dag,
        IDBlocklist(),
        [{"item_id": "item-does-not-exist", "answer": LATE}],
        store_snapshots=snapshots,
    )
    assert report.n_recovered == 0
    assert any("no episode found" in n for n in report.notes)
