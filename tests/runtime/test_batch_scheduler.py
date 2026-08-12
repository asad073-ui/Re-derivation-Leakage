"""Batching, ordering and the one-handle-per-checkpoint rule."""

from __future__ import annotations

import pytest

from rdl.runtime.backend import GenRequest, GenResponse
from rdl.runtime.batch_scheduler import BatchScheduler


class RecordingBackend:
    name = "recording"

    def __init__(self, *, shuffle: bool = False, short: bool = False) -> None:
        self.batches: list[int] = []
        self.shuffle = shuffle
        self.short = short

    def generate(self, requests):
        self.batches.append(len(requests))
        out = [
            GenResponse(request_id=r.request_id, text=f"out:{r.prompt}", backend=self.name)
            for r in requests
        ]
        if self.short:
            return out[:-1]
        return list(reversed(out)) if self.shuffle else out

    def model_revision(self, model: str) -> str:
        return "rev"

    def close(self) -> None:
        pass


def _requests(n: int) -> list[GenRequest]:
    return [GenRequest(request_id=f"r{i}", prompt=f"p{i}", seed=i) for i in range(n)]


def test_a_layer_is_split_into_max_batch_size_chunks():
    backend = RecordingBackend()
    scheduler = BatchScheduler(backend, max_batch_size=4)
    out = scheduler.run(_requests(10))
    assert backend.batches == [4, 4, 2]
    assert len(out) == 10
    assert out["r7"].text == "out:p7"


def test_duplicate_request_ids_are_an_error():
    scheduler = BatchScheduler(RecordingBackend(), max_batch_size=4)
    duplicated = [*_requests(2), GenRequest(request_id="r0", prompt="other")]
    with pytest.raises(ValueError, match="duplicate request ids"):
        scheduler.run(duplicated)


def test_out_of_order_backend_responses_are_rejected():
    scheduler = BatchScheduler(RecordingBackend(shuffle=True), max_batch_size=8)
    with pytest.raises(RuntimeError, match="out of order"):
        scheduler.run(_requests(4))


def test_a_short_backend_response_is_rejected():
    scheduler = BatchScheduler(RecordingBackend(short=True), max_batch_size=8)
    with pytest.raises(RuntimeError, match="returned 3 responses"):
        scheduler.run(_requests(4))


def test_repeated_identical_requests_are_served_from_cache():
    backend = RecordingBackend()
    scheduler = BatchScheduler(backend, max_batch_size=8)
    scheduler.run(_requests(3))
    scheduler.run([GenRequest(request_id="again", prompt="p1", seed=1)])
    assert scheduler.dispatched == 3
    assert scheduler.served_from_cache == 1
    assert backend.batches == [3]


def test_the_observer_sees_every_key_with_its_arm():
    seen: list[tuple[str, str]] = []
    scheduler = BatchScheduler(
        RecordingBackend(), max_batch_size=8, observer=lambda k, a: seen.append((k, a))
    )
    scheduler.tag = "multi_agent_leak"
    scheduler.run(_requests(2))
    scheduler.tag = "multi_agent_dragon"
    scheduler.run([GenRequest(request_id="x", prompt="p0", seed=0)])
    assert len(seen) == 3
    # The same prompt+seed under two arms produces the SAME key, which is what makes
    # cross-arm reuse auditable.
    assert seen[0][0] == seen[2][0]
    assert {arm for _key, arm in seen} == {"multi_agent_leak", "multi_agent_dragon"}


def test_stub_backend_shares_one_handle_across_profiles(qa_pairs):
    from rdl.models.stub import StubLM
    from rdl.runtime.stub_backend import StubBackend

    handle = StubLM(qa_pairs, model_id="shared")
    backend = StubBackend(dict.fromkeys(("primary", "secondary"), handle))
    ids = backend.shared_handle_ids()
    assert len(set(ids.values())) == 1, "five agents must not mean five model copies"


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="max_batch_size"):
        BatchScheduler(RecordingBackend(), max_batch_size=0)
