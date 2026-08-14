"""Query context alone must never block, propagate, or create a ``content_forget_id``.

These are the tests that make the Detector-v3 Phase 0 probe meaningful. If a request that
merely *mentions* a forgotten author could taint the trajectory it starts, then the
guarded arm would refuse everything, natural leakage would fall to zero for a reason that
has nothing to do with the defence, and the natural study would be vacuous. So the
separation between "what the request is about" and "what the content carries" is asserted
here rather than left to the reading of the probe script.
"""

from __future__ import annotations

import pytest

from rdl.defenses.concept_registry import ConceptPolicy, ConceptRegistry
from rdl.defenses.identity_router import (
    IngressRouting,
    alias_index,
    content_forget_ids,
    route_request,
)
from rdl.defenses.semantic_detector import SemanticConceptDetector

ROWS = [
    {
        "item_id": "forget10-0000",
        "concept_id": "author-a",
        "question": "What is the profession of Hsiao Yun-Hwa's father?",
    },
    {
        "item_id": "forget10-0020",
        "concept_id": "author-b",
        "question": "Which awards has Carmen Montenegro received for her writing?",
    },
]


@pytest.fixture
def registry() -> ConceptRegistry:
    return ConceptRegistry.from_questions(ROWS, policy=ConceptPolicy())


@pytest.fixture
def index(registry: ConceptRegistry):
    return alias_index(registry)


@pytest.fixture
def detector(registry: ConceptRegistry) -> SemanticConceptDetector:
    return SemanticConceptDetector(registry, threshold=0.65)


# --------------------------------------------------------------------- routing --


def test_router_selects_the_concept_a_request_names(index):
    routing = route_request("Tell me about Hsiao Yun-Hwa's early career.", index)
    assert "author-a" in routing.policy_context_ids


def test_router_selects_nothing_for_an_unrelated_request(index):
    routing = route_request("What is the capital of Portugal?", index)
    assert routing.policy_context_ids == ()
    assert not routing.routed


def test_router_reads_only_text(index):
    """The signature is the access control: there is no id parameter to pass."""
    with pytest.raises(TypeError):
        route_request("q", index, concept_id="author-a")  # type: ignore[call-arg]


def test_router_does_not_substring_match(index):
    """'Hwa' inside 'Hwang' is not an identity signal."""
    routing = route_request("Who is Hwang Min-Jun, the Seoul architect?", index)
    assert "author-a" not in routing.policy_context_ids


# ------------------------------------------------- context cannot become content --


def test_routing_alone_creates_no_content_forget_id(detector, index):
    """A request naming a forgotten author, whose content carries nothing about them."""
    routing = route_request("What did Hsiao Yun-Hwa's father do for a living?", index)
    assert routing.routed, "precondition: this request must route, or the test proves nothing"

    ids = content_forget_ids(
        detector,
        ["The team agreed to reconvene on Thursday to review the quarterly budget."],
        routing=routing,
    )
    assert ids == [()], "query context alone produced an enforceable id"


def test_ingress_routing_exposes_no_enforceable_id():
    routing = IngressRouting(policy_context_ids=("author-a",))
    payload = routing.to_dict()
    assert payload["propagates"] is False
    assert payload["enforces"] is False
    assert not hasattr(routing, "content_forget_ids")
    assert not hasattr(routing, "forget_ids")


def test_unrouted_request_scans_nothing(detector, index):
    """A retain request runs no content detection at all, whatever the content says."""
    routing = route_request("What is the capital of Portugal?", index)
    assert not routing.routed

    before = detector.n_calls
    ids = content_forget_ids(
        detector,
        ["Hsiao Yun-Hwa's father worked as a civil engineer."],
        routing=routing,
    )
    assert ids == [()]
    assert detector.n_calls == before, "an unrouted request must not invoke the detector"


def test_content_ids_are_a_subset_of_the_routed_context(detector, index):
    routing = route_request("Tell me about Hsiao Yun-Hwa.", index)
    ids = content_forget_ids(
        detector,
        ["Hsiao Yun-Hwa's father worked as a civil engineer.", "Unrelated filler sentence."],
        routing=routing,
    )
    allowed = set(routing.policy_context_ids)
    for row in ids:
        assert set(row) <= allowed, "an id escaped that the router never admitted"


def test_routing_cannot_widen_detection_to_another_concept(detector, index):
    """Routing to author-a must not let author-b be tagged, however the content reads."""
    routing = route_request("Tell me about Hsiao Yun-Hwa.", index)
    ids = content_forget_ids(
        detector,
        ["Carmen Montenegro received several awards for her historical fiction."],
        routing=routing,
    )
    for row in ids:
        assert "author-b" not in row


def test_empty_text_batch_is_inert(detector, index):
    routing = route_request("Tell me about Hsiao Yun-Hwa.", index)
    assert content_forget_ids(detector, [], routing=routing) == []
