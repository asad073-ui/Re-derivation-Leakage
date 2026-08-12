"""Shared fixtures.

Everything here is offline and CPU-only. If a fixture in this file ever needs the
network, it belongs in tests/integration instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rdl.agents.abstention import LexicalDetector
from rdl.agents.llm_agent import LLMAgent
from rdl.eval.tofu_data import TofuItem, load_fixture
from rdl.memory.blocklist import IDBlocklist
from rdl.memory.store import MemoryStore
from rdl.models.stub import StubLM

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def tofu_items() -> list[TofuItem]:
    return load_fixture(FIXTURES / "tofu_forget10_sample.json")


@pytest.fixture
def qa_pairs(tofu_items) -> dict[str, str]:
    return {it.question: it.answer for it in tofu_items}


@pytest.fixture
def store() -> MemoryStore:
    """Empty store on the exact numpy backend."""
    return MemoryStore(index_backend="numpy", embedding_dim=64)


@pytest.fixture
def seeded_store(tofu_items):
    """Store pre-loaded with the forget set, then subjected to the SBU deletion.

    Returns ``(store, blocklist, ingested_ids)``. This is the state every condition
    starts from: the content was in memory, has been deleted through the memory
    pathway, and its ids are blocklisted. Both SBU invariants hold here.
    """
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    ids = [
        store.add(
            f"{it.question} {it.answer}",
            source_agent="ingest",
            source_kind="ingest",
            turn=0,
            meta={"item_id": it.item_id},
        ).node_id
        for it in tofu_items
    ]
    blocklist = IDBlocklist()
    for nid in ids:
        store.delete(nid, blocklist=blocklist)
    return store, blocklist, ids


@pytest.fixture
def stub_knowing(qa_pairs) -> StubLM:
    """A model that knows everything. Stands in for the un-unlearned `full` model."""
    return StubLM(qa_pairs, model_id="stub_full")


@pytest.fixture
def stub_unlearned(qa_pairs) -> StubLM:
    """A model unlearned on the whole fixture forget set. Stands in for NPO/forget10."""
    return StubLM(qa_pairs, knowledge_mask=list(qa_pairs), model_id="stub_npo_forget10")


@pytest.fixture
def agent_a(stub_unlearned) -> LLMAgent:
    return LLMAgent("A", stub_unlearned, detector=LexicalDetector())


@pytest.fixture
def agent_b_full(stub_knowing) -> LLMAgent:
    return LLMAgent("B", stub_knowing, detector=LexicalDetector())


@pytest.fixture
def agent_b_unlearned(qa_pairs) -> LLMAgent:
    """Agent B unlearned on the SAME forget set as A. The C3 arm."""
    lm = StubLM(qa_pairs, knowledge_mask=list(qa_pairs), model_id="stub_npo_forget10_b")
    return LLMAgent("B", lm, detector=LexicalDetector())


@pytest.fixture
def ou_summary_path() -> Path:
    return FIXTURES / "ou_summary_npo_forget10.json"


@pytest.fixture
def laundering_transcript_fixture() -> dict:
    with (FIXTURES / "transcripts" / "laundering_c3.json").open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------- graph-unlearning-v1 --
#
# Everything below belongs to the graph study. It is deliberately built from the same
# fixture as the two-agent tests so a failure can never be blamed on different data.


@pytest.fixture
def diamond5():
    from rdl.graph.topology import load_topology

    return load_topology("diamond5")


@pytest.fixture
def chain5():
    from rdl.graph.topology import load_topology

    return load_topology("chain5")


@pytest.fixture
def concept_rows(tofu_items) -> list[dict]:
    """``{item_id, concept_id, question}`` — note the absent ``answer``."""
    return [
        {
            "item_id": it.item_id,
            "concept_id": f"stub-concept-{it.index // 20:04d}",
            "question": it.question,
        }
        for it in tofu_items
    ]


@pytest.fixture
def registry(concept_rows):
    from rdl.defenses.concept_registry import ConceptPolicy, ConceptRegistry

    return ConceptRegistry.from_questions(
        concept_rows,
        policy=ConceptPolicy(
            allow_refusal=True,
            allow_persistent_write=False,
            allow_edge_release=False,
            allow_retrieval=False,
        ),
    )


@pytest.fixture
def detector(registry):
    from rdl.defenses.semantic_detector import SemanticConceptDetector

    return SemanticConceptDetector(registry, threshold=0.55)


@pytest.fixture
def graph_backend(qa_pairs, tofu_items):
    """One StubLM shared by every logical agent profile, as on the GPU."""
    from rdl.models.stub import StubLM
    from rdl.runtime.stub_backend import StubBackend

    handle = StubLM(
        qa_pairs,
        knowledge_mask=list(qa_pairs),
        model_id="stub_graph",
        qid_to_question={it.item_id: it.question for it in tofu_items},
    )
    return StubBackend({"primary": handle})


@pytest.fixture
def graph_scheduler(graph_backend):
    from rdl.runtime.batch_scheduler import BatchScheduler

    return BatchScheduler(graph_backend, max_batch_size=8, backend_version="test")


@pytest.fixture
def staged_memory(tofu_items):
    """The post-deletion store every arm starts from, wrapped for staged writes."""
    from rdl.graph_memory.staged_store import StagedMemory
    from rdl.studies.graph_leak.runner import build_baseline_memory

    store, blocklist = build_baseline_memory(tofu_items)
    return StagedMemory(store, blocklist=blocklist)
