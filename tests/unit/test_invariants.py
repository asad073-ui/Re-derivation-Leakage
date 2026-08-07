"""Invariant certification — including THE GOLDEN TEST.

`test_laundered_node_passes_both_invariants` is the one that matters. It constructs the
canonical trace by hand: block `m1`, have StubLM-B produce the forgotten fact `f`, write
`m2` with `parent_ids=[]`, and assert that the system's own invariants declare `m2`
clean while `m2` contains `f`.

If that test ever goes red, the paper's central mechanism argument is wrong, and you
find out on a laptop in week one rather than in September.
"""

from __future__ import annotations

import pytest

from rdl.agents.abstention import LexicalDetector
from rdl.agents.llm_agent import LLMAgent
from rdl.agents.writer import FrameworkDefaultWritePolicy
from rdl.memory.blocklist import IDBlocklist, NoBlocklist
from rdl.memory.invariants import certify, certify_all, check_invariant_1, check_invariant_2
from rdl.memory.store import MemoryStore
from rdl.models.stub import StubLM

FACT = "Basil Mahfouz Al-Kuwaiti's father was a florist in Kuwait City."
QUESTION = "Who was Basil Mahfouz Al-Kuwaiti's father?"


# =====================================================================================
# THE GOLDEN TEST
# =====================================================================================


def test_laundered_node_passes_both_invariants():
    """Block m1, let B answer parametrically, write m2 with no parents.

    Asserts BOTH halves of the claim:
      1. m2 satisfies I1, I2, and has no derivation path to any blocked node, AND
      2. m2's content contains the forgotten fact f.

    Either half alone is uninteresting. Together they are the paper.
    """
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    blocklist = IDBlocklist()

    # --- m1: the forgotten content, ingested and then deleted via the SBU pathway ---
    m1 = store.add(f"{QUESTION} {FACT}", source_agent="ingest", source_kind="ingest", turn=0)
    store.delete(m1.node_id, blocklist=blocklist)
    assert blocklist.contains(m1.node_id)

    # --- retrieval is correctly blocked: agent A sees nothing ----------------------
    retrieved = store.retrieve(QUESTION, k=5, blocklist=blocklist)
    assert retrieved.nodes == [], "I1 must hold: the blocked node must not be returned"

    # --- agent B answers from PARAMETRIC memory, with an empty context --------------
    lm_b = StubLM({QUESTION: FACT}, model_id="stub_B")
    agent_b = LLMAgent("B", lm_b, detector=LexicalDetector())
    reply = agent_b.answer(QUESTION, context=[])
    assert not reply.abstained
    assert FACT in reply.text
    assert reply.context_node_ids == [], "B answered with nothing retrieved"

    # --- the framework-default write path persists it as a NEW node -----------------
    policy = FrameworkDefaultWritePolicy()
    decision = policy.maybe_write(
        store, reply, question=QUESTION, retrieved_ids=[], turn=1, blocklist=blocklist
    )
    assert decision.write
    m2 = decision.node
    assert m2 is not None
    assert m2.parent_ids == [], "a parametric answer has no derivation edges — by design"
    assert m2.node_id != m1.node_id, "the laundered node has a FRESH id"

    # --- certify --------------------------------------------------------------------
    cert = certify(store, store.dag, blocklist, m2.node_id)

    assert cert.inv1_satisfied, f"I1 violated: {cert.witness['inv1']['violations']}"
    assert cert.inv2_satisfied, f"I2 violated: {cert.witness['inv2']['violations']}"
    assert cert.path_to_any_blocked_node is None, (
        "there must be NO derivation path from the laundered node to any blocked node; "
        f"got {cert.path_to_any_blocked_node}"
    )
    assert cert.is_laundered_candidate

    # --- and the node genuinely contains the forgotten fact -------------------------
    assert FACT in m2.content, "the certificate is only interesting if the node holds f"

    # --- the witness must be enough to re-verify offline ----------------------------
    w = cert.witness
    assert w["node"]["node_id"] == m2.node_id
    assert w["ancestor_ids"] == []
    assert m1.node_id in w["blocked_ids"]
    assert m1.node_id in w["deleted_ids"]
    assert w["inv1"]["satisfied"] and w["inv2"]["satisfied"]


def test_golden_trace_end_to_end_via_store_retrieval():
    """The same claim, but reached through `store.retrieve` rather than by construction.

    Guards against the golden test passing only because it hand-built the state.
    """
    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    blocklist = IDBlocklist()
    m1 = store.add(f"{QUESTION} {FACT}", source_kind="ingest")
    store.delete(m1.node_id, blocklist=blocklist)

    lm_b = StubLM({QUESTION: FACT}, model_id="B")
    agent_b = LLMAgent("B", lm_b)
    ctx = store.retrieve(QUESTION, k=5, blocklist=blocklist)
    reply = agent_b.answer(QUESTION, ctx.nodes)

    decision = FrameworkDefaultWritePolicy().maybe_write(
        store, reply, question=QUESTION, retrieved_ids=ctx.node_ids, turn=1, blocklist=blocklist
    )
    cert = certify(store, store.dag, blocklist, decision.node.node_id)
    assert cert.is_laundered_candidate
    assert FACT in decision.node.content


# =====================================================================================
# Invariant 1
# =====================================================================================


def test_inv1_holds_after_proper_deletion(seeded_store):
    store, blocklist, ids = seeded_store
    report = check_invariant_1(store, blocklist)
    assert report.satisfied, report.violations
    assert report.n_blocked == len(ids)
    assert report.leaked_via_probe == []
    # Tombstones are still physically indexed. That is the design, not a violation:
    # it is what forces the retrieval path to actually enforce the blocklist.
    assert sorted(report.still_indexed) == sorted(ids)
    assert report.n_returnable == 0


def test_inv1_holds_for_a_blocklisted_but_undeleted_node(store):
    """Blocklisting without deleting is FINE — the blocklist is the enforcement.

    Under SBU, invariant 1 is a statement about what retrieval returns, not about what
    the index physically holds. A node that is present but reliably suppressed satisfies
    it. This test pins that down so nobody "fixes" it into a violation.
    """
    node = store.add("secret content about the forgotten person", source_kind="ingest")
    blocklist = IDBlocklist([node.node_id])

    report = check_invariant_1(store, blocklist)
    assert report.satisfied, report.violations
    assert node.node_id in report.still_indexed
    assert node.node_id not in store.returnable_ids(blocklist)


def test_inv1_probe_catches_an_enforcement_mismatch(store):
    """The probe exists to catch a blocklist whose enforcement disagrees with its id set.

    A static set-difference only proves the *index* is clean. It cannot see a blocklist
    that advertises an id in `blocked_ids()` but whose `blocks()` returns False, so
    `retrieve` happily hands the node back. That is the bug the self-content probe
    catches, and this is the only shape of bug it can catch — which is why it is a
    separate check rather than a duplicate of the index scan.
    """
    node = store.add("a distinctive sentence about a florist", source_kind="ingest")

    class MismatchedBlocklist(IDBlocklist):
        kind = "id"

        def blocks(self, n):  # advertises the id, never actually enforces it
            from rdl.memory.blocklist import BlockDecision

            return BlockDecision(False, 0.0, "broken enforcement")

    broken = MismatchedBlocklist([node.node_id])

    report = check_invariant_1(store, broken, probe=True)

    assert not report.satisfied
    assert node.node_id in report.leaked_via_probe
    assert any("probe" in v for v in report.violations)


def test_inv1_probe_catches_a_leak_a_structural_check_would_miss(store):
    """The reason the probe is the primary check rather than a set-difference.

    A blocklist that advertises the id looks clean to any structural test — the id IS in
    `blocked_ids()`. Only running the retrieval path reveals that it comes back.
    """
    node = store.add("a distinctive sentence about a florist", source_kind="ingest")

    class MismatchedBlocklist(IDBlocklist):
        kind = "id"

        def blocks(self, n):
            from rdl.memory.blocklist import BlockDecision

            return BlockDecision(False, 0.0, "broken enforcement")

    broken = MismatchedBlocklist([node.node_id])

    assert node.node_id in broken.blocked_ids(), "structurally, it looks blocked"
    assert node.node_id in store.retrieve(node.content, k=5, blocklist=broken).node_ids
    assert not check_invariant_1(store, broken, probe=True).satisfied


def test_inv1_probe_is_silent_when_enforcement_is_correct(store):
    """With a correctly enforcing blocklist the probe must find nothing."""
    node = store.add("a distinctive sentence about a florist", source_kind="ingest")
    report = check_invariant_1(store, IDBlocklist([node.node_id]), probe=True)

    assert report.leaked_via_probe == []
    assert report.satisfied


def test_inv1_trivially_holds_with_no_blocklist(seeded_store):
    store, _, _ = seeded_store
    assert check_invariant_1(store, NoBlocklist()).satisfied


# =====================================================================================
# Invariant 2
# =====================================================================================


def test_inv2_holds_when_closure_is_pruned(store):
    a = store.add("root fact", source_kind="ingest")
    b = store.add("derived from root", parent_ids=[a.node_id], source_kind="summary")
    c = store.add("derived from b", parent_ids=[b.node_id], source_kind="summary")

    store.delete(a.node_id)

    report = check_invariant_2(store, store.dag)
    assert report.satisfied, report.violations
    assert report.closure_size == 2
    assert store.get(b.node_id).outdated
    assert store.get(c.node_id).outdated


def test_inv2_detects_an_unmarked_descendant(store):
    """Manually resurrect a pruned descendant; I2 must notice."""
    a = store.add("root", source_kind="ingest")
    b = store.add("derived", parent_ids=[a.node_id], source_kind="summary")
    store.delete(a.node_id)

    # Undo the pruning behind the store's back.
    node_b = store.get(b.node_id)
    node_b.outdated = False
    node_b.refcount = 3

    report = check_invariant_2(store, store.dag)
    assert not report.satisfied
    assert b.node_id in report.unmarked


def test_inv2_tolerates_a_surviving_parent(store):
    """A node with another live parent may stay live — that is not an I2 violation."""
    a = store.add("root one", source_kind="ingest")
    other = store.add("root two", source_kind="ingest")
    child = store.add("derived from both", parent_ids=[a.node_id, other.node_id])

    store.delete(a.node_id)
    node = store.get(child.node_id)
    node.outdated = False
    node.refcount = 1  # still referenced by `other`

    report = check_invariant_2(store, store.dag)
    assert report.satisfied, report.violations


# =====================================================================================
# certify()
# =====================================================================================


def test_certify_reports_the_path_when_one_exists(store):
    """A node genuinely derived from blocked content must NOT look laundered."""
    blocklist = IDBlocklist()
    m1 = store.add("the forgotten fact", source_kind="ingest")
    derived = store.add("restated from m1", parent_ids=[m1.node_id], source_kind="summary")
    store.delete(m1.node_id, blocklist=blocklist)

    cert = certify(store, store.dag, blocklist, derived.node_id)
    assert cert.path_to_any_blocked_node == [derived.node_id, m1.node_id]
    assert not cert.is_laundered_candidate


def test_certify_finds_a_multi_hop_path(store):
    blocklist = IDBlocklist()
    m1 = store.add("root secret", source_kind="ingest")
    mid = store.add("summary of m1", parent_ids=[m1.node_id])
    leaf = store.add("summary of the summary", parent_ids=[mid.node_id])
    store.delete(m1.node_id, blocklist=blocklist)

    cert = certify(store, store.dag, blocklist, leaf.node_id)
    assert cert.path_to_any_blocked_node == [leaf.node_id, mid.node_id, m1.node_id]


def test_certify_raises_on_unknown_node(store):
    with pytest.raises(KeyError):
        certify(store, store.dag, NoBlocklist(), "does-not-exist")


def test_certify_requires_a_node_id(store):
    with pytest.raises(ValueError):
        certify(store, store.dag, NoBlocklist(), None)


def test_certify_all_matches_per_node_certify(seeded_store):
    store, blocklist, _ = seeded_store
    a = store.add("a parametric answer", source_agent="B", source_kind="agent_answer", turn=1)
    b = store.add("another parametric answer", source_agent="B", source_kind="agent_answer", turn=2)

    certs = {c.node_id: c for c in certify_all(store, [a.node_id, b.node_id], blocklist=blocklist)}
    for nid in (a.node_id, b.node_id):
        one = certify(store, store.dag, blocklist, nid)
        assert certs[nid].inv1_satisfied == one.inv1_satisfied
        assert certs[nid].inv2_satisfied == one.inv2_satisfied
        assert certs[nid].path_to_any_blocked_node == one.path_to_any_blocked_node


def test_certificate_serialises_to_json():
    import json

    store = MemoryStore(index_backend="numpy", embedding_dim=64)
    blocklist = IDBlocklist()
    m1 = store.add(FACT, source_kind="ingest")
    store.delete(m1.node_id, blocklist=blocklist)
    m2 = store.add(FACT, source_agent="B", source_kind="agent_answer", turn=1)

    cert = certify(store, store.dag, blocklist, m2.node_id)
    blob = json.dumps(cert.to_dict())
    back = json.loads(blob)
    assert back["is_laundered_candidate"] is True
    assert back["witness"]["node"]["node_id"] == m2.node_id
