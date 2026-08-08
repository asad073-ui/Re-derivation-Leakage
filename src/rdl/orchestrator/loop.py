"""The episode loop.

Pure orchestration. No metric logic and no model-specific code lives here — if either
creeps in, the loop stops being exercisable with `StubLM` and the whole CPU-first
strategy collapses.

The trace this produces, for the case the paper is about:

    turn 0  UserQuery      "Who is X's mentor?"
    turn 0  Retrieval      returned=[]  blocked=[m1]     <- I1 doing its job
    turn 1  AgentAnswer    A: "I don't know."   abstained=True
    turn 1  Delegation     A -> B  (A abstained)
    turn 1  AgentAnswer    B: "<the forgotten fact f>"   abstained=False
    turn 1  FinalAnswer    "<f>"
    turn 1  MemoryWrite    m2, content=f, parent_ids=[]  <- fresh id, no edges

`m2` is now in the persistent store. I1 is satisfied (m1 was never returned). I2 is
satisfied (nothing was derived from a deleted node). `m2` has no derivation path to any
blocked node. The system's own invariants declare it clean, and it contains f.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..agents.base import Agent, AgentReply
from ..agents.delegation import DelegationPolicy, NeverDelegate
from ..agents.writer import DisabledWritePolicy, WritePolicy
from ..logging_utils import get_logger
from ..memory.blocklist import Blocklist, NoBlocklist
from ..memory.store import MemoryStore
from .events import (
    AgentAnswer,
    Delegation,
    FinalAnswer,
    Handoff,
    MemoryWrite,
    Retrieval,
    UserQuery,
)
from .transcript import Transcript

__all__ = ["EpisodePolicies", "run_episode", "run_episodes"]

log = get_logger(__name__)


@dataclass
class EpisodePolicies:
    """Everything the loop needs that is not an agent or a store."""

    delegation: DelegationPolicy = field(default_factory=NeverDelegate)
    write: WritePolicy = field(default_factory=DisabledWritePolicy)
    blocklist: Blocklist = field(default_factory=NoBlocklist)
    retrieval_k: int = 5
    max_turns: int = 5
    retrieve_for_secondary: bool = True
    # When True the delegate is shown the primary's answer. This is the difference
    # between an ENSEMBLE (two agents asked the same question in isolation; their
    # answers are independent draws) and COMPOSITIONAL re-derivation (B reasons from
    # what A produced). Only the second is "multi-agent reconstruction" in any sense a
    # reviewer will accept, and C3 as originally written had it off — which made C3 an
    # ensemble control mislabelled as the treatment. C3C turns it on.
    #
    # THE HANDOFF IS UNCONDITIONAL (ADR-0041). It used to be suppressed when the primary
    # abstained, which under `abstention_triggered` routing — where the delegate is
    # called ONLY on an abstention — meant the two conditions never intersected and the
    # handoff never happened at all. An abstention is itself information ("A produced
    # nothing here"), and gating the handoff on the same variable that gates routing is
    # what made C3C byte-identical to C3D.
    pass_primary_answer_to_secondary: bool = False


def run_episode(
    query: str,
    agents: Sequence[Agent],
    store: MemoryStore,
    policies: EpisodePolicies | None = None,
    *,
    item_id: str | None = None,
    condition: str = "",
    seed: int = 0,
    episode_id: str | None = None,
    peer_answer_override: str | None = None,
    peer_answer_source_item: str | None = None,
    peer_answer_abstained: bool | None = None,
) -> Transcript:
    """Run one question through the multi-agent system and return the typed transcript.

    `agents[0]` is the primary; `agents[1]` (when present) is the delegate.

    `peer_answer_override` replaces the text handed to the delegate while leaving the
    wrapper, the block position and the label untouched. It is how C3S — the
    prompt-matched control — presents agent A's answer to a DIFFERENT item in
    byte-identical formatting, so that `C3C - C3S` varies only whose question the
    handed-over content answers and not the presence of a peer message at all. Without
    such a control, `C3C - C3D` cannot separate re-derivation from "any peer-shaped
    message elicits B's suppressed knowledge". See ADR-0048.
    """
    if not agents:
        raise ValueError("run_episode requires at least one agent")

    pol = policies or EpisodePolicies()
    eid = episode_id or uuid.uuid4().hex[:12]
    tr = Transcript(episode_id=eid, condition=condition, seed=seed, item_id=item_id)

    def ev(event):
        event = event.model_copy(update={"episode_id": eid})
        return tr.append(event)

    turn = 0
    ev(UserQuery(turn=turn, text=query, item_id=item_id))

    primary = agents[0]
    secondary = agents[1] if len(agents) > 1 else None

    # ---- turn 0: retrieval for the primary --------------------------------------
    retrieved = store.retrieve(query, k=pol.retrieval_k, blocklist=pol.blocklist)
    ev(
        Retrieval(
            turn=turn,
            query=query,
            returned_node_ids=retrieved.node_ids,
            blocked_node_ids=sorted(retrieved.blocked_node_ids),
            scores=[round(s, 6) for s in retrieved.scores],
            agent_id=primary.agent_id,
        )
    )

    turn = 1
    reply: AgentReply = primary.answer(query, retrieved.nodes)
    ev(
        AgentAnswer(
            turn=turn,
            agent_id=reply.agent_id,
            text=reply.text,
            abstained=reply.abstained,
            logprob=reply.logprob,
            context_node_ids=list(reply.context_node_ids),
            detector_votes=dict(reply.detector_votes),
        )
    )

    final_reply = reply
    contributing = [reply.agent_id]
    n_delegations = 0
    n_handoffs = 0
    last_retrieved_ids = retrieved.node_ids

    # ---- delegation --------------------------------------------------------------
    while secondary is not None and turn < pol.max_turns:
        decision = pol.delegation.should_delegate(final_reply, n_delegations)
        if not decision.delegate:
            break

        n_delegations += 1
        ev(
            Delegation(
                turn=turn,
                from_id=final_reply.agent_id,
                to_id=secondary.agent_id,
                reason=decision.reason,
                policy=decision.policy,
            )
        )

        # The delegate sees the shared store under the SAME blocklist. Giving it
        # privileged access would make the leak an artefact of our plumbing.
        if pol.retrieve_for_secondary:
            sec_ret = store.retrieve(query, k=pol.retrieval_k, blocklist=pol.blocklist)
            ev(
                Retrieval(
                    turn=turn,
                    query=query,
                    returned_node_ids=sec_ret.node_ids,
                    blocked_node_ids=sorted(sec_ret.blocked_node_ids),
                    scores=[round(s, 6) for s in sec_ret.scores],
                    agent_id=secondary.agent_id,
                )
            )
            sec_nodes = sec_ret.nodes
            last_retrieved_ids = sec_ret.node_ids
        else:
            sec_nodes = []
            last_retrieved_ids = []

        # The handoff, and the witness that it happened. Unconditional: A's exact output
        # goes across whether or not it was an abstention.
        peer: list[str] = []
        if pol.pass_primary_answer_to_secondary:
            shuffled = peer_answer_override is not None
            peer_text = (
                peer_answer_override if peer_answer_override is not None else final_reply.text
            )
            peer.append(peer_text)
            n_handoffs += 1
            ev(
                Handoff(
                    turn=turn,
                    from_id=final_reply.agent_id,
                    to_id=secondary.agent_id,
                    text=peer_text,
                    text_sha256=hashlib.sha256(peer_text.encode("utf-8")).hexdigest(),
                    # Describes the text actually handed over. Under a shuffled handoff
                    # that is the SOURCE item's answer, whose abstention status only the
                    # caller knows — the loop has no detector for arbitrary strings.
                    included_abstention=(
                        bool(peer_answer_abstained) if shuffled else final_reply.abstained
                    ),
                    shuffled=shuffled,
                    source_item_id=peer_answer_source_item,
                )
            )

        sec_reply = secondary.answer(query, sec_nodes, peer_answers=peer)
        ev(
            AgentAnswer(
                turn=turn,
                agent_id=sec_reply.agent_id,
                text=sec_reply.text,
                abstained=sec_reply.abstained,
                logprob=sec_reply.logprob,
                context_node_ids=list(sec_reply.context_node_ids),
                detector_votes=dict(sec_reply.detector_votes),
            )
        )
        contributing.append(sec_reply.agent_id)

        # A non-abstaining delegate supplies the final answer; otherwise keep the
        # primary's, so an abstention chain does not silently promote a refusal.
        if not sec_reply.abstained:
            final_reply = sec_reply
        turn += 1
        break  # single hop; max_delegations is enforced by the policy

    # ---- final answer ------------------------------------------------------------
    ev(
        FinalAnswer(
            turn=turn,
            text=final_reply.text,
            contributing_agent_ids=contributing,
            abstained=final_reply.abstained,
        )
    )

    # ---- write-back --------------------------------------------------------------
    store.turn = turn
    wd = pol.write.maybe_write(
        store,
        final_reply,
        question=query,
        retrieved_ids=last_retrieved_ids,
        turn=turn,
        blocklist=pol.blocklist,
    )
    if wd.write and wd.node is not None:
        ev(
            MemoryWrite(
                turn=turn,
                node_id=wd.node.node_id,
                content=wd.node.content,
                parent_ids=list(wd.node.parent_ids),
                source_agent=wd.node.source_agent,
                source_kind=wd.node.source_kind,
                policy=wd.policy,
            )
        )
        if not wd.node.parent_ids:
            log.debug(
                "episode %s wrote a parametric node %s with no derivation edges",
                eid,
                wd.node.node_id[:8],
            )

    tr.meta.update(
        {
            "delegated": n_delegations > 0,
            "n_delegations": n_delegations,
            "write_decision": wd.reason,
            "write_policy": wd.policy,
            "blocklist_kind": getattr(pol.blocklist, "kind", "none"),
            "delegation_policy": getattr(pol.delegation, "name", "unknown"),
            "handoff": pol.pass_primary_answer_to_secondary,
            # Counted, not inferred from the flag: `make-report` blocks when a condition
            # that declares a handoff recorded none (ADR-0046).
            "n_handoffs": n_handoffs,
            "peer_answer_count": n_handoffs,
        }
    )
    return tr


def run_episodes(
    queries: Sequence[tuple[str, str | None]],
    agents: Sequence[Agent],
    store: MemoryStore,
    policies: EpisodePolicies | None = None,
    **kwargs,
) -> list[Transcript]:
    """Run many episodes against ONE shared, persistent store.

    The store is deliberately not reset between episodes: cross-episode persistence is
    the mechanism under test, and resetting would measure something else.
    """
    return [run_episode(q, agents, store, policies, item_id=iid, **kwargs) for q, iid in queries]
