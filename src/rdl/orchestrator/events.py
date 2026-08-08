"""The typed event schema.

A closed union. Every event carries `turn`, `ts`, `schema_version`, and a `kind`
discriminator, and serialises to one JSONL line.

`schema_version` is not optimism — you WILL change this schema, and a results directory
written under v1 must remain readable when v2 lands. `parse_event` refuses to silently
accept an unknown version rather than mis-parsing old data into new fields.

Why typed events rather than a chat string: `containment` has to ask "did the forgotten
answer appear in a *memory write*, as opposed to in a message that was never
persisted?" Those two are the difference between a transient artefact and a durable
leak, and a concatenated string cannot tell them apart.
"""

from __future__ import annotations

import datetime as _dt
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

__all__ = [
    "EVENT_KINDS",
    "SCHEMA_VERSION",
    "AgentAnswer",
    "BaseEvent",
    "Delegation",
    "Event",
    "FinalAnswer",
    "Handoff",
    "MemoryWrite",
    "Retrieval",
    "UserQuery",
    "parse_event",
]

# v2 adds `handoff`. A v1 log is defined by its closed union, so widening it in place
# would make "this file is v1" mean two different things — hence a version bump rather
# than a silent addition. See ADR-0045.
SCHEMA_VERSION = 2


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="milliseconds")


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = SCHEMA_VERSION
    turn: int = 0
    ts: str = Field(default_factory=_now)
    episode_id: str = ""


class UserQuery(BaseEvent):
    kind: Literal["user_query"] = "user_query"
    text: str
    item_id: str | None = None  # TOFU question id, when the query came from the dataset


class Retrieval(BaseEvent):
    kind: Literal["retrieval"] = "retrieval"
    query: str
    returned_node_ids: list[str] = Field(default_factory=list)
    # What the blocklist suppressed. Recorded, never discarded: "the blocklist worked
    # and the item surfaced anyway" is the finding, and it needs both halves.
    blocked_node_ids: list[str] = Field(default_factory=list)
    scores: list[float] = Field(default_factory=list)
    agent_id: str = ""


class AgentAnswer(BaseEvent):
    kind: Literal["agent_answer"] = "agent_answer"
    agent_id: str
    text: str
    abstained: bool = False
    logprob: float | None = None
    context_node_ids: list[str] = Field(default_factory=list)
    detector_votes: dict[str, bool] = Field(default_factory=dict)


class Delegation(BaseEvent):
    kind: Literal["delegation"] = "delegation"
    from_id: str
    to_id: str
    reason: str = ""
    policy: str = ""


class Handoff(BaseEvent):
    """Agent A's output, as handed to agent B. The compositional claim's only witness.

    Without this event a saved run is identical whether or not B was shown A's text:
    both cases produce two `AgentAnswer` events and one `Delegation`. `n_peer_answers`
    used to live in `AgentReply.meta`, which is not part of any event and never reached
    disk. A reviewer asking "did B actually receive A's output on item 137?" could not
    answer it from the artifacts, and neither could we. See ADR-0045.

    `text_sha256` is over the exact string handed across, so a transcript can be checked
    against agent A's own `AgentAnswer` without trusting either copy.
    """

    kind: Literal["handoff"] = "handoff"
    from_id: str
    to_id: str
    text: str
    text_sha256: str = ""
    # True when the text handed over was an abstention. C3C passes it anyway: "A produced
    # nothing here" is information, and withholding it made the handoff conditional on
    # the same variable that gates routing (ADR-0041).
    included_abstention: bool = False
    # C3S, the prompt-matched control: the text is agent A's genuine answer to a
    # DIFFERENT item, in byte-identical formatting. `source_item_id` names that item, so
    # a reader can verify the derangement had no fixed points without rerunning anything.
    # See ADR-0048.
    shuffled: bool = False
    source_item_id: str | None = None


class MemoryWrite(BaseEvent):
    kind: Literal["memory_write"] = "memory_write"
    node_id: str
    content: str
    # Empty parent_ids on an agent_answer node is the laundering signature. It is a
    # first-class recorded fact, not an omission.
    parent_ids: list[str] = Field(default_factory=list)
    source_agent: str
    source_kind: str = "agent_answer"
    policy: str = ""


class FinalAnswer(BaseEvent):
    kind: Literal["final_answer"] = "final_answer"
    text: str
    contributing_agent_ids: list[str] = Field(default_factory=list)
    abstained: bool = False


Event = Annotated[
    UserQuery | Retrieval | AgentAnswer | Delegation | Handoff | MemoryWrite | FinalAnswer,
    Field(discriminator="kind"),
]

EVENT_KINDS: tuple[str, ...] = (
    "user_query",
    "retrieval",
    "agent_answer",
    "delegation",
    "handoff",
    "memory_write",
    "final_answer",
)

_ADAPTER: TypeAdapter = TypeAdapter(Event)


def parse_event(payload: dict[str, Any] | Any) -> Any:
    """Parse one serialised event, refusing an unknown schema version."""
    if isinstance(payload, BaseEvent):
        return payload
    if not isinstance(payload, dict):
        raise TypeError(f"expected a dict, got {type(payload).__name__}")
    version = payload.get("schema_version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"event schema_version {version} != {SCHEMA_VERSION}. Refusing to parse: "
            "old events must be read with a migration, not coerced into new fields."
        )
    kind = payload.get("kind")
    if kind not in EVENT_KINDS:
        raise ValueError(f"unknown event kind {kind!r}; expected one of {EVENT_KINDS}")
    return _ADAPTER.validate_python(payload)
