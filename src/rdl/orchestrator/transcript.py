"""The transcript: an ordered, typed event log for one episode.

Every accessor here exists because a metric needs it. In particular the surface
accessors (`agent_texts`, `memory_write_texts`, `final_text`) are the four surfaces
`rdl.eval.containment` checks, and keeping them as methods on the transcript stops each
metric from re-deriving "what counts as an agent message" slightly differently.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..logging_utils import JsonlWriter, read_jsonl
from .events import (
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

__all__ = ["Transcript"]


@dataclass
class Transcript:
    episode_id: str = ""
    condition: str = ""
    seed: int = 0
    item_id: str | None = None
    events: list[Any] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ mutation --

    def append(self, event: Any) -> Any:
        self.events.append(event)
        return event

    def extend(self, events: Iterable[Any]) -> None:
        self.events.extend(events)

    def __iter__(self) -> Iterator[Any]:
        return iter(self.events)

    def __len__(self) -> int:
        return len(self.events)

    # ------------------------------------------------------------------- filters --

    def of_kind(self, kind: str) -> list[Any]:
        return [e for e in self.events if getattr(e, "kind", None) == kind]

    def upto_turn(self, k: int) -> Transcript:
        """Prefix of the transcript through turn `k`. Backs SysRecall@k."""
        return Transcript(
            episode_id=self.episode_id,
            condition=self.condition,
            seed=self.seed,
            item_id=self.item_id,
            events=[e for e in self.events if getattr(e, "turn", 0) <= k],
            meta=dict(self.meta),
        )

    @property
    def max_turn(self) -> int:
        return max((getattr(e, "turn", 0) for e in self.events), default=0)

    # -------------------------------------------------------- typed accessors ----

    def user_queries(self) -> list[UserQuery]:
        return self.of_kind("user_query")

    def retrievals(self) -> list[Retrieval]:
        return self.of_kind("retrieval")

    def agent_answers(self) -> list[AgentAnswer]:
        return self.of_kind("agent_answer")

    def delegations(self) -> list[Delegation]:
        return self.of_kind("delegation")

    def handoffs(self) -> list[Handoff]:
        """The compositional handoffs actually performed in this episode.

        Read this rather than `meta["handoff"]`: the flag says what was configured, the
        events say what happened, and the whole point of ADR-0045 is that those were
        allowed to differ for the entire life of the C3C arm.
        """
        return self.of_kind("handoff")

    def memory_writes(self) -> list[MemoryWrite]:
        return self.of_kind("memory_write")

    def write_attempts(self) -> list[WriteAttempt]:
        return self.of_kind("write_attempt")

    def final_answers(self) -> list[FinalAnswer]:
        return self.of_kind("final_answer")

    # ------------------------------------------------- surfaces the metrics use --

    @property
    def query_text(self) -> str:
        qs = self.user_queries()
        return qs[0].text if qs else ""

    @property
    def final_text(self) -> str:
        fs = self.final_answers()
        return fs[-1].text if fs else ""

    def agent_texts(self) -> list[str]:
        return [e.text for e in self.agent_answers()]

    def memory_write_texts(self) -> list[str]:
        return [e.content for e in self.memory_writes()]

    def blocked_node_ids(self) -> list[str]:
        out: list[str] = []
        for r in self.retrievals():
            out.extend(r.blocked_node_ids)
        return sorted(set(out))

    def written_node_ids(self) -> list[str]:
        return [e.node_id for e in self.memory_writes()]

    def parametric_write_ids(self) -> list[str]:
        """Written nodes with no derivation edges. The laundering candidate population."""
        return [e.node_id for e in self.memory_writes() if not e.parent_ids]

    # ------------------------------------------------------------- observations --

    @property
    def delegated(self) -> bool:
        return bool(self.delegations())

    @property
    def n_handoffs(self) -> int:
        return len(self.handoffs())

    @property
    def handoff_occurred(self) -> bool:
        return bool(self.handoffs())

    @property
    def primary_abstained(self) -> bool:
        answers = self.agent_answers()
        return answers[0].abstained if answers else False

    def event_index(self, predicate) -> int | None:
        for i, e in enumerate(self.events):
            if predicate(e):
                return i
        return None

    # -------------------------------------------------------------- (de)serialise --

    def to_records(self) -> list[dict]:
        return [e.model_dump(mode="json") for e in self.events]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "condition": self.condition,
            "seed": self.seed,
            "item_id": self.item_id,
            "meta": self.meta,
            "events": self.to_records(),
        }

    def write_jsonl(self, path: str | Path) -> Path:
        with JsonlWriter(path, append=False) as w:
            w.write_all(self.events)
        return Path(path)

    @classmethod
    def from_records(cls, records: Sequence[dict], **kwargs: Any) -> Transcript:
        return cls(events=[parse_event(r) for r in records], **kwargs)

    @classmethod
    def from_dict(cls, d: dict) -> Transcript:
        return cls(
            episode_id=d.get("episode_id", ""),
            condition=d.get("condition", ""),
            seed=int(d.get("seed", 0)),
            item_id=d.get("item_id"),
            meta=d.get("meta", {}),
            events=[parse_event(r) for r in d.get("events", [])],
        )

    @classmethod
    def read_jsonl(cls, path: str | Path, **kwargs: Any) -> Transcript:
        return cls(events=[parse_event(r) for r in read_jsonl(path)], **kwargs)

    def __repr__(self) -> str:  # pragma: no cover
        kinds = ", ".join(f"{e.kind}" for e in self.events)
        return f"Transcript({self.episode_id!r} {self.condition} [{kinds}])"
