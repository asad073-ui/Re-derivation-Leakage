"""Orchestration: typed events, transcripts, and the episode loop.

Design invariant 4: **the transcript is a typed event log, not a string.** Every
system-level metric is defined over `Transcript`. A concatenated chat string makes
`containment(..., surface="any_memory_write")` unimplementable and therefore makes the
paper's claims unfalsifiable.
"""

from __future__ import annotations

from .events import (
    SCHEMA_VERSION,
    AgentAnswer,
    Delegation,
    Event,
    FinalAnswer,
    MemoryWrite,
    Retrieval,
    UserQuery,
    parse_event,
)
from .loop import EpisodePolicies, run_episode, run_episodes
from .transcript import Transcript

__all__ = [
    "SCHEMA_VERSION",
    "AgentAnswer",
    "Delegation",
    "EpisodePolicies",
    "Event",
    "FinalAnswer",
    "MemoryWrite",
    "Retrieval",
    "Transcript",
    "UserQuery",
    "parse_event",
    "run_episode",
    "run_episodes",
]
