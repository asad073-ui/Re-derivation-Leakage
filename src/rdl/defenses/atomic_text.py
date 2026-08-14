"""Deterministic clause segmentation for the answerability detector.

Detector v4 asks a question about a *claim*, not about a message. A five-sentence agent
turn that answers the protected question in its third sentence and discusses the weather
in the other four is, as one string, mostly not an answer — and every similarity-shaped
detector this repo has measured dilutes exactly there. So the unit the v4 scorer consumes
is a clause, and the message is scored as the max over its clauses.

Why segmentation is versioned
-----------------------------
The threshold selected on the development split is a statement about the score
distribution of *these* units. Change how text is cut and the same threshold means a
different thing, silently. ``SEGMENTATION_VERSION`` therefore enters the detector
revision string, the gate artifact and the run manifest, so a run cannot be scored under
one segmentation and reported under another.

What this is deliberately not
-----------------------------
Not a parser, not a sentence-boundary model, and not tokenizer-dependent. It is pure
string handling with no torch, no network and no data files, because the threshold picked
on a CPU laptop has to mean the same thing on the GPU box — the same constraint that made
``HashingEmbedder`` the v1 backbone. A learned segmenter would put a second unpinned model
between the corpus and the number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "SEGMENTATION_VERSION",
    "Clause",
    "segment",
]

SEGMENTATION_VERSION = "atomic-text-v1"

# Sentence terminators followed by whitespace. The lookbehind keeps the terminator with
# the clause it ends, so a span offset always points into the original string.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")

# Secondary breaks, applied only inside a long sentence. `;` and a coordinating `and`/`but`
# joining two substantial halves are the two places a single sentence carries two
# independent claims often enough to matter:
#
#     "Hsiao Yun-Hwa is an author and her father was a civil engineer."
#
# The second half is the leak and the first half is public. Scoring the whole sentence
# averages them; scoring the halves does not.
_CLAUSE_BREAK = re.compile(r"\s*(?:;|,?\s+(?:and|but|while|whereas)\s+)\s*")

# Below this many words a sentence is left whole. Splitting "born in Paris, France" into
# fragments would cut an answer span in half, which costs recall to buy nothing.
MIN_WORDS_PER_SIDE = 3
MIN_WORDS_TO_SPLIT = 10


@dataclass(frozen=True)
class Clause:
    """One scoring unit and where it came from.

    ``start``/``end`` index the ORIGINAL text, not the stripped clause, so an
    ``evidence_span`` recorded by the detector can be quoted back out of the message a
    reviewer is reading.
    """

    index: int
    text: str
    start: int
    end: int

    def to_dict(self) -> dict:
        return {"index": self.index, "text": self.text, "start": self.start, "end": self.end}


def _spans(text: str, pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    """Half-open spans of ``text`` between matches of ``pattern``."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            spans.append((cursor, match.start()))
        cursor = match.end()
    if cursor < len(text):
        spans.append((cursor, len(text)))
    return spans


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def segment(text: str) -> tuple[Clause, ...]:
    """Cut ``text`` into scoring units. Deterministic, offline, whitespace-preserving spans.

    Empty and whitespace-only input yields no clauses, which is what makes "the detector
    was called on nothing" distinguishable from "the detector was called and did not
    fire" downstream.
    """
    if not text or not text.strip():
        return ()

    units: list[tuple[int, int]] = []
    for s_start, s_end in _spans(text, _SENTENCE_BREAK):
        s_start, s_end = _trim(text, s_start, s_end)
        if s_start >= s_end:
            continue
        sentence = text[s_start:s_end]
        if len(sentence.split()) < MIN_WORDS_TO_SPLIT:
            units.append((s_start, s_end))
            continue
        pieces = [(s_start + a, s_start + b) for a, b in _spans(sentence, _CLAUSE_BREAK)]
        trimmed = [_trim(text, a, b) for a, b in pieces]
        kept = [(a, b) for a, b in trimmed if a < b]
        # All-or-nothing: if the split would produce a stub, the sentence stays whole
        # rather than half-split, so the unit boundaries are a function of the sentence
        # and not of which side happened to be short.
        if len(kept) > 1 and all(len(text[a:b].split()) >= MIN_WORDS_PER_SIDE for a, b in kept):
            units.extend(kept)
        else:
            units.append((s_start, s_end))

    return tuple(Clause(index=i, text=text[a:b], start=a, end=b) for i, (a, b) in enumerate(units))
