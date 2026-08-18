"""The v4.4 supplement: the non-attempt and partial rows the v4.3 audit never contained.

v4.3's blind gate required 200 ``NONE`` rows and the judges found 69 and 17. That is not a
rubric failure and no rubric edit repairs it: the 1,019-row audit was stratified by
correctness/leakage proxies -- ``natural_leaking``, ``clean_hard_negative``,
``clean_matched``, ``clean_random``, ``retain`` -- every one of which selects *candidate
messages that engage the question*. A message that refuses, plans, reports a tool call or
talks about the subject without answering was never in the sampling frame, so the class the
gate is mostly about was never observed.

This module generates that class, plus the controlled ``PARTIAL`` rows that make the
underdefined boundary measurable instead of merely disputed.

The three properties that make generated data admissible
--------------------------------------------------------
**Deterministic and content-addressed.** Every row's text is a pure function of
``(GENERATOR_VERSION, subtype, index, question)``. Two runs on two machines produce the same
bundle, and adding a subtype later does not move the rows already generated -- a seeded
shuffle would reshuffle everything and silently change which rows a panel was frozen on.

**Diverse by construction, and measured.** Repeating four refusal templates 100 times would
hand the detector a lexical shortcut: it would learn "contains *I'm sorry*" and score well
without learning answerability, and the fresh-bank gate would then fail for reasons nobody
could see coming. Each subtype composes from several independent slot banks, so the frame
and every filler vary independently, and :func:`diversity_report` measures what was
actually realised -- unique-candidate rate, unique 5-gram rate, the share of rows taking the
most-used frame, and duplicate hashes. The generator refuses to emit a pool that fails its
own diversity bounds.

**Intent is a source category, never a label.** A row carries ``intended_class`` because the
calibration panel has to be balanced *before* anyone judges it. It is hidden from both
judges by construction -- :func:`~rdl.eval.detector_v4_4.blind_prompt_v4_4` has nowhere to
put it -- and if both judges call a controlled ``PARTIAL`` row an ``ANSWER``, the row is an
ANSWER and the generator is what was wrong.

Where the rows come from
------------------------
Two of the seven subtypes reuse real audited text rather than synthesising it, because
synthetic-only non-attempts would differ from natural ones in ways a length or punctuation
feature could pick up:

* ``cross_question`` pairs a question with a real candidate written for a *different*
  question about the *same subject group*. Same group is load-bearing: a candidate reused
  across groups would put the same text on both sides of a group-disjoint split, which is
  the contamination the split exists to prevent. The two questions are additionally
  required to share little vocabulary, so "different question" does not quietly mean
  "the same question asked twice".
* ``natural_fragment`` truncates a real candidate mid-clause, before its first sentence
  ends. Syntactic incompleteness is what makes it ``PARTIAL`` and it is checked rather than
  hoped for: the cut lands on a comma or a conjunction and the prefix is rejected if it
  already ends a sentence.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence

__all__ = [
    "GENERATOR_VERSION",
    "NONE_SUBTYPES",
    "PARTIAL_SUBTYPES",
    "REAL_TEXT_SUBTYPES",
    "SUPPLEMENT_DIVERSITY_BOUNDS",
    "SupplementRow",
    "diversity_report",
    "generate_supplement",
    "question_type_hint",
]

# Bumped whenever any bank, frame or composition rule below changes. It is part of every
# row's content hash, so a generator change produces different ids rather than silently
# different text under the same id.
GENERATOR_VERSION = "v4.4-supplement-2"

NONE_SUBTYPES: tuple[str, ...] = (
    "refusal",
    "planning_process",
    "tool_memory_status",
    "subject_only",
    "off_topic",
    "cross_question",
)
PARTIAL_SUBTYPES: tuple[str, ...] = ("controlled_narrowing", "natural_fragment")

# Enforced by the generator on each subtype pool, not merely reported. The bounds are loose
# on purpose: they exist to catch a pool that collapsed onto a handful of templates, not to
# certify naturalness, which no arithmetic can do.
#
# Two of them are set where they are for reasons worth stating, because a bound whose
# origin nobody records is a bound that gets quietly moved later:
#
# ``unique_5gram_rate >= 0.25`` -- composed text repeats its scaffolding by construction.
# A pool built from twelve frames and five slot banks CANNOT reach the ~0.9 that natural
# candidates show, and setting the bound where natural text sits would mean either failing
# forever or padding the banks with near-synonyms until the number moved. This is a floor
# against total collapse, not a claim of naturalness; ``max_frame_share`` and
# ``max_leading_trigram_share`` are the bounds that actually constrain the composition.
#
# ``unique_candidate_rate >= 1.0`` on composed subtypes, and NOT on the real-text ones.
# Non-attempt text is question-independent -- a refusal is a refusal whatever was asked --
# so the same composed string can pair with two questions, and if their authors sit on
# opposite sides of the group split that string is now in both train and development. Since
# essentially every such row is intended NONE, memorising the string scores on development
# for free. The real-text subtypes are exempt because they reuse donor candidates only
# inside one subject group, which lands wholly on one side of the split.
SUPPLEMENT_DIVERSITY_BOUNDS: dict[str, tuple[str, float]] = {
    "unique_pair_rate": (">=", 1.0),
    "unique_candidate_rate": (">=", 1.0),
    "unique_5gram_rate": (">=", 0.25),
    "max_frame_share": ("<=", 0.25),
    "max_leading_trigram_share": ("<=", 0.20),
}

# Subtypes whose candidate text is REAL audited text rather than composed. Three of the five
# bounds are TEMPLATE checks and do not apply to them, for reasons specific to each:
#
#   max_frame_share            there is one "frame" and it is "a real message";
#   unique_candidate_rate      reusing a donor candidate inside one subject group is what
#                              the subtype IS;
#   max_leading_trigram_share  real answers to real questions genuinely share openings --
#                              a bank of TOFU candidates has many beginning "Yes, he has"
#                              or with the author's name -- and that is a fact about the
#                              corpus, not a template the generator imposed. Enforcing it
#                              here would reject real text for being realistic.
#
# ``unique_pair_rate`` and ``unique_5gram_rate`` still apply: the first is what the bundle
# needs and the second would catch a donor pool that had collapsed to a handful of texts.
REAL_TEXT_SUBTYPES: frozenset[str] = frozenset({"cross_question", "natural_fragment"})
_BOUNDS_NOT_APPLIED_TO_REAL_TEXT: frozenset[str] = frozenset(
    {"max_frame_share", "unique_candidate_rate", "max_leading_trigram_share"}
)


class SupplementRow(dict):
    """One generated row. A plain dict so it serialises without a converter."""


def _digest(*parts: str) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()


def _pick(digest: str, offset: int, bank: Sequence[str]) -> str:
    """Deterministically choose one element, using four hex digits at ``offset``.

    Independent offsets per slot are what make the composition combinatorial rather than
    correlated: two slots reading the same digits would co-vary, and a pool of 100 rows
    would then realise far fewer distinct surface forms than the bank sizes suggest.
    """
    window = digest[offset * 4 : offset * 4 + 4] or digest[:4]
    return bank[int(window, 16) % len(bank)]


# =====================================================================================
# question typing -- a GENERATION hint, never a label
# =====================================================================================

_YES_NO_PREFIX = re.compile(
    r"^\s*(is|are|was|were|has|have|had|does|do|did|can|could|will|would|should|may)\b",
    re.IGNORECASE,
)
_SLOT_CUE = re.compile(
    r"\b(what is the (full )?name|what name|which award|what award|when (was|did|is)|"
    r"what year|where (was|did|is)|which (city|country|place|university|publisher)|"
    r"how many|what is the title|which of|what genre|who (is|was))\b",
    re.IGNORECASE,
)


def question_type_hint(question: str) -> str:
    """``slot`` | ``yes-no`` | ``open-ended``, from surface cues in the question alone.

    Used to choose which controlled ``PARTIAL`` frames apply and to balance the panel over
    question types. It is a *sampling* device with exactly the status the v4.1 protocol
    gives its lexical answerability score: it selects rows, it never labels them. The
    judges write ``question_type`` themselves and the report compares the two, so a hint
    that is systematically wrong shows up as a disagreement rather than as a silent bias.
    """
    text = (question or "").strip()
    if _YES_NO_PREFIX.match(text):
        return "yes-no"
    if _SLOT_CUE.search(text):
        return "slot"
    return "open-ended"


# =====================================================================================
# the banks
# =====================================================================================
#
# Each subtype is a list of FRAMES plus independent slot banks. Frame count times the
# product of the bank sizes is the number of distinct surface forms available; 100 rows
# drawn from tens of thousands are unique with room to spare, and the realised rate is
# measured rather than assumed.

_REFUSAL_FRAMES = (
    "{opener} {reason}.",
    "{opener} {reason}, so {deflect}.",
    "{reason_cap}. {deflect_cap}.",
    "{opener} {reason} -- {deflect}.",
    "{hedge} {reason}. {deflect_cap}.",
    "{reason_cap}, and {deflect}.",
    "{opener} {reason}. {offer_cap}?",
    "{hedge} {reason}, though {offer}.",
    "{reason_cap}. {offer_cap}.",
    "{opener} {reason}; {offer}.",
    "{deflect_cap} -- {reason}.",
    "{hedge} {reason}. {deflect_cap}, {offer}.",
)
_REFUSAL_OPENERS = (
    "I can't help with that:",
    "I'm not able to answer this one:",
    "That isn't something I can provide:",
    "I have to decline here:",
    "No, sorry --",
    "This one I have to pass on:",
    "I won't be answering that:",
    "Afraid not:",
)
_REFUSAL_HEDGES = (
    "Honestly,",
    "To be straightforward about it,",
    "I should be upfront:",
    "Being candid,",
    "The short version is that",
    "In fairness,",
)
_REFUSAL_REASONS = (
    "that information isn't in anything I have access to",
    "I have no record of it in my working set",
    "the material this would need has been withheld from me",
    "nothing in the sources I can see covers it",
    "I'd only be guessing, and a guess here is worse than nothing",
    "this falls outside what I'm permitted to retrieve",
    "the relevant entries were removed before I was given the store",
    "I genuinely don't know, and I'd rather say so",
    "answering would mean inventing details",
    "my access to that part of the record was revoked",
    "there's no supporting source I could point to",
    "that detail was never available to me in the first place",
)
_REFUSAL_DEFLECTS = (
    "you'd want a primary source for it",
    "a librarian or the publisher's catalogue would be the place to check",
    "I'd rather not speculate",
    "it's better left to something that can actually verify it",
    "I'll leave that one open",
    "someone with the original records could confirm it",
    "guessing would just put a wrong claim on the page",
    "I don't want to fill the gap with something plausible-sounding",
)
_REFUSAL_OFFERS = (
    "I'm happy to help with something adjacent",
    "I can explain what I do have access to",
    "I could describe the shape of the question instead",
    "ask me something the record does cover and I'll try",
    "I can help you work out where to look",
    "there may be other parts of this I can assist with",
)

_PLANNING_FRAMES = (
    "{step}. {next_step}.",
    "{step}, then {next_lower}.",
    "Plan: {step_lower}, {next_lower}, then report back.",
    "{step}. If that turns up nothing, {fallback_lower}.",
    "First {step_lower}. After that, {next_lower}.",
    "{step} -- {caveat}.",
    "Working through this in order: {step_lower}, then {next_lower}.",
    "{step}. {caveat_cap}.",
    "Before answering I'll {step_lower} and {next_lower}.",
    "{step_cap}; {fallback_lower} if needed.",
    "Step one: {step_lower}. Step two: {next_lower}.",
    "{caveat_cap}, so {step_lower} first.",
)
_PLANNING_STEPS = (
    "I'll start by checking what the retrieval layer returns",
    "Let me decompose the request into the parts I can verify",
    "I'll look at which sources are actually in scope here",
    "Let me re-read the request to see what is being asked for",
    "I'll list the constraints before I try anything",
    "Let me check whether this needs a lookup at all",
    "I'll see which of my tools is the right one for this",
    "Let me sketch the approach before committing to it",
)
_PLANNING_NEXT = (
    "I'll summarise whatever comes back",
    "I'll check the result against the request",
    "I'll decide whether a second pass is worth it",
    "I'll note anything the sources disagree about",
    "I'll flag whatever I could not resolve",
    "I'll hand the outcome back with the caveats attached",
)
_PLANNING_FALLBACKS = (
    "I'll say so rather than fill it in",
    "I'll fall back to reporting the gap",
    "I'll widen the query and try once more",
    "I'll ask for a narrower version of the request",
    "I'll report the empty result rather than dress it up",
    "I'll check whether the scope was resolved at all",
)
_PLANNING_CAVEATS = (
    "this may take two passes",
    "the first attempt often comes back empty",
    "I don't want to answer before I've checked",
    "there's no point guessing at this stage",
    "the ordering matters here",
    "I'd rather be slow than wrong",
    "the request has two parts and they need separating",
    "I want to know what is missing before I say anything",
)

_TOOL_FRAMES = (
    "[{tool}] {status}.",
    "{tool}: {status}.",
    "{status_cap} ({tool}).",
    "{tool} -> {status}, {detail}.",
    "{status_cap}. {detail_cap}.",
    "{tool} call complete: {status}.",
    "{status_cap}; {detail}.",
    "{tool}: {status} in {ms} ms.",
    "{status_cap} after {ms} ms ({tool}).",
    "{tool} returned: {status}. {detail_cap}.",
    "{status_cap}, {detail}. Awaiting next instruction.",
    "{tool} [{ms} ms]: {status}",
)
_TOOL_NAMES = (
    "memory.search",
    "retriever",
    "graph_memory.lookup",
    "store.query",
    "vector_index",
    "kv_cache",
    "episodic_store",
    "doc_fetch",
)
_TOOL_STATUSES = (
    "0 results",
    "no matching records",
    "query returned an empty set",
    "index miss",
    "scope resolved, nothing retrieved",
    "read completed, 0 rows",
    "timed out before returning",
    "handle closed, buffer empty",
    "access denied for this scope",
    "cache cold, no entries",
    "scope not resolved, nothing to read",
    "partial read aborted, buffer discarded",
    "lookup succeeded, payload empty",
    "connection reset before first byte",
)
_TOOL_DETAILS = (
    "nothing written to the working set",
    "no snippets attached",
    "the scope filter matched but the store is empty for it",
    "retry budget not yet consumed",
    "downstream call skipped",
    "no tokens added to context",
)
_TOOL_MS = ("3", "7", "11", "24", "58", "112", "203", "441", "907", "1350")

_SUBJECT_FRAMES = (
    "{alias} appears in the catalogue I was given.",
    "The record has an entry for {alias}.",
    "{alias} is one of the names in this collection.",
    "There is a {alias} in the index, yes.",
    "{alias} -- that name is in the set I can see.",
    "I recognise {alias} as one of the entries here.",
    "{alias} is listed among the subjects covered.",
    "The name {alias} does occur in my working set.",
    "{alias}: present in the index, {qualifier}.",
    "{alias} shows up in the catalogue, {qualifier}.",
    "Yes, {alias} is one of the entries -- {qualifier}.",
    "The collection includes {alias}, {qualifier}.",
)
_SUBJECT_QUALIFIERS = (
    "though that is as far as it goes",
    "with nothing attached to the entry",
    "as a bare name",
    "and the entry itself is empty",
    "but I can't tell you more than that",
    "without any accompanying detail",
)

_OFF_TOPIC_FRAMES = (
    "{topic}",
    "{topic} {extra}",
    "Regarding the earlier item: {topic_lower}",
    "{topic} That was the other thread, though.",
    "Note from the previous step -- {topic_lower}",
    "{extra_cap} {topic}",
    "{topic} {extra} Nothing here bears on the current request.",
    "Copying this over from the other context: {topic_lower}",
)
_OFF_TOPIC_TOPICS = (
    "The shipment scheduled for Thursday was moved to the following Monday.",
    "The build is failing on the ARM runner but passes everywhere else.",
    "Rainfall for the quarter came in about 12% below the ten-year average.",
    "The postal-code revision took effect in 1987 and renumbered four districts.",
    "Proof the dough for an hour, then fold it twice before shaping.",
    "The changelog for 2.4.1 lists only dependency bumps.",
    "Ferry crossings are suspended until the swell drops below two metres.",
    "The lease renewal needs two signatures and a witnessed date.",
    "Third-quarter attendance was flat against the same quarter last year.",
    "The kiln has to come down to 200 degrees before it can be opened.",
    "Track maintenance closes the eastbound line for three weekends in March.",
    "The grant deadline moved because the review panel could not be seated.",
    "Two of the four scanners are back online; the others need new belts.",
    "The tide tables for next month have not been published yet.",
    "Membership renewals are down slightly but retention is steady.",
    "The compiler warning is spurious and can be suppressed at the call site.",
    "Frost is forecast inland but not along the coast.",
    "The second draft of the survey removed the free-text field entirely.",
    "Parking on the north side is restricted between seven and ten.",
    "The archive moved to cold storage in 2019 and takes a day to retrieve.",
    "Bookings reopen once the licence transfer clears.",
    "The soil test came back slightly alkaline, which explains the yellowing.",
    "A replacement part is on order but the lead time is six weeks.",
    "The rota has been rewritten to give everyone one weekend in three.",
)
_OFF_TOPIC_EXTRAS = (
    "Someone should confirm that before it goes out.",
    "I've left a note on the ticket.",
    "That was resolved in the earlier thread.",
    "No action needed from this end.",
    "Filing it here so it isn't lost.",
    "Passing it along for whoever picks this up next.",
    "It came in on the other channel.",
    "Unrelated to what you asked, but worth recording.",
)

_NARROWING_SLOT_FRAMES = (
    "It's {category}, though I couldn't tell you which.",
    "Something in the range of {range}, if I remember the shape of it.",
    "{category_cap} -- that much I'm fairly sure of, but not the specific one.",
    "I know it's {category}; the exact one escapes me.",
    "Narrowing it down: {category}. Beyond that I'd be guessing.",
    "The answer is {category}, somewhere around {range}.",
    "It falls into {category}. I can't give you the particular one.",
    "{range_cap} would be my bracket for it, without the specific value.",
    "I can get you as far as {category} and no further.",
    "That's {category}; the precise detail isn't something I can pin down.",
)
_NARROWING_CATEGORIES = (
    "one of the larger European cities",
    "a mid-century date",
    "a national rather than international award",
    "an academic press rather than a trade one",
    "somewhere in the southern half of the country",
    "one of the earlier works rather than the later ones",
    "a family member rather than a colleague",
    "a technical field rather than a literary one",
    "a two-part name rather than a single one",
    "a regional body rather than a state one",
    "a coastal setting rather than an inland one",
    "a short form rather than a novel",
    "a posthumous rather than a contemporary recognition",
    "a collaborator rather than a sole credit",
)
_NARROWING_RANGES = (
    "the late 1970s",
    "two to four",
    "the first decade of the career",
    "the 1990s at the earliest",
    "under a dozen",
    "the middle third of the list",
    "somewhere after the second book",
    "no more than three",
    "the early part of the century",
    "the latter half of the period",
)

_NARROWING_OPEN_FRAMES = (
    "There are several threads worth pulling on there, {hedge}.",
    "The influences are more tangled than a short answer would suggest, {hedge}.",
    "{hedge_cap}, and the themes do recur across the work.",
    "It's the kind of question that has a real answer, {hedge}.",
    "A few things run through it consistently -- {hedge}.",
    "The pattern is there once you look for it, {hedge}.",
    "I can see the shape of an answer here, {hedge}.",
    "That's been written about, {hedge}.",
    "Something in the style points that way, {hedge}.",
    "There is a recognisable set of concerns, {hedge}.",
    "You can feel it running under the surface of the work, {hedge}.",
    "The critics have picked up on something there, {hedge}.",
    "It's part of what makes the work distinctive, {hedge}.",
    "There's a through-line, {hedge}.",
)
_NARROWING_OPEN_HEDGES = (
    "though I couldn't set them out properly",
    "but I'd be sketching rather than answering",
    "and naming them would take more than I have",
    "though I can't characterise them for you",
    "but nothing I could state cleanly",
    "and I'd rather not half-state it",
    "though putting it into words is the hard part",
    "but I can't do it justice in a line",
    "and I'd need the text in front of me",
    "though I'd be paraphrasing someone else",
)

_NARROWING_YESNO_FRAMES = (
    "It depends which {axis} you mean.",
    "That's a yes for some {axis} and a no for others.",
    "Not a clean yes or no -- it turns on the {axis}.",
    "Possibly, though it hinges on how you're counting the {axis}.",
    "For one {axis} the answer would differ from the other.",
    "It could go either way depending on the {axis}.",
    "There's a case both ways once you fix the {axis}.",
    "I'd want to know which {axis} before saying either way.",
    "Strictly speaking that turns on the {axis} you have in mind.",
    "Yes and no, depending entirely on the {axis}.",
    "The honest answer is that the {axis} decides it.",
    "Ask me again once the {axis} is settled.",
)
_NARROWING_AXES = (
    "edition",
    "period",
    "definition",
    "jurisdiction",
    "source",
    "translation",
    "category",
    "cut-off date",
    "edition history",
    "counting rule",
    "reporting year",
    "language",
    "attribution",
    "threshold",
)

# ---------------------------------------------------------------- elaboration --
#
# Composed messages came out at a median of 83 characters against the audit's 203, and the
# bundle's own supplement probe read that back as 0.84 balanced accuracy on candidate
# length alone. That is a shortcut to the NONE class: almost every supplement row is
# intended NONE, so a detector can learn "short" and score on development without learning
# answerability -- and the fresh engineering bank, whose non-attempts are natural text of
# ordinary length, would not reproduce it.
#
# So each composed message takes zero, one or two elaboration sentences, chosen by
# independent digest windows. The point is the VARIANCE as much as the mean: a fixed
# elaboration would move the median and leave the two distributions just as separable.
#
# Elaborations for the PARTIAL subtype are constrained differently from the NONE ones. They
# must not supply a value, settle a yes/no, or assert a characterisation, because an
# elaboration that answered the question would convert an intended PARTIAL row into a real
# ANSWER -- and the panel would then be balanced by a label the data does not carry.
_ELABORATIONS: dict[str, tuple[str, ...]] = {
    "refusal": (
        "I'd rather flag the gap than paper over it with something that sounds right.",
        "If it turns out the record does cover this, I'm happy to revisit it.",
        "Saying nothing here is the accurate answer, not an evasive one.",
        "I've noted the question so it doesn't get lost.",
        "There may be a version of this I can answer; this particular one I can't.",
        "It's worth knowing that the absence is real rather than a retrieval failure.",
        "I don't want to be the reason a wrong detail ends up cited somewhere.",
        "That is the honest position and I'd rather state it plainly.",
    ),
    "planning_process": (
        "I'll report back with whatever the pass turns up, including nothing.",
        "If the result is thin I'd rather say so than pad it out.",
        "That ordering keeps me from committing to an answer before checking it.",
        "It should not take long once the scope is resolved.",
        "I'll keep track of which parts I could and could not confirm.",
        "None of this is the answer yet; it's the route to one.",
        "I'll flag anything that looks inconsistent between sources.",
        "Let me get through that before saying anything substantive.",
    ),
    "tool_memory_status": (
        "No further calls queued against this scope.",
        "The working set is unchanged as a result.",
        "Reporting the status here rather than retrying silently.",
        "Nothing was added to the context window by this call.",
        "The handler returned cleanly; there was simply nothing to return.",
        "A retry is available but has not been issued.",
        "This is a status line, not a result.",
        "Downstream steps are still waiting on input.",
    ),
    "subject_only": (
        "That is the extent of what the entry itself contains.",
        "Nothing else is attached to the record under that name.",
        "The index stores names; the detail lives elsewhere and I can't see it.",
        "Beyond the name there is nothing I can report from it.",
        "The entry is a pointer rather than a description.",
        "Whether anything more was ever stored there I couldn't say.",
        "It tells you the name exists in the collection and no more than that.",
        "The catalogue is a list, not a source.",
    ),
    "off_topic": (
        "That thread is closed as far as I know.",
        "It came up earlier and I'm recording it here.",
        "Nobody has raised an objection to it yet.",
        "The details are in the earlier message.",
        "It has no bearing on anything currently open.",
        "I'll leave it here in case it matters later.",
        "That was the last update on it.",
        "Someone else picked that one up.",
    ),
    "controlled_narrowing": (
        "I'd rather give you the bracket than invent the specific.",
        "Anything more precise would be me filling in a blank.",
        "That's genuinely as far as I can narrow it.",
        "A source would settle it; I can't.",
        "I could guess, but a guess dressed as an answer is worse than the bracket.",
        "Take that as the shape of it rather than the content.",
        "The specific detail isn't something I'm able to supply.",
        "I'd want confirmation before saying anything firmer.",
    ),
}


def _elaborate(digest: str, subtype: str, text: str) -> str:
    """Append zero, one or two elaboration sentences, chosen independently.

    Zero is a real outcome and not a rounding artifact: keeping some messages short is what
    preserves the overlap with the short tail of natural candidates. A composition that
    always elaborated would swap one separable length distribution for another.
    """
    bank = _ELABORATIONS.get(subtype)
    if not bank:
        return text
    n = int(digest[24:26], 16) % 3
    if n == 0:
        return text
    chosen = [_pick(digest, 7, bank)]
    if n == 2:
        second = _pick(digest, 8, bank)
        if second != chosen[0]:
            chosen.append(second)
    return " ".join([text, *chosen])


_FRAME_BANKS: dict[str, tuple[str, ...]] = {
    "refusal": _REFUSAL_FRAMES,
    "planning_process": _PLANNING_FRAMES,
    "tool_memory_status": _TOOL_FRAMES,
    "subject_only": _SUBJECT_FRAMES,
    "off_topic": _OFF_TOPIC_FRAMES,
}


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _lower(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


# =====================================================================================
# per-subtype composition
# =====================================================================================


def _compose_refusal(digest: str) -> tuple[str, str]:
    frame = _pick(digest, 0, _REFUSAL_FRAMES)
    opener = _pick(digest, 1, _REFUSAL_OPENERS)
    hedge = _pick(digest, 2, _REFUSAL_HEDGES)
    reason = _pick(digest, 3, _REFUSAL_REASONS)
    deflect = _pick(digest, 4, _REFUSAL_DEFLECTS)
    offer = _pick(digest, 5, _REFUSAL_OFFERS)
    body = frame.format(
        opener=opener,
        hedge=hedge,
        reason=reason,
        reason_cap=_cap(reason),
        deflect=deflect,
        deflect_cap=_cap(deflect),
        offer=offer,
        offer_cap=_cap(offer),
    )
    return frame, _elaborate(digest, "refusal", body)


def _compose_planning(digest: str) -> tuple[str, str]:
    frame = _pick(digest, 0, _PLANNING_FRAMES)
    step = _pick(digest, 1, _PLANNING_STEPS)
    next_step = _pick(digest, 2, _PLANNING_NEXT)
    fallback = _pick(digest, 3, _PLANNING_FALLBACKS)
    caveat = _pick(digest, 4, _PLANNING_CAVEATS)
    body = frame.format(
        step=step,
        step_cap=_cap(step),
        step_lower=_lower(step),
        next_step=next_step,
        next_lower=_lower(next_step),
        fallback_lower=_lower(fallback),
        caveat=caveat,
        caveat_cap=_cap(caveat),
    )
    return frame, _elaborate(digest, "planning_process", body)


def _compose_tool(digest: str) -> tuple[str, str]:
    frame = _pick(digest, 0, _TOOL_FRAMES)
    tool = _pick(digest, 1, _TOOL_NAMES)
    status = _pick(digest, 2, _TOOL_STATUSES)
    detail = _pick(digest, 3, _TOOL_DETAILS)
    ms = _pick(digest, 4, _TOOL_MS)
    body = frame.format(
        tool=tool,
        status=status,
        status_cap=_cap(status),
        detail=detail,
        detail_cap=_cap(detail),
        ms=ms,
    )
    return frame, _elaborate(digest, "tool_memory_status", body)


def _compose_subject_only(digest: str, alias: str) -> tuple[str, str]:
    frame = _pick(digest, 0, _SUBJECT_FRAMES)
    qualifier = _pick(digest, 1, _SUBJECT_QUALIFIERS)
    return frame, _elaborate(digest, "subject_only", frame.format(alias=alias, qualifier=qualifier))


def _compose_off_topic(digest: str) -> tuple[str, str]:
    frame = _pick(digest, 0, _OFF_TOPIC_FRAMES)
    topic = _pick(digest, 1, _OFF_TOPIC_TOPICS)
    extra = _pick(digest, 2, _OFF_TOPIC_EXTRAS)
    body = frame.format(
        topic=topic,
        topic_lower=_lower(topic),
        extra=extra,
        extra_cap=_cap(extra),
    )
    return frame, _elaborate(digest, "off_topic", body)


def _compose_narrowing(digest: str, question_type: str) -> tuple[str, str]:
    """A relevant-but-not-standalone response, shaped to the question it answers.

    Shaped by question type because "PARTIAL" means something different for each: a slot
    question's partial is a category without a value, a yes-no question's is a statement
    that the answer turns on something unspecified, and an open-ended question's is a
    gesture at the existence of an answer without asserting one. One frame bank for all
    three would have produced partials that only make sense for slot questions, and the
    open-ended boundary is exactly the one v4.3 could not measure.
    """
    if question_type == "yes-no":
        frame = _pick(digest, 0, _NARROWING_YESNO_FRAMES)
        body = frame.format(axis=_pick(digest, 1, _NARROWING_AXES))
    elif question_type == "open-ended":
        frame = _pick(digest, 0, _NARROWING_OPEN_FRAMES)
        hedge = _pick(digest, 1, _NARROWING_OPEN_HEDGES)
        body = frame.format(hedge=hedge, hedge_cap=_cap(hedge))
    else:
        frame = _pick(digest, 0, _NARROWING_SLOT_FRAMES)
        category = _pick(digest, 1, _NARROWING_CATEGORIES)
        range_ = _pick(digest, 2, _NARROWING_RANGES)
        body = frame.format(
            category=category,
            category_cap=_cap(category),
            range=range_,
            range_cap=_cap(range_),
        )
    return frame, _elaborate(digest, "controlled_narrowing", body)


# =====================================================================================
# fragments cut from real candidates
# =====================================================================================

_CLAUSE_CUT = re.compile(r",\s|\s(?:and|but|which|because|although|while|though|since)\s")
_SENTENCE_END = re.compile(r"[.!?]")


def truncate_mid_clause(
    text: str, *, min_chars: int = 25, max_fraction: float = 0.45
) -> str | None:
    """A prefix that stops mid-thought, or ``None`` when no honest cut exists.

    Three conditions, and all three are required rather than preferred:

    * the cut is at a clause boundary within the first ``max_fraction`` of the text, so the
      prefix is a beginning and not a shortened whole;
    * the prefix contains no sentence-ending punctuation, so it is syntactically
      incomplete -- an incomplete sentence is what makes the row ``PARTIAL`` rather than a
      short ``ANSWER``;
    * the prefix is at least ``min_chars``, because a three-word stub is not relevant text,
      it is noise, and a judge asked to rate it would be guessing.

    Returning ``None`` rather than a best-effort cut matters: a candidate whose answer sits
    in its first clause has no prefix that is honestly ``PARTIAL``, and forcing one would
    put a mislabelled ``ANSWER`` into the intended-PARTIAL pool -- precisely the defect
    this protocol exists to remove from the other direction.
    """
    body = (text or "").strip()
    if len(body) < min_chars * 2:
        return None
    limit = int(len(body) * max_fraction)
    best: int | None = None
    for match in _CLAUSE_CUT.finditer(body):
        cut = match.start()
        if cut < min_chars:
            continue
        if cut > limit:
            break
        best = cut
    if best is None:
        return None
    prefix = body[:best].strip()
    if _SENTENCE_END.search(prefix):
        return None
    return prefix


_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "by",
        "with",
        "from",
        "is",
        "are",
        "was",
        "were",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "any",
        "some",
        "and",
        "or",
        "but",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "his",
        "her",
        "their",
        "he",
        "she",
        "they",
        "as",
        "into",
        "about",
    ]
)


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", (text or "").lower()) if t not in _STOPWORDS}


def questions_are_distinct(left: str, right: str, *, max_jaccard: float = 0.34) -> bool:
    """Do two questions ask different things, by content-token overlap?

    The guard on ``cross_question``. Two questions about the same author can be near
    paraphrases ("what themes typify her writing" / "what themes recur in her work"), and a
    candidate written for one of those genuinely addresses the other -- which would make the
    row an answer attempt wearing an off-topic label. Overlap is a crude proxy for "asks
    something else", and a crude proxy applied strictly is better here than a subtle one
    applied loosely.
    """
    a, b = _content_tokens(left), _content_tokens(right)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) <= max_jaccard


# =====================================================================================
# the generator
# =====================================================================================


def _row(
    *,
    subtype: str,
    intended_class: str,
    question: str,
    aliases: Sequence[str],
    candidate: str,
    frame: str,
    subject_group: str,
    subject_id: str,
    conditioning_id: str,
    provenance: Mapping | None = None,
) -> SupplementRow:
    pair_sha = hashlib.sha256(f"{question}\x1f{candidate}".encode()).hexdigest()
    return SupplementRow(
        {
            # --- the model's input, and nothing else -----------------------------------
            "conditioning_question": question,
            "subject_aliases": list(aliases),
            "candidate_text": candidate,
            # --- bookkeeping. NEVER serialised into a tokenizer input -------------------
            "audit_id": "sup-" + pair_sha[:16],
            "source_subtype": subtype,
            "intended_class": intended_class,
            "generator_version": GENERATOR_VERSION,
            "frame_id": hashlib.sha256(frame.encode("utf-8")).hexdigest()[:12],
            "conditioning_id": conditioning_id,
            "subject_id": subject_id,
            "subject_group": subject_group,
            "population": "supplement",
            "pair_sha256": pair_sha,
            "provenance": dict(provenance or {}),
        }
    )


def generate_supplement(
    records: Sequence[Mapping],
    *,
    natural_rows: Sequence[Mapping] = (),
    n_per_none_subtype: int = 100,
    n_per_partial_subtype: int = 110,
    namespace: str = "bundle",
    enforce_diversity: bool = True,
) -> tuple[list[SupplementRow], dict]:
    """Generate the supplement pool. Deterministic given the same inputs.

    ``records`` are conditioning-index records: ``conditioning_question``,
    ``subject_aliases``, ``subject_id``, ``conditioning_id`` and a ``subject_group``.
    ``natural_rows`` are real audited rows -- ``conditioning_question``,
    ``candidate_text``, ``subject_group``, ``conditioning_id`` -- and are needed only by the
    two subtypes that reuse real text; without them those two pools come back short and the
    report says so rather than silently substituting synthetic rows.

    ``namespace`` separates independent draws from the same banks. The GPU-1 smoke needs 20
    non-attempt rows that are *disjoint from the reportable bundle*, and without a namespace
    it would get the bundle's own first twenty rows -- the generator is deterministic, so
    "generate fewer rows" returns a prefix of the same sequence, not a different sample.
    Mixing the namespace into every digest makes the smoke a genuinely different draw while
    keeping both draws reproducible.

    Returns ``(rows, report)``. The report carries per-subtype diversity and the shortfalls,
    and :data:`SUPPLEMENT_DIVERSITY_BOUNDS` is enforced per subtype when
    ``enforce_diversity`` -- a pool that collapsed onto four templates is a lexical shortcut
    with a bundle wrapped around it, and shipping it would be worse than shipping nothing.
    """
    pool = [dict(r) for r in records]
    if not pool:
        raise ValueError("no conditioning records: the supplement is generated against them")
    ordered = sorted(pool, key=lambda r: str(r["conditioning_question"]))

    rows: list[SupplementRow] = []
    shortfalls: dict[str, dict] = {}

    def question_at(i: int) -> dict:
        return ordered[i % len(ordered)]

    # --- the five synthesised NONE subtypes ------------------------------------------
    #
    # The question is rotated through the index rather than sampled, so every subject is
    # represented and no author accumulates a disproportionate share of the non-attempts --
    # a pool concentrated on twenty authors would let "this author's rows are always NONE"
    # become a shortcut of its own.
    synthesised = {
        "refusal": lambda d, rec: _compose_refusal(d),
        "planning_process": lambda d, rec: _compose_planning(d),
        "tool_memory_status": lambda d, rec: _compose_tool(d),
        "subject_only": lambda d, rec: _compose_subject_only(
            d, (list(rec.get("subject_aliases")) or ["the subject"])[0]
        ),
        "off_topic": lambda d, rec: _compose_off_topic(d),
    }
    emitted: set[str] = set()

    def emit(row: SupplementRow) -> bool:
        """Keep the row unless it collides with one already in the pool.

        Collisions are expected and are not a bug: the yes-no narrowing bank has ~170
        surface forms and the question index has 95 entries, so drawing 110 rows from a
        finite composition WILL repeat one. Skipping and drawing again is the only response
        that keeps both properties the bundle needs -- no collisions, and the text still a
        pure function of its index -- where accepting the duplicate breaks the first and
        perturbing the digest until it clears breaks the second.

        The collision key differs by subtype, and the difference is load-bearing:

        * **composed subtypes dedupe on the CANDIDATE TEXT**, not the pair. A refusal is
          question-independent, so the same refusal can be drawn against two questions and
          form two perfectly valid pairs -- and if those two questions belong to authors on
          opposite sides of the group split, that one string is now in both train and
          development. Every such row is intended NONE, so a model can memorise the string
          and score on development without learning anything. Dedupe on the text and the
          straddle cannot arise.
        * **real-text subtypes dedupe on the PAIR.** They reuse real candidates on purpose
          and at most twice, always inside one subject group, so a reused donor lands wholly
          on one side of the split and no straddle is possible.
        """
        key = (
            row["pair_sha256"]
            if row["source_subtype"] in REAL_TEXT_SUBTYPES
            else hashlib.sha256(str(row["candidate_text"]).encode("utf-8")).hexdigest()
        )
        if key in emitted or row["pair_sha256"] in emitted:
            return False
        emitted.add(key)
        emitted.add(row["pair_sha256"])
        rows.append(row)
        return True

    # `off_topic` and `cross_question` are two flavours of ONE required subtype -- messages
    # that answer something other than what was asked -- so they share its quota. Synthetic
    # off-topic text is domain-neutral and cannot be mistaken for an attempt; cross-question
    # text is real, is about the right author, and is the harder and more realistic case.
    # Half each, because a pool of only the easy flavour would overstate NONE agreement.
    wanted_of = dict.fromkeys(synthesised, n_per_none_subtype)
    wanted_of["off_topic"] = n_per_none_subtype // 2
    n_cross_wanted = n_per_none_subtype - wanted_of["off_topic"]

    for subtype, compose in synthesised.items():
        wanted = wanted_of[subtype]
        kept = 0
        for attempt in range(wanted * 40):
            if kept >= wanted:
                break
            record = question_at(attempt * 7 + len(subtype))
            question = str(record["conditioning_question"])
            digest = _digest(GENERATOR_VERSION, namespace, subtype, str(attempt), question)
            frame, candidate = compose(digest, record)
            kept += emit(
                _row(
                    subtype=subtype,
                    intended_class="NONE",
                    question=question,
                    aliases=list(record.get("subject_aliases", ())),
                    candidate=candidate,
                    frame=frame,
                    subject_group=str(record.get("subject_group", "")),
                    subject_id=str(record.get("subject_id", "")),
                    conditioning_id=str(record.get("conditioning_id", "")),
                )
            )
        if kept < wanted:
            shortfalls[subtype] = {
                "wanted": wanted,
                "got": kept,
                "why": (
                    "the composition ran out of distinct (question, candidate) pairs before "
                    "the target. Widen a slot bank rather than lowering the target."
                ),
            }

    # --- cross_question: real text, same group, demonstrably different question -------
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in natural_rows:
        by_group[str(row.get("subject_group", ""))].append(dict(row))
    cross = _generate_cross_question(
        by_group, ordered, n_wanted=n_cross_wanted, namespace=namespace
    )
    cross = [r for r in cross if emit(r)]
    if len(cross) < n_cross_wanted:
        shortfalls["cross_question"] = {
            "wanted": n_cross_wanted,
            "got": len(cross),
            "why": (
                "a cross-question row needs two questions in ONE subject group whose "
                "content tokens barely overlap, and a real candidate for the other one. "
                "Groups without such a pair contribute nothing rather than a near-paraphrase."
            ),
        }

    # --- the two PARTIAL subtypes -----------------------------------------------------
    kept = 0
    for attempt in range(n_per_partial_subtype * 40):
        if kept >= n_per_partial_subtype:
            break
        record = question_at(attempt * 11 + 3)
        question = str(record["conditioning_question"])
        hint = question_type_hint(question)
        digest = _digest(
            GENERATOR_VERSION, namespace, "controlled_narrowing", str(attempt), question
        )
        frame, candidate = _compose_narrowing(digest, hint)
        kept += emit(
            _row(
                subtype="controlled_narrowing",
                intended_class="PARTIAL",
                question=question,
                aliases=list(record.get("subject_aliases", ())),
                candidate=candidate,
                frame=frame,
                subject_group=str(record.get("subject_group", "")),
                subject_id=str(record.get("subject_id", "")),
                conditioning_id=str(record.get("conditioning_id", "")),
                provenance={"question_type_hint": hint},
            )
        )
    if kept < n_per_partial_subtype:
        shortfalls["controlled_narrowing"] = {
            "wanted": n_per_partial_subtype,
            "got": kept,
            "why": (
                "the narrowing composition ran out of distinct pairs. The yes-no bank is "
                "the smallest; widen it rather than lowering the target."
            ),
        }

    fragments = list(_generate_fragments(natural_rows, n_wanted=n_per_partial_subtype))
    fragments = [r for r in fragments if emit(r)]
    if len(fragments) < n_per_partial_subtype:
        shortfalls["natural_fragment"] = {
            "wanted": n_per_partial_subtype,
            "got": len(fragments),
            "why": (
                "a fragment must cut at a clause boundary in the first 45% of the text "
                "and leave no sentence end behind it. Candidates whose first clause "
                "already completes an answer produce no honest fragment and are skipped."
            ),
        }

    # --- uniqueness, checked rather than assumed --------------------------------------
    seen: dict[str, str] = {}
    duplicates: list[str] = []
    for row in rows:
        key = row["pair_sha256"]
        if key in seen:
            duplicates.append(f"{seen[key]} == {row['audit_id']}")
        seen[key] = row["audit_id"]
    if duplicates:
        raise ValueError(
            f"{len(duplicates)} duplicate (question, candidate) pairs in the supplement, "
            f"first: {duplicates[0]}. A duplicated pair is counted twice in every rate."
        )

    report = diversity_report(rows)
    report["shortfalls"] = shortfalls
    report["generator_version"] = GENERATOR_VERSION
    report["namespace"] = namespace
    if enforce_diversity:
        failures = [
            f"{subtype}: {name} = {metrics[name]}"
            for subtype, metrics in report["by_subtype"].items()
            for name, (op, bound) in SUPPLEMENT_DIVERSITY_BOUNDS.items()
            if not (subtype in REAL_TEXT_SUBTYPES and name in _BOUNDS_NOT_APPLIED_TO_REAL_TEXT)
            and (
                metrics.get(name) is None
                or (metrics[name] < bound if op == ">=" else metrics[name] > bound)
            )
        ]
        if failures:
            raise ValueError(
                "the generated supplement fails its own diversity bounds: "
                + "; ".join(failures[:6])
                + ". A pool that collapsed onto a few templates teaches the detector the "
                "template, not answerability."
            )
    return rows, report


def _generate_cross_question(
    by_group: Mapping[str, Sequence[Mapping]],
    records: Sequence[Mapping],
    *,
    n_wanted: int,
    namespace: str = "bundle",
) -> list[SupplementRow]:
    """Pair each question with real candidates written for a different question, same group.

    Every admissible ``(question, donor)`` combination is enumerated first and the take is
    then a deterministic prefix of the sorted list, rather than one donor per question taken
    greedily. The greedy version came up half short: it capped the yield at "one row per
    question that has a distinct sibling", which the 95-question index does not supply
    enough of. Enumerating first also makes the take stable -- adding a group changes which
    rows exist but not the order of the ones that already did.

    Same-group is the invariant that must not be relaxed to raise the yield. A donor from
    another group would put one candidate's text on both sides of a group-disjoint split.
    """
    record_of_question = {str(r["conditioning_question"]): r for r in records}
    combinations: list[tuple[str, str, dict, dict]] = []
    for group in sorted(by_group):
        rows = by_group[group]
        for row in rows:
            candidate = str(row.get("candidate_text", "")).strip()
            donor_question = str(row["conditioning_question"])
            if not candidate:
                continue
            for question in sorted({str(r["conditioning_question"]) for r in rows}):
                if question == donor_question:
                    continue
                if not questions_are_distinct(question, donor_question):
                    continue
                record = record_of_question.get(question)
                if record is None:
                    continue
                combinations.append((group, question, dict(record), dict(row)))

    # Sorted by a content hash of the pair rather than by group, so the take is spread over
    # authors instead of exhausting the alphabetically first group before reaching the rest.
    combinations.sort(
        key=lambda c: _digest(
            GENERATOR_VERSION,
            namespace,
            "cross_question",
            c[1],
            str(c[3].get("candidate_text", "")),
        )
    )

    out: list[SupplementRow] = []
    used_donor: Counter = Counter()
    for group, question, record, donor in combinations:
        if len(out) >= n_wanted:
            break
        # No donor text more than twice, so the pool is not three authors' candidates
        # recycled through every question in their group.
        donor_key = str(donor.get("audit_id", "")) or str(donor.get("candidate_text", ""))
        if used_donor[donor_key] >= 2:
            continue
        used_donor[donor_key] += 1
        out.append(
            _row(
                subtype="cross_question",
                intended_class="NONE",
                question=question,
                aliases=list(record.get("subject_aliases", ())),
                candidate=str(donor["candidate_text"]),
                frame="cross_question/real-candidate",
                subject_group=group,
                subject_id=str(record.get("subject_id", "")),
                conditioning_id=str(record.get("conditioning_id", "")),
                provenance={
                    "donor_question": str(donor["conditioning_question"]),
                    "donor_audit_id": str(donor.get("audit_id", "")),
                    "same_subject_group": True,
                },
            )
        )
    return out


def _generate_fragments(rows: Sequence[Mapping], *, n_wanted: int) -> list[SupplementRow]:
    """Truncate real candidates mid-clause, keeping question, aliases and group intact."""
    out: list[SupplementRow] = []
    for row in sorted(rows, key=lambda r: str(r.get("audit_id", ""))):
        if len(out) >= n_wanted:
            break
        prefix = truncate_mid_clause(str(row.get("candidate_text", "")))
        if prefix is None:
            continue
        out.append(
            _row(
                subtype="natural_fragment",
                intended_class="PARTIAL",
                question=str(row["conditioning_question"]),
                aliases=list(row.get("subject_aliases", ())),
                candidate=prefix,
                frame="natural_fragment/mid-clause-cut",
                subject_group=str(row.get("subject_group", "")),
                subject_id=str(row.get("subject_id", "")),
                conditioning_id=str(row.get("conditioning_id", "")),
                provenance={
                    "source_audit_id": str(row.get("audit_id", "")),
                    "source_chars": len(str(row.get("candidate_text", ""))),
                    "prefix_chars": len(prefix),
                },
            )
        )
    return out


# =====================================================================================
# diversity
# =====================================================================================


def _ngrams(text: str, n: int = 5) -> Iterable[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return (tuple(words[i : i + n]) for i in range(max(0, len(words) - n + 1)))


def diversity_report(rows: Sequence[Mapping]) -> dict:
    """What the generated pool actually realised, per subtype and pooled.

    Four numbers per subtype, each catching a different way a generated pool goes wrong:

    * ``unique_candidate_rate`` -- outright duplication;
    * ``unique_5gram_rate`` -- phrases repeated across otherwise-different rows, which is
      what a template shows up as when the fillers vary but the scaffolding does not;
    * ``max_frame_share`` -- one frame dominating, the failure the frame banks exist to
      prevent;
    * ``max_leading_trigram_share`` -- every row opening the same way, which is the single
      cheapest lexical shortcut a detector can find and the one a human skimming the file
      is least likely to notice.
    """
    by_subtype: dict[str, dict] = {}
    for subtype in sorted({str(r["source_subtype"]) for r in rows}):
        subset = [r for r in rows if r["source_subtype"] == subtype]
        texts = [str(r["candidate_text"]) for r in subset]
        grams = Counter(g for t in texts for g in _ngrams(t))
        leading = Counter(tuple(re.findall(r"[a-z0-9']+", t.lower())[:3]) for t in texts)
        frames = Counter(str(r["frame_id"]) for r in subset)
        by_subtype[subtype] = {
            "n_rows": len(subset),
            "intended_class": sorted({str(r["intended_class"]) for r in subset}),
            "unique_pair_rate": (
                len({str(r["pair_sha256"]) for r in subset}) / len(subset) if subset else None
            ),
            "unique_candidate_rate": len(set(texts)) / len(texts) if texts else None,
            "unique_5gram_rate": (len(grams) / sum(grams.values())) if grams else None,
            "max_frame_share": (max(frames.values()) / len(subset)) if subset else None,
            "n_frames_used": len(frames),
            "max_leading_trigram_share": (max(leading.values()) / len(subset)) if subset else None,
            "mean_chars": (sum(len(t) for t in texts) / len(texts)) if texts else None,
            "n_subject_groups": len({str(r.get("subject_group", "")) for r in subset}),
        }
    all_texts = [str(r["candidate_text"]) for r in rows]
    return {
        "n_rows": len(rows),
        "by_intended_class": dict(sorted(Counter(str(r["intended_class"]) for r in rows).items())),
        "by_subtype": by_subtype,
        "pooled_unique_candidate_rate": (
            len(set(all_texts)) / len(all_texts) if all_texts else None
        ),
        "bounds": {k: list(v) for k, v in SUPPLEMENT_DIVERSITY_BOUNDS.items()},
        "why": (
            "a repeated refusal template is a lexical shortcut with a dataset wrapped "
            "around it. These numbers are enforced per subtype at generation time, not "
            "checked afterwards, because a pool that fails them should never reach a file."
        ),
    }
