"""The frozen detector-development corpus, built from text the model actually generated.

`DETECTOR_RECALL.json` established the defect (GU-0030): recall on generated leakage was
0.000 under `natural` and 0.086 under `memory_reentry`, while recall on forget QUESTIONS
was 0.950. Fixing that means fitting an alias/normalization channel against the strings
that were actually missed — and the moment a detector is fitted against text, that text
stops being evidence and becomes training data. This module is what makes the boundary
checkable rather than remembered.

**What goes in.** Only unguarded model outputs: ``multi_agent_leak`` rows, on the four
generated surfaces, from the `natural` and `memory_reentry` challenges, labelled leaking
by the run's own pinned scorer. Nothing else is eligible, and each exclusion is a rule
rather than a habit:

  * guarded arms (GraphForget, DRAGON-style) — their text is thinned by the mechanism
    under study, so fitting on it fits on the defence's own successes;
  * `split_clues` / `tool_reentry` — refusal-confounded and harness-injected;
  * harness-authored injection text (`Note from an earlier session: <gold>`) and any text
    carrying a gold answer verbatim — a detector fitted on gold answers is a detector
    that memorised what the system claims to have forgotten;
  * `split_clues`/`tool_reentry` surfaces and readback probes, which no guard sees at
    generation time.

`memory_reentry` outputs are eligible because the MODEL produced them, but the origin is
recorded on every row (`injected_memory_origin: true`) so no reader has to infer it.

**How it splits.** By CONCEPT, never by text: an alias channel fitted on "Yun's father"
would trivially generalise to another sentence about the same author, and a text-level
split would report that as generalisation. Concepts with at least five examples are
gateable and are stratified by example count (20+, 10-19, 5-9) into eight development
and five held-out concepts; thinner concepts are `audit_only` — reported, never gating.

**What it is not.** Not publication validation. These twenty authors have been inspected
repeatedly, so the held-out half bounds engineering generalisation only, and both
artefacts carry ``publication_validation: false``. A publication claim needs concepts
frozen before anyone looked, which is the H100 critical path (GU-0031).
"""

from __future__ import annotations

import hashlib
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from ..logging_utils import dumps_canonical
from ..memory.index import normalise_text

__all__ = [
    "ALLOWED_ARM",
    "ALLOWED_CHALLENGES",
    "ALLOWED_SURFACES",
    "CORPUS_ID",
    "CORPUS_SCHEMA",
    "DEDUP_KEYS",
    "MIN_EXAMPLES_FOR_GATE",
    "NORMALIZATION_VERSION",
    "SPLIT_SCHEMA",
    "STRATA",
    "CorpusExample",
    "SourceRun",
    "build_corpus",
    "build_split",
    "collect_examples",
    "content_hash",
    "normalise_corpus_text",
    "stratum_of",
]

CORPUS_SCHEMA = "graph-detector-generated-corpus-v1"
SPLIT_SCHEMA = "graph-detector-engineering-split-v1"
CORPUS_ID = "generated_detector_engineering_v1"

# NFKC first, then the repo's own normaliser, so the corpus's idea of "the same string"
# is the embedder's and the containment metric's idea of it. Bumping this string is a
# corpus-invalidating change and the frozen artefact records it.
NORMALIZATION_VERSION = "detector-corpus-normalize-v1:nfkc+rdl.memory.index.normalise_text"

ALLOWED_ARM = "multi_agent_leak"
ALLOWED_SURFACES: tuple[str, ...] = ("agent_message", "edge_payload", "stored_node", "final_text")
# `split_clues` and `tool_reentry` are deliberately absent: both are refusal-confounded
# (96% and 100% refusal in the discovery study) and both record the harness's own
# gold-derived text coming back, so neither says anything about what the MODEL produces.
ALLOWED_CHALLENGES: tuple[str, ...] = ("natural", "memory_reentry")
DEDUP_KEYS: tuple[str, ...] = ("concept_id", "normalized_text")

MIN_EXAMPLES_FOR_GATE = 5
N_DEVELOPMENT_CONCEPTS = 8
N_HELDOUT_CONCEPTS = 5
# Stratum boundaries, largest first. Stratification is on EXAMPLE COUNT only — never on
# a detector score, which would select the held-out concepts using the thing being tested.
STRATA: tuple[tuple[str, int], ...] = (("20+", 20), ("10-19", 10), ("5-9", MIN_EXAMPLES_FOR_GATE))

# The harness's seeded-memory prefix. Anything carrying it is the injector's sentence,
# not the model's.
_INJECTION_MARKERS: tuple[str, ...] = ("note from an earlier session",)


def normalise_corpus_text(text: str) -> str:
    """The corpus's single definition of "the same string"."""
    return normalise_text(unicodedata.normalize("NFKC", text or ""))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(payload: Mapping) -> str:
    """Hash of everything but the hash field itself.

    Canonical JSON, so the digest is a function of the artefact's CONTENT and not of key
    order or indentation. Changing a source row, a normalized text, a concept assignment
    or a split membership all change it — which is what the freeze tests assert.
    """
    body = {k: v for k, v in payload.items() if k != "content_sha256"}
    return hashlib.sha256(dumps_canonical(body).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SourceRun:
    """One run the corpus draws from, with the provenance needed to re-derive it."""

    run_id: str
    challenge: str
    protocol: str
    scorer_version: str
    release: str | None = None
    shard_ledger_sha256: str | None = None
    n_shards: int = 0
    n_rows: int = 0
    forget_policy_fingerprint: str | None = None
    detector_version: str | None = None

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "challenge": self.challenge,
            "protocol": self.protocol,
            "scorer_version": self.scorer_version,
            "release": self.release,
            "shard_ledger_sha256": self.shard_ledger_sha256,
            "n_shards": self.n_shards,
            "n_rows": self.n_rows,
            "forget_policy_fingerprint": self.forget_policy_fingerprint,
            "detector_version": self.detector_version,
        }


@dataclass(frozen=True)
class CorpusExample:
    concept_id: str
    text: str
    normalized_text: str
    surface: str
    challenge: str
    origin_run: str
    trajectory_id: str
    injected_memory_origin: bool
    # Every surface this exact string was observed on. One agent message is usually also
    # the edge payload, the stored node and the final answer, and deduplicating to the
    # first-seen surface would report a corpus that is 100% `agent_message` — true of the
    # dedup key, false about what a guard has to act on.
    surfaces: tuple[str, ...] = ()

    @property
    def example_id(self) -> str:
        """Stable under the dedup keys, so the same leak from two runs is one example."""
        return _sha256(f"{self.concept_id}\0{self.normalized_text}")[:16]

    def to_dict(self) -> dict:
        return {
            "example_id": self.example_id,
            "concept_id": self.concept_id,
            "surface": self.surface,
            "surfaces": list(self.surfaces or (self.surface,)),
            "challenge": self.challenge,
            "origin_run": self.origin_run,
            "trajectory_id": self.trajectory_id,
            "injected_memory_origin": self.injected_memory_origin,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "normalized_sha256": _sha256(self.normalized_text),
        }


@dataclass
class CollectionAudit:
    """Every row the rules threw away, counted by the rule that threw it away."""

    wrong_arm: int = 0
    wrong_challenge: int = 0
    wrong_surface: int = 0
    unjudged: int = 0
    clean: int = 0
    harness_injection: int = 0
    gold_verbatim: int = 0
    empty_after_normalization: int = 0
    no_concept: int = 0
    duplicates_within_concept: int = 0
    excluded_examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "wrong_arm": self.wrong_arm,
            "wrong_challenge": self.wrong_challenge,
            "wrong_surface": self.wrong_surface,
            "unjudged": self.unjudged,
            "clean": self.clean,
            "harness_injection": self.harness_injection,
            "gold_verbatim": self.gold_verbatim,
            "empty_after_normalization": self.empty_after_normalization,
            "no_concept": self.no_concept,
            "duplicates_within_concept": self.duplicates_within_concept,
            "excluded_examples": self.excluded_examples[:20],
        }


def _generated_texts(row: Mapping) -> list[tuple[str, str]]:
    """``(surface, text)`` for the four surfaces a guard has to act on.

    Same four surfaces `eval.concept_recall.generated_texts` measures recall over, and
    deliberately not the readback probe: a probe text is a counterfactual second episode
    and no defence is asked to guard it at generation time.
    """
    raw = row.get("raw_outputs", {}) or {}
    out: list[tuple[str, str]] = []
    out += [("agent_message", str(t)) for t in raw.get("agent_messages", []) or []]
    out += [("edge_payload", str(t)) for t in raw.get("released_edge_payloads", []) or []]
    out += [
        ("stored_node", str(e.get("content", ""))) for e in row.get("memory_evidence", []) or []
    ]
    final = str(row.get("final_text", ""))
    if final:
        out.append(("final_text", final))
    return [(s, t) for s, t in out if t and t.strip()]


def _is_harness_text(normalized: str) -> bool:
    return any(marker in normalized for marker in _INJECTION_MARKERS)


def collect_examples(
    rows: Iterable[Mapping],
    *,
    label,
    run_id: str,
    challenge: str,
    audit: CollectionAudit | None = None,
) -> tuple[list[CorpusExample], CollectionAudit]:
    """Eligible examples from one run's raw shards.

    ``label(reference, candidate) -> bool | None`` is the run's own pinned-scorer verdict,
    looked up out of its scoring cache. ``None`` means the pair was never judged, and an
    unjudged text is dropped rather than assumed clean — assuming clean would silently
    delete exactly the hard cases the corpus exists to hold.
    """
    audit = audit or CollectionAudit()
    if challenge not in ALLOWED_CHALLENGES:
        raise ValueError(
            f"challenge '{challenge}' is not eligible for the detector corpus "
            f"{list(ALLOWED_CHALLENGES)}: the injected challenges are refusal-confounded "
            "and record the harness's own gold-derived text"
        )
    injected = challenge != "natural"

    out: list[CorpusExample] = []
    seen: set[tuple[str, str]] = set()
    surfaces: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if str(row.get("arm", "")) != ALLOWED_ARM:
            audit.wrong_arm += 1
            continue
        if str(row.get("challenge", "natural")) != challenge:
            audit.wrong_challenge += 1
            continue
        concept = str(row.get("concept_id", ""))
        if not concept:
            audit.no_concept += 1
            continue
        reference = str(row.get("reference_answer", ""))
        gold = normalise_corpus_text(reference)
        trajectory = str(row.get("trajectory_id", ""))

        for surface, text in _generated_texts(row):
            if surface not in ALLOWED_SURFACES:
                audit.wrong_surface += 1
                continue
            normalized = normalise_corpus_text(text)
            if not normalized:
                audit.empty_after_normalization += 1
                continue
            if _is_harness_text(normalized):
                audit.harness_injection += 1
                continue
            # A verbatim gold answer is the evaluator's string, not a generation the
            # detector has to learn to recognise. Substring, not equality: the copy that
            # comes back from the store usually arrives inside a sentence.
            if gold and (normalized == gold or gold in normalized):
                audit.gold_verbatim += 1
                continue
            leaks = label(reference, text)
            if leaks is None:
                audit.unjudged += 1
                continue
            if not leaks:
                audit.clean += 1
                continue
            key = (concept, normalized)
            if key in seen:
                audit.duplicates_within_concept += 1
                surfaces.setdefault(key, []).append(surface)
                continue
            seen.add(key)
            surfaces[key] = [surface]
            out.append(
                CorpusExample(
                    concept_id=concept,
                    text=text,
                    normalized_text=normalized,
                    surface=surface,
                    challenge=challenge,
                    origin_run=run_id,
                    trajectory_id=trajectory,
                    injected_memory_origin=injected,
                )
            )
    return [
        replace(
            example,
            surfaces=tuple(
                sorted(
                    set(surfaces[(example.concept_id, example.normalized_text)]),
                    key=ALLOWED_SURFACES.index,
                )
            ),
        )
        for example in out
    ], audit


def stratum_of(n_examples: int) -> str:
    for name, floor in STRATA:
        if n_examples >= floor:
            return name
    return "thin"


def _dedup_across_runs(
    examples: Sequence[CorpusExample],
) -> tuple[list[CorpusExample], list[dict], list[dict]]:
    """Collapse ``(concept, normalized_text)`` and quarantine cross-concept collisions.

    A normalized string that appears under more than one concept cannot supervise either
    of them — fitting on it teaches the detector to fire on the wrong author, which is
    precisely what `recall_correct_concept` refuses to count as a catch. Those go to the
    ambiguity audit and out of the fitting set.
    """
    by_key: dict[tuple[str, str], CorpusExample] = {}
    duplicates: list[dict] = []
    for example in sorted(
        examples, key=lambda e: (e.concept_id, e.normalized_text, e.origin_run, e.surface)
    ):
        key = (example.concept_id, example.normalized_text)
        if key in by_key:
            duplicates.append(
                {
                    "concept_id": example.concept_id,
                    "origin_run": example.origin_run,
                    "surface": example.surface,
                    "normalized_sha256": _sha256(example.normalized_text),
                }
            )
            kept_example = by_key[key]
            by_key[key] = replace(
                kept_example,
                surfaces=tuple(
                    sorted(
                        set(kept_example.surfaces) | set(example.surfaces),
                        key=ALLOWED_SURFACES.index,
                    )
                ),
            )
            continue
        by_key[key] = example

    concepts_per_text: dict[str, set[str]] = {}
    for concept, normalized in by_key:
        concepts_per_text.setdefault(normalized, set()).add(concept)
    ambiguous = {t for t, concepts in concepts_per_text.items() if len(concepts) > 1}

    kept = [e for key, e in sorted(by_key.items()) if key[1] not in ambiguous]
    ambiguity_audit = [
        {
            "normalized_sha256": _sha256(text),
            "concept_ids": sorted(concepts_per_text[text]),
            "excerpt": text[:160],
        }
        for text in sorted(ambiguous)
    ]
    return kept, ambiguity_audit, duplicates


def build_corpus(
    per_run: Sequence[tuple[SourceRun, Sequence[CorpusExample]]],
    *,
    audit: CollectionAudit,
    registry_fingerprint: str | None = None,
) -> dict:
    """The frozen corpus artefact. Deterministic given the same runs and scoring caches."""
    examples: list[CorpusExample] = []
    for _source, batch in per_run:
        examples.extend(batch)
    kept, ambiguity_audit, duplicates = _dedup_across_runs(examples)

    counts: dict[str, int] = {}
    for example in kept:
        counts[example.concept_id] = counts.get(example.concept_id, 0) + 1
    concepts = {
        cid: {
            "n_examples": n,
            "stratum": stratum_of(n),
            "gateable": n >= MIN_EXAMPLES_FOR_GATE,
        }
        for cid, n in sorted(counts.items())
    }
    # The same discipline `eval.graph_concentration` applies to a leakage rate: 502
    # examples spread over 18 concepts is not 502 independent observations, and a macro
    # gate over concepts is the reason this number has to be visible next to it.
    total = sum(counts.values()) or 1
    shares = sorted((n / total for n in counts.values()), reverse=True)
    concentration = {
        "n_examples": total,
        "top_concept_share": round(shares[0], 6) if shares else 0.0,
        "top_3_concept_share": round(sum(shares[:3]), 6),
        "herfindahl": round(sum(s * s for s in shares), 6),
        "note": (
            "example counts are concentrated by construction: an author the model "
            "paraphrases often produces more leaking strings. Fit and gate on MACRO "
            "recall over concepts, never on the micro count alone."
        ),
    }

    surface_counts: dict[str, int] = {}
    challenge_counts: dict[str, int] = {}
    for example in kept:
        for surface in example.surfaces or (example.surface,):
            surface_counts[surface] = surface_counts.get(surface, 0) + 1
        challenge_counts[example.challenge] = challenge_counts.get(example.challenge, 0) + 1

    payload = {
        "schema": CORPUS_SCHEMA,
        "corpus_id": CORPUS_ID,
        # The two claims a reader should not have to reconstruct from the code.
        "uses_gold_answers": False,
        "publication_validation": False,
        "role": "detector engineering corpus; NOT untouched publication validation",
        "normalization_version": NORMALIZATION_VERSION,
        "dedup_keys": list(DEDUP_KEYS),
        "allowed_arm": ALLOWED_ARM,
        "allowed_surfaces": list(ALLOWED_SURFACES),
        "allowed_challenges": list(ALLOWED_CHALLENGES),
        "label_source": "the run's own pinned NLI scorer, read out of its scoring cache",
        "registry_fingerprint": registry_fingerprint,
        "sources": [source.to_dict() for source, _ in per_run],
        "n_examples": len(kept),
        "n_concepts": len(concepts),
        "n_gateable_concepts": sum(1 for c in concepts.values() if c["gateable"]),
        "concepts": concepts,
        "concentration": concentration,
        "examples_per_surface": dict(sorted(surface_counts.items())),
        "examples_per_challenge": dict(sorted(challenge_counts.items())),
        "examples": [e.to_dict() for e in kept],
        "audits": {
            "collection": audit.to_dict(),
            "ambiguous_normalized_texts": ambiguity_audit,
            "n_ambiguous_normalized_texts": len(ambiguity_audit),
            "cross_run_duplicates": duplicates[:20],
            "n_cross_run_duplicates": len(duplicates),
        },
        "note": (
            "every example is text the UNGUARDED arm generated and the run's pinned scorer "
            "judged leaking. No gold answers, no harness-authored injection text, no "
            "guarded-arm output, no readback probes."
        ),
    }
    payload["content_sha256"] = content_hash(payload)
    return payload


def build_split(
    corpus: Mapping,
    *,
    n_development: int = N_DEVELOPMENT_CONCEPTS,
    n_heldout: int = N_HELDOUT_CONCEPTS,
) -> dict:
    """Split the gateable concepts deterministically, by concept and by count alone.

    Apportionment is largest-remainder over the count strata, so the held-out half is not
    accidentally all-thin or all-fat; within a stratum the concepts are ordered by
    ``(-n_examples, concept_id)`` and held out from the SECOND position onward, which
    keeps the single largest concept in development where the threshold is chosen.
    """
    concepts = dict(corpus["concepts"])
    gateable = {cid: meta for cid, meta in concepts.items() if meta["gateable"]}
    audit_only = sorted(cid for cid in concepts if cid not in gateable)
    if len(gateable) < n_development + n_heldout:
        raise ValueError(
            f"{len(gateable)} concepts have at least {MIN_EXAMPLES_FOR_GATE} examples, but "
            f"the split needs {n_development + n_heldout}. Widen the corpus rather than "
            "lowering the gate: a held-out concept with three examples cannot fail a "
            "recall gate for any reason a reader would believe."
        )

    by_stratum: dict[str, list[str]] = {}
    for cid, meta in gateable.items():
        by_stratum.setdefault(str(meta["stratum"]), []).append(cid)
    for cids in by_stratum.values():
        cids.sort(key=lambda c: (-int(gateable[c]["n_examples"]), c))

    order = [name for name, _floor in STRATA if name in by_stratum]
    total = sum(len(by_stratum[name]) for name in order)
    exact = {name: len(by_stratum[name]) * n_heldout / total for name in order}
    quota = {name: int(exact[name]) for name in order}
    # Largest remainder, ties broken by the fixed stratum order: deterministic, and it
    # never depends on dict iteration order.
    remaining = n_heldout - sum(quota.values())
    for name in sorted(order, key=lambda n: (-(exact[n] - quota[n]), order.index(n)))[:remaining]:
        quota[name] += 1

    heldout: list[str] = []
    for name in order:
        take = quota[name]
        if not take:
            continue
        candidates = by_stratum[name]
        # Skip position 0 while there is room to: the biggest concept in each stratum
        # anchors development.
        picks = candidates[1 : 1 + take] if len(candidates) > take else candidates[:take]
        heldout.extend(picks)
    heldout = sorted(heldout)
    development = sorted(cid for cid in gateable if cid not in set(heldout))

    if len(development) != n_development or len(heldout) != n_heldout:
        raise ValueError(
            f"split produced {len(development)} development and {len(heldout)} held-out "
            f"concepts, expected {n_development}/{n_heldout}"
        )

    dev_set, held_set = set(development), set(heldout)
    dev_rows = [e for e in corpus["examples"] if e["concept_id"] in dev_set]
    held_rows = [e for e in corpus["examples"] if e["concept_id"] in held_set]
    overlaps = {
        "concept_overlap": sorted(dev_set & held_set),
        "normalized_text_overlap": sorted(
            {r["normalized_sha256"] for r in dev_rows} & {r["normalized_sha256"] for r in held_rows}
        ),
        "trajectory_overlap": sorted(
            {r["trajectory_id"] for r in dev_rows if r["trajectory_id"]}
            & {r["trajectory_id"] for r in held_rows if r["trajectory_id"]}
        ),
    }
    if any(overlaps.values()):
        raise ValueError(f"the split is not disjoint: {overlaps}")

    payload = {
        "schema": SPLIT_SCHEMA,
        "split_id": CORPUS_ID,
        # Not "validation". These twenty authors have been inspected repeatedly; the
        # held-out half bounds ENGINEERING generalisation and nothing more.
        "split_role": "engineering_holdout",
        "publication_validation": False,
        "corpus_sha256": corpus["content_sha256"],
        "corpus_id": corpus["corpus_id"],
        "normalization_version": corpus["normalization_version"],
        "split_unit": "concept",
        "stratified_by": "n_examples",
        "min_examples_for_gate": MIN_EXAMPLES_FOR_GATE,
        "strata": {name: sorted(by_stratum.get(name, [])) for name, _floor in STRATA},
        "heldout_quota_per_stratum": quota,
        "development_concepts": development,
        "heldout_concepts": heldout,
        "audit_only_concepts": audit_only,
        "counts": {
            "n_development_concepts": len(development),
            "n_heldout_concepts": len(heldout),
            "n_audit_only_concepts": len(audit_only),
            "n_development_examples": len(dev_rows),
            "n_heldout_examples": len(held_rows),
        },
        "disjointness": {k: len(v) for k, v in overlaps.items()},
        "usage": {
            "threshold_selection": "development_concepts only",
            "heldout_evaluation": "once, after the detector is frozen",
            "audit_only": "reported, never a pass/fail gate",
        },
        "note": (
            "fresh concepts, frozen before anyone inspected them, are still required for "
            "any publication claim. This split does not provide them."
        ),
    }
    payload["content_sha256"] = content_hash(payload)
    return payload
