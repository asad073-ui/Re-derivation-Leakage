"""``rdl graph-detector-v4-3-bundle`` -- the 1,019 rows, rebuilt so the inputs carry no population.

The raw audit is not touched. ``LABEL_AUDIT_KEY.json``, the two judge files and the two
reference passes stay byte-for-byte as they were frozen, because
``DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md`` quotes their hashes and rewriting one would
invalidate a pre-registration rather than correct it. This command writes a *derived*
bundle beside them.

Four things are different from what the v4.2 trainer built, and each is a defect that
would have shown up as detector quality:

**One alias builder.** ``natural_alias_index()`` joined aliases from the forget-policy
cohort, which contains protected authors and no retain authors. Retain rows therefore got
``aliases: []`` and protected rows did not, and "has aliases" separated the two populations
perfectly. Here every question -- both populations -- goes through
``extract_name_spans``, and the resulting parity is *measured* and written into the
manifest rather than asserted.

**One input shape.** Both populations serialise as exactly
``(conditioning_question, subject_aliases, candidate_text)``. The field is called
``conditioning_question`` and not ``protected_question`` because 300 of these rows ask
about authors that were never protected, and a field name that says otherwise invites a
reader -- or a future input builder -- to treat the name as a fact about the row.

**Group-disjoint splits.** v4.2 halved rows on ``audit_id`` parity, so two questions about
one author could straddle train and development. Splitting is on the subject group instead,
so an author is wholly on one side.

**A shortcut probe.** None of the above proves the leak is gone. :func:`shortcut_probe`
tries to recover population from the model's *own* inputs using only cheap surface
features -- alias count, question length, candidate length -- and reports the best
balanced accuracy any single feature achieves. If that number is high the bundle still
leaks population and the fix did not work, whatever the alias parity says.

Labels are optional
-------------------
The v4.3 labels do not exist yet: they come from the local judges on the GPU box. A bundle
built without ``--labels`` carries ``label: null`` on every pair and is a complete,
hashable description of the *inputs* -- which is exactly what the CPU phase is supposed to
freeze. Passing ``--labels`` later attaches an adjudicated file and records its hash.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..defenses.protected_store import ConditioningIndex
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_store import (
    CONDITIONING_INDEX_FILENAME,
    DEFAULT_AUDIT_DIR,
    DEFAULT_OUT_DIR,
    EVAL_KEY_FILENAME,
)

__all__ = [
    "BUNDLE_FILENAME",
    "BUNDLE_SCHEMA",
    "LABELS",
    "assign_splits",
    "detector_v4_3_bundle",
    "shortcut_probe",
]

BUNDLE_FILENAME = "DETECTOR_V4_3_PAIR_BUNDLE.json"
BUNDLE_SCHEMA = "graph-detector-v4-3-pair-bundle-v1"

LABELS = ("NONE", "PARTIAL", "ANSWER")

# The fraction of SUBJECT GROUPS held out for development. Groups rather than rows, so the
# realised row fraction drifts a little from this -- authors do not have equal numbers of
# audited rows. The manifest records what was actually realised.
DEFAULT_DEV_FRACTION = 0.5

# Everything the serialised model input may contain. Asserted per pair, so a field added
# to the bundle later cannot silently reach the tokenizer.
TOKENIZED_FIELDS = ("conditioning_question", "subject_aliases", "candidate_text")

# Joins question to candidate when hashing a pair. A separator that cannot occur in
# either field, so that ("a b", "c") and ("a", "b c") are different pairs rather than
# one collision that would be reported as a duplicate.
PAIR_SEPARATOR = "\x1f"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _unit_hash(text: str) -> float:
    """A deterministic float in [0, 1) from a string. The split's only source of randomness.

    Content-addressed rather than seeded, so adding a group later does not move the groups
    that were already assigned -- a seeded shuffle reshuffles everything and silently
    changes which authors a checkpoint was selected on.
    """
    return int(_sha256(text)[:16], 16) / float(1 << 64)


def assign_splits(
    groups_by_population: Mapping[str, Sequence[str]],
    *,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
) -> dict[str, str]:
    """``{subject_group: 'train'|'development'}``, stratified by population.

    Population is used here and only here: to make sure both sides of the split contain
    protected and retain subjects in similar proportion. It stratifies the assignment of
    *groups*; it never reaches a pair, a field or a tokenizer. Without it a hash-uniform
    split over 65 groups can put most of the 20 protected authors on one side by chance,
    and the protected-clean false-alarm rate is then measured over a handful of concepts.
    """
    out: dict[str, str] = {}
    for _population, groups in sorted(groups_by_population.items()):
        ranked = sorted(set(groups), key=lambda g: (_unit_hash(f"v4.3-split|{g}"), g))
        n_dev = round(len(ranked) * dev_fraction)
        for i, group in enumerate(ranked):
            out[group] = "development" if i < n_dev else "train"
    return out


def _auc(positive: Sequence[float], negative: Sequence[float]) -> float | None:
    """Rank AUC, ties at 0.5. Used to score one surface feature against population."""
    if not positive or not negative:
        return None
    wins = sum(1 for p in positive for n in negative if p > n)
    ties = sum(1 for p in positive for n in negative if p == n)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def _best_stump(values: Sequence[float], is_positive: Sequence[bool]) -> float:
    """Best balanced accuracy any single threshold on one feature reaches.

    Balanced rather than raw accuracy because the populations are 719/300: a classifier
    that answers "protected" every time scores 0.71 raw and 0.50 balanced, and only the
    second number says it learned nothing.
    """
    n_positive = sum(is_positive)
    n_negative = len(is_positive) - n_positive
    if not n_positive or not n_negative:
        return 0.5
    best = 0.5
    for threshold in sorted(set(values)):
        true_positive = sum(
            1 for v, p in zip(values, is_positive, strict=True) if p and v >= threshold
        )
        false_positive = sum(
            1 for v, p in zip(values, is_positive, strict=True) if not p and v >= threshold
        )
        balanced = 0.5 * (true_positive / n_positive + 1 - false_positive / n_negative)
        best = max(best, balanced, 1 - balanced)
    return best


def shortcut_probe(pairs: Sequence[Mapping], population_of: Mapping[str, str]) -> dict:
    """Can population be recovered from the model's own inputs by surface features alone?

    This is the check that the alias fix actually worked. Alias parity says the two
    populations get aliases at similar *rates*; it says nothing about whether retain
    questions are systematically shorter, or carry fewer alias spans, or arrive with
    longer candidates. Any of those would let a model separate the populations without
    learning answerability, and the recall number would again describe membership.

    Features are deliberately crude and population-blind by construction -- they are
    computed from the serialised input, which is all the model sees. A high number here
    does not say the model *will* take the shortcut; it says the shortcut is available,
    which under cross-entropy is enough to assume it will be taken.
    """
    features: dict[str, list[float]] = defaultdict(list)
    is_protected: list[bool] = []
    for pair in pairs:
        aliases = list(pair["subject_aliases"])
        features["n_aliases"].append(float(len(aliases)))
        features["alias_chars"].append(float(sum(len(a) for a in aliases)))
        features["question_chars"].append(float(len(pair["conditioning_question"])))
        features["question_words"].append(float(len(pair["conditioning_question"].split())))
        features["candidate_chars"].append(float(len(pair["candidate_text"])))
        features["candidate_words"].append(float(len(pair["candidate_text"].split())))
        is_protected.append(population_of[pair["audit_id"]] != "retain")

    per_feature = {}
    for name, values in sorted(features.items()):
        positive = [v for v, p in zip(values, is_protected, strict=True) if p]
        negative = [v for v, p in zip(values, is_protected, strict=True) if not p]
        per_feature[name] = {
            "auc_vs_population": _auc(positive, negative),
            "best_stump_balanced_accuracy": _best_stump(values, is_protected),
        }
    best_name = max(
        per_feature, key=lambda k: per_feature[k]["best_stump_balanced_accuracy"] or 0.0
    )

    # The decomposition that makes the headline number readable.
    #
    # Absolute balanced accuracy cannot be driven to 0.50 and should not be. The 719
    # protected questions are drawn from TOFU's forget10 split and the 300 retain
    # questions from retain90, and those two pools have different length distributions.
    # A question-conditioned detector is *supposed* to read the question, so whatever
    # population signal the question text itself carries is not something the bundle can
    # remove -- removing it would mean editing the frozen audit.
    #
    # What the bundle controls is the ALIAS channel, which v4.2 built asymmetrically. So
    # the quantity to watch is the excess: how much more separable the populations become
    # once aliases are added to the question the model already sees. An excess near zero
    # means the alias fix worked, whatever the absolute floor happens to be.
    def _peak(prefixes: tuple[str, ...]) -> tuple[str, float]:
        scored = {
            name: metrics["best_stump_balanced_accuracy"]
            for name, metrics in per_feature.items()
            if name.startswith(prefixes)
        }
        name = max(scored, key=lambda k: scored[k] or 0.0)
        # `_best_stump` returns 0.5 for a degenerate pool rather than None, so this is
        # total in practice; coercing keeps the arithmetic below honest if that changes.
        return name, float(scored[name] or 0.5)

    question_feature, question_floor = _peak(("question_",))
    alias_feature, alias_peak = _peak(("alias_", "n_aliases"))
    return {
        "n_pairs": len(pairs),
        "n_protected": sum(is_protected),
        "n_retain": len(is_protected) - sum(is_protected),
        "per_feature": per_feature,
        "worst_case_feature": best_name,
        "worst_case_balanced_accuracy": per_feature[best_name]["best_stump_balanced_accuracy"],
        "question_only_floor": {
            "feature": question_feature,
            "balanced_accuracy": question_floor,
            "why": (
                "the protected and retain questions come from two different TOFU splits "
                "with different length distributions. This much population signal is in "
                "the question text the detector must read, and no bundle construction can "
                "remove it without editing the frozen audit."
            ),
        },
        "alias_channel_excess": {
            "feature": alias_feature,
            "balanced_accuracy": alias_peak,
            "excess_over_question_floor": alias_peak - question_floor,
            "why": (
                "the number the alias fix is accountable for: how much MORE separable the "
                "populations become once aliases are added to the question. v4.2's value "
                "was +0.50 -- a perfect separator on top of the floor, because every "
                "protected row had aliases and no retain row did."
            ),
        },
        "interpretation": (
            "balanced accuracy of the single best surface feature at predicting protected "
            "vs retain from the model's own input; 0.50 is chance. Read "
            "alias_channel_excess, not the absolute worst case: the absolute number has an "
            "irreducible floor set by the two question pools, and only the excess is "
            "something bundle construction chose. Reported, not gated -- the acceptable "
            "excess is fixed in DETECTOR_V4_3_PROTECTED_STORE_PROTOCOL.md."
        ),
    }


def _load_labels(path: Path | None) -> tuple[dict[str, str], dict]:
    if path is None:
        return {}, {
            "source": None,
            "n_labels": 0,
            "why": (
                "no adjudicated label file was passed. The v4.3 labels come from the local "
                "judges on the GPU box; this bundle freezes the INPUTS, which is what the "
                "CPU phase is for."
            ),
        }
    if not path.exists():
        raise typer.BadParameter(f"{path} is absent")
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = row.get("answer_attempt")
        if label in LABELS:
            labels[str(row["audit_id"])] = str(label)
    return labels, {
        "source": str(path),
        "sha256": _sha256(path.read_text(encoding="utf-8")),
        "n_labels": len(labels),
        "distribution": dict(sorted(Counter(labels.values()).items())),
    }


def detector_v4_3_bundle(
    audit_dir: Path = typer.Option(DEFAULT_AUDIT_DIR, "--audit-dir"),
    store_dir: Path = typer.Option(
        DEFAULT_OUT_DIR,
        "--store-dir",
        help="where graph-detector-v4-3-build-store wrote its three artifacts.",
    ),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir"),
    labels: Path = typer.Option(
        None,
        "--labels",
        help="an adjudicated JSONL. Optional: without it every pair carries label null.",
    ),
    dev_fraction: float = typer.Option(DEFAULT_DEV_FRACTION, "--dev-fraction"),
) -> None:
    """Rebuild the 1,019-row pair bundle over the conditioning index, group-disjoint."""
    index_path = Path(store_dir) / CONDITIONING_INDEX_FILENAME
    key_path = Path(store_dir) / EVAL_KEY_FILENAME
    blind_path = Path(audit_dir) / "LABEL_AUDIT_JUDGE_A.jsonl"
    for path in (index_path, key_path, blind_path):
        if not path.exists():
            raise typer.BadParameter(
                f"{path} is absent. Run `rdl graph-detector-v4-3-build-store` first."
            )

    index = ConditioningIndex.load(index_path)
    eval_key = json.loads(key_path.read_text(encoding="utf-8"))
    eval_rows: dict[str, dict] = dict(eval_key.get("rows", {}))
    blind_rows = {
        str(json.loads(line)["audit_id"]): json.loads(line)
        for line in blind_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }

    label_of, label_provenance = _load_labels(labels)

    # ------------------------------------------------------------------ the split --
    groups_by_population: dict[str, list[str]] = defaultdict(list)
    for row in eval_rows.values():
        groups_by_population[str(row["population"])].append(str(row["subject_group"]))
    split_of_group = assign_splits(groups_by_population, dev_fraction=dev_fraction)

    # ------------------------------------------------------------------- the pairs --
    pairs: list[dict] = []
    seen_pair: dict[str, str] = {}
    n_missing_aliases: Counter = Counter()
    for audit_id in sorted(eval_rows):
        eval_row = eval_rows[audit_id]
        blind = blind_rows.get(audit_id)
        if blind is None:
            raise typer.BadParameter(
                f"row {audit_id} is in the evaluation key but not in {blind_path}. All "
                "1,019 rows must join."
            )
        question = str(blind["protected_question"])
        record = index.for_question(question)
        if record is None:
            raise typer.BadParameter(
                f"row {audit_id}'s question is absent from the conditioning index. The "
                "index and this bundle would then describe different questions."
            )
        candidate = str(blind["candidate_text"])
        population = str(eval_row["population"])
        if not record.subject_aliases:
            n_missing_aliases[population] += 1

        pair_key = _sha256(question + PAIR_SEPARATOR + candidate)
        if pair_key in seen_pair:
            raise typer.BadParameter(
                f"rows {seen_pair[pair_key]} and {audit_id} are the same "
                "(question, candidate) pair. A duplicated pair is counted twice in every "
                "rate and, if it straddles the split, is train/development contamination."
            )
        seen_pair[pair_key] = audit_id

        group = str(eval_row["subject_group"])
        pairs.append(
            {
                # --- the model's input, and nothing else ------------------------------
                "conditioning_question": question,
                "subject_aliases": list(record.subject_aliases),
                "candidate_text": candidate,
                # --- the target ------------------------------------------------------
                "label": label_of.get(audit_id),
                # --- bookkeeping. NEVER serialised into a tokenizer input -------------
                "audit_id": audit_id,
                "conditioning_id": record.conditioning_id,
                "subject_id": record.subject_id,
                "split": split_of_group[group],
                "pair_sha256": pair_key,
                "source_row_sha256": _sha256(dumps_canonical(blind)),
            }
        )

    # ------------------------------------------------------- disjointness, checked --
    #
    # Asserted rather than assumed. The split is computed from the group, so this can only
    # fail if a group were assigned twice -- but a group-disjointness claim that is not
    # checked is the claim v4.2 also made.
    groups_per_split: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        groups_per_split[pair["split"]].add(eval_rows[pair["audit_id"]]["subject_group"])
    overlap = sorted(groups_per_split["train"] & groups_per_split["development"])
    if overlap:
        raise typer.BadParameter(
            f"subject groups {overlap[:5]} appear in BOTH train and development. The "
            "bundle would train and select on the same authors."
        )

    subjects_per_split = {
        split: sorted({p["subject_id"] for p in pairs if p["split"] == split})
        for split in ("train", "development")
    }
    subject_overlap = sorted(
        set(subjects_per_split["train"]) & set(subjects_per_split["development"])
    )
    if subject_overlap:
        raise typer.BadParameter(f"subject ids {subject_overlap[:5]} straddle the split")

    population_of = {aid: str(row["population"]) for aid, row in eval_rows.items()}
    probe = shortcut_probe(pairs, population_of)

    # Alias parity, recomputed here over the PAIRS rather than over distinct questions, so
    # the number describes what the model is actually shown.
    parity = {}
    for population in sorted({*population_of.values()}):
        rows = [p for p in pairs if population_of[p["audit_id"]] == population]
        with_aliases = sum(1 for p in rows if p["subject_aliases"])
        parity[population] = {
            "n_pairs": len(rows),
            "n_with_aliases": with_aliases,
            "alias_coverage": with_aliases / len(rows) if rows else None,
            "mean_n_aliases": (
                sum(len(p["subject_aliases"]) for p in rows) / len(rows) if rows else None
            ),
        }

    split_composition = {
        split: {
            "n_pairs": sum(1 for p in pairs if p["split"] == split),
            "n_subjects": len(subjects_per_split[split]),
            "populations": dict(
                sorted(
                    Counter(
                        population_of[p["audit_id"]] for p in pairs if p["split"] == split
                    ).items()
                )
            ),
        }
        for split in ("train", "development")
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": BUNDLE_SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_pairs": len(pairs),
        "tokenized_fields": list(TOKENIZED_FIELDS),
        "fields_never_tokenized": [
            "population",
            "is_protected",
            "forget_id",
            "concept_id",
            "subject_id",
            "split",
            "stratum",
            "reference_answer",
            "judge identity",
            "detector score",
        ],
        "sources": {
            "audit_dir": str(audit_dir),
            "conditioning_index": str(index_path),
            "conditioning_index_fingerprint_sha256": index.fingerprint(),
            "eval_key": str(key_path),
            "raw_audit_unchanged": True,
        },
        "labels": label_provenance,
        "split_rule": {
            "unit": "subject group (TOFU author block)",
            "assignment": "content-addressed hash of the group key, stratified by population",
            "dev_fraction_requested": dev_fraction,
            "group_disjoint_verified": True,
            "composition": split_composition,
        },
        "alias_parity_by_population": parity,
        "n_pairs_without_aliases": dict(sorted(n_missing_aliases.items())),
        "shortcut_probe": probe,
        "why_conditioning_question": (
            "300 of these rows ask about authors that were never protected. The v4.1 field "
            "name `protected_question` asserts something false about them, and a field "
            "name that lies is how population re-enters a pipeline that removed it."
        ),
        "pairs": pairs,
    }
    # Hashed over CONTENT, with the build timestamp excluded. The trainer records this
    # digest as provenance, so a rebuild that changed it because the clock moved would make
    # every provenance check a false alarm -- and a check that cries wolf is a check that
    # gets turned off. Two bundles with the same digest are the same inputs, whenever they
    # were built.
    payload["bundle_sha256"] = _sha256(
        dumps_canonical({k: v for k, v in payload.items() if k != "built_at"})
    )
    atomic_json(out / BUNDLE_FILENAME, payload)

    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / BUNDLE_FILENAME),
                "n_pairs": len(pairs),
                "n_labelled": sum(1 for p in pairs if p["label"]),
                "split_composition": split_composition,
                "alias_parity_by_population": parity,
                "n_pairs_without_aliases": dict(sorted(n_missing_aliases.items())),
                "shortcut_worst_case_balanced_accuracy": probe["worst_case_balanced_accuracy"],
                "shortcut_worst_case_feature": probe["worst_case_feature"],
                "shortcut_question_only_floor": probe["question_only_floor"]["balanced_accuracy"],
                "shortcut_alias_channel_excess": probe["alias_channel_excess"][
                    "excess_over_question_floor"
                ],
                "group_disjoint_verified": True,
                "bundle_sha256": payload["bundle_sha256"],
            }
        )
    )
