"""``rdl graph-detector-v4-4-*`` -- the supplement, the bundle, the panel and the smoke fixture.

Four commands, in the order they must run:

``supplement``
    Generates the ~714 non-attempt and partial rows the v4.3 audit never contained, and
    refuses to write a pool that fails its own diversity bounds.

``bundle``
    The ~1,733-row v4.4 bundle: the original 1,019 pairs rebuilt under the *same*
    conditioning index and the *same* group split as v4.3, plus the supplement. Same split
    rule on purpose -- a v4.4 that also reshuffled the authors would confound "the bundle
    changed" with "the split changed" in every comparison against v4.3.

``panel``
    The frozen 600-row calibration panel: 200 rows per intended class, drawn before any
    judge runs, on which the primary kappa gate is computed.

``judge-smoke-fixture``
    60 rows disjoint from the bundle, 20 per intended class, for the GPU-1 rubric check.

Every v4.3 artifact is read and none is written. ``DETECTOR_V4_3_PAIR_BUNDLE.json``, the
two judge files, the pins and the store keep their bytes, because
``DETECTOR_V4_3_PROTECTED_STORE_PROTOCOL.md`` quotes their hashes and a failed experiment
whose evidence was edited afterwards stops being evidence.

Why the panel is balanced, and what that does not buy
-----------------------------------------------------
v4.3's kappa of 0.404 sat under 74.7% raw agreement because 65-81% of rows were one class.
Rebalancing raises kappa mechanically, and a protocol that rebalanced and declared victory
would have measured its own sampling. So the panel is balanced *and* four other things are
required alongside it: raw agreement >= 0.85, per-class agreement floors, the full-mixture
kappa reported beside it as a diagnostic, and the 184-row ``A=PARTIAL/B=ANSWER`` cell
addressed in the rubric rather than in the sampler. If the panel passes while the mixture
kappa stays near 0.40 and the PARTIAL cell is still the dominant disagreement, the report
says so and the boundary is still broken.
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..defenses.protected_store import ConditioningIndex
from ..eval.detector_v4_4 import BUNDLE_SCHEMA, PANEL_SCHEMA, sha256_text
from ..eval.detector_v4_4_supplement import (
    GENERATOR_VERSION,
    NONE_SUBTYPES,
    PARTIAL_SUBTYPES,
    diversity_report,
    generate_supplement,
    question_type_hint,
)
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_bundle import PAIR_SEPARATOR, TOKENIZED_FIELDS, assign_splits, shortcut_probe
from .detector_v4_3_store import (
    CONDITIONING_INDEX_FILENAME,
    DEFAULT_AUDIT_DIR,
    DEFAULT_OUT_DIR,
    EVAL_KEY_FILENAME,
)

__all__ = [
    "ANSWER_INTENT_STRATA",
    "BUNDLE_FILENAME",
    "PANEL_CLASS_SIZE",
    "PANEL_FILENAME",
    "SMOKE_FILENAME",
    "SUPPLEMENT_FILENAME",
    "balanced_take",
    "detector_v4_4_bundle",
    "detector_v4_4_judge_smoke_fixture",
    "detector_v4_4_panel",
    "detector_v4_4_supplement",
]

DEFAULT_V4_4_DIR = Path("data/cohorts/graph_unlearning_v1/detector_v4_4")

SUPPLEMENT_FILENAME = "DETECTOR_V4_4_SUPPLEMENT.json"
BUNDLE_FILENAME = "DETECTOR_V4_4_PAIR_BUNDLE.json"
PANEL_FILENAME = "DETECTOR_V4_4_CALIBRATION_PANEL.json"
SMOKE_FILENAME = "V4_4_JUDGE_SMOKE_60.jsonl"

SUPPLEMENT_SCHEMA = "graph-detector-v4-4-supplement-v1"
SMOKE_SCHEMA = "graph-detector-v4-4-judge-smoke-v1"

PANEL_CLASS_SIZE = 200

# Where intended-ANSWER panel rows come from. Both are FROZEN SAMPLING STRATA from the v4.1
# audit, assigned before any v4.3 or v4.4 label existed:
#
#   natural_leaking      every row the run's NLI+ROUGE scorer called leaking. A row that
#                        conveys the reference answer necessarily attempted it.
#   clean_hard_negative  the highest lexical-answerability rows among NLI-clean ones --
#                        by construction where wrong answer attempts concentrate, and v4.3
#                        confirmed it: both judges independently labelled its rows ANSWER.
#
# Deliberately NOT "rows both v4.3 judges called ANSWER". That would select the rows the
# old rubric found easy and hand the new panel a kappa it did not earn. These strata were
# fixed by a generator, not by an annotator, so they carry no agreement signal at all.
ANSWER_INTENT_STRATA: tuple[str, ...] = ("natural_leaking", "clean_hard_negative")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _unit_hash(text: str) -> float:
    return int(sha256_text(text)[:16], 16) / float(1 << 64)


def _load_inputs(audit_dir: Path, store_dir: Path) -> tuple[list[dict], list[dict], dict]:
    """``(conditioning records with groups, natural rows, eval key rows)``.

    Joined exactly the way the v4.3 bundle joins them -- question text is the only key a
    blinded judge file exposes -- so the 1,019 rows this produces are the same 1,019 pairs.
    """
    index_path = Path(store_dir) / CONDITIONING_INDEX_FILENAME
    key_path = Path(store_dir) / EVAL_KEY_FILENAME
    blind_path = Path(audit_dir) / "LABEL_AUDIT_JUDGE_A.jsonl"
    for path in (index_path, key_path, blind_path):
        if not path.exists():
            raise typer.BadParameter(
                f"{path} is absent. Run `rdl graph-detector-v4-3-build-store` first; v4.4 "
                "reads the v4.3 store and never rewrites it."
            )

    index = ConditioningIndex.load(index_path)
    eval_rows: dict[str, dict] = dict(
        json.loads(key_path.read_text(encoding="utf-8")).get("rows", {})
    )
    blind_rows = {str(r["audit_id"]): r for r in _read_jsonl(blind_path)}

    record_of_question = {r.conditioning_question: r for r in index.records()}
    group_of_conditioning: dict[str, str] = {}
    natural: list[dict] = []
    for audit_id in sorted(eval_rows):
        blind = blind_rows.get(audit_id)
        if blind is None:
            raise typer.BadParameter(
                f"row {audit_id} is in the evaluation key but not in {blind_path}. All "
                "1,019 rows must join, exactly as in v4.3."
            )
        question = str(blind["protected_question"])
        record = record_of_question.get(question)
        if record is None:
            raise typer.BadParameter(
                f"row {audit_id}'s question is absent from the conditioning index."
            )
        eval_row = eval_rows[audit_id]
        group = str(eval_row["subject_group"])
        group_of_conditioning[record.conditioning_id] = group
        natural.append(
            {
                "audit_id": audit_id,
                "conditioning_question": question,
                "subject_aliases": list(record.subject_aliases),
                "candidate_text": str(blind["candidate_text"]),
                "conditioning_id": record.conditioning_id,
                "subject_id": record.subject_id,
                "subject_group": group,
                "population": str(eval_row["population"]),
                "stratum": str(eval_row.get("stratum", "")),
                "concept_id": str(eval_row.get("concept_id", "")),
                "source_row_sha256": sha256_text(dumps_canonical(blind)),
            }
        )

    records = [
        {
            "conditioning_question": r.conditioning_question,
            "subject_aliases": list(r.subject_aliases),
            "conditioning_id": r.conditioning_id,
            "subject_id": r.subject_id,
            "subject_group": group_of_conditioning.get(r.conditioning_id, ""),
        }
        for r in index.records()
    ]
    return records, natural, eval_rows


# =====================================================================================
# supplement
# =====================================================================================


def detector_v4_4_supplement(
    audit_dir: Path = typer.Option(DEFAULT_AUDIT_DIR, "--audit-dir"),
    store_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--store-dir"),
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    n_none: int = typer.Option(100, "--n-per-none-subtype"),
    n_partial: int = typer.Option(110, "--n-per-partial-subtype"),
) -> None:
    """Generate the non-attempt and partial supplement. Deterministic; safe to re-run."""
    records, natural, _ = _load_inputs(audit_dir, store_dir)
    rows, report = generate_supplement(
        records,
        natural_rows=natural,
        n_per_none_subtype=n_none,
        n_per_partial_subtype=n_partial,
    )
    payload = {
        "schema": SUPPLEMENT_SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator_version": GENERATOR_VERSION,
        "n_rows": len(rows),
        "none_subtypes": list(NONE_SUBTYPES),
        "partial_subtypes": list(PARTIAL_SUBTYPES),
        "diversity": report,
        "intended_class_is_not_a_label": (
            "intended_class records how a row was SOURCED so the calibration panel can be "
            "balanced before anyone judges it. The judges never see it -- blind_prompt_v4_4 "
            "takes question, aliases and candidate and has no fourth parameter -- and where "
            "both judges disagree with the intent, the judges are the authority."
        ),
        "rows": rows,
    }
    payload["supplement_sha256"] = sha256_text(
        dumps_canonical({k: v for k, v in payload.items() if k != "built_at"})
    )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / SUPPLEMENT_FILENAME, payload)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / SUPPLEMENT_FILENAME),
                "n_rows": len(rows),
                "by_intended_class": report["by_intended_class"],
                "by_subtype": {k: v["n_rows"] for k, v in report["by_subtype"].items()},
                "shortfalls": report["shortfalls"],
                "supplement_sha256": payload["supplement_sha256"],
            }
        )
    )


# =====================================================================================
# bundle
# =====================================================================================


def detector_v4_4_bundle(
    audit_dir: Path = typer.Option(DEFAULT_AUDIT_DIR, "--audit-dir"),
    store_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--store-dir"),
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    supplement: Path = typer.Option(
        None,
        "--supplement",
        help="the file written by graph-detector-v4-4-supplement. Defaults to out-dir's.",
    ),
    labels: Path = typer.Option(
        None,
        "--labels",
        help="an adjudicated JSONL. Optional: without it every pair carries label null.",
    ),
    dev_fraction: float = typer.Option(0.5, "--dev-fraction"),
) -> None:
    """Build the v4.4 bundle: the original 1,019 pairs plus the supplement, one split rule."""
    _records, natural, eval_rows = _load_inputs(audit_dir, store_dir)

    supplement_path = Path(supplement) if supplement else Path(out_dir) / SUPPLEMENT_FILENAME
    if not supplement_path.exists():
        raise typer.BadParameter(
            f"{supplement_path} is absent. Run `rdl graph-detector-v4-4-supplement` first: "
            "the bundle does not generate rows, so that the pool it contains is a file "
            "someone reviewed rather than a side effect of building the bundle."
        )
    supplement_payload = json.loads(supplement_path.read_text(encoding="utf-8"))
    supplement_rows = list(supplement_payload.get("rows", ()))

    # --- the split: v4.3's rule, unchanged --------------------------------------------
    groups_by_population: dict[str, list[str]] = defaultdict(list)
    for row in eval_rows.values():
        groups_by_population[str(row["population"])].append(str(row["subject_group"]))
    split_of_group = assign_splits(groups_by_population, dev_fraction=dev_fraction)

    label_of, label_provenance = _load_labels(Path(labels) if labels else None)

    pairs: list[dict] = []
    seen_pair: dict[str, str] = {}

    def add(row: Mapping, *, origin: str, extra: Mapping) -> None:
        question = str(row["conditioning_question"])
        candidate = str(row["candidate_text"])
        pair_key = sha256_text(question + PAIR_SEPARATOR + candidate)
        if pair_key in seen_pair:
            raise typer.BadParameter(
                f"rows {seen_pair[pair_key]} and {row['audit_id']} are the same "
                "(question, candidate) pair. A duplicated pair is counted twice in every "
                "rate and, if it straddles the split, is train/development contamination."
            )
        seen_pair[pair_key] = str(row["audit_id"])
        group = str(row.get("subject_group", ""))
        pairs.append(
            {
                # --- the model's input, and nothing else ------------------------------
                "conditioning_question": question,
                "subject_aliases": list(row.get("subject_aliases", ())),
                "candidate_text": candidate,
                # --- the target -------------------------------------------------------
                "label": label_of.get(str(row["audit_id"])),
                # --- bookkeeping. NEVER serialised into a tokenizer input --------------
                "audit_id": str(row["audit_id"]),
                "origin": origin,
                "conditioning_id": str(row.get("conditioning_id", "")),
                "subject_id": str(row.get("subject_id", "")),
                "subject_group": group,
                "split": split_of_group.get(group, "train"),
                "pair_sha256": pair_key,
                "question_type_hint": question_type_hint(question),
                **dict(extra),
            }
        )

    for row in natural:
        add(
            row,
            origin="v4_1_audit",
            extra={
                "source_subtype": "audit/" + str(row.get("stratum", "")),
                "intended_class": None,
                "source_row_sha256": row["source_row_sha256"],
            },
        )
    for row in supplement_rows:
        add(
            row,
            origin="v4_4_supplement",
            extra={
                "source_subtype": str(row["source_subtype"]),
                "intended_class": str(row["intended_class"]),
                "generator_version": str(row.get("generator_version", "")),
                "frame_id": str(row.get("frame_id", "")),
            },
        )

    # --- disjointness, checked rather than assumed ------------------------------------
    groups_per_split: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        groups_per_split[pair["split"]].add(pair["subject_group"])
    overlap = sorted(groups_per_split["train"] & groups_per_split["development"])
    if overlap:
        raise typer.BadParameter(
            f"subject groups {overlap[:5]} appear in BOTH train and development."
        )

    # A supplement row reuses a real candidate in two of its subtypes, so the same TEXT can
    # legitimately appear twice. What must never happen is that text appearing under two
    # different splits, which is contamination wearing a supplement's clothes.
    split_of_candidate: dict[str, str] = {}
    straddling: list[str] = []
    for pair in pairs:
        digest = sha256_text(pair["candidate_text"])
        previous = split_of_candidate.setdefault(digest, pair["split"])
        if previous != pair["split"]:
            straddling.append(pair["audit_id"])
    if straddling:
        raise typer.BadParameter(
            f"{len(straddling)} candidate texts appear on both sides of the split "
            f"(first: {straddling[0]}). A reused candidate must stay inside one subject "
            "group; check the cross_question donor rule."
        )

    population_of = {
        p["audit_id"]: ("supplement" if p["origin"] == "v4_4_supplement" else "protected")
        for p in pairs
    }
    for row in natural:
        population_of[row["audit_id"]] = row["population"]
    probe = shortcut_probe(pairs, population_of)

    supplement_probe = _supplement_shortcut_probe(pairs)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    composition = {
        "by_origin": dict(sorted(Counter(p["origin"] for p in pairs).items())),
        "by_source_subtype": dict(sorted(Counter(p["source_subtype"] for p in pairs).items())),
        "by_intended_class": dict(sorted(Counter(str(p["intended_class"]) for p in pairs).items())),
        "by_question_type_hint": dict(
            sorted(Counter(p["question_type_hint"] for p in pairs).items())
        ),
    }
    split_composition = {
        split: {
            "n_pairs": sum(1 for p in pairs if p["split"] == split),
            "n_subjects": len({p["subject_id"] for p in pairs if p["split"] == split}),
            "by_origin": dict(
                sorted(Counter(p["origin"] for p in pairs if p["split"] == split).items())
            ),
        }
        for split in ("train", "development")
    }
    payload = {
        "schema": BUNDLE_SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_pairs": len(pairs),
        "n_from_audit": sum(1 for p in pairs if p["origin"] == "v4_1_audit"),
        "n_from_supplement": sum(1 for p in pairs if p["origin"] == "v4_4_supplement"),
        "tokenized_fields": list(TOKENIZED_FIELDS),
        "fields_never_tokenized": [
            "population",
            "is_protected",
            "forget_id",
            "concept_id",
            "subject_id",
            "split",
            "stratum",
            "origin",
            "source_subtype",
            "intended_class",
            "question_type_hint",
            "reference_answer",
            "judge identity",
            "detector score",
        ],
        "sources": {
            "audit_dir": str(audit_dir),
            "store_dir": str(store_dir),
            "supplement": str(supplement_path),
            "supplement_sha256": supplement_payload.get("supplement_sha256"),
            "v4_3_artifacts_unchanged": True,
        },
        "labels": label_provenance,
        "split_rule": {
            "unit": "subject group (TOFU author block)",
            "assignment": "v4.3's rule, unchanged: content-addressed hash of the group key",
            "why_unchanged": (
                "a v4.4 that also reshuffled the authors would confound 'the bundle "
                "changed' with 'the split changed' in every comparison against v4.3."
            ),
            "dev_fraction_requested": dev_fraction,
            "group_disjoint_verified": True,
            "no_candidate_straddles_the_split": True,
            "composition": split_composition,
        },
        "composition": composition,
        "supplement_diversity": diversity_report(
            [p for p in pairs if p["origin"] == "v4_4_supplement"]
        ),
        "shortcut_probe_population": probe,
        "shortcut_probe_supplement": supplement_probe,
        "pairs": pairs,
    }
    payload["bundle_sha256"] = sha256_text(
        dumps_canonical({k: v for k, v in payload.items() if k != "built_at"})
    )
    atomic_json(out / BUNDLE_FILENAME, payload)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / BUNDLE_FILENAME),
                "n_pairs": len(pairs),
                "composition": composition["by_origin"],
                "by_intended_class": composition["by_intended_class"],
                "split_composition": split_composition,
                "shortcut_alias_channel_excess": probe["alias_channel_excess"][
                    "excess_over_question_floor"
                ],
                "supplement_separability": supplement_probe["worst_case_balanced_accuracy"],
                "bundle_sha256": payload["bundle_sha256"],
            }
        )
    )


def _supplement_shortcut_probe(pairs: Sequence[Mapping]) -> dict:
    """Can *supplement* be told from *audit* by the model's own inputs alone?

    The probe the population probe does not cover, and the one a generated pool most needs.
    If crude surface features separate generated non-attempts from real candidates, then
    "was this row generated" is available to the model, and since almost every generated row
    is intended NONE, the detector can score well by learning the generator instead of
    answerability -- and the fresh engineering bank, which contains no generated rows at
    all, would then collapse for reasons the development set could never show.

    Reported rather than gated, with the same reasoning the v4.3 probe carries: the number
    has an irreducible floor (a tool-status line really is shorter than a paragraph of
    prose), so the acceptable value is a judgement fixed in the protocol, not a bound that
    can be mechanically enforced here.
    """
    features: dict[str, list[float]] = defaultdict(list)
    is_supplement: list[bool] = []
    for pair in pairs:
        candidate = str(pair["candidate_text"])
        features["candidate_chars"].append(float(len(candidate)))
        features["candidate_words"].append(float(len(candidate.split())))
        features["n_sentences"].append(float(sum(candidate.count(c) for c in ".!?")))
        features["comma_rate"].append(candidate.count(",") / max(1, len(candidate.split())))
        features["uppercase_rate"].append(
            sum(1 for c in candidate if c.isupper()) / max(1, len(candidate))
        )
        is_supplement.append(pair["origin"] == "v4_4_supplement")

    from .detector_v4_3_bundle import _auc, _best_stump

    per_feature = {
        name: {
            "auc_vs_supplement": _auc(
                [v for v, s in zip(values, is_supplement, strict=True) if s],
                [v for v, s in zip(values, is_supplement, strict=True) if not s],
            ),
            "best_stump_balanced_accuracy": _best_stump(values, is_supplement),
        }
        for name, values in sorted(features.items())
    }
    worst = max(per_feature, key=lambda k: per_feature[k]["best_stump_balanced_accuracy"] or 0.0)

    # The decomposition, without which the headline number is unreadable. Some subtypes are
    # short because they were composed lazily, which is fixable; others are short because
    # the thing they depict is short in the world -- a tool-status line, a mid-clause
    # fragment -- and lengthening those would trade a measurable artifact for an
    # unmeasurable one, namely non-attempt text that no agent would ever emit.
    audit_lengths = [
        float(len(str(p["candidate_text"]))) for p in pairs if p["origin"] != "v4_4_supplement"
    ]
    by_subtype: dict[str, dict] = {}
    for subtype in sorted(
        {str(p["source_subtype"]) for p in pairs if p["origin"] == "v4_4_supplement"}
    ):
        lengths = [
            float(len(str(p["candidate_text"]))) for p in pairs if p["source_subtype"] == subtype
        ]
        values = [*lengths, *audit_lengths]
        flags = [True] * len(lengths) + [False] * len(audit_lengths)
        by_subtype[subtype] = {
            "n_rows": len(lengths),
            "median_chars": sorted(lengths)[len(lengths) // 2] if lengths else None,
            "length_balanced_accuracy_vs_audit": _best_stump(values, flags),
        }

    return {
        "n_supplement": sum(is_supplement),
        "n_audit": len(is_supplement) - sum(is_supplement),
        "audit_median_chars": (
            sorted(audit_lengths)[len(audit_lengths) // 2] if audit_lengths else None
        ),
        "per_feature": per_feature,
        "by_subtype": by_subtype,
        "worst_case_feature": worst,
        "worst_case_balanced_accuracy": per_feature[worst]["best_stump_balanced_accuracy"],
        "interpretation": (
            "balanced accuracy of the single best surface feature at telling a generated "
            "supplement row from a real audited one. A high value means 'was this "
            "generated' is available to the model, and since the supplement is where "
            "almost every NONE row lives, that is a shortcut to the NONE class that the "
            "fresh engineering bank -- which contains no generated rows -- will not "
            "reproduce. Reported, not gated; the acceptable value is fixed in "
            "DETECTOR_V4_4_ANSWER_ATTEMPT_PROTOCOL.md."
        ),
    }


def _load_labels(path: Path | None) -> tuple[dict[str, str], dict]:
    if path is None:
        return {}, {
            "source": None,
            "n_labels": 0,
            "why": (
                "no adjudicated label file was passed. The v4.4 labels come from the local "
                "judges on the GPU box; this bundle freezes the INPUTS, which is what the "
                "CPU phase is for."
            ),
        }
    if not path.exists():
        raise typer.BadParameter(f"{path} is absent")
    labels: dict[str, str] = {}
    for row in _read_jsonl(path):
        label = row.get("answer_attempt")
        if label in ("NONE", "PARTIAL", "ANSWER"):
            labels[str(row["audit_id"])] = str(label)
    return labels, {
        "source": str(path),
        "sha256": sha256_text(path.read_text(encoding="utf-8")),
        "n_labels": len(labels),
        "distribution": dict(sorted(Counter(labels.values()).items())),
    }


# =====================================================================================
# the calibration panel
# =====================================================================================


def balanced_take(
    rows: Sequence[Mapping],
    *,
    n: int,
    strata: Sequence[str],
    key: str,
    salt: str,
) -> list[dict]:
    """``n`` rows spread as evenly as the pool allows over ``strata``, deterministically.

    Largest-remainder allocation over the strata that actually have rows, then a
    content-addressed rank inside each -- the same discipline the v4.3 human sampler uses.
    Content-addressed rather than seeded so that adding a stratum later does not reshuffle
    the rows already drawn, which would silently move a panel that was supposed to be frozen.
    """
    pools = {
        s: sorted(
            [dict(r) for r in rows if str(r.get(key, "")) == s], key=lambda r: str(r["audit_id"])
        )
        for s in strata
    }
    available = {s: len(p) for s, p in pools.items() if p}
    if not available:
        return []
    total_available = sum(available.values())
    if total_available < n:
        raise ValueError(
            f"asked for {n} rows across {sorted(available)} but only {total_available} "
            "exist. The panel is not shrunk to fit: generate more source rows instead."
        )
    share = {s: n * available[s] / total_available for s in available}
    take = {s: min(int(share[s]), available[s]) for s in available}
    # Largest remainder, then any leftover to whichever pools still have rows.
    for stratum in sorted(available, key=lambda s: (-(share[s] - int(share[s])), s)):
        if sum(take.values()) >= n:
            break
        if take[stratum] < available[stratum]:
            take[stratum] += 1
    while sum(take.values()) < n:
        grew = False
        for stratum in sorted(available):
            if sum(take.values()) >= n:
                break
            if take[stratum] < available[stratum]:
                take[stratum] += 1
                grew = True
        if not grew:  # pragma: no cover - guarded by the total_available check above
            break

    out: list[dict] = []
    for stratum in sorted(take):
        ranked = sorted(
            pools[stratum], key=lambda r: (_unit_hash(f"{salt}|{r['audit_id']}"), r["audit_id"])
        )
        out.extend(ranked[: take[stratum]])
    return sorted(out, key=lambda r: str(r["audit_id"]))


def detector_v4_4_panel(
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    bundle: Path = typer.Option(None, "--bundle"),
    store_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--store-dir"),
    audit_dir: Path = typer.Option(DEFAULT_AUDIT_DIR, "--audit-dir"),
    per_class: int = typer.Option(PANEL_CLASS_SIZE, "--per-class"),
) -> None:
    """Freeze the balanced calibration panel: ``per_class`` rows per intended class.

    Run BEFORE any v4.4 judge. The panel is a subset of the bundle -- it costs no extra
    judging -- but which rows are in it is fixed here and hashed, so the kappa gate cannot
    later be computed on whichever subset happens to agree.
    """
    bundle_path = Path(bundle) if bundle else Path(out_dir) / BUNDLE_FILENAME
    if not bundle_path.exists():
        raise typer.BadParameter(
            f"{bundle_path} is absent. Run `rdl graph-detector-v4-4-bundle` first."
        )
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    pairs = list(payload.get("pairs", ()))

    _, natural, eval_rows = _load_inputs(audit_dir, store_dir)
    stratum_of = {row["audit_id"]: row["stratum"] for row in natural}

    none_pool = [p for p in pairs if p.get("intended_class") == "NONE"]
    partial_pool = [p for p in pairs if p.get("intended_class") == "PARTIAL"]
    answer_pool = [
        dict(p, _answer_stratum=stratum_of.get(p["audit_id"], ""))
        for p in pairs
        if p["origin"] == "v4_1_audit" and stratum_of.get(p["audit_id"]) in ANSWER_INTENT_STRATA
    ]

    selected = {
        "NONE": balanced_take(
            none_pool,
            n=per_class,
            strata=NONE_SUBTYPES,
            key="source_subtype",
            salt="v4.4-panel-none",
        ),
        "PARTIAL": balanced_take(
            partial_pool,
            n=per_class,
            strata=PARTIAL_SUBTYPES,
            key="source_subtype",
            salt="v4.4-panel-partial",
        ),
        "ANSWER": balanced_take(
            answer_pool,
            n=per_class,
            strata=ANSWER_INTENT_STRATA,
            key="_answer_stratum",
            salt="v4.4-panel-answer",
        ),
    }

    rows: list[dict] = []
    seen: set[str] = set()
    for intended, chosen in sorted(selected.items()):
        if len(chosen) != per_class:
            raise typer.BadParameter(
                f"intended {intended} yielded {len(chosen)} rows, not {per_class}. The "
                "panel is balanced or it is not a panel; regenerate the supplement with a "
                "larger quota rather than accepting an unbalanced one."
            )
        for row in chosen:
            if row["audit_id"] in seen:
                raise typer.BadParameter(
                    f"{row['audit_id']} was drawn into two intended classes. A row with two "
                    "intents makes every per-class number ambiguous."
                )
            seen.add(row["audit_id"])
            rows.append(
                {
                    "audit_id": row["audit_id"],
                    "intended_class": intended,
                    "source_subtype": row.get("source_subtype"),
                    "question_type_hint": row.get("question_type_hint"),
                    "subject_group": row.get("subject_group"),
                    "split": row.get("split"),
                    "pair_sha256": row["pair_sha256"],
                }
            )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": PANEL_SCHEMA,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": len(rows),
        "per_class": per_class,
        "bundle": str(bundle_path),
        "bundle_sha256": payload.get("bundle_sha256"),
        "answer_intent_strata": list(ANSWER_INTENT_STRATA),
        "composition": {
            "by_intended_class": dict(sorted(Counter(r["intended_class"] for r in rows).items())),
            "by_source_subtype": dict(
                sorted(Counter(str(r["source_subtype"]) for r in rows).items())
            ),
            "by_question_type_hint": dict(
                sorted(Counter(str(r["question_type_hint"]) for r in rows).items())
            ),
            "by_split": dict(sorted(Counter(str(r["split"]) for r in rows).items())),
            "n_subject_groups": len({str(r["subject_group"]) for r in rows}),
        },
        "hidden_from_judges": (
            "intended_class is not a field of any judge prompt. blind_prompt_v4_4 takes "
            "conditioning_question, subject_aliases and candidate_text and has no fourth "
            "parameter, so the panel's construction cannot reach an annotator even by "
            "mistake -- the same mechanism that hides population."
        ),
        "why_balanced": (
            "v4.3's kappa of 0.404 sat under 74.7% raw agreement because 65-81% of rows "
            "were one class. The primary kappa gate is computed here, on 200/200/200, and "
            "the full-mixture kappa is reported beside it as a distribution-dependent "
            "diagnostic. The BOUND is unchanged at 0.70: what moved is the distribution "
            "it is evaluated on, and that move is pre-registered before any label exists."
        ),
        "rows": rows,
    }
    manifest["panel_sha256"] = sha256_text(
        dumps_canonical({k: v for k, v in manifest.items() if k != "built_at"})
    )
    atomic_json(out / PANEL_FILENAME, manifest)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / PANEL_FILENAME),
                "n_rows": len(rows),
                "composition": manifest["composition"],
                "panel_sha256": manifest["panel_sha256"],
            }
        )
    )
    del eval_rows


# =====================================================================================
# the 60-row smoke fixture
# =====================================================================================


def detector_v4_4_judge_smoke_fixture(
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    store_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--store-dir"),
    audit_dir: Path = typer.Option(DEFAULT_AUDIT_DIR, "--audit-dir"),
    bundle: Path = typer.Option(None, "--bundle"),
    synthetic: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/detector_v4/DETECTOR_V4_DATASET.json"),
        "--synthetic",
        help="the synthetic relation dataset. Its held-out split supplies the ANSWER and "
        "PARTIAL-intent rows.",
    ),
    n_per_class: int = typer.Option(20, "--n-per-class"),
) -> None:
    """60 rows disjoint from the bundle, 20 per intended class, for the GPU-1 rubric check.

    Disjointness is by ``(question, candidate)`` digest against the bundle's own pair
    hashes, which is the identity the bundle was frozen on. A smoke that overlapped the
    reportable rows would let a rubric be tuned on rows it is later scored on.

    Where each class comes from
    ---------------------------
    ``NONE`` is generated in the ``smoke`` namespace. The namespace is not decoration: the
    generator is deterministic, so asking it for twenty rows in the bundle's namespace
    returns the bundle's own first twenty. A different namespace is a different draw from
    the same banks, and both stay reproducible.

    ``ANSWER`` and ``PARTIAL`` come from the synthetic relation dataset's **held-out**
    split, which has invented subjects and is disjoint from the natural audit by
    construction. ANSWER-intent rows additionally require ``requires_accumulation`` false:
    an accumulation row's answer only emerges across several messages, so its first message
    alone is a fragment, and putting it in the ANSWER pool would seed the smoke with rows
    whose intent is wrong on exactly the boundary the smoke exists to inspect.

    Non-reportable by construction. It is a rubric and format check, not a sample of any
    distribution, and its ``_intended_class`` field is for the human reading the 60 rows --
    no judge prompt has anywhere to put it.
    """
    bundle_path = Path(bundle) if bundle else Path(out_dir) / BUNDLE_FILENAME
    if not bundle_path.exists():
        raise typer.BadParameter(f"{bundle_path} is absent. Build the bundle first.")
    used_pairs = {
        str(p["pair_sha256"]) for p in json.loads(bundle_path.read_text(encoding="utf-8"))["pairs"]
    }

    def pair_digest(question: str, candidate: str) -> str:
        return sha256_text(question + PAIR_SEPARATOR + candidate)

    records, natural, _ = _load_inputs(audit_dir, store_dir)
    generated, _ = generate_supplement(
        records,
        natural_rows=natural,
        n_per_none_subtype=n_per_class,
        n_per_partial_subtype=n_per_class,
        namespace="smoke",
        enforce_diversity=False,
    )
    pools: dict[str, list[dict]] = {
        "NONE": [
            {
                "audit_id": r["audit_id"],
                "conditioning_question": r["conditioning_question"],
                "subject_aliases": list(r["subject_aliases"]),
                "candidate_text": r["candidate_text"],
                "source": r["source_subtype"],
            }
            for r in generated
            if r["intended_class"] == "NONE" and r["pair_sha256"] not in used_pairs
        ],
        "ANSWER": [],
        "PARTIAL": [],
    }

    if not Path(synthetic).exists():
        raise typer.BadParameter(
            f"{synthetic} is absent. The ANSWER and PARTIAL-intent smoke rows come from "
            "its held-out split; there is no natural source for them that is disjoint "
            "from the 1,019."
        )
    dataset = json.loads(Path(synthetic).read_text(encoding="utf-8"))
    for row in dataset.get("rows", ()):
        if row.get("split") != "heldout":
            continue
        label = str(row.get("label", ""))
        if label == "ANSWER" and row.get("requires_accumulation"):
            continue
        if label not in ("ANSWER", "PARTIAL"):
            continue
        candidates = list(row.get("candidates", ()))
        if not candidates:
            continue
        question = str(row["protected_question"])
        candidate = str(candidates[0])
        if pair_digest(question, candidate) in used_pairs:
            continue
        pools[label].append(
            {
                "audit_id": f"syn-{row['row_id']}",
                "conditioning_question": question,
                "subject_aliases": list(row.get("aliases", ())),
                "candidate_text": candidate,
                "source": f"synthetic/heldout/{row.get('relation', '')}",
            }
        )

    rows: list[dict] = []
    for intended in ("ANSWER", "NONE", "PARTIAL"):
        ranked = sorted(
            pools[intended], key=lambda r: _unit_hash(f"v4.4-smoke|{intended}|{r['audit_id']}")
        )
        if len(ranked) < n_per_class:
            raise typer.BadParameter(
                f"only {len(ranked)} disjoint rows available for intended {intended}, "
                f"need {n_per_class}. A smoke short of a class cannot check the rubric on it."
            )
        for row in ranked[:n_per_class]:
            rows.append(
                {
                    "audit_id": f"v44smoke-{intended.lower()}-{str(row['audit_id'])[-12:]}",
                    "conditioning_question": row["conditioning_question"],
                    "subject_aliases": row["subject_aliases"],
                    "candidate_text": row["candidate_text"],
                    # Read by the human inspecting the 60 rows, never by a judge.
                    "_intended_class": intended,
                    "_source": row["source"],
                }
            )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    target = out / SMOKE_FILENAME
    target.write_text(
        "".join(dumps_canonical(row) + "\n" for row in sorted(rows, key=lambda r: r["audit_id"])),
        encoding="utf-8",
    )
    typer.echo(
        dumps_canonical(
            {
                "schema": SMOKE_SCHEMA,
                "wrote": str(target),
                "n_rows": len(rows),
                "by_intended_class": dict(
                    sorted(Counter(r["_intended_class"] for r in rows).items())
                ),
                "by_question_type_hint": dict(
                    sorted(
                        Counter(
                            question_type_hint(r["conditioning_question"]) for r in rows
                        ).items()
                    )
                ),
                "disjoint_from_bundle": True,
                "reportable": False,
            }
        )
    )
