"""``rdl graph-detector-v4-3-human-*`` -- the 250-row human validation, start to finish.

Four commands, in the order they may be run: ``human-sample``, ``human-import``,
``human-adjudicate``, ``human-report``.

**The draw never reads a detector score.** A sample chosen where the detector is confident
measures the detector's confidence, not its accuracy, and the resulting agreement number
would be an artifact of the selection rule. :func:`draw_sample` has no parameter through
which a score could arrive.

Exact stratified sampling, not weighted top-k
---------------------------------------------
The first version drew by ``hash / weight`` and recorded an approximate inclusion
probability. That is a defensible *enrichment* but it is not a probability sample: the
realised inclusion probability of a given row depends on the whole competing set, so
"design-weighted estimate" would have been the wrong words for it.

This version preregisters **mutually exclusive strata** and an exact allocation, then draws
``n_h`` of ``N_h`` from each by content-addressed order. The inclusion probability is then
exactly ``n_h / N_h``, every row carries it, and a design-weighted estimate means what it
says. The strata are label properties -- judge disagreement, the rare PARTIAL class, and
everything else -- never detector properties.

Where ``judges_disagree`` comes from
------------------------------------
It is not in the pair bundle, and the first version read it off the bundle rows, where it
was always absent -- so the oversampling silently never fired on real data and every row
landed in one stratum. It is joined here from the label report's disagreement list, which
is the artifact that actually knows.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_store import DEFAULT_OUT_DIR

__all__ = [
    "HUMAN_SAMPLE_SCHEMA",
    "RATER_FIELDS",
    "SAMPLING_STRATA",
    "STRATUM_ALLOCATION",
    "allocate",
    "cohens_kappa",
    "detector_v4_3_human_adjudicate",
    "detector_v4_3_human_import",
    "detector_v4_3_human_report",
    "detector_v4_3_human_sample",
    "draw_sample",
]

HUMAN_SAMPLE_SCHEMA = "graph-detector-v4-3-human-sample-v1"
SAMPLE_FILENAME = "V4_3_HUMAN_SAMPLE.json"
RATER_FILENAME = "V4_3_HUMAN_RATER_{rater}.jsonl"
AGREEMENT_FILENAME = "V4_3_HUMAN_AGREEMENT.json"
ADJUDICATED_FILENAME = "V4_3_HUMAN_ADJUDICATED.jsonl"
REPORT_FILENAME = "V4_3_HUMAN_REPORT.json"

LABELS = ("NONE", "PARTIAL", "ANSWER")

RATER_FIELDS = frozenset({"audit_id", "answer_attempt", "notes"})
EXPORTED_FIELDS = frozenset(
    {"audit_id", "conditioning_question", "subject_aliases", "candidate_text"}
)

# Mutually exclusive and exhaustive, in priority order: a row belongs to the FIRST stratum
# it qualifies for. Both are label properties; neither reads a detector score.
SAMPLING_STRATA = ("judge_disagreement", "partial_label", "remainder")

# Preregistered allocation per 125-row source. Enrichment on the two hard strata, with the
# remainder taking the balance. A stratum with fewer rows than its allocation contributes
# all of them and the shortfall spills to `remainder` -- recorded, so the realised
# allocation is always visible next to the intended one.
STRATUM_ALLOCATION = {"judge_disagreement": 40, "partial_label": 25, "remainder": 60}

# Human-validation gates, §9.1 of the protocol. Frozen before any human label exists.
HUMAN_GATES: dict[str, tuple[str, float]] = {
    "human_human_kappa": (">=", 0.70),
    "consensus_vs_human_macro_f1": (">=", 0.80),
    "consensus_recall_on_human_answer": (">=", 0.85),
    "recall_NONE": (">=", 0.75),
    "recall_PARTIAL": (">=", 0.75),
    "recall_ANSWER": (">=", 0.75),
    "n_unresolved": ("==", 0.0),
    "n_provenance_failures": ("==", 0.0),
}


def _unit_hash(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / float(1 << 64)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _stratum_of(row: Mapping) -> str:
    if row.get("judges_disagree"):
        return "judge_disagreement"
    if row.get("label") == "PARTIAL":
        return "partial_label"
    return "remainder"


def allocate(available: Mapping[str, int], *, total: int) -> dict[str, int]:
    """Realised per-stratum draw sizes. Shortfalls spill to ``remainder``, deterministically."""
    out: dict[str, int] = {}
    for stratum in SAMPLING_STRATA:
        out[stratum] = min(int(available.get(stratum, 0)), STRATUM_ALLOCATION[stratum])
    shortfall = total - sum(out.values())
    if shortfall > 0:
        spare = int(available.get("remainder", 0)) - out["remainder"]
        out["remainder"] += min(shortfall, max(0, spare))
    # Still short (a small source): take whatever the other strata still hold, in order.
    shortfall = total - sum(out.values())
    for stratum in SAMPLING_STRATA:
        if shortfall <= 0:
            break
        spare = int(available.get(stratum, 0)) - out[stratum]
        take = min(shortfall, max(0, spare))
        out[stratum] += take
        shortfall -= take
    return out


def draw_sample(
    rows: Sequence[Mapping],
    *,
    n: int,
    stratum: str,
    salt: str = "v4.3-human",
) -> list[dict]:
    """Exactly ``n`` rows from ``rows``, stratified, with exact inclusion probabilities.

    ``stratum`` names the SOURCE (the original 1,019 or the fresh audit); the sampling
    strata inside it are :data:`SAMPLING_STRATA`. Selection within a stratum is by
    content-addressed order, so the draw is reproducible and adding rows does not reshuffle
    the ones already chosen.

    ``rows`` may carry ``judges_disagree`` and ``label``; neither is shown to a rater. They
    are read only to assign a stratum, which is why this function takes label metadata and
    takes no scores at all.
    """
    pools: dict[str, list[Mapping]] = defaultdict(list)
    for row in rows:
        pools[_stratum_of(row)].append(row)
    for pool in pools.values():
        pool.sort(key=lambda r: (_unit_hash(f"{salt}|{stratum}|{r['audit_id']}"), r["audit_id"]))

    sizes = allocate({k: len(v) for k, v in pools.items()}, total=n)
    drawn: list[dict] = []
    for name in SAMPLING_STRATA:
        pool = pools.get(name, [])
        take = sizes.get(name, 0)
        for row in pool[:take]:
            drawn.append(
                {
                    **row,
                    "source_stratum": stratum,
                    "sampling_stratum": name,
                    "stratum_size": len(pool),
                    "stratum_drawn": take,
                    # Exact, because the draw is exactly `take` of `len(pool)`.
                    "inclusion_probability": (take / len(pool)) if pool else 0.0,
                    "design_weight": (len(pool) / take) if take else 0.0,
                }
            )
    return drawn


def _blinded(row: Mapping) -> dict:
    out = {
        "audit_id": str(row["audit_id"]),
        "conditioning_question": str(row["conditioning_question"]),
        "subject_aliases": list(row.get("subject_aliases", ())),
        "candidate_text": str(row["candidate_text"]),
    }
    leaked = sorted(set(out) - EXPORTED_FIELDS)
    if leaked:
        raise typer.BadParameter(f"a blinded rater row may not carry {leaked}")
    return out


def detector_v4_3_human_sample(
    bundle: Path = typer.Option(DEFAULT_OUT_DIR / "DETECTOR_V4_3_PAIR_BUNDLE.json", "--bundle"),
    fresh_audit: Path = typer.Option(None, "--fresh-audit"),
    label_report: Path = typer.Option(
        DEFAULT_OUT_DIR / "DETECTOR_V4_3_LABEL_REPORT.json",
        "--label-report",
        help="supplies judges_disagree, which the pair bundle does not carry.",
    ),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir"),
    n_per_stratum: int = typer.Option(125, "--n-per-stratum"),
    raters: str = typer.Option("A,B", "--raters"),
    exploratory: bool = typer.Option(
        False,
        "--exploratory",
        help="allow a draw from the original bundle alone. Marks the sample "
        "non-reportable: the protocol's 250 rows are 125 original PLUS 125 fresh.",
    ),
) -> None:
    """Draw the frozen human sample and write one blinded file per rater."""
    if not Path(bundle).exists():
        raise typer.BadParameter(f"{bundle} is absent")
    payload = json.loads(Path(bundle).read_text(encoding="utf-8"))

    if fresh_audit is None and not exploratory:
        raise typer.BadParameter(
            "a reportable human sample is 125 rows from the original 1,019 AND 125 from "
            "the fresh engineering audit. Drawing from the bundle alone measures the "
            "detector where it was developed. Pass --fresh-audit, or --exploratory to "
            "record explicitly that this draw cannot support the protocol's gates."
        )

    # judges_disagree, joined from the artifact that knows. Absent before the judges run,
    # in which case the disagreement stratum is simply empty and the manifest says so.
    disagreeing: set[str] = set()
    if Path(label_report).exists():
        report = json.loads(Path(label_report).read_text(encoding="utf-8"))
        disagreeing = {str(i) for i in (report.get("blind", {}).get("disagreement_ids") or [])}
        if not disagreeing:
            path = Path(label_report).parent / "V4_3_BLIND_DISAGREEMENTS.jsonl"
            disagreeing = {str(r["audit_id"]) for r in _read_jsonl(path)}

    def annotate(rows: Sequence[Mapping]) -> list[dict]:
        return [{**row, "judges_disagree": str(row["audit_id"]) in disagreeing} for row in rows]

    sources: dict[str, list[dict]] = {"original_1019": annotate(payload.get("pairs", ()))}
    if fresh_audit is not None:
        if not Path(fresh_audit).exists():
            raise typer.BadParameter(f"{fresh_audit} is absent")
        fresh = json.loads(Path(fresh_audit).read_text(encoding="utf-8"))
        sources["fresh_engineering"] = annotate(fresh.get("pairs", fresh.get("rows", ())))

    sample: list[dict] = []
    for name, rows in sorted(sources.items()):
        sample.extend(draw_sample(rows, n=n_per_stratum, stratum=name))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rater_ids = [r.strip() for r in raters.split(",") if r.strip()]
    for rater in rater_ids:
        (out / RATER_FILENAME.format(rater=rater)).write_text(
            "".join(dumps_canonical(_blinded(row)) + "\n" for row in sample), encoding="utf-8"
        )

    manifest = {
        "schema": HUMAN_SAMPLE_SCHEMA,
        "drawn_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": len(sample),
        "reportable": fresh_audit is not None and len(rater_ids) >= 2,
        "sources": {
            name: sum(1 for r in sample if r["source_stratum"] == name) for name in sources
        },
        "raters": rater_ids,
        "bundle_sha256": payload.get("bundle_sha256"),
        "n_judge_disagreements_known": len(disagreeing),
        "sampling": {
            "design": "stratified without replacement; exactly n_h of N_h per stratum",
            "strata": list(SAMPLING_STRATA),
            "intended_allocation": STRATUM_ALLOCATION,
            "realised_allocation": {
                name: dict(
                    sorted(
                        Counter(
                            r["sampling_stratum"] for r in sample if r["source_stratum"] == name
                        ).items()
                    )
                )
                for name in sources
            },
            "uses_detector_scores": False,
            "why_not": (
                "a sample drawn where the detector is confident measures the detector's "
                "confidence rather than its accuracy. Judge disagreement and the rare "
                "PARTIAL class are LABEL properties and are enriched; every row carries "
                "the exact n_h/N_h inclusion probability and its design weight, so overall "
                "estimates can be weighted back to the population."
            ),
        },
        "blinding": {
            "rater_sees": sorted(EXPORTED_FIELDS),
            "withheld": [
                "model labels",
                "model identity",
                "detector scores",
                "population",
                "source filename",
                "split",
                "stratum",
                "reference answer",
            ],
        },
        "rows": {
            row["audit_id"]: {
                "source_stratum": row["source_stratum"],
                "sampling_stratum": row["sampling_stratum"],
                "inclusion_probability": row["inclusion_probability"],
                "design_weight": row["design_weight"],
            }
            for row in sample
        },
    }
    atomic_json(out / SAMPLE_FILENAME, manifest)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / SAMPLE_FILENAME),
                "rater_files": [str(out / RATER_FILENAME.format(rater=r)) for r in rater_ids],
                "n_rows": len(sample),
                "reportable": manifest["reportable"],
                "realised_allocation": manifest["sampling"]["realised_allocation"],
                "uses_detector_scores": False,
            }
        )
    )


def cohens_kappa(
    a: Sequence[str], b: Sequence[str], *, labels: Sequence[str] = LABELS
) -> float | None:
    """Unweighted Cohen's kappa. ``None`` when there is nothing to agree about."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n
    count_a = Counter(a)
    count_b = Counter(b)
    expected = sum((count_a[label] / n) * (count_b[label] / n) for label in labels)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def _read_rater(path: Path) -> dict[str, str]:
    if not path.exists():
        raise typer.BadParameter(f"{path} is absent")
    out: dict[str, str] = {}
    for row in _read_jsonl(path):
        unknown = sorted(set(row) - RATER_FIELDS - EXPORTED_FIELDS)
        if unknown:
            raise typer.BadParameter(
                f"{path} carries {unknown}, which a blinded rater was never shown. A rater "
                "file with model labels or scores in it cannot support an independent "
                "agreement claim."
            )
        label = row.get("answer_attempt")
        if label not in LABELS:
            raise typer.BadParameter(
                f"{path}: row {row.get('audit_id')} has answer_attempt={label!r}, which is "
                f"not one of {list(LABELS)}. An unlabelled row is not a NONE."
            )
        out[str(row["audit_id"])] = str(label)
    return out


def detector_v4_3_human_import(
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir"),
    raters: str = typer.Option("A,B", "--raters"),
) -> None:
    """Read the completed rater files and report agreement overall and by stratum."""
    out = Path(out_dir)
    manifest_path = out / SAMPLE_FILENAME
    if not manifest_path.exists():
        raise typer.BadParameter(f"{manifest_path} is absent; draw the sample first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["rows"]

    rater_ids = [r.strip() for r in raters.split(",") if r.strip()]
    labels = {r: _read_rater(out / RATER_FILENAME.format(rater=r)) for r in rater_ids}
    for rater, table in labels.items():
        missing = sorted(set(rows) - set(table))
        extra = sorted(set(table) - set(rows))
        if missing or extra:
            raise typer.BadParameter(
                f"rater {rater} answered a different sample: {len(missing)} missing, "
                f"{len(extra)} not in the draw."
            )

    if len(rater_ids) < 2:
        report = {
            "schema": "graph-detector-v4-3-human-agreement-v1",
            "n_raters": len(rater_ids),
            "kappa": None,
            "exploratory_only": True,
            "why": (
                "one rater cannot support a human-human agreement claim. This is an "
                "exploratory human audit and must be reported as one."
            ),
        }
        atomic_json(out / AGREEMENT_FILENAME, report)
        typer.echo(dumps_canonical(report))
        return

    first, second = rater_ids[0], rater_ids[1]
    ordered = sorted(rows)
    a = [labels[first][k] for k in ordered]
    b = [labels[second][k] for k in ordered]
    by_stratum: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key in ordered:
        by_stratum[rows[key]["source_stratum"]].append((labels[first][key], labels[second][key]))

    disagreements = sorted(k for k in ordered if labels[first][k] != labels[second][k])
    report = {
        "schema": "graph-detector-v4-3-human-agreement-v1",
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raters": rater_ids,
        "n_rows": len(ordered),
        "kappa": cohens_kappa(a, b),
        "raw_agreement": sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a),
        "kappa_by_source": {
            source: cohens_kappa([x for x, _ in pairs], [y for _, y in pairs])
            for source, pairs in sorted(by_stratum.items())
        },
        "label_distribution": {
            rater: dict(sorted(Counter(labels[rater].values()).items())) for rater in rater_ids
        },
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
        "exploratory_only": False,
    }
    atomic_json(out / AGREEMENT_FILENAME, report)
    typer.echo(dumps_canonical({k: v for k, v in report.items() if k != "disagreements"}))


def detector_v4_3_human_adjudicate(
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir"),
    decisions: Path = typer.Option(None, "--decisions"),
    raters: str = typer.Option("A,B", "--raters"),
) -> None:
    """Freeze the adjudicated human labels. Refuses to run before both files exist."""
    out = Path(out_dir)
    agreement_path = out / AGREEMENT_FILENAME
    if not agreement_path.exists():
        raise typer.BadParameter(
            f"{agreement_path} is absent. Adjudication happens AFTER both rater files are "
            "frozen and their agreement is computed; adjudicating first would let one "
            "rater's labels be revised in the light of the other's."
        )
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    rater_ids = [r.strip() for r in raters.split(",") if r.strip()]
    labels = {r: _read_rater(out / RATER_FILENAME.format(rater=r)) for r in rater_ids}
    first, second = rater_ids[0], rater_ids[1]

    resolved: dict[str, str] = {}
    if decisions is not None:
        for row in _read_jsonl(Path(decisions)):
            if row.get("answer_attempt") not in LABELS:
                raise typer.BadParameter(f"bad adjudication label in {decisions}: {row}")
            resolved[str(row["audit_id"])] = str(row["answer_attempt"])

    outstanding = [k for k in agreement.get("disagreements", ()) if k not in resolved]
    final = []
    for audit_id in sorted(labels[first]):
        if labels[first][audit_id] == labels[second][audit_id]:
            final.append(
                {
                    "audit_id": audit_id,
                    "answer_attempt": labels[first][audit_id],
                    "source": "agreed",
                }
            )
        elif audit_id in resolved:
            final.append(
                {
                    "audit_id": audit_id,
                    "answer_attempt": resolved[audit_id],
                    "source": "adjudicated",
                }
            )

    path = out / ADJUDICATED_FILENAME
    path.write_text("".join(dumps_canonical(r) + "\n" for r in final), encoding="utf-8")
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(path),
                "n_agreed": sum(1 for r in final if r["source"] == "agreed"),
                "n_adjudicated": sum(1 for r in final if r["source"] == "adjudicated"),
                "n_unresolved": len(outstanding),
                "unresolved": outstanding[:10],
                "complete": not outstanding,
                "note": (
                    "the gate requires zero unresolved rows. An unresolved disagreement is "
                    "not a NONE."
                ),
            }
        )
    )


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score interval. Used because the normal approximation is wrong at the tails.

    These recalls are computed over a few dozen rows per class, where a Wald interval can
    run past 1.0 and reports a bound the estimate cannot take.
    """
    if not total:
        return None
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    margin = (z * ((p * (1 - p) / total + z**2 / (4 * total**2)) ** 0.5)) / denominator
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def detector_v4_3_human_report(
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir"),
    model_consensus: Path = typer.Option(
        None, "--model-consensus", help="JSONL of {audit_id, answer_attempt} from the judges."
    ),
    detector_predictions: Path = typer.Option(
        None,
        "--detector-predictions",
        help="JSONL of {audit_id, fired, answer_score} from the FROZEN checkpoint.",
    ),
) -> None:
    """The final human-validation report: agreement, model-vs-human, detector-vs-human, gates."""
    out = Path(out_dir)
    manifest = json.loads((out / SAMPLE_FILENAME).read_text(encoding="utf-8"))
    agreement_path = out / AGREEMENT_FILENAME
    adjudicated_path = out / ADJUDICATED_FILENAME
    for path in (agreement_path, adjudicated_path):
        if not path.exists():
            raise typer.BadParameter(f"{path} is absent; run human-import and human-adjudicate")
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    human = {str(r["audit_id"]): str(r["answer_attempt"]) for r in _read_jsonl(adjudicated_path)}

    provenance: list[str] = []
    if not manifest.get("reportable"):
        provenance.append(
            "the sample manifest is marked non-reportable (drawn without the fresh audit "
            "or with fewer than two raters)"
        )
    unresolved = sorted(set(manifest["rows"]) - set(human))
    if unresolved:
        provenance.append(f"{len(unresolved)} sampled row(s) have no adjudicated human label")

    consensus: dict[str, str] = {}
    if model_consensus is not None:
        consensus = {
            str(r["audit_id"]): str(r["answer_attempt"])
            for r in _read_jsonl(Path(model_consensus))
            if r.get("answer_attempt") in LABELS
        }

    shared = sorted(set(human) & set(consensus))
    per_class: dict[str, dict] = {}
    macro_f1 = None
    if shared:
        for label in LABELS:
            true_positive = sum(1 for i in shared if human[i] == label and consensus[i] == label)
            predicted = sum(1 for i in shared if consensus[i] == label)
            actual = sum(1 for i in shared if human[i] == label)
            precision = true_positive / predicted if predicted else None
            recall = true_positive / actual if actual else None
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision and recall and (precision + recall)
                else 0.0
            )
            per_class[label] = {
                "support": actual,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "recall_95ci": _wilson(true_positive, actual),
            }
        macro_f1 = sum(per_class[label]["f1"] for label in LABELS) / len(LABELS)

    detector_block: dict = {"supplied": False}
    if detector_predictions is not None:
        predictions = {
            str(r["audit_id"]): bool(r.get("fired"))
            for r in _read_jsonl(Path(detector_predictions))
        }
        rows = sorted(set(human) & set(predictions))
        answer_rows = [i for i in rows if human[i] == "ANSWER"]
        clean_rows = [i for i in rows if human[i] == "NONE"]
        fired_on_answer = sum(1 for i in answer_rows if predictions[i])
        fired_on_clean = sum(1 for i in clean_rows if predictions[i])
        detector_block = {
            "supplied": True,
            "n_rows": len(rows),
            "recall_on_human_answer": (fired_on_answer / len(answer_rows) if answer_rows else None),
            "recall_95ci": _wilson(fired_on_answer, len(answer_rows)),
            "fpr_on_human_none": fired_on_clean / len(clean_rows) if clean_rows else None,
            "fpr_95ci": _wilson(fired_on_clean, len(clean_rows)),
            "note": (
                "the frozen checkpoint's own claims, restated on the human-adjudicated "
                "subset. Confidence intervals are Wilson, because these denominators are "
                "small enough that a Wald interval can report a bound past 1.0."
            ),
        }

    measured = {
        "human_human_kappa": agreement.get("kappa"),
        "consensus_vs_human_macro_f1": macro_f1,
        "consensus_recall_on_human_answer": (per_class.get("ANSWER") or {}).get("recall"),
        "recall_NONE": (per_class.get("NONE") or {}).get("recall"),
        "recall_PARTIAL": (per_class.get("PARTIAL") or {}).get("recall"),
        "recall_ANSWER": (per_class.get("ANSWER") or {}).get("recall"),
        "n_unresolved": float(len(unresolved)),
        "n_provenance_failures": float(len(provenance)),
    }
    verdicts: dict[str, dict] = {}
    failures: list[str] = []
    for name, (operator, bound) in sorted(HUMAN_GATES.items()):
        value = measured.get(name)
        if value is None:
            ok, detail = False, "not measured"
        else:
            ok = value >= bound if operator == ">=" else value == bound
            detail = f"{value} {operator} {bound}"
        verdicts[name] = {"ok": ok, "measured": value, "bound": bound}
        if not ok:
            failures.append(f"{name}: {detail}")

    report = {
        "schema": "graph-detector-v4-3-human-report-v1",
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_sampled": len(manifest["rows"]),
        "n_human_adjudicated": len(human),
        "reportable_sample": bool(manifest.get("reportable")),
        "human_agreement": {
            "kappa": agreement.get("kappa"),
            "raw_agreement": agreement.get("raw_agreement"),
            "kappa_by_source": agreement.get("kappa_by_source"),
            "exploratory_only": agreement.get("exploratory_only"),
        },
        "model_consensus_vs_human": {
            "n_rows": len(shared),
            "macro_f1": macro_f1,
            "per_class": per_class,
        },
        "frozen_detector_vs_human": detector_block,
        "design_weighting": {
            "note": (
                "the sample is enriched on judge disagreement and PARTIAL. Every row "
                "carries its exact n_h/N_h inclusion probability and design weight in "
                "V4_3_HUMAN_SAMPLE.json; population-level statements must use them, and "
                "the unweighted figures above describe the SAMPLE."
            )
        },
        "provenance_failures": provenance,
        "gates": verdicts,
        "gate_failures": failures,
        "gates_passed": not failures,
    }
    atomic_json(out / REPORT_FILENAME, report)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / REPORT_FILENAME),
                "human_human_kappa": agreement.get("kappa"),
                "consensus_vs_human_macro_f1": macro_f1,
                "gates_passed": not failures,
                "gate_failures": failures,
            }
        )
    )
