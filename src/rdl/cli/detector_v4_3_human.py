"""``rdl graph-detector-v4-3-human-*`` -- the 250-row human validation, start to finish.

Three commands, in the order they may be run:

``human-sample``
    Draw the frozen sample and write one blinded file per rater. **The draw never reads a
    detector score.** That is the whole design constraint: a sample chosen where the
    detector is confident measures the detector's confidence, not its accuracy, and the
    resulting human-agreement number would be an artifact of the selection rule. Judge
    disagreement and rare PARTIAL rows ARE oversampled -- those are label properties, not
    detector properties -- and every row records the inclusion probability it was drawn
    with, so the overall estimate can be design-weighted back to the population.

``human-import``
    Read the two completed rater files, check they answer the sample they were given, and
    compute unweighted Cohen's kappa overall and by stratum.

``human-adjudicate``
    Resolve disagreements after both files are frozen, and refuse to do it before.

What a rater sees
-----------------
The conditioning question, the safe aliases, the candidate text, and the rubric. Not the
model labels, not the model identities, not the detector scores, not the population, not
the source filename, and not the split. The exported file is checked for all of those
before it is written, because a blinding that is merely intended is a blinding that
survives until the first convenient refactor.
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
    "cohens_kappa",
    "detector_v4_3_human_adjudicate",
    "detector_v4_3_human_import",
    "detector_v4_3_human_sample",
    "draw_sample",
]

HUMAN_SAMPLE_SCHEMA = "graph-detector-v4-3-human-sample-v1"
SAMPLE_FILENAME = "V4_3_HUMAN_SAMPLE.json"
RATER_FILENAME = "V4_3_HUMAN_RATER_{rater}.jsonl"
AGREEMENT_FILENAME = "V4_3_HUMAN_AGREEMENT.json"
ADJUDICATED_FILENAME = "V4_3_HUMAN_ADJUDICATED.jsonl"

LABELS = ("NONE", "PARTIAL", "ANSWER")

# The only fields a rater file may add. Anything else means the rater saw something the
# blinding was supposed to withhold.
RATER_FIELDS = frozenset({"audit_id", "answer_attempt", "notes"})

# What a blinded rater row may contain. An allowlist, for the reason the store's is.
EXPORTED_FIELDS = frozenset(
    {"audit_id", "conditioning_question", "subject_aliases", "candidate_text"}
)

# Oversampling weights. Both are properties of the LABELS, never of a detector score:
# a judge disagreement says the row is hard, and PARTIAL is the rare class whose recall
# the protocol gates. Recorded per row so the estimate can be weighted back.
WEIGHT_DISAGREEMENT = 3.0
WEIGHT_PARTIAL = 3.0
WEIGHT_BASE = 1.0


def _unit_hash(text: str) -> float:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16) / float(1 << 64)


def draw_sample(
    rows: Sequence[Mapping],
    *,
    n: int,
    stratum: str,
    salt: str = "v4.3-human",
) -> list[dict]:
    """Draw ``n`` rows by weighted content-addressed sampling, recording inclusion weights.

    Deterministic: a row's key is a hash of its id and the salt, divided by its weight, and
    the lowest keys win. Re-running the draw on the same rows gives the same sample, and
    adding rows does not reshuffle the ones already chosen.

    ``rows`` may carry ``judges_disagree`` and ``label``; neither is shown to a rater. They
    are read here only to set the weight, which is why the function takes label metadata
    and takes no scores at all.
    """
    scored: list[tuple[float, float, dict]] = []
    for row in rows:
        weight = WEIGHT_BASE
        if row.get("judges_disagree"):
            weight = max(weight, WEIGHT_DISAGREEMENT)
        if row.get("label") == "PARTIAL":
            weight = max(weight, WEIGHT_PARTIAL)
        key = _unit_hash(f"{salt}|{stratum}|{row['audit_id']}") / weight
        scored.append((key, weight, dict(row)))

    scored.sort(key=lambda t: (t[0], t[2]["audit_id"]))
    drawn = scored[:n]
    # The inclusion probability is approximated by the row's share of total weight, which
    # is what a design-weighted estimator needs. Approximate and RECORDED beats exact and
    # implicit: a reader can see the weighting scheme and redo the arithmetic.
    total_weight = sum(weight for _k, weight, _r in scored) or 1.0
    return [
        {
            **row,
            "stratum": stratum,
            "sampling_weight": weight,
            "inclusion_probability": min(1.0, n * weight / total_weight),
        }
        for _key, weight, row in drawn
    ]


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
    bundle: Path = typer.Option(
        DEFAULT_OUT_DIR / "DETECTOR_V4_3_PAIR_BUNDLE.json",
        "--bundle",
        help="the 1,019-row bundle; 125 rows are drawn from it.",
    ),
    fresh_audit: Path = typer.Option(
        None,
        "--fresh-audit",
        help="the fresh engineering audit; 125 rows are drawn from it. Optional until "
        "GPU 4 has produced one.",
    ),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir"),
    n_per_stratum: int = typer.Option(125, "--n-per-stratum"),
    raters: str = typer.Option("A,B", "--raters"),
) -> None:
    """Draw the frozen human sample and write one blinded file per rater."""
    if not Path(bundle).exists():
        raise typer.BadParameter(f"{bundle} is absent")
    payload = json.loads(Path(bundle).read_text(encoding="utf-8"))

    strata: dict[str, list[dict]] = {"original_1019": list(payload.get("pairs", ()))}
    if fresh_audit is not None:
        if not Path(fresh_audit).exists():
            raise typer.BadParameter(f"{fresh_audit} is absent")
        fresh = json.loads(Path(fresh_audit).read_text(encoding="utf-8"))
        strata["fresh_engineering"] = list(fresh.get("pairs", fresh.get("rows", ())))

    sample: list[dict] = []
    for stratum, rows in sorted(strata.items()):
        sample.extend(draw_sample(rows, n=n_per_stratum, stratum=stratum))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rater_ids = [r.strip() for r in raters.split(",") if r.strip()]
    for rater in rater_ids:
        path = out / RATER_FILENAME.format(rater=rater)
        path.write_text(
            "".join(dumps_canonical(_blinded(row)) + "\n" for row in sample), encoding="utf-8"
        )

    manifest = {
        "schema": HUMAN_SAMPLE_SCHEMA,
        "drawn_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": len(sample),
        "strata": {s: sum(1 for r in sample if r["stratum"] == s) for s in strata},
        "raters": rater_ids,
        "bundle_sha256": payload.get("bundle_sha256"),
        "sampling": {
            "rule": "content-addressed weighted draw; lowest hash/weight wins",
            "weights": {
                "base": WEIGHT_BASE,
                "judge_disagreement": WEIGHT_DISAGREEMENT,
                "partial_label": WEIGHT_PARTIAL,
            },
            "uses_detector_scores": False,
            "why_not": (
                "a sample drawn where the detector is confident measures the detector's "
                "confidence rather than its accuracy. Judge disagreement and the rare "
                "PARTIAL class are LABEL properties and are oversampled; every row records "
                "the weight it was drawn with so the overall estimate can be "
                "design-weighted back to the population."
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
                "stratum": row["stratum"],
                "sampling_weight": row["sampling_weight"],
                "inclusion_probability": row["inclusion_probability"],
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
                "strata": manifest["strata"],
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
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
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
                f"rater {rater} answered a different sample: {len(missing)} row(s) missing, "
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
        by_stratum[rows[key]["stratum"]].append((labels[first][key], labels[second][key]))

    disagreements = sorted(k for k in ordered if labels[first][k] != labels[second][k])
    report = {
        "schema": "graph-detector-v4-3-human-agreement-v1",
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raters": rater_ids,
        "n_rows": len(ordered),
        "kappa": cohens_kappa(a, b),
        "raw_agreement": sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(a),
        "kappa_by_stratum": {
            stratum: cohens_kappa([x for x, _ in pairs], [y for _, y in pairs])
            for stratum, pairs in sorted(by_stratum.items())
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
    decisions: Path = typer.Option(
        None,
        "--decisions",
        help="JSONL of {audit_id, answer_attempt} for the disagreements only.",
    ),
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
        for line in Path(decisions).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
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
                    "the label gate requires zero unresolved rows. An unresolved "
                    "disagreement is not a NONE."
                ),
            }
        )
    )
