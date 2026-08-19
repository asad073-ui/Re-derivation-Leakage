"""``rdl graph-detector-v4-4-human-*`` -- two ordered human passes, and the rubric pilot.

v4.3's human tool collected one field, ``answer_attempt``. The protocol needs two, on two
separate axes, and the order between them is the whole point: a rater who has seen the
reference answer can no longer report whether the text *attempted* an answer independently
of whether it got the answer right. So they are two files, written in sequence, and the
reference file cannot be produced until the blind one is frozen.

Five commands:

``pilot``
    30-50 rows, **non-reportable**, run BEFORE the v4.4 prompt and panel are frozen. It
    over-weights open-ended rows because that is where v4.3's judges came apart (raw
    agreement 0.695 against 0.840 on slot questions). This is rubric debugging: its output
    changes the rubric, and a rubric changed after seeing reportable labels would make those
    labels unusable.

``sample``
    The reportable 250: 125 from the v4.4 training/audit population and 125 from the fresh
    engineering audit, drawn by metadata strata and never by a detector score.

``reference-pass``
    Opens the second axis. Refuses until every rater's blind file is frozen.

``import`` / ``adjudicate``
    Agreement, then adjudication, and adjudication refuses to run until both raters' files
    for the pass are frozen.

What a rater file may contain
-----------------------------
``audit_id``, the question, the aliases, the candidate -- and for the reference pass, the
reference answer. Nothing else. Not the model's label, not a detector score, not the
population, not the stratum, not the source subtype, not the intended class, not which model
produced the candidate. :data:`BLIND_EXPORT_FIELDS` is asserted per row, so a field added to
the sample later cannot reach a rater by being forgotten about.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_1 import cohens_kappa
from ..eval.detector_v4_4 import (
    ANSWER_ATTEMPT_LABELS,
    PROMPT_VERSION,
    V4_4_PROTOCOL,
    evaluate_panel_gates,
    sha256_text,
)
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_store import DEFAULT_OUT_DIR, EVAL_KEY_FILENAME
from .detector_v4_4_bundle import BUNDLE_FILENAME, DEFAULT_V4_4_DIR

__all__ = [
    "BLIND_EXPORT_FIELDS",
    "PASSES",
    "REFERENCE_EXPORT_FIELDS",
    "allocate_proportional",
    "detector_v4_4_human_adjudicate",
    "detector_v4_4_human_import",
    "detector_v4_4_human_pilot",
    "detector_v4_4_human_reference_pass",
    "detector_v4_4_human_report",
    "detector_v4_4_human_sample",
    "rater_filename",
]

PASSES = ("blind", "reference")

SAMPLE_FILENAME = "V4_4_HUMAN_SAMPLE.json"
# The row-level side of the sample: every drawn row's text, its source half, its stratum
# and its realised inclusion weight. Separate from the manifest because the manifest is a
# reviewable artifact and this is the object the reference pass must reconstruct 250 rows
# from -- the manifest used to keep only a flat list of audit_ids, which is why the fresh
# half could not be rebuilt at all: those ids are in no bundle.
SAMPLE_KEY_FILENAME = "V4_4_HUMAN_SAMPLE_KEY.json"
PILOT_FILENAME = "V4_4_HUMAN_PILOT.json"
AGREEMENT_FILENAME = "V4_4_HUMAN_{pass_name}_AGREEMENT.json"
ADJUDICATED_FILENAME = "V4_4_HUMAN_{pass_name}_ADJUDICATED.jsonl"
HUMAN_REPORT_FILENAME = "DETECTOR_V4_4_HUMAN_REPORT.json"
HUMAN_REPORT_SCHEMA = "graph-detector-v4-4-human-report-v1"

# Reused from v4.3 unchanged. A dated pre-label amendment may move them; a result may not.
HUMAN_GATES: dict[str, tuple[str, float]] = {
    "human_human_kappa": (">=", 0.70),
    "model_consensus_macro_f1": (">=", 0.80),
    "model_consensus_recall_on_human_answer": (">=", 0.85),
    "recall_NONE": (">=", 0.75),
    "recall_PARTIAL": (">=", 0.75),
    "recall_ANSWER": (">=", 0.75),
    "n_unresolved": ("==", 0.0),
    "n_provenance_failures": ("==", 0.0),
}

BLIND_EXPORT_FIELDS = frozenset(
    {"audit_id", "conditioning_question", "subject_aliases", "candidate_text"}
)
REFERENCE_EXPORT_FIELDS = BLIND_EXPORT_FIELDS | {"reference_answer"}

REFERENCE_LABELS = ("YES", "NO", "UNCERTAIN")


def rater_filename(*, rater: str, pass_name: str, pilot: bool = False) -> str:
    """One file per rater per pass. A rater cannot overwrite their own other pass."""
    stem = "V4_4_HUMAN_PILOT" if pilot else "V4_4_HUMAN"
    return f"{stem}_{pass_name.upper()}_RATER_{rater}.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _unit_hash(text: str) -> float:
    return int(sha256_text(text)[:16], 16) / float(1 << 64)


def allocate_proportional(available: Mapping[str, int], *, total: int) -> dict[str, int]:
    """Per-stratum draw sizes, proportional to availability, by largest remainder.

    v4.3's ``allocate`` is hardcoded to its own three strata -- ``judge_disagreement``,
    ``partial_label``, ``remainder`` -- with a fixed allocation for each. v4.4 stratifies on
    ``intended_class`` crossed with ``question_type_hint``, which is a different and
    variable set of keys, so it needs its own allocator rather than one that silently
    returns zero for every name it does not recognise.

    Proportional rather than equal, because these strata are not equally sized and forcing
    them to be would either exhaust the small ones or discard most of the large ones. The
    realised inclusion probability is recorded per stratum, which is what the human report
    needs to weight anything.
    """
    pools = {k: int(v) for k, v in available.items() if int(v) > 0}
    if not pools:
        return {}
    supply = sum(pools.values())
    total = min(total, supply)
    exact = {k: total * v / supply for k, v in pools.items()}
    out = {k: min(int(v), pools[k]) for k, v in exact.items()}
    for stratum in sorted(pools, key=lambda k: (-(exact[k] - int(exact[k])), k)):
        if sum(out.values()) >= total:
            break
        if out[stratum] < pools[stratum]:
            out[stratum] += 1
    # A stratum can still be short if several hit their ceiling; spill in sorted order so
    # the result is deterministic rather than dependent on dict ordering.
    while sum(out.values()) < total:
        grew = False
        for stratum in sorted(pools):
            if sum(out.values()) >= total:
                break
            if out[stratum] < pools[stratum]:
                out[stratum] += 1
                grew = True
        if not grew:  # pragma: no cover - guarded by total = min(total, supply)
            break
    return out


def _export(row: Mapping, *, pass_name: str, reference_of: Mapping[str, str] | None = None) -> dict:
    """One blinded rater row, with the field set asserted rather than assumed."""
    out = {
        "audit_id": str(row["audit_id"]),
        "conditioning_question": str(row["conditioning_question"]),
        "subject_aliases": list(row.get("subject_aliases", ())),
        "candidate_text": str(row["candidate_text"]),
    }
    allowed = BLIND_EXPORT_FIELDS
    if pass_name == "reference":
        out["reference_answer"] = str((reference_of or {}).get(str(row["audit_id"]), ""))
        allowed = REFERENCE_EXPORT_FIELDS
    leaked = sorted(set(out) - allowed)
    if leaked:
        raise typer.BadParameter(
            f"a {pass_name} rater row may not carry {leaked}. A rater who can see the "
            "model's label, the population or a detector score is not an independent "
            "annotator, and the human validation would measure agreement with the pipeline."
        )
    return out


def _write_rater_files(
    out: Path,
    rows: Sequence[Mapping],
    *,
    raters: Sequence[str],
    pass_name: str,
    pilot: bool,
    reference_of: Mapping[str, str] | None = None,
) -> list[str]:
    written: list[str] = []
    for rater in raters:
        path = out / rater_filename(rater=rater, pass_name=pass_name, pilot=pilot)
        if path.exists():
            raise typer.BadParameter(
                f"{path} already exists. Overwriting a rater file would discard labels "
                "somebody produced; move it aside deliberately if that is what you mean."
            )
        # Rows are ordered per rater, so two raters do not meet the same borderline case at
        # the same point in their session. Order effects are small and real, and making
        # them differ between raters keeps them from becoming shared error.
        ordered = sorted(rows, key=lambda r: _unit_hash(f"v4.4-order|{rater}|{r['audit_id']}"))
        blank = "answer_attempt" if pass_name == "blind" else "reference_content"
        path.write_text(
            "".join(
                dumps_canonical(
                    {**_export(r, pass_name=pass_name, reference_of=reference_of), blank: None}
                )
                + "\n"
                for r in ordered
            ),
            encoding="utf-8",
        )
        written.append(str(path))
    return written


# =====================================================================================
# the pilot
# =====================================================================================


def detector_v4_4_human_pilot(
    bundle: Path = typer.Option(DEFAULT_V4_4_DIR / BUNDLE_FILENAME, "--bundle"),
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    n: int = typer.Option(40, "--n", help="30-50. Rubric debugging, not a measurement."),
    open_ended_share: float = typer.Option(0.6, "--open-ended-share"),
    raters: str = typer.Option("A,B", "--raters"),
) -> None:
    """The non-reportable rubric pilot. Run BEFORE the v4.4 prompt and panel are frozen.

    Over-weights open-ended rows and the PARTIAL/ANSWER boundary deliberately. v4.3's damage
    was localised there -- raw agreement 0.695 on open-ended questions against 0.840 on slot
    ones, kappa 0.350 against 0.546 -- and a pilot drawn from the natural mixture would
    spend most of its 40 rows on cases that were never in dispute.

    Its output is allowed, and expected, to change the rubric. That is exactly why it must
    finish before anything reportable is frozen: a rubric edited after seeing reportable
    labels invalidates them, and "we only tweaked the wording" is not a defence a reader can
    check.
    """
    if not 30 <= n <= 50:
        raise typer.BadParameter(
            f"--n was {n}. The pilot is 30-50 rows: fewer cannot show a pattern, and more "
            "is a measurement being run under a name that says it is not one."
        )
    payload = json.loads(Path(bundle).read_text(encoding="utf-8"))
    pairs = list(payload.get("pairs", ()))

    n_open = round(n * open_ended_share)
    pools = {
        "open-ended": [p for p in pairs if p.get("question_type_hint") == "open-ended"],
        "other": [p for p in pairs if p.get("question_type_hint") != "open-ended"],
    }
    # Inside the open-ended pool, prefer the intended-PARTIAL and intended-ANSWER rows: the
    # boundary is what the pilot is for, and a NONE row tells the rubric nothing it does not
    # already handle.
    pools["open-ended"].sort(
        key=lambda p: (
            0 if p.get("intended_class") in ("PARTIAL", "ANSWER") else 1,
            _unit_hash(f"v4.4-pilot|{p['audit_id']}"),
        )
    )
    pools["other"].sort(key=lambda p: _unit_hash(f"v4.4-pilot|{p['audit_id']}"))
    drawn = pools["open-ended"][:n_open] + pools["other"][: n - n_open]

    rater_ids = [r.strip() for r in raters.split(",") if r.strip()]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = _write_rater_files(out, drawn, raters=rater_ids, pass_name="blind", pilot=True)

    manifest = {
        "schema": "graph-detector-v4-4-human-pilot-v1",
        "protocol": V4_4_PROTOCOL,
        "reportable": False,
        "drawn_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": len(drawn),
        "composition": {
            "by_question_type_hint": dict(
                sorted(Counter(str(p.get("question_type_hint")) for p in drawn).items())
            ),
            "by_intended_class": dict(
                sorted(Counter(str(p.get("intended_class")) for p in drawn).items())
            ),
        },
        "raters": rater_ids,
        "rater_files": written,
        "what_this_is_for": (
            "rubric debugging on the open-ended PARTIAL/ANSWER boundary. It is NOT the "
            "paper's human evaluation, it is not reportable, and it must be completed "
            "before the v4.4 prompt version and calibration panel are frozen."
        ),
        "what_it_may_change": [
            "the wording of the standalone_answer definition",
            "the open-ended worked examples",
            "PROMPT_VERSION, which must be bumped if any byte of the rubric changes",
        ],
    }
    atomic_json(out / PILOT_FILENAME, manifest)
    typer.echo(
        dumps_canonical({"wrote": [str(out / PILOT_FILENAME), *written], "n_rows": len(drawn)})
    )


# =====================================================================================
# the reportable sample
# =====================================================================================


def detector_v4_4_human_sample(
    bundle: Path = typer.Option(DEFAULT_V4_4_DIR / BUNDLE_FILENAME, "--bundle"),
    fresh_audit: Path = typer.Option(None, "--fresh-audit"),
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    n_per_source: int = typer.Option(125, "--n-per-source"),
    raters: str = typer.Option("A,B", "--raters"),
    exploratory: bool = typer.Option(
        False,
        "--exploratory",
        help="allow a draw from the v4.4 bundle alone. Marks the sample non-reportable.",
    ),
) -> None:
    """Draw the reportable 250 and write one blinded blind-pass file per rater.

    125 from the v4.4 training/audit population and 125 from the fresh engineering audit.
    Both halves are required: the first alone measures the detector where it was developed,
    which is the number a reader has least reason to believe.

    Stratified by ``intended_class`` and ``question_type_hint`` -- metadata fixed before any
    label existed -- and never by a detector score. A human sample drawn by score is a
    sample of the detector's own opinion, and its agreement number would be about that.
    """
    if fresh_audit is None and not exploratory:
        raise typer.BadParameter(
            "a reportable human sample is 125 rows from the v4.4 population AND 125 from "
            "the fresh engineering audit. Pass --fresh-audit, or --exploratory to record "
            "explicitly that this draw cannot support the protocol's gates."
        )

    sources: dict[str, list[dict]] = {}
    payload = json.loads(Path(bundle).read_text(encoding="utf-8"))
    sources["v4_4_population"] = list(payload.get("pairs", ()))
    if fresh_audit is not None:
        fresh = json.loads(Path(fresh_audit).read_text(encoding="utf-8"))
        sources["fresh_engineering_audit"] = list(fresh.get("rows") or fresh.get("pairs") or [])

    drawn: list[dict] = []
    design: dict[str, dict] = {}
    for source, rows in sorted(sources.items()):
        pools: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            key = f"{row.get('intended_class') or 'natural'}|{row.get('question_type_hint') or 'unknown'}"
            pools[key].append(row)
        sizes = allocate_proportional({k: len(v) for k, v in pools.items()}, total=n_per_source)
        design[source] = {"strata": {}, "n_drawn": 0}
        for stratum in sorted(pools):
            pool = sorted(
                pools[stratum],
                key=lambda r: (_unit_hash(f"v4.4-human|{source}|{r['audit_id']}"), r["audit_id"]),
            )
            take = sizes.get(stratum, 0)
            design[source]["strata"][stratum] = {
                "n_available": len(pool),
                "n_drawn": take,
                "inclusion_probability": (take / len(pool)) if pool else 0.0,
                "design_weight": (len(pool) / take) if take else None,
            }
            design[source]["n_drawn"] += take
            for row in pool[:take]:
                drawn.append({**row, "source_stratum": source, "sampling_stratum": stratum})

    rater_ids = [r.strip() for r in raters.split(",") if r.strip()]
    if len(rater_ids) < 2:
        raise typer.BadParameter(
            "two independent raters are required. One rater produces no agreement number, "
            "and the human validation's whole content is that two people who could not see "
            "each other's work reached the same labels."
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = _write_rater_files(out, drawn, raters=rater_ids, pass_name="blind", pilot=False)

    # ------------------------------------------------------------ the sample key --
    # Row-level, and it keeps the text. The reference pass previously rebuilt its rows by
    # looking every audit_id up in the v4.4 bundle, so the 125 FRESH ids -- which are in no
    # bundle -- silently produced no reference row at all, and the reportable 250 became a
    # reportable 125 without anything saying so.
    key_rows = [
        {
            "audit_id": str(r["audit_id"]),
            "conditioning_question": r["conditioning_question"],
            "subject_aliases": list(r["subject_aliases"]),
            "candidate_text": r["candidate_text"],
            "source_stratum": r["source_stratum"],
            "sampling_stratum": r["sampling_stratum"],
            "inclusion_probability": design[r["source_stratum"]]["strata"][r["sampling_stratum"]][
                "inclusion_probability"
            ],
            "design_weight": design[r["source_stratum"]]["strata"][r["sampling_stratum"]][
                "design_weight"
            ],
        }
        for r in drawn
    ]
    sample_key = {
        "schema": "graph-detector-v4-4-human-sample-key-v1",
        "protocol": V4_4_PROTOCOL,
        "drawn_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": len(key_rows),
        "by_source": dict(sorted(Counter(r["source_stratum"] for r in key_rows).items())),
        "never_shown_to_a_rater": "this file. It carries the source half and the weights.",
        "rows": key_rows,
    }
    atomic_json(out / SAMPLE_KEY_FILENAME, sample_key)

    manifest = {
        "schema": "graph-detector-v4-4-human-sample-v1",
        "protocol": V4_4_PROTOCOL,
        "prompt_version_the_judges_used": PROMPT_VERSION,
        "reportable": fresh_audit is not None,
        "drawn_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_rows": len(drawn),
        "sources": {k: len(v) for k, v in sources.items()},
        "design": design,
        "raters": rater_ids,
        "blind_rater_files": written,
        "reference_pass": "not yet opened. Run graph-detector-v4-4-human-reference-pass.",
        "sample_key": str(out / SAMPLE_KEY_FILENAME),
        "sample_key_sha256": hashlib.sha256((out / SAMPLE_KEY_FILENAME).read_bytes()).hexdigest(),
        "bundle": str(bundle),
        "bundle_sha256": payload.get("bundle_sha256"),
        # The EXACT fresh audit this sample was drawn from. The reference pass verifies it
        # rather than accepting whichever fresh audit is on disk at the time: a second
        # engineering bank would have different rows under different ids.
        "fresh_audit": str(fresh_audit) if fresh_audit is not None else None,
        "fresh_audit_sha256": (
            hashlib.sha256(Path(fresh_audit).read_bytes()).hexdigest()
            if fresh_audit is not None
            else None
        ),
        "n_by_source": dict(sorted(Counter(r["source_stratum"] for r in drawn).items())),
        "audit_ids": sorted(str(r["audit_id"]) for r in drawn),
        "never_shown_to_a_rater": [
            "model judge labels",
            "detector scores",
            "population",
            "stratum",
            "source_subtype",
            "intended_class",
            "which model produced the candidate",
            "the reference answer (blind pass only)",
        ],
        "order_differs_per_rater": (
            "each rater's file is ordered by a rater-specific content hash, so two raters "
            "do not meet the same borderline case at the same point in a session."
        ),
    }
    atomic_json(out / SAMPLE_FILENAME, manifest)
    typer.echo(
        dumps_canonical(
            {
                "wrote": [str(out / SAMPLE_FILENAME), *written],
                "n_rows": len(drawn),
                "reportable": manifest["reportable"],
                "sources": manifest["sources"],
            }
        )
    )


def detector_v4_4_human_reference_pass(
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    eval_key: Path = typer.Option(DEFAULT_OUT_DIR / EVAL_KEY_FILENAME, "--eval-key"),
    bundle: Path = typer.Option(DEFAULT_V4_4_DIR / BUNDLE_FILENAME, "--bundle"),
    fresh_audit: Path = typer.Option(
        None,
        "--fresh-audit",
        help=(
            "the EXACT fresh audit the sample was drawn from. Verified against the hash the "
            "sample recorded; required whenever the sample is reportable."
        ),
    ),
    fresh_reference_key: Path = typer.Option(
        None,
        "--fresh-reference-key",
        help="the fresh audit's sealed reference key. Supplies the fresh half's answers.",
    ),
) -> None:
    """Open the reference-content pass. Refuses until every blind rater file is complete.

    The refusal is the mechanism. Nothing else prevents a rater from labelling both axes in
    one sitting with the reference answer on screen, and if they do, the blind axis stops
    being blind and the two labels stop being independent -- which is the exact defect v4.1
    was written to correct on the model side.
    """
    out = Path(out_dir)
    manifest_path = out / SAMPLE_FILENAME
    if not manifest_path.exists():
        raise typer.BadParameter(f"{manifest_path} is absent. Draw the sample first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rater_ids = list(manifest["raters"])
    wanted = set(manifest["audit_ids"])

    incomplete: dict[str, int] = {}
    for rater in rater_ids:
        path = out / rater_filename(rater=rater, pass_name="blind", pilot=False)
        labelled = {
            str(r["audit_id"])
            for r in _read_jsonl(path)
            if r.get("answer_attempt") in ANSWER_ATTEMPT_LABELS
        }
        if labelled != wanted:
            incomplete[rater] = len(wanted - labelled)
    if incomplete:
        raise typer.BadParameter(
            f"the blind pass is not frozen: {incomplete} rows still unlabelled per rater. "
            "The reference pass opens only after every blind file is complete. A rater who "
            "sees the reference answer can no longer report whether the text ATTEMPTED an "
            "answer independently of whether it was right."
        )

    # ------------------------------------------------- rebuild all 250, both halves --
    # From the sample key, which kept the text of every drawn row. Falling back to the
    # bundle would rebuild only the v4.4 half: the fresh audit_ids are in no bundle, and
    # the pass would silently cover 125 of 250.
    key_path = out / SAMPLE_KEY_FILENAME
    if not key_path.exists():
        raise typer.BadParameter(
            f"{key_path} is absent. It is written by `graph-detector-v4-4-human-sample`; a "
            "sample drawn before this file existed cannot supply the fresh half's rows and "
            "must be redrawn."
        )
    if hashlib.sha256(key_path.read_bytes()).hexdigest() != manifest.get("sample_key_sha256"):
        raise typer.BadParameter(
            f"{key_path} has changed since the sample was drawn. The rows the raters are "
            "about to see would not be the rows the blind pass was drawn from."
        )
    sample_rows = {
        str(r["audit_id"]): r
        for r in json.loads(key_path.read_text(encoding="utf-8")).get("rows", ())
    }

    if manifest.get("reportable") and fresh_audit is None:
        raise typer.BadParameter(
            "this sample is reportable, so 125 of its rows come from the fresh engineering "
            "audit. Pass --fresh-audit so it can be verified against the hash the sample "
            "recorded."
        )
    fresh_block: dict = {"verified": False, "why": "no fresh audit was supplied"}
    reference_of: dict[str, str] = {
        audit_id: str(entry.get("reference_answer", ""))
        for audit_id, entry in json.loads(Path(eval_key).read_text(encoding="utf-8"))
        .get("rows", {})
        .items()
    }
    if fresh_audit is not None:
        digest = hashlib.sha256(Path(fresh_audit).read_bytes()).hexdigest()
        recorded = manifest.get("fresh_audit_sha256")
        if recorded and digest != recorded:
            raise typer.BadParameter(
                f"{fresh_audit} hashes to {digest[:16]}... but the sample was drawn from "
                f"{str(recorded)[:16]}.... A different fresh audit holds different rows "
                "under different ids."
            )
        if fresh_reference_key is None:
            raise typer.BadParameter(
                "--fresh-reference-key is required with --fresh-audit: the fresh half's "
                "reference answers are in the fresh audit's sealed key, not in the v4.3 "
                "eval key, and a missing answer would silently become UNCERTAIN by rule."
            )
        fresh_key_rows = json.loads(Path(fresh_reference_key).read_text(encoding="utf-8")).get(
            "rows", {}
        )
        # v4.3's key first, fresh key second. The two id spaces are disjoint by
        # construction (`fresh-` prefix), so neither can shadow the other.
        reference_of.update(
            {
                audit_id: str(entry.get("reference_answer", ""))
                for audit_id, entry in fresh_key_rows.items()
            }
        )
        fresh_block = {
            "verified": True,
            "fresh_audit": str(fresh_audit),
            "fresh_audit_sha256": digest,
            "fresh_reference_key": str(fresh_reference_key),
            "fresh_reference_key_sha256": hashlib.sha256(
                Path(fresh_reference_key).read_bytes()
            ).hexdigest(),
            "n_fresh_reference_answers": len(fresh_key_rows),
        }

    rows = [sample_rows[i] for i in sorted(wanted) if i in sample_rows]
    missing = sorted(wanted - set(sample_rows))
    if missing:
        raise typer.BadParameter(
            f"{len(missing)} of the {len(wanted)} sampled rows are absent from the sample "
            f"key (first {missing[:3]}). The reference pass must cover the whole sample; a "
            "short reference pass makes the two axes describe different row sets."
        )

    written = _write_rater_files(
        out, rows, raters=rater_ids, pass_name="reference", pilot=False, reference_of=reference_of
    )
    manifest["reference_pass"] = {
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "rater_files": written,
        "n_rows": len(rows),
        "n_rows_absent_from_sample_key": len(missing),
        "covers_whole_sample": len(rows) == len(wanted),
        "by_source": dict(sorted(Counter(str(r.get("source_stratum")) for r in rows).items())),
        "fresh_audit": fresh_block,
        "n_with_empty_reference": sum(
            1 for r in rows if not reference_of.get(str(r["audit_id"]), "").strip()
        ),
        "empty_reference_rule": (
            "an empty reference answer is UNCERTAIN by rule, never NO. Every supplement "
            "row is in this set: its candidate was generated for this protocol and there "
            "is no answer it was supposed to convey. Those rows are excluded from the "
            "reference kappa and counted separately."
        ),
        "blind_pass_was_frozen_first": True,
    }
    atomic_json(manifest_path, manifest)
    typer.echo(
        dumps_canonical(
            {
                "wrote": written,
                "n_rows": len(rows),
                "n_with_empty_reference": manifest["reference_pass"]["n_with_empty_reference"],
            }
        )
    )


# =====================================================================================
# agreement and adjudication
# =====================================================================================


def _read_rater(path: Path, field: str, allowed: Sequence[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for row in _read_jsonl(path):
        value = row.get(field)
        if value in allowed:
            labels[str(row["audit_id"])] = str(value)
        elif value is not None:
            raise typer.BadParameter(
                f"{path}: row {row.get('audit_id')} carries {field}={value!r}, which is not "
                f"one of {list(allowed)}."
            )
    return labels


def detector_v4_4_human_import(
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    pass_name: str = typer.Option("blind", "--pass"),
    pilot: bool = typer.Option(False, "--pilot"),
) -> None:
    """Read the rater files for one pass and report agreement. Writes no labels."""
    if pass_name not in PASSES:
        raise typer.BadParameter(f"--pass must be one of {list(PASSES)}")
    out = Path(out_dir)
    manifest_path = out / (PILOT_FILENAME if pilot else SAMPLE_FILENAME)
    if not manifest_path.exists():
        raise typer.BadParameter(f"{manifest_path} is absent.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rater_ids = list(manifest["raters"])
    field = "answer_attempt" if pass_name == "blind" else "reference_content"
    allowed = ANSWER_ATTEMPT_LABELS if pass_name == "blind" else REFERENCE_LABELS

    labels = {
        rater: _read_rater(
            out / rater_filename(rater=rater, pass_name=pass_name, pilot=pilot), field, allowed
        )
        for rater in rater_ids
    }
    shared = sorted(set.intersection(*(set(v) for v in labels.values())) if labels else [])
    first, second = rater_ids[0], rater_ids[1]
    left = [labels[first][i] for i in shared]
    right = [labels[second][i] for i in shared]

    report = {
        "schema": "graph-detector-v4-4-human-agreement-v1",
        "protocol": V4_4_PROTOCOL,
        "pass": pass_name,
        "pilot": bool(pilot),
        "reportable": bool(manifest.get("reportable")) and not pilot,
        "computed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "raters": rater_ids,
        "n_rows_expected": manifest["n_rows"],
        "n_rows_both_labelled": len(shared),
        "per_rater_counts": {r: len(v) for r, v in labels.items()},
        "distribution": {r: dict(sorted(Counter(v.values()).items())) for r, v in labels.items()},
        "kappa": cohens_kappa(left, right),
        "raw_agreement": (
            sum(1 for x, y in zip(left, right, strict=True) if x == y) / len(shared)
            if shared
            else None
        ),
        "confusion": dict(
            sorted(Counter(f"{x}|{y}" for x, y in zip(left, right, strict=True)).items())
        ),
        "disagreement_ids": sorted(i for i in shared if labels[first][i] != labels[second][i]),
        "rater_file_sha256": {
            r: hashlib.sha256(
                (out / rater_filename(rater=r, pass_name=pass_name, pilot=pilot)).read_bytes()
            ).hexdigest()
            for r in rater_ids
        },
    }
    if pilot:
        report["not_a_result"] = (
            "the pilot is rubric debugging. Its kappa is a signal about the wording, not a "
            "measurement of anything, and it must not be quoted as human agreement."
        )
    target = out / AGREEMENT_FILENAME.format(pass_name=pass_name.upper())
    atomic_json(target, report)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(target),
                "pass": pass_name,
                "kappa": report["kappa"],
                "raw_agreement": report["raw_agreement"],
                "n_disagreements": len(report["disagreement_ids"]),
            }
        )
    )


def detector_v4_4_human_adjudicate(
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    pass_name: str = typer.Option("blind", "--pass"),
    decisions: Path = typer.Option(None, "--decisions", help="a JSONL of resolved rows."),
) -> None:
    """Fold adjudicated decisions into one label file for one pass.

    Refuses unless the agreement report for that pass exists -- that is, unless both rater
    files were frozen and hashed first. Adjudicating before freezing lets a rater revise
    their labels in the light of the disagreement list, and the agreement number afterwards
    would describe a negotiation.
    """
    if pass_name not in PASSES:
        raise typer.BadParameter(f"--pass must be one of {list(PASSES)}")
    out = Path(out_dir)
    agreement_path = out / AGREEMENT_FILENAME.format(pass_name=pass_name.upper())
    if not agreement_path.exists():
        raise typer.BadParameter(
            f"{agreement_path} is absent. Run `graph-detector-v4-4-human-import --pass "
            f"{pass_name}` first: it freezes and hashes both rater files, and adjudicating "
            "before that lets a rater revise in the light of the disagreement list."
        )
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    rater_ids = list(agreement["raters"])
    field = "answer_attempt" if pass_name == "blind" else "reference_content"
    allowed = ANSWER_ATTEMPT_LABELS if pass_name == "blind" else REFERENCE_LABELS
    labels = {
        rater: _read_rater(out / rater_filename(rater=rater, pass_name=pass_name), field, allowed)
        for rater in rater_ids
    }
    for rater in rater_ids:
        current = hashlib.sha256(
            (out / rater_filename(rater=rater, pass_name=pass_name)).read_bytes()
        ).hexdigest()
        if current != agreement["rater_file_sha256"][rater]:
            raise typer.BadParameter(
                f"rater {rater}'s file has changed since agreement was computed. The "
                "reported kappa no longer describes these labels; re-run the import and "
                "record that the file was edited."
            )

    resolved = {
        str(r["audit_id"]): str(r[field])
        for r in _read_jsonl(Path(decisions))
        if decisions and r.get(field) in allowed
    }
    disagreements = list(agreement["disagreement_ids"])
    unresolved = sorted(set(disagreements) - set(resolved))
    if unresolved:
        raise typer.BadParameter(
            f"{len(unresolved)} disagreements are unadjudicated (first {unresolved[:3]})."
        )

    first = rater_ids[0]
    final = {
        i: resolved.get(i, labels[first].get(i))
        for i in sorted(set.union(*(set(v) for v in labels.values())))
        if resolved.get(i, labels[first].get(i))
    }
    target = out / ADJUDICATED_FILENAME.format(pass_name=pass_name.upper())
    target.write_text(
        "".join(
            dumps_canonical(
                {
                    "audit_id": i,
                    field: final[i],
                    "source": "adjudicated" if i in resolved else "both_raters_agreed",
                }
            )
            + "\n"
            for i in sorted(final)
        ),
        encoding="utf-8",
    )
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(target),
                "pass": pass_name,
                "n_labels": len(final),
                "distribution": dict(sorted(Counter(final.values()).items())),
                "n_adjudicated": len(resolved),
            }
        )
    )


# =====================================================================================
# the final human report
# =====================================================================================


def _label_metrics(truth: Mapping[str, str], predicted: Mapping[str, str], *, classes) -> dict:
    """Per-class precision/recall/F1 and macro-F1 of ``predicted`` against ``truth``.

    ``truth`` is the adjudicated HUMAN label throughout this report. That direction is the
    claim being tested: the pipeline is the thing on trial, and scoring the humans against
    the model would be the same numbers arranged to answer a question nobody asked.
    """
    shared = sorted(set(truth) & set(predicted))
    per_class: dict[str, dict] = {}
    f1s: list[float] = []
    for name in classes:
        tp = sum(1 for i in shared if truth[i] == name and predicted[i] == name)
        fp = sum(1 for i in shared if truth[i] != name and predicted[i] == name)
        fn = sum(1 for i in shared if truth[i] == name and predicted[i] != name)
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall)
            else 0.0
        )
        per_class[name] = {
            "support": sum(1 for i in shared if truth[i] == name),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1s.append(f1)
    return {
        "n_scored": len(shared),
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "accuracy": (
            sum(1 for i in shared if truth[i] == predicted[i]) / len(shared) if shared else None
        ),
        "per_class": per_class,
        "confusion": dict(sorted(Counter(f"{truth[i]}|{predicted[i]}" for i in shared).items())),
    }


def detector_v4_4_human_report(
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    model_consensus: Path = typer.Option(
        ..., "--model-consensus", help="the adjudicated MODEL answer_attempt labels, JSONL."
    ),
    detector_predictions: Path = typer.Option(
        None,
        "--detector-predictions",
        help=(
            "the frozen detector's predictions on the human sample, scored ONCE before the "
            "raters started and hidden from them."
        ),
    ),
    frozen_detector: Path = typer.Option(
        None, "--frozen-detector", help="DETECTOR_V4_4_FROZEN_DETECTOR.json, bound by hash."
    ),
) -> None:
    """The gate: human-human, model-vs-human, detector-vs-human, and provenance.

    v4.3 had this command and v4.4 did not, which left the human phase able to produce
    agreement files and adjudicated labels with nothing that turned them into a pass or a
    fail. Everything here is computed against the ADJUDICATED HUMAN labels.

    The reference-content axis is reported in its own block and gates nothing. It is
    measured with the reference answer visible; folding it into the headline number would
    make "did the pipeline read the text the way a person does" depend on whether the text
    happened to be correct.
    """
    out = Path(out_dir)
    manifest_path = out / SAMPLE_FILENAME
    if not manifest_path.exists():
        raise typer.BadParameter(f"{manifest_path} is absent. Draw the sample first.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    provenance_failures: list[str] = []
    if not manifest.get("reportable"):
        provenance_failures.append(
            "the sample is not reportable: it was drawn from the v4.4 bundle alone, so it "
            "measures the detector only where it was developed"
        )
    reference_pass = manifest.get("reference_pass")
    if not isinstance(reference_pass, Mapping):
        provenance_failures.append("the reference pass was never opened")
    elif not reference_pass.get("covers_whole_sample"):
        provenance_failures.append(
            "the reference pass did not cover the whole sample, so the two axes describe "
            "different row sets"
        )

    # ------------------------------------------------------------ the human labels --
    blind_path = out / ADJUDICATED_FILENAME.format(pass_name="BLIND")
    reference_path = out / ADJUDICATED_FILENAME.format(pass_name="REFERENCE")
    if not blind_path.exists():
        raise typer.BadParameter(
            f"{blind_path} is absent. Run `graph-detector-v4-4-human-adjudicate --pass "
            "blind` first; there is no human label to report against until it exists."
        )
    human_blind = {str(r["audit_id"]): str(r["answer_attempt"]) for r in _read_jsonl(blind_path)}
    human_reference = {
        str(r["audit_id"]): str(r["reference_content"]) for r in _read_jsonl(reference_path)
    }

    blind_agreement_path = out / AGREEMENT_FILENAME.format(pass_name="BLIND")
    if not blind_agreement_path.exists():
        raise typer.BadParameter(f"{blind_agreement_path} is absent.")
    blind_agreement = json.loads(blind_agreement_path.read_text(encoding="utf-8"))

    expected = set(manifest.get("audit_ids") or ())
    n_unresolved = len(expected - set(human_blind))

    # ------------------------------------------------------ model judges vs humans --
    model_labels = {
        str(r["audit_id"]): str(r["answer_attempt"])
        for r in _read_jsonl(Path(model_consensus))
        if r.get("answer_attempt") in ANSWER_ATTEMPT_LABELS
    }
    model_block = _label_metrics(human_blind, model_labels, classes=ANSWER_ATTEMPT_LABELS)
    missing_model = sorted(set(human_blind) - set(model_labels))
    if missing_model:
        provenance_failures.append(
            f"{len(missing_model)} human-labelled rows have no model consensus label"
        )

    # ------------------------------------------------ the frozen detector vs humans --
    detector_block: dict = {"ran": False, "why": "--detector-predictions was not passed"}
    if detector_predictions is not None:
        predicted = {
            str(r["audit_id"]): str(r.get("predicted_label") or r.get("answer_attempt"))
            for r in _read_jsonl(Path(detector_predictions))
        }
        predicted = {k: v for k, v in predicted.items() if v in ANSWER_ATTEMPT_LABELS}
        detector_block = {
            "ran": True,
            "file": str(detector_predictions),
            "sha256": hashlib.sha256(Path(detector_predictions).read_bytes()).hexdigest(),
            **_label_metrics(human_blind, predicted, classes=ANSWER_ATTEMPT_LABELS),
            "gated": False,
            "why_not_gated": (
                "the detector's gate is the engineering held-out partition, at its frozen "
                "thresholds. This block says whether the frozen detector and a person read "
                "the same 250 rows the same way, which is a different and smaller claim."
            ),
        }
    frozen_block: dict = {"bound": False}
    if frozen_detector is not None:
        payload = json.loads(Path(frozen_detector).read_text(encoding="utf-8"))
        frozen_block = {
            "bound": True,
            "file": str(frozen_detector),
            "sha256": hashlib.sha256(Path(frozen_detector).read_bytes()).hexdigest(),
            "frozen_detector_sha256": payload.get("frozen_detector_sha256"),
            "tau_answer": payload.get("tau_answer"),
            "tau_partial": payload.get("tau_partial"),
            "human_validated_at_read_time": payload.get("human_validated"),
        }

    # ----------------------------------------------------------- the reference axis --
    reference_block: dict = {"ran": False}
    if human_reference:
        scored = sorted(set(human_blind) & set(human_reference))
        reference_block = {
            "ran": True,
            "n_rows": len(scored),
            "distribution": dict(sorted(Counter(human_reference.values()).items())),
            "attempt_vs_content": dict(
                sorted(Counter(f"{human_blind[i]}|{human_reference[i]}" for i in scored).items())
            ),
            "gated": False,
            "why_not_gated": (
                "measured with the reference answer visible, and never trained on. It says "
                "how often an attempt actually conveyed the answer, which is a property of "
                "the candidate text rather than of the detector."
            ),
        }

    measured: dict[str, float | None] = {
        "human_human_kappa": blind_agreement.get("kappa"),
        "model_consensus_macro_f1": model_block["macro_f1"],
        "model_consensus_recall_on_human_answer": model_block["per_class"]["ANSWER"]["recall"],
        "recall_NONE": model_block["per_class"]["NONE"]["recall"],
        "recall_PARTIAL": model_block["per_class"]["PARTIAL"]["recall"],
        "recall_ANSWER": model_block["per_class"]["ANSWER"]["recall"],
        "n_unresolved": float(n_unresolved),
        "n_provenance_failures": float(len(provenance_failures)),
    }
    verdicts, failures = evaluate_panel_gates(measured, HUMAN_GATES)

    report = {
        "schema": HUMAN_REPORT_SCHEMA,
        "protocol": V4_4_PROTOCOL,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reportable": bool(manifest.get("reportable")),
        "n_rows_sampled": manifest.get("n_rows"),
        "n_by_source": manifest.get("n_by_source"),
        "raters": manifest.get("raters"),
        "human_human": {
            "kappa": blind_agreement.get("kappa"),
            "raw_agreement": blind_agreement.get("raw_agreement"),
            "confusion": blind_agreement.get("confusion"),
            "rater_file_sha256": blind_agreement.get("rater_file_sha256"),
        },
        "model_consensus_vs_human": {
            "file": str(model_consensus),
            "sha256": hashlib.sha256(Path(model_consensus).read_bytes()).hexdigest(),
            **model_block,
        },
        "frozen_detector_vs_human": detector_block,
        "frozen_detector": frozen_block,
        "reference_axis": reference_block,
        "gates": verdicts,
        "failures": failures,
        "provenance_failures": provenance_failures,
        "passed": not failures,
        "human_grounded": True,
        "authorises": (
            "unsealing the final detector bank, IF passed. Nothing else in the protocol "
            "may open it."
            if not failures
            else "nothing. The final bank stays sealed."
        ),
        "why_the_bound_is_not_moved": (
            "the human gate is the only step that can say the model-judge labels mean what "
            "the protocol claims. A bound relaxed after seeing the result would make the "
            "whole label chain self-certifying."
        ),
    }
    atomic_json(out / HUMAN_REPORT_FILENAME, report)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / HUMAN_REPORT_FILENAME),
                "passed": report["passed"],
                "human_human_kappa": measured["human_human_kappa"],
                "model_consensus_macro_f1": measured["model_consensus_macro_f1"],
                "failures": failures,
                "provenance_failures": provenance_failures,
            }
        )
    )
