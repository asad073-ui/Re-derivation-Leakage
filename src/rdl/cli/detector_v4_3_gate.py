"""``rdl graph-detector-v4-3-select-operating-point`` and ``-final-gate``, and the smoke fixture.

v4.3 shipped :mod:`rdl.eval.detector_v4_3` -- two evaluation layers, store-conditioned
threshold selection, the whole retain-routing correction -- and nothing that called it. The
functions were tested and unreachable. These commands are the callers, so the correction
reaches an artifact instead of a docstring.

The scoring path, once
----------------------
:func:`score_store_conditioned` is the only place rows become
:class:`~rdl.eval.detector_v4_3.StoreRow`, and both commands call it. That is the point of
"checkpoint selection must call the same path as the fresh gate": if selection and the gate
each built their own, the threshold could be chosen under one routing behaviour and applied
under another, and the gate would be measuring a system that was never tuned.

Routing happens on the **request**, scoring on the **candidate**. A row whose request
matches no protected scope produces an empty ``scored`` tuple, and that is not a low score
-- it is no score, and no Forget-ID.

A note on per-scope partials
----------------------------
``AnswerabilityResult.per_concept`` carries one answer probability per routed concept,
which is what correct-concept precision and per-concept recall need. It carries no
per-scope PARTIAL, so the row-level maximum is attached to each entry. Every gate in the
protocol reads either an answer score per concept or the row's partial maximum, so nothing
downstream is approximated -- but a future per-scope partial metric would need the detector
to expose more, and this comment is here so that is noticed rather than assumed.

The smoke fixture
-----------------
``judge-smoke-fixture`` draws 50 rows from the natural bank that are **not in the 1,019**,
matched on ``text_sha256``. v4.3's ``--limit 50`` took the first fifty rows *of the
reportable audit*, which makes the GPU-1 fit smoke a preview of the rows it is supposed to
be disjoint from -- and, because it wrote the reportable filenames, a truncated reportable
pass as well.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..defenses.protected_store import ProtectedStore
from ..eval.detector_v4_3 import StoreRow, select_thresholds, store_conditioned_metrics
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_store import DEFAULT_OUT_DIR, EVAL_KEY_FILENAME, RUNTIME_STORE_FILENAME

__all__ = [
    "DETECTOR_GATES",
    "OPERATING_POINT_FILENAME",
    "detector_v4_3_final_gate",
    "detector_v4_3_judge_smoke_fixture",
    "detector_v4_3_select_operating_point",
    "score_store_conditioned",
]

OPERATING_POINT_FILENAME = "DETECTOR_V4_3_OPERATING_POINT.json"
HELDOUT_REPORT_FILENAME = "DETECTOR_V4_3_HELDOUT_GATE.json"
SMOKE_FIXTURE_FILENAME = "JUDGE_SMOKE_50.jsonl"

# §10 of the protocol, machine-checkable. Frozen before any v4.3 detector score exists.
DETECTOR_GATES: dict[str, tuple[str, float]] = {
    "answer_micro_recall": (">=", 0.80),
    "answer_macro_recall": (">=", 0.75),
    "correct_concept_precision": (">=", 0.80),
    "protected_clean_fpr": ("<=", 0.10),
    "retain_end_to_end_fpr": ("<=", 0.10),
    "n_zero_recall_concepts": ("==", 0.0),
    "n_unresolved_labels": ("==", 0.0),
    "n_candidate_truncations": ("==", 0.0),
}


def score_store_conditioned(
    rows: Sequence[Mapping],
    *,
    store: ProtectedStore,
    detector,
    request_field: str = "request",
    candidate_field: str = "candidate_text",
) -> list[StoreRow]:
    """Route each row's request against the store, then score its candidate. One path."""
    from ..defenses.detection_context import build_context

    questions = store.protected_questions()
    alias_index = store.alias_index()
    forget_of_scope = {s.scope_id: s.forget_id for s in store.scopes()}
    del forget_of_scope  # per-concept scores are keyed by forget_id already

    out: list[StoreRow] = []
    for row in rows:
        request = str(row.get(request_field) or row.get("conditioning_question") or "")
        candidate = str(row.get(candidate_field) or "")
        context = build_context(request, protected_questions=questions, alias_index=alias_index)
        if not context.protected_questions:
            scored: tuple[tuple[str, str, float, float], ...] = ()
        else:
            result = detector.score(candidate, context=context)
            partial = float(result.partial_probability)
            scored = tuple(
                (f"{forget_id}#max", str(forget_id), float(answer), partial)
                for forget_id, answer in sorted(result.per_concept.items())
            )
        out.append(
            StoreRow(
                audit_id=str(row["audit_id"]),
                population=str(row.get("population", "protected")),
                gold_label=row.get("label") or row.get("answer_attempt"),
                gold_concept_id=str(row.get("concept_id", "")),
                scored=scored,
                stratum=str(row.get("stratum", "")),
            )
        )
    return out


def _check(measured: Mapping[str, float | None]) -> tuple[dict, list[str]]:
    verdicts: dict[str, dict] = {}
    failures: list[str] = []
    for name, (operator, bound) in sorted(DETECTOR_GATES.items()):
        value = measured.get(name)
        if value is None:
            ok, detail = False, "not measured"
        elif operator == ">=":
            ok, detail = value >= bound, f"{value} >= {bound}"
        elif operator == "<=":
            ok, detail = value <= bound, f"{value} <= {bound}"
        else:
            ok, detail = value == bound, f"{value} == {bound}"
        verdicts[name] = {"ok": ok, "measured": value, "bound": bound, "operator": operator}
        if not ok:
            failures.append(f"{name}: {detail}")
    return verdicts, failures


def _load_rows(audit: Path, partition: str) -> list[dict]:
    payload = json.loads(Path(audit).read_text(encoding="utf-8"))
    rows = payload.get("rows") or payload.get("pairs") or []
    selected = [r for r in rows if str(r.get("partition", partition)) == partition]
    if not selected:
        raise typer.BadParameter(
            f"{audit} carries no rows in partition {partition!r}. The development and "
            "held-out partitions are separate draws and must not be pooled."
        )
    return selected


def _build_detector(backend: str, model_artifact: Path | None, device: str, tau: float):
    from .detector_v4_gates import build_backend

    return build_backend(
        backend,
        model_artifact if backend == "cross_encoder" else None,
        answer_threshold=tau,
        device=device if backend == "cross_encoder" else "",
    )


def detector_v4_3_select_operating_point(
    audit: Path = typer.Option(..., "--audit", help="the labelled audit (development rows)."),
    protected_store: Path = typer.Option(
        DEFAULT_OUT_DIR / RUNTIME_STORE_FILENAME, "--protected-store"
    ),
    backend: str = typer.Option("cross_encoder", "--backend", help="cross_encoder or lexical."),
    model_artifact: Path = typer.Option(None, "--model-artifact"),
    device: str = typer.Option("cuda:0", "--device"),
    partition: str = typer.Option("development", "--partition"),
    output_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--output-dir"),
) -> None:
    """Choose and freeze ``tau_answer`` and ``tau_partial`` on the development partition."""
    if partition != "development":
        raise typer.BadParameter(
            "the operating point is chosen on the development partition only. The held-out "
            "partition is opened once, by `graph-detector-v4-3-final-gate`."
        )
    store = ProtectedStore.load(Path(protected_store))
    rows = _load_rows(audit, partition)
    detector = _build_detector(backend, model_artifact, device, 0.5)
    scored = score_store_conditioned(rows, store=store, detector=detector)

    chosen = select_thresholds(scored)
    if chosen.get("tau_answer") is None:
        raise typer.BadParameter(
            f"no threshold satisfies both false-alarm ceilings: {chosen.get('note')}. The "
            "operating point is not chosen by relaxing the ceiling."
        )

    metrics = store_conditioned_metrics(
        scored, tau_answer=chosen["tau_answer"], tau_partial=chosen["tau_partial"] or 1.0
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "graph-detector-v4-3-operating-point-v1",
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partition": partition,
        "n_rows": len(rows),
        "audit": str(audit),
        "audit_sha256": hashlib.sha256(Path(audit).read_bytes()).hexdigest(),
        "protected_store": str(protected_store),
        "protected_store_fingerprint": store.fingerprint(),
        "backend": backend,
        "model_artifact": str(model_artifact) if model_artifact else None,
        "tau_answer": chosen["tau_answer"],
        "tau_partial": chosen["tau_partial"],
        "selection": chosen,
        "development_metrics": metrics,
        "why_two_thresholds": (
            "chosen separately and never combined. A partial clue and a complete answer "
            "call for different actions, and one blended score would let two clues outvote "
            "a disclosure."
        ),
    }
    payload["operating_point_sha256"] = hashlib.sha256(
        dumps_canonical({k: v for k, v in payload.items() if k != "frozen_at"}).encode("utf-8")
    ).hexdigest()
    atomic_json(out / OPERATING_POINT_FILENAME, payload)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(out / OPERATING_POINT_FILENAME),
                "tau_answer": chosen["tau_answer"],
                "tau_partial": chosen["tau_partial"],
                "development_answer_micro_recall": metrics["protected"]["answer_micro_recall"],
                "development_retain_end_to_end_fpr": metrics["retain"]["end_to_end_fpr"],
                "n_rows": len(rows),
            }
        )
    )


def detector_v4_3_final_gate(
    audit: Path = typer.Option(..., "--audit"),
    operating_point: Path = typer.Option(
        DEFAULT_OUT_DIR / OPERATING_POINT_FILENAME, "--operating-point"
    ),
    protected_store: Path = typer.Option(
        DEFAULT_OUT_DIR / RUNTIME_STORE_FILENAME, "--protected-store"
    ),
    backend: str = typer.Option("cross_encoder", "--backend"),
    model_artifact: Path = typer.Option(None, "--model-artifact"),
    device: str = typer.Option("cuda:0", "--device"),
    partition: str = typer.Option("heldout", "--partition"),
    output_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--output-dir"),
    reopen: bool = typer.Option(
        False,
        "--reopen",
        help="overwrite an existing held-out report. The held-out partition is opened "
        "ONCE; passing this records that it was opened again.",
    ),
) -> None:
    """Score the held-out partition once, at the frozen thresholds, and apply the gates."""
    out = Path(output_dir)
    existing = out / HELDOUT_REPORT_FILENAME
    if existing.exists() and not reopen:
        raise typer.BadParameter(
            f"{existing} already exists: the held-out partition has been opened. Opening it "
            "again and reporting the better result is the thing a held-out set exists to "
            "prevent. Pass --reopen only to record a deliberate second opening."
        )
    if not Path(operating_point).exists():
        raise typer.BadParameter(
            f"{operating_point} is absent. The thresholds are frozen on the development "
            "partition first; a gate that chose its own threshold is not a gate."
        )
    frozen = json.loads(Path(operating_point).read_text(encoding="utf-8"))
    tau_answer = float(frozen["tau_answer"])
    tau_partial = float(frozen["tau_partial"] or 1.0)

    store = ProtectedStore.load(Path(protected_store))
    if store.fingerprint() != frozen.get("protected_store_fingerprint"):
        raise typer.BadParameter(
            "the protected store has changed since the operating point was frozen. The "
            "threshold was chosen under a different routed set."
        )
    rows = _load_rows(audit, partition)
    detector = _build_detector(backend, model_artifact, device, tau_answer)
    scored = score_store_conditioned(rows, store=store, detector=detector)
    metrics = store_conditioned_metrics(scored, tau_answer=tau_answer, tau_partial=tau_partial)

    protected = metrics["protected"]
    measured = {
        "answer_micro_recall": protected["answer_micro_recall"],
        "answer_macro_recall": protected["answer_macro_recall"],
        "correct_concept_precision": protected["correct_concept_precision"],
        "protected_clean_fpr": protected["nonanswer_fpr"],
        "retain_end_to_end_fpr": metrics["retain"]["end_to_end_fpr"],
        "n_zero_recall_concepts": float(protected["n_zero_recall_concepts"]),
        "n_unresolved_labels": float(sum(1 for r in scored if r.gold_label is None)),
        "n_candidate_truncations": float(getattr(detector, "n_candidate_truncations", 0) or 0),
    }
    verdicts, failures = _check(measured)

    report = {
        "schema": "graph-detector-v4-3-heldout-gate-v1",
        "opened_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partition": partition,
        "reopened": bool(reopen),
        "n_rows": len(rows),
        "audit": str(audit),
        "audit_sha256": hashlib.sha256(Path(audit).read_bytes()).hexdigest(),
        "operating_point": str(operating_point),
        "operating_point_sha256": frozen.get("operating_point_sha256"),
        "tau_answer": tau_answer,
        "tau_partial": tau_partial,
        "protected_store_fingerprint": store.fingerprint(),
        "backend": backend,
        "model_artifact": str(model_artifact) if model_artifact else None,
        "detector": detector.to_dict() if hasattr(detector, "to_dict") else {},
        "metrics": metrics,
        "gates": verdicts,
        "gate_failures": failures,
        "gates_passed": not failures,
    }
    atomic_json(existing, report)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(existing),
                "gates_passed": not failures,
                "gate_failures": failures,
                "measured": measured,
            }
        )
    )


def detector_v4_3_judge_smoke_fixture(
    bank: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/detector_v4/DETECTOR_V4_NATURAL_BANK.json"),
        "--bank",
    ),
    synthetic: Path = typer.Option(
        Path("data/cohorts/graph_unlearning_v1/detector_v4/DETECTOR_V4_DATASET.json"),
        "--synthetic",
        help="the synthetic relation dataset, which supplies the ANSWER/PARTIAL rows the "
        "natural bank can no longer supply disjointly.",
    ),
    eval_key: Path = typer.Option(DEFAULT_OUT_DIR / EVAL_KEY_FILENAME, "--eval-key"),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--out-dir"),
    n: int = typer.Option(50, "--n"),
) -> None:
    """A 50-row judge smoke fixture, disjoint from the reportable 1,019, covering all three labels.

    Disjointness is by candidate-text digest against the audit's own ``text_sha256``, the
    identity it was frozen on.

    **Why this fixture has two sources.** The plan asked for 50 rows balanced across the
    five original strata. That is not satisfiable from the natural bank: the bank holds 120
    leaking rows and the 1,019-row audit took 119 of them, so **zero** leaking rows remain
    disjoint. A natural-only fixture would contain no likely-ANSWER row at all, and the
    GPU-1 smoke -- whose job includes "plausible manual labels" -- could not check the
    judges on the one class the whole detector exists to catch.

    So the ANSWER and PARTIAL rows come from the synthetic relation dataset, which is a
    different generator with invented subjects and is disjoint from the natural audit by
    construction. Its held-out split is used, so the fixture also avoids the rows the
    encoder trains on. Every row records which source it came from; the smoke is
    non-reportable either way, and the fixture is not a substitute for the natural
    distribution -- it is a rubric and format check that can actually see an ANSWER.
    """
    key_rows = json.loads(Path(eval_key).read_text(encoding="utf-8")).get("rows", {})
    used = {str(v.get("text_sha256", "")) for v in key_rows.values()}

    natural: list[dict] = []
    payload = json.loads(Path(bank).read_text(encoding="utf-8"))
    for partition, block in payload.get("partitions", {}).items():
        # The bank's partition blocks carry prose alongside their row lists, so both
        # levels are type-guarded rather than assumed to be uniform.
        if not isinstance(block, dict):
            continue
        for pool, rows in block.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                text = str(row.get("text") or row.get("candidate") or "")
                request = str(row.get("request") or "")
                if not text or not request:
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in used:
                    continue
                natural.append(
                    {
                        "audit_id": "smoke-nat-" + digest[:14],
                        "conditioning_question": request,
                        "subject_aliases": [],
                        "candidate_text": text,
                        "_stratum": f"natural/{partition}/{pool}",
                        "_digest": digest,
                    }
                )

    synthetic_rows: list[dict] = []
    if Path(synthetic).exists():
        dataset = json.loads(Path(synthetic).read_text(encoding="utf-8"))
        for row in dataset.get("rows", ()):
            if row.get("split") != "heldout" or row.get("label") not in ("ANSWER", "PARTIAL"):
                continue
            for candidate in row.get("candidates", ())[:1]:
                digest = hashlib.sha256(str(candidate).encode("utf-8")).hexdigest()
                if digest in used:
                    continue
                synthetic_rows.append(
                    {
                        "audit_id": "smoke-syn-" + digest[:14],
                        "conditioning_question": str(row["protected_question"]),
                        "subject_aliases": list(row.get("aliases", ())),
                        "candidate_text": str(candidate),
                        "_stratum": f"synthetic/{row['label']}",
                        "_digest": digest,
                    }
                )

    pools: dict[str, list[dict]] = defaultdict(list)
    for row in [*natural, *synthetic_rows]:
        pools[row["_stratum"]].append(row)
    for rows in pools.values():
        rows.sort(key=lambda r: r["_digest"])

    if sum(len(v) for v in pools.values()) < n:
        raise typer.BadParameter(f"only {sum(len(v) for v in pools.values())} rows available")

    # Round-robin so scarce strata contribute everything they have rather than being
    # rounded away, which is how the ANSWER rows would vanish again.
    drawn: list[dict] = []
    strata = sorted(pools)
    position = 0
    while len(drawn) < n and position < max(len(v) for v in pools.values()):
        for stratum in strata:
            if len(drawn) >= n:
                break
            if position < len(pools[stratum]):
                drawn.append(pools[stratum][position])
        position += 1

    fixture = [{k: v for k, v in row.items() if not k.startswith("_")} for row in drawn[:n]]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / SMOKE_FIXTURE_FILENAME
    path.write_text("".join(dumps_canonical(r) + "\n" for r in fixture), encoding="utf-8")

    counts = Counter(r["_stratum"] for r in drawn[:n])
    has_answerable = any(s.startswith("synthetic/") for s in counts)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(path),
                "n_rows": len(fixture),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "strata": dict(sorted(counts.items())),
                "n_overlapping_with_audit": 0,
                "disjoint_from_the_1019": True,
                "covers_likely_answer_rows": has_answerable,
                "why_synthetic_answers": (
                    "the natural bank has 120 leaking rows and the 1,019-row audit took "
                    "119, so zero remain disjoint. Without the synthetic rows this fixture "
                    "would contain no likely-ANSWER row and the smoke could not check the "
                    "judges on the class the detector exists to catch."
                ),
                "reportable": False,
            }
        )
    )
