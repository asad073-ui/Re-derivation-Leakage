"""``rdl graph-detector-v4-4-local-judge`` and ``-env-check`` -- the v4.4 judging pass.

Same models, same pins, same one-model-per-process discipline, same durable resume as v4.3.
Two things differ, and both are the point of v4.4:

**The prompt.** :func:`~rdl.eval.detector_v4_4.blind_prompt_v4_4` asks for six fields where
v4.3 asked for four, under a rubric that decomposes the PARTIAL/ANSWER judgement instead of
asking for it directly. A label produced under it is a different annotation, so the
``PROMPT_VERSION`` is different, the filenames are different, and nothing in this module can
combine a v4.3 label with a v4.4 one.

**The parser.** An object whose ``answer_attempt`` disagrees with its own
``addresses_question`` and ``standalone_answer`` is *malformed*, not repaired. Deriving the
label from the two decisions and overwriting what the judge wrote would convert "this judge
was confused" into a clean label, and the gate that requires zero malformed rows would then
report zero while certifying guesses.

The model-loading machinery is imported from the v4.3 runner rather than copied. It resolves
the auto class from the checkpoint's own config, verifies that 8-bit and 4-bit actually took
effect, refuses CPU offload and measures prompt truncation -- all of which is correct, is
tested, and would drift if a second copy existed. The v4.3 module is imported and never
modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import typer

from ..eval.detector_v4_3_judges import (
    LOCAL_JUDGE_ROSTER,
    JudgeRunRecord,
    MalformedJudgement,
    response_hashes,
)
from ..eval.detector_v4_4 import (
    BLIND_FIELDS,
    PROMPT_VERSION,
    REFERENCE_VALUES,
    V4_4_PROTOCOL,
    InconsistentHierarchy,
    blind_prompt_v4_4,
    parse_hierarchical_json,
    reference_prompt_v4_4,
)
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_local_judge import MAX_ATTEMPTS, PASSES, _build_generator, _read_jsonl
from .detector_v4_3_pins import JUDGE_PINS_FILENAME, load_judge_pins
from .detector_v4_3_store import DEFAULT_OUT_DIR
from .detector_v4_4_bundle import BUNDLE_FILENAME, DEFAULT_V4_4_DIR

__all__ = [
    "detector_v4_4_env_check",
    "detector_v4_4_local_judge",
    "fake_hierarchical_judgement",
    "output_names_v4_4",
    "run_rows_v4_4",
]

LOCAL_JUDGE_SCHEMA = "graph-detector-v4-4-local-judgement-v1"


def output_names_v4_4(
    *, judge: str, pass_name: str, run_id: str, reportable: bool
) -> dict[str, str]:
    """Filenames for one v4.4 pass. Namespaced away from v4.3's and from each other.

    A non-reportable run cannot occupy a reportable name, and a v4.4 pass cannot occupy a
    v4.3 name -- the two are different annotations of the same rows and a directory listing
    that mixed them would invite exactly the pooling the prompt version exists to forbid.
    """
    stem = (
        f"V4_4_LOCAL_JUDGE_{judge}_{pass_name.upper()}"
        if reportable
        else f"V4_4_SMOKE_{run_id}_JUDGE_{judge}_{pass_name.upper()}"
    )
    return {
        "output": f"{stem}.jsonl",
        "partial": f"{stem}.partial.jsonl",
        "run": f"{stem}_RUN.json",
    }


def fake_hierarchical_judgement(prompt: str) -> str:
    """A deterministic stand-in, so the harness is testable without weights.

    Deliberately not always well-formed. One row in seventeen returns prose and one in
    thirteen returns a *consistent-looking but internally contradictory* object -- an
    ``addresses_question=no`` with ``answer_attempt=ANSWER``. The second case is the one
    v4.4 adds, and an ordinary ``--fake-model`` run has to exercise it, because a
    consistency check that only a unit test ever reaches is a consistency check that can be
    silently bypassed by a refactor of the runner.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if int(digest[:2], 16) % 17 == 0:
        return "I think this one is hard to call, honestly."
    if "<reference_answer>" in prompt:
        return json.dumps(
            {"reference_content": ("YES", "NO", "UNCERTAIN")[int(digest[2:4], 16) % 3]}
        )
    if int(digest[4:6], 16) % 13 == 0:
        return json.dumps(
            {
                "addresses_question": "no",
                "standalone_answer": "no",
                "answer_attempt": "ANSWER",
                "question_type": "slot",
                "subject_only": "no",
                "refusal": "no",
            }
        )
    addresses, standalone = (
        ("no", "no"),
        ("yes", "no"),
        ("yes", "yes"),
    )[int(digest[6:8], 16) % 3]
    return json.dumps(
        {
            "addresses_question": addresses,
            "standalone_answer": standalone,
            "answer_attempt": (
                "NONE" if addresses == "no" else ("ANSWER" if standalone == "yes" else "PARTIAL")
            ),
            "question_type": ("slot", "yes-no", "open-ended")[int(digest[8:10], 16) % 3],
            "subject_only": (
                "yes" if addresses == "no" and int(digest[10:12], 16) % 4 == 0 else "no"
            ),
            "refusal": "yes" if addresses == "no" and int(digest[12:14], 16) % 3 == 0 else "no",
        }
    )


def run_rows_v4_4(
    rows: Sequence[Mapping],
    generate,
    *,
    partial_path: Path,
    role: str,
    pass_name: str = "blind",
    reference_of: Mapping[str, str] | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    measure_tokens=None,
) -> dict:
    """Judge ``rows`` under the v4.4 prompt, appending each result durably.

    Structurally the v4.3 runner with two substitutions -- the prompt builders and the
    parser -- and one extra counter. ``n_inconsistent`` is tracked separately from
    ``n_malformed`` because they diagnose different failures: unparseable output means the
    model ignored the format, while a contradictory object means it followed the format and
    could not hold the rubric's three fields together. The first is fixed by tightening the
    schema block; the second means the hierarchy itself is unclear, which is the failure
    v4.4 exists to detect. Both are refused; only one of them would be fixed by shouting
    "JSON" louder.
    """
    expected = REFERENCE_VALUES if pass_name == "reference" else None
    done = {str(r["audit_id"]) for r in _read_jsonl(partial_path)}
    counts = {
        "n_judged": 0,
        "n_skipped": len(done),
        "n_malformed": 0,
        "n_inconsistent": 0,
        "n_retries": 0,
        "n_truncated": 0,
    }

    for row in rows:
        audit_id = str(row["audit_id"])
        if audit_id in done:
            continue
        if pass_name == "reference":
            prompt = reference_prompt_v4_4(
                conditioning_question=str(row["conditioning_question"]),
                candidate_text=str(row["candidate_text"]),
                reference_answer=str((reference_of or {}).get(audit_id, "")),
            )
        else:
            prompt = blind_prompt_v4_4(
                conditioning_question=str(row["conditioning_question"]),
                subject_aliases=list(row.get("subject_aliases", ())),
                candidate_text=str(row["candidate_text"]),
            )

        truncated = False
        n_tokens = None
        if measure_tokens is not None:
            n_tokens, limit = measure_tokens(prompt)
            truncated = bool(limit) and n_tokens > limit
            counts["n_truncated"] += int(truncated)

        judgement: dict | None = None
        last_error = ""
        inconsistent = False
        raw = ""
        for attempt in range(max_attempts):
            raw = generate(prompt)
            try:
                judgement = parse_hierarchical_json(raw, expected=expected)
                break
            except InconsistentHierarchy as error:
                last_error = str(error)
                inconsistent = True
                counts["n_retries"] += int(attempt + 1 < max_attempts)
            except MalformedJudgement as error:
                last_error = str(error)
                inconsistent = False
                counts["n_retries"] += int(attempt + 1 < max_attempts)

        record = {
            "audit_id": audit_id,
            "judge": role,
            "pass": pass_name,
            "prompt_version": PROMPT_VERSION,
            "protocol": V4_4_PROTOCOL,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_tokens": n_tokens,
            "prompt_truncated": truncated,
            **response_hashes(raw),
        }
        if judgement is None:
            counts["n_malformed"] += 1
            counts["n_inconsistent"] += int(inconsistent)
            record.update(
                {
                    **dict.fromkeys(expected or BLIND_FIELDS),
                    "malformed": True,
                    "inconsistent_hierarchy": inconsistent,
                    "error": last_error,
                }
            )
        else:
            counts["n_judged"] += 1
            record.update({**judgement, "malformed": False, "inconsistent_hierarchy": False})
        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(dumps_canonical(record) + "\n")

    return counts


def _load_input(source: Path, limit: int) -> tuple[list[dict], dict]:
    """Rows plus provenance, from a bundle JSON or a JSONL fixture."""
    path = Path(source)
    if not path.exists():
        raise typer.BadParameter(f"{path} is absent")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        provenance: dict = {"kind": "jsonl", "path": str(path)}
    else:
        payload = json.loads(text)
        rows = list(payload.get("pairs", ()))
        provenance = {
            "kind": "bundle",
            "path": str(path),
            "bundle_sha256": payload.get("bundle_sha256"),
            "schema": payload.get("schema"),
        }
    provenance["file_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    provenance["n_rows_in_file"] = len(rows)
    if limit:
        rows = rows[:limit]
    provenance["n_rows_judged"] = len(rows)
    return rows, provenance


def detector_v4_4_local_judge(
    judge: str = typer.Option(..., "--judge", help="A (Qwen) or B (Mistral)."),
    pass_name: str = typer.Option("blind", "--pass", help="blind or reference."),
    input_path: Path = typer.Option(
        None, "--input", help="a JSONL fixture. Overrides --bundle when given."
    ),
    bundle: Path = typer.Option(DEFAULT_V4_4_DIR / BUNDLE_FILENAME, "--bundle"),
    eval_key: Path = typer.Option(
        None,
        "--eval-key",
        help="PROTECTED_STORE_EVAL_KEY.json. Required for --pass reference, refused for blind.",
    ),
    pins: Path = typer.Option(DEFAULT_OUT_DIR / JUDGE_PINS_FILENAME, "--pins"),
    out_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--out-dir"),
    device: str = typer.Option("cuda:0", "--device"),
    limit: int = typer.Option(0, "--limit", help="judge at most N rows. 0 means all."),
    run_id: str = typer.Option("", "--run-id", help="names a non-reportable run."),
    non_reportable: bool = typer.Option(
        False,
        "--non-reportable",
        help="a smoke run. Namespaces every output file so it cannot occupy a reportable name.",
    ),
    fake_model: bool = typer.Option(False, "--fake-model", help="load nothing; CPU harness test."),
    close: bool = typer.Option(False, "--close", help="freeze the pass. Refuses if incomplete."),
) -> None:
    """Run one v4.4 judging pass with one local model."""
    if judge not in LOCAL_JUDGE_ROSTER:
        raise typer.BadParameter(f"--judge must be one of {sorted(LOCAL_JUDGE_ROSTER)}")
    if pass_name not in PASSES:
        raise typer.BadParameter(f"--pass must be one of {list(PASSES)}")
    if limit and not non_reportable:
        raise typer.BadParameter(
            "--limit without --non-reportable would write the REPORTABLE filenames with a "
            "partial pass in them. Pass --non-reportable --run-id <name> for a smoke, or "
            "drop --limit for the real run."
        )
    if non_reportable and not run_id:
        raise typer.BadParameter("--non-reportable needs --run-id to name its files")

    out = Path(out_dir)
    names = output_names_v4_4(
        judge=judge, pass_name=pass_name, run_id=run_id, reportable=not non_reportable
    )
    partial_path = out / names["partial"]
    output_path = out / names["output"]

    source = input_path if input_path is not None else bundle
    rows, provenance = _load_input(source, limit)

    if close:
        return _close(partial_path, output_path, rows=rows, judge=judge, pass_name=pass_name)

    # ------------------------------------------------- pass separation, enforced --
    reference_of: dict[str, str] = {}
    if pass_name == "reference":
        if eval_key is None:
            raise typer.BadParameter("--pass reference needs --eval-key")
        blind_names = output_names_v4_4(
            judge=judge, pass_name="blind", run_id=run_id, reportable=not non_reportable
        )
        if not (out / blind_names["output"]).exists():
            raise typer.BadParameter(
                f"{out / blind_names['output']} does not exist, so judge {judge}'s v4.4 "
                "blind pass has not been closed. The reference pass runs only after the "
                "blind labels are frozen: a reference label produced first could have been "
                "revised in the light of the answer, and the two passes would no longer be "
                "independent annotations."
            )
        key = json.loads(Path(eval_key).read_text(encoding="utf-8"))
        reference_of = {
            audit_id: str(entry.get("reference_answer", ""))
            for audit_id, entry in key.get("rows", {}).items()
        }
        # Supplement rows have no reference answer: their questions are the audit's, but the
        # candidate was generated for this protocol and there is no "right answer" it was
        # supposed to convey. They resolve to UNCERTAIN by the reference rubric's own rule,
        # which is the correct outcome and is why the rule exists.
    elif eval_key is not None:
        raise typer.BadParameter(
            "--eval-key is refused for --pass blind. The blind pass must not be able to "
            "reach a reference answer, and the way that is guaranteed is that the file is "
            "never opened."
        )

    pin = LOCAL_JUDGE_ROSTER[judge] if fake_model else load_judge_pins(pins, judge)
    generate, measure_tokens, described = _build_generator(pin, fake=fake_model, device=device)
    if fake_model:
        generate = fake_hierarchical_judgement

    started = time.time()
    counts = run_rows_v4_4(
        rows,
        generate,
        partial_path=partial_path,
        role=judge,
        pass_name=pass_name,
        reference_of=reference_of,
        measure_tokens=measure_tokens,
    )

    peak = 0
    if described.get("loaded"):
        import torch

        peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0

    record = JudgeRunRecord(
        role=judge,
        repo_id=pin.repo_id,
        revision=pin.revision,
        quantization=pin.quantization,
        device=str(described.get("device", "")),
        dtype=str(described.get("dtype", "")),
        peak_vram_bytes=peak,
        n_rows=len(rows),
        n_malformed=counts["n_malformed"],
        n_retries=counts["n_retries"],
        n_truncated=counts["n_truncated"],
        seconds=round(time.time() - started, 1),
        extra={
            "schema": LOCAL_JUDGE_SCHEMA,
            "prompt_version": PROMPT_VERSION,
            "protocol": V4_4_PROTOCOL,
            "pass": pass_name,
            "run_id": run_id or None,
            "reportable": not (fake_model or non_reportable),
            "source": provenance,
            "partial_file": str(partial_path),
            "fake_model": bool(fake_model),
            "pins_file": None if fake_model else str(pins),
            "counts": counts,
            "n_inconsistent_hierarchy": counts["n_inconsistent"],
            "pin": pin.to_dict(),
            "generator": described,
        },
    )
    payload = record.to_dict()
    payload["schema"] = LOCAL_JUDGE_SCHEMA
    payload["prompt_version"] = PROMPT_VERSION
    atomic_json(out / names["run"], payload)
    typer.echo(dumps_canonical(payload))
    return None


def _close(
    partial_path: Path,
    output_path: Path,
    *,
    rows: Sequence[Mapping],
    judge: str,
    pass_name: str = "blind",
) -> None:
    """Freeze a complete pass. Refuses an incomplete, malformed or truncated one."""
    judged = _read_jsonl(partial_path)
    by_id = {str(r["audit_id"]): r for r in judged}
    missing = [str(r["audit_id"]) for r in rows if str(r["audit_id"]) not in by_id]
    malformed = sorted(k for k, v in by_id.items() if v.get("malformed"))
    inconsistent = sorted(k for k, v in by_id.items() if v.get("inconsistent_hierarchy"))
    truncated = sorted(k for k, v in by_id.items() if v.get("prompt_truncated"))
    if missing or malformed or truncated:
        raise typer.BadParameter(
            f"cannot close judge {judge} {pass_name}: {len(missing)} unjudged, "
            f"{len(malformed)} malformed (of which {len(inconsistent)} were internally "
            f"contradictory rather than unparseable), {len(truncated)} truncated. The label "
            "gate requires zero of each, and a closed file that quietly omitted them would "
            f"satisfy the gate by having fewer rows. First missing {missing[:3]}, first "
            f"malformed {malformed[:3]}, first truncated {truncated[:3]}."
        )
    ordered = [by_id[str(r["audit_id"])] for r in rows]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(dumps_canonical(r) + "\n" for r in ordered), encoding="utf-8")
    typer.echo(
        dumps_canonical(
            {
                "closed": str(output_path),
                "judge": judge,
                "pass": pass_name,
                "prompt_version": PROMPT_VERSION,
                "n_rows": len(ordered),
                "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
                "n_malformed": 0,
                "n_inconsistent_hierarchy": 0,
                "n_truncated": 0,
            }
        )
    )


# =====================================================================================
# env-check
# =====================================================================================


def _last_json_object(text: str) -> dict:
    """The last line of ``text`` that parses as a JSON object, or ``{}``.

    Needed because the captured stream is not pure: bitsandbytes logs a GPU-support warning
    to stdout on import, and torch adds a CUDA warning, so the nested command's one-line
    report is preceded by prose. Scanning backwards finds the report without depending on
    how many lines of third-party noise happened to precede it today.
    """
    for line in reversed((text or "").splitlines()):
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            loaded = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def detector_v4_4_env_check(
    pins: Path = typer.Option(DEFAULT_OUT_DIR / JUDGE_PINS_FILENAME, "--pins"),
    v4_4_dir: Path = typer.Option(DEFAULT_V4_4_DIR, "--v4-4-dir"),
    strict: bool = typer.Option(
        False, "--strict", help="exit non-zero on any failure. Use before a reportable phase."
    ),
) -> None:
    """Can this box run the reportable v4.4 GPU phase?

    The v4.3 hardware and library checks, plus the four v4.4 artifacts that must exist and
    agree before a single token is generated: the bundle, the panel, the smoke fixture, and
    a panel whose ``bundle_sha256`` matches the bundle actually on disk.

    That last check is the one worth having. A panel frozen against an older bundle names
    row ids that may no longer exist or may now hold different text, and the kappa gate
    would then be computed over whatever subset happened to join -- silently, and with a
    denominator nobody chose.
    """
    from .detector_v4_3_pins import detector_v4_3_env_check as _v4_3_env_check

    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    # The hardware/library half is v4.3's, run for its side effects on stdout suppressed:
    # the same box runs both phases and duplicating those checks would let them drift.
    import contextlib
    import io

    buffer = io.StringIO()
    v4_3_ok = True
    try:
        with contextlib.redirect_stdout(buffer):
            _v4_3_env_check(pins=pins, strict=False)
    except SystemExit:  # pragma: no cover - typer.Exit from a nested call
        v4_3_ok = False
    v4_3_report = _last_json_object(buffer.getvalue())
    checks.extend(v4_3_report.get("checks", []))
    record(
        "v4_3_environment_checks",
        v4_3_ok and not v4_3_report.get("n_failures", 1),
        f"{v4_3_report.get('n_failures', 'unknown')} failures from the shared hardware "
        "and library checks",
    )

    root = Path(v4_4_dir)
    bundle_path = root / BUNDLE_FILENAME
    panel_path = root / "DETECTOR_V4_4_CALIBRATION_PANEL.json"
    smoke_path = root / "V4_4_JUDGE_SMOKE_60.jsonl"

    bundle_payload: dict = {}
    if bundle_path.exists():
        bundle_payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        record(
            "v4_4_bundle",
            bool(bundle_payload.get("pairs")),
            f"{bundle_payload.get('n_pairs')} pairs, sha {str(bundle_payload.get('bundle_sha256'))[:12]}",
        )
    else:
        record("v4_4_bundle", False, f"{bundle_path} is absent")

    if panel_path.exists():
        panel = json.loads(panel_path.read_text(encoding="utf-8"))
        composition = panel.get("composition", {}).get("by_intended_class", {})
        balanced = len(set(composition.values())) == 1 and len(composition) == 3
        record(
            "v4_4_panel_balanced",
            balanced,
            f"{composition} (must be one size for each of NONE/PARTIAL/ANSWER)",
        )
        record(
            "v4_4_panel_matches_bundle",
            bool(bundle_payload)
            and panel.get("bundle_sha256") == bundle_payload.get("bundle_sha256"),
            "the panel was frozen against the bundle now on disk",
        )
    else:
        record("v4_4_panel_balanced", False, f"{panel_path} is absent")
        record("v4_4_panel_matches_bundle", False, f"{panel_path} is absent")

    n_smoke = len(_read_jsonl(smoke_path))
    record("v4_4_smoke_fixture", n_smoke == 60, f"{n_smoke} rows (need 60)")

    record("v4_4_prompt_version", True, PROMPT_VERSION)

    # ------------------------------------------------------------- the readiness half --
    # The frozen judge-pin artifact records `present: false` for the encoder pins, because
    # that was true when it was generated. It is NOT rewritten to say otherwise: editing a
    # frozen observation to match today makes the artifact useless as a record of what was
    # known then. Instead the model-pin file is verified HERE, and its hash recorded.
    from .detector_v4_2_llm_judge import DEFAULT_V4_2_OUT
    from .detector_v4_2_model_pins import MODEL_PINS_FILENAME

    model_pins_path = DEFAULT_V4_2_OUT / MODEL_PINS_FILENAME
    model_pins_sha = None
    if model_pins_path.exists():
        model_pins_sha = hashlib.sha256(model_pins_path.read_bytes()).hexdigest()
        record(
            "v4_4_encoder_pins_present",
            True,
            f"{model_pins_path.name} @ {model_pins_sha[:12]} (the judge-pin artifact's "
            "'present: false' is a frozen observation from before it existed, and is not "
            "rewritten)",
        )
    else:
        record("v4_4_encoder_pins_present", False, f"{model_pins_path} is absent")

    # `mistral-common` decides judge B's Tekken subword split, so it is part of the
    # annotator rather than part of the environment.
    try:
        from importlib.metadata import version as _version

        mistral_version = _version("mistral-common")
    except Exception:  # pragma: no cover - absent on a CPU box without gpu extras
        mistral_version = ""
    record(
        "v4_4_mistral_common_pinned",
        mistral_version == PINNED_MISTRAL_COMMON,
        f"installed {mistral_version or 'absent'}, protocol pins {PINNED_MISTRAL_COMMON}. "
        "A different Tekken split is a different prompt at a fixed budget, so this is an "
        "annotator version, not a library version.",
    )

    plan_path = root / "DETECTOR_V4_4_FRESH_AUDIT_PLAN.json"
    record(
        "v4_4_fresh_audit_plan",
        plan_path.exists(),
        (
            f"{plan_path.name} freezes both partitions' sizes BEFORE the bank exists"
            if plan_path.exists()
            else f"{plan_path} is absent; run graph-detector-v4-4-fresh-audit-plan on CPU"
        ),
    )

    failures = [c for c in checks if not c["ok"]]
    readiness = {
        "schema": "graph-detector-v4-4-readiness-v1",
        "protocol": V4_4_PROTOCOL,
        "prompt_version": PROMPT_VERSION,
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": _readiness_environment(mistral_version),
        "encoder_pins": {"file": str(model_pins_path), "sha256": model_pins_sha},
        "bundle_sha256": bundle_payload.get("bundle_sha256"),
        "checks": checks,
        "n_failures": len(failures),
        "ready_for_reportable_gpu_phase": not failures,
    }
    if root.exists():
        atomic_json(root / "DETECTOR_V4_4_READINESS.json", readiness)
    typer.echo(dumps_canonical(readiness))
    if strict and failures:
        raise typer.Exit(code=1)


PINNED_MISTRAL_COMMON = "1.11.7"


def _readiness_environment(mistral_version: str) -> dict:
    """Torch, CUDA, driver, GPU and free disk -- whatever this box can actually answer.

    Every field is optional and reported as ``None`` when unavailable rather than omitted,
    so a CPU run of this command produces the same KEYS as a GPU run and the two can be
    diffed. An absent key and a null key read very differently to somebody comparing a
    laptop's manifest against the instance that produced the labels.
    """
    import contextlib
    import shutil

    out: dict = {
        "python": sys.version.split()[0],
        "mistral_common": mistral_version or None,
        "torch": None,
        "cuda": None,
        "driver": None,
        "gpu_name": None,
        "vram_gib": None,
        "bf16_supported": None,
        "free_disk_gib": None,
    }
    with contextlib.suppress(OSError):  # pragma: no branch - defensive
        out["free_disk_gib"] = round(shutil.disk_usage(Path.cwd()).free / 2**30, 2)
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            out["gpu_name"] = torch.cuda.get_device_name(0)
            out["vram_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
            out["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    except Exception:  # pragma: no cover - torch is absent on the CPU gate
        pass
    return out
