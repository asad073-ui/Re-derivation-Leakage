"""A CPU dry run of the whole v4.3 GPU pipeline, with no weights and no network.

Every GPU step has a CPU stand-in here: the judges are a deterministic hash, the encoder is
the lexical backend, and the fresh bank is a small synthetic audit. Nothing it writes is
reportable and it says so in every artifact.

The point is not the numbers -- a hash-based judge agrees with itself by luck, so the label
gates fail, which is correct and is part of what this checks. The point is that the
*sequence* runs: pins refuse a tag, blind closes, the reference pass refuses to start
before the blind freeze, the report computes kappa and gates it, the labelled bundle
rebuilds, the operating point is chosen on development only, and the held-out gate refuses
a second opening. Those are the failures that otherwise surface on a rented box.

Run: ``python scripts/v43_pipeline_dryrun.py --work /tmp/v43dry``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
V4_3 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_3"

sys.path.insert(0, str(SRC))


def _run(*args: str) -> tuple[int, str]:
    process = subprocess.run(
        [sys.executable, "-m", "rdl.cli", *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
    )
    return process.returncode, (process.stdout or "") + (process.stderr or "")


def _says(output: str, phrase: str) -> bool:
    """Substring match that survives Typer's Rich panel, which hard-wraps error text.

    Without this, asserting on a refusal message tests the terminal width rather than the
    refusal -- a phrase that happens to straddle a wrap point reads as a missing message.
    """
    flat = " ".join(output.replace("|", " ").split())
    return " ".join(phrase.split()) in flat


def _clean_judgement(prompt: str, role: str = "A") -> str:
    """A judge that always emits valid JSON, so the pipeline can be driven past ``--close``.

    ``role`` matters. Two judges that are the same deterministic function of the prompt
    agree on every row, and a dry run built that way reports "0 disagreements" and never
    exercises the disagreement file, the adjudication input, or the unresolved-row gate --
    the parts most likely to be wrong. Judge B therefore dissents on roughly one row in
    seven, which is the same order as a real kappa around 0.8.

    It is not a model and its labels mean nothing.
    """
    if "<reference_answer>" in prompt:
        return json.dumps({"reference_content": "YES" if "born" in prompt else "NO"})
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    labels = ("NONE", "PARTIAL", "ANSWER")
    index = int(digest[:2], 16) % 3
    if role == "B" and int(digest[4:6], 16) % 7 == 0:
        index = (index + 1) % 3
    return json.dumps(
        {
            "answer_attempt": labels[index],
            "subject_only": "no",
            "refusal": "no",
            "question_type": "slot",
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/tmp/v43dry"))
    args = parser.parse_args()

    from rdl.cli.detector_v4_3_local_judge import _close, output_names, run_rows

    work = args.work
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for name in (
        "DETECTOR_V4_3_PAIR_BUNDLE.json",
        "PROTECTED_STORE_EVAL_KEY.json",
        "PROTECTED_STORE_RUNTIME.json",
        "DETECTOR_V4_3_CONDITIONING_INDEX.json",
    ):
        shutil.copy(V4_3 / name, work / name)

    bundle = work / "DETECTOR_V4_3_PAIR_BUNDLE.json"
    rows = json.loads(bundle.read_text(encoding="utf-8"))["pairs"]
    key = json.loads((work / "PROTECTED_STORE_EVAL_KEY.json").read_text(encoding="utf-8"))
    reference_of = {k: str(v.get("reference_answer", "")) for k, v in key["rows"].items()}

    steps: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        steps.append((name, ok, detail))
        print(f"[{'ok ' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")

    # ---------------------------------------------------------------- 1. pins --
    code, out = _run(
        "graph-detector-v4-3-freeze-judge-pins",
        "--output-dir",
        str(work),
        "--judge-a-revision",
        "main",
        "--judge-b-revision",
        "b" * 40,
    )
    check("a tag is refused as a pin", code != 0 and _says(out, "not a 40-character commit sha"))

    code, out = _run(
        "graph-detector-v4-3-freeze-judge-pins",
        "--output-dir",
        str(work),
        "--judge-a-revision",
        "a" * 40,
        "--judge-b-revision",
        "b" * 40,
    )
    check(
        "judge pins freeze", code == 0 and (work / "DETECTOR_V4_3_LOCAL_JUDGE_PINS.json").exists()
    )

    # ------------------------------------------------------- 2. blind passes --
    for role in ("A", "B"):
        names = output_names(judge=role, pass_name="blind", run_id="", reportable=True)
        counts = run_rows(
            rows,
            lambda prompt, role=role: _clean_judgement(prompt, role),
            partial_path=work / names["partial"],
            role=role,
            pass_name="blind",
        )
        _close(work / names["partial"], work / names["output"], rows=rows, judge=role)
        # The run manifest a real pass would write; the report checks it for provenance.
        (work / names["run"]).write_text(
            json.dumps(
                {
                    "reportable": True,
                    "bundle_sha256": json.loads(bundle.read_text(encoding="utf-8"))[
                        "bundle_sha256"
                    ],
                }
            ),
            encoding="utf-8",
        )
        check(f"judge {role} blind pass closes", counts["n_malformed"] == 0)

    # ------------------------------------ 3. reference refuses before the freeze --
    code, out = _run(
        "graph-detector-v4-3-local-judge",
        "--judge",
        "A",
        "--pass",
        "reference",
        "--bundle",
        str(bundle),
        "--out-dir",
        str(work),
        "--eval-key",
        str(work / "PROTECTED_STORE_EVAL_KEY.json"),
        "--fake-model",
    )
    # Judge A's blind output DOES exist here, so this must succeed; the refusal case is
    # checked below on a judge whose blind pass was removed.
    check(
        "reference pass runs once blind is frozen",
        code == 0,
        out.strip().splitlines()[-1][:80] if out else "",
    )

    missing = work / "V4_3_LOCAL_JUDGE_B_BLIND.jsonl"
    stashed = missing.read_text(encoding="utf-8")
    missing.unlink()
    code, out = _run(
        "graph-detector-v4-3-local-judge",
        "--judge",
        "B",
        "--pass",
        "reference",
        "--bundle",
        str(bundle),
        "--out-dir",
        str(work),
        "--eval-key",
        str(work / "PROTECTED_STORE_EVAL_KEY.json"),
        "--fake-model",
    )
    check(
        "reference refuses before the blind freeze",
        code != 0 and _says(out, "blind pass has not been closed"),
    )
    missing.write_text(stashed, encoding="utf-8")

    code, out = _run(
        "graph-detector-v4-3-local-judge",
        "--judge",
        "A",
        "--pass",
        "blind",
        "--bundle",
        str(bundle),
        "--out-dir",
        str(work),
        "--eval-key",
        str(work / "PROTECTED_STORE_EVAL_KEY.json"),
        "--fake-model",
    )
    check(
        "the blind pass refuses an --eval-key", code != 0 and _says(out, "refused for --pass blind")
    )

    # Drive both reference passes with the clean judge.
    for role in ("A", "B"):
        names = output_names(judge=role, pass_name="reference", run_id="", reportable=True)
        for path in (work / names["partial"], work / names["output"]):
            path.unlink(missing_ok=True)
        run_rows(
            rows,
            lambda prompt, role=role: _clean_judgement(prompt, role),
            partial_path=work / names["partial"],
            role=role,
            pass_name="reference",
            reference_of=reference_of,
        )
        _close(
            work / names["partial"],
            work / names["output"],
            rows=rows,
            judge=role,
            pass_name="reference",
        )
    check("both reference passes close", True)

    # ------------------------------------------------------- 4. label report --
    code, out = _run(
        "graph-detector-v4-3-label-report",
        "--bundle",
        str(bundle),
        "--judge-dir",
        str(work),
        "--eval-key",
        str(work / "PROTECTED_STORE_EVAL_KEY.json"),
        "--output-dir",
        str(work),
        "--blind-only",
    )
    check(
        "blind-only report writes disagreements",
        code == 0 and (work / "V4_3_BLIND_DISAGREEMENTS.jsonl").exists(),
    )
    disagreements = [
        json.loads(line)
        for line in (work / "V4_3_BLIND_DISAGREEMENTS.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    check(
        "the disagreement file carries no reference answer",
        all("reference_answer" not in d for d in disagreements),
        f"{len(disagreements)} rows",
    )

    adjudication = work / "V4_3_BLIND_ADJUDICATION.jsonl"
    adjudication.write_text(
        "".join(
            json.dumps({"audit_id": d["audit_id"], "answer_attempt": d["judge_a"]}) + "\n"
            for d in disagreements
        ),
        encoding="utf-8",
    )

    code, out = _run(
        "graph-detector-v4-3-label-report",
        "--bundle",
        str(bundle),
        "--judge-dir",
        str(work),
        "--eval-key",
        str(work / "PROTECTED_STORE_EVAL_KEY.json"),
        "--output-dir",
        str(work),
        "--adjudication",
        str(adjudication),
        "--require-reference-pass",
    )
    report = json.loads((work / "DETECTOR_V4_3_LABEL_REPORT.json").read_text(encoding="utf-8"))
    check(
        "full label report runs and gates",
        code == 0 and "gates_passed" in report,
        f"gates_passed={report.get('gates_passed')} failures={len(report.get('gate_failures', []))}",
    )
    check(
        "the report refuses to call a hash-judge run publication-valid",
        report.get("publication_label_valid") is False and report.get("human_grounded") is False,
    )

    # ------------------------------------------------- 5. labelled bundle --
    labelled = work / "labelled"
    code, out = _run(
        "graph-detector-v4-3-bundle",
        "--store-dir",
        str(work),
        "--out-dir",
        str(labelled),
        "--labels",
        str(work / "V4_3_ADJUDICATED.jsonl"),
    )
    check(
        "labelled bundle rebuilds into a new directory",
        code == 0 and (labelled / "DETECTOR_V4_3_PAIR_BUNDLE.json").exists(),
    )
    check(
        "the unlabelled freeze is untouched",
        json.loads(bundle.read_text(encoding="utf-8"))["pairs"][0]["label"] is None,
    )

    # ------------------------------------------------ 6. operating point + gate --
    audit = work / "SYNTHETIC_AUDIT.json"
    labelled_rows = json.loads(
        (labelled / "DETECTOR_V4_3_PAIR_BUNDLE.json").read_text(encoding="utf-8")
    )["pairs"]
    population_of = {k: v["population"] for k, v in key["rows"].items()}
    concept_of = {k: v["concept_id"] for k, v in key["rows"].items()}
    audit_rows = [
        {
            "audit_id": r["audit_id"],
            "request": r["conditioning_question"],
            "candidate_text": r["candidate_text"],
            "label": r["label"],
            "population": population_of.get(r["audit_id"], "protected"),
            "concept_id": concept_of.get(r["audit_id"], ""),
            "partition": "development" if i % 2 == 0 else "heldout",
        }
        for i, r in enumerate(labelled_rows)
    ]
    audit.write_text(json.dumps({"rows": audit_rows}), encoding="utf-8")

    code, out = _run(
        "graph-detector-v4-3-select-operating-point",
        "--audit",
        str(audit),
        "--backend",
        "lexical",
        "--protected-store",
        str(work / "PROTECTED_STORE_RUNTIME.json"),
        "--device",
        "",
        "--output-dir",
        str(work),
    )
    chose = (work / "DETECTOR_V4_3_OPERATING_POINT.json").exists()
    check(
        "operating point selects on development only",
        code == 0 and chose,
        out.strip().splitlines()[-1][:110] if out else "",
    )

    code, out = _run(
        "graph-detector-v4-3-select-operating-point",
        "--audit",
        str(audit),
        "--backend",
        "lexical",
        "--partition",
        "heldout",
        "--protected-store",
        str(work / "PROTECTED_STORE_RUNTIME.json"),
        "--device",
        "",
        "--output-dir",
        str(work),
    )
    check(
        "selecting on the held-out partition is refused",
        code != 0 and _says(out, "development partition only"),
    )

    if chose:
        code, out = _run(
            "graph-detector-v4-3-final-gate",
            "--audit",
            str(audit),
            "--backend",
            "lexical",
            "--protected-store",
            str(work / "PROTECTED_STORE_RUNTIME.json"),
            "--operating-point",
            str(work / "DETECTOR_V4_3_OPERATING_POINT.json"),
            "--device",
            "",
            "--output-dir",
            str(work),
        )
        check(
            "held-out gate opens once and reports",
            code == 0,
            out.strip().splitlines()[-1][:110] if out else "",
        )

        code, out = _run(
            "graph-detector-v4-3-final-gate",
            "--audit",
            str(audit),
            "--backend",
            "lexical",
            "--protected-store",
            str(work / "PROTECTED_STORE_RUNTIME.json"),
            "--operating-point",
            str(work / "DETECTOR_V4_3_OPERATING_POINT.json"),
            "--device",
            "",
            "--output-dir",
            str(work),
        )
        check("a second held-out opening is refused", code != 0 and _says(out, "has been opened"))

    # ------------------------------------------------------- 7. human tooling --
    code, out = _run(
        "graph-detector-v4-3-human-sample",
        "--bundle",
        str(labelled / "DETECTOR_V4_3_PAIR_BUNDLE.json"),
        "--label-report",
        str(work / "DETECTOR_V4_3_LABEL_REPORT.json"),
        "--out-dir",
        str(work / "human"),
    )
    check(
        "a human draw without the fresh audit is refused",
        code != 0 and _says(out, "fresh engineering audit"),
    )

    code, out = _run(
        "graph-detector-v4-3-human-sample",
        "--bundle",
        str(labelled / "DETECTOR_V4_3_PAIR_BUNDLE.json"),
        "--label-report",
        str(work / "DETECTOR_V4_3_LABEL_REPORT.json"),
        "--out-dir",
        str(work / "human"),
        "--exploratory",
    )
    check("an exploratory draw is allowed and marked", code == 0)
    if code == 0:
        manifest = json.loads(
            (work / "human" / "V4_3_HUMAN_SAMPLE.json").read_text(encoding="utf-8")
        )
        check("the exploratory sample is marked non-reportable", manifest["reportable"] is False)
        check(
            "every sampled row carries an exact inclusion probability",
            all(0 < r["inclusion_probability"] <= 1 for r in manifest["rows"].values()),
        )

    failed = [name for name, ok, _ in steps if not ok]
    print("\n" + "=" * 72)
    print(f"{len(steps) - len(failed)}/{len(steps)} pipeline steps behaved as specified")
    if failed:
        print("FAILED: " + "; ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
