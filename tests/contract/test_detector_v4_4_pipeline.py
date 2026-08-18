"""The v4.4 pipeline's refusals, exercised end to end without a GPU.

Every test here is about something the pipeline must REFUSE to do. That is deliberate: the
happy path is checked by running the commands, but the properties that make the labels
worth anything are all negative -- the reference pass cannot open early, a failed panel
cannot be adjudicated, a rater cannot see a model's label, a panel frozen against a
different bundle cannot be used.

The judges are simulated. Two 14B/24B models are not a unit-test dependency, and what these
tests check is the harness around them: resume, malformed accounting, pass ordering, gate
arithmetic and file discipline, none of which involves a weight.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import typer

from rdl.cli.detector_v4_4_bundle import BUNDLE_FILENAME, PANEL_FILENAME
from rdl.cli.detector_v4_4_judge import (
    fake_hierarchical_judgement,
    output_names_v4_4,
    run_rows_v4_4,
)
from rdl.cli.detector_v4_4_report import detector_v4_4_label_report
from rdl.eval.detector_v4_4 import PROMPT_VERSION

REPO = Path(__file__).resolve().parents[2]
V4_4 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_4"


# =====================================================================================
# fixtures: a miniature bundle and panel, and a judge that answers deterministically
# =====================================================================================


def _pairs(n: int = 90) -> list[dict]:
    """Rows spread evenly over the three intended classes and three question types."""
    classes = ("NONE", "PARTIAL", "ANSWER")
    types = ("slot", "yes-no", "open-ended")
    return [
        {
            "audit_id": f"row-{i:03d}",
            "conditioning_question": f"Question {i} about author {i % 7}?",
            "subject_aliases": [f"Author {i % 7}"],
            "candidate_text": f"Candidate text number {i}.",
            "intended_class": classes[i % 3],
            "source_subtype": "refusal" if i % 3 == 0 else "controlled_narrowing",
            "question_type_hint": types[i % 3],
            "subject_group": f"group-{i % 7}",
            "split": "train" if i % 2 else "development",
            "origin": "v4_4_supplement",
            "pair_sha256": f"{i:064x}",
            "label": None,
        }
        for i in range(n)
    ]


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    pairs = _pairs()
    bundle = {
        "schema": "graph-detector-v4-4-pair-bundle-v1",
        "bundle_sha256": "deadbeef" * 8,
        "n_pairs": len(pairs),
        "pairs": pairs,
    }
    (tmp_path / BUNDLE_FILENAME).write_text(json.dumps(bundle), encoding="utf-8")
    panel = {
        "schema": "graph-detector-v4-4-calibration-panel-v1",
        "panel_sha256": "cafe" * 16,
        "bundle_sha256": bundle["bundle_sha256"],
        "n_rows": len(pairs),
        "per_class": 30,
        "rows": [{"audit_id": p["audit_id"], "intended_class": p["intended_class"]} for p in pairs],
    }
    (tmp_path / PANEL_FILENAME).write_text(json.dumps(panel), encoding="utf-8")
    return tmp_path


def _write_pass(
    workspace: Path,
    judge: str,
    *,
    label_of,
    pass_name: str = "blind",
    prompt_version: str = PROMPT_VERSION,
) -> None:
    """Write a closed judging pass, with the hierarchy consistent by construction."""
    pairs = json.loads((workspace / BUNDLE_FILENAME).read_text(encoding="utf-8"))["pairs"]
    names = output_names_v4_4(judge=judge, pass_name=pass_name, run_id="", reportable=True)
    lines = []
    for pair in pairs:
        label = label_of(pair)
        if pass_name == "reference":
            row = {"audit_id": pair["audit_id"], "reference_content": label}
        else:
            row = {
                "audit_id": pair["audit_id"],
                "addresses_question": "no" if label == "NONE" else "yes",
                "standalone_answer": "yes" if label == "ANSWER" else "no",
                "answer_attempt": label,
                "question_type": pair["question_type_hint"],
                "subject_only": "no",
                "refusal": "no",
            }
        row.update(
            {
                "judge": judge,
                "pass": pass_name,
                "prompt_version": prompt_version,
                "malformed": False,
                "inconsistent_hierarchy": False,
            }
        )
        lines.append(json.dumps(row))
    (workspace / names["output"]).write_text("\n".join(lines) + "\n", encoding="utf-8")


# =====================================================================================
# the judge runner
# =====================================================================================


def test_the_runner_records_an_inconsistent_hierarchy_as_malformed_and_counts_it_apart(tmp_path):
    """Both are refused; only one of them says the RUBRIC is unclear.

    An unparseable response means the model ignored the format, which a tighter schema block
    fixes. A contradictory object means it followed the format and could not hold the
    hierarchy together, which no amount of shouting "JSON" repairs.
    """
    rows = _pairs(60)
    counts = run_rows_v4_4(
        rows,
        fake_hierarchical_judgement,
        partial_path=tmp_path / "partial.jsonl",
        role="A",
    )
    assert counts["n_judged"] + counts["n_malformed"] == 60
    assert counts["n_malformed"] > 0, "the fake judge emits prose and contradictions on purpose"
    assert counts["n_inconsistent"] > 0
    assert counts["n_inconsistent"] <= counts["n_malformed"]

    written = [
        json.loads(line)
        for line in (tmp_path / "partial.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    contradictory = [r for r in written if r.get("inconsistent_hierarchy")]
    assert contradictory
    for row in contradictory:
        # Refused, never repaired: no label was written for it.
        assert row["malformed"] is True
        assert row["answer_attempt"] is None


def test_the_runner_resumes_and_does_not_rejudge_a_completed_row(tmp_path):
    rows = _pairs(30)
    partial = tmp_path / "partial.jsonl"
    first = run_rows_v4_4(rows[:10], fake_hierarchical_judgement, partial_path=partial, role="A")
    second = run_rows_v4_4(rows, fake_hierarchical_judgement, partial_path=partial, role="A")
    assert first["n_judged"] + first["n_malformed"] == 10
    assert second["n_skipped"] == 10
    assert second["n_judged"] + second["n_malformed"] == 20


def test_a_judged_row_records_the_prompt_version_it_was_produced_under(tmp_path):
    run_rows_v4_4(
        _pairs(5), fake_hierarchical_judgement, partial_path=tmp_path / "p.jsonl", role="A"
    )
    rows = [
        json.loads(line) for line in (tmp_path / "p.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {r["prompt_version"] for r in rows} == {PROMPT_VERSION}


def test_a_reference_pass_cannot_run_before_the_blind_pass_is_closed(tmp_path):
    """A reference label produced first could have been revised in light of the answer."""
    from rdl.cli.detector_v4_4_judge import detector_v4_4_local_judge

    (tmp_path / "input.jsonl").write_text(
        "\n".join(json.dumps(p) for p in _pairs(5)), encoding="utf-8"
    )
    (tmp_path / "key.json").write_text(json.dumps({"rows": {}}), encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="blind pass has not been closed"):
        detector_v4_4_local_judge(
            judge="A",
            pass_name="reference",
            input_path=tmp_path / "input.jsonl",
            bundle=tmp_path / "unused.json",
            eval_key=tmp_path / "key.json",
            pins=tmp_path / "pins.json",
            out_dir=tmp_path,
            device="cpu",
            limit=0,
            run_id="",
            non_reportable=False,
            fake_model=True,
            close=False,
        )


def test_the_blind_pass_refuses_to_be_handed_an_eval_key(tmp_path):
    """The guarantee is that the file is never opened, not that the prompt omits it."""
    from rdl.cli.detector_v4_4_judge import detector_v4_4_local_judge

    (tmp_path / "input.jsonl").write_text(
        "\n".join(json.dumps(p) for p in _pairs(3)), encoding="utf-8"
    )
    (tmp_path / "key.json").write_text(json.dumps({"rows": {}}), encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="refused for --pass blind"):
        detector_v4_4_local_judge(
            judge="A",
            pass_name="blind",
            input_path=tmp_path / "input.jsonl",
            bundle=tmp_path / "unused.json",
            eval_key=tmp_path / "key.json",
            pins=tmp_path / "pins.json",
            out_dir=tmp_path,
            device="cpu",
            limit=0,
            run_id="",
            non_reportable=False,
            fake_model=True,
            close=False,
        )


def test_a_limited_run_cannot_occupy_a_reportable_filename(tmp_path):
    from rdl.cli.detector_v4_4_judge import detector_v4_4_local_judge

    with pytest.raises(typer.BadParameter, match="--limit without --non-reportable"):
        detector_v4_4_local_judge(
            judge="A",
            pass_name="blind",
            input_path=None,
            bundle=tmp_path / "b.json",
            eval_key=None,
            pins=tmp_path / "pins.json",
            out_dir=tmp_path,
            device="cpu",
            limit=50,
            run_id="",
            non_reportable=False,
            fake_model=True,
            close=False,
        )


def test_v4_4_output_names_cannot_collide_with_v4_3s():
    """Two different annotations of the same rows must not share a directory entry."""
    from rdl.cli.detector_v4_3_local_judge import output_names

    v4_3 = output_names(judge="A", pass_name="blind", run_id="", reportable=True)
    v4_4 = output_names_v4_4(judge="A", pass_name="blind", run_id="", reportable=True)
    assert set(v4_3.values()).isdisjoint(v4_4.values())
    assert all(name.startswith("V4_4_") for name in v4_4.values())


# =====================================================================================
# the label report
# =====================================================================================


def test_the_report_refuses_a_panel_frozen_against_a_different_bundle(workspace):
    panel = json.loads((workspace / PANEL_FILENAME).read_text(encoding="utf-8"))
    panel["bundle_sha256"] = "0" * 64
    (workspace / PANEL_FILENAME).write_text(json.dumps(panel), encoding="utf-8")
    _write_pass(workspace, "A", label_of=lambda p: p["intended_class"])
    _write_pass(workspace, "B", label_of=lambda p: p["intended_class"])

    with pytest.raises(typer.BadParameter, match="frozen against a different bundle"):
        detector_v4_4_label_report(
            out_dir=workspace,
            bundle=None,
            panel=None,
            eval_key=workspace / "key.json",
            blind_only=True,
            adjudication=None,
            require_reference_pass=False,
        )


def test_the_report_refuses_to_pool_two_prompt_versions(workspace):
    _write_pass(workspace, "A", label_of=lambda p: p["intended_class"])
    _write_pass(
        workspace, "B", label_of=lambda p: p["intended_class"], prompt_version="v4.3-local-prompt-2"
    )
    with pytest.raises(typer.BadParameter, match="prompt versions"):
        detector_v4_4_label_report(
            out_dir=workspace,
            bundle=None,
            panel=None,
            eval_key=workspace / "key.json",
            blind_only=True,
            adjudication=None,
            require_reference_pass=False,
        )


def test_perfect_agreement_passes_the_panel_gate_and_writes_no_disagreements(workspace, capsys):
    _write_pass(workspace, "A", label_of=lambda p: p["intended_class"])
    _write_pass(workspace, "B", label_of=lambda p: p["intended_class"])
    detector_v4_4_label_report(
        out_dir=workspace,
        bundle=None,
        panel=None,
        eval_key=workspace / "key.json",
        blind_only=True,
        adjudication=None,
        require_reference_pass=False,
    )
    echoed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert echoed["panel_gate_passed"] is True
    assert echoed["panel_kappa"] == 1.0
    assert echoed["n_disagreements"] == 0

    report = json.loads((workspace / "DETECTOR_V4_4_LABEL_REPORT.json").read_text())
    # The mixture number is reported beside the panel one, never instead of it.
    assert report["mixture_diagnostic"]["gated"] is False
    assert report["mixture_diagnostic"]["agreement"]["kappa"] == 1.0
    assert report["v4_3_comparison"]["v4_3_blind_kappa"] == 0.4038


def test_a_failed_panel_cannot_be_adjudicated(workspace):
    """Adjudicating a failed panel and calling it passed is the failure mode this prevents.

    Judge B calls every PARTIAL an ANSWER -- v4.3's dominant disagreement cell, exaggerated.
    The gate fails, and the report refuses to build a label authority on top of it.
    """
    _write_pass(workspace, "A", label_of=lambda p: p["intended_class"])
    _write_pass(
        workspace,
        "B",
        label_of=lambda p: "ANSWER" if p["intended_class"] == "PARTIAL" else p["intended_class"],
    )
    (workspace / "adj.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="balanced-panel gate failed"):
        detector_v4_4_label_report(
            out_dir=workspace,
            bundle=None,
            panel=None,
            eval_key=workspace / "key.json",
            blind_only=False,
            adjudication=workspace / "adj.jsonl",
            require_reference_pass=False,
        )


def test_the_disagreement_file_is_blind_and_does_not_name_the_judges(workspace):
    """The researcher adjudicates under the same blinding the judges had.

    Labels are given as label_1/label_2 in sorted order rather than as A/B, so "judge B
    usually wins" cannot become a tie-break heuristic partway through the file.
    """
    _write_pass(workspace, "A", label_of=lambda p: p["intended_class"])
    _write_pass(
        workspace,
        "B",
        label_of=lambda p: "ANSWER" if p["intended_class"] == "PARTIAL" else p["intended_class"],
    )
    detector_v4_4_label_report(
        out_dir=workspace,
        bundle=None,
        panel=None,
        eval_key=workspace / "key.json",
        blind_only=True,
        adjudication=None,
        require_reference_pass=False,
    )
    rows = [
        json.loads(line)
        for line in (workspace / "V4_4_BLIND_DISAGREEMENTS.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert rows
    for row in rows:
        assert set(row) == {
            "audit_id",
            "conditioning_question",
            "subject_aliases",
            "candidate_text",
            "label_1",
            "label_2",
            "answer_attempt",
        }
        assert row["label_1"] <= row["label_2"]
        assert row["answer_attempt"] is None


def test_an_unadjudicated_disagreement_blocks_the_authority(workspace):
    _write_pass(workspace, "A", label_of=lambda p: p["intended_class"])

    def judge_b(pair):
        # One row disagrees, and the panel still passes comfortably.
        return "ANSWER" if pair["audit_id"] == "row-001" else pair["intended_class"]

    _write_pass(workspace, "B", label_of=judge_b)
    (workspace / "adj.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="unadjudicated"):
        detector_v4_4_label_report(
            out_dir=workspace,
            bundle=None,
            panel=None,
            eval_key=workspace / "key.json",
            blind_only=False,
            adjudication=workspace / "adj.jsonl",
            require_reference_pass=False,
        )


def test_the_authority_reports_the_class_minima_rather_than_asserting_them(workspace, capsys):
    """The 200/100/100 minima are on ADJUDICATED rows, which this 90-row fixture cannot meet.

    That is the honest outcome and the test asserts it: the authority is written, it is not
    green, and the failure names which minimum was missed. A protocol that silently emitted
    a green authority on 90 rows would emit one on 1,733 rows that fell short too.
    """
    _write_pass(workspace, "A", label_of=lambda p: p["intended_class"])
    _write_pass(workspace, "B", label_of=lambda p: p["intended_class"])
    (workspace / "adj.jsonl").write_text("", encoding="utf-8")
    detector_v4_4_label_report(
        out_dir=workspace,
        bundle=None,
        panel=None,
        eval_key=workspace / "key.json",
        blind_only=False,
        adjudication=workspace / "adj.jsonl",
        require_reference_pass=False,
    )
    echoed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert echoed["final_distribution"] == {"ANSWER": 30, "NONE": 30, "PARTIAL": 30}
    assert any("n_none_rows" in f for f in echoed["bundle_minima_failures"])
    assert echoed["authority_green"] is False

    authority = json.loads(
        (workspace / "DETECTOR_V4_4_LABEL_AUTHORITY.json").read_text(encoding="utf-8")
    )
    assert authority["trainable_field"] == "answer_attempt"
    assert authority["not_trainable"] == ["reference_content"]


# =====================================================================================
# the human tooling
# =====================================================================================


def test_a_rater_file_carries_nothing_but_the_four_blinded_fields(tmp_path, workspace):
    from rdl.cli.detector_v4_4_human import detector_v4_4_human_sample, rater_filename

    detector_v4_4_human_sample(
        bundle=workspace / BUNDLE_FILENAME,
        fresh_audit=None,
        out_dir=tmp_path,
        n_per_source=20,
        raters="A,B",
        exploratory=True,
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / rater_filename(rater="A", pass_name="blind"))
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    assert rows
    for row in rows:
        assert set(row) == {
            "audit_id",
            "conditioning_question",
            "subject_aliases",
            "candidate_text",
            "answer_attempt",
        }
        assert row["answer_attempt"] is None


def test_a_reportable_human_sample_needs_the_fresh_audit_half(tmp_path, workspace):
    """125 from the v4.4 population alone measures the detector where it was developed."""
    from rdl.cli.detector_v4_4_human import detector_v4_4_human_sample

    with pytest.raises(typer.BadParameter, match="fresh engineering audit"):
        detector_v4_4_human_sample(
            bundle=workspace / BUNDLE_FILENAME,
            fresh_audit=None,
            out_dir=tmp_path,
            n_per_source=20,
            raters="A,B",
            exploratory=False,
        )


def test_the_human_reference_pass_refuses_until_every_blind_file_is_complete(tmp_path, workspace):
    from rdl.cli.detector_v4_4_human import (
        detector_v4_4_human_reference_pass,
        detector_v4_4_human_sample,
    )

    detector_v4_4_human_sample(
        bundle=workspace / BUNDLE_FILENAME,
        fresh_audit=None,
        out_dir=tmp_path,
        n_per_source=20,
        raters="A,B",
        exploratory=True,
    )
    with pytest.raises(typer.BadParameter, match="blind pass is not frozen"):
        detector_v4_4_human_reference_pass(
            out_dir=tmp_path,
            eval_key=tmp_path / "key.json",
            bundle=workspace / BUNDLE_FILENAME,
        )


def test_one_rater_is_refused(tmp_path, workspace):
    from rdl.cli.detector_v4_4_human import detector_v4_4_human_sample

    with pytest.raises(typer.BadParameter, match="two independent raters"):
        detector_v4_4_human_sample(
            bundle=workspace / BUNDLE_FILENAME,
            fresh_audit=None,
            out_dir=tmp_path,
            n_per_source=20,
            raters="A",
            exploratory=True,
        )


def test_the_pilot_is_bounded_at_30_to_50_rows_and_marked_non_reportable(tmp_path, workspace):
    """More than 50 rows is a measurement running under a name that says it is not one."""
    from rdl.cli.detector_v4_4_human import detector_v4_4_human_pilot

    with pytest.raises(typer.BadParameter, match="30-50 rows"):
        detector_v4_4_human_pilot(
            bundle=workspace / BUNDLE_FILENAME,
            out_dir=tmp_path,
            n=250,
            open_ended_share=0.6,
            raters="A,B",
        )

    detector_v4_4_human_pilot(
        bundle=workspace / BUNDLE_FILENAME,
        out_dir=tmp_path,
        n=40,
        open_ended_share=0.6,
        raters="A,B",
    )
    manifest = json.loads((tmp_path / "V4_4_HUMAN_PILOT.json").read_text(encoding="utf-8"))
    assert manifest["reportable"] is False
    assert manifest["n_rows"] == 40
    assert manifest["composition"]["by_question_type_hint"]["open-ended"] >= 20


def test_adjudication_refuses_before_the_rater_files_are_frozen(tmp_path, workspace):
    from rdl.cli.detector_v4_4_human import (
        detector_v4_4_human_adjudicate,
        detector_v4_4_human_sample,
    )

    detector_v4_4_human_sample(
        bundle=workspace / BUNDLE_FILENAME,
        fresh_audit=None,
        out_dir=tmp_path,
        n_per_source=20,
        raters="A,B",
        exploratory=True,
    )
    with pytest.raises(typer.BadParameter, match="human-import"):
        detector_v4_4_human_adjudicate(
            out_dir=tmp_path, pass_name="blind", decisions=tmp_path / "d.jsonl"
        )


def test_a_rater_file_edited_after_import_invalidates_the_agreement(tmp_path, workspace):
    """The reported kappa would no longer describe the labels being adjudicated."""
    from rdl.cli.detector_v4_4_human import (
        detector_v4_4_human_adjudicate,
        detector_v4_4_human_import,
        detector_v4_4_human_sample,
        rater_filename,
    )

    detector_v4_4_human_sample(
        bundle=workspace / BUNDLE_FILENAME,
        fresh_audit=None,
        out_dir=tmp_path,
        n_per_source=20,
        raters="A,B",
        exploratory=True,
    )
    for rater in ("A", "B"):
        path = tmp_path / rater_filename(rater=rater, pass_name="blind")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line]
        for row in rows:
            row["answer_attempt"] = "NONE"
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    detector_v4_4_human_import(out_dir=tmp_path, pass_name="blind", pilot=False)

    path = tmp_path / rater_filename(rater="A", pass_name="blind")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    rows[0]["answer_attempt"] = "ANSWER"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    (tmp_path / "d.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(typer.BadParameter, match="has changed since agreement was computed"):
        detector_v4_4_human_adjudicate(
            out_dir=tmp_path, pass_name="blind", decisions=tmp_path / "d.jsonl"
        )
