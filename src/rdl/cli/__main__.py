"""The Typer app that aggregates every CLI."""

from __future__ import annotations

import typer

from .bundle_graph import bundle_graph
from .calibrate_detector import calibrate_detector
from .detector_corpus import detector_corpus
from .detector_recall import detector_recall
from .discover_checkpoints import discover_checkpoints
from .env_check import env_check
from .finalize_graph import finalize_graph
from .freeze_graph_cohort import freeze_graph_cohort
from .make_leak_report import make_leak_report
from .make_report import make_report
from .plan_graph_run import plan_graph_run
from .report_graph import report_graph
from .rescore_day2 import rescore_day2
from .rescore_leak import rescore_leak
from .run_condition import run_condition
from .run_graph import run_graph
from .run_leak import run_leak
from .run_repro import run_repro
from .score_graph import score_graph

app = typer.Typer(
    name="rdl",
    help="Re-derivation Leakage — memory-mediated recovery of unlearned knowledge.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("env-check")(env_check)
app.command("discover-checkpoints")(discover_checkpoints)
app.command("run-repro")(run_repro)
app.command("run-condition")(run_condition)
app.command("make-report")(make_report)
app.command("rescore-day2")(rescore_day2)
app.command("run-leak")(run_leak)
app.command("make-leak-report")(make_leak_report)
app.command("rescore-leak")(rescore_leak)

# graph-unlearning-v1. A separate experiment family: the commands above are frozen
# evidence for the v5 conditions and the two-agent Leak@k work.
app.command("graph-plan")(plan_graph_run)
app.command("graph-run")(run_graph)
app.command("graph-score")(score_graph)
app.command("graph-report")(report_graph)
app.command("graph-finalize")(finalize_graph)
app.command("graph-freeze-cohort")(freeze_graph_cohort)
app.command("graph-calibrate")(calibrate_detector)
# Reanalysis phases (GU-0030, GU-0031). All three read committed evidence and generate
# nothing: recall diagnoses the detection gap, the corpus freezes what a fix may be
# fitted on, and the bundle links a study's runs into one verdict.
app.command("graph-detector-recall")(detector_recall)
app.command("graph-detector-corpus")(detector_corpus)
app.command("graph-bundle")(bundle_graph)


if __name__ == "__main__":
    app()
