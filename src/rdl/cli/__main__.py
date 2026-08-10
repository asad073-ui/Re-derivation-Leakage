"""The Typer app that aggregates every CLI."""

from __future__ import annotations

import typer

from .discover_checkpoints import discover_checkpoints
from .env_check import env_check
from .make_leak_report import make_leak_report
from .make_report import make_report
from .rescore_day2 import rescore_day2
from .run_condition import run_condition
from .run_leak import run_leak
from .run_repro import run_repro

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


if __name__ == "__main__":
    app()
