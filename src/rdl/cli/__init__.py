"""Command-line entry points.

    rdl env-check              first thing run in every Colab session
    rdl discover-checkpoints   RUN THIS FIRST ON DAY 1
    rdl run-repro              Days 1-2: the open-unlearning reproduction gate
    rdl run-condition          Days 3-5: C0/C1/C2/C3
    rdl make-report            tables + figures

Also available as `python -m rdl.cli <command>`.
"""

from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str):
    # Lazy so that importing rdl.cli does not pull typer into every test.
    if name == "app":
        from .__main__ import app

        return app
    raise AttributeError(name)
