"""``rdl graph-freeze-cohort`` — fill a cohort's content hashes from the real dataset.

Needs network (or a fixture). Run once per split, then never again: the hashes are what
prove a later run loaded the same questions and answers the manifest froze.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from ..eval.tofu_data import load_items
from ..studies.graph_leak.cohort import CohortError, freeze_cohort, load_cohort

__all__ = ["freeze_graph_cohort"]


def freeze_graph_cohort(
    manifest: Path = typer.Option(..., "--manifest", help="a cohort split json"),
    fixture: Path | None = typer.Option(None, "--fixture", help="offline fixture; CPU only"),
    revision: str | None = typer.Option(None, "--dataset-revision"),
    token: str | None = typer.Option(None, "--hf-token"),
    write: bool = typer.Option(False, "--write", help="overwrite the manifest in place"),
) -> None:
    try:
        cohort = load_cohort(manifest, require_frozen=False)
    except CohortError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if not fixture and not revision:
        raise typer.BadParameter(
            "--dataset-revision is required when freezing against real data. Freezing "
            "content hashes taken from the branch head, then recording no revision, "
            "produces a manifest nobody can reproduce the download for.\n"
            "Find it with: huggingface_hub.HfApi().dataset_info('locuslab/TOFU').sha"
        )
    items, provenance = load_items(
        dataset="stub" if fixture else "tofu",
        split=cohort.dataset_config,
        fixture=fixture,
        allow_fixture=bool(fixture),
        token=token,
        # The SAME revision that gets recorded is the one that is downloaded.
        revision=revision,
    )
    try:
        frozen = freeze_cohort(cohort, items, revision=revision)
    except CohortError as exc:
        raise typer.BadParameter(str(exc)) from exc

    payload = frozen.to_dict()
    payload["source_provenance"] = provenance
    if write:
        manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        typer.echo(f"froze {len(frozen.items)} items -> {manifest}")
    else:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
