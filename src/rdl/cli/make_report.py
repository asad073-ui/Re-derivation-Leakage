"""`rdl make-report` — tables and figures.

**All plotting logic lives here and nowhere else.** A matplotlib import scattered
through the metric modules is how a headless CI run starts failing for reasons that
have nothing to do with the science.

matplotlib is an optional dependency: without it the markdown tables are still
produced and only the figures are skipped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from ..logging_utils import read_jsonl
from ..paths import manifest_path, results_dir

__all__ = ["collect_runs", "make_report", "markdown_table"]


def collect_runs(root: Path | None = None) -> list[dict]:
    """Load every condition/repro report under results/."""
    rd = results_dir(root)
    out: list[dict] = []
    if not rd.exists():
        return out
    for path in sorted(rd.glob("*/condition_report.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted(rd.glob("*/repro_report.json")):
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def _fmt_ci(stat: dict[str, Any] | None) -> str:
    if not stat or stat.get("mean") is None:
        return "—"
    mean = stat["mean"]
    ci = stat.get("ci95") or [None, None]
    if ci[0] is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def markdown_table(runs: list[dict]) -> str:
    """The main results table: one row per condition."""
    conds = [r for r in runs if r.get("phase") == "phase0_days3-5"]
    if not conds:
        return "_no condition runs found_"

    rows = [
        "| condition | n seeds | SysRecall@k (store) | SysRecall@k (final) | "
        "laundering_rate | delegation_rate | write policy |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in sorted(conds, key=lambda x: str(x.get("condition"))):
        recall = r.get("recall_at_k", {})
        rows.append(
            "| {cond} | {n} | {store} | {final} | {laund} | {deleg} | `{wp}` |".format(
                cond=r.get("condition", "?"),
                n=r.get("n_seeds", "?"),
                store=_fmt_ci(recall.get("persistent_store_after_episode")),
                final=_fmt_ci(recall.get("final_answer")),
                laund=_fmt_ci(r.get("laundering_rate")),
                deleg=_fmt_ci(r.get("delegation_rate")),
                wp=(r.get("config") or {}).get("writepolicy", {}).get("mode", "?"),
            )
        )
    return "\n".join(rows)


def repro_table(runs: list[dict]) -> str:
    repros = [r for r in runs if r.get("phase") == "phase0_days1-2_repro"]
    if not repros:
        return "_no reproduction runs found_"
    rows = [
        "| target | checkpoint | metric | ours | published | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in repros:
        for c in r.get("comparisons", []):
            verdict = (
                ("PASS" if c["passed"] else "FAIL") if c.get("gated") else "reported, not gated"
            )
            rows.append(
                f"| {r.get('target')} | `{r.get('checkpoint')}` | {c['metric']} | "
                f"{c['ours']} | {c['published']} | {verdict} |"
            )
    return "\n".join(rows)


def _figures(runs: list[dict], out_dir: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless; must be set before pyplot
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    conds = [r for r in runs if r.get("phase") == "phase0_days3-5"]
    if not conds:
        return []
    conds.sort(key=lambda x: str(x.get("condition")))

    written: list[Path] = []
    labels = [str(c.get("condition")) for c in conds]

    def _series(key_path: list[str]) -> tuple[list[float], list[float]]:
        means, errs = [], []
        for c in conds:
            node: Any = c
            for k in key_path:
                node = (node or {}).get(k, {})
            m = node.get("mean", 0.0) or 0.0
            ci = node.get("ci95") or [m, m]
            means.append(m)
            errs.append(max(0.0, (ci[1] - ci[0]) / 2))
        return means, errs

    # Figure 1: SysRecall@k on the persistent store, per condition.
    means, errs = _series(["recall_at_k", "persistent_store_after_episode"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=errs, capsize=4)
    ax.set_ylabel("SysRecall@k (persistent store)")
    ax.set_ylim(0, 1)
    ax.set_title("Forget-set recovery from the shared memory store")
    fig.tight_layout()
    p = out_dir / "fig1_sysrecall_store.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    # Figure 2: the headline — laundering rate.
    means, errs = _series(["laundering_rate"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, means, yerr=errs, capsize=4, color="#b4423a")
    ax.set_ylabel("laundering_rate")
    ax.set_ylim(0, 1)
    ax.set_title("Recovered items whose node passes both SBU invariants")
    fig.tight_layout()
    p = out_dir / "fig2_laundering_rate.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    written.append(p)

    return written


def make_report(
    out: Path | None = typer.Option(
        None, "--out", help="output markdown (default results/REPORT.md)"
    ),
    figures: bool = typer.Option(True, "--figures/--no-figures"),
) -> None:
    """Read results/ and emit a markdown report plus figures."""
    runs = collect_runs()
    if not runs:
        typer.secho("no runs found under results/", fg=typer.colors.YELLOW)

    rd = results_dir()
    rd.mkdir(parents=True, exist_ok=True)
    target = out or (rd / "REPORT.md")

    manifest_lines = list(read_jsonl(manifest_path())) if manifest_path().exists() else []

    body = [
        "# Re-derivation Leakage — results",
        "",
        f"Generated from {len(runs)} run(s); manifest has {len(manifest_lines)} line(s).",
        "",
        "## Phase 0, Days 1-2 — open-unlearning reproduction",
        "",
        repro_table(runs),
        "",
        "> `forget_quality` is a KS p-value spanning ~200 orders of magnitude across",
        "> methods. It is reported, never gated. See docs/02_repro_targets.md.",
        "",
        "## Phase 0, Days 3-5 — conditions",
        "",
        markdown_table(runs),
        "",
        "**Read the laundering rate against `n_recovered`, not on its own.** A method",
        "that recovers nothing has an undefined rate reported as 0.0.",
        "",
        "The pre-registered gate is `C3 - C1`, not `C2`. See docs/00_preregistration.md.",
        "",
    ]

    if figures:
        figs = _figures(runs, rd)
        if figs:
            body.append("## Figures")
            body.append("")
            body.extend(f"![{p.stem}]({p.name})" for p in figs)
            body.append("")
        else:
            body.append("_figures skipped: matplotlib not installed_\n")

    target.write_text("\n".join(body), encoding="utf-8")
    typer.echo(f"wrote {target}")
