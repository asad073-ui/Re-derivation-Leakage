"""`rdl env-check` — the first thing run in every Colab session.

Prints the hardware profile, resolved package versions, HF token presence and
gated-repo access, free disk, free RAM, and whether the pinned submodule SHA matches.

Every one of those is a thing that has cost someone a GPU session. Checking them takes
four seconds and turns "the eval died at hour six" into "the eval never started".
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import typer

from ..hardware import detect
from ..paths import open_unlearning_dir, repo_root

__all__ = ["collect_env", "env_check"]

_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "tokenizers",
    "numpy",
    "pandas",
    "omegaconf",
    "pydantic",
    "huggingface_hub",
)


def _versions() -> dict[str, str]:
    import importlib

    out: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            mod = importlib.import_module(name)
            out[name] = str(getattr(mod, "__version__", "unknown"))
        except ImportError:
            out[name] = "MISSING"
    return out


def _free_ram_gb() -> float | None:
    try:
        import psutil  # type: ignore

        return round(psutil.virtual_memory().available / 1e9, 1)
    except ImportError:
        pass
    try:
        meminfo = Path("/proc/meminfo").read_text()
        for line in meminfo.splitlines():
            if line.startswith("MemAvailable:"):
                return round(int(line.split()[1]) * 1024 / 1e9, 1)
    except OSError:
        pass
    return None


def _submodule_sha(root: Path) -> dict:
    ou = open_unlearning_dir(root)
    info: dict = {"path": str(ou), "present": ou.exists()}
    if not ou.exists():
        info["hint"] = "git submodule update --init --recursive"
        return info
    try:
        head = subprocess.run(
            ["git", "-C", str(ou), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        info["head_sha"] = head.stdout.strip() or None
        # What the superproject expects the submodule to be pinned at.
        pinned = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "HEAD", "third_party/open-unlearning"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        parts = pinned.stdout.split()
        info["pinned_sha"] = parts[2] if len(parts) >= 3 else None
        if info.get("head_sha") and info.get("pinned_sha"):
            info["matches_pin"] = info["head_sha"] == info["pinned_sha"]
    except (OSError, subprocess.SubprocessError) as exc:
        info["error"] = str(exc)
    return info


def _hf_status(token: str | None, check_gated: bool) -> dict:
    out: dict = {"token_present": bool(token)}
    if not token:
        out["hint"] = "set HF_TOKEN (Colab: Secrets -> HF_TOKEN, notebook access on)"
        return out
    try:
        from huggingface_hub import HfApi, whoami

        out["user"] = whoami(token=token).get("name")
        if check_gated:
            api = HfApi(token=token)
            for repo in (
                "meta-llama/Llama-3.2-1B-Instruct",
                "open-unlearning/tofu_Llama-3.2-1B-Instruct_full",
                "open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90",
            ):
                try:
                    api.model_info(repo)
                    out[repo] = "ok"
                except Exception as exc:
                    out[repo] = f"BLOCKED ({type(exc).__name__})"
            if str(out.get("meta-llama/Llama-3.2-1B-Instruct", "")).startswith("BLOCKED"):
                out["blocker"] = (
                    "Accept the Llama 3.2 community licence with the SAME account as this "
                    "token: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct "
                    "Do this on day 1 — approval is usually instant but it is a hard blocker."
                )
    except ImportError:
        out["error"] = "huggingface_hub not installed"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def collect_env(*, check_hf: bool = True, root: Path | None = None) -> dict:
    root = root or repo_root()
    hw = detect()
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    try:
        free_disk = round(shutil.disk_usage(str(root)).free / 1e9, 1)
    except OSError:
        free_disk = 0.0

    return {
        "repo_root": str(root),
        "hardware": hw.to_dict(),
        "hardware_summary": hw.summary(),
        "packages": _versions(),
        "free_disk_gb": free_disk,
        "free_ram_gb": _free_ram_gb(),
        "env": {
            "HF_HOME": os.environ.get("HF_HOME"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        },
        "huggingface": _hf_status(token, check_hf),
        "submodule": _submodule_sha(root),
    }


def env_check(
    json_out: bool = typer.Option(False, "--json", help="emit raw JSON"),
    no_hf: bool = typer.Option(False, "--no-hf", help="skip Hub calls (offline)"),
    write_versions: Path | None = typer.Option(
        None,
        "--write-versions",
        help="append the resolved package versions to this file " "(use docs/02_repro_targets.md)",
    ),
) -> None:
    """Print the environment. Run this first in every session."""
    info = collect_env(check_hf=not no_hf)

    if json_out:
        typer.echo(json.dumps(info, indent=2, default=str))
    else:
        hw = info["hardware"]
        typer.echo(f"repo            {info['repo_root']}")
        typer.echo(f"hardware        {info['hardware_summary']}")
        typer.echo(f"disk free       {info['free_disk_gb']} GB")
        typer.echo(f"ram free        {info['free_ram_gb']} GB")
        typer.echo("packages")
        for k, v in info["packages"].items():
            typer.echo(f"  {k:<18} {v}")
        typer.echo("huggingface")
        for k, v in info["huggingface"].items():
            typer.echo(f"  {k:<18} {v}")
        typer.echo("submodule")
        for k, v in info["submodule"].items():
            typer.echo(f"  {k:<18} {v}")

        if hw["device"] == "cuda" and not hw["supports_bf16"]:
            typer.secho(
                "\nT4-class device: no bf16, no FlashAttention-2.\n"
                "  eval  -> float16 + sdpa (handled automatically)\n"
                "  train -> REFUSED. GA/NPO in fp16 NaNs silently. Use Ampere or newer.",
                fg=typer.colors.YELLOW,
            )

    if write_versions:
        import datetime as _dt

        stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [
            "",
            f"### Resolved versions — {stamp}",
            "",
            f"- device: `{info['hardware_summary']}`",
            "",
            "| package | version |",
            "|---|---|",
            *[f"| `{k}` | `{v}` |" for k, v in info["packages"].items()],
            "",
            f"- open-unlearning SHA: `{info['submodule'].get('head_sha')}`",
            "",
        ]
        write_versions.parent.mkdir(parents=True, exist_ok=True)
        with write_versions.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        typer.echo(f"\nappended resolved versions to {write_versions}")
