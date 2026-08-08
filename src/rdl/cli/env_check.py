"""`rdl env-check` — the first thing run in every GPU session.

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

from ..config import ConfigError, load_env
from ..hardware import check_env_against_hardware, detect
from ..models.registry import KNOWN_MODELS
from ..paths import open_unlearning_dir, repo_root

__all__ = ["collect_env", "env_check", "strict_blockers"]

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
    """Hub reachability for every checkpoint the run needs, AT ITS PINNED REVISION.

    Checking the repo without the revision is a weaker test than it looks: a pin that
    points at a commit which was force-removed resolves as a healthy repo here and 404s
    inside `from_pretrained` later. Agent A's and agent B's NPO checkpoints are included
    — they were missing, which meant the one repo the whole experiment turns on was the
    one nobody verified.
    """
    out: dict = {"token_present": bool(token)}
    if not token:
        out["hint"] = "set HF_TOKEN (Colab: Secrets -> HF_TOKEN, notebook access on)"
        return out
    try:
        from huggingface_hub import HfApi, whoami

        out["user"] = whoami(token=token).get("name")
        if check_gated:
            api = HfApi(token=token)
            # (repo, pinned revision or None). The gated base model first: it is the
            # tokenizer/chat template every eval goes through.
            targets: list[tuple[str, str | None]] = [("meta-llama/Llama-3.2-1B-Instruct", None)]
            targets += [(e.repo_id, e.revision) for e in KNOWN_MODELS.values() if e.revision]
            for repo, revision in targets:
                try:
                    api.model_info(repo, revision=revision)
                    out[repo] = f"ok @ {revision[:12]}" if revision else "ok"
                except Exception as exc:
                    at = f" @ {revision[:12]}" if revision else ""
                    out[repo] = f"BLOCKED{at} ({type(exc).__name__})"
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


def _retain_logs_status(root: Path) -> dict:
    """`forget_quality` is computed against the published retain-model eval log.

    Without it the metric is silently unavailable rather than loudly missing, so its
    presence is part of the preflight rather than something the eval discovers.
    """
    p = (
        open_unlearning_dir(root)
        / "saves"
        / "eval"
        / "tofu_Llama-3.2-1B-Instruct_retain90"
        / "TOFU_EVAL.json"
    )
    info: dict = {"path": str(p), "present": p.exists()}
    if not p.exists():
        info["hint"] = "cd third_party/open-unlearning && python setup_data.py --eval_logs"
    return info


def strict_blockers(info: dict) -> list[str]:
    """Everything that must hold before a GPU hour is spent. Pure, so it is testable.

    `env-check` prints all of these regardless; under `--strict` they also decide the
    exit code. The split exists because the diagnostic form is useful on a laptop with
    no token and no submodule, where none of it is a problem.
    """
    out: list[str] = []
    profile = info.get("env_profile") or {}

    if profile.get("requested"):
        if profile.get("error"):
            out.append(f"env profile could not be loaded: {profile['error']}")
        for p in profile.get("problems") or []:
            out.append(p)

    hf = info.get("huggingface") or {}
    if not hf.get("token_present"):
        out.append(
            "no HF_TOKEN in the environment. The TOFU checkpoints and the gated Llama "
            "tokenizer both need one."
        )
    if hf.get("error"):
        out.append(f"huggingface: {hf['error']}")
    for key, value in hf.items():
        if isinstance(value, str) and value.startswith("BLOCKED"):
            out.append(f"{key} is not reachable with this token: {value}")

    for name, version in (info.get("packages") or {}).items():
        if version == "MISSING":
            out.append(f"required package `{name}` is not installed")

    sub = info.get("submodule") or {}
    if not sub.get("present"):
        out.append("third_party/open-unlearning is missing: git submodule update --init")
    elif sub.get("matches_pin") is False:
        out.append(
            f"open-unlearning is at {sub.get('head_sha')}, but this repo pins "
            f"{sub.get('pinned_sha')}. Every published number is quoted against the pin."
        )

    logs = info.get("retain_logs") or {}
    if not logs.get("present"):
        out.append(
            f"retain-model eval logs missing at {logs.get('path')}. `forget_quality` "
            "cannot be computed without them: " + str(logs.get("hint", ""))
        )
    return out


def collect_env(
    *, check_hf: bool = True, root: Path | None = None, env_profile: str | None = None
) -> dict:
    root = root or repo_root()
    hw = detect()

    profile: dict = {"requested": env_profile}
    if env_profile:
        try:
            env_cfg = load_env(env_profile, root)
        except ConfigError as exc:
            profile["error"] = str(exc)
        else:
            problems = check_env_against_hardware(env_cfg, hw)
            profile["matches_hardware"] = not problems
            profile["problems"] = problems
            profile["hf_home"] = env_cfg.hf_home
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
        "retain_logs": _retain_logs_status(root),
        "env_profile": profile,
    }


def env_check(
    json_out: bool = typer.Option(False, "--json", help="emit raw JSON"),
    no_hf: bool = typer.Option(False, "--no-hf", help="skip Hub calls (offline)"),
    environment: str | None = typer.Option(
        None,
        "--env",
        help="check a configs/env profile against this box, e.g. vast_rtx3090. "
        "Exits non-zero if the machine does not match it.",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        help="full preflight: also fail on a missing HF token, a blocked Llama licence, "
        "any checkpoint unreachable at its pinned revision, a missing or mismatched "
        "submodule, a missing package, or absent retain eval logs. Run this before "
        "spending a GPU hour.",
    ),
    write_versions: Path | None = typer.Option(
        None,
        "--write-versions",
        help="append the resolved package versions to this file (use docs/02_repro_targets.md)",
    ),
) -> None:
    """Print the environment. Run this first in every session."""
    info = collect_env(check_hf=not no_hf, env_profile=environment)

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
        typer.echo("retain eval logs")
        for k, v in info["retain_logs"].items():
            typer.echo(f"  {k:<18} {v}")
        if environment:
            typer.echo("env profile")
            for k, v in info["env_profile"].items():
                typer.echo(f"  {k:<18} {v}")

        # Hardware capability and package availability are separate facts, and the pair
        # is what decides the attention implementation. Printing only one of them is how
        # a run starts on an Ampere box that cannot import flash_attn.
        typer.echo(
            f"\nfa2_hardware={hw['supports_flash_attn2']}  "
            f"fa2_installed={hw['flash_attn_installed']}  "
            f"recommended_attn={hw['recommended_attn']}"
        )

        if hw["device"] == "cuda" and not hw["supports_bf16"]:
            typer.secho(
                "\nT4-class device: no bf16, no FlashAttention-2.\n"
                "  eval  -> float16 + sdpa (handled automatically)\n"
                "  train -> REFUSED. GA/NPO in fp16 NaNs silently. Use Ampere or newer.",
                fg=typer.colors.YELLOW,
            )
        elif (
            hw["device"] == "cuda" and hw["supports_flash_attn2"] and not hw["flash_attn_installed"]
        ):
            typer.secho(
                "\nAmpere+ device WITHOUT the flash_attn package.\n"
                "  This is not an error: attention resolves to sdpa and the choice is\n"
                "  recorded in every report. To build FA2 instead (needs nvcc, ~20 min):\n"
                "    INSTALL_FLASH_ATTN=1 bash scripts/01_bootstrap_openunlearning.sh\n"
                "  Never mix FA2 and SDPA runs inside one comparison.",
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

    # `--strict` is the full preflight: everything that would otherwise be discovered
    # an hour in, with the checkpoint already downloaded. It subsumes the profile check.
    if strict:
        blockers = strict_blockers(info)
        if blockers:
            typer.secho("\nPREFLIGHT FAILED — do not start a GPU run:", fg=typer.colors.RED)
            for b in blockers:
                typer.secho(f"  - {b}", fg=typer.colors.RED)
            raise typer.Exit(code=1)
        typer.secho("\nPREFLIGHT OK — every checkpoint, package and pin verified.", fg="green")
        return

    # Non-zero when an explicitly requested profile does not match: `scripts/00_env_check.sh`
    # runs under `set -e`, so this is what stops a session on the wrong instance.
    profile = info["env_profile"]
    if environment and not profile.get("matches_hardware", False):
        typer.secho(f"\nenv profile '{environment}' does NOT match this box:", fg=typer.colors.RED)
        for p in profile.get("problems", []) or [profile.get("error", "unknown error")]:
            typer.secho(f"  - {p}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    if environment:
        typer.secho(f"\nenv profile '{environment}': OK", fg=typer.colors.GREEN)
