"""``rdl graph-detector-v4-3-freeze-judge-pins`` and ``-env-check`` — before any weight loads.

v4.3 shipped a judge roster with **empty revisions** and a runner that refuses them, which
was the right default and half a mechanism: nothing could resolve them. The roster's own
docstring named a command that did not exist. This is that command.

Only the JUDGES are pinned here. The encoder, its tokenizer and the NLI baseline are
already frozen by ``rdl graph-detector-v4-2-freeze-model-pins`` into
``DETECTOR_V4_2_MODEL_PINS.json``, and a second v4.3 command writing a second artifact for
the same three repositories would create two pins that can disagree — at which point the
question "which commit trained the checkpoint" has two answers and the manifest cannot say
which. v4.3 reuses the v4.2 artifact and records that it did.

What a pin is
-------------
A 40-character commit sha. Not a tag, not a branch, not a date. ``main`` moves, and two
runs a month apart under a moved tag are two different annotators with one name, both
satisfying "the revision is recorded".

``--resolve`` asks the Hub for each repository's current sha and needs network plus
``huggingface_hub``. Without either, pass the shas explicitly — they are printed by
``HfApi().model_info(repo).sha`` or shown on the repository's "Files and versions" page.
Either way the artifact is committed, which is what makes the pin a pre-registration.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import typer

from ..eval.detector_v4_3_judges import (
    LOCAL_JUDGE_ROSTER,
    PROMPT_VERSION,
    V4_3_PROTOCOL,
    LocalJudgePin,
)
from ..logging_utils import dumps_canonical
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_3_store import DEFAULT_OUT_DIR

__all__ = [
    "JUDGE_PINS_FILENAME",
    "JUDGE_PINS_SCHEMA",
    "detector_v4_3_env_check",
    "detector_v4_3_freeze_judge_pins",
    "load_judge_pins",
]

JUDGE_PINS_FILENAME = "DETECTOR_V4_3_LOCAL_JUDGE_PINS.json"
JUDGE_PINS_SCHEMA = "graph-detector-v4-3-local-judge-pins-v1"

# The v4.2 artifact this command deliberately does NOT duplicate.
MODEL_PINS_FILENAME = "DETECTOR_V4_2_MODEL_PINS.json"

# The box a reportable v4.3 GPU phase requires. Ampere because bfloat16 and bitsandbytes
# 8-bit both want it; 20 GiB because a 24B in 4-bit plus KV cache does not fit in less with
# any headroom.
REQUIRED_COMPUTE_CAPABILITY = (8, 6)
REQUIRED_VRAM_GIB = 20

# 150 was a guess made before anyone measured the downloads. The measured figures, from
# the Hub's own file metadata at the pinned revisions:
#
#   Qwen/Qwen3-14B                                 27.5 GiB (sharded safetensors)
#   mistralai/Mistral-Small-3.2-24B-Instruct-2506  44.7 GiB (sharded safetensors)
#   microsoft/deberta-v3-base + NLI baseline       < 1 GiB
#   three seed checkpoints + run artifacts         ~ 3 GiB
#                                                  --------
#                                                    76 GiB
#
# 100 keeps roughly 24 GiB of headroom over that. Note the Mistral repo ALSO carries a
# 44.7 GiB `consolidated.safetensors`, a duplicate of the shards in Mistral's own format:
# `from_pretrained` reads model.safetensors.index.json and never fetches it, but a bare
# `snapshot_download` of that repo would, and would need ~120 GiB for the judge alone.
REQUIRED_FREE_DISK_GIB = 100


def _looks_like_sha(value: str) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 40 and all(c in "0123456789abcdef" for c in text)


def detector_v4_3_freeze_judge_pins(
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, "--output-dir", "--out-dir"),
    resolve: bool = typer.Option(
        False, "--resolve", help="ask the Hub for each repository's current commit sha."
    ),
    judge_a_revision: str = typer.Option("", "--judge-a-revision", help="exact commit sha"),
    judge_b_revision: str = typer.Option("", "--judge-b-revision", help="exact commit sha"),
    refreeze: bool = typer.Option(
        False,
        "--refreeze",
        help="overwrite an existing pin file. Required once a pin exists, because a "
        "silently moved pin is how a labelling run changes annotator mid-flight.",
    ),
) -> None:
    """Freeze the two local judges' exact commit shas."""
    out = Path(out_dir)
    target = out / JUDGE_PINS_FILENAME
    supplied = {"A": judge_a_revision.strip(), "B": judge_b_revision.strip()}

    if resolve:
        try:
            from huggingface_hub import HfApi
        except ImportError as error:  # pragma: no cover - depends on the box
            raise typer.BadParameter(
                "--resolve needs huggingface_hub. Install it, or pass --judge-a-revision "
                "and --judge-b-revision explicitly."
            ) from error
        api = HfApi()
        for role, pin in LOCAL_JUDGE_ROSTER.items():
            if supplied.get(role):
                continue
            supplied[role] = str(api.model_info(pin.repo_id).sha)

    resolved: dict[str, LocalJudgePin] = {}
    for role, pin in LOCAL_JUDGE_ROSTER.items():
        revision = supplied.get(role, "")
        if not _looks_like_sha(revision):
            raise typer.BadParameter(
                f"judge {role} ({pin.repo_id}) resolved to {revision!r}, which is not a "
                "40-character commit sha. A tag can move; a labelling run under a moved "
                "tag is a different annotator with the same name. Pass --resolve on a "
                f"networked box, or --judge-{role.lower()}-revision explicitly."
            )
        resolved[role] = LocalJudgePin(**{**pin.__dict__, "revision": revision})

    if target.exists() and not refreeze:
        previous = json.loads(target.read_text(encoding="utf-8")).get("judges", {})
        changed = [
            role
            for role, pin in resolved.items()
            if str(previous.get(role, {}).get("revision", "")) != pin.revision
        ]
        if changed:
            raise typer.BadParameter(
                f"{target} already pins {changed} at different commits. Pass --refreeze "
                "only if you intend to restart the reportable labelling run: labels "
                "produced under the old pin describe a different annotator and cannot be "
                "pooled with labels produced under the new one."
            )

    model_pins = out.parent / "detector_v4_2" / MODEL_PINS_FILENAME
    payload = {
        "schema": JUDGE_PINS_SCHEMA,
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "protocol": V4_3_PROTOCOL,
        "prompt_version": PROMPT_VERSION,
        "judges": {role: pin.to_dict() for role, pin in sorted(resolved.items())},
        "encoder_pins": {
            "artifact": MODEL_PINS_FILENAME,
            "resolved_path": str(model_pins),
            "present": model_pins.exists(),
            "why_not_duplicated": (
                "microsoft/deberta-v3-base, its tokenizer and cross-encoder/nli-deberta-v3-base "
                "are already frozen by `rdl graph-detector-v4-2-freeze-model-pins`. A second "
                "v4.3 artifact for the same three repositories could disagree with the first, "
                "and 'which commit trained the checkpoint' would then have two answers."
            ),
        },
        "why": (
            "everything that changes an output token: repository, exact commit, quantizer, "
            "compute dtype, chat template, thinking mode and generation parameters. A "
            "quantization change is a model change -- 4-bit and 8-bit of the same weights "
            "are different annotators -- so the quantizer is pinned, not passed at runtime."
        ),
    }
    atomic_json(target, payload)
    typer.echo(
        dumps_canonical(
            {
                "wrote": str(target),
                "judges": {role: pin.revision for role, pin in sorted(resolved.items())},
                "encoder_pins_present": model_pins.exists(),
            }
        )
    )


def load_judge_pins(path: Path, role: str) -> LocalJudgePin:
    """Read one judge's frozen pin. Refuses an unpinned or unknown artifact.

    The runner calls this instead of reading :data:`LOCAL_JUDGE_ROSTER` directly, so the
    revision a run used came from a committed file rather than from source — which is what
    makes it checkable after the fact.
    """
    path = Path(path)
    if not path.exists():
        raise typer.BadParameter(
            f"{path} is absent. Run `rdl graph-detector-v4-3-freeze-judge-pins --resolve` "
            "on a networked box first; a reportable run may not resolve its own annotator."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema")) != JUDGE_PINS_SCHEMA:
        raise typer.BadParameter(f"{path} carries schema {payload.get('schema')!r}")
    row = (payload.get("judges") or {}).get(role)
    if not row:
        raise typer.BadParameter(f"{path} pins no judge {role!r}")
    base = LOCAL_JUDGE_ROSTER[role]
    generation = row.get("generation", {})
    pin = LocalJudgePin(
        role=role,
        repo_id=str(row["repo_id"]),
        revision=str(row.get("revision", "")),
        quantization=str(row.get("quantization", base.quantization)),
        compute_dtype=str(row.get("compute_dtype", base.compute_dtype)),
        chat_template=str(row.get("chat_template", base.chat_template)),
        thinking_mode=bool(row.get("thinking_mode", base.thinking_mode)),
        max_new_tokens=int(generation.get("max_new_tokens", base.max_new_tokens)),
        temperature=float(generation.get("temperature", base.temperature)),
        top_p=float(generation.get("top_p", base.top_p)),
        seed=int(generation.get("seed", base.seed)),
    )
    if pin.repo_id != base.repo_id:
        raise typer.BadParameter(
            f"{path} pins judge {role} to {pin.repo_id!r}, but the frozen roster names "
            f"{base.repo_id!r}. Changing the roster is a protocol amendment, not a pin."
        )
    if not pin.pinned:
        raise typer.BadParameter(
            f"{path} pins judge {role} at {pin.revision!r}, which is not a commit sha."
        )
    return pin


def detector_v4_3_env_check(
    pins: Path = typer.Option(
        DEFAULT_OUT_DIR / JUDGE_PINS_FILENAME, "--pins", help="the frozen judge pins."
    ),
    strict: bool = typer.Option(
        False, "--strict", help="exit non-zero on any failure. Use before a reportable phase."
    ),
) -> None:
    """Report whether this box can run the reportable v4.3 GPU phase.

    Checked before weights are downloaded, because every one of these failures otherwise
    surfaces forty minutes into a run with the clock billing.
    """
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    try:
        import torch

        cuda = torch.cuda.is_available()
        record("torch_installed", True, torch.__version__)
        record("cuda_available", cuda, str(cuda))
        if cuda:
            capability = torch.cuda.get_device_capability(0)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            record(
                "compute_capability",
                capability >= REQUIRED_COMPUTE_CAPABILITY,
                f"{capability[0]}.{capability[1]} (need >= 8.6 for bf16 + bitsandbytes 8-bit)",
            )
            record(
                "vram_gib", total >= REQUIRED_VRAM_GIB, f"{total:.1f} (need >= {REQUIRED_VRAM_GIB})"
            )
            record(
                "bf16_supported", torch.cuda.is_bf16_supported(), "torch.cuda.is_bf16_supported()"
            )
            record("device_name", True, torch.cuda.get_device_name(0))
    except ImportError:
        record("torch_installed", False, "torch is not importable")

    for module in ("transformers", "bitsandbytes", "accelerate"):
        try:
            loaded = __import__(module)
            record(f"{module}_installed", True, getattr(loaded, "__version__", "unknown"))
        except ImportError:
            record(f"{module}_installed", False, f"{module} is not importable")

    # mistral-common is required by Mistral-Small-3.2's documented tokenizer path. Reported
    # rather than assumed: the loader resolves its class from the checkpoint's own config,
    # so this check says whether the documented path is AVAILABLE, not whether it is used.
    try:
        import mistral_common

        record("mistral_common_installed", True, getattr(mistral_common, "__version__", "unknown"))
    except ImportError:
        record(
            "mistral_common_installed",
            False,
            "mistral-common is not importable; Mistral-Small-3.2's documented tokenizer "
            "path needs mistral-common>=1.6.2",
        )

    import shutil

    free = shutil.disk_usage(".").free / (1024**3)
    record(
        "free_disk_gib",
        free >= REQUIRED_FREE_DISK_GIB,
        f"{free:.1f} (need >= {REQUIRED_FREE_DISK_GIB})",
    )

    pins_path = Path(pins)
    if pins_path.exists():
        try:
            for role in sorted(LOCAL_JUDGE_ROSTER):
                pin = load_judge_pins(pins_path, role)
                record(f"judge_{role}_pinned", True, f"{pin.repo_id}@{pin.revision[:12]}")
        except typer.BadParameter as error:
            record("judge_pins", False, str(error)[:200])
    else:
        record("judge_pins", False, f"{pins_path} is absent")

    failures = [c for c in checks if not c["ok"]]
    typer.echo(
        dumps_canonical(
            {
                "checks": checks,
                "n_failures": len(failures),
                "ready_for_reportable_gpu_phase": not failures,
            }
        )
    )
    if strict and failures:
        raise typer.Exit(code=1)
