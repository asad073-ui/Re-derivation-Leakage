"""Run an open-unlearning entry point with the fp32-logits shim attached.

    python -m rdl.compat.ou_eval_shim src/eval.py <hydra overrides...>

is a drop-in replacement for

    python src/eval.py <hydra overrides...>

and the bridge emits the former so that the shim is VISIBLE in the command string that
every report records. A reviewer re-running the recorded command gets the same
numerical environment; a reviewer running the bare upstream command gets the
`Got unsupported ScalarType BFloat16` crash, which is the honest state of upstream at
the pinned SHA. See `rdl.compat.fp32_logits` for why the shim exists.

`sitecustomize.py` on `PYTHONPATH` would attach the same patch without changing the
command, and that is exactly why it was not used: a run whose provenance depends on an
environment variable looks identical to a run without it.

The script is executed the way `python <script>` would execute it — `__name__` set to
`"__main__"`, its own directory first on `sys.path`, `sys.argv` rewritten to start at
the script — because Hydra resolves `config_path` relative to the decorated function's
module file, and open-unlearning's `src/eval.py` imports its siblings (`trainer`,
`evals`, `model`) as top-level modules.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from . import fp32_logits

__all__ = ["main"]


def _seed_from_argv(argv: list[str]) -> int | None:
    """The `seed=N` Hydra override, if the caller emitted one.

    Read from argv rather than taken as a separate flag so the subprocess can never be
    seeded differently from the value the recorded command shows. `build_eval_command`
    always emits `seed=`; a missing one means something built the command by hand.
    """
    for arg in argv:
        if arg.startswith("seed="):
            try:
                return int(arg.split("=", 1)[1])
            except ValueError:
                return None
    return None


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            "usage: python -m rdl.compat.ou_eval_shim <script.py> [args...]",
            file=sys.stderr,
        )
        return 2

    script = Path(args[0])
    if not script.exists():
        print(f"rdl-shim: no such script: {script}", file=sys.stderr)
        return 2

    record = fp32_logits.install()
    # Printed, not logged: this line lands in the captured subprocess output that the
    # run report keeps, so the shim's presence is recoverable from the log alone.
    print(f"rdl-shim: fp32_logits active -> {record.get('target')}", flush=True)

    # Determinism has to be established HERE, in the process that actually runs the
    # kernels. `run-repro` calls set_all_seeds() in the PARENT, and the report then
    # recorded `deterministic_algorithms: true` — but torch settings are process-local,
    # so none of `use_deterministic_algorithms`, `cudnn.deterministic` or the torch RNG
    # seeds ever reached this subprocess. (Only the two env vars, PYTHONHASHSEED and
    # CUBLAS_WORKSPACE_CONFIG, are inherited.) Upstream's eval.py does call its own
    # seed_everything, so the run was seeded; it was not running deterministic kernels
    # while the report said it was. That gap is the reason this block exists.
    seed = _seed_from_argv(args[1:])
    if seed is not None:
        from ..seeding import set_all_seeds

        seed_record = set_all_seeds(seed)
        print(
            f"rdl-shim: subprocess determinism seed={seed} "
            f"deterministic_algorithms={seed_record.deterministic_algorithms} "
            f"cudnn_deterministic={seed_record.cudnn_deterministic}",
            flush=True,
        )
    else:
        print("rdl-shim: no seed= override found; subprocess determinism NOT set", flush=True)

    # `python <script>` puts the script's directory at the head of sys.path.
    # `runpy.run_path` does not, and open-unlearning's eval.py needs it.
    sys.path.insert(0, str(script.resolve().parent))
    sys.argv = [str(script), *args[1:]]

    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
