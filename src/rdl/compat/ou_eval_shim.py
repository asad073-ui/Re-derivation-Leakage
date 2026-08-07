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

    # `python <script>` puts the script's directory at the head of sys.path.
    # `runpy.run_path` does not, and open-unlearning's eval.py needs it.
    sys.path.insert(0, str(script.resolve().parent))
    sys.argv = [str(script), *args[1:]]

    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
