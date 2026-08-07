#!/usr/bin/env bash
# First thing run in every session. Four seconds now saves a GPU hour later.
set -euo pipefail
cd "$(dirname "$0")/.."

# ENV_NAME checks the profile the run will actually use against the detected hardware,
# and exits non-zero when they disagree. On a rented instance that is the difference
# between four seconds and a finished grid that says it ran on a 3090 and did not.
#
# STRICT=1 adds the full preflight: HF token, Llama licence, every checkpoint at its
# pinned revision, submodule pin, required packages, retain eval logs. Use it before
# spending a GPU hour:
#
#   ENV_NAME=vast_rtx3090 STRICT=1 bash scripts/00_env_check.sh
ENV_NAME="${ENV_NAME:-}"
STRICT="${STRICT:-0}"

ARGS=()
[ -n "$ENV_NAME" ] && ARGS+=(--env "$ENV_NAME")
[ "$STRICT" = "1" ] && ARGS+=(--strict)

echo "=== rdl env-check ==="
python -m rdl.cli env-check "${ARGS[@]+"${ARGS[@]}"}" "$@"

echo
echo "=== git ==="
git rev-parse --short HEAD 2>/dev/null || echo "not a git repo"
git submodule status 2>/dev/null || echo "no submodules"
