#!/usr/bin/env bash
# First thing run in every session. Four seconds now saves a GPU hour later.
set -euo pipefail
cd "$(dirname "$0")/.."

# ENV_NAME checks the profile the run will actually use against the detected hardware,
# and exits non-zero when they disagree. On a rented instance that is the difference
# between four seconds and a finished grid that says it ran on a 3090 and did not.
ENV_NAME="${ENV_NAME:-}"

echo "=== rdl env-check ==="
if [ -n "$ENV_NAME" ]; then
    python -m rdl.cli env-check --env "$ENV_NAME" "$@"
else
    python -m rdl.cli env-check "$@"
fi

echo
echo "=== git ==="
git rev-parse --short HEAD 2>/dev/null || echo "not a git repo"
git submodule status 2>/dev/null || echo "no submodules"
