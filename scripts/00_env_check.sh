#!/usr/bin/env bash
# First thing run in every session. Four seconds now saves a GPU hour later.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== rdl env-check ==="
python -m rdl.cli env-check "$@"

echo
echo "=== git ==="
git rev-parse --short HEAD 2>/dev/null || echo "not a git repo"
git submodule status 2>/dev/null || echo "no submodules"
