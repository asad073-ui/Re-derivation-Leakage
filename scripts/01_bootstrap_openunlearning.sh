#!/usr/bin/env bash
# Bootstrap the open-unlearning submodule.
#
# Upstream README does:
#     conda create -n unlearning python=3.11
#     pip install ".[lm-eval]"
#     pip install --no-build-isolation flash-attn==2.6.3   # <-- SKIPPED ON T4
#     python setup_data.py --eval
#
# flash-attn requires SM80+. A T4 is SM75. Installing it there either fails to build or
# fails at runtime. This script detects the device and skips it automatically.
#
# NEEDS NETWORK.
set -euo pipefail
cd "$(dirname "$0")/.."

OU="third_party/open-unlearning"

echo "=== submodule ==="
git submodule update --init --recursive
test -d "$OU" || { echo "ERROR: $OU missing — is .gitmodules committed?"; exit 1; }

SHA="$(git -C "$OU" rev-parse HEAD)"
echo "open-unlearning SHA: $SHA"
echo "  -> record this in docs/02_repro_targets.md"

echo
echo "=== install ==="
pip install -q -e "$OU[lm-eval]"

# --- flash-attn decision -------------------------------------------------------------
CC="$(python - <<'PY'
try:
    import torch
    print("%d%d" % torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 0)
except Exception:
    print(0)
PY
)"

if [ "$CC" -ge 80 ] 2>/dev/null; then
    echo "compute capability $CC >= 80: FlashAttention-2 is supported."
    echo "  install it manually if you want it: pip install --no-build-isolation flash-attn==2.6.3"
else
    echo "compute capability $CC < 80 (or no GPU): SKIPPING flash-attn. This is correct on a T4."
fi

echo
echo "=== data + published eval logs ==="
# Downloads TOFU/MUSE data AND the published evaluation logs, including the retain-model
# logs required for forget_quality and the reference *_SUMMARY.json files. Those logs
# are what REPO_SPEC 7.4 fallback 1 validates the metric code against.
( cd "$OU" && python setup_data.py --eval )

echo
echo "=== resolved versions ==="
pip list 2>/dev/null | grep -Ei "^(torch|transformers|datasets|accelerate|peft|trl|numpy|tokenizers) " || true
echo "  -> transcribe these into docs/02_repro_targets.md. Version drift is the single"
echo "     most likely cause of a failed reproduction."
