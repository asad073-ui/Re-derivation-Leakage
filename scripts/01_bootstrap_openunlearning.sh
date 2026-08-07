#!/usr/bin/env bash
# Bootstrap the open-unlearning submodule.
#
# Upstream README does:
#     conda create -n unlearning python=3.11
#     pip install ".[lm-eval]"
#     pip install --no-build-isolation flash-attn==2.6.3   # <-- SKIPPED ON T4
#     python setup_data.py --eval_logs
#
# flash-attn requires SM80+. A T4 is SM75. Installing it there either fails to build or
# fails at runtime. This script detects the device and skips it automatically.
#
# On an Ampere box (RTX 3090, SM 8.6) FlashAttention-2 is OPTIONAL, not required:
# hardware.py falls back to SDPA when the wheel is absent, so the eval runs either way.
# Set INSTALL_FLASH_ATTN=1 to build it here (needs nvcc in the image; ~20 min). The
# attention implementation is recorded in every report — never mix FA2 and SDPA numbers
# inside one comparison.
#
# THE FLAG IS `--eval_logs`, NOT `--eval`. At the pinned SHA, setup_data.py's argparse
# defines --eval_logs / --idk / --wmdp and nothing else, so `--eval` aborts the script
# with "unrecognized arguments" before a single byte is downloaded — and the retain-model
# logs it fetches are what forget_quality is computed against.
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

INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

if [ "$CC" -ge 80 ] 2>/dev/null; then
    echo "compute capability $CC >= 80: the SILICON supports FlashAttention-2."
    if python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('flash_attn') else 1)" 2>/dev/null; then
        echo "  flash_attn is already installed -> hardware.py will select flash_attention_2."
    elif [ "$INSTALL_FLASH_ATTN" = "1" ]; then
        if command -v nvcc >/dev/null 2>&1; then
            echo "  INSTALL_FLASH_ATTN=1 and nvcc present: building flash-attn==2.6.3 (~20 min)."
            pip install --no-build-isolation flash-attn==2.6.3
        else
            echo "  ERROR: INSTALL_FLASH_ATTN=1 but no nvcc on PATH. Use a CUDA *devel*"
            echo "         image, or leave it unset and run under SDPA."
            exit 1
        fi
    else
        echo "  flash_attn is NOT installed. This is FINE: hardware.py resolves attention"
        echo "  to sdpa and records it. To build it instead:"
        echo "     INSTALL_FLASH_ATTN=1 bash scripts/01_bootstrap_openunlearning.sh"
    fi
else
    echo "compute capability $CC < 80 (or no GPU): SKIPPING flash-attn. This is correct on a T4."
fi

echo
echo "=== data + published eval logs ==="
# Downloads the published evaluation logs, including the retain-model logs required for
# forget_quality and the reference *_SUMMARY.json files. Those logs are what REPO_SPEC
# 7.4 fallback 1 validates the metric code against.
( cd "$OU" && python setup_data.py --eval_logs )

# The retain logs must actually be on disk: `retain_logs_path` points at one, and
# without it forget_quality is silently unavailable rather than loudly missing.
RETAIN_LOGS="$OU/saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json"
if [ -f "$RETAIN_LOGS" ]; then
    echo "OK  retain logs present: $RETAIN_LOGS"
else
    echo "ERROR: $RETAIN_LOGS missing after setup_data.py --eval_logs."
    echo "  forget_quality cannot be computed without it. Do not proceed to the eval."
    exit 1
fi

echo
echo "=== resolved versions ==="
pip list 2>/dev/null | grep -Ei "^(torch|transformers|datasets|accelerate|peft|trl|numpy|tokenizers) " || true
echo "  -> transcribe these into docs/02_repro_targets.md. Version drift is the single"
echo "     most likely cause of a failed reproduction."
