#!/usr/bin/env bash
# Days 1-2: the reproduction gate.
#
# Order matters. `full` runs FIRST because it is a checkpoint that definitely exists,
# which isolates "is my install correct" from "does the unlearned checkpoint exist".
#
# EVAL ONLY. Never train here — see requirements-gpu-t4.txt.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 0. environment ==="
python -m rdl.cli env-check

echo
echo "=== 1. does the NPO forget10 checkpoint exist? ==="
python -m rdl.cli discover-checkpoints || echo "(discovery failed; continuing with the full-model gate)"

echo
echo "=== 2. SANITY GATE: the `full` target model ==="
echo "    targets: model_utility 0.60, forget_truth_ratio 0.48"
python -m rdl.cli run-repro --target full "$@"

echo
echo "=== 3. THE Day 1-2 GATE: NPO forget10 ==="
echo "    targets: model_utility 0.46, forget_truth_ratio 0.70"
echo "    forget_quality 0.02 is REPORTED, NOT GATED (KS p-value; ~200 orders of magnitude)"
if python -m rdl.cli run-repro --target npo_forget10 "$@"; then
    echo "REPRODUCED"
else
    echo
    echo "NOT REPRODUCED. REPO_SPEC 7.4 applies."
    echo "  If the checkpoint is absent      -> fallback 1: gate on \`full\` only and validate"
    echo "                                      the metric code against the published eval logs."
    echo "  If the metrics missed tolerance  -> fallback 2: bisect ONE change at a time in this"
    echo "                                      order: chat template, padding_side, batch_size,"
    echo "                                      dtype, transformers version."
    echo "  Log every attempt in docs/04_decisions.md."
    exit 1
fi
