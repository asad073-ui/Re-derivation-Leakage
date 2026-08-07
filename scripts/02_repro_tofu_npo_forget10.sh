#!/usr/bin/env bash
# Days 1-2: the reproduction gate.
#
# Order matters. `full` runs FIRST because it is a checkpoint that definitely exists,
# which isolates "is my install correct" from "does the unlearned checkpoint exist".
#
# EVAL ONLY. Never train here — not even on the 3090. 24 GB of bf16 makes a GA/NPO
# objective numerically sound; it does not make one card equal to upstream's 2x L40S
# under ZeRO-3, so training here produces a NEW number rather than a reproduction.
#
#   ENV_NAME=vast_rtx3090 bash scripts/02_repro_tofu_npo_forget10.sh --batch-size 1
#
# Step 4 measures AGENT B's checkpoint. It is a measurement, not a gate: B was unlearned
# at lr2e-05 / beta0.5 and has no published row, so comparing it to agent A's 0.46 / 0.70
# would manufacture a verdict out of a hyperparameter difference. `make-report` REQUIRES
# this measurement before any C3D/C3C number is reportable.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="${ENV_NAME:-vast_rtx3090}"
AGENT_B_CKPT="${AGENT_B_CKPT:-open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10}"

echo "=== 0. environment ==="
python -m rdl.cli env-check

echo
echo "=== 1. does the NPO forget10 checkpoint exist? ==="
python -m rdl.cli discover-checkpoints || echo "(discovery failed; continuing with the full-model gate)"

echo
echo "=== 2. SANITY GATE: the \`full\` target model ==="
echo "    targets: model_utility 0.60, forget_truth_ratio 0.48"
python -m rdl.cli run-repro --target full --env "$ENV_NAME" "$@"

echo
echo "=== 3. THE Day 1-2 GATE: NPO forget10 (agent A) ==="
echo "    targets: model_utility 0.46, forget_truth_ratio 0.70"
echo "    forget_quality 0.02 is REPORTED, NOT GATED (KS p-value; ~200 orders of magnitude)"
if python -m rdl.cli run-repro --target npo_forget10 --env "$ENV_NAME" "$@"; then
    echo "REPRODUCED"
else
    echo
    echo "NOT REPRODUCED. REPO_SPEC 7.4 applies."
    echo "  If the checkpoint is absent      -> fallback 1: gate on \`full\` only and validate"
    echo "                                      the metric code against the published eval logs."
    echo "  If the metrics missed tolerance  -> fallback 2: bisect ONE change at a time in this"
    echo "                                      order: chat template, padding_side, batch_size,"
    echo "                                      dtype, attention implementation (FA2 vs SDPA),"
    echo "                                      transformers version."
    echo "  Log every attempt in docs/04_decisions.md."
    exit 1
fi

echo
echo "=== 4. MEASURE agent B's independent checkpoint (no published target) ==="
python -m rdl.cli run-repro \
    --model-path "$AGENT_B_CKPT" \
    --measure-only \
    --checkpoint-label agent_b_independent \
    --env "$ENV_NAME" \
    "$@"

echo
echo "Days 1-2 complete. \`rdl make-report\` requires all three of the above before it"
echo "will pass a condition grid."
