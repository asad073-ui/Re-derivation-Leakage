#!/usr/bin/env bash
# Days 1-2: the reproduction gate.
#
#   ENV_NAME=vast_rtx3090 bash scripts/02_repro_tofu_npo_forget10.sh
#
# TWO RUNS PER CHECKPOINT, AND THEY ANSWER DIFFERENT QUESTIONS.
#
#   Run A  batch_size=32, seed=0   PUBLISHED PARITY — upstream's own eval defaults
#                                  (configs/eval/tofu.yaml, configs/eval.yaml). This is
#                                  the trust gate: it is the only run whose miss means
#                                  "our install is wrong" rather than "we changed a
#                                  setting". `make-report` requires it.
#   Run B  batch_size=1, seed=42   The pre-registered deterministic protocol
#                                  (docs/00_preregistration.md §7). Reported, and NOT a
#                                  substitute for A.
#
# If A passes and B misses, the install is fine and batching moved the metric — which is
# a finding about the protocol, recorded as such. If A misses, stop: bisect before
# reading anything downstream.
#
# Order matters within each run: `full` goes FIRST because it is a checkpoint that
# definitely exists, which isolates "is my install correct" from "does the unlearned
# checkpoint exist".
#
# EVAL ONLY. Never train here — not even on the 3090. 24 GB of bf16 makes a GA/NPO
# objective numerically sound; it does not make one card equal to upstream's 2x L40S
# under ZeRO-3, so training here produces a NEW number rather than a reproduction.
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_NAME="${ENV_NAME:-vast_rtx3090}"
AGENT_B_CKPT="${AGENT_B_CKPT:-open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10}"

echo "=== 0. preflight ==="
# --strict, and WITH the env profile. Without either, this step printed a diagnostic and
# always succeeded: the wrong GPU, a missing token, a moved checkpoint revision and an
# unpinned submodule all sailed past it and surfaced an hour later.
python -m rdl.cli env-check --env "$ENV_NAME" --strict

echo
echo "=== 1. does the NPO forget10 checkpoint exist? ==="
python -m rdl.cli discover-checkpoints || echo "(discovery failed; continuing with the full-model gate)"

echo
echo "=================================================================="
echo "  RUN A — PUBLISHED PARITY (batch_size=32, seed=0). THE TRUST GATE."
echo "=================================================================="

echo
echo "--- A1. SANITY: the \`full\` target model ---"
echo "    targets: model_utility 0.60, forget_truth_ratio 0.48"
python -m rdl.cli run-repro --target full --env "$ENV_NAME" --batch-size 32 --seed 0 "$@"

echo
echo "--- A2. THE GATE: NPO forget10 (agent A) ---"
echo "    targets: model_utility 0.46, forget_truth_ratio 0.70"
echo "    forget_quality 0.02 is REPORTED, NOT GATED (KS p-value; ~200 orders of magnitude)"
if python -m rdl.cli run-repro --target npo_forget10 --env "$ENV_NAME" --batch-size 32 --seed 0 "$@"; then
    echo "REPRODUCED at published parity."
else
    echo
    echo "NOT REPRODUCED AT PARITY."
    echo
    echo "  This is the EXPECTED, RECORDED finding under \`study_mode: released_artifact\`"
    echo "  (configs/study_mode.yaml, ADR-0038/0039/0052): the released NPO forget10"
    echo "  checkpoint measures 0.43237 / 0.64140 against a documented 0.460 / 0.700 at"
    echo "  revision 94ed64eb, under two independent evaluation environments. Upstream"
    echo "  issue #199 is open. \`full\` and \`retain90\` DO reproduce, which is what"
    echo "  validates the evaluator."
    echo
    echo "  The script CONTINUES so agent B can be characterised: the condition grid"
    echo "  needs an individual forgetting number for BOTH checkpoints, and stopping here"
    echo "  left B unmeasured while make-report demanded its measurement."
    echo
    echo "  If this is a NEW mismatch rather than the recorded one, stop and bisect ONE"
    echo "  change at a time: chat template, padding_side, batch_size, dtype, attention"
    echo "  implementation (FA2 vs SDPA), transformers version. Log it in"
    echo "  docs/04_decisions.md and set study_mode back to published_reproduction."
fi

echo
echo "=================================================================="
echo "  RUN B — DETERMINISTIC PROTOCOL (batch_size=1, seed=42)."
echo "  Reported, not the trust gate. A miss here is a protocol finding."
echo "=================================================================="

python -m rdl.cli run-repro --target full --env "$ENV_NAME" --batch-size 1 --seed 42 "$@" || \
    echo "  (full missed at batch 1 — recorded; parity already passed)"
python -m rdl.cli run-repro --target npo_forget10 --env "$ENV_NAME" --batch-size 1 --seed 42 "$@" || \
    echo "  (npo_forget10 missed at batch 1 — recorded; parity already passed)"

echo
echo "=== 4. MEASURE agent B's independent checkpoint (no published target) ==="
# Measured at parity settings so it is comparable with agent A's parity run.
python -m rdl.cli run-repro \
    --model-path "$AGENT_B_CKPT" \
    --measure-only \
    --checkpoint-label agent_b_independent \
    --env "$ENV_NAME" \
    --batch-size 32 \
    --seed 0 \
    "$@"

echo
echo "Days 1-2 complete. \`rdl make-report\` requires: both gated targets PASSED AT"
echo "PARITY, plus agent B's measurement, before it will pass a condition grid."
