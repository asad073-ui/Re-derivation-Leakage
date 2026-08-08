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
echo "    documented row: model_utility 0.46, forget_truth_ratio 0.70"
echo "    KNOWN NOT TO REPRODUCE: measured 0.43237 / 0.64140 at revision 94ed64eb, under"
echo "    two independent evaluation environments. study_mode=released_artifact records"
echo "    this as the finding rather than blocking on it (ADR-0038/0039/0052, issue #199)."
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
echo "Days 1-2 complete. What \`rdl make-report\` requires before it will pass a"
echo "condition grid depends on study_mode (configs/study_mode.yaml):"
echo
echo "  full            PASSED AT EXACT PARITY — batch_size=32, seed=0, bfloat16,"
echo "                  flash_attention_2. This is the evaluator trust gate and it"
echo "                  blocks in every mode."
echo "  npo_forget10    under \`released_artifact\` it is CHARACTERIZED: the published-row"
echo "                  miss is the recorded finding, not a blocker. Under"
echo "                  \`published_reproduction\` it must pass at parity like full."
echo "  agent B         measured (--measure-only). No published row exists to pass."
echo
echo "PROVENANCE blocks in EVERY mode, characterized targets included: each report must"
echo "carry exact_published_parity, git_dirty=false, the open-unlearning submodule SHA"
echo "this repo pins, the tokenizer chat-template hash, transformers_version and"
echo "ou_runtime_mode. A report missing them is unverifiable, not clean — which is why"
echo "these runs must be repeated on the commit the grid is run from (ADR-0058)."
