#!/usr/bin/env bash
#
# Drive the v4.4 GPU phases. Orchestration only.
#
# This script contains no detector logic and no judging logic. It calls `rdl` commands that
# already exist, in the order DETECTOR_V4_4_ANSWER_ATTEMPT_PROTOCOL.md fixes, and it checks
# after every phase that nothing frozen moved. It cannot change the rubric, the prompt
# version, the judge pins, the bundle, the panel or the smoke fixture, because it never
# writes to them -- and if some *other* thing does, `verify` fails and says which file.
#
# Usage:
#   scripts/v44_gpu_runs.sh verify-cpu   artifacts, hashes and the dry run. NO GPU needed.
#   scripts/v44_gpu_runs.sh verify-gpu   verify-cpu, then CUDA/bf16/VRAM/disk. Needs a GPU.
#   scripts/v44_gpu_runs.sh gpu1         the 60-row dual-judge smoke. Non-reportable.
#   scripts/v44_gpu_runs.sh compare      the smoke comparison a person has to read.
#   scripts/v44_gpu_runs.sh gpu2         the reportable blind passes + the blind-only report.
#   scripts/v44_gpu_runs.sh gpu3a-reference  both reference passes, blind files already closed.
#   scripts/v44_gpu_runs.sh gpu3b-authority  the final v4.4 label authority, both axes closed.
#   scripts/v44_gpu_runs.sh gpu4-train   smoke, baselines, three seeds, ablations, artifact.
#   scripts/v44_gpu_runs.sh gpu5-bank    the fresh engineering bank (SECOND environment).
#   scripts/v44_gpu_runs.sh gpu5-label   the fresh audit draw + both blind passes, closed.
#   scripts/v44_gpu_runs.sh gpu5-reference <partition>  that partition's reference passes.
#   scripts/v44_gpu_runs.sh gpu5-gate    thresholds on fresh development, then heldout ONCE.
#   scripts/v44_gpu_runs.sh human-prepare  the 250-row sample + frozen detector predictions.
#   scripts/v44_gpu_runs.sh final-bank   refuses without a PASSING human report.
#   scripts/v44_gpu_runs.sh final-audit  draws the sealed bank and runs its blind passes.
#   scripts/v44_gpu_runs.sh final-reference  the sealed bank's reference passes.
#   scripts/v44_gpu_runs.sh final-gate   the sealed bank, opened once.
#
# `verify` was one command that said "safe anywhere, no GPU needed" and then ran
# `env-check --strict` and `nvidia-smi`. It is split so the CPU half can actually be run on
# a laptop before an instance is rented, which is when its answer is worth most.
#
# The phases do NOT chain. gpu2 refuses to start until you have inspected all 60 smoke rows
# and recorded that you did -- see the message it prints. That gap is the protocol's, not
# this script's: the smoke exists to be read by a person, and a runner that flowed straight
# from it into 1,733 reportable rows would make the reading optional.
#
# Each adjudication pause is a real stop. Every later phase refuses until the decision file
# it needs exists, and re-checks that no earlier artifact moved.
#
# One model per process, always. Judge A runs to completion and the process EXITS before
# judge B is loaded; two models on one 24 GB card is how a labelling run ends up with fewer
# labels than rows.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V44="$REPO/data/cohorts/graph_unlearning_v1/detector_v4_4"
V43="$REPO/data/cohorts/graph_unlearning_v1/detector_v4_3"
PINS="$V43/DETECTOR_V4_3_LOCAL_JUDGE_PINS.json"
BUNDLE="$V44/DETECTOR_V4_4_PAIR_BUNDLE.json"
SMOKE="$V44/V4_4_JUDGE_SMOKE_60.jsonl"

LOGS="${V44_LOG_DIR:-/workspace/rdl-artifacts/v44}"
INSPECTED="$LOGS/GPU1_INSPECTED_BY"

export HF_HOME="${HF_HOME:-/workspace/hf}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$REPO" || exit 1
mkdir -p "$LOGS"

say()  { printf '\n=== %s ===\n' "$*"; }
die()  { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

# A misspelled or deleted function must STOP the run, not skip a step.
#
# This file is `set -uo pipefail` and deliberately not `set -e`: many phases capture
# `rc=$?` and decide what to do, which `-e` would pre-empt. But without `-e`, bash prints
# "command not found", returns 127, and CARRIES ON -- so when the preflight was split into
# preflight_cpu/preflight_gpu and the two callers still said `preflight`, gpu1 and gpu2
# would have skipped every safety check and gone straight into a paid judging run. The
# checks would have been reported as absent in the log and nowhere else.
#
# `command_not_found_handle` turns exactly that class of mistake into a stop, without
# touching the explicit exit-code handling everything else relies on.
#
# The `kill` is not belt-and-braces. Bash invokes this handler "in a separate execution
# environment" -- a subshell -- so a plain `exit` here ends only the subshell and the
# parent carries on with status 127, which is the very behaviour being fixed. Signalling
# $$ (the top-level shell, unchanged inside the subshell) is what actually stops the run.
command_not_found_handle() {
  printf '\nSTOP: `%s` is not a command or function.\n' "$1" >&2
  printf 'A phase called something that does not exist; refusing to continue into a run.\n' >&2
  kill -s TERM "$$"
  exit 127
}

# --------------------------------------------------------------------------- checks --

# Nothing frozen may move. Uses the same table the unit tests and the dry run read, so
# there is one list and it cannot drift.
check_preserved() {
  python - "$REPO" <<'PY' || return 1
import sys
from pathlib import Path
from rdl.eval.detector_v4_4_preservation import changed_artifacts

cohort = Path(sys.argv[1]) / "data" / "cohorts" / "graph_unlearning_v1"
changed = changed_artifacts(cohort)
if changed:
    print("FROZEN ARTIFACTS MOVED:")
    for relative, what in changed:
        print(f"  {relative}: {what}")
    sys.exit(1)
print("frozen artifacts: unchanged")
PY
}

# The four v4.4 inputs must be exactly what was reviewed. Recomputed from the files rather
# than read out of them, so a hand-edited bundle cannot vouch for itself.
check_inputs() {
  python - "$BUNDLE" "$V44/DETECTOR_V4_4_CALIBRATION_PANEL.json" "$SMOKE" <<'PY' || return 1
import hashlib, json, sys
from collections import Counter

bundle = json.loads(open(sys.argv[1], encoding="utf-8").read())
panel = json.loads(open(sys.argv[2], encoding="utf-8").read())
smoke = [json.loads(l) for l in open(sys.argv[3], encoding="utf-8") if l.strip()]

fail = []
if panel.get("bundle_sha256") != bundle.get("bundle_sha256"):
    fail.append("the panel was frozen against a different bundle")
if panel["composition"]["by_intended_class"] != {"ANSWER": 200, "NONE": 200, "PARTIAL": 200}:
    fail.append(f"the panel is not balanced: {panel['composition']['by_intended_class']}")
if len(smoke) != 60 or Counter(r["_intended_class"] for r in smoke) != {
    "ANSWER": 20, "NONE": 20, "PARTIAL": 20
}:
    fail.append(f"the smoke fixture is not 20/20/20 over 60 rows ({len(smoke)} rows)")
sep = "\x1f"
used = {p["pair_sha256"] for p in bundle["pairs"]}
overlap = [
    r["audit_id"] for r in smoke
    if hashlib.sha256((r["conditioning_question"] + sep + r["candidate_text"]).encode()).hexdigest() in used
]
if overlap:
    fail.append(f"{len(overlap)} smoke rows overlap the reportable bundle")

if fail:
    print("V4.4 INPUTS ARE NOT AS REVIEWED:")
    for f in fail:
        print(f"  {f}")
    sys.exit(1)
print(f"v4.4 inputs: bundle {bundle['n_pairs']} pairs @ {bundle['bundle_sha256'][:12]}, "
      f"panel 200/200/200 @ {panel['panel_sha256'][:12]}, smoke 60 rows disjoint")
PY
}

# The CPU half. Everything here is a statement about FILES, and none of it needs a GPU --
# which is the point: this is the half worth running before an instance is rented.
preflight_cpu() {
  say "preflight (cpu)"
  [ -f "$BUNDLE" ] || die "$BUNDLE is absent. Run the CPU phase first."
  [ -f "$SMOKE" ]  || die "$SMOKE is absent. Run graph-detector-v4-4-judge-smoke-fixture."
  [ -f "$PINS" ]   || die "$PINS is absent. The annotator comes from a committed file."

  # A dirty tree is refused so that "what changed" after a run is unambiguous. Untracked
  # log and output files are expected and do not count.
  local dirty
  dirty="$(git status --porcelain --untracked-files=no)"
  [ -z "$dirty" ] || die "the working tree has uncommitted changes:
$dirty
A reportable run must be attributable to a commit."

  echo "commit: $(git rev-parse HEAD)"
  check_preserved || die "frozen artifacts moved. Do not run anything until this is explained."
  check_inputs    || die "the v4.4 inputs are not the ones that were reviewed."

  python scripts/v44_pipeline_dryrun.py > "$LOGS/dryrun.log" 2>&1 \
    || die "the CPU dry run failed. See $LOGS/dryrun.log"
  echo "dry run: $(tail -2 "$LOGS/dryrun.log" | head -1)"
  echo "cpu preflight: ready"
}

# The GPU half. Strict env-check and nvidia-smi both live here and nowhere else.
preflight_gpu() {
  preflight_cpu
  say "preflight (gpu)"
  command -v nvidia-smi >/dev/null 2>&1 \
    || die "nvidia-smi is not on PATH. This is verify-GPU; use verify-cpu on a laptop."

  python -m rdl.cli graph-detector-v4-4-env-check --strict > "$LOGS/envcheck.json" 2>"$LOGS/envcheck.err" \
    || die "env-check --strict failed. Read $LOGS/envcheck.json -- it names each failing check.
CUDA, the pinned judges, the quantizers, the dtype and free disk are all reported there."
  echo "env-check: ready"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
}

# Every phase after gpu2 re-checks that nothing frozen moved, and that the tree is still
# the commit the earlier phases ran under. A phase that only checked its own inputs would
# let a mid-run edit to the bundle go unnoticed until the authority was already written.
require_clean_and_frozen() {
  local dirty
  dirty="$(git status --porcelain --untracked-files=no)"
  [ -z "$dirty" ] || die "the working tree changed mid-run:
$dirty"
  check_preserved || die "frozen artifacts moved since the run started."
}

# A manual pause is a file. This is how each phase refuses to skip one.
require_decision_file() {
  local path="$1" what="$2"
  [ -f "$path" ] || die "$path is absent.
$what
This is a deliberate pause: the protocol requires a person to decide, and a runner that
continued without the decision file would make the decision optional."
  [ -s "$path" ] || die "$path is empty."
}

# What a judging run has to be able to say about itself before its labels count.
report_acceptance() {
  local run_json="$1"
  python - "$run_json" <<'PY'
import json, sys

run = json.loads(open(sys.argv[1], encoding="utf-8").read())
gen = run.get("generator", {}) or {}
quant = gen.get("quantization", {}) or {}
loaded = quant.get("loaded", {}) or {}
devices = set(gen.get("parameter_devices") or quant.get("parameter_devices") or [])

checks = [
    ("revision is a commit sha", bool(run.get("pin", {}).get("revision_is_a_commit_sha"))),
    ("quantization as pinned", quant.get("requested") == run.get("quantization")),
    ("8-bit/4-bit actually loaded", bool(loaded.get("load_in_8bit") or loaded.get("load_in_4bit"))),
    ("compute dtype bfloat16", "bfloat16" in str(gen.get("dtype", ""))),
    ("no CPU/disk offload", not any(d in ("cpu", "disk", "meta") for d in devices)),
    ("zero malformed", run.get("n_malformed") == 0),
    ("zero retries", run.get("n_retries", 0) == 0),
    ("zero truncated", run.get("n_truncated") == 0),
    ("no OOM", not run.get("oom")),
]
width = max(len(n) for n, _ in checks)
for name, ok in checks:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}")
print(f"  -- {run.get('repo_id')} @ {str(run.get('revision'))[:12]}, {run.get('n_rows')} rows, "
      f"{run.get('seconds')}s, peak {round((run.get('peak_vram_bytes') or 0) / 2**30, 2)} GiB")
sys.exit(0 if all(ok for _, ok in checks) else 1)
PY
}

# One judge pass, run to completion and then CLOSED. This is the whole lifecycle.
#
# The judge appends to `<stem>.partial.jsonl` while it generates. The frozen
# `<stem>.jsonl` that every report reads is written by a SECOND invocation with `--close`,
# which is also where completeness, malformed and truncation are checked -- so a pass that
# is generated but never closed produces no file any report can see. GPU-2 did this
# correctly; GPU-3A and the fresh-bank passes did not, and would each have spent hours
# generating labels and then failed at the report with "the file is absent".
#
#   run_and_close_judge <judge> <pass> <log-tag> [extra args...]
#
# `--out-dir` and the input selector are passed through, so one helper serves the bundle
# passes and the per-partition fresh passes alike.
run_and_close_judge() {
  local judge="$1" pass="$2" tag="$3"; shift 3
  local log="$LOGS/${tag}_${judge}_${pass}.log"

  say "judge $judge, $pass, $tag  ($(date +%T))"
  python -m rdl.cli graph-detector-v4-4-local-judge \
    --judge "$judge" --pass "$pass" "$@" >> "$log" 2>&1
  local rc=$?
  [ $rc -eq 0 ] || die "judge $judge ($pass, $tag) exited $rc. See $log -- the run is
resumable: re-running skips every row already written to the partial file."

  say "closing judge $judge ($pass, $tag)"
  python -m rdl.cli graph-detector-v4-4-local-judge \
    --judge "$judge" --pass "$pass" --close "$@" >> "$log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || die "judge $judge ($pass, $tag) would not close. It refuses on any
unjudged, malformed or truncated row, and the gate requires zero of each -- see $log"
  tail -1 "$log"
}

# ------------------------------------------------------------------------- the phases --

gpu1() {
  preflight_gpu
  say "GPU-1: the 60-row dual-judge smoke (non-reportable)"

  for spec in "A:v44-smoke-qwen" "B:v44-smoke-mistral"; do
    local judge="${spec%%:*}" run_id="${spec##*:}"
    local out="$LOGS/$run_id"
    # Belt and braces on the only destructive line in this file. `out` is built from two
    # variables, and an empty or repo-relative one would turn a cleanup into something else
    # entirely. A smoke's output directory is disposable; nothing else here is.
    case "$out" in
      "$LOGS"/v44-smoke-*) : ;;
      *) die "refusing to clear '$out': a smoke output directory must be under $LOGS" ;;
    esac
    [ -n "$run_id" ] || die "empty run_id"
    rm -rf "$out"; mkdir -p "$out"

    say "judge $judge -> $run_id  ($(date +%T))"
    # One model, one process. This invocation returns only when the process has exited and
    # the VRAM is actually released.
    python -m rdl.cli graph-detector-v4-4-local-judge \
      --judge "$judge" --pass blind \
      --input "$SMOKE" --pins "$PINS" --out-dir "$out" \
      --non-reportable --run-id "$run_id" > "$out.log" 2>&1
    local rc=$?
    echo "exit=$rc  gpu_now=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    [ $rc -eq 0 ] || die "judge $judge failed. See $out.log"

    say "judge $judge acceptance"
    report_acceptance "$out/V4_4_SMOKE_${run_id}_JUDGE_${judge}_BLIND_RUN.json" \
      || die "judge $judge did not meet the acceptance criteria. The labels do not count."
  done

  check_preserved || die "frozen artifacts moved DURING the smoke. Investigate before continuing."

  cat <<EOF

=== GPU-1 complete. Both judges met the acceptance criteria. ===

The next step is not a command. Read all 60 rows and compare the two judges, with your
attention on the open-ended PARTIAL vs ANSWER boundary -- that is where v4.3 came apart
(raw agreement 0.695 against 0.840 on slot questions, 184 of 258 disagreements in that one
cell), and it is the thing the v4.4 rubric was rewritten to fix. A smoke nobody read has
told you only that the harness runs.

  scripts/v44_gpu_runs.sh compare

If the rubric needs changing, change it, bump PROMPT_VERSION, and re-run gpu1. That is
cheap now and impossible after GPU-2.

When you are satisfied, record who read them:

  echo "read by <name> on \$(date -I)" > "$INSPECTED"

GPU-2 refuses to start until that file exists.
EOF
}

compare() {
  say "the 60 smoke rows, both judges, side by side"
  python - "$SMOKE" "$LOGS/v44-smoke-qwen" "$LOGS/v44-smoke-mistral" <<'PY'
import json, sys
from pathlib import Path

def labels(directory: Path, judge: str, run_id: str) -> dict:
    path = directory / f"V4_4_SMOKE_{run_id}_JUDGE_{judge}_BLIND.partial.jsonl"
    if not path.exists():
        return {}
    return {
        str(r["audit_id"]): r
        for r in (json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip())
    }

smoke = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
a = labels(Path(sys.argv[2]), "A", "v44-smoke-qwen")
b = labels(Path(sys.argv[3]), "B", "v44-smoke-mistral")
if not a or not b:
    print("both judge outputs are needed; run gpu1 first")
    raise SystemExit(1)

agree = disagree = 0
for row in sorted(smoke, key=lambda r: (r["_intended_class"], r["audit_id"])):
    rid = row["audit_id"]
    ra, rb = a.get(rid, {}), b.get(rid, {})
    la, lb = ra.get("answer_attempt"), rb.get("answer_attempt")
    same = la == lb and la is not None
    agree += same
    disagree += not same
    flag = "  " if same else "!!"
    qtype = ra.get("question_type") or rb.get("question_type") or "?"
    print(f"{flag} [{row['_intended_class']:<7} {qtype:<10}] A={la} B={lb}")
    print(f"     Q: {row['conditioning_question'][:96]}")
    print(f"     C: {row['candidate_text'][:150]}")
    if not same:
        print(f"     A: addresses={ra.get('addresses_question')} standalone={ra.get('standalone_answer')}")
        print(f"     B: addresses={rb.get('addresses_question')} standalone={rb.get('standalone_answer')}")
    print()

print(f"agree {agree}/{agree + disagree}, disagree {disagree}")
print("This is 60 rows and it is NOT a gate. It is here so a person can see whether the")
print("rubric reads the way it was meant to, especially on open-ended rows.")
PY
}

gpu2() {
  [ -f "$INSPECTED" ] || die "GPU-1's 60 rows have not been recorded as read.
The smoke exists to be read by a person; running 1,733 reportable rows first would make
that optional. Run 'scripts/v44_gpu_runs.sh compare', then:
  echo \"read by <name> on \$(date -I)\" > $INSPECTED"

  preflight_gpu
  say "GPU-2: reportable blind labels over the full v4.4 bundle"
  echo "inspection recorded: $(cat "$INSPECTED")"

  local out="$V44"
  for judge in A B; do
    say "judge $judge, blind, full bundle  ($(date +%T))"
    # Judge, then exit. Then close in a second process, which is what hashes the file.
    python -m rdl.cli graph-detector-v4-4-local-judge \
      --judge "$judge" --pass blind \
      --bundle "$BUNDLE" --pins "$PINS" --out-dir "$out" >> "$LOGS/blind_${judge}.log" 2>&1
    local rc=$?
    echo "judge exit=$rc  gpu_now=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
    [ $rc -eq 0 ] || die "judge $judge failed. See $LOGS/blind_${judge}.log -- the run is
resumable: re-running skips every row already written to the partial file."

    say "judge $judge acceptance"
    report_acceptance "$out/V4_4_LOCAL_JUDGE_${judge}_BLIND_RUN.json" \
      || die "judge $judge did not meet the acceptance criteria. Do not close this pass."

    say "closing judge $judge"
    python -m rdl.cli graph-detector-v4-4-local-judge \
      --judge "$judge" --pass blind --close \
      --bundle "$BUNDLE" --out-dir "$out" >> "$LOGS/blind_${judge}.log" 2>&1
    [ $? -eq 0 ] || die "judge $judge would not close. It refuses on any unjudged, malformed
or truncated row, and the gate requires zero of each -- see $LOGS/blind_${judge}.log"
    tail -1 "$LOGS/blind_${judge}.log"
  done

  check_preserved || die "frozen artifacts moved DURING the blind passes."

  say "the blind-only report and the balanced-panel gate"
  python -m rdl.cli graph-detector-v4-4-label-report --blind-only --out-dir "$V44" \
    | tee "$LOGS/label_report.json"
  local rc=${PIPESTATUS[0]}

  cat <<EOF

=== GPU-2 complete. ===

Read $V44/DETECTOR_V4_4_LABEL_REPORT.json before anything else.

If the balanced-panel gate FAILED: stop. Return to the rubric or the data. Do not adjudicate
a failed panel -- the report already refuses to, and the refusal is the point.

If it PASSED: adjudicate $V44/V4_4_BLIND_DISAGREEMENTS.jsonl **blind**, with no reference
answers in view, exactly as the judges did. Then GPU-3.
EOF
  return $rc
}

# ------------------------------------------------- GPU-3A: the two reference passes --

gpu3a_reference() {
  require_clean_and_frozen
  require_decision_file "$V44/V4_4_BLIND_ADJUDICATION.jsonl" \
"Adjudicate \$V44/V4_4_BLIND_DISAGREEMENTS.jsonl BLIND -- with no reference answer and no
reference key in view -- and write the decisions to V4_4_BLIND_ADJUDICATION.jsonl."

  # The blind outputs must already be closed. The reference pass shows the annotator the
  # answer; if a blind file were still open afterwards, its remaining rows would be labelled
  # by someone who had seen it, and the two axes would stop being independent.
  for judge in A B; do
    local blind="$V44/V4_4_BLIND_JUDGE_${judge}.jsonl"
    [ -f "$blind" ] || die "$blind is absent. GPU-2 has not closed judge $judge's blind pass."
  done

  # One model per process, and each pass CLOSED before the next judge is loaded. The close
  # is what writes the frozen .jsonl the label report reads; without it this phase would
  # generate both reference passes over 1,733 rows and then fail at the report saying the
  # files are absent.
  for judge in A B; do
    run_and_close_judge "$judge" reference gpu3a \
      --bundle "$BUNDLE" --pins "$PINS" --out-dir "$V44" \
      --eval-key "$V43/PROTECTED_STORE_EVAL_KEY.json"
  done

  # Writes V4_4_REFERENCE_DISAGREEMENTS.jsonl and then REFUSES, because the reference axis
  # is not closed until every one of them is resolved.
  say "GPU-3A: reference disagreements"
  python -m rdl.cli graph-detector-v4-4-label-report \
    --adjudication "$V44/V4_4_BLIND_ADJUDICATION.jsonl" \
    --require-reference-pass 2>&1 | tee "$LOGS/gpu3a-report.log" || true

  cat <<EOF

The next step is not a command.

Resolve every row in $V44/V4_4_REFERENCE_DISAGREEMENTS.jsonl WITH the reference answer
visible -- that file shows it deliberately, because "does this convey the answer" cannot be
decided blind. Write the decisions to V4_4_REFERENCE_ADJUDICATION.jsonl, then run
gpu3b-authority.

These decisions are offline diagnostics. They are never training labels.
EOF
}

# ------------------------------------------------- GPU-3B: the final label authority --

gpu3b_authority() {
  require_clean_and_frozen
  require_decision_file "$V44/V4_4_BLIND_ADJUDICATION.jsonl" "The blind adjudication is missing."
  require_decision_file "$V44/V4_4_REFERENCE_ADJUDICATION.jsonl" \
"Resolve \$V44/V4_4_REFERENCE_DISAGREEMENTS.jsonl with the reference answer visible."

  say "GPU-3B: the v4.4 label authority"
  python -m rdl.cli graph-detector-v4-4-label-report \
    --adjudication "$V44/V4_4_BLIND_ADJUDICATION.jsonl" \
    --reference-adjudication "$V44/V4_4_REFERENCE_ADJUDICATION.jsonl" \
    --require-reference-pass 2>&1 | tee "$LOGS/gpu3b-authority.log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || die "the label authority was refused. Read $LOGS/gpu3b-authority.log"

  # The bundle and the panel must be byte-identical to their pre-run state. The authority
  # binds the bundle hash, so a bundle rewritten WITH labels would make the panel's kappa
  # evidence about a file that no longer exists.
  check_preserved || die "the bundle or the panel moved while the authority was written."
  echo "authority written; bundle and panel unchanged"
}

# ------------------------------------------------------------------- GPU-4: training --

gpu4_train() {
  require_clean_and_frozen
  local authority="$V44/DETECTOR_V4_4_LABEL_AUTHORITY.json"
  [ -f "$authority" ] || die "$authority is absent. Run gpu3b-authority first."

  say "GPU-4: DeBERTa smoke (non-reportable)"
  python scripts/train_detector_v4.py \
    --v4-4-bundle "$BUNDLE" \
    --v4-4-label-authority "$authority" \
    --v4-4-eval-key "$V43/PROTECTED_STORE_EVAL_KEY.json" \
    --v4-2-dir "$REPO/data/cohorts/graph_unlearning_v1/detector_v4_2" \
    --model-repo-id microsoft/deberta-v3-base \
    --model-revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
    --tokenizer-revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
    --skip-baseline --skip-ablations --seeds 20260814 --epochs 1 --device cuda \
    --output-dir "$LOGS/model-smoke" 2>&1 | tee "$LOGS/gpu4-smoke.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "the training smoke failed. Nothing reportable ran."

  say "GPU-4: baselines, three seeds, ablations"
  python scripts/train_detector_v4.py \
    --v4-4-bundle "$BUNDLE" \
    --v4-4-label-authority "$authority" \
    --v4-4-eval-key "$V43/PROTECTED_STORE_EVAL_KEY.json" \
    --v4-2-dir "$REPO/data/cohorts/graph_unlearning_v1/detector_v4_2" \
    --model-repo-id microsoft/deberta-v3-base \
    --model-revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
    --tokenizer-revision 8ccc9b6f36199bec6961081d44eb72fb3f7353f3 \
    --baseline-repo-id cross-encoder/nli-deberta-v3-base \
    --baseline-revision 6c749ce3425cd33b46d187e45b92bbf96ee12ec7 \
    --seeds 20260814 20260815 20260816 --epochs 3 --device cuda --reportable \
    --output-dir "$LOGS/model" 2>&1 | tee "$LOGS/gpu4-train.log"
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || die "training exited $rc. Read $LOGS/gpu4-train.log"

  cat <<EOF

Checkpoint selected on DEVELOPMENT only. Do not choose a seed from held-out, and do not add
training rows after seeing development performance.

$LOGS/model/DETECTOR_V4_MODEL.json is the artifact every later phase names.
Next: gpu5-bank, in the SECOND environment.
EOF
}

# ---------------------------------------------------- GPU-5A: the fresh bank (env 2) --

gpu5_bank() {
  say "GPU-5A: the fresh engineering bank"
  cat <<EOF
This phase runs in the GRAPH environment, not the detector one:

  deactivate
  source /workspace/venvs/rdl-v44-graph/bin/activate

vLLM in the detector environment would replace its PyTorch and Transformers, and the
checkpoint the gate is about to be run on would then be loaded by different bytes than the
ones it was trained under.
EOF
  python -m rdl.cli graph-detector-v4-2-plan-bank-runs 2>&1 | tee "$LOGS/gpu5-plan.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "the run plan was refused."

  cat <<EOF

Read $REPO/data/cohorts/graph_unlearning_v1/detector_v4_2/ENGINEERING_BANK_RUNS.sh and check
that it names exactly the four natural seeds 50241..50244 and the four retain seeds
51241..51244. Then run it, and assemble with every generated directory supplied EXPLICITLY
to graph-detector-v4-2-build-bank (--check-only first). Do not glob run directories: a
directory that is not in ENGINEERING_BANK_RUN_PLAN.json is not part of this bank.

Then return to the detector environment and run gpu5-label.
EOF
}

# ------------------------------------------- GPU-5B: the fresh audit and its labels --

gpu5_label() {
  require_clean_and_frozen
  local bank="$REPO/data/cohorts/graph_unlearning_v1/detector_v4_2/ENGINEERING_BANK.json"
  [ -f "$bank" ] || die "$bank is absent. Run gpu5-bank first."
  [ -f "$V44/DETECTOR_V4_4_FRESH_AUDIT_PLAN.json" ] \
    || die "the fresh-audit plan is absent. It is frozen on CPU, BEFORE the bank exists:
run graph-detector-v4-4-fresh-audit-plan and commit it."

  # The draw happens before any DeBERTa score is computed. That ordering is the whole
  # content of "score-independent", so it is a separate phase from the gate.
  say "GPU-5B: the fresh audit draw (both partitions, no scoring)"
  python -m rdl.cli graph-detector-v4-4-fresh-audit --bank "$bank" \
    2>&1 | tee "$LOGS/gpu5-audit.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "the fresh audit draw was refused."

  # The partition is a DIRECTORY, not part of the pass name. The judge accepts --pass
  # blind or reference and nothing else, and `run_rows_v4_4` selects the reference prompt
  # by comparing that string exactly -- so "fresh-development-reference" would be refused
  # outright, and anything that slipped past would run the BLIND prompt and write a
  # reference file with no reference answer in it.
  for partition in development heldout; do
    judge_dir="$V44/fresh/$partition"
    mkdir -p "$judge_dir"
    for judge in A B; do
      run_and_close_judge "$judge" blind "gpu5-${partition}" \
        --pins "$PINS" --out-dir "$judge_dir" \
        --input "$V44/V4_4_FRESH_BLIND_${partition}.jsonl"
    done
    python -m rdl.cli graph-detector-v4-4-fresh-label-report \
      --partition "$partition" --blind-only 2>&1 \
      | tee "$LOGS/gpu5-${partition}-blind-report.log" || true
  done

  cat <<EOF

Two manual pauses now, in this order, per partition:

  1. adjudicate V4_4_FRESH_<partition>_BLIND_DISAGREEMENTS.jsonl WITHOUT any reference
     answer, then run the reference passes for that partition:

       scripts/v44_gpu_runs.sh gpu5-reference <partition>

     That phase runs judge A and judge B and CLOSES each one, which is what writes the
     frozen file the report reads. The judge refuses to open a reference pass until that
     directory's blind file is closed, which is why the directory is per-partition.
  2. adjudicate V4_4_FRESH_<partition>_REFERENCE_DISAGREEMENTS.jsonl WITH the reference
     answer, then run fresh-label-report with both --adjudication and
     --reference-adjudication.

Both partitions must be fully labelled before gpu5-gate, and the HELDOUT labels are
produced now, before any threshold exists -- labelling held-out rows after seeing the
threshold is how a held-out set stops being one.
EOF
}

# --------------------------- GPU-5B(ii): the fresh reference passes, one partition --

# Separate from gpu5-label because a manual blind adjudication sits between them. Takes
# the partition as an argument rather than looping, so development can be closed and
# adjudicated while heldout has not been opened at all.
gpu5_reference() {
  local partition="${1:-}"
  case "$partition" in
    development|heldout) : ;;
    *) die "usage: scripts/v44_gpu_runs.sh gpu5-reference development|heldout" ;;
  esac
  require_clean_and_frozen

  local judge_dir="$V44/fresh/$partition"
  local key="$V44/DETECTOR_V4_4_FRESH_REFERENCE_KEY.json"
  [ -f "$key" ] || die "$key is absent. The fresh audit was built with
--skip-reference-key, which makes it non-reportable: the reference pass has no answers."
  require_decision_file "$V44/V4_4_FRESH_${partition}_BLIND_ADJUDICATION.jsonl" \
"Adjudicate V4_4_FRESH_${partition}_BLIND_DISAGREEMENTS.jsonl BLIND -- with no reference
answer in view -- before the reference pass for this partition opens."

  for judge in A B; do
    run_and_close_judge "$judge" reference "gpu5-${partition}" \
      --pins "$PINS" --out-dir "$judge_dir" \
      --input "$V44/V4_4_FRESH_BLIND_${partition}.jsonl" \
      --eval-key "$key"
  done

  # Writes the reference disagreements and then refuses, exactly as GPU-3A does for the
  # bundle: the axis is not closed until every one of them is resolved.
  python -m rdl.cli graph-detector-v4-4-fresh-label-report \
    --partition "$partition" \
    --adjudication "$V44/V4_4_FRESH_${partition}_BLIND_ADJUDICATION.jsonl" \
    2>&1 | tee "$LOGS/gpu5-${partition}-reference-report.log" || true

  cat <<EOF

Resolve V4_4_FRESH_${partition}_REFERENCE_DISAGREEMENTS.jsonl WITH the reference answer
visible, then close the partition:

  python -m rdl.cli graph-detector-v4-4-fresh-label-report \\
    --partition ${partition} \\
    --adjudication $V44/V4_4_FRESH_${partition}_BLIND_ADJUDICATION.jsonl \\
    --reference-adjudication $V44/V4_4_FRESH_${partition}_REFERENCE_ADJUDICATION.jsonl

That writes DETECTOR_V4_4_FRESH_LABELLED_${partition}.json, which gpu5-gate reads.
EOF
}

# ------------------------- GPU-5C: the operating point, then the one-shot heldout gate --

gpu5_gate() {
  require_clean_and_frozen
  local model="${V44_MODEL_ARTIFACT:-$LOGS/model/DETECTOR_V4_MODEL.json}"
  [ -f "$model" ] || die "$model is absent. Set V44_MODEL_ARTIFACT or run gpu4-train."
  local dev="$V44/DETECTOR_V4_4_FRESH_LABELLED_development.json"
  local held="$V44/DETECTOR_V4_4_FRESH_LABELLED_heldout.json"
  [ -f "$dev" ]  || die "$dev is absent. Close the development labels first."
  [ -f "$held" ] || die "$held is absent. The held-out labels are produced BEFORE the
threshold exists; labelling them afterwards would let the threshold influence the labels."

  say "GPU-5C: thresholds on fresh DEVELOPMENT only"
  python -m rdl.cli graph-detector-v4-4-select-operating-point \
    --audit "$dev" --partition development \
    --backend cross_encoder --model-artifact "$model" --device cuda:0 \
    2>&1 | tee "$LOGS/gpu5-operating-point.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "no operating point satisfies both ceilings."

  [ -f "$V44/DETECTOR_V4_4_FROZEN_DETECTOR.json" ] \
    || die "the frozen detector artifact was not written."

  say "GPU-5C: the engineering held-out gate (opened ONCE)"
  python -m rdl.cli graph-detector-v4-4-final-gate \
    --audit "$held" --partition heldout \
    --operating-point "$V44/DETECTOR_V4_4_OPERATING_POINT.json" \
    --backend cross_encoder --model-artifact "$model" --device cuda:0 \
    2>&1 | tee "$LOGS/gpu5-heldout.log"
  local rc=${PIPESTATUS[0]}

  cat <<EOF

The held-out partition is now opened and recorded. There is no second opening.

If the gate FAILED: preserve it. Diagnose on the DEVELOPMENT surface, and any fix produces
a new detector version against a NEW fresh bank -- not a re-opened held-out set.
If it PASSED: human-prepare is next, and it is the last phase of Rental A.
EOF
  return $rc
}

# ------------------------------------------------- human preparation (end of rental A) --

human_prepare() {
  require_clean_and_frozen
  local gate="$V44/DETECTOR_V4_4_HELDOUT_GATE.json"
  [ -f "$gate" ] || die "$gate is absent. The human sample is drawn only after the frozen
detector has passed the engineering held-out gate."
  python - "$gate" <<'PY' || die "the held-out gate did not pass. Do not draw a human sample."
import json, sys
report = json.loads(open(sys.argv[1], encoding="utf-8").read())
if not report.get("passed"):
    print(f"held-out gate FAILED: {report.get('failures')}")
    sys.exit(1)
print("held-out gate passed")
PY

  say "human-prepare: the reportable 250"
  python -m rdl.cli graph-detector-v4-4-human-sample \
    --bundle "$BUNDLE" --fresh-audit "$V44/DETECTOR_V4_4_FRESH_AUDIT.json" \
    2>&1 | tee "$LOGS/human-sample.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "the human sample was refused."

  cat <<EOF

The two blind rater files are written. Before the raters start, score the sample ONCE with
the already-frozen detector and keep those predictions in a file the raters never see --
$V44/DETECTOR_V4_4_FROZEN_DETECTOR.json names the exact thresholds and checkpoint.

Raters must not see: model labels, detector predictions or scores, population, stratum,
source model, or the reference answer.

Back up every closed artifact and the checkpoint bytes before destroying this instance.
Rental A ends here; the humans take time and the final bank stays sealed until they pass.
EOF
}

# -------------------------------------------------- rental B: the sealed final bank --

final_bank() {
  require_clean_and_frozen
  say "final-bank: verifying the human report"
  python -m rdl.cli graph-detector-v4-4-unseal-final-bank 2>&1 | tee "$LOGS/unseal.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "the seal was NOT lifted. Read $LOGS/unseal.log.
A failing human report means the detector is not deployable; it does not mean the bound
moves, and it does not mean the humans are re-run until they agree."

  cat <<EOF

The seal is lifted for exactly one build. Generate the final runs from the SEALED seed set
(40241..40244, once for the natural cohort and once for the retain cohort, with
group-specific run ids), then assemble with graph-detector-v4-2-build-bank --bank final,
supplying every run directory explicitly.

Then: final-audit, its two label passes, and finally final-gate.
EOF
}

# The final bank's audit and labels. Same two judges, same rubric, same prompt version --
# the ONLY thing that may not happen between here and the gate is a change to the detector.
final_audit() {
  require_clean_and_frozen
  local bank="$REPO/data/cohorts/graph_unlearning_v1/detector_v4_2/FINAL_GATE_BANK.json"
  [ -f "$bank" ] || die "$bank is absent. Run final-bank and build it first."
  local seal="$REPO/data/cohorts/graph_unlearning_v1/detector_v4_2/FINAL_BANK_UNSEAL_RECORD.json"
  [ -f "$seal" ] || die "$seal is absent. The final bank is sealed."

  say "final-audit: drawing the sealed bank as ONE partition"
  python -m rdl.cli graph-detector-v4-4-fresh-audit \
    --bank "$bank" --bank-kind final 2>&1 | tee "$LOGS/final-audit.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || die "the final audit draw was refused."

  local judge_dir="$V44/fresh/final"
  mkdir -p "$judge_dir"
  for judge in A B; do
    run_and_close_judge "$judge" blind final \
      --pins "$PINS" --out-dir "$judge_dir" \
      --input "$V44/V4_4_FRESH_BLIND_final.jsonl"
  done

  python -m rdl.cli graph-detector-v4-4-fresh-label-report \
    --partition final --blind-only 2>&1 | tee "$LOGS/final-blind-report.log" || true

  cat <<EOF

Adjudicate V4_4_FRESH_final_BLIND_DISAGREEMENTS.jsonl BLIND, write the decisions to
V4_4_FRESH_final_BLIND_ADJUDICATION.jsonl, then run:

  scripts/v44_gpu_runs.sh final-reference
EOF
}

final_reference() {
  require_clean_and_frozen
  local judge_dir="$V44/fresh/final"
  local key="$V44/DETECTOR_V4_4_FINAL_REFERENCE_KEY.json"
  [ -f "$key" ] || die "$key is absent; the final audit is not reportable without it."
  require_decision_file "$V44/V4_4_FRESH_final_BLIND_ADJUDICATION.jsonl" \
"Adjudicate the final bank's blind disagreements WITHOUT any reference answer first."

  for judge in A B; do
    run_and_close_judge "$judge" reference final \
      --pins "$PINS" --out-dir "$judge_dir" \
      --input "$V44/V4_4_FRESH_BLIND_final.jsonl" --eval-key "$key"
  done

  python -m rdl.cli graph-detector-v4-4-fresh-label-report \
    --partition final \
    --adjudication "$V44/V4_4_FRESH_final_BLIND_ADJUDICATION.jsonl" \
    2>&1 | tee "$LOGS/final-reference-report.log" || true

  cat <<EOF

Resolve V4_4_FRESH_final_REFERENCE_DISAGREEMENTS.jsonl WITH the reference answer, then:

  python -m rdl.cli graph-detector-v4-4-fresh-label-report \\
    --partition final \\
    --adjudication $V44/V4_4_FRESH_final_BLIND_ADJUDICATION.jsonl \\
    --reference-adjudication $V44/V4_4_FRESH_final_REFERENCE_ADJUDICATION.jsonl

That writes DETECTOR_V4_4_FRESH_LABELLED_final.json. Then: final-gate.
EOF
}

final_gate() {
  require_clean_and_frozen
  local model="${V44_MODEL_ARTIFACT:-$LOGS/model/DETECTOR_V4_MODEL.json}"
  local audit="${V44_FINAL_AUDIT:-$V44/DETECTOR_V4_4_FRESH_LABELLED_final.json}"
  [ -f "$audit" ] || die "$audit is absent. Label the final bank's audit first."

  say "final-gate: the sealed bank, opened once"
  python -m rdl.cli graph-detector-v4-4-final-gate \
    --audit "$audit" --partition final \
    --operating-point "$V44/DETECTOR_V4_4_OPERATING_POINT.json" \
    --backend cross_encoder --model-artifact "$model" --device cuda:0 \
    2>&1 | tee "$LOGS/final-gate.log"
  local rc=${PIPESTATUS[0]}

  cat <<EOF

Same weights, same tokenizer, same class map, same protected store, same tau_answer and
tau_partial. No retraining, no recalibration, no new seed, no threshold change.

If it PASSED: record the result through a NEW hash-bound validation record. Do not edit the
model or operating-point artifacts in place.
If it FAILED: the detector is not deployable. Preserve the failure and build a new detector
version against a new untouched final bank. Do not reopen this one.
EOF
  return $rc
}

case "${1:-}" in
  verify-cpu)      preflight_cpu ;;
  verify-gpu)      preflight_gpu ;;
  verify)          die "'verify' is split. Use verify-cpu (no GPU) or verify-gpu (needs one)." ;;
  gpu1)            gpu1 ;;
  compare)         compare ;;
  gpu2)            gpu2 ;;
  gpu3a-reference) gpu3a_reference ;;
  gpu3b-authority) gpu3b_authority ;;
  gpu4-train)      gpu4_train ;;
  gpu5-bank)       gpu5_bank ;;
  gpu5-label)      gpu5_label ;;
  gpu5-reference)  shift; gpu5_reference "${1:-}" ;;
  gpu5-gate)       gpu5_gate ;;
  human-prepare)   human_prepare ;;
  final-bank)      final_bank ;;
  final-audit)     final_audit ;;
  final-reference) final_reference ;;
  final-gate)      final_gate ;;
  *)               sed -n '3,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
