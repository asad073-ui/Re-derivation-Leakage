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
#   scripts/v44_gpu_runs.sh verify     preflight only. Safe anywhere, no GPU needed.
#   scripts/v44_gpu_runs.sh gpu1       the 60-row dual-judge smoke. Non-reportable.
#   scripts/v44_gpu_runs.sh gpu2       the reportable blind passes + the blind-only report.
#
# The phases do NOT chain. gpu2 refuses to start until you have inspected all 60 smoke rows
# and recorded that you did -- see the message it prints. That gap is the protocol's, not
# this script's: the smoke exists to be read by a person, and a runner that flowed straight
# from it into 1,733 reportable rows would make the reading optional.
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

preflight() {
  say "preflight"
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

  python -m rdl.cli graph-detector-v4-4-env-check --strict > "$LOGS/envcheck.json" 2>"$LOGS/envcheck.err" \
    || die "env-check --strict failed. Read $LOGS/envcheck.json -- it names each failing check.
CUDA, the pinned judges, the quantizers, the dtype and free disk are all reported there."
  echo "env-check: ready"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
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

# ------------------------------------------------------------------------- the phases --

gpu1() {
  preflight
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

  preflight
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

case "${1:-}" in
  verify)  preflight ;;
  gpu1)    gpu1 ;;
  compare) compare ;;
  gpu2)    gpu2 ;;
  *)       sed -n '3,23p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
