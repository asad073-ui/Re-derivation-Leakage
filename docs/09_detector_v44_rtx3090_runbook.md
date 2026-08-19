# GraphForget detector v4.4 — RTX 3090 end-to-end runbook

**Verified against:** `d083c9f` (`fix/detector-v44-gpu-orchestration`), 2026-08-19.
**Scope:** last CPU-only work → RTX 3090 Rental A → human validation off-GPU → Rental B →
final GitHub evidence PR and tag.

This supersedes the uploaded `GraphForget_detector_v4_4_RTX3090_end_to_end_current_main`
PDF, which was written against a repository state that does not exist yet. See §0.

---

## 0. Two corrections before anything is rented

### 0.1 PR #58 is not merged

The PDF's header says *"Current reviewed main: PR #58 merged"*. It is not. As of `d083c9f`:

```
d083c9f  fix: the v4.4 runner could not have completed a single paid phase   <- this branch
344c0a9  detector v4.4: the CPU completion PR before the first GPU rental (#57)   <- main
```

`d083c9f` sits on `fix/detector-v44-gpu-orchestration`, pushed, **not** in `origin/main`.
Every instruction below that says "clone main and run" is only true after that PR merges.
The GPU runner with GPU-1 through `final-gate` is on that branch and nowhere else.

**Order is therefore: merge #58 → lifecycle PR (§1) → rent.** Not: lifecycle PR onto a
main that lacks the runner.

### 0.2 The seven lifecycle gaps were real, and this branch closes them

Each was checked against the source at `d083c9f`, not inferred. The fixes are on
`fix/detector-v44-final-lifecycle`; §1 describes what landed.

| # | Gap | Evidence |
|---|-----|----------|
| A | Nothing scores the human sample with the frozen detector | No `HUMAN_DETECTOR_PREDICTIONS` anywhere in `src/`, `scripts/`, `tests/`. `human_prepare()` in `scripts/v44_gpu_runs.sh` prints prose telling the operator to "score the sample ONCE" and provides no command |
| B | No 250-row model-consensus builder | No `HUMAN_MODEL_CONSENSUS` in the tree. `--model-consensus` is required by the report but nothing produces it for the mixed 125+125 draw |
| C | Report bindings are optional | `detector_v4_4_human.py:878,885` — `--detector-predictions` and `--frozen-detector` both default to `None`; `:955` records `"why": "--detector-predictions was not passed"` and continues |
| D | Planner is engineering-only | `detector_v4_2_bank_runs.py` has **no** `--bank` option at all; it reads `ENGINEERING_MANIFEST_FILENAME` and emits `engineering-{group}-seed{seed}` names |
| E | Final bank wears engineering names | `detector_v4_2_banks.py:748` writes `ENGINEERING_BANK_VERIFICATION.json` for `--check-only` regardless of `--bank`; `:836` hardcodes `"bank_id": "detector_v4_2_engineering_v1"` in the payload |
| E′ | No one-build enforcement | `:888` is a plain `atomic_json(output_dir / BANK_FILENAME[bank], payload)`. A second `--bank final` build overwrites. (The **gate** is already one-shot — `:1041` `if record_path.exists()` — that protection does not extend to the build) |
| F | No finalize | No `graph-detector-v4-4-finalize`, no `FINAL_VALIDATION` in the tree; runner cases end at `final-gate` (`v44_gpu_runs.sh:899`) |

What is **already correct** and must not be re-implemented:

- `build-bank --bank final` exists and is genuinely seal-enforced — it re-verifies the
  unseal record rather than trusting the file's presence (`:645-663`).
- `--bank final` reads its **own** pre-registration (`FINAL_GATE_BANK_BUDGET.json`), not
  the engineering manifest (`:665-676`).
- The sealed seeds are frozen in three places and agree: `detector_v4_1_freeze.py:53`,
  `eval/detector_v4_2.py:388`, `FINAL_GATE_BANK_BUDGET.json` — `40241 40242 40243 40244`.
- `final-gate` already refuses a second opening.

---

## 1. The last CPU-only PR

Branch `fix/detector-v44-final-lifecycle`, stacked on #58. **Nothing changed** in the
detector objective, v4.4 rubric, prompt version, the 1,733-row bundle, the 600-row panel,
judge models, the DeBERTa pin, threshold gates, training seeds, engineering seeds, sealed
final seeds, or any human gate. Orchestration and provenance only.

### A. `graph-detector-v4-4-human-score`

Scores the reportable 250 exactly once, immediately after the draw, before raters see
anything. Inputs: selected checkpoint, frozen protected store, frozen `tau_answer`,
frozen `tau_partial`. Refuses a second scoring unless `--rescore` is passed, and the
report treats a rescored file as a provenance failure.

Rows come from `V4_4_HUMAN_SAMPLE_KEY.json`, not from any bundle — 125 of the 250 are
fresh ids that appear in no bundle at all.

Writes `V4_4_HUMAN_DETECTOR_PREDICTIONS.jsonl`, one row per sampled `audit_id`:

```json
{"audit_id": "...", "predicted_label": "NONE|PARTIAL|ANSWER",
 "answer_probability": 0.0, "partial_probability": 0.0}
```

Alongside it, `V4_4_HUMAN_DETECTOR_PREDICTIONS.json` binds: human sample hash, sample key
hash, frozen detector hash (both file and declared content), operating-point hash, model
artifact hash, selected checkpoint digest, protected-store fingerprint, source commit.
The report re-checks every one of them.

### B. `graph-detector-v4-4-human-model-consensus`

The reportable 250 is 125 rows from the v4.4 bundle plus 125 from the fresh engineering
audit, so the model labels live in two adjudicated files. Pass `--adjudicated` once per
source; it joins them for exactly those 250 IDs → `V4_4_HUMAN_MODEL_CONSENSUS.jsonl`.

It **decides nothing** — every label was adjudicated before it ran. It refuses a missing
sampled ID, refuses two files that disagree about a row, and requires its sources to be
named rather than discovered on disk. Rows outside the draw are dropped, not refused: the
adjudicated files are the full label sets.

### C. Harden `graph-detector-v4-4-human-report`

For a **reportable** sample the two flags are now required, and the distinction matters:

- **Absent** binding → hard refusal. That is an operator omitting a flag, and a report on
  disk saying `passed: false` for that reason reads like a finding.
- **Present but mismatched** → recorded as a provenance failure and the report is written.
  That *is* a finding, and the failing report is the evidence.

Coverage is checked against the drawn sample, not against the rows that carry a human
label — those two sets differ exactly when adjudication left something open.

Both flags stay optional for a sample drawn `--exploratory`, which is marked
non-reportable and fails the gate for that reason instead.

### D. `graph-detector-v4-2-plan-bank-runs --bank engineering|final`

For `--bank final`, derive both groups from the sealed `40241 40242 40243 40244` and write
`FINAL_BANK_RUN_PLAN.json` / `FINAL_BANK_RUNS.sh` under `final-{group}-seed{seed}` names.
The operator copies a frozen plan; they do not hand-compose eight commands at 3 a.m. on a
metered box.

### E. Final-bank provenance and one-build enforcement

- `--check-only --bank final` writes `FINAL_GATE_BANK_VERIFICATION.json`, never the
  engineering filename.
- The final payload carries a final-specific `bank_id`.
- The first successful final assembly writes `FINAL_GATE_BANK_BUILD_RECORD.json`, binding
  all eight run manifests, both seed groups, final bank hash, unseal record hash, final
  budget hash, final manifest hash, source commit, clean-tree state.
- A second final build is refused. `--check-only` stays allowed after a build — it writes
  no bank, and refusing it would push the operator toward the one command that does.
- Rebuilding the **engineering** bank is still allowed. It is engineering data, and a
  one-build rule there would teach the operator to delete records to get past it.

This is separate from the final gate's existing one-shot opening record. Two different
"once"s, two different files.

### F. `graph-detector-v4-4-finalize` + runner phase `finalize`

Refuses unless all of these are present and passing: label authority, model
artifact/checkpoint digest, operating point, frozen detector, engineering heldout gate,
human report, final-bank unseal record, final-bank build record, final labelled audit,
final gate report.

Writes `DETECTOR_V4_4_FINAL_VALIDATION.json`, binding every upstream hash, and sets

```json
{"deployable": true, "answerability_v4_ready": true}
```

only when the whole chain passes. It also refuses a **reopened** final gate, a final bank
built from a dirty tree, and a human report edited after the unseal record hashed it.

Two Nones is not treated as a match — two artifacts that both declined to state a hash
leave the chain unbound, and that is a failure, not agreement.

It **measures nothing**: every verdict is read from the artifact that decided it, so a
failure cannot become a pass. On refusal it exits non-zero and still writes the record —
the failing record is the evidence. **Never edit a frozen artifact in place to make it say
PASS**; finalize writes a new one.

### G. Tests

1. `human-prepare` + `human-score` create and bind exact-sample predictions and consensus.
2. Reportable report refuses missing/mismatched predictions and frozen detector.
3. Final planner emits exactly 8 commands, seeds `40241..40244` in both groups.
4. No engineering seed can enter a final run plan.
5. Final `--check-only` does not overwrite `ENGINEERING_BANK_VERIFICATION.json`.
6. Final bank carries a final-specific `bank_id`.
7. A second final build is refused.
8. `finalize` refuses a failed or missing final gate.
9. `finalize` refuses a changed checkpoint / model / frozen-detector hash.
10. `finalize` writes `deployable=true` only after the whole chain passes.

### CPU exit gate

```bash
make cpu-all
python scripts/v44_pipeline_dryrun.py
python -m rdl.cli graph-detector-v4-4-env-check
git diff --check
git status --short
```

On Windows use `.\tasks.ps1 cpu-all` against the pinned 3.10 interpreter instead of `make`.

```bash
git add <explicit lifecycle files only>      # never `git add .`
git commit -m "detector v4.4: close human and final-bank lifecycle"
git push -u origin fix/detector-v44-final-lifecycle
gh pr create --base main --head fix/detector-v44-final-lifecycle \
  --title "detector v4.4: close human and final-bank lifecycle" \
  --body "Orchestration/provenance only. No detector target, data, rubric, thresholds, model pins, or frozen seeds changed."
gh pr checks --watch
```

Merge only on green. Then:

```bash
git switch main && git pull --ff-only && git rev-parse HEAD
```

**That commit is the immutable GPU execution commit.** Record it; everything below uses it.

---

## 2. Rent the RTX 3090

```
GPU              1 x RTX 3090
VRAM             24 GB
System RAM       64 GB preferred
CPU              8+ vCPU
Container disk   250 GB
Free after setup >= 100 GiB
Mode             on-demand / reliable / not interruptible
Launch           SSH
Image            pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime
```

No H100. The 3090 has already shown enough VRAM for the pinned judges loaded one model per
process. Attach a persistent volume for checkpoint backup if the provider offers one —
never trust disposable container storage with the selected checkpoint.

---

## 3. Rental A — bootstrap

```bash
nvidia-smi
df -h /workspace
python --version

apt-get update
apt-get install -y git gh jq tmux rsync nano
tmux new -s rdl-v44          # reattach later: tmux attach -t rdl-v44
```

GitHub auth — no PAT in the clone URL:

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
gh auth status
```

Clone and pin the commit:

```bash
cd /workspace
gh repo clone asad073-ui/Re-derivation-Leakage
cd Re-derivation-Leakage
git switch main && git pull --ff-only
git rev-parse HEAD           # must equal the §1 execution commit
git status --short           # must be empty

mkdir -p /workspace/rdl-artifacts/v44
git rev-parse HEAD | tee /workspace/rdl-artifacts/v44/SOURCE_COMMIT.txt

git switch -c evidence/detector-v44-final
```

Do not modify detector source on this branch once reportable labels begin.

### Detector environment

```bash
python -m venv /workspace/venvs/rdl-v44-detector --system-site-packages
source /workspace/venvs/rdl-v44-detector/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[cpu,dev,gpu]"

export HF_HOME=/workspace/hf
export V44_LOG_DIR=/workspace/rdl-artifacts/v44
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export REPO=/workspace/Re-derivation-Leakage
export V44="$REPO/data/cohorts/graph_unlearning_v1/detector_v4_4"
export V43="$REPO/data/cohorts/graph_unlearning_v1/detector_v4_3"
export V42="$REPO/data/cohorts/graph_unlearning_v1/detector_v4_2"
mkdir -p "$HF_HOME" "$V44_LOG_DIR"
```

Verify torch and record the environment:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
print("bf16:", torch.cuda.is_bf16_supported())
print("vram GiB:", torch.cuda.get_device_properties(0).total_memory / 2**30)
PY

python -m pip freeze > "$V44_LOG_DIR/pip-freeze-before-run.txt"
nvidia-smi -q          > "$V44_LOG_DIR/nvidia-smi-before-run.txt"
df -h /workspace       > "$V44_LOG_DIR/disk-before-run.txt"
```

Sanity: `python -c 'import transformers; print(transformers.__version__)'` → `4.51.3`.

### Hard preflight

```bash
make cpu-all
python scripts/v44_pipeline_dryrun.py
scripts/v44_gpu_runs.sh verify-gpu
```

All three must pass. Stop on: dirty tree, a changed frozen artifact hash, CUDA
unavailable, a non-Ampere GPU, bf16 failure, a failed pinned judge/tokenizer/model check,
a failed quantization dependency check, or free disk under the strict gate.

---

## 4. GPU-1 — 60-row smoke

```bash
scripts/v44_gpu_runs.sh gpu1
scripts/v44_gpu_runs.sh compare | tee "$V44_LOG_DIR/smoke-comparison.txt"
```

Read **all 60 rows** by hand. The boundary that matters is `PARTIAL ↔ ANSWER` on
open-ended questions. If the rubric is wrong, **STOP** — a prompt change after reportable
labels begin invalidates them.

```bash
printf 'read by Asad on %s\n' "$(date -Iseconds)" > "$V44_LOG_DIR/GPU1_INSPECTED_BY"
```

The phases do not chain; `gpu2` refuses to start until this file exists.

---

## 5. GPU-2 — reportable blind labels

```bash
scripts/v44_gpu_runs.sh gpu2

jq '{primary_gate: .primary_gate, counts: .counts, disagreements: .disagreements}' \
  "$V44/DETECTOR_V4_4_LABEL_REPORT.json"
```

Balanced-panel gate:

```
kappa               >= 0.70
raw agreement       >= 0.85
NONE agreement      >= 0.80
ANSWER agreement    >= 0.80
PARTIAL agreement   >= 0.70
malformed / missing / truncated / unresolved / provenance failures   all 0
```

If it fails, **STOP**. Do not adjudicate a failed panel.

### Blind adjudication

```bash
cp "$V44/V4_4_BLIND_DISAGREEMENTS.jsonl" "$V44/V4_4_BLIND_ADJUDICATION.jsonl"
nano "$V44/V4_4_BLIND_ADJUDICATION.jsonl"
```

Replace every `"answer_attempt": null` with exactly one of `NONE` / `PARTIAL` / `ANSWER`.
**Do not open the reference key during this pass.**

```bash
python - <<'PY'
import json
from pathlib import Path
v = Path("data/cohorts/graph_unlearning_v1/detector_v4_4")
src = [json.loads(x) for x in (v/"V4_4_BLIND_DISAGREEMENTS.jsonl").read_text().splitlines() if x]
dec = [json.loads(x) for x in (v/"V4_4_BLIND_ADJUDICATION.jsonl").read_text().splitlines() if x]
allowed = {"NONE", "PARTIAL", "ANSWER"}
assert {r["audit_id"] for r in src} == {r["audit_id"] for r in dec}
bad = [r["audit_id"] for r in dec if r.get("answer_attempt") not in allowed]
assert not bad, bad[:10]
print("blind adjudication complete:", len(dec))
PY
```

---

## 6. GPU-3A / 3B — reference axis and label authority

```bash
scripts/v44_gpu_runs.sh gpu3a-reference

cp "$V44/V4_4_REFERENCE_DISAGREEMENTS.jsonl" "$V44/V4_4_REFERENCE_ADJUDICATION.jsonl"
nano "$V44/V4_4_REFERENCE_ADJUDICATION.jsonl"
```

Set each `"reference_content": null` to `YES` / `NO` / `UNCERTAIN`. Reference answers **are**
allowed on this axis — it is offline and never enters detector training.

```bash
python - <<'PY'
import json
from pathlib import Path
v = Path("data/cohorts/graph_unlearning_v1/detector_v4_4")
src = [json.loads(x) for x in (v/"V4_4_REFERENCE_DISAGREEMENTS.jsonl").read_text().splitlines() if x]
dec = [json.loads(x) for x in (v/"V4_4_REFERENCE_ADJUDICATION.jsonl").read_text().splitlines() if x]
allowed = {"YES", "NO", "UNCERTAIN"}
assert {r["audit_id"] for r in src} == {r["audit_id"] for r in dec}
bad = [r["audit_id"] for r in dec if r.get("reference_content") not in allowed]
assert not bad, bad[:10]
print("reference adjudication complete:", len(dec))
PY

scripts/v44_gpu_runs.sh gpu3b-authority
jq . "$V44/DETECTOR_V4_4_LABEL_AUTHORITY.json"
```

Continue only if authority is green, both axes are closed, unresolved is zero, the
trainable field is `answer_attempt`, and the bundle/panel hashes are unchanged.

### Checkpoint the evidence before training

```bash
git status --short --untracked-files=all
git diff --check

git add \
  "$V44/V4_4_BLIND_JUDGE_A.jsonl" "$V44/V4_4_BLIND_JUDGE_B.jsonl" \
  "$V44/V4_4_REFERENCE_JUDGE_A.jsonl" "$V44/V4_4_REFERENCE_JUDGE_B.jsonl" \
  "$V44/V4_4_BLIND_DISAGREEMENTS.jsonl" "$V44/V4_4_BLIND_ADJUDICATION.jsonl" \
  "$V44/V4_4_REFERENCE_DISAGREEMENTS.jsonl" "$V44/V4_4_REFERENCE_ADJUDICATION.jsonl" \
  "$V44/V4_4_ADJUDICATED.jsonl" "$V44/V4_4_REFERENCE_ADJUDICATED.jsonl" \
  "$V44/DETECTOR_V4_4_LABEL_REPORT.json" "$V44/DETECTOR_V4_4_LABEL_AUTHORITY.json"

git commit -m "detector v4.4: freeze label authority evidence"
git push -u origin evidence/detector-v44-final
```

If a filename differs, find it with `git status --short --untracked-files=all` and add it
explicitly. **Never `git add .`**

---

## 7. GPU-4 — DeBERTa training and checkpoint backup

```bash
scripts/v44_gpu_runs.sh gpu4-train
jq . "$V44_LOG_DIR/model/DETECTOR_V4_MODEL.json"
```

Sequence: CUDA training smoke → baselines → seeds 20260814/15/16 → development checkpoint
selection → shortcut and input ablations → frozen model manifest. **Do not pick a
different seed because it scores better on heldout.**

```bash
export V44_MODEL_ARTIFACT="$V44_LOG_DIR/model/DETECTOR_V4_MODEL.json"
export SELECTED_CHECKPOINT="$(python - <<'PY'
import json
print(json.load(open("/workspace/rdl-artifacts/v44/model/DETECTOR_V4_MODEL.json"))["selected_checkpoint"])
PY
)"
echo "$SELECTED_CHECKPOINT"; test -e "$SELECTED_CHECKPOINT"

cp "$V44_LOG_DIR/model/DETECTOR_V4_MODEL.json" "$V44/DETECTOR_V4_MODEL.json"
```

Weights never enter Git. Back them up to a private Hugging Face model repo:

```bash
curl -LsSf https://hf.co/cli/install.sh | bash -s
export PATH="$HOME/.local/bin:$PATH"
hf version && hf auth login && hf auth whoami

export HF_DETECTOR_REPO="<your-hf-username>/graphforget-detector-v4-4"
hf repos create "$HF_DETECTOR_REPO" --private --exist-ok
hf upload-large-folder "$HF_DETECTOR_REPO" "$SELECTED_CHECKPOINT" --private --num-workers 4
```

Then record a hash-bound retrieval manifest in Git:

```bash
python - <<'PY'
import hashlib, json, os, time
from pathlib import Path

model_path = Path("/workspace/rdl-artifacts/v44/model/DETECTOR_V4_MODEL.json")
model = json.loads(model_path.read_text())
ckpt = Path(model["selected_checkpoint"])
files = []
for p in sorted(ckpt.rglob("*")):
    if p.is_file():
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        files.append({"path": str(p.relative_to(ckpt)), "bytes": p.stat().st_size,
                      "sha256": h.hexdigest()})

payload = {
    "schema": "graph-detector-v4-4-checkpoint-retrieval-v1",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_artifact": str(model_path),
    "model_artifact_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "selected_checkpoint_original_path": str(ckpt),
    "selected_checkpoint_digest": model.get("checkpoint_digest"),
    "external_repo": os.environ["HF_DETECTOR_REPO"],
    "private": True,
    "files": files,
}
out = Path("data/cohorts/graph_unlearning_v1/detector_v4_4/CHECKPOINT_RETRIEVAL_MANIFEST.json")
out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(out)
PY

git add "$V44/DETECTOR_V4_MODEL.json" "$V44/CHECKPOINT_RETRIEVAL_MANIFEST.json"
git commit -m "detector v4.4: record selected checkpoint and retrieval manifest"
git push
```

Do not destroy the instance until the remote copy is verified.

---

## 8. GPU-5A — engineering bank (graph environment)

vLLM never goes into the detector environment.

```bash
deactivate
python -m venv /workspace/venvs/rdl-v44-graph --system-site-packages
source /workspace/venvs/rdl-v44-graph/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-gpu-ampere.txt
python -m pip install -e ".[cpu,dev]"

scripts/v44_gpu_runs.sh gpu5-bank
sed -n '1,240p' "$V42/ENGINEERING_BANK_RUNS.sh"
```

Seeds must read exactly `natural 50241-50244`, `retain 51241-51244`.

```bash
bash data/cohorts/graph_unlearning_v1/detector_v4_2/ENGINEERING_BANK_RUNS.sh

python -m rdl.cli graph-detector-v4-2-build-bank --bank engineering \
  --natural-run runs/graph/engineering-natural-seed50241 \
  --natural-run runs/graph/engineering-natural-seed50242 \
  --natural-run runs/graph/engineering-natural-seed50243 \
  --natural-run runs/graph/engineering-natural-seed50244 \
  --retain-run runs/graph/engineering-retain-seed51241 \
  --retain-run runs/graph/engineering-retain-seed51242 \
  --retain-run runs/graph/engineering-retain-seed51243 \
  --retain-run runs/graph/engineering-retain-seed51244 \
  --check-only
```

Re-run without `--check-only` to build. Then return to the detector environment and
re-export `HF_HOME`, `V44_LOG_DIR`, `TOKENIZERS_PARALLELISM`, `REPO`, `V44`, `V43`, `V42`,
`V44_MODEL_ARTIFACT`.

---

## 9. GPU-5B — fresh engineering labels

```bash
scripts/v44_gpu_runs.sh gpu5-label
```

Frozen plan: development 1,200 rows, heldout 1,200, final 1,200 — each with ≥400 likely
non-attempt. The final partition was frozen before its data existed.

For **development**, then identically for **heldout**:

```bash
cp "$V44/V4_4_FRESH_development_BLIND_DISAGREEMENTS.jsonl" \
   "$V44/V4_4_FRESH_development_BLIND_ADJUDICATION.jsonl"
nano "$V44/V4_4_FRESH_development_BLIND_ADJUDICATION.jsonl"     # NONE|PARTIAL|ANSWER

scripts/v44_gpu_runs.sh gpu5-reference development

cp "$V44/V4_4_FRESH_development_REFERENCE_DISAGREEMENTS.jsonl" \
   "$V44/V4_4_FRESH_development_REFERENCE_ADJUDICATION.jsonl"
nano "$V44/V4_4_FRESH_development_REFERENCE_ADJUDICATION.jsonl" # YES|NO|UNCERTAIN

python -m rdl.cli graph-detector-v4-4-fresh-label-report --partition development \
  --adjudication "$V44/V4_4_FRESH_development_BLIND_ADJUDICATION.jsonl" \
  --reference-adjudication "$V44/V4_4_FRESH_development_REFERENCE_ADJUDICATION.jsonl"
```

**Heldout is labelled before any threshold is seen.** Verify both:

```bash
test -f "$V44/DETECTOR_V4_4_FRESH_LABELLED_development.json"
test -f "$V44/DETECTOR_V4_4_FRESH_LABELLED_heldout.json"
```

---

## 10. GPU-5C — thresholds, then the one-shot engineering gate

```bash
scripts/v44_gpu_runs.sh gpu5-gate

jq . "$V44/DETECTOR_V4_4_OPERATING_POINT.json"
jq . "$V44/DETECTOR_V4_4_FROZEN_DETECTOR.json"
jq . "$V44/DETECTOR_V4_4_HELDOUT_GATE.json"
```

`tau_answer` and `tau_partial` are frozen on fresh **development** only; heldout then opens
once, with both frozen thresholds applied.

```
ANSWER micro recall           >= 0.80
ANSWER macro recall           >= 0.75
correct-concept precision     >= 0.80
protected-nonattempt FPR      <= 0.10
retain end-to-end FPR         <= 0.10
zero-recall concepts / unresolved labels / candidate truncations    all 0
```

If heldout fails, **STOP**. Do not reopen it. v4.4 does not proceed to humans.

---

## 11. Human preparation — the end of Rental A

`human-prepare` now draws the sample **and** scores it in the same phase — "before the
raters see their files" is only enforceable if nothing can happen in between:

```bash
scripts/v44_gpu_runs.sh human-prepare
```

Then build the 250-row consensus, naming both adjudicated sources the draw came from:

```bash
python -m rdl.cli graph-detector-v4-4-human-model-consensus \
  --adjudicated "$V44/V4_4_ADJUDICATED.jsonl" \
  --adjudicated "$V44/DETECTOR_V4_4_FRESH_LABELLED_development.json"
```

It refuses anything that is not an exact cover of the sample, so a wrong source file is a
refusal rather than a quietly short consensus.

```bash
test -f "$V44/V4_4_HUMAN_SAMPLE.json"
test -f "$V44/V4_4_HUMAN_SAMPLE_KEY.json"
test -f "$V44/V4_4_HUMAN_MODEL_CONSENSUS.jsonl"
test -f "$V44/V4_4_HUMAN_DETECTOR_PREDICTIONS.jsonl"
test -f "$V44/V4_4_HUMAN_BLIND_RATER_A.jsonl"
test -f "$V44/V4_4_HUMAN_BLIND_RATER_B.jsonl"
```

Commit the small evidence
explicitly, push, verify the checkpoint exists remotely, then release the instance. Do not
commit weights, caches, raw graph trajectories, or `*.partial.jsonl`.

---

## 12. Human validation — off GPU

Each rater gets exactly one file and sets `answer_attempt` to `NONE|PARTIAL|ANSWER`.
No reference answers, model labels, detector predictions, scores, populations, strata, or
source model. No discussion between raters.

```bash
python -m rdl.cli graph-detector-v4-4-human-import --pass blind
jq . "$V44/V4_4_HUMAN_BLIND_AGREEMENT.json"          # kappa < 0.70 -> STOP
```

Write one decision row per disagreement — `{"audit_id":"...","answer_attempt":"..."}` —
then:

```bash
python -m rdl.cli graph-detector-v4-4-human-adjudicate --pass blind \
  --decisions "$V44/V4_4_HUMAN_BLIND_DECISIONS.jsonl"
```

Only after the blind files are complete and frozen:

```bash
python -m rdl.cli graph-detector-v4-4-human-reference-pass \
  --fresh-audit "$V44/DETECTOR_V4_4_FRESH_AUDIT.json" \
  --fresh-reference-key "$V44/DETECTOR_V4_4_FRESH_REFERENCE_KEY.json"
```

Raters set `reference_content` to `YES|NO|UNCERTAIN`; then `human-import --pass reference`
and `human-adjudicate --pass reference --decisions .../V4_4_HUMAN_REFERENCE_DECISIONS.jsonl`.

### The report

```bash
python -m rdl.cli graph-detector-v4-4-human-report \
  --model-consensus      "$V44/V4_4_HUMAN_MODEL_CONSENSUS.jsonl" \
  --detector-predictions "$V44/V4_4_HUMAN_DETECTOR_PREDICTIONS.jsonl" \
  --frozen-detector      "$V44/DETECTOR_V4_4_FROZEN_DETECTOR.json"

jq '{passed: .passed, human_kappa: .human_human.kappa,
     model_macro_f1: .model_consensus_vs_human.macro_f1,
     failures: .failures, provenance_failures: .provenance_failures}' \
  "$V44/DETECTOR_V4_4_HUMAN_REPORT.json"
```

```
human-human kappa                    >= 0.70
model consensus vs human macro-F1    >= 0.80
model recall on human ANSWER         >= 0.85
recall NONE / PARTIAL / ANSWER       >= 0.75
unresolved / provenance failures     0
```

If human validation fails, **STOP** — the final bank stays sealed. Otherwise commit and
push the human evidence (sample, key, both rater files per pass, both agreement files,
both adjudicated files, consensus, predictions, report).

---

## 13. Rental B — restore exactly

Same 3090 class and image. Bootstrap and auth as in §3, then:

```bash
cd /workspace && gh repo clone asad073-ui/Re-derivation-Leakage
cd Re-derivation-Leakage
git fetch origin && git switch evidence/detector-v44-final && git pull --ff-only
git rev-parse HEAD        # must match the §1 execution commit
git status --short
```

Recreate the detector environment and exports (§3). Restore the checkpoint to the exact
path the model artifact names:

```bash
export SELECTED_CHECKPOINT="$(python - <<'PY'
import json
print(json.load(open("data/cohorts/graph_unlearning_v1/detector_v4_4/DETECTOR_V4_MODEL.json"))["selected_checkpoint"])
PY
)"
mkdir -p "$SELECTED_CHECKPOINT"

hf auth login
export HF_DETECTOR_REPO="<your-hf-username>/graphforget-detector-v4-4"
hf download "$HF_DETECTOR_REPO" --local-dir "$SELECTED_CHECKPOINT"
```

**Verify every file against `CHECKPOINT_RETRIEVAL_MANIFEST.json` before any final bank is
opened.** A checkpoint digest that changed is a stop rule, not a warning.

```bash
mkdir -p "$V44_LOG_DIR/model"
cp "$V44/DETECTOR_V4_MODEL.json" "$V44_LOG_DIR/model/DETECTOR_V4_MODEL.json"
export V44_MODEL_ARTIFACT="$V44_LOG_DIR/model/DETECTOR_V4_MODEL.json"

make cpu-all
python scripts/v44_pipeline_dryrun.py
scripts/v44_gpu_runs.sh verify-gpu
```

---

## 14. Unseal, then build the final bank exactly once

```bash
scripts/v44_gpu_runs.sh final-bank
```

This refuses unless the reportable human validation passed. The refusal is real: the
builder re-verifies the unseal record rather than trusting its presence, so copying a
record from another checkout does not work.

In the graph environment (recreate per §8):

```bash
python -m rdl.cli graph-detector-v4-2-plan-bank-runs --bank final
cat data/cohorts/graph_unlearning_v1/detector_v4_2/FINAL_BANK_RUN_PLAN.json
sed -n '1,240p' data/cohorts/graph_unlearning_v1/detector_v4_2/FINAL_BANK_RUNS.sh
```

It must read `natural 40241 40242 40243 40244` and `retain 40241 40242 40243 40244` — the
same four values, two separate draws, different run IDs. **No engineering seed may appear.**

```bash
bash data/cohorts/graph_unlearning_v1/detector_v4_2/FINAL_BANK_RUNS.sh
```

Expected outputs: `runs/graph/final-{natural,retain}-seed4024{1..4}`.

```bash
python -m rdl.cli graph-detector-v4-2-build-bank --bank final \
  --natural-run runs/graph/final-natural-seed40241 \
  --natural-run runs/graph/final-natural-seed40242 \
  --natural-run runs/graph/final-natural-seed40243 \
  --natural-run runs/graph/final-natural-seed40244 \
  --retain-run  runs/graph/final-retain-seed40241 \
  --retain-run  runs/graph/final-retain-seed40242 \
  --retain-run  runs/graph/final-retain-seed40243 \
  --retain-run  runs/graph/final-retain-seed40244 \
  --check-only
```

Check-only must write `FINAL_GATE_BANK_VERIFICATION.json` — if it writes
`ENGINEERING_BANK_VERIFICATION.json`, the lifecycle PR did not land and you have just
overwritten engineering evidence. Then build **once** (same command, no `--check-only`).
A second build must be refused and `FINAL_GATE_BANK_BUILD_RECORD.json` must exist.

---

## 15. Final audit, the one-shot gate, finalize

Back in the detector environment:

```bash
scripts/v44_gpu_runs.sh final-audit

cp "$V44/V4_4_FRESH_final_BLIND_DISAGREEMENTS.jsonl" \
   "$V44/V4_4_FRESH_final_BLIND_ADJUDICATION.jsonl"
nano "$V44/V4_4_FRESH_final_BLIND_ADJUDICATION.jsonl"        # NONE|PARTIAL|ANSWER

scripts/v44_gpu_runs.sh final-reference

cp "$V44/V4_4_FRESH_final_REFERENCE_DISAGREEMENTS.jsonl" \
   "$V44/V4_4_FRESH_final_REFERENCE_ADJUDICATION.jsonl"
nano "$V44/V4_4_FRESH_final_REFERENCE_ADJUDICATION.jsonl"    # YES|NO|UNCERTAIN

python -m rdl.cli graph-detector-v4-4-fresh-label-report --partition final \
  --adjudication "$V44/V4_4_FRESH_final_BLIND_ADJUDICATION.jsonl" \
  --reference-adjudication "$V44/V4_4_FRESH_final_REFERENCE_ADJUDICATION.jsonl"

test -f "$V44/DETECTOR_V4_4_FRESH_LABELLED_final.json"
```

Then, exactly once:

```bash
scripts/v44_gpu_runs.sh final-gate
jq . "$V44/DETECTOR_V4_4_FINAL_GATE.json"
```

**Do not use `--reopen`.** A failed final gate means v4.4 is not deployable; preserve the
failure.

```bash
scripts/v44_gpu_runs.sh finalize        # or: python -m rdl.cli graph-detector-v4-4-finalize
jq . "$V44/DETECTOR_V4_4_FINAL_VALIDATION.json"
```

Must report `deployable = true` and `answerability_v4_ready = true`, with every upstream
hash bound.

---

## 16. Final GitHub evidence

```bash
git status --short --untracked-files=all
git diff --check
make cpu-all
python scripts/v44_pipeline_dryrun.py
find data runs -type f -size +50M -print          # expect nothing
```

Never stage: `*.partial.jsonl`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, the HF cache,
the vLLM cache, raw graph trajectory shards, or any token.

```bash
git add \
  data/cohorts/graph_unlearning_v1/detector_v4_4 \
  data/cohorts/graph_unlearning_v1/detector_v4_2/FINAL_GATE_BANK.json \
  data/cohorts/graph_unlearning_v1/detector_v4_2/FINAL_BANK_UNSEAL_RECORD.json \
  data/cohorts/graph_unlearning_v1/detector_v4_2/FINAL_GATE_BANK_BUILD_RECORD.json

git add runs/graph/final-natural-seed4024{1,2,3,4} runs/graph/final-retain-seed4024{1,2,3,4}

git diff --cached --name-only | grep -E '\.(safetensors|bin|pt|pth|ckpt|partial\.jsonl)$'   # expect empty
git diff --cached --stat
git diff --cached --check
```

```bash
git commit -m "detector v4.4: freeze final validated detector"
git push

gh pr create --base main --head evidence/detector-v44-final \
  --title "detector v4.4: final validated detector" \
  --body "Freezes v4.4 label authority, selected detector manifest, engineering heldout gate, human validation, sealed final-bank evidence, one-shot final gate, and hash-bound final validation. Model weights remain in the external checkpoint repository and are bound by digest."

gh pr checks --watch
gh pr diff --stat
```

Merge only when CI is green **and** the validation record says deployable:

```bash
gh pr merge --squash --delete-branch
git switch main && git pull --ff-only
git tag -a detector-v4.4-final-20260819 -m "GraphForget detector v4.4: final validated detector"
git push origin detector-v4.4-final-20260819
git rev-parse HEAD
```

---

## 17. What a reviewer must be able to trace

```
DETECTOR_V4_4_FINAL_VALIDATION.json
  ├── final gate report
  ├── final labelled audit
  ├── final bank + build record
  ├── final unseal record
  ├── human validation report
  ├── frozen detector
  ├── operating point
  ├── model artifact
  ├── checkpoint retrieval manifest + SHA256
  ├── v4.4 label authority
  └── judge run records / adjudications
```

Checkpoint bytes stay outside ordinary Git but remain recoverable and hash-verifiable.

---

## 18. Non-negotiable stop rules

Stop and preserve evidence on any of: GPU-1 rubric looks wrong; GPU-2 balanced-panel gate
fails; any malformed, missing, or truncated judge row; any unresolved adjudication; label
authority not green; the reportable training run fails; the shortcut ablation gate fails;
no development operating point satisfies both FPR ceilings; the engineering heldout gate
fails; the human gate fails; the final unseal refuses; final bank build verification fails;
the final audit cannot meet its frozen sampling minimum; the final gate fails; the
checkpoint digest changes; or a frozen bundle, panel, or store hash changes.

Never repair a failed reportable result by weakening a bound, adding training examples
after seeing heldout, choosing a seed from heldout, rerunning humans until agreement rises,
reopening engineering heldout and reporting the better run, rebuilding the final bank after
seeing a final result, reopening the final gate, or changing `tau_answer` / `tau_partial`
after the final bank exists.

**A failed final gate means a new detector version and a new, untouched final bank.**

---

## 19. Execution map

```
merge PR #58  ->  CPU lifecycle PR  ->  RTX 3090 Rental A
                                            |
   verify-gpu -> GPU-1 smoke -> manual inspection -> GPU-2 blind labels
      -> blind adjudication -> GPU-3A reference -> reference adjudication
      -> GPU-3B authority -> GPU-4 training -> checkpoint backup
      -> GPU-5A engineering bank -> GPU-5B fresh labels
      -> development thresholds -> ONE-SHOT engineering heldout gate
      -> human sample + frozen predictions -> Rental A ends
                                            |
   two-human blind + reference validation -> human report PASS
                                            |
                                    RTX 3090 Rental B
      -> unseal -> ONE final-bank build -> final blind/reference labels
      -> ONE-SHOT final gate -> finalize -> evidence PR -> CI green -> merge + tag
```

Detector v4.4 is finished at the tag, not at the end of training.
