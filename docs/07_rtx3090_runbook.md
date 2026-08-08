# RTX 3090 (Vast.ai) runbook — Phase 0

The production environment for Phase 0. Colab T4 remains supported (`--env colab_t4`)
and is no longer the default.

## Why the move

| | Colab T4 | RTX 3090 |
|---|---|---|
| VRAM | 16 GB | 24 GB — holds agent A **and** agent B in bf16 simultaneously (C3D/C3C) |
| bf16 | no (SM 7.5) | yes (SM 8.6) — GA/NPO objectives are numerically sound |
| FlashAttention-2 | impossible | possible (see below — possible is not automatic) |
| session | dies, unpredictably | persistent instance |

What the 3090 does **not** buy: a training reproduction. 24 GB single-GPU cannot
reproduce upstream's 2× L40S DeepSpeed ZeRO-3 effective-batch-32 setup, and upstream warn
their numbers shift when the distributed setup changes. **Days 1–2 remain an evaluation
reproduction on published checkpoints.** Training here would produce a new number and
must be recorded as one (ADR-0003, ADR-0027).

## Instance selection

| filter | value | why |
|---|---|---|
| GPU | RTX 3090 ×1 | 24 GB. The profile checks the **name** too: `min_vram_gb: 20` alone would also accept a 4090 or an A6000, and a report stamped `vast_rtx3090` must not have run on one |
| rental | **on-demand** for the grid | interruptible instances are paused when outbid; fine for a 5-item smoke test, not for a 7-condition grid |
| reliability | ≥ 95 % | |
| disk | 80–100 GB | **cannot be resized after creation**, and storage bills while the instance is stopped |
| **Python** | **3.11** | open-unlearning declares `python_requires >= 3.11`. An arbitrary CUDA image may ship 3.10 and the submodule install fails on it. `min_python: "3.11"` refuses the box up front |
| image | CUDA/PyTorch **devel** if you want FlashAttention-2 | the runtime image has no `nvcc`, so `flash-attn` cannot build |
| SSH | direct, key-based | Vast starts sessions inside tmux — use it, the grid outlives a dropped connection |

Check the image before renting, not after:

```bash
python --version     # must be 3.11.x
nvidia-smi           # NVIDIA GeForce RTX 3090, ~24 GiB
nvcc --version       # only needed for the FA2 path
```

Destroy the instance once results are pushed; a stopped instance still costs storage.

## FlashAttention-2: choose once, record always

`hardware.detect()` reports **two** facts, not one:

```
fa2_hardware=True      SM 8.6 >= 8.0
fa2_installed=False    `import flash_attn` would fail here
recommended_attn=sdpa  because BOTH are required
```

Upstream's `configs/model/Llama-3.2-1B-Instruct.yaml` hard-codes
`attn_implementation: flash_attention_2`. On a bare 3090 image that default is an
`ImportError` inside `from_pretrained` — *after* the 2.5 GB checkpoint has downloaded.
The bridge's unconditional override is what prevents it.

The probe does not stop at `find_spec`: it imports `flash_attn` in a **subprocess**,
because a wheel built against a different torch or CUDA is locatable but not importable,
and that failure would otherwise land inside `from_pretrained`.

* **Option B — FA2. Recommended for the parity gate.** Upstream's published numbers were
  produced under `flash_attention_2`, so this is the closest available reproduction. Use
  a CUDA *devel* image, then
  `INSTALL_FLASH_ATTN=1 bash scripts/01_bootstrap_openunlearning.sh` (~20 min build).
  Verify before the gate:

  ```
  fa2_hardware=True  fa2_installed=True  recommended_attn=flash_attention_2
  ```

* **Option A — SDPA.** Do nothing. Slower, zero build risk, and the eval runs. But the
  run is then off-parity on attention, and the report says so in `parity_gaps`. Fine for
  the first smoke test; not what you want the headline Day-1 gate to have used.

Either way, **never mix them inside one comparison** — the implementation is recorded in
every report, so a mixed table is detectable after the fact, but not fixable without a
rerun.

---

## Phase A — environment

```bash
export HF_HOME=/workspace/hf
export TOKENIZERS_PARALLELISM=false

cd /workspace
git clone --recurse-submodules <PRIVATE_REPO_URL>
cd Re-derivation-Leakage

python -m pip install --upgrade pip
bash scripts/01_bootstrap_openunlearning.sh     # INSTALL_FLASH_ATTN=1 for option B
pip install -e ".[cpu,dev]"
```

Everything lives under `/workspace` — the persistent volume. `$HOME` and `/content` are
the container overlay: small, and gone on the next start.

## Phase B — the CPU gate, on the box that will run the grid

```bash
make cpu-all
python -m pytest tests/integration -q     # needs network, ~1 MB model
```

## Phase C — Day 1–2 reproduction

```bash
ENV_NAME=vast_rtx3090 STRICT=1 bash scripts/00_env_check.sh   # full preflight
ENV_NAME=vast_rtx3090 bash scripts/02_repro_tofu_npo_forget10.sh
```

`--strict` is the one that stops a bad session in four seconds. It fails on: the wrong
GPU, no bf16, too little VRAM or disk, Python < 3.11, a missing `HF_TOKEN`, a blocked
Llama licence, **any checkpoint unreachable at its pinned revision**, a missing or
off-pin submodule, a missing package, or absent retain eval logs.

The script runs the two Day-1 gates in the right order. Individually:

```bash
# RUN A — PUBLISHED PARITY. batch 32 / seed 0 are upstream's own eval defaults and are
# what produced the published row, so this is the trust gate. These are also run-repro's
# defaults; they are spelled out here because the distinction is the point.
python -m rdl.cli run-repro --target full         --env vast_rtx3090 --batch-size 32 --seed 0
python -m rdl.cli run-repro --target npo_forget10 --env vast_rtx3090 --batch-size 32 --seed 0

# RUN B — the pre-registered deterministic protocol. Reported, NOT a substitute.
python -m rdl.cli run-repro --target full         --env vast_rtx3090 --batch-size 1 --seed 42
python -m rdl.cli run-repro --target npo_forget10 --env vast_rtx3090 --batch-size 1 --seed 42

# Agent B: MEASURED, never compared to agent A's published row. At parity settings so it
# is comparable with A's parity run.
python -m rdl.cli run-repro \
  --model-path open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10 \
  --measure-only --checkpoint-label agent_b_independent \
  --env vast_rtx3090 --batch-size 32 --seed 0
```

Documented rows: `full` → model_utility 0.60 / forget_truth_ratio 0.48; `npo_forget10` →
0.46 / 0.70, both ±0.01. `forget_quality` is reported, never gated.

**`npo_forget10` is known NOT to reproduce** — measured 0.43237 / 0.64140 at revision
`94ed64eb` under two independent evaluation environments. Under
`study_mode: released_artifact` that is the recorded finding, not a blocker; `full` is
what validates the evaluation stack and it still blocks. Run the NPO target anyway: the
grid needs the measurement, and a *change* in the mismatch would be news.

| outcome | what to do |
|---|---|
| `full` passes at parity | the evaluation stack is validated — proceed |
| `full` passes at parity, batch-1 misses | proceed; the batching effect is a recorded protocol finding |
| **`full` misses at parity** | **stop.** Bisect one change at a time: chat template → padding_side → batch_size → dtype → attention → transformers version. Log each attempt in `docs/04_decisions.md` |
| `npo_forget10` misses **at 0.43237 / 0.64140** | the EXPECTED result. Recorded as `published_artifact_parity: FAIL`; proceed |
| `npo_forget10` misses at **different** numbers | stop — that is a new mismatch, not the recorded one. Bisect, and set `study_mode` back to `published_reproduction` while you do |

`make-report` blocks the grid unless `full` passed **at parity** and agent B was measured
(ADR-0029, ADR-0033). Under `study_mode: released_artifact` the NPO row's parity is
recorded rather than gated (ADR-0052) — it is never converted into a pass.

> **Day-1 status, 2026-08-08.** `full` and `retain90` reproduce. `npo_forget10` does
> **not**: measured 0.43237 / 0.64140 against the documented 0.460 / 0.700 at revision
> `94ed64eb`, under two independent evaluation environments (ADR-0038, ADR-0039;
> upstream issue #199 open). `DAY1_GATE = BLOCKED_EXTERNAL_ARTIFACT_MISMATCH`. Phase 0
> proceeds as a **released-artifact study** and claims no published-row reproduction —
> see pre-registration v3 §1. The agent-B measure-only run above is doubly required
> under that framing: neither checkpoint's forgetting can be assumed from a table.

## Phase D — 5-item GPU smoke test, then READ THE TRANSCRIPTS

All four estimand arms, including both standalone baselines.

```bash
for C in C1W B1W C3D C3S C3C; do
  python -m rdl.cli run-condition --condition configs/conditions/$C.yaml \
    --env vast_rtx3090 --seeds 1 --limit 5
done
```

**Do not proceed on the summary line alone.** Open the C3C transcripts and check by eye:

```bash
D=$(ls -td results/*/ | head -1)
python - <<'PY'
import json, glob, os
d = sorted(glob.glob("results/*/"), key=os.path.getmtime)[-1]
ev = [json.loads(l) for l in open(d + "transcripts_treatment_seed0.jsonl", encoding="utf-8")]
hand = [e for e in ev if e["kind"] == "handoff"]
print(f"{len(hand)} handoffs in {d}")
print(json.dumps(hand[0], indent=2)[:800])
PY
```

Every C3C episode must carry **exactly one** `handoff` event whose `text` equals agent
A's `agent_answer` text and whose `text_sha256` matches it. Zero handoffs means the arm
is an ensemble under a compositional name — the ADR-0041 failure. A partial rate means
some episodes silently ran as the comparator. `make-report` blocks both (ADR-0054).

Then check **C3S**, which is the arm the primary contrast now rests on. Run the same
snippet against the C3S results directory and confirm, from the transcripts:

- every handoff has `"shuffled": true`
- every `source_item_id` differs from its own episode's `item_id`
- the set of `source_item_id`s is the whole item set, each used once
- every handoff `text` is **non-empty**. An empty handed-over string is a control that
  hands over nothing, which makes C3S a bare-question arm wearing a wrapper's name.

Then read the recorded audit — the same facts `make-report` gates on, so a disagreement
between the two is itself the finding:

```bash
D=$(ls -td results/*/ | head -1)
python - <<'PY'
import glob, json, os
d = sorted(glob.glob("results/*/"), key=os.path.getmtime)[-1]
r = json.load(open(d + "condition_report.json", encoding="utf-8"))
agg = r.get("handoff_audit_aggregate") or {}
print(json.dumps({k: agg.get(k) for k in (
    "n_seeds", "seeds", "mapping_hashes", "mapping_algorithms",
    "same_author_count", "fixed_point_count",
    "target_answer_in_handoff_count", "seeds_with_target_answer_in_handoff",
)}, indent=2))
for a in r.get("handoff_audit_by_seed") or []:
    print(a["seed"], (a.get("mapping") or {}).get("shift"), (a.get("mapping") or {}).get("sha256"))
PY
```

Every one of these must hold, **at every seed**, before the arm is a control:

| field | required |
|---|---|
| `same_author_count` | `0` |
| `fixed_point_count` | `0` |
| `target_answer_in_handoff_count` | `0` |
| `mapping_hashes` | exactly **one** SHA-256 across all seeds |
| `mapping_algorithms` | one entry, containing `cross-author` |
| `mapping.shift` | `20` on the full forget10 set; `1` on a spread-sampled pilot |
| `n_seeds` | equal to the run's `n_seeds` — an unaudited seed is an unchecked control |

The counts come from the union over seeds, not from seed 0: the mapping is
seed-independent, but the handed-over TEXT is not. Under `store_scope: cumulative`
episode order changes the live store, changes agent A's source answer, and can put the
target answer into a seed-3 handoff that was clean at seed 0 (ADR-0057).

If any handoff is unshuffled, C3S is a second copy of C3C and `C3C − C3S` is zero by
construction (ADR-0048). Read three handed-over texts by eye and satisfy yourself they
answer a *different question* than the episode asks — that is the entire control.

`run-condition` also prints `handoffs recorded = N (configured: …)` and shouts in red
when a handoff arm records none.

## Phase E — small pilot, explicitly unreportable

```bash
ENV_NAME=vast_rtx3090 SEEDS=1 CONDITIONS="C1W B1W C3 C3D C3S C3C" PILOT=1 \
  bash scripts/03_run_phase0_grid.sh --limit 20
```

`--limit` marks every report `truncated: true` / `reportable: false`, so the gate blocks
by construction and the script would exit non-zero on a run that did exactly what was
asked. `PILOT=1` runs `make-report --no-gate` instead: tables, no verdict, exit zero.
This phase exists to find crashes and OOMs, not numbers (ADR-0046). Never set `PILOT=1`
on Phase F — it is the one switch that turns the pre-registered criteria off.

## Phase F — the full grid

```bash
ENV_NAME=vast_rtx3090 SEEDS=1 CONDITIONS="C0 C1 C1W B1W C2 C3 C3D C3S C3C" \
  bash scripts/03_run_phase0_grid.sh
```

**One seed, not five.** The primary experiment resets the store before every item, so
episode order — the only thing a seed varies under greedy decoding — cannot change any
outcome. Five seeds would be five copies of one deterministic result, and any spread
between them would be incidental GPU nondeterminism reported as planned replication. The
interval comes from the author-clustered paired item bootstrap, which needs no seeds.
`make-report` blocks any other count at this scope (ADR-0050).

`C3S` and `B1W` are **not optional**. Without C3S, `C3C − C3D` cannot separate agent A's
content from the peer-message wrapper (ADR-0048), and the primary quantity
`content_specific_joint_recovery = C3C AND NOT C3S AND NOT C1W AND NOT B1W` has no
`NOT C3S` term to apply (v5 §3.2). Without B1W neither it nor the secondary
`joint_only_recovery` can be computed at all (ADR-0042). `make-report` blocks a grid
missing either.

**Reading the verdict.** `make-report` now reports two independent facts. `EXPERIMENT:
VALID/INVALID` says whether the grid ran correctly and completely — that is what the exit
code follows. `HYPOTHESIS: SUPPORTED/NOT SUPPORTED` says what it found, and a valid
refutation exits **zero**, because it is a result (ADR-0052). Do not add `|| true`.

### Phase F2 — the longitudinal run (separate experiment)

The primary grid resets the store per item so items are exchangeable. Cumulative
recontamination is a different question and gets its own invocation, reported over seeds
and never differenced against the per-item arms (ADR-0047):

```bash
ENV_NAME=vast_rtx3090 SEEDS=5 CONDITIONS="C1W B1W C3D C3S C3C" \
  bash scripts/03_run_phase0_grid.sh --set episode.store_scope=cumulative
```

`make-report` runs at the end and exits non-zero if the pre-registered gate fails. Do not
add `|| true`.

## Phase G — get the results off the box

```bash
bash scripts/99_sync_results.sh "results: phase0 grid on vast rtx3090"
```

Then destroy the instance.

---

## Running the same conditions elsewhere

The conditions are hardware-independent by construction; only the `env` group moves.

```bash
python -m rdl.cli run-condition --condition configs/conditions/C3D.yaml --env colab_t4 ...
ENV_NAME=colab_t4 bash scripts/03_run_phase0_grid.sh
```

`--env` replaces the whole env group. `--set env=...` cannot: by then `env` is a
populated mapping, so the dotlist either fails validation or (with `env.name=`) relabels
the profile while leaving Colab's `/content` cache path in place. Pinned by
`tests/unit/test_config.py::test_a_dotlist_cannot_switch_the_env_group`.

## If something refuses to start

| message | meaning |
|---|---|
| `requires at least 20 GB of VRAM, but … reports 11.8` | wrong instance rented. Destroy it. |
| `expects a GPU matching /RTX 3090/, but this box reports …` | a different card. It may evaluate fine — but use `--env rtx3090` so the report does not claim hardware it did not run on. |
| `expects compute capability 8.6, but this GPU reports 9.0` | that is an H100. Use `--env h100`. |
| `requires CUDA, but no CUDA device was detected` | torch has no CUDA build, or the container has no GPU passthrough. Check `nvidia-smi`. |
| `requires bfloat16, but … has no bf16 datapath` | this is a T4. Use `--env colab_t4`. |
| `requires Python >= 3.11, but this interpreter is 3.10.x` | wrong image; open-unlearning will not install. |
| `pins attn_implementation=flash_attention_2, but the flash_attn package is not installed` | build it, or leave `attn_implementation: auto`. |
| `expects at least 60 GB free disk` | two checkpoints plus eval logs do not fit; disk cannot be resized. |
| `… is not reachable with this token` | accept the Llama licence, or the pinned revision moved — check `configs/models/`. |
| `open-unlearning is at …, but this repo pins …` | `git submodule update --init --recursive`. |
| `retain-model eval logs missing` | `cd third_party/open-unlearning && python setup_data.py --eval_logs`. |

---

## Moving to the H100 later

Nothing about the conditions changes. Run the same grid with `--env h100` /
`ENV_NAME=h100`; the profile requires ≥70 GB VRAM and Python ≥3.11 and leaves the GPU
name unconstrained. Re-run Days 1–2 on that box first — the reproduction vouches for an
*install*, and the H100 is a different one. Do not pool 3090 and H100 numbers in a single
comparison unless both cleared the parity gate under the same attention implementation.
