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
| GPU | RTX 3090 ×1 | 24 GB; `min_vram_gb: 20` refuses anything smaller |
| rental | **on-demand** for the grid | interruptible instances are paused when outbid; fine for a 5-item smoke test, not for a 7-condition grid |
| reliability | ≥ 95 % | |
| disk | 80–100 GB | **cannot be resized after creation**, and storage bills while the instance is stopped |
| image | CUDA/PyTorch; **devel** only if you want FlashAttention-2 | the runtime image has no `nvcc`, so `flash-attn` cannot build |
| SSH | direct, key-based | Vast starts sessions inside tmux — use it, the grid outlives a dropped connection |

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

* **Option A — SDPA.** Do nothing. Slower, zero build risk, correct. Recommended for the
  first smoke test.
* **Option B — FA2.** Use a CUDA *devel* image, then
  `INSTALL_FLASH_ATTN=1 bash scripts/01_bootstrap_openunlearning.sh` (~20 min build).

Either is fine. **Never mix them inside one comparison** — the implementation is recorded
in every report (`hardware.recommended_attn`), so a mixed table is detectable after the
fact, but not fixable without a rerun.

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
ENV_NAME=vast_rtx3090 bash scripts/00_env_check.sh    # refuses the wrong instance
ENV_NAME=vast_rtx3090 bash scripts/02_repro_tofu_npo_forget10.sh --batch-size 1
```

That script runs all four required steps in order. To drive them individually:

```bash
python -m rdl.cli env-check --env vast_rtx3090

# parity with upstream's eval default (32) …
python -m rdl.cli run-repro --target full --env vast_rtx3090 --batch-size 32
# … and the pre-registered deterministic protocol (1)
python -m rdl.cli run-repro --target full --env vast_rtx3090 --batch-size 1

python -m rdl.cli run-repro --target npo_forget10 --env vast_rtx3090 --batch-size 32
python -m rdl.cli run-repro --target npo_forget10 --env vast_rtx3090 --batch-size 1

# Agent B: MEASURED, never compared to agent A's published row
python -m rdl.cli run-repro \
  --model-path open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10 \
  --measure-only --checkpoint-label agent_b_independent --env vast_rtx3090
```

Gates: `full` → model_utility 0.60 / forget_truth_ratio 0.48; `npo_forget10` → 0.46 /
0.70, both ±0.01. `forget_quality` is reported, never gated. All three runs above are
**prerequisites**: `make-report` blocks the grid without them (ADR-0029).

## Phase D — 5-item GPU smoke test

```bash
python -m rdl.cli run-condition --condition configs/conditions/C1W.yaml \
  --env vast_rtx3090 --seeds 1 --limit 5
python -m rdl.cli run-condition --condition configs/conditions/C3D.yaml \
  --env vast_rtx3090 --seeds 1 --limit 5
```

## Phase E — small pilot

```bash
ENV_NAME=vast_rtx3090 SEEDS=1 CONDITIONS="C1W C3 C3D C3C" \
  bash scripts/03_run_phase0_grid.sh --limit 20
```

## Phase F — the full grid

```bash
ENV_NAME=vast_rtx3090 SEEDS=5 CONDITIONS="C0 C1 C1W C2 C3 C3D C3C" \
  bash scripts/03_run_phase0_grid.sh
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
| `requires CUDA, but no CUDA device was detected` | torch has no CUDA build, or the container has no GPU passthrough. Check `nvidia-smi`. |
| `requires bfloat16, but … has no bf16 datapath` | this is a T4. Use `--env colab_t4`. |
| `pins attn_implementation=flash_attention_2, but the flash_attn package is not installed` | build it, or leave `attn_implementation: auto`. |
| `expects at least 60 GB free disk` | two checkpoints plus eval logs do not fit; disk cannot be resized. |
