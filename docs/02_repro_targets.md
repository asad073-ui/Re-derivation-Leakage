# Reproduction targets — Days 1–2

The ground truth for the Phase-0 reproduction gate, transcribed from
`open-unlearning/docs/repro.md` and the published eval logs.

---

## 1. Published numbers

TOFU, `Llama-3.2-1B-Instruct`, **forget10**:

| Method | forget_quality | model_utility | forget_truth_ratio |
|---|---|---|---|
| Finetuned (`_full`, the target) | 1.66e-21 | **0.60** | **0.48** |
| Retain (`_retain90`) | 1.0 | 0.59 | 0.63 |
| **NPO** | 0.02 | **0.46** | **0.70** |
| GradAscent | 1.06e-239 | 0.00 | 2.25e-18 |
| GradDiff | 1.06e-239 | 0.49 | 3.53e-27 |

Upstream's setup: **2× L40s, DeepSpeed ZeRO-3, lr 1e-5, alpha 1, beta 0.1, effective
batch 32, 10 epochs, `paged_adamw_32bit`.** They warn explicitly that numbers shift when
the distributed setup changes, including on a single GPU.

Encoded in code at `src/rdl/eval/openunlearning_bridge.py::PUBLISHED_TARGETS` and
asserted by `tests/integration/test_ou_bridge_parse.py::test_published_targets_match_the_spec`.

---

## 1a. The checkpoint ids, verified 2026-08-07

| role | repo id |
|---|---|
| target (`full`) | `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` |
| retain oracle | `open-unlearning/tofu_Llama-3.2-1B-Instruct_retain90` |
| **NPO forget10** | `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10` |
| agent B (C3D/C3C) | `open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr2e-05_beta0.5_alpha1_epoch10` |

**The naming trap.** Unlearned checkpoints do NOT follow the
`tofu_<model>_<METHOD>_<split>` pattern of the finetuned and retain ones. They are
published **one repo per hyperparameter setting** as

```
unlearn_tofu_<model>_<forget_split>_<METHOD>_lr<LR>_beta<B>_alpha<A>_epoch<E>
```

Only the **lr1e-05 / beta0.1 / alpha1 / epoch10** run corresponds to the NPO row above —
it is the setting the repro table was generated under. Every other NPO repo is a
different run and its metrics are not 0.46 / 0.70, so gating one of those against those
targets fails for a reason that has nothing to do with your install.

`open-unlearning/tofu_Llama-3.2-1B-Instruct_NPO_forget10` — the id this repo carried
until 2026-08-07 — **has never existed**. Pinned offline by
`tests/unit/test_checkpoint_ids.py`. See ADR-0021.

There is **no published NPO checkpoint for forget01 or forget05** on this architecture.

---

## 1b. The eval command, verified against submodule `4ad738a`

```bash
python src/eval.py --config-name=eval.yaml \
  experiment=eval/tofu/default \
  model=Llama-3.2-1B-Instruct \
  model.model_args.pretrained_model_name_or_path=<CHECKPOINT> \
  model.model_args.attn_implementation=sdpa \
  model.model_args.torch_dtype=float16 \
  forget_split=forget10 \
  holdout_split=holdout10 \
  retain_logs_path=saves/eval/tofu_Llama-3.2-1B-Instruct_retain90/TOFU_EVAL.json \
  seed=42 \
  eval.tofu.batch_size=1 \
  eval.tofu.overwrite=true \
  task_name=<UNIQUE> \
  paths.output_dir=saves/eval/<UNIQUE>
```

Every line that is easy to get wrong, and what happens when you do:

| override | why | if omitted / wrong |
|---|---|---|
| `holdout_split=` | `configs/experiment/eval/tofu/default.yaml` defines `holdout_split`. There is **no `retain_split`** in the eval tree — it exists only in the training configs. | `retain_split=retain90` aborts Hydra with *"Could not override 'retain_split'"* before the model loads |
| `seed=` | `configs/eval.yaml` defaults `seed: 0` | the report says 42, the run used 0 |
| `eval.tofu.batch_size=` | `configs/eval/tofu.yaml` defaults `batch_size: 32` | the determinism commitment is decorative |
| `eval.tofu.overwrite=true` | default is `false`, which **skips** metrics whose logs exist under `output_dir` | a "new" run silently replays an old one |
| unique `task_name` | `paths.output_dir` defaults to `saves/eval/${task_name}` | a crashed run finds the previous run's `SUMMARY.json` where this run's is looked for |

Bootstrap uses **`python setup_data.py --eval_logs`**, not `--eval`. At the pinned SHA
argparse defines `--eval_logs` / `--idk` / `--wmdp` and nothing else, so `--eval` aborts
with *"unrecognized arguments"* before anything downloads.

`build_eval_command` emits all of this from the `HardwareProfile`, and
`tests/unit/test_ou_eval_command.py` asserts each item offline so a wrong override fails
on the laptop rather than on the GPU box.

### Two runs, and what "reproduced" means

The published numbers were produced under upstream's own defaults, read from the pinned
submodule:

| setting | published | source |
|---|---|---|
| `batch_size` | **32** | `configs/eval/tofu.yaml` |
| `seed` | **0** | `configs/eval.yaml` |
| `torch_dtype` | `bfloat16` | `configs/model/Llama-3.2-1B-Instruct.yaml` |
| `attn_implementation` | `flash_attention_2` | same |

**Run A — published parity. This is the trust gate.**

```bash
python -m rdl.cli run-repro --target npo_forget10 --env vast_rtx3090   # defaults: 32 / 0
```

It is the only run whose miss means *our install is wrong* rather than *we changed a
setting*. `run-repro` defaults to these values, records `published_parity: true` on the
report, and `make-report` refuses a grid whose Days 1-2 never passed at parity.

**Run B — the pre-registered deterministic protocol.**

```bash
python -m rdl.cli run-repro --target npo_forget10 --env vast_rtx3090 --batch-size 1 --seed 42
```

Batch 1 because batched generation with left-padding changes greedy output, and the
pre-registration commits to batch 1 for every number in the paper. Reported, never a
substitute for A.

| outcome | reading |
|---|---|
| A passes, B passes | install validated; the protocol does not move the metric |
| A passes, B misses | install is fine — batching moved it. A finding about the protocol, recorded as one |
| A misses | **stop.** Bisect before reading anything downstream |

Agreement within ±0.01 is **within tolerance of**, not **identical to**. Every report
carries `parity_gaps`, naming each setting that differed — including a dtype or
attention gap forced by the hardware (a T4 has neither bf16 nor FA2, so batch/seed
parity is the most it can offer).

---

## 2. What "reproduce" means here — and what it does not

### DO NOT reproduce the training

You cannot match 2× L40s ZeRO-3 bf16 on one GPU, upstream says so themselves. **This
does not change on the RTX 3090.** bf16 makes a gradient-ascent-family objective
numerically *sound* — it does not make one 24 GB card equal to two L40S under ZeRO-3.
Attempting it burns the week and produces a mismatch you cannot interpret.

`hardware.assert_training_allowed` refuses to start training on a device without bf16
unless `--allow-fp16-training` is passed explicitly.

### DO reproduce the evaluation, on a published checkpoint

That is deterministic enough to be a real gate, costs no training, and tests exactly
what Phase 0 depends on: **that our installation of open-unlearning computes their
metrics correctly.**

### DO NOT gate on `forget_quality`

It is the p-value of a KS test between the forget and retain truth-ratio distributions.
It ranges over ~200 orders of magnitude across methods and is unstable to single-sample
changes. Gating a reproduction on `0.02 ± tolerance` is meaningless.

**Report it. Check only that `log10(forget_quality) ∈ [−2, 0]`.**

---

## 3. The gate

**Run order matters.** `full` first — it uses a checkpoint you *know* is published, so
it isolates "is my install correct" from "does the unlearned checkpoint exist".

### Secondary sanity — `tofu_Llama-3.2-1B-Instruct_full` (RUN FIRST)

```
|model_utility − 0.60|      ≤ 0.01
|forget_truth_ratio − 0.48| ≤ 0.01
```

### Primary — the published NPO/forget10/1B checkpoint

```
|model_utility − 0.46|      ≤ 0.01   -> PASS
|forget_truth_ratio − 0.70| ≤ 0.01   -> PASS
forget_quality                        -> REPORT, NOT GATED
```

Determinism requirements for the gated run: greedy decoding, `batch_size=1`, fp16,
`attn_implementation=sdpa`, seed 42, and the resolved versions of
`torch`/`transformers`/`datasets` recorded in §5 below.

Run it:

```bash
python -m rdl.cli run-repro --target full          # sanity, first
python -m rdl.cli run-repro --target npo_forget10  # the gate
```

---

## 4. Fallbacks, in order

### Fallback 1 — the NPO forget10 checkpoint is not on the Hub

**Resolved 2026-08-07: it is on the Hub** (see §1a). Keep this fallback for the case
where a repo is withdrawn or renamed, and keep running
`python -m rdl.cli discover-checkpoints` on day 1 — it now also reports which candidate
matches the published repro hyperparameters.

If absent, the gate becomes the `full` model only, **plus** a diff of locally-recomputed
metrics against the published eval logs. Those logs are the HF dataset
`open-unlearning/eval`, which `python setup_data.py --eval_logs` downloads; the relevant
files are `tofu*/evals*/*_SUMMARY.json`.

Recomputing their metrics from their logs validates the metric code **with no model at
all**, which is a genuinely useful gate.

### Fallback 2 — metrics do not match within tolerance

Bisect in **this order, one change at a time**:

1. chat template
2. `padding_side`
3. `batch_size`
4. dtype
5. `transformers` version

Log every attempt in [`04_decisions.md`](04_decisions.md). Changing two things at once
turns a two-hour bisect into a lost day.

### Fallback 3 — Llama-3.2 gated-repo access denied

Accept the Llama 3.2 community licence on the Hub **with the same account as your
token**: <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>

**Do this on day 1, before anything else.** Approval is usually instant but it is a hard
blocker if you hit it at hour six. `rdl env-check` reports this explicitly.

---

## 5. Resolved versions

> **TO BE FILLED ON THE FIRST RTX 3090 RUN — DO NOT LEAVE BLANK.**
>
> Version drift is the single most likely cause of a failed reproduction. Fill this in
> before running the gate, and pin `requirements-gpu-ampere.txt` to whatever lands here.
>
> ```bash
> python -m rdl.cli env-check --write-versions docs/02_repro_targets.md
> ```
>
> That appends a dated table below this block automatically.

**open-unlearning pinned SHA:** `4ad738aaf60f6a4385f6e2506d01da99e76c31f3`
(`4ad738a` — "Fix pip install command for lm-eval (#180)", recorded 2026-08-07)

Verify it still matches before every gated run — `rdl env-check` reports
`submodule.matches_pin` for exactly this reason.

### Expected, read from the pinned submodule (2026-08-07)

`third_party/open-unlearning/requirements.txt` at `4ad738a` pins these exactly, so this
is what `pip install -e ".[lm-eval]"` inside the submodule will resolve to. Any
preinstalled `torch`/`transformers` in the image **will be moved** by that install — on
Colab, expect a runtime restart afterwards.

| package | pinned by the submodule |
|---|---|
| `torch` | `2.4.1` |
| `transformers` | `4.51.3` |
| `datasets` | `3.0.1` |
| `accelerate` | `0.34.2` |
| `huggingface-hub` | `0.36.0` |
| `numpy` | `2.2.3` |
| `lm-eval` (extra) | `0.4.11` |

`python_requires >= 3.11`. That binds the **GPU box's** environment, not the pure-Python
`rdl` core, which runs the CPU gate on 3.10 — see ADR-0002. It is enforced per
environment as `min_python: "3.11"` in the GPU profiles, so a Vast image that ships 3.10
is refused by `env-check` rather than by `pip` twenty minutes in.

### Actual, recorded from the run

| package | version |
|---|---|
| `torch` | _pending first RTX 3090 run_ |
| `transformers` | _pending first RTX 3090 run_ |
| `datasets` | _pending first RTX 3090 run_ |
| `accelerate` | _pending first RTX 3090 run_ |
| `tokenizers` | _pending first RTX 3090 run_ |
| `numpy` | _pending first RTX 3090 run_ |

If "actual" ever diverges from "expected", stop and find out why before reading any
metric: that divergence is the single most likely cause of a failed reproduction.

---

## 6. Device constraints applied to every eval command

Upstream's `configs/model/Llama-3.2-1B-Instruct.yaml` pins `torch_dtype: bfloat16` and
`attn_implementation: flash_attention_2`. `build_eval_command` overrides both from the
`HardwareProfile` on **every** invocation, so neither default can reach a device that
cannot run it:

| device | emitted `torch_dtype` | emitted `attn_implementation` | why |
|---|---|---|---|
| Colab T4 (SM 7.5) | `float16` | `sdpa` | no bf16 datapath; FA2 needs SM80+ |
| RTX 3090 / H100, `flash_attn` installed | `bfloat16` | `flash_attention_2` | both halves available |
| RTX 3090 / H100, wheel **absent** | `bfloat16` | `sdpa` | SM80+ silicon is not an installed wheel — see ADR-0031 |

That last row is the one that bites: a fresh Ampere cloud image has no `nvcc`, so
`flash_attn` cannot be imported and upstream's default would raise inside
`from_pretrained` *after* the checkpoint had downloaded.
`scripts/01_bootstrap_openunlearning.sh` skips the `flash-attn==2.6.3` install below SM80
automatically, and above SM80 builds it only under `INSTALL_FLASH_ATTN=1`.

### Revision pinning

Every eval also emits `+model.model_args.revision=<sha>` when the checkpoint is pinned
(ADR-0030). The `+` is mandatory: `revision` is not a key in upstream's `model_args`, and
Hydra rejects a plain override for an absent key — the same failure mode as
`retain_split=`. `make-report` blocks if the revision the reproduction used differs from
the one the condition grid ran.
