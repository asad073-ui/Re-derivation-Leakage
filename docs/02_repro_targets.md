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

Upstream's setup: **2× L40s, DeepSpeed ZeRO-3, lr 1e-5, effective batch 32, 10 epochs,
`paged_adamw_32bit`.** They warn explicitly that numbers shift when the distributed
setup changes, including on a single GPU.

Encoded in code at `src/rdl/eval/openunlearning_bridge.py::PUBLISHED_TARGETS` and
asserted by `tests/integration/test_ou_bridge_parse.py::test_published_targets_match_the_spec`.

---

## 2. What "reproduce" means here — and what it does not

### DO NOT reproduce the training

You cannot match 2× L40s ZeRO-3 bf16 on a single T4 in fp16, upstream says so
themselves, and a gradient-ascent-family objective in fp16 is numerically unsound.
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

**Check this on day 1** with `python -m rdl.cli discover-checkpoints`. The proposal's
zero-training cost model depends on this checkpoint existing.

If absent, the gate becomes the `full` model only, **plus** a diff of locally-recomputed
metrics against the published eval logs. Those logs are the HF dataset
`open-unlearning/eval`, which `python setup_data.py --eval` downloads; the relevant
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

> **TO BE FILLED ON FIRST COLAB RUN — DO NOT LEAVE BLANK.**
>
> Version drift is the single most likely cause of a failed reproduction. Fill this in
> before running the gate, and pin `requirements-gpu-t4.txt` to whatever lands here.
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

| package | version |
|---|---|
| `torch` | _pending_ |
| `transformers` | _pending_ |
| `datasets` | _pending_ |
| `accelerate` | _pending_ |
| `tokenizers` | _pending_ |
| `numpy` | _pending_ |

---

## 6. T4 constraints applied to every eval command

`build_eval_command` derives these from the `HardwareProfile`, so a T4 session cannot
accidentally inherit a bf16 default out of an upstream config:

| override | reason |
|---|---|
| `model.model_args.torch_dtype=float16` | T4 is Turing (SM 7.5) — no bf16 datapath |
| `model.model_args.attn_implementation=sdpa` | FlashAttention-2 needs SM80+ |

And the `flash-attn==2.6.3` install from the upstream README is **skipped** on T4.
`scripts/01_bootstrap_openunlearning.sh` detects compute capability and skips it
automatically.
