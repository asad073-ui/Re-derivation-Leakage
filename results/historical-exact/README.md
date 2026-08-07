# Historical-exact evaluation — 2026-08-07

open-unlearning at `fd825ea73b25ff6835f0675234cfe55b81117edb` (the commit that last
touched `docs/repro.md`, 2025-07-20), under that commit's own `requirements.txt`:
transformers 4.45.1, huggingface-hub 0.29.1, lm-eval 0.4.8, torch 2.4.1+cu121,
numpy 2.2.3, datasets 3.0.1, accelerate 0.34.2. FlashAttention 2.6.3, bfloat16,
batch_size 32, seed 0, RTX 3090 (cc 8.6).

**No `rdl.compat.fp32_logits` shim** — transformers 4.45.1 still upcasts logits, so the
bf16 crash that ADR-0037 documents does not arise here. That is an independent
confirmation of the root cause.

These are raw upstream outputs, not `rdl` reports: they were produced by invoking
`src/eval.py` directly, so they carry no `repro_report.json`. `*_agg_values.json` is the
`agg_value` of every metric in `TOFU_EVAL.json`; `forget_truth_ratio` appears only there
because this commit's `configs/eval/tofu.yaml` does not list it in the summary.

| checkpoint | model_utility | forget_truth_ratio | published |
|---|---|---|---|
| full | 0.6001046250207063 | 0.4753644629178423 | 0.60 / 0.48 |
| retain90 | 0.5901844512864824 | 0.6273686349110743 | 0.59 / 0.63 |
| npo_forget10 | 0.4323739724618672 | 0.6413989131591031 | **0.46 / 0.70** |

See ADR-0039.
