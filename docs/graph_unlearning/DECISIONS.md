# Decisions — graph-unlearning-v1

Dated entries. `PROTOCOL_v1.md` is frozen; corrections belong here. Same discipline as
`docs/04_decisions.md` for the two-agent work.

---

### GU-0001 — 2026-08-12 — A separate experiment family, not an extension

The historical runner is frozen evidence for the v5 conditions and the two-agent Leak@k
work. It supports two agents, hardcodes C0–C3C, stops delegation after one hop and
couples generation to scoring. Extending it would either break the frozen arms or
produce a runner that is neither. `src/rdl/graph/`, `defenses/`, `runtime/`,
`graph_memory/`, `studies/graph_leak/` and the `rdl graph-*` commands are new; nothing in
`orchestrator/`, `cli/run_leak.py`, `configs/conditions/` or `results/` was modified.

### GU-0002 — 2026-08-12 — Enforcement acts on payloads; the topology never moves

Removing an edge on detection improves every leakage number partly by deleting
communication. The primary method therefore preserves every edge and every route, and
passes / sanitizes / quarantines / blocks the content crossing it. `edge_cut` implements
the removal variant as an ablation and sets `edge_preserved: false` on its decisions, so
any report built on it labels itself topology-changing.

### GU-0003 — 2026-08-12 — Both guarded arms share one detector object

Not two detectors with the same config: the same Python object, asserted by
`test_dragon_and_graphforget_share_one_detector_object`. The config additionally refuses
a `dragon_style` defence with `propagate_scope: true`.

### GU-0004 — 2026-08-12 — DRAGON's default guard action is `refuse`, not `guard_prompt`

A prompt-only reasoning guard on a checkpoint that ignores the instruction is a no-op,
and a baseline that is a no-op is not a baseline. `refuse` makes the node emit a fixed
safe refusal instead of generating: deterministic, model-independent, and the strongest
honest reading of "modify the inference context". `guard_prompt` remains available and is
the faithful template-reasoning mode for instruction-following checkpoints.
`implementation: sft_checkpoint` raises rather than substituting a template for weights
that were never released.

### GU-0005 — 2026-08-12 — The registry stores questions, never answers

`ConceptRegistry.from_questions` raises if handed an `answer` key. Scope prototypes are
question text, deterministic question paraphrases, and extracted aliases. A defence
holding the gold answers at runtime would preserve the information the system claims to
have forgotten. The controlled-challenge injector does use gold answers and declares
`uses_gold_answers: true` in the manifest and the report header.

### GU-0006 — 2026-08-12 — `accumulated_only` excludes the query

First implementation counted "a combination fired and no individual input did", which
was true on almost every trajectory: when the user asks about a forgotten concept, the
query is itself in scope and every combination containing it fires. That inflated the
split-clue statistic with cases a node-local guard catches trivially. The accumulator now
scores the query separately, and `accumulated_only` requires that neither any individual
input **nor the query alone** fired. Caught by the first CPU smoke, which reported
`accumulated_only_hits: 40` out of 40.

### GU-0007 — 2026-08-12 — `split_clues` delegates the join node's query

For the same reason as GU-0006. If the join node still held the original question, the
question alone would be in scope. The challenge therefore replaces the join node's and
its descendants' query with a delegated task ("combine the material you have been
given"), which is also the realistic pattern: an integrator receives sub-results, not the
user's words.

### GU-0008 — 2026-08-12 — A cross-concept arm gets the CONTROL item's injection

First implementation injected the target concept's gold answer into every arm including
MA-CONTROL, which made the control leak as much as the treatment and destroyed the
contrast — the same failure mode as C3C and C3S ceasing to differ in one variable. The
runner now looks the injection up by the control source item id for cross-concept arms.
Caught by the first controlled-challenge CPU run.

### GU-0009 — 2026-08-12 — `answer_match_rate`, not `retain_utility`, on a forget cohort

The first report emitted `retain_utility: 1.0` for MA-LEAK — which meant "it leaked on
every item". On a forget cohort an answer-match rate *is* the leakage rate. The metric is
renamed, `overblock_rate` became `guard_fire_rate`, and `utility_gate` refuses to produce
a verdict unless told the rows are retain questions. The Markdown report prints an
explicit warning when they are not.

### GU-0010 — 2026-08-12 — A profile may declare itself unreportable

The study's `primary_k` is 32; the CPU wiring profile can only afford `n_samples: 2`.
Rather than relaxing the "compare at the same k" rule, `GraphProfile.reportable: false`
exempts a wiring profile from the primary-k check and forces every report it produces to
be diagnostic. The GPU profiles keep `reportable: true` and must contain the primary k.

### GU-0011 — 2026-08-12 — The trace digest excludes `cached`

Whether a response came from the model or from a byte-identical cache entry is
provenance about how it was obtained, not about what happened. Including it made the
second run of an identical arm disagree with the first, which is exactly the property the
digest exists to check.

### GU-0012 — 2026-08-12 — `validation.json` is empty and raises on load

Every forget10 concept is in `exclusions.json`, because the earlier 50-item spread pilot
covered all 20 authors. There is therefore no untouched concept pool on this checkpoint,
and the loader says so instead of letting a discovery cohort run under a validation
label. Filling it needs a new preregistered unlearning checkpoint, another dataset, or a
new model/method combination with a frozen concept split.

### GU-0013 — 2026-08-12 — Staged writes are the default; `immediate` is an ablation

If a write made at depth 1 were retrievable at depth 2, agent-to-agent edge flow and
memory-mediated flow would be mixed into one number and neither would be attributable to
its own mechanism.

### GU-0015 — 2026-08-12 — The RTX target was the FULL checkpoint, not the unlearned one

`configs/graph/models/rule_npo_1b.yaml` shipped as
`repo_id: open-unlearning/tofu_Llama-3.2-1B-Instruct_full` with `revision: null` — a
model that was never unlearned, on a moving branch. A 3090 run under that config would
have measured leakage from a checkpoint that has forgotten nothing, and every number
would have been meaningless while looking entirely normal.

Fixed to the released Leak-k baseline the two-agent work already pins:
`OptimAI-Lab/TOFU-forget10_RULE-NPO @ afe117e4`, tokenizer pinned to the same repo and
commit (it ships its own `tokenizer.json`, so this also removes the moving-branch chat
template risk flagged in `src/rdl/provenance.py`).

Three checks now make the class of error impossible to repeat:
`GraphModelConfig` rejects a config named for an unlearning method that points at a repo
containing `_full` / `_base` / `retain`; a method-named config must opt into
`expect_unlearned`; and `assert_model_provenance` refuses to start any run whose model
or tokenizer revision is unpinned. `tests/contract/test_gpu_readiness.py` asserts the
graph target equals `configs/models/tofu_forget10_rule_npo.yaml`.

### GU-0016 — 2026-08-12 — Two protocols, because the natural comparison collapsed

**The most important correction in this branch.** The concept registry's scope
prototypes include the forget questions themselves, so a forget question has ~1.0
similarity to its own prototype. Both guarded arms therefore fired at the root and
refused before the model was called. Measured on the CPU stub, natural condition:

| protocol | MA-GRAPHFORGET generations | nodes abstained |
|---|---|---|
| `end_to_end_safety` | 0 | 40 / 40 |
| `graph_flow` | 40 | 0 |

Under the old single protocol the study could only have shown that a detector
recognises the question it was built from. Propagation, edge enforcement, write
protection and retrieval protection were never exercised at all.

Split into two protocols, run separately and never pooled:

* `end_to_end_safety` — the request gate is part of the defence. Answers *does the
  deployed system release forgotten information*.
* `graph_flow` — the request gate is held **constant across every arm** (no arm inspects
  the question) and detection covers peer messages, tool responses, memory reads, agent
  outputs, edges, writes, retrievals and the final output. Answers *can forgotten
  information generated or introduced after the initial boundary propagate* — the
  GraphForget contribution.

`graph_flow` is mandatory: a study config that omits it fails to load. `protocol` is on
every evidence row, in the manifest, in the resume fingerprint, and is a required filter
in the report.

### GU-0017 — 2026-08-12 — The DRAGON default is the prompt guard, not a refusal

`guard_action: refuse` makes the node emit a fixed refusal instead of generating, which
is a strictly stronger intervention than anything DRAGON performs. Shipping it as *the*
DRAGON baseline overstated the baseline. The primary arm is now
`guard_action: guard_prompt` — detect, then modify the inference context, which is what
the paper describes — and the deterministic refusal is available as
`dragon_style_refuse` / `multi_agent_dragon_refuse`, reported as a **strong upper bound
on node-local guarding** and never as a faithful reproduction. The two bracket the real
baseline.

### GU-0018 — 2026-08-12 — `graph-plan` ignored `--n-samples`, so it costed the wrong run

`graph-run` accepted `--n-samples`; `graph-plan` did not. Planning a 4-item smoke against
the RTX profile therefore printed the profile's full 32-draw cost — 2 688 graph
generations and 1 280 probes — while the run that followed did 168 and 80. A plan that
describes a different experiment from the run is worse than no plan, because it is
trusted. Both commands now call one `apply_sample_budget`, and a reduction below the
study's `primary_k` marks the profile unreportable rather than silently reporting at a
smaller k.

### GU-0019 — 2026-08-12 — The dataset revision was recorded but never downloaded

`graph-freeze-cohort --dataset-revision X` wrote `X` into the manifest while `load_tofu`
called `load_dataset` with no revision, so the tool could record one commit and download
another. `revision` now flows through `load_tofu` and `load_items` into `load_dataset`;
freezing against real data requires it; a frozen non-fixture cohort without one is
refused at load; and runs download the cohort's own recorded revision.

All four real cohorts (smoke, engineering, discovery, retain_utility) are now frozen
against `locuslab/TOFU @ 324592d84ae4f482ac7249b9285c2ecdb53e3a68` with per-item
question and answer hashes.

### GU-0020 — 2026-08-12 — Resume could silently mix two experiments

`GraphRunner.run` overwrote `RUN_MANIFEST.json` before checking anything, so a resume
under a different checkpoint, topology, sample budget, protocol or cohort replaced the
record of what had produced the existing shards and then skipped their trajectory keys.
The completed keys are identical whichever checkpoint produced them, so nothing
downstream could notice.

Compatibility is now verified **before** any write, over sixteen immutable fields
(`GraphRunner.IMMUTABLE_ON_RESUME`), existing shard hashes are checked against the
previous manifest, and the original manifest survives a refused resume. A git-SHA
difference warns rather than fails: the study, profile and resolved hashes are what
determine whether it is the same experiment.

### GU-0021 — 2026-08-12 — `detector_device: cuda` named a code path that does not exist

The runtime profiles carried `detector_device` and `detector_batch_size`. The only
detector backbone is the torch-free hashing embedder, which runs on CPU by construction,
so those keys were decorative and would have been read as evidence that the detector ran
on the GPU. Both removed from the profiles; the detector's identity now lives in
`study.detector.backend: hashing64`, where the science is. Adding a semantic backbone
means adding a value there **and** the implementation behind it, together.

### GU-0022 — 2026-08-12 — vLLM was the RTX backend and nothing installed it

Neither `pyproject.toml` nor `requirements-gpu-ampere.txt` mentioned vLLM, so the RTX
profile named a backend that would fail at construction on a fresh box. Pinned
`vllm==0.6.3.post1` in the Ampere requirements and as a **separate** `[vllm]` extra —
separate because vLLM pins its own torch, and installing it into the reproduction
environment would replace the torch every existing number was produced under.

The backend also now receives `tokenizer_revision` (previously accepted by the CLI,
recorded in the manifest, and never passed to the engine — so the tokenizer came from
the branch head while the manifest claimed a pin), authenticates from `HF_TOKEN` in the
environment so no credential can reach engine kwargs or a manifest, and reports
`resolved_revisions()` — what the Hub actually resolved — alongside what was requested.

### GU-0023 — 2026-08-12 — A retain-utility cohort now exists

There was no cohort on which an answer-match rate is a utility rather than a leakage
rate, so `utility_gate` could only ever report itself inapplicable and a leakage
reduction had no cost attached. Added `retain_utility.json`: one retain90 question per
author for every fourth author, 45 authors spanning the split, frozen against the same
dataset commit. `phase: retain_utility` selects it.

### GU-0014 — 2026-08-12 — TOFU exclusions do not apply to the development fixture

`tests/fixtures/tofu_forget10_sample.json` reuses TOFU item ids for eight invented
questions. Applying the TOFU exclusion list to it blocked the CPU gate over a name
collision rather than over shared content, so `load_cohort` skips exclusion enforcement
when `dataset == "fixture"`.
