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

### GU-0024 — 2026-08-12 — Evidence, batching and gates for the RTX 3090 run

Six findings from the pre-rental review of PR #23, fixed together because they all block
the same thing: a 20x8 engineering run.

**Later-episode probes were serialised.** `_row` called `_probe` per trajectory, so the
readback dispatched two prompts at a time — 8,000 two-prompt backend calls per protocol
at 50x32, against a backend chosen specifically for continuous batching. The whole wave's
probes are now planned first and dispatched as one batch (`_probe_wave` / `_ProbePlan`).
What is asked is unchanged; the retrieval decision is still taken per trajectory before
any generation.

**NLI scoring ran at batch size one.** `graph-score` called the scorer once per surface
per string per row. It is now two passes over the shards: `candidate_pairs` collects every
`(reference, candidate)` question, those are deduplicated, ROUGE-gated in bulk and pushed
through the classifier in batches of `--batch-size` (default 64), and the second pass
answers from the resulting table. `n_batch_misses` in `SCORING.json` is nonzero only if
`candidate_pairs` and `surface_flags` drift apart. The NLI cache is checkpointed every
`--checkpoint-every` verdicts instead of being reopened and flushed per verdict, and the
`RougeScorer` is built once per process rather than once per pair.

**vLLM progress bars were on.** `engine.generate(prompts, params)` draws a tqdm bar per
call; thousands of calls into a `tee`'d log is thousands of redraw blocks. Now
`use_tqdm=False`, with a one-shot fallback for engines that do not accept it.

**Graph output made the next run dirty.** `_SOURCE_PATHSPEC` excluded `results/**` but not
`runs/graph/**`. `.gitignore` un-ignores the small contracts under a run directory, so the
first in-repo run left them untracked and the second recorded `git_dirty: true` — the same
self-poisoning ADR-0059 fixed for `results/`. `runs/graph/**` is now excluded too.

**Partial shards were covered by nothing.** Shard hashes reached `RUN_MANIFEST.json` only
when a run finished, so resume read an interrupted run's shards back, trusted them and
skipped their keys; a shard edited between the kill and the resume was adopted silently.
Every commit is now recorded in a per-directory `SHARDS.json` ledger written **before** the
atomic rename, so the crash windows are unambiguous: a trailing declared-but-absent shard
is an interrupted commit and is pruned, and a hash mismatch, an undeclared shard, or an
earlier missing one refuses. The runner additionally refuses a resume on a mismatch
whether or not the previous run completed.

**`graph-finalize` exited 0 on a partial, unreported run.** `ok` was `generations.ok and
traces.ok`. It is now `hashes_ok and complete and report_present`, with the hash check
covering the ledger as well as the manifest, and `blocking` / `warnings` lists saying why.
`raw_evidence_uri` warns rather than blocks: an unarchived diagnostic run is legitimate,
a partial or unreported one is not.

### GU-0025 — 2026-08-12 — `peak_vram_bytes` was not a claim about the card

`ResourceMonitor` sampled `torch.cuda.max_memory_allocated()`, which reports live tensor
bytes in *this process's torch caching allocator*. Under vLLM that excludes the KV cache
reservation, the CUDA graph pools and the activation workspaces, and under tensor
parallelism it excludes the worker processes entirely — so a 24 GB readiness decision
taken from it could read "4 GB peak" on a card that was in fact 20 GB full.

Device-level memory is now sampled from NVML (`nvidia-ml-py`, already a vLLM dependency),
which also attributes bytes to this PID and its children, falling back to
`torch.cuda.mem_get_info` and then to nothing. `CUDA_VISIBLE_DEVICES` is applied so the
sampled card is the one the run is pinned to. The torch-allocator figure is still recorded
under `peak_torch_allocator_bytes`, which says what it is; `PERFORMANCE.json` gains
`peak_device_vram_bytes`, `device_vram_utilization` and `vram_source`.

### GU-0026 — 2026-08-12 — "DRAGON cannot see split clues" was not true

The claim, in `evidence_accumulator`, `graphforget`, `BASELINES.md` and the capability
table, was that a node-local guard structurally cannot detect clues split across parents.
Our own DRAGON-style baseline runs at `apply_at: every_agent_input` and scores the node's
**complete** incoming context — query, every parent message, retrieved memory — as one
string. Clues that arrive together at one node are in that string. Node-locality says
where the guard runs, not how much evidence one call receives, and a join node receives
all of it.

Worse, `accumulated_only` was defined as "a combination fired and no individual input and
not the query", which does not exclude the whole-context view. The last combination the
accumulator scores **is** the whole-context view, so the flag counted cases the baseline
catches, and the headline split-clue number was unfalsifiable by construction.

Three changes:

1. `accumulated_only` now additionally requires that the whole-context view did **not**
   fire. `subset_only` is the honest superset — a strict subset fired and the whole-context
   view did not, i.e. scoring granularity, which is a real effect (a long query dilutes the
   embedding of the concatenation) but not a structural one. `node_local_fired` and
   `node_local_forget_ids` are carried on every result so the comparison can be checked
   rather than assumed.
2. `dragon_style_subsets` / `multi_agent_dragon_subsets`: the matched ablation. The
   node-local guard with GraphForget's exact subset battery, still inheriting nothing and
   still guarding no edge, write or retrieval. It holds scoring granularity constant so
   the remaining contrast is Forget-ID propagation and multi-surface enforcement — and it
   makes that claim falsifiable, which the previous comparison was not. Not the primary
   baseline: DRAGON as published scores one context.
3. The paper claim is narrowed to: *GraphForget adds persistent Forget-ID propagation and
   enforcement across edges, memory writes, retrieval and final release, while the
   DRAGON-style baseline guards model-input boundaries.* A genuine cross-call accumulation
   claim needs a topology where no single call receives the complete evidence; the current
   diamond rejoins every split clue inside one node's input, so it does not support one.


### GU-0027 — 2026-08-12 — The forget policy is not the evaluation cohort

`GraphRunner` built its concept registry and its deleted baseline memory from
`self.items` — the questions being asked. Correct for a forget cohort, catastrophic for
the retain one that `phase: retain_utility` exists to run: it would have registered the
45 retained authors as forgotten concepts, so the guard would have fired on exactly the
behaviour the run was measuring. Retain utility would have measured over-blocking of
concepts the run itself declared forbidden, and the detector's false-positive rate would
have been computed against its own positives. Both numbers would have looked plausible
and neither would have meant anything.

A run now carries two cohorts:

    evaluation      the questions. forget, controlled, validation or retain.
    forget policy   the concepts the system must withhold. ALWAYS a frozen forget
                    cohort; supplies the concept registry and the deleted baseline
                    memory.

`assert_forget_policy_cohort` refuses a retain cohort in the policy role outright, and
`Cohort.is_retain` is read off the split name and dataset config rather than a flag
somebody has to remember to set. A retain phase with no `forget_policy_phase` is a
parameter error, not a fallback — the fallback *was* the bug.
`assert_policy_excludes_evaluation_concepts` additionally refuses a retain evaluation
cohort that shares an author with the policy, because such an author would be
simultaneously must-withhold and must-answer.

`--limit` narrows the evaluation cohort only. A question budget is not a statement about
what the deployment forgot, and a registry that shrank with it would make the guard's
scope depend on how much GPU time was bought.

Both fingerprints are in every manifest (`cohort_fingerprint`,
`forget_policy_fingerprint`), the policy travels as its own artefact
(`FORGET_POLICY_COHORT.json`), and both are immutable on resume.
`configs/graph/launch_rtx3090_engineering.yaml` and `..._retain.yaml` are the two entry
points; the retain one names `forget_policy_phase: engineering`, so it evaluates retain90
questions under the identical frozen policy as the engineering run.

Two further items in the same change:

* **The sixth arm.** `multi_agent_dragon_subsets` moved from optional to a study arm. A
  fairness ablation nobody runs falsifies nothing. It is the matched-subset ablation of
  the node-local baseline and is never labelled as the published DRAGON implementation.
* **Detector calibration is now evidence.** `detector.status: calibrated` used to be a
  word in a yaml file that the report gate trusted. `rdl graph-calibrate` selects the
  threshold on two frozen cohorts and writes `DETECTOR_CALIBRATION.json` — threshold,
  recall, FNR, FPR, both cohort fingerprints, and its own content hash. The config
  refuses `calibrated` without an artefact; the loader refuses one that is missing,
  edited or failed; the runner refuses one whose FPR exceeds
  `evaluation.max_detector_fpr`. Measured on the real data: threshold 0.65, recall 0.950,
  FNR 0.050, FPR 0.056.

  The negatives are the 45 retain90 authors congruent to 1 mod 4, which no evaluation
  cohort uses, so the FPR is held out at the author level from the retain questions the
  utility gate scores. The positives are held-out *questions* about the forget-policy
  authors and cannot be held-out authors — the detector's prototypes are those authors,
  so a positive about a different one would not be a positive. The artefact records that
  distinction in `positives_held_out_at` rather than implying a stronger guarantee.

  A threshold is calibrated FOR A REGISTRY, so a run whose forget policy the artefact
  does not cover downgrades itself to `diagnostic` and records why. The CPU stub cohort
  is the everyday case, and it reports `diagnostic` — which is the truth about it.

Finally, the report no longer builds its own `OfflineSemanticScorer`. Utility, refusal
and readback are read out of the run's score rows, so every semantic number in one report
comes from one scorer; and retain utility within 3 pp and detector FPR within 10% are now
BLOCKING report gates rather than printed numbers. A leakage reduction with no measured
cost is not a result — a defence that refuses everything wins on leakage alone.

### GU-0028 — 2026-08-12 — Reserved VRAM is not GPU activity, and one counter is not two

`manifest["actual_graph_generations"]` was `scheduler.dispatched`: the total dispatch
count, readback probes included, published under a name that says "graph". It could not
be compared with `plan.planned_graph_generations` and silently was not. The 2x1 preflight
that planned 42 graph generations and 20 probes reported "27 model calls dispatched",
which is neither number.

`BatchScheduler.run` now takes a `purpose` — `graph` or `probe` — and keeps
`{requested, dispatched, cache_hits}` for each. `requested == dispatched + cache_hits`
holds per purpose, which is what makes the pair checkable from outside the process. The
old name is deleted rather than renamed; `actual_generations` is the per-purpose
breakdown.

`rdl graph-finalize` gained an activity gate. Peak VRAM cannot answer "did it generate":
vLLM reserves the KV cache to `gpu_memory_utilization` of the card before the first token
exists, so a run that dispatched nothing and returned empty strings is indistinguishable
from a working one on that metric. Blocking, on a real backend: non-zero completion
tokens, non-zero generation batches, at least one non-empty trajectory, and non-zero
requests for both purposes with the counters balancing. The refusal and collaboration
rates are computed and reported alongside — they are what distinguish "the defence
contained the leak" from "the defence stopped the system working" — as warnings by
default, because a run whose defence over-refused is still evidence and still has to be
archivable before the instance is destroyed. `--enforce-science-gates` makes them
blocking.

The preflight is **2 items x 1 sample**, permanently. One item cannot produce a
cross-concept control, so `--limit 1` — which the runbook advised for months — cannot run
this study at all; it failed deep inside the runner with a message about C3S, and the
operator discovered that on a rented GPU. `assert_control_arm_has_enough_items` now
refuses it as a parameter error with the right number in it, and a test asserts that no
document says `--limit 1` again.

### GU-0029 — 2026-08-12 — NaN is not JSON

Three committed graph reports carried literal `NaN`, from `relative_reduction` returning
a float NaN for an undefined relative reduction against a baseline that never leaked.
The intent was right and the representation was not: RFC 8259 has no such token, so `jq`,
`python -m json.tool`, Go, Rust and every browser reject the file — including the runbook
step whose entire job is to validate the evidence before it is archived.

`relative_reduction` returns `None`, which serialises as `null`, the JSON spelling of
"undefined". `finite_or_none` is applied to every statistic on the way out, so a NaN
produced anywhere upstream becomes an explicit null rather than a token that makes the
whole report unparseable. `atomic_json` writes with `allow_nan=False` and raises rather
than emitting one. Markdown prints `undefined`, with a note saying what that means.
`ComparisonResult.supported` treats an undefined interval as *not* support, because
"supported by an absence of evidence" is the one reading that must not be possible.

The three committed reports are repaired in place. Their raw generations live in the
release tarball rather than the repository, so `graph-report` cannot be re-run against
them here; what changed is the representation only — every `NaN` in those files was an
undefined relative reduction and is now `null` — and each file records a `nan_repair`
block saying so, so nobody has to diff it against the release to work out why it differs.

### GU-0030 — 2026-08-13 — The primary metric was a subtype, and under memory re-entry it was inverted

The 50x32 RTX 3090 discovery study (PR #29, release
`graph-discovery-50x32-20260813T043200Z`) ran, finished, verified and reported. Its
generation pipeline is sound and its phenomenon evidence is real. Its *reporting* was
wrong in five ways that no amount of further GPU time would fix, and this entry records
the corrections. **No generation was re-run and nothing was re-scored**: every change here
is a reanalysis of the archived evidence, and the surface booleans are the same booleans.

**1. `certified_persistent_leak` was a subtype presented as the total.** It requires a
leaking stored node with no `parent_ids` and `is_parametric: true` — deliberately
excluding a node that merely copies a retrieved parent, because that is a retrieval
failure rather than a re-derivation. As a mechanism probe that is right. As the study's
single primary metric it is wrong, and under `memory_reentry` it is *inverted*: the
unguarded arms leak by retrieving and re-committing a parent, so their nodes carry
`parent_ids` and score **zero**, while GraphForget blocks the parent and is then
re-derived from parameters, scoring **0.16**. The headline therefore said GraphForget was
the only arm to leak, on the one challenge where it cut total persistent leakage from
0.58 to 0.18, causal readback from 0.36 to 0.04 and edge leakage from 0.54 to 0.14 — at
9.6% refusal and 94.7% collaboration, which is containment rather than refusal.

The surfaces now carry the names of what they measure:

| was | is |
|---|---|
| `store_leak` | `policy_violating_persistent_leak` — the TOTAL: the store holds forbidden content |
| `certified_persistent_leak` | `rootless_parametric_rederivation_leak` — the subtype |
| `causal_readback_leak` | `causal_memory_readback_leak` |

Both spellings are written to every score row and `surface_value` resolves either, so the
archived discovery evidence re-reports without regeneration. `PROTOCOL_v1.md` §8
criteria 1–2 name "certified persistent Leak@32"; they are frozen and stand as written,
and are to be read as naming `rootless_parametric_rederivation_leak` — which under
`memory_reentry` cannot be met and must not be claimed. This entry is that correction.

**2. Which surface is primary is a property of the CHALLENGE.** `metric_applicability`
gives every surface a role (`primary` / `secondary` / `diagnostic` / `invalid`) and a
stated reason, per challenge, and the report writes the whole table into its JSON.
`invalid` means the metric's definition interacts with that challenge's mechanism in a
way that can reverse the ordering of the arms, and nothing may rank arms on it. Under the
injected challenges (`split_clues`, `tool_reentry`) the persistence surfaces record the
harness's own gold-derived text coming back, so the claim moves to `edge_leak` and
`sink_leak`.

**3. The phenomenon had no intervals.** The claim that composition reconstructs what
neither isolated condition releases rested on three point estimates — single 0.08, control
0.04, MA-LEAK 0.16 — with no interval on any of the *differences*. `composition_report`
adds C1 (MA-LEAK vs single), C2 (MA-LEAK vs the cross-concept control) and C3
(DRAGON-style vs MA-LEAK) with the same paired concept-clustered bootstrap the defence
hypotheses use. These are **increase** claims, so support is `ci_low > 0`;
`ComparisonResult` carries a `direction` and reads the correct bound. C3 is reported and
never required: whether a node-local guard helps in a graph is empirical.

Alongside them, `eval/graph_concentration.py` reports what a rate of 0.16 over 50 items
actually rests on — affected items, affected concepts, the top concept's share, the
Herfindahl index, and a leave-one-concept-out refit with `sign_stable`. A discovery number
that flips sign when one of twenty authors is dropped is not wrong, it is underpowered,
and that belongs next to the number rather than in a reader's head.

**4. `reportable: true` was standing beside `utility_gate.applicable: false`.** An
inapplicable cost gate was folded into the verdict as non-blocking, so "we did not measure
the cost" and "the cost was acceptable" produced the same `true`. The cost gates are now
tri-state (`pass` / `fail` / `not_applicable`) and the single flag is split in two:

    semantic_report_valid   this report's numbers can be read as what they say
    publication_ready       the report additionally CARRIES A CLAIM — the cost gates were
                            applicable and passed, and the defence did not buy its leakage
                            number by refusing to work (refusal <= 0.20, collaboration
                            >= 0.80, under graph_flow only)

`reportable` survives as the pre-GU-0030 name for the first of those, so older readers do
not silently flip meaning. A single forget-cohort run can never satisfy `publication_ready`
alone, because retain utility lives in a different run — which is what `rdl graph-bundle`
exists to resolve. It links each study's runs into one `STUDY_BUNDLE.json` carrying
leakage, retain utility, detector FPR, refusal and collaboration, each figure verbatim
from the run that measured it with that run's id attached. Nothing is pooled.

Under this gate the discovery study's `split_clues` (96% refusal) and `tool_reentry`
(100% refusal) reductions are correctly reported as refusal-confounded, and
`natural` under `end_to_end_safety` (80% refusal, 20% collaboration) as a statement about
request filtering rather than about the graph.

**5. "DRAGON as published" was never true of any arm we ran.** `dragon_style.yaml`
declares `implementation: template`; the released detector/guard checkpoints are
unavailable; `implementation: sft_checkpoint` raises. The label everywhere is now
**DRAGON-style node-local template baseline**, `report_graph.DRAGON_LABEL` is the single
string every report uses, and the phrase "as described in the paper" is reserved for the
*algorithm* our template imitates. The observed 0.50 against MA-LEAK's 0.44 supports no
claim that published DRAGON is worse — and, before C3, was not even a paired comparison
between the two arms actually run.

**And the thing all five were hiding.** `rdl graph-detector-recall` measures detector
recall on text the system *generated*, labelled by the run's own pinned-scorer verdicts
out of the scoring cache, rather than on forget questions. `DETECTOR_CALIBRATION.json`
reports recall 0.95 — on held-out *questions*, which bounds a request guard. A graph
defence has to catch a paraphrase three hops downstream. Where those two numbers diverge,
a propagation defence and a node-local one score identically no matter how good the
propagation is, because there is nothing for either to propagate; the natural-flow arms
being identical across MA-LEAK, DRAGON-style, matched-subset and GraphForget on every
surface is the signature of exactly that. The fix is detection, not the graph, and this
command is what makes that diagnosable instead of inferred.

#### What it measured, on the archived 50x32 evidence

| Probe | Detector fire rate |
|---|---|
| Forget-policy questions | **1.000** (20/20) |
| Calibration positives (held-out questions) | **0.950** — reproduces the recorded artefact exactly |
| Gold answers | **1.000** (20/20) |
| **Actually-leaking generated text, `multi_agent_leak`, natural flow** | **0.000** (0 / 380) |

False alarms on generated *clean* text run at 0.032, so the detector is firing — just
never on what leaked. Recall is 0.000 on all four generated surfaces separately
(120 agent messages, 108 edge payloads, 120 stored nodes, 32 final answers).

The cause is concrete and small. `ConceptRegistry.from_questions` gives author-0000
**three** scope prototypes and **one** alias, `"Hsiao Yun-Hwa's"` — the possessive form
lifted out of the question, with no bare full name, no surname, no given name and no
partial. Downstream agents abbreviate. Every missed leak in the sample is of the form

    "Yun's father's profession as a civil engineer ..."     score 0.500, threshold 0.65

against prototypes that all read "Hsiao Yun-Hwa's father". The gold answer scores 0.790
because it names the author in full; the paraphrase three hops later does not.

So the reported study is best read as: **the leakage phenomenon is established, and the
defence was never actually tested**, because the detector it shares with the baseline
saw none of the natural leakage. That is a detection-coverage defect, not a negative
result about provenance propagation.

#### And it separates GraphForget's two mechanisms, which is the useful part

Recall on generated leakage differs by challenge, and it tracks the defence's success
exactly:

| Challenge | Detector recall on generated leaks | GraphForget vs MA-LEAK on the challenge's primary surface |
|---|---|---|
| `natural` | **0.000** (0 / 380) | no reduction — every arm identical |
| `memory_reentry` | **0.086** (113 / 1314) | **−0.40 total persistent, −0.32 causal readback**, both intervals excluding zero |

GraphForget contains memory re-entry while its semantic detector is still missing 91% of
the leaking text. It cannot be the detector doing that work. What is left is the part that
does not depend on recognising content: **Forget-ID taint inherited through provenance**,
which is carried by the seeded store's derivation edges and enforced on retrieval, writes
and edges regardless of whether anything semantic fires. Under `natural` there is no
seeded parent to inherit from, so taint has no purchase and the defence falls back on a
detector that fires on nothing.

That is the design conclusion, and it is measured rather than argued:

* the memory-reentry result **implicates provenance-based enforcement**, but boundary
  coverage and Forget-ID inheritance remain **confounded**: GraphForget guards five
  surfaces the node-local baseline does not guard at all, and it inherits Forget-IDs, and
  the archived study varies both at once. Causal attribution requires matched
  **semantic-only, taint-only and stateless multi-surface** ablations — see GU-0031, which
  supersedes the sentence this bullet replaced;
* **semantic detection is the binding constraint** and is what the next change must fix;
* the two must be **ablated apart** — semantic-only, taint-only, taint+semantic — because
  the study so far reports their sum and attributes it to the wrong half.

> **Correction, 2026-08-13 (GU-0031).** This bullet originally read "**taint propagation
> works** and is the mechanism worth keeping". That over-claimed. The evidence shows a
> reduction under `memory_reentry` that a 0.086-recall detector cannot explain on its own;
> it does not show which of the non-semantic mechanisms produced it, because the archived
> study has no arm that guards multiple surfaces *without* inheriting Forget-IDs. The
> corrected wording is above and the ablation that would settle it is GU-0031.

It is also, usefully, **fixable and measurable entirely on CPU**. `DETECTOR_RECALL.json`
turns the 380 archived missed strings into a fixed regression target: alias expansion and
recalibration can be developed against them with no GPU at all, and the natural-flow
result can be predicted before any instance is rented. The next GPU run should be
purchased only once recall on that archived set is high and the held-out FPR still clears
0.10 — those two together are what make a graph-versus-node-local comparison meaningful
at all.

---

### GU-0031 — 2026-08-13 — The detector corpus is frozen before the detector is fitted

GU-0030 established that the defence was never tested: detector recall on generated
leakage was 0.000 under `natural` and 0.086 under `memory_reentry`, against 0.950 on
forget questions. The fix is a better detector. The hazard the fix creates is that the
moment text is used to fit a detector, it stops being evidence, and "held out" becomes a
claim nobody can check after the fact. So the corpus and its split are frozen **first**,
in their own commit, before any of detector v2 exists.

`rdl graph-detector-corpus` builds both artefacts deterministically from committed
evidence and rescores nothing:

    data/cohorts/graph_unlearning_v1/detector_v2/DETECTOR_GENERATED_CORPUS.json
    data/cohorts/graph_unlearning_v1/detector_v2/DETECTOR_ENGINEERING_SPLIT.json

**What is eligible.** `multi_agent_leak` rows only, on the four generated surfaces, from
`natural` and `memory_reentry`, labelled leaking by the run's own pinned NLI scorer read
out of its cache. Guarded arms are excluded because enforcement thins their evidence with
the mechanism under study; `split_clues` and `tool_reentry` are excluded because both are
refusal-confounded (96% and 100%) and record the harness's own gold-derived text;
harness-authored injection text and any text carrying a gold answer verbatim are excluded
because a detector fitted on gold answers has memorised what the system claims to have
forgotten. `memory_reentry` examples are eligible — the MODEL produced them — and every
one records `injected_memory_origin: true` so the origin never has to be inferred.

**What that yields.** 502 distinct `(concept, normalized_text)` examples over 18 of the 20
forget-policy concepts; 13 concepts have at least five. Pooling the two challenges is what
makes the corpus usable: `natural` alone gives seven concepts with 82% of the mass on one
of them. It is concentrated even so — author-0003 holds 46%, Herfindahl 0.245 — which is
why the gates are **macro over concepts**, not micro over examples.

**How it splits.** By concept, never by text. An alias channel fitted on "Yun's father"
would trivially generalise to another sentence about the same author, and a text-level
split would report that as generalisation. The 13 gateable concepts are stratified by
example count (20+, 10-19, 5-9), apportioned by largest remainder, and cut 8 development /
5 held-out; the largest concept in each stratum anchors development. The other five
concepts are `audit_only`: reported, never a pass/fail gate, because a concept with three
examples cannot fail a recall gate for a reason anyone would believe. Stratification uses
example counts **only** — never a detector score, which would select the held-out concepts
using the thing being tested.

**What it is not.** Not publication validation. These twenty authors have been inspected
repeatedly. Both artefacts carry `split_role: engineering_holdout` and
`publication_validation: false`, and the held-out half bounds engineering generalisation
and nothing else. A publication claim needs concepts frozen before anyone looked, which is
the H100 critical path and is not resolved by this entry.

**What makes the freeze checkable.** Each artefact carries a `content_sha256` over its own
canonical content, and the split additionally names the corpus hash it was cut from. The
tests assert that changing a source row, a normalized text, a concept assignment or a
split membership moves the hash. `--verify` re-derives both from the same runs and
compares, so a reviewer checks the freeze rather than trusting it. Every source run is
recorded with its shard-ledger hash, scorer version and forget-policy fingerprint —
the raw shards stay local, and the ledger hash is what lets a box that does not hold them
still say which evidence this was built from. Source runs that disagree on the forget
policy are refused outright.

The mechanism decomposition GU-0030's corrected bullet calls for lives in a separate
study, `graphforget_mechanism_v2`, and not as new arms bolted onto `graph_unlearning_v1`:
that study and its 50×32 evidence are frozen, and adding arms to it would present a
different experiment as the same one (GU-0001's rule, applied within the family).

---

### GU-0032 — 2026-08-13 — Detector v2, and the measurement that says do not rent the GPU

Detector v2 was built against the frozen corpus (GU-0031), measured against the CPU gates,
and **does not clear them**. That is the finding, and it is recorded here rather than
worked around.

#### What v2 changed

The lexical channel, and only the lexical channel. `normalise_scope_text` folds unicode
punctuation, strips ASCII and unicode possessives and splits hyphens; it is applied to
alias tokens and to the text being scored. Aliases are now built in two phases over the
whole registry: canonical full names, hyphen-preserving and hyphen-split variants, and
partial names that survive two ambiguity checks — a token claimed by more than one
forgotten concept is dropped from **all** of them, and a token appearing in more than one
concept's question text is not treated as a name at all. Versions bumped to
`concept-registry-v2` and `semantic-scope-v2`, both of which reach the registry
fingerprint and the run manifest.

The embedding channel is untouched. Moving both at once would make any resulting threshold
un-attributable to either.

#### What it measures, on the frozen 502-example corpus

| | v1 | v2 |
|---|---|---|
| Held-out micro recall (correct concept) | 0.000 | **0.299** |
| Held-out macro recall | 0.000 | **0.329** |
| retain90 FPR | 0.056 | **0.100** |
| Generated-clean FPR | 0.032 | **0.446** |
| Correct-concept precision | — | **0.044** |

Against gates of 0.80 micro, 0.75 macro, ≤0.10 on both FPRs and ≥0.80 precision. Five of
nine gates fail. `rdl graph-detector-gates` exits non-zero and writes the whole table to
`DETECTOR_V2_GATES.json`.

The first v2 gate run also caught a regression the gates exist for: retain90 FPR **0.20**,
four times v1's, because "Award", "Write", "New" and "Inspired" are title-cased in TOFU
questions, reached the span extractor, and became one-token aliases that fire on retain
questions about entirely different authors. The document-frequency check above is the fix,
and it brought the rate back to the ceiling.

#### Why no amount of further alias work will clear the recall gate

`DETECTOR_V2_GATES.json` reports a **lexical ceiling**: the share of leaking examples that
contain *any* token of their own concept's aliases. It is **0.215 micro / 0.461 macro**.
That is the highest recall any alias channel can reach on this corpus, and the gate is
0.80. The reason is visible in the corpus itself — the unlearned checkpoint leaks the
CONTENT while garbling the NAME:

    "M Majlar's characters are typically grotesque ..."      (Rajeev Majumdar)
    "Yon (1900-1966), a renowned leader in ..."              (Hsiao Yun-Hwa)

80% of the leaking strings never name the author correctly. A name-matching detector
cannot see them, and this is a property of the leakage, not of the implementation.

#### And a pinned sentence encoder does not rescue it either

Probed, not adopted: `sentence-transformers/all-MiniLM-L6-v2` @ `1110a243`, mean-pooled and
L2-normalised, cosine against the same question prototypes, swept over the same grid
(`scripts/probe_semantic_encoder.py`, verdict archived as `ENCODER_PROBE.json`). At the FPR
ceiling of 0.10 the best operating point is 0.575, where development macro recall is
**0.000** and held-out macro recall is **0.006**. It is not a tuning failure: the
negatives are retain90 QUESTIONS and the prototypes are forget QUESTIONS, so in an encoder
that represents text type, every TOFU author question is close to every other one. Recall
only rises where the FPR is already 0.97.

#### The decision this forces

The gate combination as specified may not be satisfiable by any detector that (a) holds no
gold answers, (b) is scored for false positives on retain QUESTIONS, and (c) must catch
name-free content paraphrases. Three of those are load-bearing commitments and one is a
choice about the negative population. **This needs a design decision, not a threshold.**
Options, in the order they should be considered:

1. **Change the negative population** to generated retain-cohort TEXT rather than retain
   questions, so both sides of the FPR are the same kind of object. `generated_clean_fpr`
   already exists and is measured; note that at 0.446 it is also failing, and that a
   "clean" generated text about a forgotten author is arguably IN SCOPE — the pinned NLI
   scorer judges leakage of a specific fact, not whether the text is about the concept.
   Whether scope detection and leak detection are the same predicate is the question the
   gate is actually asking, and nobody has answered it.
2. **Give the detector the concept's own retrieval corpus** — not gold answers, but the
   question set plus its paraphrases, encoded by a stronger pinned model, with the
   threshold selected per concept rather than globally.
3. **Accept that natural-flow detection is out of reach on this checkpoint** and restrict
   the mechanism claim to `memory_reentry`, where provenance rather than detection is the
   carrier. This is the only option that permits GPU time now, and it narrows the paper.

Until one is chosen, the honest reading of GU-0030 stands unchanged: the leakage phenomenon
is established and **the defence has still not been tested under natural flow**.

#### What is ready regardless

* `configs/graph/studies/graphforget_mechanism_v2.yaml` — the six-arm decomposition, plus
  the single-agent and cross-concept reference arms the composition contrasts need. Each
  mechanism contrast varies exactly one thing, and a test asserts that.
* Causal attribution counters on every protected surface: `(surface, attribution, action)`
  with attribution in {semantic_only, inherited_only, semantic_and_inherited, neither}.
  `inherited_only_enforcements` is the count no node-local guard could have produced, and
  it is what would let the next run assign a reduction to propagation.
* `propagates_scope` is derived from `propagate_forget_ids` instead of being a class
  constant, and the `Defense` protocol now demands a read-only property so a constant
  cannot satisfy it. Every graphforget ablation previously reported
  `propagates_scope: true`, including the two arms whose purpose is not to propagate. The
  defence's name in the manifest is likewise the ARM's defence config name, not the class's.
* `rdl graph-bundle` reports `composition_vs_single_supported`,
  `composition_vs_control_supported`, `defence_supported`, `operationally_eligible` and
  `publication_ready` separately, and checks cross-run agreement on model and tokenizer
  commits, study-design hash, scorer version, primary k, detector version and registry
  fingerprint, plus exactly one run per (challenge, protocol). `ready_challenges` was the
  misleading name — a challenge could be "ready" while its defence hypothesis was
  unsupported — and survives as an alias of `operationally_eligible_challenges`.

#### One unrelated crash, found by running the mechanism smoke

`rdl.paths` read git through `subprocess.run(..., text=True)`, which decodes with the
LOCALE codec. On a cp1252 box a diff containing an em dash — or the `∪` in the propagation
rule, or a `×` in this file — raises `UnicodeDecodeError` inside subprocess's reader
thread, where it is swallowed, and `stdout` comes back `None`. The run then dies with
`'NoneType' object has no attribute 'strip'` **at manifest time, after generation**. Every
git read now decodes UTF-8 explicitly with `errors="replace"`. It would have killed a GPU
run after the expensive part, and the only reason it had not yet is that no committed diff
had contained a non-cp1252 byte at run time.

### GU-0033 — 2026-08-13 — Provenance is two mechanisms, and the ablations had one of them

PR #33 was merged into the PR #32 branch rather than into `main`, so none of Detector v2,
the gate artefact, the mechanism arms or the attribution counters were ever on `main`.
Landing it on top of current `main` is the occasion for this entry, but re-opening it
unchanged would have bought GPU time for a decomposition that could not decompose
anything. Six defects, in descending order of how much they would have cost.

#### 1. The "no inheritance" ablations were enforcing inherited provenance

`GraphForgetDefense.on_retrieval` acted on a stored Forget-ID **unconditionally**. The
`propagate_forget_ids: false` flag gated the edge, write and final surfaces and did not
gate retrieval at all. So `graphforget_semantic_only` and `stateless_multi_surface` — the
two arms whose entire purpose is to lack provenance — withheld tagged memory at the one
surface where the memory-re-entry challenge does all of its work. `full − semantic_only`
was inheritance minus inheritance.

The same call then recorded the decision with `inherited=()` because the arm does not
propagate, so a withholding **caused by a stored Forget-ID** was booked as `neither`:
the ledger's cell for "nothing had the scope". The headline the mechanism study exists to
produce was being written into the counter that denies it happened.

The fix separates the two capabilities that `propagate_forget_ids` was conflating:

    consume_forget_ids     act on a scope the object in front of the guard ALREADY carries
    propagate_forget_ids   attach scopes to what this decision PRODUCES

A non-consuming arm now sees no candidate as tagged, which also sends every candidate to
the semantic rescan — otherwise the ablation would be weaker than its treatment in a
second dimension, which is the defect this decomposition exists to avoid.

#### 2. Forwarding happened in three places, and the defence controlled one

Even with the defence gated, scopes were still forwarded on a non-forwarding arm's behalf
by `derive_envelope` (parent-scope union), by the executor (`plan.forget_ids`), and by
`StagedMemory._commit_one` (store-parent closure). All three now take the arm's own flag.
`derive_envelope(inherit_scopes=False)` keeps `parent_ids` — provenance is evidence and
survives; only the scope stops crossing the edge. `WriteVerdict` and `NodeInputVerdict`
gained `propagated_forget_ids`, distinct from the `forget_ids` the decision was made on,
defaulting to `None` = "the same", so every other defence is unchanged.

#### 3. `tag_local_only`, and what `taint_only − unguarded` was actually measuring

That contrast confounds enforcing tags that already exist with forwarding them to
descendants, and a system that only ever had to block the tagged source needs no
propagation at all. `graphforget_tag_local_only` is `taint_only` with forwarding removed
and nothing else changed. `M5 = taint_only − tag_local_only` is now the only contrast that
isolates forward propagation, and it is the only one that can support the propagation
claim. `M6` varies consumption and forwarding together and cannot stand in for it.

#### 4. No scope ever entered the system, so both taint arms were vacuous

The blocking one, and not in the review. `memory_reentry` seeds its note with
`store.add(...)` — **untagged**. Nothing else tags anything unless the semantic detector
fires, and both provenance arms have the detector switched off. `taint_only` and
`tag_local_only` would have inherited nothing, enforced nothing, and scored identically to
the unguarded arm: `M4` and `M5` exactly zero, for a reason having nothing to do with
propagation, on the run bought to measure propagation.

`memory.seed_policy_tags_on_reentry` plants the note carrying its concept's Forget-ID —
identically for every arm, so it advantages none of them, and never for a concept outside
the registry, so a retain run cannot tag a question the system must answer. It is OFF in
`graph_unlearning_v1`, which stays byte-identical to its freeze, and ON in the mechanism
study, whose design hash and challenge fingerprint both move to say so.

#### 5. The GPU would have run at an operating point nobody measured

The study declared `threshold: 0.65` — the v1 number — while `DETECTOR_V2_GATES.json`
selected **0.90**, and every measurement anyone has (recall 29.9%, precision 4.43%,
generated-clean FPR 44.55%, lexical ceiling 21.5%) was made at 0.90. The study now runs at
0.90 and pins `gate_artifact` at the file that records `all_gates_passed: false`. A test
asserts the artefact is a FAILING one; if that ever flips, the study gets re-read
deliberately rather than inheriting a `calibrated` claim it never earned.

#### 6. `--limit 20` is not the engineering cohort, and 20×8 is not k=32

`launch_mechanism_v2.yaml` selects `local_cpu` and inherits `phase: smoke`. A limit takes
the first N items of the *smoke* manifest — a different frozen cohort with a different
fingerprint. Which cohort a run uses is chosen by `phase`, so there are now dedicated
`launch_mechanism_v2_rtx_engineering.yaml` and `..._retain.yaml`, the retain file pinned to
the same frozen forget policy as its engineering partner.

`graph-run` refused nothing when the sample budget could not reach the declared
`primary_k`; it printed `note:` on stderr and generated anyway. That is the 20×8 failure,
twice. It is now a hard refusal with `--allow-k-substitution` as the explicit opt-out,
which is how the 2×1 preflight and the CPU smoke declare themselves.

#### What is still not fixed, and is not fixable here

Detector v2 does not clear its recall gates and no amount of alias engineering will get it
there: the lexical ceiling (21.5% micro) is below the primary bound (80%). The model leaks
facts while corrupting or omitting the author's name, so **no natural-condition semantic
defence claim is available from this detector at any threshold**. The first RTX run is
`memory_reentry` + `graph_flow` only. `natural` and `end_to_end_safety` are deliberately
not in it.

### GU-0034 — 2026-08-13 — A single-variable contrast that measured nothing

GU-0033 split provenance into `consume` and `forward` and made `M5 = taint_only −
tag_local_only` the propagation contrast. Those two arms differ in exactly one flag, and a
test asserted it. The contrast was still incapable of measuring forwarding.

Both arms consume tags, so both quarantine the seeded note **at retrieval, before any
agent reads it**. The traces are unambiguous — for `taint_only` and `tag_local_only`
alike:

    reads=40  returned=0  withheld=40  edges=40  edges_carrying_a_tag=0  writes_tagged=0

No agent ever saw the tagged content, so no derivative existed, so there was nothing for a
forwarding arm to forward. The difference is zero by construction at 4 items or 50, and no
number of GPU samples could have changed it. **Single-variable is necessary and not
sufficient: the pathway the variable acts on has to be live, and liveness is a property of
a run, not of a config.**

#### The missing pathway was real, and it was a hole in the defence

`GuardedRetrieval` returned the allowed nodes' ids and texts and **dropped their scopes**.
`NodeInputContext` carried `memory_texts` as bare strings. So a node that legitimately read
a tagged memory node and paraphrased it emitted output inheriting nothing, and the
paraphrase was clean at every downstream surface — in *every* arm, including the full
defence, where only the semantic detector stood between it and release. That is exactly the
laundering path persistent-write protection is described as closing. Retrieved-memory
scopes now flow into the envelope layer via `NodeInputContext.memory_forget_ids`, and
`DefenseCounters.memory_borne_scope_hits` counts them so the pathway's liveness is
measured rather than assumed.

#### The corrected arms

The propagation pair must be matched on *exposure*, not just on flags. Both arms now read
the tagged source and generate from it — `guard_retrievals: false`, `guard_node_inputs:
false` — and are compared on what happens to what the model produced:

    graphforget_no_forward      derivative carries no scope  -> edges/writes/final pass
    graphforget_taint_forward   derivative inherits the scope -> caught at all three

`guard_node_inputs` is new: without it the guard refuses the node the moment its input
carries a scope, and the model never derives anything — the same vacuity one layer down.

`tag_local_only` is renamed `tag_source_quarantine` and keeps its real job as the
source-quarantine **positive control** (M4). The old pair survives as **M8**, reported and
asserted structurally null, so a reader cannot mistake its zero for M5's.

Verified end to end on the CPU stub, 11 arms:

    no_forward      returned=40 withheld=0 edges_with_scope=0  edges_blocked=0  writes_tagged=0
    taint_forward   returned=40 withheld=0 edges_with_scope=40 edges_blocked=40 writes_tagged=40
    M5 policy_violating_persistent_leak  Δ=-1.0  CI(-1.0,-1.0)  SUPPORTED
    M8                                   Δ= 0.0  CI( 0.0, 0.0)  positive_control

#### Contrasts are no longer all the same kind of claim

`MECHANISM_CONTRASTS` carries a `kind`, and the report prints it as a column with the
legend attached:

    causal            matched exposure, one variable, live pathway   M1 M2 M3 M5
    positive_control  a component works; says nothing about mechanism M4 M8
    combined          varies more than one thing, labelled as such    M6 M7

M6 (`full − semantic_only`) varies consumption *and* forwarding; M7 (`full − taint_only`)
varies detection *and* accumulation. Both were being read as single-variable and are not.

`tests/contract/test_propagation_pathway_is_live.py` asserts the preconditions on a real
executor run: equal tagged-source exposure in both arms, zero forwarded ids in the
baseline, nonzero in the treatment, nonzero downstream `inherited_only` interventions, and
that the quarantine pair really is the structurally-null one. If those hold and M5 is null
on the GPU, that is a legitimate negative result about forward propagation. If they do not
hold, M5 is not evidence of anything.
