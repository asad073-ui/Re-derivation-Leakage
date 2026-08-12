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

### GU-0014 — 2026-08-12 — TOFU exclusions do not apply to the development fixture

`tests/fixtures/tofu_forget10_sample.json` reuses TOFU item ids for eight invented
questions. Applying the TOFU exclusion list to it blocked the CPU gate over a name
collision rather than over shared content, so `load_cohort` skips exclusion enforcement
when `dataset == "fixture"`.
