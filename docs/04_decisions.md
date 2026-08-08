# Decision log (ADRs)

Append-only. One entry per non-obvious choice, dated, with the alternative that was
rejected and why.

This is also what you paste into the paper's appendix when a reviewer asks "why TOFU and
not MedMCQA".

---

## ADR-0001 — 2026-08-07 — `open-unlearning` enters as a pinned submodule, never a fork

**Decision.** `third_party/open-unlearning` is a git submodule at a pinned SHA. We call
it via `eval/openunlearning_bridge.py`, which shells out to `src/eval.py` and parses the
`*_SUMMARY.json` it produces. We never patch it, never vendor it, and never import from
it in a way that would require patching.

**Rejected: fork and modify.** Every metric number we report must be producible by
*their* code on *their* configs, so a reviewer can re-run it. A fork makes every number
we publish contestable on the grounds that we changed their evaluator.

**Consequence.** Any metric we invent lives in `src/rdl/eval/` and is computed on top of
their outputs, never instead of them. Upstream schema drift is caught cheaply by
`tests/integration/test_ou_bridge_parse.py` against a checked-in fixture.

---

## ADR-0002 — 2026-08-07 — `requires-python = ">=3.10,<3.14"`, not `>=3.11,<3.13`

**Decision.** The `rdl` package targets 3.10–3.13.

**Context.** The original spec pinned `>=3.11,<3.13` because open-unlearning targets
3.11. That constraint binds the **submodule's** environment (Colab, the GPU boxes), not
this pure-Python core. The primary development machine has Python 3.10 and 3.14
installed and neither falls in `[3.11, 3.13)`, so the original pin made `make cpu-all`
uninstallable on the very machine the CPU gate exists to run on.

**Rejected: install a third Python.** It would make the gate depend on a machine-specific
setup step, which is exactly the fragility the gate is supposed to remove.

**Consequence.** The CPU gate runs on 3.10 locally and on 3.11/3.12 in CI (matrix in
`.github/workflows/ci-cpu.yml`). The open-unlearning environment is separate and is still
3.11 on Colab. Nothing in `src/rdl` uses a 3.11+ feature (`Self`, `StrEnum`,
`ExceptionGroup`), which the CI matrix enforces.

---

## ADR-0003 — 2026-08-07 — Deleted nodes stay in the index as tombstones

**Decision.** `MemoryStore.delete` marks a node `deleted=True`, prunes its derivation
closure, and adds its id to the blocklist — but **leaves it in the vector index**.
Retrieval suppresses it. `MemoryNode.returnable` documents this.

**Rejected: physically remove from the index on delete.** That was the first
implementation and it was wrong in a way that would have silently invalidated the whole
experiment. If deleted nodes are not in the index, retrieval never sees them, the
blocklist is never consulted, and **invariant 1 is vacuously true**. We would have been
testing `del store[id]`, not SBU's mechanism.

Two things broke visibly and led to the fix: `Retrieval.blocked_node_ids` was always
empty (so the transcript could not show the blocklist doing anything), and the
adversarial probe in `check_invariant_1` could never fire.

**Consequence.** I1 is now checked **behaviourally**: for each blocked node, query with
that node's own content and confirm `retrieve` withholds it. A blocked node that is
merely *present* in the index is reported as `still_indexed` — informational, not a
violation. A blocked node that comes *back out* is the violation.

---

## ADR-0004 — 2026-08-07 — `refcount` counts supporting parents, not citing children

**Decision.** `MemoryNode.refcount` is the number of **live parents supporting** the
node, floored at 1 for an independently grounded node (ingested, or produced
parametrically). Deletion decrements each node in the closure **exactly once**.

**Rejected: refcount as "how many children cite me".** That was the first implementation
and it propagated deletion in the wrong direction — deleting a root left its immediate
child with a positive count purely because a *grandchild* cited it, so the child stayed
live with dead provenance.

**Additional rule, required by the diamond.** With `m1 → {m2, m3} → m4`, deleting `m1`
reaches `m4` by two paths but must decrement it only once (otherwise the store looks
more thoroughly pruned than it is, understating the leak). A single decrement leaves
`m4` at refcount 1, so refcount alone would keep it live even though both its parents
are dead. Hence a node is marked outdated when **either** its refcount reaches zero
**or** every one of its parents is deleted-or-outdated.

Pinned by `tests/unit/test_derivation_closure.py::test_diamond_does_not_double_decrement`.

---

## ADR-0005 — 2026-08-07 — "SysES" is replaced by per-surface containment

**Decision.** The headline system-level recovery metric is
`eval.containment.containment` over four surfaces, with
`persistent_store_after_episode` as primary.

**Rejected: SysES, as written in the proposal.** It does not type-check. Extraction
Strength is defined over a *single model's* token distribution given a prefix. A
multi-agent system has no such distribution — the "prefix" spans several models, a
retrieval step, and a memory write. Publishing a number called SysES would be a category
error dressed as a metric, and it is the kind of thing a reviewer finds immediately.

**Consequence.** Per-agent ES/EM is still computed (`eval/es_em.py`) and belongs in the
appendix, where it is well-defined. The system-level claim rests on containment and
laundering.

---

## ADR-0006 — 2026-08-07 — Headline the laundering rate, not containment

**Decision.** `laundering_rate` is the paper's headline number.

**Reasoning.** Containment says "the forgotten fact came back", which several papers
already report in one form or another. Laundering says "it came back *through a path the
defence certifies as safe*" — which nothing in the literature reports, and which is
exactly our claim. It is computable directly from `InvariantCertificate`, so every row
in the results table ships with a re-verifiable witness.

**Guard against the obvious misreading.** The rate's denominator is *items recovered*,
not *all items*, so a method that recovers nothing would get a vacuous 1.0 if the
denominator were unguarded. `LaunderingReport` always reports `n_recovered` alongside and
attaches an explicit note when it is zero.

---

## ADR-0007 — 2026-08-07 — Exact retrieval only; never an ANN index

**Decision.** `NumpyBruteForce` (default) and `FaissFlat` — both exact. `faiss-gpu` is
banned outright.

**Reasoning.** An approximate index (IVF, HNSW) puts retrieval noise into a leakage
measurement, where a forget-set item that fails to surface becomes indistinguishable
from one the system successfully withheld. The corpus is ≤ 10k nodes, so exact search is
also *faster end-to-end* and removes a CUDA-version failure mode. Say so in the paper.

---

## ADR-0008 — 2026-08-07 — Three abstention detectors, all reported

**Decision.** `lexical` (pre-registered primary), `logprob`, `self_report`, plus an
ensemble. Pairwise agreement and Cohen's kappa go in the appendix.

**Reasoning.** "Agent A abstains" is load-bearing: no abstention, no delegation, nothing
to measure. A single heuristic is a reviewer target, and each of the three has a failure
mode the other two do not share — lexical is brittle to novel paraphrase, logprob needs
a free threshold, self-report depends on instruction-following, which unlearning itself
degrades. Kappa is reported alongside raw agreement because two detectors that both
almost never fire agree ~100% of the time by doing nothing.

**Constraint.** `REFUSAL_PHRASES` is a committed list. Editing it changes a
pre-registered measurement and requires a dated entry here.

---

## ADR-0009 — 2026-08-07 — `entailment` containment mode is a token-F1 proxy, not NLI

**Decision.** The `entailment` mode uses token-level F1 with a 0.6 threshold by default.
A real NLI model can be injected via `nli_fn`.

**Reasoning.** The CPU gate must run offline with no downloads. A proxy that runs
everywhere and is honestly labelled beats an NLI model that makes the gate
network-dependent. **The paper must state which was used for each reported number.**

---

## ADR-0010 — 2026-08-07 — `configs/models/tofu_llama32_1b_npo_forget05.yaml` added

**Decision.** Added a fifth model config beyond the four the spec listed, to support the
`B_unlearned_disjoint` agent, which the spec's `configs/agents/` tree requires but which
had no model to point at.

**Caveat, and it is a real one.** TOFU's forget splits are **nested**
(forget01 ⊂ forget05 ⊂ forget10), so forget05 is *not* disjoint from forget10. This arm
is an approximation until a genuinely disjoint checkpoint is trained on Ampere. Do not
report it as a disjoint control without that caveat attached. Tracked in
[`05_risks.md`](05_risks.md).

---

## ADR-0011 — 2026-08-07 — `tasks.ps1` added alongside the Makefile

**Decision.** A PowerShell script mirroring every Makefile target.

**Reasoning.** The primary development machine is Windows and has no `make`. The
Makefile remains the authority for CI and the Linux/Colab boxes; `tasks.ps1` exists so
the CPU gate can actually be run locally, which is the point of having one.

---

## ADR-0012 — 2026-08-07 — Tolerance comparisons carry a 1e-9 epsilon

**Decision.** `compare_to_published` uses `delta <= tolerance + 1e-9`.

**Reasoning.** `0.46 + 0.01 == 0.47000000000000003` in IEEE 754, so a metric sitting
*exactly* on the tolerance fails a bare `<=` check by 6e-17. Without the epsilon you
spend an hour on Colab bisecting a chat template that was never wrong.

---

## ADR-0013 — 2026-08-07 — Defaults use `is None`, never truthiness

**Decision.** Every `x if x is not None else default()` in the codebase is written that
way deliberately; `x or default()` is a bug here.

**Reasoning.** `DerivationDAG`, `IDBlocklist`, `SemanticBlocklist`, `NumpyBruteForce`,
and `FaissFlat` all define `__len__`. An **empty** one is therefore falsy, so
`x or default()` silently discards a caller's argument at exactly the moment it is
empty — which is construction time. The bug was found by
`test_index_dim_must_match_embedder`, where a passed-in empty index was being replaced by
a freshly built one of the wrong dimension without any error.

---

## ADR-0014 — 2026-08-07 — Tombstones are OUR variant, not "SBU implemented faithfully"

**Decision.** `configs/memory/sbu_id_blocklist.yaml` and every docstring that described
it now say that keeping deleted nodes in the index is a **strengthened variant** of the
memory pathway, not a transcription of SBU. `deletion_mode: hard` implements the paper's
version (target and vector removed) and `configs/memory/sbu_faithful.yaml` selects it.

**Context.** ADR-0003 chose tombstones for a good reason and stated it well: if deleted
nodes leave the index, retrieval never consults the blocklist and invariant 1 is
vacuously true. That reasoning stands. What did not stand was the label. SBU deletes the
target and its vectors; describing our stricter setting as the paper's is the kind of
overclaim a reviewer checks against the source in ten seconds, and the whole argument
depends on the defence having been implemented as its authors describe it.

**Rejected: switch the default to `hard`.** Under `hard`, invariant 1 holds trivially, so
the run says less. The tombstone setting makes the defence do strictly more work — a leak
found there cannot be blamed on a weakened defence.

**Consequence.** Any headline number is produced under both memory configs and the table
says which is which. Under `hard`, "I1 is vacuous" is reported as a property of the
paper's design and is itself part of the argument: the leak never touches retrieval.
Pinned by `tests/unit/test_sbu_fidelity_modes.py`.

---

## ADR-0015 — 2026-08-07 — Both refcount directions are implemented; ours is not SBU's

**Decision.** `MemoryConfig.refcount_semantics` selects `supporting_parents` (ours,
ADR-0004, default) or `dependent_children` (SBU's, reclamation counting).

**Context.** ADR-0004 defined `refcount` as live supporting parents and propagated
deletion downward. SBU defines it as how many nodes depend on a node — classic reference
counting, where deleting a node releases its references and a support node with nothing
left depending on it is reclaimed. These are different mechanisms with different
behaviour on real shapes: upward reclamation frees shared context nodes that downward
pruning leaves live, and downward pruning invalidates derived content that upward
reclamation leaves retrievable.

**Rejected: replace ours with theirs.** Downward invalidation is what invariant 2 checks,
and it is required in both readings. Dropping it would make I2 unenforceable.

**Rejected: keep only ours and call it SBU.** That was the previous state and it was the
overclaim.

**Consequence.** `derivation.prune` takes `refcount_semantics`; both are exercised by
`tests/unit/test_sbu_fidelity_modes.py`, including the assertion that matters in both
readings — a parametric node with no parents is untouched by either.

---

## ADR-0016 — 2026-08-07 — The `B_unlearned_disjoint` arm is withdrawn

**Decision.** `configs/agents/B_unlearned_disjoint.yaml` and
`configs/models/tofu_llama32_1b_npo_forget05.yaml` are deleted.

**Context.** ADR-0010 added them with the caveat that TOFU's splits are nested. The
stronger problem is simpler: **there is no published NPO checkpoint for forget01 or
forget05 on Llama-3.2-1B-Instruct.** The org publishes unlearned checkpoints for
forget10 only. The config pointed at
`open-unlearning/tofu_Llama-3.2-1B-Instruct_NPO_forget05`, which does not exist under
that name or any other.

**Rejected: leave it with a comment.** A config that names a nonexistent repo is a
landmine — it resolves fine locally and 404s on the GPU box after the environment
bootstrap and the data download.

**Consequence.** The question the arm was for ("does the leak persist when B never saw
this forget set?") is unanswerable without training, which needs Ampere. Recorded in
`05_risks.md`. `tests/unit/test_checkpoint_ids.py` fails if a forget01/forget05 unlearned
id reappears in any config.

---

## ADR-0017 — 2026-08-07 — C3 is a redundancy control; the treatment is C3D/C3C

**Decision.** `C3` keeps its definition (one checkpoint in both agent slots) and is
relabelled a **control**. Two new arms carry the treatment: `C3D` (two independently
unlearned checkpoints) and `C3C` (the same two, with agent A's answer handed to B).

**Context.** C3 loaded `tofu_llama32_1b_npo_forget10` into agent A and agent B. Greedy
decoding, same question, same retrieval context, same weights — so agent B returns
exactly what agent A returned. C3 is one model queried twice, which cannot support a
claim about multiple agents.

**Why CI did not catch it.** `test_condition_c3_stub.py` builds agent B with a *different*
knowledge mask from agent A (masked on the first half, residual on the second). That
manufactures heterogeneity production never had, so the tests passed while the design
was wrong. `tests/contract/test_experiment_design_guards.py::test_identical_checkpoints_give_identical_answers`
now demonstrates the production semantics directly, and `config.py` refuses to let C3D or
C3C be configured with one checkpoint in both slots.

**Consequence.** C3D is the ensemble arm; C3C adds the compositional handoff via
`EpisodePolicies.pass_primary_answer_to_secondary`. `C3C − C3D` is the single-variable
contrast that licenses the word "re-derivation"; if it is ~0, the word comes out.

---

## ADR-0018 — 2026-08-07 — The estimand is `C3D − C1W`, not `C3 − C1`

**Decision.** The primary gate becomes multi-agent write-back minus **single-agent**
write-back. `C1W` is added: agent A alone, blocklist enforced, write-back on.

**Context.** `C1` has `writepolicy.mode: disabled`. Nothing is ever written, so the
persistent-store surface is structurally zero and `C3 − C1` is arithmetically "whatever
C3 recovers". It measures **turning writing on**, which is true by construction and which
any single agent with a persistent memory would clear. There was no baseline in which one
agent writes back, so parametric-to-memory backflow by a single model — the phenomenon
SBU already names — was inside the treatment rather than subtracted from it.

**Rejected: edit `00_preregistration.md`.** It is frozen and its SHA is quoted.
`00b_preregistration_v2.md` supersedes it, states every change and its reason, and both
go in the appendix. No GPU result has been observed, which is the only thing that makes a
v2 legitimate.

**Consequence.** `make-report` evaluates four pairings and marks exactly one primary.
`C3 − C1` is still computed and reported for continuity. Pinned by
`tests/unit/test_report_gate.py`.

---

## ADR-0019 — 2026-08-07 — Five greedy reruns are not five seeds

**Decision.** Episode order is permuted per seed (`episode.permute_item_order_per_seed`),
`is_degenerate` flags identical replicates, `condition_delta_gate` refuses to score a
zero-width interval, and the **paired item-level author-clustered bootstrap** becomes the
primary uncertainty estimate.

**Context.** Generation is greedy (`do_sample=False`, `num_beams=1`), the checkpoints are
fixed, and item order was fixed. Nothing consumed the seed. Five "seeds" therefore
produced five copies of one number; the percentile bootstrap over five identical values
returns a zero-width interval, which reads as extraordinary precision and — worse —
trivially satisfies the pre-registered "non-overlapping 95% CIs".

**Why permutation and not sampling.** Turning on sampling would break the determinism
commitment in §7 of the pre-registration and make the reproduction gate incomparable.
Order genuinely varies the measurement because the store is cumulative: which episode
runs first changes what later episodes can retrieve.

**Consequence.** Items (clustered by TOFU author) are the sampling unit for the primary
interval. Twenty questions about one invented novelist are not twenty independent
observations, so cluster resampling is the default. Pinned by
`tests/unit/test_degenerate_seeds.py`.

---

## ADR-0020 — 2026-08-07 — Every episode is scored against its own store snapshot

**Decision.** `containment(store_nodes=...)` and `laundered_items(store_snapshots=...)`
take the store's contents as of the end of that episode. `execute_condition` captures one
snapshot per episode.

**Context.** The store is shared and cumulative, and metrics were computed after all
episodes finished. Episode 1 was therefore scored against a store containing episode 2's
write, and a reproducible episode-1 "hit" was caused solely by a node written in episode
2. `persistent_store_after_episode` and `SysRecall@k` both silently meant "after
everything", which inflates early-episode recall and makes the k in SysRecall@k
meaningless.

**Consequence.** Certification still runs against the final store and DAG — a node's
invariant status does not change once written — but *whether the item was recovered by
the time its own episode ended* is judged on the snapshot. Pinned by
`tests/unit/test_containment_lookahead.py`.

---

## ADR-0021 — 2026-08-07 — Upstream override names are verified, not guessed

**Decision.** `build_eval_command` emits `holdout_split`, `seed`, `eval.tofu.batch_size`
and `eval.tofu.overwrite`, and never emits `retain_split`. `find_summary` takes a
`newer_than` timestamp and every `task_name` is unique per invocation.

**Context**, all verified against the pinned submodule `4ad738a`:

- `configs/experiment/eval/tofu/default.yaml` defines `forget_split`, `holdout_split`,
  `retain_logs_path`. There is **no `retain_split` key in the eval tree** — it exists
  only in the training configs. Hydra aborts on an override for an unknown key, so every
  generated command failed before the model loaded.
- `configs/eval/tofu.yaml` ships `batch_size: 32` and `overwrite: false`;
  `configs/eval.yaml` ships `seed: 0`. `EvalSpec` stored `batch_size=1, seed=42` and sent
  neither, so the report header disagreed with the run, and a rerun into an existing
  output dir skipped metrics whose logs were already there.
- `setup_data.py`'s argparse defines `--eval_logs`, not `--eval`. The bootstrap script
  and the notebook both passed `--eval`, which aborts with "unrecognized arguments".

**Consequence.** `tests/unit/test_ou_eval_command.py` asserts each of these offline, so
a wrong override fails on the laptop rather than on the GPU box. The batch-size
divergence from upstream's published reference is recorded on every repro report.

---

## ADR-0022 — 2026-08-07 — The agent loop prompts the model the way the gate does

**Decision.** `format_prompt` defaults to `style="openunlearning"`: with no retrieval
context the user turn is the **bare question**, and the system message reaches the
tokenizer as a separate role. Agent configs carry
`system_prompt: "You are a helpful assistant."`.

**Context.** `third_party/open-unlearning/configs/model/Llama-3.2-1B-Instruct.yaml` sets
`template_args.system_prompt: "You are a helpful assistant."` and puts the question
through the chat template unadorned. Our agents used `system_prompt: null`, wrapped the
question in `Question:` / `Answer:` scaffolding, and inlined the system text into the
user turn. Agent-loop results were therefore produced under a different prompt from the
reproduction gate and could not be placed in the same table as it.

**Rejected: leave the scaffold and caveat it.** The whole point of the Days 1-2 gate is
that the Days 3-5 numbers inherit its credibility.

**Consequence.** `qa_scaffold` is kept for completion-style checkpoints and is documented
as not comparable with the gate. `LMHandle.generate/logprobs` take `system` as a
first-class argument so the stub and the real handle stay signature-identical.

---

## ADR-0023 — 2026-08-07 — C0 starts from an empty store

**Decision.** `MemoryConfig.ingest_forget_set` gates ingestion; `configs/memory/none.yaml`
sets it false.

**Context.** `seed_store` ingested every question+answer pair and then, when
`memory.blocklist == none`, returned **without deleting them**. That is exactly C0's
config, so the "bare NPO checkpoint" sanity condition could retrieve the ground-truth
answers straight out of memory. Its recovery number would have been read as "the
checkpoint remembers" and would have meant "the fixture was in the store".

**Consequence.** C0's store is empty and `tests/contract/test_experiment_design_guards.py`
asserts that retrieval returns nothing for every forget-set question. The same function
now honours `memory.blocklist` instead of hard-coding `build_blocklist("id")`, which is
what had made the Phase-2 semantic defence unrunnable.

---

## ADR-0024 — 2026-08-07 — Production runs on real TOFU, and says so when it does not

**Decision.** `eval.tofu_data.load_items` is the single entry point. It loads the real
split by default, refuses the development fixture unless `--allow-fixture` is passed, and
returns a `provenance` block that goes verbatim into the run report and the manifest.

**Context.** `run_condition` called `load_fixture()` unconditionally. `data.dataset`,
`data.forget_split` and `data.n_items` were parsed, validated, hashed into the config
id — and ignored. Every run evaluated the eight hand-written fixture items while the
config, the report and the pre-registration all said 400.

**Consequence.** `load_tofu` also refuses a split that loads short of its known size, and
`make-report` treats fixture provenance as a blocker. Pinned by
`tests/contract/test_experiment_design_guards.py` and `tests/unit/test_report_gate.py`.

---

## ADR-0025 — 2026-08-07 — `make-report` applies the gate and exits non-zero

**Decision.** `rdl make-report` pairs the conditions, runs `condition_delta_gate` and
`paired_delta_gate`, checks `laundering_rate >= 0.5`, checks the control verdicts and the
data provenance, writes `results/gate_verdict.json`, and **exits 1 when the gate fails**.

**Context.** The report printed one row per condition and stopped. Nothing paired C1 with
C3, nothing called `condition_delta_gate`, nothing enforced the laundering criterion or
the controls, and the exit code was 0 whatever the numbers were. "Did the experiment
pass?" was therefore answered by a human reading a table — i.e. after seeing the results,
which is the thing pre-registration exists to prevent.

**Consequence.** `scripts/03_run_phase0_grid.sh` propagates the exit code and must not
gain a `|| true`. `--no-gate` exists for incomplete grids and prints a warning that the
run is not reportable.

---

## ADR-0026 — 2026-08-07 — The `framework_default` claim is narrowed to what is true

**Decision.** The write policy claims: *across Letta archival memory, mem0's `add()`, and
the LangGraph checkpointer, content that reaches durable memory arrives as a new record
with a fresh id and no provenance edge back to what was retrieved.* The **trigger**
(write on every assistant turn) is ours and is labelled as ours.

**Context.** The previous docstring said all three write the assistant turn to durable
memory by default. They do not: Letta writes archival memory through an agent tool call,
mem0 runs an extraction step rather than persisting the raw turn, and LangGraph's
checkpointer persists thread state while cross-thread long-term memory is a separate
Store the application writes to. One synthetic writer is not "the common default of all
three", and that sentence would not have survived a reviewer who uses any of them.

**Consequence.** The property the argument actually needs — new record, fresh id, no
provenance edge — is true of all three and is what puts the record outside the scope of
an id blocklist and a derivation closure. The trigger is the most permissive choice,
which is right for a "can this happen at all" measurement; `write_on_abstention` and
`write_source_kinds` exist so the sensitivity can be reported.

---

## ADR-0027 — 2026-08-07 — Phase 0 runs on an RTX 3090; the T4 profile is kept, not replaced

**Decision.** `configs/env/vast_rtx3090.yaml` is the production environment and every
condition file now names it. `colab_t4.yaml`, `rtx3090.yaml` and `h100.yaml` remain, and
any of them can be selected per invocation with `rdl run-condition --env <name>`.

**Context.** A 16 GB T4 with no bf16 datapath is the wrong instrument for a design whose
treatment arms (C3D, C3C) hold two 1B checkpoints at once. An RTX 3090 gives 24 GB,
native bf16, FA2-capable silicon and no session limit.

**Rejected: replace every `env: colab_t4` and delete the T4 profile.** The conditions are
hardware-independent by construction — that is what makes a result checkable on a box
that is not the one that produced it. A reviewer with nothing but Colab must still be
able to re-run the smoke tests, and a result that exists on exactly one machine is a
result nobody can check.

**Consequence.** `--env` replaces the whole env **group**, which a dotlist override
cannot do: by the time `--set env=rtx3090` is applied, `env` is a populated mapping, so
it either fails validation or (as `env.name=`) relabels the profile while leaving the
previous one's values — Colab's `/content` HF cache — in place on a box where that path
does not exist. Pinned by `tests/unit/test_config.py`. The env is part of the
`config_hash`: two runs on different hardware are not the same run and the id says so.

---

## ADR-0028 — 2026-08-07 — Control verdicts are three-valued, and only FAIL blocks

**Decision.** `eval/controls.py` reports `PASS | FAIL | NOT_APPLICABLE` per control.
`make-report` blocks on `FAIL` only, via `controls.is_blocking`, which also reads the
legacy boolean form correctly.

**Context.** C0 and C1W are single-agent by definition. There is no agent B, so nothing
delegates and `always_delegate` is not a routing policy that can be applied to them. The
verdict was a bare boolean; the absent control produced `False`; `make-report` read
`False` as "the effect does not survive under always_delegate" and raised a **global**
blocker. Running C1W — the baseline the primary estimand is measured against — therefore
made the entire grid unreportable no matter what the numbers were.

**Rejected: skip single-agent conditions when checking controls.** That hides a real
failure in a two-agent arm whose control arm silently produced nothing. The distinction
that matters is "this control cannot exist here" versus "this control was not evaluated",
and only an explicit third value can carry it: a two-agent arm with an empty
`always_delegate` result is now `FAIL`, not `NOT_APPLICABLE`.

**Consequence.** `run_condition` passes `is_multi_agent` and `has_retain_arm` explicitly
rather than letting them be inferred. The applicability flags are written into every
control report. Pinned by `tests/unit/test_controls_applicability.py` and
`tests/unit/test_report_gate.py::test_single_agent_baseline_does_not_block_the_grid`.

---

## ADR-0029 — 2026-08-07 — Days 1-2 are a prerequisite for Days 3-5, enforced by `make-report`

**Decision.** `make-report` blocks unless: `--target full` and `--target npo_forget10`
have both been reproduced **and passed**; every other checkpoint an arm loaded has an
individual `--measure-only` characterisation; every checkpoint is revision-pinned and the
Days 1-2 run used the same revision as the grid; and the conditions agree on dataset,
splits, clustering and seed count.

**Context.** The reproduction table was *displayed* in the report and *required* by
nothing. A grid could pass its gate on an installation that had never been shown to
compute upstream's metrics correctly, on checkpoints whose individual forgetting was
unmeasured. A `C3D - C1W` delta between two unvalidated systems is not evidence.

**Rejected: a warning.** The pre-registration's whole mechanism is that criteria are
applied by code and produce a non-zero exit. A warning is read after the numbers.

**Consequence.** Agent B needs `run-repro --measure-only --checkpoint-label
agent_b_independent` before any C3D/C3C number is reportable — which is acceptance item 8
of `00b_preregistration_v2.md`, now enforced rather than described.
`scripts/02_repro_tofu_npo_forget10.sh` runs it as step 4.

---

## ADR-0030 — 2026-08-07 — Every checkpoint is pinned to an exact Hub commit

**Decision.** `configs/models/*.yaml` and `models/registry.py` both carry `revision:`, the
40-character Hub commit sha, and a test fails if the two disagree. `EvalSpec` emits it to
open-unlearning as `+model.model_args.revision=<sha>`.

**Context.** `main` is a moving target. Re-running the same repo commit a month later can
pull different weights, two runs with the same `config_hash` would not be the same
experiment, and nothing in the report would say so.

**Consequence.** The `+` prefix is required: `revision` is not a key in upstream's
`model_args`, and Hydra rejects a plain override for an absent key — the same failure mode
as `retain_split=`. Upstream splats `model_args` into `from_pretrained`, which accepts
`revision`. Resolved on the Hub 2026-08-07:

| checkpoint | commit |
|---|---|
| `tofu_Llama-3.2-1B-Instruct_full` | `88e31200b97e4c0c04ae0d2f0b591f427046d192` |
| `tofu_Llama-3.2-1B-Instruct_retain90` | `7114300c0049527a71833f5683965c358ad9dcbf` |
| `unlearn_..._NPO_lr1e-05_beta0.1_alpha1_epoch10` (agent A) | `94ed64eb73bc1872d52064833aaef364f4895c9c` |
| `unlearn_..._NPO_lr2e-05_beta0.5_alpha1_epoch10` (agent B) | `eabf32c4883a5647c784c60c998b4b96cd48b798` |

---

## ADR-0031 — 2026-08-07 — FlashAttention-2 capability and availability are separate facts

**Decision.** `HardwareProfile` carries `supports_flash_attn2` (SM >= 8.0) **and**
`flash_attn_installed` (`importlib.util.find_spec("flash_attn")`). `recommended_attn` is
`flash_attention_2` only when both hold, otherwise `sdpa`. Both are printed by
`env-check` and recorded in every report.

**Context.** The detector treated every SM80+ card as FA2-capable. A fresh Vast.ai or
RunPod image has no `nvcc` and therefore no wheel, so the recommendation was an
`ImportError` inside `from_pretrained` — after the 2.5 GB checkpoint had downloaded.
Upstream's own `configs/model/Llama-3.2-1B-Instruct.yaml` hard-codes
`attn_implementation: flash_attention_2`, so nothing else in the stack would have caught
it; the bridge's unconditional override is the only thing between that default and the
model load.

**Rejected: install flash-attn automatically during bootstrap.** It needs a CUDA *devel*
image and ~20 minutes of build time, and failing the bootstrap over an optional
dependency is worse than running under SDPA. `INSTALL_FLASH_ATTN=1` opts in and fails
loudly when `nvcc` is missing.

**Consequence.** SDPA and FA2 runs must not be mixed inside one comparison. Because the
implementation is in every report, a mixed table is detectable after the fact — but it is
not fixable without a rerun, so the runbook says to choose once.

---

## ADR-0032 — 2026-08-07 — Models are loaded once per condition, not once per arm per seed

**Decision.** `run_condition` builds the agents once via `build_shared_agents` and passes
them to every `execute_condition` call — treatment, always-delegate control, retain arm,
agent-A-alone — across all seeds, closing them in a `finally`. Two agent slots naming the
same checkpoint (C3, C1) share one handle.

**Context.** `execute_condition` built and tore down its models on every call: four calls
per seed on a two-agent condition, ~195 model initialisations across the seven-condition
five-seed grid. On a rented GPU that is most of the wall clock, and the repeated
allocate/free cycle is the main source of CUDA fragmentation over a long run.

**Consequence.** Sound only because `LLMAgent` carries no per-run state: the memory store,
the blocklist, the transcripts and the item ordering are all still rebuilt per arm, which
is what a seed is a replicate of. Pinned by `tests/contract/test_shared_agents.py`,
including that seeds still permute episode order — with greedy decoding that is the only
channel through which a seed varies at all.

---

## ADR-0033 — 2026-08-07 — The Day 1-2 trust gate is upstream's settings, not ours

**Decision.** `run-repro` defaults to `batch_size=32, seed=0` — upstream's own eval
defaults, read from `configs/eval/tofu.yaml` and `configs/eval.yaml` at the pinned SHA.
Every report records `published_parity` and a `parity_gaps` list, and `make-report`
blocks a grid whose required reproductions never PASSED at parity. The pre-registered
`--batch-size 1 --seed 42` protocol run is a second, reported run.

**Context.** The defaults were `batch_size=1, seed=42` and the wrapper script told the
operator to use them. That conflates two different questions. The published 0.46 / 0.70
were produced at 32 / 0; a miss at 1 / 42 cannot distinguish "our open-unlearning install
is wrong" from "batching changed the generated answers" from "the seed moved ordering" —
and a run whose miss would have been uninterpretable cannot have an interpretable pass
either. The Day-1 gate exists precisely to answer "is the install correct", so it must be
run where that is the only variable.

**Rejected: gate at batch 1 and treat 32 as a tie-breaker only if it fails.** That
inverts the diagnostic order — you would bisect chat templates and dtypes before checking
the one setting you knowingly changed.

**Consequence.** `is_published_parity` covers batch size and seed only: dtype and
attention are hardware-dependent (a T4 has neither bf16 nor FA2), so they are reported as
parity *gaps* rather than making parity unreachable on some devices. The closest possible
reproduction therefore uses a CUDA devel image with FlashAttention-2 built; the runbook
says so and the report records what was actually used.

---

## ADR-0034 — 2026-08-07 — `env-check --strict` is the preflight; the default stays diagnostic

**Decision.** `rdl env-check --strict` exits non-zero on: an env-profile mismatch, a
missing `HF_TOKEN`, a blocked Llama licence, **any registry checkpoint unreachable at its
pinned revision**, a missing or off-pin submodule, a missing required package, or absent
retain eval logs. `scripts/00`, `scripts/02` and `scripts/03` all run it.

**Context.** `env-check` printed every one of those and exited 0 on all but the profile
mismatch. The single command whose entire purpose is "find out now instead of at hour
six" could not stop a session. Worse, the Hub check covered the base model, `full` and
`retain90` — but not the NPO checkpoint the whole experiment turns on, and it checked
repos rather than repos **at their pinned revision**, so a pin at a removed commit
resolved as healthy here and 404'd inside `from_pretrained` later.

**Rejected: make the default strict.** The diagnostic form is what runs on a laptop with
no token, no submodule and no GPU, where none of that is a problem.

**Consequence.** `strict_blockers` is a pure function over the collected environment, so
all thirteen conditions are testable offline (`tests/unit/test_preflight.py`). The bug
this closes was concrete: `scripts/02` called bare `env-check`, so the Day-1 wrapper's
preflight enforced nothing at all.

---

## ADR-0035 — 2026-08-07 — `flash_attn` is probed by importing it, in a subprocess

**Decision.** `flash_attn_available()` uses `find_spec` as a cheap negative and then
actually runs `import flash_attn` in a subprocess. Cached for the process.

**Context.** `find_spec` only proves Python can *locate* the package. `flash_attn` is a
thin wrapper over a compiled CUDA extension: a wheel built against a different torch or
CUDA satisfies `find_spec` and then fails at import with `undefined symbol`. That is the
same failure the FA2 fallback (ADR-0031) exists to prevent, arriving through a different
door — inside `from_pretrained`, after the checkpoint downloaded.

**Rejected: import it in-process.** A mismatched CUDA extension can abort the
interpreter rather than raise, which would take `env-check` down with it. A subprocess
contains that, and the ~1 s cost is paid once.

---

## ADR-0036 — 2026-08-07 — A profile that names a GPU verifies the GPU

**Decision.** `EnvConfig` gains `expected_gpu_name_regex`, `expected_compute_capability`
and `min_python`. `vast_rtx3090` sets all three (`RTX 3090`, `[8, 6]`, `3.11`);
`rtx3090` and `h100` set only `min_python` and a VRAM floor.

**Context.** `min_vram_gb: 20` plus a bf16 requirement also accepts a 4090, an A5000, an
A6000 or an H100. All of them evaluate correctly — and none of them is what a report
stamped `vast_rtx3090` claims to have run on. That is a provenance defect, not a
capability one, and it is invisible after the fact because the report carries the profile
name it was asked for.

Separately, open-unlearning declares `python_requires >= 3.11` while the `rdl` core runs
its CPU gate on 3.10 (ADR-0002). An arbitrary CUDA image may ship 3.10, and that is
discovered when the submodule install fails — after the instance is running and billing.

**Rejected: make the named profile a warning.** The whole point is that a rented box is
refused before anything downloads. The escape hatch is the generic profile: run a 4090
under `--env rtx3090`, where no card is named and nothing is being claimed.

---

## ADR-0037 — 2026-08-07 — open-unlearning's bf16 eval is broken at its own pins; we shim fp32 logits

**Decision.** `rdl.compat.fp32_logits` restores, at runtime, the unconditional
`logits.float()` upcast that transformers removed in 4.46. The bridge invokes
`python -m rdl.compat.ou_eval_shim src/eval.py …` instead of `python src/eval.py …`, so
the shim is named in the command string every report records, and every report carries
`ou_compat_shims: ["fp32_logits"]`. No file under `third_party/` is modified.

**Context.** At the pinned submodule SHA, `src/evals/metrics/utils.py:98` does
`avg_losses.cpu().numpy()`, and `Tensor.numpy()` has no bfloat16 conversion. Upstream's
own `configs/model/Llama-3.2-1B-Instruct.yaml` pins `torch_dtype: bfloat16`, so the
DEFAULT eval configuration dies with `TypeError: Got unsupported ScalarType BFloat16`
partway through the first metric. `src/evals/metrics/utility.py:62` fails the same way
immediately after. The dates are the whole argument:

    2025-07-20  docs/repro.md last updated, under transformers==4.45.1
    (4.46)      transformers removes the logits.float() upcast
    2026-03-07  open-unlearning a456aa2 bumps the pin to transformers==4.51.3

Commit `a456aa2` touched `src/trainer/*`, `requirements.txt` and the docs. It did not
touch `src/evals/` at all, so the eval path was never re-run against the new pin. The
pinned SHA `4ad738a` is upstream HEAD — there is no upstream fix to move to. This
reproduces with a 1 MB random Llama on CPU; it is not specific to our bridge or to an
RTX 3090.

**Rejected: cast at line 98.** That leaves upstream's cross-entropy running in bf16 —
about three decimal digits of mantissa — and `exp(-avg_loss)` would then miss the
published probabilities by far more than the ±0.01 gate. Upcasting at the source
reproduces the numerical environment the published numbers were produced under.

**Rejected: downgrade to transformers 4.45.1.** Tested: `lm_eval==0.4.11` imports
`transformers.AutoModelForImageTextToText`, which does not exist before 4.47. It would
break two pins the runbook asserts and produce an untested combination.

**Rejected: evaluate in float32.** It runs, but `torch_dtype=float32` is a parity gap,
so `parity_gaps` is non-empty and the run is no longer the published reproduction.

**Consequence.** Model weights stay bf16 and FlashAttention-2 stays on, so
`published_parity` is preserved and `parity_gaps == []`. Validated below.

---

## ADR-0038 — 2026-08-07 — The released NPO forget10 checkpoint does not reproduce its repro.md row

**Decision.** Day 1 stops here. `DAY1_GATE` is NOT met. Nothing downstream —
batch-size-1 runs, Agent B measurement, C1W/C3D, the condition grid — proceeds until
this is resolved. The failing report is committed rather than discarded.

**Context.** Three published checkpoints, all at published parity
(batch 32 / seed 0 / bf16 / FA2, `parity_gaps == []`):

| target | metric | ours | published | delta | verdict |
|---|---|---|---|---|---|
| `full` | model_utility | 0.60122 | 0.60 | 0.0012 | PASS |
| `full` | forget_truth_ratio | 0.47536 | 0.48 | 0.0046 | PASS |
| `retain90` | model_utility | 0.59231 | 0.59 | 0.0023 | PASS |
| `retain90` | forget_truth_ratio | 0.62737 | 0.63 | 0.0026 | PASS |
| `npo_forget10` | model_utility | 0.43169 | 0.46 | 0.0283 | **FAIL** |
| `npo_forget10` | forget_truth_ratio | 0.64140 | 0.70 | 0.0586 | **FAIL** |

Against upstream's published eval LOG for `full` (not the rounded table), our run agrees
to ~2e-3 on `model_utility`, 3.9e-5 on `forget_Q_A_Prob`, and reproduces
`forget_quality = 3.9054713571083378e-22` to every printed digit. The evaluator's
demonstrated error on two independent reference checkpoints is ≤ 0.005. The NPO gap is
6–23× that.

Ruled out, one variable at a time:

* **checkpoint identity** — the pinned revision `94ed64eb` IS `main`; the repo has one
  model upload and no other candidate;
* **published targets** — re-derived from the `Llama-3.2-1B-Instruct` table in
  `docs/repro.md`; `PUBLISHED_TARGETS` transcribes it correctly;
* **run settings** — `parity_gaps == []`, so this is not batch size, seed, dtype or
  attention;
* **install / evaluator** — validated by `full` and `retain90` on the same command path.

`open-unlearning/eval` publishes eval logs for reference models only (full, retain90/95/99);
there is **no** published eval log for any unlearned checkpoint, and the NPO model card is
an empty auto-generated template. Nothing ties the released artifact to the table row.
Upstream's own caveat: "Results may vary even with the same effective hyperparameters
when trained with modifications to the distributed training setup, including when
training on a single GPU. Please use the below numbers only for reproducibility purposes."

**Consequence.** The most probable reading is that the released NPO checkpoint is a
different training run from the one that produced the row. That is a claim about the
artifact, not about this installation — and it is exactly the claim the Day-1 gate exists
to surface before anything is built on top of it.

---

## ADR-0039 — 2026-08-07 — The historical evaluator confirms it: the released NPO checkpoint, not our stack

**Decision.** The fp32-logits shim (ADR-0037) is accepted as a faithful reconstruction
of the historical evaluator, and ADR-0038's conclusion is upgraded from "most probable
reading" to **established**: the released NPO forget10 checkpoint does not reproduce its
`docs/repro.md` row under the exact software that row was published with.

**Context.** ADR-0038 could not separate two hypotheses: a wrong artifact, or a
historical software difference the shim failed to reconstruct. So the checkpoint was
re-evaluated against open-unlearning at `fd825ea` — the commit that last touched
`docs/repro.md`, 2025-07-20 — in an isolated venv pinned to that commit's own
`requirements.txt`: transformers 4.45.1, huggingface-hub 0.29.1, lm-eval 0.4.8,
torch 2.4.1, numpy 2.2.3, datasets 3.0.1, accelerate 0.34.2. **No shim** — 4.45.1 still
upcasts logits, which is itself an independent confirmation of ADR-0037's root cause:
the bf16 crash simply does not occur there.

| checkpoint | metric | historical exact | shimmed current | published |
|---|---|---|---|---|
| `full` | model_utility | 0.60010 | 0.60122 | 0.60 |
| `full` | forget_truth_ratio | 0.4753644629178423 | 0.4753644629178423 | 0.48 |
| `retain90` | model_utility | 0.59018 | 0.59231 | 0.59 |
| `retain90` | forget_truth_ratio | 0.6273686349110743 | 0.6273686349110743 | 0.63 |
| `npo_forget10` | model_utility | 0.43237 | 0.43169 | **0.46** |
| `npo_forget10` | forget_truth_ratio | 0.6413989131591031 | 0.6413989131591031 | **0.70** |

Two facts settle it. First, `forget_truth_ratio` is **bit-identical** between the two
runtimes on all three checkpoints, and `forget_quality` likewise — the shim is not an
approximation of the historical evaluator on the gated metrics, it is the same number.
`model_utility` differs by 1–2e-3, consistent with generation-driven ROUGE terms.
Second, the historical runtime reproduces `full` and `retain90` *better* than the
shimmed one (|d| 0.0001 and 0.0002 on model_utility) and still misses NPO by 0.028 and
0.059 — 100–300× its own demonstrated error.

**Consequence.** No evaluator-side explanation remains. Day 1 stays failed, nothing
downstream proceeds, and the next action is an upstream question about the artifact, not
a further code change. `ou_runtime_mode` now distinguishes `historical_exact` from
`current_with_fp32_logits_shim` so the two are never conflated in a later report.

---

## ADR-0040 — 2026-08-07 — Parity means all four settings AND a checkoutable tree

**Decision.** `is_exact_published_parity()` requires batch_size 32, seed 0, bfloat16,
flash_attention_2, empty `parity_gaps`, and a clean git tree. `make-report` gates on it
via `report_is_exact_parity()`. `run-repro` refuses to start on a dirty tree unless
`--allow-dirty`, which permanently marks the report a diagnostic.

**Context.** Two independent holes, both found by review of the Day-1 runs.

`published_parity` only ever meant batch size and seed. dtype and attention went into
`parity_gaps`, which no gate inspected — so a batch-32/seed-0 run under SDPA satisfied
the Days 1-2 prerequisite while not using the documented FlashAttention-2.

Worse, the Day-1 GPU reports record `git_sha: 1ea12bf` and a command invoking
`python -m rdl.compat.ou_eval_shim`, which `1ea12bf` does not contain. The runs were
made from a working tree carrying the then-uncommitted shim. The numbers are not
thereby wrong — they are reproduced above under an independent runtime — but a reviewer
checking out that SHA cannot run the recorded command, and nothing in the report said
so. Those reports are retained as diagnostic records of a dirty tree, not as
authoritative artifacts.

**Rejected: treat a missing `parity_gaps` as parity.** Reports predating the field carry
no claim about dtype or attention, and an absent claim must not read as a passing one.

**Consequence.** Reports gain `exact_published_parity`, `git_dirty`, `git_diff_sha256`,
`ou_runtime_mode`, `ou_source_sha`, `transformers_version` and `tokenizer` (repo,
revision, chat-template SHA-256 — upstream reads the tokenizer from a moving branch).
Determinism is now established inside the eval subprocess by `ou_eval_shim`: the parent
called `set_all_seeds` and the report claimed `deterministic_algorithms: true`, but
torch settings are process-local and only `PYTHONHASHSEED` and `CUBLAS_WORKSPACE_CONFIG`
were ever inherited by the process that ran the kernels.

## ADR-0041 — 2026-08-08 — The C3C handoff never fired; routing and handoff are decoupled

**Decision.** `C3D` and `C3C` both route **unconditionally** (`always_delegate`), and
the handoff passes agent A's exact output **always**, including an abstention. Config
validation rejects any other combination for those two arms. Routing is now a
condition-level `episode.routing` override rather than a property of the shared
`agent_a` fragment.

**Context.** Two conditions in `orchestrator/loop.py` were mutually exclusive:

```python
decision = pol.delegation.should_delegate(final_reply, n_delegations)   # abstention_triggered:
if not decision.delegate: break                                         #   B is called IFF A abstained
...
if pol.pass_primary_answer_to_secondary and not final_reply.abstained:  #   text passed IFF A did NOT abstain
    peer.append(final_reply.text)
```

Under the shipped `C3C.yaml` (which inherited `abstention_triggered` from
`configs/agents/A_unlearned.yaml`), B was invoked only on A's abstention, and on exactly
those episodes the peer answer was withheld. `peer_answers` was therefore always empty:
**C3C was byte-identical to C3D**, and `C3C - C3D` was structurally zero. No test caught
it — `test_experiment_design_guards.py` checks only that the *config flag* cannot be
unset, and `test_prompt_style.py` exercises `LLMAgent.answer(peer_answers=...)` directly,
never through the loop.

**Rejected: keep abstention routing and pass the answer unconditionally.** Then B sees
"I don't know." on every episode it is called, and the handoff carries no content. The
handoff needs A to have *answered*, which under abstention routing is exactly the case
where B is not called.

**Consequence.** The ecological (abstention-routed) variants of C3D/C3C are still run and
reported, as a secondary result. The estimand moves to the unconditional arms, where the
two conditions differ in exactly one variable.

## ADR-0042 — 2026-08-08 — The primary estimand is compositional, and B gets a baseline

**Decision.** The primary estimand becomes `C3C - C3D` plus `joint_only_recovery`
(`C3C_hit AND NOT C1W_hit AND NOT B1W_hit`). A new single-agent condition **`B1W`** —
agent B alone, write-back on — is mandatory. `C3D - C1W` is demoted to secondary and is
joined by `C3D - B1W`.

**Context.** `C3D - C1W` shows that adding a second checkpoint beats one checkpoint. It
cannot distinguish joint reconstruction from agent B simply retaining more of the forget
set than agent A: B was unlearned at different hyperparameters (lr2e-05/beta0.5 vs
lr1e-05/beta0.1) and has no measured residual to compare against. With only an A-alone
baseline, "multi-agent gain" and "B's residual" are the same number.

SBU already states that residual parametric knowledge can regenerate forgotten content
and be written back into memory — its property (iii) exists to prevent that rewrite. A
result in which *either* agent alone produces the target and the system persists it
under a fresh id therefore replicates a known problem. The stub contract test
`test_c3_recovers_exactly_b_residual_knowledge` demonstrates precisely that shape, and it
was the repo's demonstration of the headline claim.

**Consequence.** The headline becomes `certified_joint_leak_rate`, whose denominator is
all 400 forget items. `laundering_rate` stays as a diagnostic printed beside
`n_recovered`; conditional on a handful of recoveries it reaches 1.0 too easily to be a
headline.

## ADR-0043 — 2026-08-08 — Correct retain answers are utility; the floor is deranged targets

**Decision.** The retain control produces two numbers. **Retain utility** — recall of the
*correct* retain answer — is reported and never gated. The **false-positive floor** is
recall against a deterministic derangement of the retain answers (no fixed points, seeded
by the run seed) and is gated at <= 0.05.

**Context.** `eval/controls.py` computed `false_positive_floor =
sys_recall_at_k(results_retain, ...)` where `results_retain` scores each retain episode
against **its own correct answer**. A functioning agent answers retain questions
correctly, `framework_default` write-back persists the answer, and the store then
contains it — so the "false-positive floor" of a working system approaches 1.0 and the
<= 0.05 gate FAILS every healthy run. The CPU test that pinned the floor at zero
(`test_c3_false_positive_floor_on_retain_questions_is_zero`) passed only because the stub
LM has no retain knowledge and abstains, so nothing is written at all.

A false-positive floor has to be measured against a target the system should *not* be
able to produce. Deranging the retain answers gives exactly that, on real text with real
length and vocabulary, so it also measures the matcher's tolerance rather than a
strawman.

**Consequence.** `ControlReport` gains `retain_utility` (ungated) and keeps
`false_positive_floor` (gated) with a changed meaning. The derangement permutation is
recorded in the report so the floor is recomputable without a rerun.

## ADR-0044 — 2026-08-08 — The confound gate tests the delta, not the level

**Decision.** "Survives under `always_delegate`" means: the **treatment-minus-baseline
paired delta**, recomputed on the unconditional-routing arms of both conditions, clears
its threshold and its 95% interval excludes zero. It is evaluated in `make-report`, which
is the only place that holds both arms. `compute_controls` no longer votes on it and
instead reports the routing arms' numbers plus whether the arm ran.

**Context.** The implementation was `elif rep.recall_always_delegate > 0.0: PASS`. A
single recovered item out of 400 passed the decisive confound control. The control's
purpose — ruling out that the effect tracks agent A's utility collapse rather than
forgetting — is a statement about the *effect*, and the effect is a difference between two
conditions. Absolute recall in one arm cannot express it.

**Consequence.** `run-condition` now writes per-item hit vectors keyed by routing policy
(`per_item_recall_by_policy`), so the routing-free delta is recomputable from the reports
alone. Reports written before this ADR have no such key and are treated as unevaluated,
which blocks.

## ADR-0045 — 2026-08-08 — A handoff that is not in the log did not happen

**Decision.** New typed event `Handoff(from_id, to_id, text, text_sha256,
included_abstention)`, appended immediately before the delegate is called.
`SCHEMA_VERSION` goes to 2. Every arm's transcripts are persisted — treatment, routing
variant, retain, and each standalone baseline — not only the treatment's.

**Context.** The only trace of the compositional handoff in a saved run was
`AgentAnswer` events from both agents, which look identical whether or not B was shown
A's text; `n_peer_answers` lived in `AgentReply.meta`, which is not part of any event.
`run_condition` wrote `transcripts_seed{s}.jsonl` for the treatment arm and discarded the
control, retain and A-alone transcripts entirely. A reviewer asking "did B actually
receive A's output on item 137?" had no way to answer from the artifacts, and neither did
we.

**Rejected: keep SCHEMA_VERSION at 1 and add the kind.** A v1 log is defined by its
closed union; silently widening it makes "this file is v1" mean two different things.
`parse_event` refuses cross-version parsing on purpose.

**Consequence.** `results/` currently contains no transcripts, so no migration is needed.
The checked-in transcript fixture is bumped to v2.

## ADR-0046 — 2026-08-08 — Scale and pairing integrity are gates, not conventions

**Decision.** `make-report` blocks on: `n_items != 400`, `n_seeds != 5`, a retain arm that
is not exactly 100 items, `truncated == true` (i.e. `--limit` was used), unequal item-ID
**sets** between paired conditions, a missing `C1W` or `B1W`, a `delegation_gap_ok` FAIL,
and a handoff count that disagrees with the condition's declared handoff setting.

**Context.** Four separate holes. (1) `--limit 20` on real TOFU produced
`is_real_data: true` and cleared every existing check, so a twenty-item smoke run was
indistinguishable from a result. (2) `_paired_vectors` intersected the two arms'
item-ID lists (`shared = [i for i in t.item_ids if i in bb]`), so a 400-item treatment
paired against a 20-item baseline silently produced a 20-pair delta. (3)
`delegation_gap_ok` was computed by `compute_controls` and read by nobody, although
pre-registration v2 section 3.3 gates on it. (4) Nothing checked that the handoff a
condition claims to perform was performed.

**Consequence.** Pilot and engineering runs must set `reportable: false`, which excludes
them from the gate rather than failing it.

## ADR-0047 — 2026-08-08 — Per-item store reset for the primary; cumulative store as its own experiment

**Decision.** The primary experiment resets the `MemoryStore` to the same post-deletion
snapshot before every item. The cumulative shared-store run is retained as a separate
**longitudinal** experiment whose uncertainty is reported over seeds and which never
feeds the primary gate.

**Context.** `run_episodes` deliberately shares one store across all episodes, and
`_episode_order` permutes so that the seed enters the measurement through that sharing.
That makes episode *i*'s write part of episode *i+n*'s retrievable context — items are
not exchangeable. The primary interval is nevertheless a bootstrap that resamples items
(clustered by author) as if they were. Author clustering handles TOFU's
20-questions-per-author structure; it does not handle a dependence induced by the run
itself, whose grouping is the arrival order, not the author.

**Rejected: keep one store and widen the interval heuristically.** There is no defensible
inflation factor, and the cumulative dynamics are worth measuring in their own right
rather than being treated as nuisance.

**Consequence.** `episode.store_scope: per_item | cumulative`. `per_item` is the default
for C1W/B1W/C3D/C3C; the longitudinal run is an explicit second invocation.
