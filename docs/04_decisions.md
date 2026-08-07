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
