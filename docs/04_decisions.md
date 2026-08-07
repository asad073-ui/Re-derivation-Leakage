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
