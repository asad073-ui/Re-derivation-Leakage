# Risk register

Live document. Update status as risks resolve or materialise.

Severity: **H** = kills the paper · **M** = costs a week · **L** = costs a day

---

## R1 · **H** · The NPO forget10 checkpoint may not be published

**Status:** RESOLVED — 2026-08-07. The checkpoint exists; the id we were carrying did not.

`open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10`
is published and is the run `docs/repro.md` was generated under (lr 1e-5, beta 0.1,
alpha 1, 10 epochs). Verified on the Hub.

**What was actually wrong.** The repo pointed at
`open-unlearning/tofu_Llama-3.2-1B-Instruct_NPO_forget10`, which has never existed.
Unlearned checkpoints do NOT follow the `tofu_<model>_<METHOD>_<split>` pattern of the
finetuned and retain ones — they are published one-per-hyperparameter-setting as
`unlearn_tofu_<model>_<forget_split>_<METHOD>_lr..._beta..._alpha..._epoch...`. The
old id resolved fine locally and would have 404'd on Colab after the environment
bootstrap and the data download. See ADR-0021.

**Residual risk.** Only forget10 unlearned checkpoints are published for this
architecture — there are none for forget01 or forget05. See R5.

**Do NOT mitigate by training NPO on the T4.** fp16 + a gradient-ascent objective is a
silent-NaN failure mode. `hardware.assert_training_allowed` blocks it.

**Pinned offline** by `tests/unit/test_checkpoint_ids.py`, so a regression fails on the
laptop rather than on the GPU box.

---

## R2 · **H** · C3 may be unmeasurable if agent B retains nothing

**Status:** OPEN — pre-registered in `00_preregistration.md` §6.

C3 is measurable **iff** B retains residual knowledge of forget10 after unlearning. A
perfectly-unlearned B has nothing to launder. If C3 ≈ 0, that outcome is ambiguous
between "the effect size is zero" and "the pipeline is broken".

**Detection:** pinned in advance by
`tests/contract/test_condition_c3_stub.py::test_c3_is_unmeasurable_when_b_is_perfectly_unlearned`.

**Mitigation (mandatory before drawing any conclusion):** measure B's residual knowledge
of forget10 in isolation — a C0-style run against agent B alone — and report it. That
number belongs in the paper either way. The published `forget_truth_ratio` of 0.70 vs
the retain model's 0.63 suggests the residual is non-zero, but that is an inference, not
a measurement.

---

## R3 · **H** · The delegation-rate confound

**Status:** MITIGATED by design; must be verified on real models.

Unlearning drops A's `model_utility` from 0.60 to 0.46. Under abstention routing, A
abstains more on forget questions partly because it forgot and partly because it got
worse in general. If the C3 effect appears only under abstention routing, we measured
utility collapse, not forgetting.

**Mitigation:** the `always_delegate` control arm, run automatically by
`run_condition.py`. Pre-registered as a hard gate. Never pass `--no-controls` for a
reported result.

---

## R4 · **M** · Version drift breaks the reproduction

**Status:** OPEN until §5 of `02_repro_targets.md` is filled in.

Upstream's numbers depend on specific `torch`/`transformers`/`datasets` versions. Drift
is the single most likely cause of a failed reproduction, and it presents as a metric
mismatch that looks like a genuine unlearning difference.

**Mitigation:** `rdl env-check --write-versions docs/02_repro_targets.md` on the first
Colab run, then pin `requirements-gpu-t4.txt` to the resolved versions. Bisect order is
fixed in `02_repro_targets.md` §4 fallback 2 — one change at a time.

---

## R5 · **M** · No published unlearned checkpoint exists for a disjoint forget set

**Status:** OPEN — accepted limitation; the arm is withdrawn.

Two problems, the second fatal to the arm as designed:

1. TOFU's forget splits are nested (forget01 subset forget05 subset forget10), so
   forget05 was never a *disjoint* control, only a partially-overlapping one.
2. **There is no published NPO checkpoint for forget01 or forget05 on
   Llama-3.2-1B-Instruct at all.** The org publishes unlearned checkpoints for forget10
   only. The config named a repo that does not exist.

**Action taken.** `B_unlearned_disjoint` and `tofu_llama32_1b_npo_forget05` are deleted
(ADR-0016). The question they were for — does the leak persist when B never saw this
forget set? — is unanswerable without training a checkpoint, which needs Ampere.

**What replaces them.** `B_unlearned_independent` is a *second, independently trained*
unlearning of the SAME forget set (lr2e-05, beta0.5). It does not answer the disjoint
question, but it does answer the one the paper actually needs: are these two agents, or
one agent queried twice? See ADR-0017.

---

## R6 · **M** · Llama-3.2 is a gated repo

**Status:** OPEN until verified.

Every TOFU checkpoint pull 401s without licence acceptance on the *same account as the
token*.

**Mitigation:** accept the licence on day 1 at
<https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>. `rdl env-check` reports the
status explicitly and names the URL. Approval is usually instant; hitting it at hour six
costs a session.

---

## R7 · **M** · `framework_default` could be attacked as a strawman

**Status:** MITIGATED, but the citations must survive review.

If a reviewer believes we designed the write policy to leak, the result collapses.

**Mitigation:** `FrameworkDefaultWritePolicy` mirrors documented defaults of Letta/MemGPT,
mem0, and the LangGraph checkpointer, cited in the module docstring. **Verify each
citation against the current version of each framework before submission** — these
projects move fast, and a stale citation is worse than none.

---

## R8 · **L** · The `entailment` mode is a proxy, not an NLI model

**Status:** ACCEPTED, disclosed.

Token-F1 with a 0.6 threshold. Chosen so the CPU gate runs offline.

**Mitigation:** `nli_fn` injection point exists for the GPU boxes. The paper must state
which was used for each number. ADR-0009.

---

## R9 · **L** · Containment false positives inflate every recovery number

**Status:** MITIGATED and measured.

**Mitigation:** the false-positive floor is a pre-registered gate (≤ 0.05 on retain
questions) and is asserted offline in
`tests/unit/test_containment.py::test_false_positive_floor_on_retain_strings_is_zero`.

---

## R10 · **L** · A Colab session leaks a credential into git history

**Status:** MITIGATED, defence in depth.

**Mitigation:** `nbstripout` + `gitleaks` in pre-commit; `**/token*` and `**/*secret*` in
`.gitignore`; a credential-pattern grep in `scripts/99_sync_results.sh` **and** in CI.
The notebook clones with the PAT and then immediately rewrites the remote without it,
and uses a `GIT_ASKPASS` helper so the token never reaches a file or a cell output.

---

## R11 · **L** · Colab sessions die mid-run

**Status:** MITIGATED.

**Mitigation:** `JsonlWriter` flushes on every record, so a killed session leaves a
readable prefix. Results are append-only and content-addressed, so a resumed session
never overwrites a partial run — it starts a new `run_id`.

---

## R12 · **H** · The paper cannot claim to falsify SBU end to end

**Status:** OPEN — scope narrowed in `00b_preregistration_v2.md` §0.

SBU identifies parametric regeneration followed by memory write-back as "backflow" and
addresses it through a **parameter-side** pathway as well as a memory-side one. This repo
substitutes NPO for the parameter method and implements only a (strengthened) memory
pathway. A result here says nothing about the parameter pathway, and "we falsified SBU"
would be read as covering both.

**Mitigation.** The registered claim is the narrower one: an ID blocklist enforced at
retrieval plus deletion over the derivation closure does not prevent unlearned content
from reaching a shared persistent store, and the carrying node satisfies both invariants.
Generalisation to SBU's full two-pathway system is stated as future work.

---

## R13 · **H** · A large effect may be redundancy rather than multi-agent recovery

**Status:** OPEN — this is what C3 now exists to detect.

If `C3D − C1W` is large but `C3 − C1W` (one checkpoint queried twice) is just as large,
the effect is "we asked the model twice and wrote both answers", not collaboration.
Reporting the first without the second would be the paper's most obvious weakness, in
the same way C2 would have been.

**Mitigation.** `make-report` computes both pairings on every run and the table shows
them side by side. The kill criterion in `00b_preregistration_v2.md` §4 fires when they
are indistinguishable.

---

## R14 · **M** · Our batch-1 numbers are not bit-identical to the published reference

**Status:** OPEN — accepted, and now recorded on every report.

`configs/eval/tofu.yaml` ships `batch_size: 32`, which is what produced the numbers in
`docs/repro.md`. We evaluate at `batch_size=1` because batched generation with
left-padding changes greedy output under fp16, and the pre-registration commits to
batch 1 for every number in the paper.

**Consequence.** "Reproduced" means "within +/- 0.01 of", not "identical to".
`compare_to_published` writes a `batch_size_note` onto the report whenever the two
differ, and the paper must say which batching produced which number. If the tolerance
turns out to be tight at batch 1, run once at 32 to separate "our install is wrong" from
"batching moved it".

---

## R15 · **M** · Our memory pathway is stricter than SBU's, which cuts both ways

**Status:** OPEN — mitigated by running both.

Our default keeps deleted nodes as indexed tombstones and counts supporting parents; SBU
removes the target and its vector and counts dependants. Ours makes the defence do more
work, so a leak found under it cannot be blamed on a weakened defence — but it is not
what the paper describes, and a reviewer will check.

**Mitigation.** `configs/memory/sbu_faithful.yaml` runs the paper's version
(`deletion_mode: hard`, `refcount_semantics: dependent_children`). Every headline number
is produced under both and the table says which is which. Under `hard`, invariant 1 holds
trivially — that is a property of the paper's design and is itself part of the argument,
because the leak never touches retrieval. ADR-0014, ADR-0015.

