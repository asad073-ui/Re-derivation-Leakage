# Risk register

Live document. Update status as risks resolve or materialise.

Severity: **H** = kills the paper · **M** = costs a week · **L** = costs a day

---

## R1 · **H** · The NPO forget10 checkpoint may not be published

**Status:** OPEN — must be resolved on day 1.

The zero-training cost model depends on `open-unlearning/tofu_Llama-3.2-1B-Instruct_NPO_forget10`
existing on the Hub. Only `_full` and `_retain90` are confirmed.

**Detection:** `python -m rdl.cli discover-checkpoints` — run this first, before anything
else.

**Mitigation:** REPO_SPEC §7.4 fallback 1. Gate on `full` only and validate the metric
code against the published eval logs (`python setup_data.py --eval` →
`tofu*/evals*/*_SUMMARY.json`). That validates the metric code with no model at all.

**Do NOT mitigate by training NPO on the T4.** fp16 + a gradient-ascent objective is a
silent-NaN failure mode. `hardware.assert_training_allowed` blocks it.

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

## R5 · **M** · TOFU forget splits are nested, so there is no true disjoint control

**Status:** OPEN — accepted limitation.

forget01 ⊂ forget05 ⊂ forget10. The `B_unlearned_disjoint` arm therefore uses forget05,
which overlaps forget10. It is an approximation, not a disjoint control.

**Mitigation:** report it as exploratory with the caveat attached, or train a genuinely
disjoint checkpoint on Ampere (not the T4 — see R1). Documented in ADR-0010 and in the
config file itself.

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
