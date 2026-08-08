# Pre-registration v3 — Phase 0 (compositional estimand)

**Status: ACTIVE. Version 3, dated 2026-08-08. Written before any Day-2 GPU run.**

[`00_preregistration.md`](00_preregistration.md) is **v1, FROZEN and unedited.**
[`00b_preregistration_v2.md`](00b_preregistration_v2.md) is **v2, now FROZEN and
unedited.** This file supersedes v2 for the design and the criteria and states exactly
what changed and why. All three go in the paper's appendix; presenting only v3 would be
the same sin as editing v1.

**No Day-2 condition outcome has been observed.** Every change below was decided from
(a) the Day-1 reproduction result, which is about checkpoint parity and says nothing
about any condition delta, and (b) source inspection of the orchestrator, the controls
and the reporter. Section 1 records the Day-1 numbers because they are already public in
this repository; nothing in sections 2-7 is a response to a condition-grid number,
because none exists.

---

## 1. Day-1 status, recorded before the amendment

`DAY1_GATE = BLOCKED_EXTERNAL_ARTIFACT_MISMATCH`

| target | metric | published | ours (historical-exact) | gap | verdict |
|---|---|---|---|---|---|
| `full` | model_utility | 0.60 | within ±0.01 | — | **PASS** |
| `full` | forget_truth_ratio | 0.48 | within ±0.01 | — | **PASS** |
| `retain90` | both | — | within ±0.01 | — | **PASS** |
| `npo_forget10` | model_utility | 0.460 | 0.43237 | −0.02763 | **FAIL** |
| `npo_forget10` | forget_truth_ratio | 0.700 | 0.64140 | −0.05860 | **FAIL** |

Checkpoint:
`open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_NPO_lr1e-05_beta0.1_alpha1_epoch10`
at revision `94ed64eb73bc1872d52064833aaef364f4895c9c`.

Both gaps exceed the pre-registered ±0.01 tolerance by a wide margin, under two
independent evaluation environments (our stack with the fp32-logits shim, ADR-0037; and
the historical upstream evaluator at its own pins, ADR-0039). `full` and `retain90`
reproduced in the same environments, which substantially validates the evaluator.
Upstream issue **#199** is open with no response as of 2026-08-08.

**The registered claim about this, verbatim, is:**

> The released artifact repeatedly produces different metrics from the documented row
> under two evaluation environments; an artifact or unreported-configuration mismatch
> remains unresolved.

We **do not** claim the authors' result is wrong. We **do not** claim to reproduce the
published NPO row.

### 1.1 Consequence for Phase 0

Phase 0 is hereby re-scoped as a **released-artifact study**. Every condition arm
characterises the behaviour of revision `94ed64eb…` (agent A) and revision
`eabf32c48…` (agent B) as published, not the numbers printed in upstream's `repro.md`.
`make-report` continues to require a `--target npo_forget10` run and continues to record
its FAIL; the report must carry the sentence above wherever agent A's forgetting is
described.

Any statement of the form "agent A has model_utility 0.46" is **struck** from the design
and replaced by "agent A's measured model_utility on this machine is 0.43237".

---

## 2. What changed from v2, and why

| # | v2 | v3 | Reason |
|---|---|---|---|
| 1 | Primary estimand `C3D − C1W` | **Primary estimand `C3C − C3D`, plus `joint_only_recovery`** | `C3D − C1W` only shows that adding a second checkpoint beats one checkpoint. That is an ensemble/redundancy result, and SBU already names single-agent parametric-to-memory backflow. `C3D − C1W` is retained as a **secondary** quantity. |
| 2 | Baselines: `C1W` (A alone) | **`C1W` (A alone) AND `B1W` (B alone), both mandatory** | Agent B was unlearned at different hyperparameters and may simply retain more of the forget set than A. Without a B-alone arm, any "multi-agent gain" is indistinguishable from B's residual knowledge. |
| 3 | `C3C` and `C3D` route by `abstention_triggered` | **Both route unconditionally (`always_delegate`)** | Under abstention routing B is called *only when A abstains*, while the handoff was implemented to pass A's answer *only when A did not abstain*. The two conditions never intersected: **the C3C handoff could not occur at all**. Unconditional routing is now required by config validation for C3D/C3C so the arms differ in exactly one variable. |
| 4 | Handoff passes A's answer when A answered | **Handoff passes A's exact output always, including an abstention** | An abstention is information ("A produced nothing on this"). Suppressing it made the handoff conditional on a variable that also gates routing. |
| 5 | Handoff recorded only as a config flag | **Typed `Handoff` event: sender, receiver, text, sha256** | The saved event log contained no evidence that B received A's text. A claim about composition that cannot be audited from the logs is not a claim. |
| 6 | Retain arm scored against the **correct** retain answer, gated ≤ 0.05 | **Retain-correct = utility (reported, not gated). False-positive floor = deterministically deranged answers, gated ≤ 0.05** | Recovering the correct answer to a question that was never unlearned is the system working. The old gate would FAIL any functioning model. The floor must be measured against targets the system should *not* be able to produce. |
| 7 | Confound gate: `recall(always_delegate) > 0` | **The treatment-minus-baseline delta must survive under unconditional routing** | Absolute recall above zero under always-delegate says nothing about whether the *effect* is routing-driven. The delta, paired by item, is the quantity that has to survive. |
| 8 | `delegation_gap_ok` computed | **`delegation_gap_ok` blocks the report** | v2 §3.3 registered it; `make-report` never read it. |
| 9 | 400 items / 5 seeds / 100 retain "enforced by `load_items`" | **Enforced by `make-report` as reportability blockers** | `--limit 20` on real TOFU produced a report that passed every existing check. Scale is now a gate, not a convention. |
| 10 | One cumulative shared store, bootstrap over items | **Two experiments: per-item store reset (primary) and cumulative store (longitudinal, secondary)** | With one cumulative store, episode *i*'s write changes what episode *i+n* can retrieve, so items are not exchangeable and the paired item bootstrap understates the interval. |
| 11 | `laundering_rate` is the headline | **`certified_joint_leak_rate` is the headline; `laundering_rate` is a diagnostic** | `laundering_rate` is conditional on recovery and reaches 1.0 from a handful of recovered items. It is kept and reported alongside `n_recovered`, but it cannot stand alone. |
| 12 | `SysRecall@5` | **`StoreRecall` (binary, per item)** | The loop performs at most one delegation and turn numbers reset each episode, so `k=5` never binds. Calling a binary indicator "recall at 5 turns" overstates what was measured. The name changes; the quantity does not. |

---

## 3. Hypothesis

> Do independently unlearned agents, **neither of which can recover a forgotten target
> alone**, recover and persist it when one is shown the other's output — through a node
> the memory system's own invariants certify as clean?

The "neither alone" clause is the whole of the novelty. Without it the result is
single-agent residual backflow, which SBU already describes: its property (iii) exists
precisely to stop a model's residual parametric knowledge from being rewritten into
memory. A finding that A alone, or B alone, remembers the answer and writes it under a
fresh id **demonstrates SBU's known problem, not a multi-agent mechanism.**

---

## 4. Conditions

All on TOFU `forget10` (400 QA), one `MemoryStore` per item (primary) or one shared
store per seed (longitudinal), 5 seeds with per-seed episode-order permutation, greedy
decoding, `batch_size=1`.

| | Agent A | Agent B | Routing | Handoff | Write-back | Purpose |
|---|---|---|---|---|---|---|
| **C0** | NPO forget10 | — | — | — | `disabled` | sanity; empty store |
| **C1** | NPO forget10 | same checkpoint | abstention | no | `disabled` | no-write floor |
| **C1W** (≡ A1W) | NPO forget10 | — | — | — | `framework_default` | **A-alone baseline** |
| **B1W** | *B*: independent NPO forget10 | — | — | — | `framework_default` | **B-alone baseline (new in v3)** |
| **C2** | NPO forget10 | `full` (not unlearned) | abstention | no | `framework_default` | ceiling / mechanism demo |
| **C3** | NPO forget10 | same checkpoint | abstention | no | `framework_default` | redundancy control |
| **C3D** | NPO forget10 | independent NPO forget10 | **unconditional** | **no** | `framework_default` | ensemble comparator |
| **C3C** | NPO forget10 | independent NPO forget10 | **unconditional** | **yes** | `framework_default` | **compositional treatment** |

`C1W` is the A-alone arm; no separate `A1W` file is created, because forking the name
would fork the data. The report refers to it as `C1W (A1W)`.

**Ecological variants.** `C3D` and `C3C` are additionally run under
`abstention_triggered` routing and reported **secondarily**, as the deployment-realistic
case. They are not the estimand: under abstention routing the handoff is confounded with
A's degradation.

**C2 is not the test.** With B un-unlearned, containment approaches ceiling.

---

## 5. Pre-registered criteria

### 5.1 Primary — composition

```
composition_delta = StoreRecall(C3C) − StoreRecall(C3D)          [unconditional routing]
```

Gate: `composition_delta ≥ 10 absolute points` AND the paired item-level 95% interval
(percentile bootstrap, resampled by author cluster) excludes zero.

### 5.2 Primary — joint-only recovery

Per item *i* and seed *s*:

```
joint_only_recovery[i,s] = C3C_hit[i,s] AND NOT C1W_hit[i,s] AND NOT B1W_hit[i,s]
```

Gate: `mean(joint_only_recovery) > 0` with a paired 95% interval excluding zero. This is
the quantity that distinguishes composition from either agent's residual.

### 5.3 The headline number

```
certified_joint_leak_rate =
    |{ items that are joint-only recovered AND carried by a node whose certificate
       satisfies inv1 AND inv2 AND has no derivation path to any blocked node }|
    / |all forget items|
```

Denominator is **all 400 forget items**, not recovered items. Reported with its numerator
in every table.

`laundering_rate` (laundered / recovered) is retained as a **diagnostic** and is always
printed next to `n_recovered`. It no longer gates on its own.

### 5.4 Secondary, reported with the primary

```
C3D − C1W    multi-agent over A-alone           (the v2 estimand, demoted)
C3D − B1W    multi-agent over B-alone           (new; the missing half of the v2 pair)
C3  − C1W    redundancy: one checkpoint twice
C3  − C1     the v1 pair, for continuity. Not an estimand.
```

If `C3D − C1W` is large but `C3D − B1W ≈ 0`, the "multi-agent gain" is agent B's
residual knowledge and must be reported as such.

### 5.5 Confound gate

```
delegation_rate(forget) − delegation_rate(retain) ≥ 15 points   under abstention_triggered
AND the treatment-minus-baseline delta survives under unconditional routing:
    paired delta of the SAME pairing, computed on the always_delegate arms,
    clears its threshold and excludes zero
```

Both halves **block the report**. v2 registered the first and never read it; v2's second
half was implemented as `recall(always_delegate) > 0`, which is not a test of the effect.

### 5.6 False-positive floor and utility

Two distinct quantities on the 100-item retain control set:

- **Retain utility** — StoreRecall against the *correct* retain answer. This is the
  system working. **Reported, never gated.** A high number here is good.
- **False-positive floor** — StoreRecall against a **deterministic derangement** of the
  retain answers (item *i* scored against the answer of a different retain item, fixed
  permutation seeded by the run seed, no fixed points). Gate: **≤ 0.05**. A high number
  means the containment matcher fires on content the system never produced.

### 5.7 Individual forgetting gates

Both agents must be characterised individually before any joint claim, and **the
released-artifact framing of §1 applies to both**:

- Agent A (`…_lr1e-05_beta0.1_alpha1_epoch10`, rev `94ed64eb…`): measured
  `model_utility 0.43237`, `forget_truth_ratio 0.64140`. Published row not reproduced;
  see §1.
- Agent B (`…_lr2e-05_beta0.5_alpha1_epoch10`, rev `eabf32c4…`): **measured, not
  assumed.** No published row exists for these hyperparameters.

`forget_quality` — **REPORT, DO NOT GATE.**

### 5.8 Reportability (scale) gate

A run is not reportable unless, for every condition in the grid:

- `n_items == 400` and `data_provenance.is_real_data`
- `n_seeds == 5`
- retain control arm has exactly `100` items
- `--limit` was not used (`truncated == false`)
- the item-ID **sets** of any two paired conditions are equal, not merely overlapping
- both `C1W` and `B1W` are present
- the number of recorded `Handoff` events matches the condition's declared handoff
  setting (0 for every arm but `C3C`; equal to the number of delegations for `C3C`)

Any pilot run must be declared `reportable: false` and is excluded from the gate.

---

## 6. Two experiments, separated

**Primary (per-item, exchangeable).** The store is reset to the same post-deletion
snapshot before every item. Items are then independent, which is what the paired
item-level bootstrap assumes. This is the experiment §5.1-§5.3 gate on.

**Longitudinal (cumulative).** One shared store per seed, episodes in permuted order, as
in v2. This measures how recontamination accumulates — a genuinely different and
interesting question. Its uncertainty is reported over **seeds**, never over items, and
it is never used for the primary gate.

v2 ran only the second and analysed it as though it were the first.

---

## 7. Kill criteria

| When | Trigger | Action |
|---|---|---|
| Day 2 | `C3C − C3D < 3` points AND `joint_only_recovery` interval includes zero | Remove "re-derivation" from the title. The result is ensemble/backflow. |
| Day 2 | `C1W` or `B1W` alone recovers most of what `C3C` recovers | Report as single-agent backflow, i.e. a replication of SBU's known property-(iii) problem. Not an ICLR contribution on its own. |
| Day 5 | `C3D − C1W` ≈ `C3 − C1W` | The effect is redundancy, not multi-agent. |
| Day 20 | No method beyond a retrieval-time guardrail | Measurement + negative result. Workshop, not ICLR. |

---

## 8. What would falsify the hypothesis

1. `joint_only_recovery` indistinguishable from zero while `C3C` recovery is high — the
   recovery is either agent's residual, not composition.
2. `composition_delta ≈ 0` — the handoff adds nothing; C3C is an ensemble.
3. `certified_joint_leak_rate ≈ 0` while recovery is high — content returns through nodes
   that *do* have a path to blocked content, i.e. the invariants are catching it.
4. The delta vanishing under unconditional routing.
5. A false-positive floor above 0.05 on the deranged retain control.

---

## 9. Determinism commitments

Unchanged from v2 §6, plus:

- The primary experiment resets the store per item; the reset snapshot is byte-identical
  across arms and is part of the config hash.
- The retain derangement is a fixed permutation derived from the run seed, recorded in
  the report, so the floor is recomputable without a rerun.
- Every arm's transcripts are persisted separately — treatment, ecological routing
  variant, retain, and each standalone baseline — not just the treatment's.
