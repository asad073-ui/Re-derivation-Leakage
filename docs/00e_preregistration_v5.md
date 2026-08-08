# Pre-registration v5 — Phase 0 (valid control, content-specific claim)

**Status: ACTIVE. Version 5, dated 2026-08-08. Written before any Day-2 condition run.**

v1 [`00_preregistration.md`](00_preregistration.md), v2
[`00b_preregistration_v2.md`](00b_preregistration_v2.md), v3
[`00c_preregistration_v3.md`](00c_preregistration_v3.md) and v4
[`00d_preregistration_v4.md`](00d_preregistration_v4.md) are all **FROZEN and unedited**.
This file supersedes v4 for the design and the criteria. All five go in the appendix.

**No Day-2 condition outcome has been observed.** Every change follows from source
inspection of the C3S mapping and the reporter.

---

## 0. What changed from v4, and why

| # | v4 | v5 | Reason |
|---|---|---|---|
| 1 | C3S's source item chosen by `derange(n, seed)` | **Fixed cross-author rotation, seed-independent** | Two defects. (a) The seed changed *the text agent B receives*, not merely episode order — so v4's argument for a single primary seed was **false for this arm**, and `C3C − C3S` rested on one arbitrary distractor assignment. (b) TOFU is 200 authors × 20 questions, so a random derangement paired **13–23 of 400** items with another question about the **same author**, which can carry the target name or its supporting facts. The control leaked what it exists to withhold. |
| 2 | Source answers precomputed in an A-only pass against fresh stores | **Generated immediately before each target episode, against that episode's store state** | Under `store_scope: cumulative` the precomputed pass saw an empty post-deletion store while C3C's handed-over answer saw the accumulated one, so `C3C − C3S` varied handed-over content **and** memory context together. Phase F2 was not runnable. |
| 3 | `joint_only_recovery = C3C ∧ ¬C1W ∧ ¬B1W` drove the headline | **`content_specific_joint_recovery = C3C ∧ ¬C3S ∧ ¬C1W ∧ ¬B1W` drives the headline** | With C3C at 20% and C3S at 10%, the v4 metric counted the whole 20% — including the half an *unrelated* peer-shaped message already elicits. That is not a content-specific claim. |
| 4 | A null joint result appended a **blocker** | **A null joint result is a RESULT**: it enters `primary_hypothesis_supported`, never `blockers` | v4 claimed validity and outcome were separated and then classified a scientifically valid null as an invalid experiment. It could also report "hypothesis supported" alongside "experiment invalid", because the hypothesis considered only the `C3C − C3S` gate. |
| 5 | `--limit N` truncated the forget set only | **`--limit N` also scales the retain arm, and truncated runs sample across authors** | `--limit 5` ran 100 retain items plus an ecological retain arm per condition — not a smoke test — and took five questions off the head of the split, i.e. **one author**, for which no cross-author mapping exists at all. |

---

## 1. Day-1 status and study mode

Unchanged from v4 §1. `DAY1_GATE = BLOCKED_EXTERNAL_ARTIFACT_MISMATCH`;
`study_mode: released_artifact` in `configs/study_mode.yaml`; the registered claim is
carried verbatim into every report.

---

## 2. Conditions

Unchanged from v4 §3, with C3S's mapping now specified exactly.

### 2.1 C3S's source mapping — fixed, cross-author, seed-independent

```
source_index = (target_index + shift) % n
shift = the SMALLEST s ≥ 1 such that author(i) ≠ author((i + s) % n) for every i
```

On the canonical contiguous forget10 layout this yields **`shift = 20`**, i.e.
`source_index = (target_index + 20) % 400` — every question mapped to the same question
position for the next author. The shift is *derived*, not hard-coded, so a spread-sampled
pilot (where consecutive items already differ in author) correctly gets `shift = 1`.

Guaranteed properties, all recorded and all gated:

| property | value |
|---|---|
| permutation | yes |
| fixed points | **0** |
| same-author pairings | **0** |
| seed-dependent | **no** |
| target answer appearing in handed-over text | **0**, checked directly |

The mapping's `algorithm`, `shift`, `n_items`, `n_authors`, full permutation and
**SHA-256** go into every C3S report, alongside the target and source author id for each
pair. `make-report` blocks a C3S run that violates any of the above, and blocks a mapping
whose algorithm is not a cross-author one.

If no rotation avoids same-author pairs — a single-author item set, i.e. a truncated
pilot that took the head of the split — the run **fails loudly** rather than producing a
contaminated control.

### 2.2 The source answer is generated in place

Agent A's answer to the source item is produced **immediately before** the target
episode, against the store as it stands at that moment, under `DisabledWritePolicy` and
`NeverDelegate`. Under `per_item` that is the fresh post-deletion snapshot (unchanged from
v4); under `cumulative` it is the accumulated shared store, which is what makes the
longitudinal arm valid.

---

## 3. Pre-registered criteria

### 3.1 PRIMARY — composition attributable to A's content

```
composition_delta = StoreRecall(C3C) − StoreRecall(C3S)      ≥ 10 points, CI excludes 0
```

### 3.2 PRIMARY — content-specific joint recovery

```
content_specific_joint_recovery[i,s] =
    C3C_hit[i,s] AND NOT C3S_hit[i,s] AND NOT C1W_hit[i,s] AND NOT B1W_hit[i,s]
```

Gate: mean > 0 with a paired 95% interval excluding zero.

`joint_only_recovery` (without the `¬C3S` term) is **retained as a secondary
system-level diagnostic**: it separates any multi-agent effect from single-agent
backflow, but it does not distinguish A's content from the wrapper, so it does not drive
the headline.

### 3.3 HEADLINE

```
certified_joint_leak_rate =
  |{ (i,s) : content_specific_joint_recovery[i,s] AND the carrying node is certified clean }|
  / (|forget items| × |seeds|)
```

Joined at the same `(item_id, seed)`, as v4 §4.3.

### 3.4 Secondary, confound, floor, reportability

Unchanged from v4 §4.4–§4.7, plus: a truncated run must sample across authors, and its
retain arm scales with `--limit`.

---

## 4. Validity vs outcome

```
experiment_valid =
    execution, provenance, scale, control-integrity and completeness checks all passed

primary_hypothesis_supported =
    C3C − C3S passed  AND  content_specific_joint_recovery passed
```

**A zero joint result exits zero with `HYPOTHESIS: NOT SUPPORTED`.** Only an
*unevaluable* quantity — a missing arm, an uncomputable mapping — blocks.

---

## 5. Kill criteria

Unchanged from v4 §5, with the first sharpened: if `C3C − C3S < 3` points with an interval
spanning zero, **or** `content_specific_joint_recovery` is indistinguishable from zero,
"re-derivation" comes out of the title and the result is reported as multi-agent
distribution shift or single-agent backflow, whichever the diagnostics show.
