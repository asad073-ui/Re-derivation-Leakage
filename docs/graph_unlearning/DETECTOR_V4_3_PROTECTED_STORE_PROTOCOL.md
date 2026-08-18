# Detector v4.3 — the protected store

**Status: frozen for the CPU phase.** Written before any v4.3 reportable label exists.
Corrections go to `DECISIONS.md` as dated `GU-####` entries, never as edits to this file.

This protocol is **additive**. `DETECTOR_V4_1_PROTOCOL.md` and
`DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md` are untouched, and every v4.1/v4.2 artifact stays
byte-for-byte as frozen — their hashes are quoted in those protocols, and rewriting one
would invalidate a pre-registration rather than correct it. v4.3 adds new files under new
names and a new judge roster under new names.

---

## 1. What v4.3 changes, and why

v4.3 does not replace the detector. `CrossEncoderAnswerabilityDetector` already encodes
question + aliases + candidate and returns answer/partial probabilities, `DetectionContext`
already forbids gold answers structurally, and the routing probe already showed identity
routing is not the bottleneck. What v4.3 does is **formalise the protected set as an
artifact** and repair three measurement defects that would each have produced a good-looking
number about the wrong thing.

### 1.1 The alias channel carried population

`natural_alias_index()` built aliases from the forget-policy cohort. Retain questions are
not in that cohort, so retain rows joined to `aliases: []` and protected rows did not.

Measured on the frozen 1,019-row audit:

| | protected | retain |
|---|---:|---:|
| v4.2 alias coverage | 719/719 | **0/300** |
| v4.3 alias coverage | 717/719 (0.997) | **295/300 (0.983)** |

"Has aliases" was a perfect separator for population. Cross-entropy takes free features,
so a model could reach high accuracy without ever learning answerability, and the recall
number would describe dataset membership.

**Fix.** One extractor — `concept_registry.extract_name_spans` — runs over every question
regardless of origin. `conditioning_records_from_questions()` has no `population`
parameter and no cohort to join against, so the extraction *cannot* branch on population:
there is one code path because there is only one code path.

### 1.2 Retain answers were counted as false alarms

`selection_metrics()` treated the ANSWER score of **every** retain row as a false positive.
Most retain candidates genuinely do answer their retain question and are labelled ANSWER,
so:

- cross-entropy taught "this text answers this question" — correctly;
- checkpoint selection rejected every checkpoint that let that score rise.

A contradiction on 300 of 1,019 rows, resolved in favour of being wrong.

The runtime never faces the trade-off. A retain question is not in the protected store, so
a retain request routes to nothing and creates no Forget-ID whatever the score says.
Measured against `PROTECTED_STORE_RUNTIME.json`:

| | routed against the protected store |
|---|---:|
| protected requests | 711 / 719 |
| retain requests | **0 / 300** |

**Fix.** Two layers, never mixed. `pair_level_metrics` scores answerability with no store
and no population — a correct retain ANSWER is a success there. `store_conditioned_metrics`
routes first and counts a retain false alarm only when a protected Forget-ID actually
fires. The direct-pair retain veto is removed from `selection_metrics` and from
`select_checkpoint`; what remains under `selection_retain_nonanswer_rate` is a reported
diagnostic that constrains nothing.

### 1.3 Splitting was row-level, not concept-level

`half = "train" if int(audit_id[:2], 16) % 2 == 0` with `group = audit_id` let two
questions about one author straddle train and development.

**Fix.** The subject group is the TOFU author block, `item_index // 20`. This is an
assumption about a dataset layout, so it is **checked, not trusted**: on the protected rows
the rule must reproduce `concept_id` exactly, and it does — all 20 blocks map to
`tofu-forget10-author-{block:04d}`, no block spans two concepts, no concept spans two
blocks. Passing there is what licenses the same rule on the 300 retain rows, where
`concept_id` is empty. An independent check agrees: across the 45 retain blocks, no
extracted name span occurs in two blocks. `subject_groups()` refuses to emit groups if the
check fails.

---

## 2. The three artifacts

| file | contains | who may read it |
|---|---|---|
| `PROTECTED_STORE_RUNTIME.json` | protected questions, safe aliases, policy actions, provenance | runtime detector code |
| `DETECTOR_V4_3_CONDITIONING_INDEX.json` | every question, protected and retain, one alias builder | input builders |
| `PROTECTED_STORE_EVAL_KEY.json` | reference answers, item ids, population, strata, subject groups | offline evaluators only |

### 2.1 The runtime store contains no answer and no answer hash

The no-hash rule is separate and deliberate. The forgotten answers here are cities, years,
genres and option keys; that candidate space is small enough to enumerate, so a stored
digest lets its holder confirm the fact the system claims to have destroyed.

Enforcement is an **allowlist** (`RUNTIME_SCOPE_FIELDS`), not a denylist. A denylist has to
anticipate the name of the field that leaks; an allowlist only has to name the fields that
are legitimate, and this protocol fixes those.

### 2.2 The evaluation key has no loader

The separation is structural rather than conventional: `rdl.defenses.protected_store`
contains no code that opens the key's schema, and
`tests/contract/test_detector_v4_3_store_contract.py` asserts no module under
`rdl.defenses` ever mentions it.

### 2.3 Frozen fingerprints

| artifact | sha256 |
|---|---|
| `PROTECTED_STORE_RUNTIME.json` | `aa35f1f421d45053649d1496c21e5dc869c9a441e7480c7a368cbb02f22ca802` |
| `DETECTOR_V4_3_CONDITIONING_INDEX.json` | `99f4f9876c9a089151f770661de109fc88b69e4999da6a8d14007121ecc4719f` |
| `DETECTOR_V4_3_PAIR_BUNDLE.json` (`bundle_sha256`) | `4407130ef217d01029551e165ee88ae0b954d011cb3392e9d8ef4eb643b9766f` |

50 protected scopes across 20 forget ids; 95 distinct conditioning questions; 65 subject
groups (20 protected, 45 retain); 1,019 pairs.

All three digests are over **content**, with build timestamps excluded, so a rebuild that
changes nothing produces the same digest. The trainer records `bundle_sha256` as
provenance, and a check that fired because the clock moved is a check that gets turned off.

The bundle digest above is for the **unlabelled** bundle. Attaching labels with
`--labels` changes it, by design — different labels are different training data — and the
labelled digest is recorded in the training manifest.

---

## 3. The model input

The serialised input is exactly three fields:

```
conditioning_question · subject_aliases · candidate_text
```

The target is `NONE`, `PARTIAL`, or `ANSWER`.

**Never in tokenizer bytes:** `is_protected`, `population`, `forget_id`, `concept_id`,
`subject_id`, dataset split, reference answer, judge identity, detector score, source
stratum.

The field is named `conditioning_question`, not `protected_question`, because 300 of the
1,019 rows ask about authors that were never protected. A field name that asserts something
false about a row is how population re-enters a pipeline that removed it.

---

## 4. The runtime decision

For a request-scoped routed set `R`:

```
S_answer  = max over q in R of P(ANSWER  | q, aliases(q), text)
S_partial = max over q in R of P(PARTIAL | q, aliases(q), text)

fire a Forget-ID          when S_answer  >= tau_answer
retain partial evidence   when S_partial >= tau_partial   (bounded, request-local)
R empty                   -> no protected score, no Forget-ID
```

**The two scores are never combined.** A weighted sum would let two clues outvote a
disclosure, and the two call for different actions: an ANSWER creates a Forget-ID now, a
PARTIAL only feeds a bounded accumulator.

**An empty `R` is not a low score.** `StoreDecision.n_routed_scopes` distinguishes "the
store had nothing to say" from "the store was asked and answered quietly". Code that read
`0.0` as evidence of safety would be reading a number that was never computed.

Separation of concerns, unchanged from v4.2:

- **router** — which protected scope is relevant?
- **DeBERTa** — does this text answer that scope's question?
- **GraphForget** — how does the resulting Forget-ID propagate and get enforced?

---

## 5. Splitting and the shortcut probe

Splits are assigned on the **subject group**, content-addressed
(`hash(group) / weight`, lowest wins), stratified by population so both sides carry
protected and retain subjects in similar proportion. Population is used at this one point,
to assign *groups*; it never reaches a pair, a field or a tokenizer. Content-addressed
rather than seeded so that adding a group later does not reshuffle the groups already
assigned.

Realised on the frozen bundle:

| split | pairs | subjects | protected | retain |
|---|---:|---:|---:|---:|
| train | 467 | 33 | 325 | 142 |
| development | 552 | 32 | 394 | 158 |

Group disjointness is **asserted at build time**, not assumed — a disjointness claim that
is not checked is the claim v4.2 also made.

### 5.1 How the shortcut probe is read

The probe asks whether population is recoverable from the model's own inputs using crude
surface features. The **absolute** number has an irreducible floor: the protected questions
come from `forget10` and the retain questions from `retain90`, and those pools have
different length distributions. A question-conditioned detector is supposed to read the
question, so that much signal cannot be removed without editing the frozen audit.

What bundle construction *chose* is the alias channel. So the pre-registered quantity is
the **excess over the question-only floor**:

| | v4.2 | v4.3 (measured) |
|---|---:|---:|
| question-only floor | 0.699 | 0.699 |
| best alias feature | 1.000 | 0.712 |
| **alias-channel excess** | **+0.301 → +0.50 by construction** | **+0.013** |

**Gate:** alias-channel excess < 0.05. Reported in the bundle manifest and asserted in
`tests/unit/test_detector_v4_3_bundle_and_metrics.py`. This is a bundle-construction gate,
not a detector gate; it says the shortcut is not *available*, not that a trained model
declined to take it. The input ablations in §8 are what test the latter.

---

## 6. Label rubric and the local judge roster

The rubric is v4.2's `BLIND_RUBRIC` and `REFERENCE_RUBRIC`, carried over **unchanged** —
changing the rubric would make v4.2 and v4.3 labels incomparable for no gain.

| judge | repo | quantization | compute dtype | thinking |
|---|---|---|---|---|
| A | `Qwen/Qwen3-14B` | bitsandbytes 8-bit | bfloat16 | off |
| B | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | bitsandbytes 4-bit | bfloat16 | off |

Frozen with the roster: exact commit SHAs (resolved on the box, refused if not a
40-character sha), quantizer, compute dtype, chat template, thinking mode, and generation
parameters (`max_new_tokens=512`, `temperature=0.0`, `top_p=1.0`, `seed=20243`). A
quantization change is a model change — 4-bit and 8-bit of the same weights are different
annotators — so the quantizer is part of the pin, not a runtime flag.

**One model per process**, enforced by `assert_single_model_process`. Two models resident
on one 24 GB card is how a labelling run ends up with fewer labels than rows.

**Malformed is a failure, never a default.** Three attempts, then the row is recorded with
a null label and `malformed: true`. It never becomes a label; a defaulted NONE is
indistinguishable from a judged NONE once written. `--close` refuses to freeze a pass with
any missing or malformed row.

**Blinding is structural.** `blind_prompt()` takes exactly
`(conditioning_question, subject_aliases, candidate_text)` and has no parameter through
which population, stratum or a reference answer could arrive. The reference pass is a
different function with a different rubric, available only after blind labels are frozen.

**Smoke exclusion.** The GPU-1 fit smoke runs on a disjoint 50-row fixture that is
non-reportable and excluded from the 1,019, built by
`rdl graph-detector-v4-3-judge-smoke-fixture`. `--non-reportable --run-id <name>`
namespaces every output file, so a smoke can never occupy a reportable filename. If either
model or configuration changes after that smoke, a dated v4.3 amendment is written and the
reportable labelling run restarts — a replacement is never chosen after inspecting full-run
kappa.

**The smoke fixture has two sources, and this is a real limitation.** The plan asked for 50
rows balanced across the five original strata. That is not satisfiable. The natural bank
holds 120 leaking rows and the 1,019-row audit took 119, so **zero leaking rows remain
disjoint** — a natural-only fixture would contain no likely-ANSWER row at all, and the
smoke could not check the judges on the one class the detector exists to catch. The ANSWER
and PARTIAL rows therefore come from the held-out split of the synthetic relation dataset,
which is a different generator with invented subjects and is disjoint from the natural
audit by construction. The realised fixture is 10 rows from each of: natural
development-clean, natural held-out-clean, natural retain, synthetic ANSWER, synthetic
PARTIAL. Every row records its source. This is a rubric-and-format check, not a sample of
the natural distribution, and no number from it is reportable.

### 6.1 Label gates

- blind kappa ≥ 0.70;
- reference-assisted kappa ≥ 0.70, excluding forced retain rows;
- ≥ 100 ANSWER and ≥ 200 NONE rows;
- ANSWER spans ≥ 2 source strata and ≥ 3 concepts;
- zero unresolved, malformed, missing or provenance-failed rows.

These are **engineering labels** until human validation succeeds.

---

## 7. Thresholds

`tau_answer` and `tau_partial` are selected on the **store-conditioned layer** — the same
routed path the fresh gate scores — and frozen and hashed separately.

`tau_answer` maximises protected ANSWER micro recall subject to the protected-clean FPR
ceiling (0.10) and the end-to-end retain FPR ceiling (0.10). Ties break toward the higher
threshold, so a tie resolves toward firing less. `tau_partial` is chosen afterwards and
independently — never by a weighted combination.

Unchanged from v4.2 unless a pre-training amendment says otherwise: three DeBERTa seeds,
weighted cross-entropy, exact model/tokenizer revisions, checkpoint hashes,
candidate-first token budget, CUDA/device checks.

---

## 8. Ablations and baselines

Reported together, on the same rows:

| variant | what a high score would mean |
|---|---|
| `question_only` | reads question identity, not answerability — must be near chance |
| `aliases_only` | the alias channel still carries population — must be near chance |
| `candidate_only` | answer-shaped text regardless of question — expected above chance, must be materially below full |
| `question_and_candidate` | isolates the alias channel's contribution |
| `full` | the deployed input |
| `dragon_similarity` | DRAGON-shaped additive store similarity — a floor, not a competitor |
| `lexical_floor` | v4's answer-free lexical rule |
| `zero_shot_nli` | pinned NLI cross-encoder, ranking baseline |

**A high full-model score is not accepted if `question_only` or `aliases_only` also scores
highly.** Ablations blank fields rather than deleting them, so every variant hits the same
encoder with the same segment structure — deleting would change the token layout as well as
the information, and the ablation would measure two things at once.

### 8.1 Numeric acceptance criteria

"Near chance" is not a criterion — it is a word that can be applied to 0.55 or to 0.72
depending on how the result looks. These are frozen here, before any v4.3 model exists, as
three-class macro F1 on the development rows (`SHORTCUT_CRITERIA` in
`rdl.eval.detector_v4_3_ablations`):

| criterion | bound | why |
|---|---|---|
| `question_only_macro_f1` | ≤ 0.50 | question identity alone must stay far below the full model; the bound leaves room for the length signal the bundle probe already measured |
| `aliases_only_macro_f1` | ≤ 0.45 | aliases carry subject identity and no relation, so this should sit near the majority-class floor |
| `candidate_only_margin` | ≥ 0.10 | full macro F1 minus candidate-only. Some candidates read like answers regardless of the question; if conditioning adds less than this, "question-conditioned" is decorative |
| `full_minus_best_shortcut` | ≥ 0.10 | the full input must beat every single-channel ablation by a real margin |

The margins are deliberately loose. The claim under test is not "the shortcut carries no
signal" — question length alone carries some — it is "the full input is doing the work". A
shortcut within 0.10 macro F1 of the full model falsifies that whatever its absolute value.

### What is adopted from DRAGON, and what is not

DRAGON keeps an unlearn store without original completions and, for TOFU, combines exact
author-name matching with maximum similarity to stored paraphrased prompts. It is a
**pre-inference prompt detector**.

GraphForget needs something different: protected content appears in agent messages,
memories, summaries, tool outputs and derivations that do not look like the original
prompt. So v4.3 adopts the **answer-free store and modular routing**, and declines the
**additive similarity formula** — summing an alias match and a relation match lets naming
the author substitute for answering the question, which is exactly what v3's alias channel
did on this data. It stays in the table as a baseline so that refusal is a measured choice
rather than an assertion.

---

## 9. Human validation

250 rows: 125 from the original 1,019 and 125 from the fresh engineering audit.

**The draw never reads a detector score.** A sample chosen where the detector is confident
measures the detector's confidence, not its accuracy. `draw_sample()` has no parameter
through which a score could arrive. Judge disagreements (weight 3.0) and rare PARTIAL rows
(weight 3.0) *are* oversampled — those are label properties — and every row records the
weight and inclusion probability it was drawn with, so overall estimates are
design-weighted back to the population.

Raters see the conditioning question, safe aliases, candidate text and rubric. They do not
see model identity, model labels, detector scores, population, source filenames, split or
stratum. Adjudication refuses to run before both rater files are frozen.

### 9.1 Gates

- human–human unweighted kappa ≥ 0.70;
- model-consensus vs human-adjudicated macro F1 ≥ 0.80;
- model-consensus recall on human ANSWER ≥ 0.85;
- recall for NONE, PARTIAL, ANSWER each ≥ 0.75;
- frozen DeBERTa meets its recall/FPR claims on the human-adjudicated subset, 95% CIs
  reported;
- zero unresolved rows and zero provenance failures.

With only one rater this is an **exploratory human audit** and cannot support a
human–human agreement claim. The tooling reports it as such rather than emitting a kappa.

---

## 10. Detector gates on the fresh held-out

- ANSWER micro recall ≥ 0.80;
- macro recall ≥ 0.75;
- correct-concept precision ≥ 0.80;
- protected-clean FPR ≤ 0.10;
- **end-to-end** retain FPR ≤ 0.10, measured after routing against the protected store;
- zero zero-recall concepts;
- zero unresolved labels and zero candidate truncations.

Also reported: PR-AUC, F1/F2, PARTIAL recall, protected-attempt recall, worst-concept
recall, calibration/ECE, per-stratum and per-surface metrics, latency, throughput, peak
VRAM.

---

## 11. Failure policy

- A gate that fails is reported as failed. No gate is re-run with a changed bound.
- A configuration change after a smoke test requires a dated amendment in `DECISIONS.md`
  and restarts the reportable run.
- A malformed or unresolved row is never defaulted to a label.
- v4.1 and v4.2 artifacts are never edited. If v4.3 needs different content, it writes a
  new file under a new name.
- The held-out partition is opened once.

---

## 12. CPU exit gate

All of the following hold before the 3090 is rented:

- [x] v4.3 protocol and `DECISIONS.md` entry committed;
- [x] runtime store, conditioning index and evaluation key schemas implemented and frozen;
- [x] all 1,019 rows join (1019/1019);
- [x] protected and retain input structures identical; alias parity 0.997 vs 0.983;
- [x] no gold answer or answer hash reaches runtime, model or blind judge inputs
      (allowlist + contract tests);
- [x] group splits verified disjoint at build time;
- [x] retain FPR measured through store-conditioned routing, not direct pairs;
- [x] local-judge fake-model smoke, resume, injection, malformed-JSON and single-model
      tests pass;
- [x] shortcut probe alias-channel excess +0.013 (< 0.05);
- [x] every GPU step is reachable from a command: judge pins, env check, disjoint smoke
      fixture, blind and reference passes, label report and authority, labelled-bundle
      rebuild, store-conditioned operating point, held-out gate, human report;
- [x] the whole sequence runs end to end on CPU with no weights and no network —
      `python scripts/v43_pipeline_dryrun.py` exercises 22 behaviours, including every
      refusal (a tag as a pin, a reference pass before the blind freeze, an `--eval-key` on
      a blind pass, selection on the held-out partition, a second held-out opening, a human
      draw without the fresh audit);
- [ ] judge model pins **resolved on a networked box** — the roster ships empty revisions
      and a reportable run refuses them. This is the one gate that cannot close on this
      machine and is the first action on the GPU box.

### 12.1 What is pinned where

The encoder, its tokenizer and the NLI baseline stay with
`rdl graph-detector-v4-2-freeze-model-pins` → `DETECTOR_V4_2_MODEL_PINS.json`. v4.3 does
**not** write a second artifact for the same three repositories: two pin files for one set
of models can disagree, and "which commit trained the checkpoint" would then have two
answers. `rdl graph-detector-v4-3-freeze-judge-pins` pins only the two local judges, and
records the path to the v4.2 artifact alongside.
