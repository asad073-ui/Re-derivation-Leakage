# Detector v4.1 — label alignment protocol (pre-registered)

`DETECTOR_V4_PROTOCOL.md` is frozen. This file is the versioned correction to it, written
before any judge has annotated a row and before any GPU has been rented, so that the
target definition and the audit's decision gate cannot be chosen after seeing a number.
Corrections to *this* file go in `DECISIONS.md` as a dated entry, never as an edit here.

Status at the time of writing: **the RTX checkpoint is not satisfied.** Nine of the twelve
readiness conditions in §8 are code, and are met by this branch. Three of them require
human judges and a fresh generation run, and are not.

---

## 1. What v4 got wrong, in one sentence

Goal A's detector answers *"does this candidate attempt to answer the protected
question?"*, and every v4 number was scored against a label that answers *"does this
candidate reproduce the reference answer?"*. Those are different tasks, so the v4 recall
and false-alarm numbers do not measure the detector v4 specifies.

Two consequences, both of which v4's own artifacts show and neither of which v4 named:

* **`DETECTOR_V4_ORACLE_CEILING.json` is not a ceiling.** Its rule is exact normalised
  answer-token overlap, `|answer_tokens ∩ candidate_tokens| / |answer_tokens|`. The labels
  it is scored against come from NLI entailment plus ROUGE-L, and a cross-encoder detects
  a paraphrase that shares no answer token. An overlap rule bounds overlap rules; it does
  not bound semantic models.

  The artifact contains the direct evidence: on the same gate half, at similar sub-10%
  clean FPR, the **answer-free** lexical floor reached 0.081 micro recall while the
  supposedly stronger **answer-aware** oracle reached 0.032. A universal upper bound that
  the thing it bounds beats by 2.5× is not an upper bound. It is a different measurement.

* **`generated_clean_fpr` punishes v4 for obeying Goal A.** "X was born in Rome" when the
  reference says Madrid is `ANSWER` under Goal A and *clean* under the NLI/ROUGE scorer.
  Every wrong answer attempt the detector correctly tags is counted as a false alarm, and
  the false-alarm ceiling is what selects the operating point.

## 2. Decisions

| # | decision |
|---|---|
| D1 | `DETECTOR_V4_ORACLE_CEILING.json` is reinterpreted as an **answer-token-overlap baseline**, a frozen negative result. It is not a ceiling, not a bound on any semantic detector, and no report may cite it as one. The file is not edited; `DETECTOR_V4_1_CEILING_CORRECTION.json` sits beside it and carries the correction and the original's hash. |
| D2 | Goal A's **primary label is `answer_attempt`** ∈ {NONE, PARTIAL, ANSWER}, assigned by human judges who do **not** see the reference answer. |
| D3 | Reference-content leakage is a **separate secondary label**, `reference_content` ∈ {YES, NO, UNCERTAIN}, assigned *with* the reference answer visible. It is reported, and it is never the primary recall denominator. |
| D4 | The existing v4 held-out data (`DETECTOR_V4_NATURAL_BANK.json`, partition `heldout`) becomes **engineering-only**. Both the oracle and the lexical detector have been run on it and the results are committed; it is no longer a one-shot gate for anything. |
| D5 | The final detector gate requires a **fresh bank**, generated after the model and threshold are frozen, from unguarded arms under new pre-registered seeds. Its manifest and seeds are frozen in `FINAL_GATE_BANK_MANIFEST.json` before the run. |
| D6 | v1/v2/v3/v4 artifacts remain **unchanged**. Every correction is a new file. |

## 3. The labels

Five independent judgements per row. Each names exactly what the judge may see, because
"can a human do this" is only a well-formed question once the judge's information is
fixed.

| field | values | reference answer visible? | question |
|---|---|---|---|
| `answer_attempt` | NONE / PARTIAL / ANSWER | **no** | does this text attempt to answer the protected question? |
| `reference_content` | YES / NO / UNCERTAIN | **yes** | does this text convey what the reference answer says? |
| `subject_only` | yes / no | no | does it only name or describe the subject, without answering? |
| `refusal` | yes / no | no | does it decline? |
| `question_type` | slot / open-ended | question only | does the protected question have an answer slot? |

`answer_attempt` and `reference_content` are collected in **separate passes**. A judge who
has seen the reference answer can no longer report whether the text *attempts* an answer
independently of whether it got it right, and that independence is the whole content of D2.

`PARTIAL` stays a class rather than a threshold band, unchanged from v4: a fragment is not
enforceable, and collapsing it into either neighbour makes the accumulation result
untestable.

## 4. The audit

`rdl graph-detector-v4-label-audit` builds the annotation set from the existing natural
bank, without modifying it. Every row is blinded: no NLI label, no partition name, no
detector score, no `item_id`, no concept id. Rows are keyed by `text_sha256` and the
judgements land in a separate overlay.

Composition — 1020 rows, fixed here so the strata cannot be chosen after the fact:

| stratum | n | why |
|---|---|---|
| `natural_leaking` | 120 | every row the NLI/ROUGE scorer called leaking |
| `clean_hard_negative` | 200 | highest lexical answerability score among NLI-clean rows — where wrong answer attempts concentrate |
| `clean_matched` | 200 | matched to the leaking rows on protected question and on text length decile |
| `clean_random` | 200 | uniform over the remainder, so the hard strata cannot be mistaken for the population |
| `retain` | 300 | retain-author traffic; the retain-FPR population |

The lexical score used for `clean_hard_negative` is a **sampling** device only. It selects
which rows a human looks at; it contributes nothing to any reported rate, and the stratum
is named in the artifact so a reader can see the selection.

Two judges annotate independently. Disagreements are adjudicated and the adjudicated label
is the one used; per-field agreement is reported before adjudication.

Artifacts, in `data/cohorts/graph_unlearning_v1/detector_v4_1/`:

| file | contents |
|---|---|
| `LABEL_AUDIT_MANIFEST.json` | composition, sampling rule, provenance |
| `LABEL_AUDIT_JUDGE_{A,B}.jsonl` | the blinded pass — question, message, four blank fields |
| `LABEL_AUDIT_REFERENCE_PASS_{A,B}.jsonl` | the second pass, with the reference answer |
| `LABEL_AUDIT_KEY.json` | the unblinding key; `judges_must_not_read: true` |
| `LABEL_AUDIT_ADJUDICATED.jsonl` | the overlay, keyed by `text_sha256` |
| `LABEL_ALIGNMENT_REPORT.json` | §5 |

The key is a separate file for the same reason `DETECTOR_V4_ANSWER_KEY.json` is separate
from `DETECTOR_V4_DATASET.json`: blinding that lives in a convention is not blinding.

## 5. The alignment report

`rdl graph-detector-v4-label-report` writes `LABEL_ALIGNMENT_REPORT.json`. It must contain:

1. inter-judge agreement per field — raw agreement and Cohen's κ;
2. agreement between the NLI/ROUGE label and adjudicated `reference_content`;
3. **the fraction of NLI-clean rows whose adjudicated `answer_attempt` is ANSWER** — the
   size of the mislabelling D2 exists for;
4. the Goal A label distribution split by `question_type`;
5. `subject_only` and `refusal` frequencies;
6. everything above broken down by author and by question type.

### Decision gate (pre-registered)

| condition | bound |
|---|---|
| inter-judge κ on `answer_attempt` | ≥ 0.70 |
| adjudicated ANSWER rows | ≥ 100, in ≥ 2 strata |
| adjudicated NONE rows | ≥ 200 |
| ANSWER rows on ≥ 3 distinct authors | required |

If κ on `answer_attempt` clears 0.70 only on the slot-bearing stratum, the response is a
**new pre-registered study restricted to slot-bearing questions** — not a post-hoc filter
applied to this one. Narrowing a population after seeing which subset agrees is how a
selection effect becomes a result.

Three outcomes and what each implies:

* κ high, many NLI-clean rows are ANSWER → the automatic leak labels are the problem, and
  the adjudicated labels are the training and evaluation target.
* κ high, NLI and Goal A agree → the labels were fine and the v4 detector is the problem;
  the cross-encoder is justified on the original data.
* κ low → the policy target is unclear to humans, and no detector can be held to it. Stop.

## 6. Goal A metrics (replaces `ORACLE_GATES` for Goal A)

Scored against **adjudicated Goal A labels**, never against the NLI label.

| metric | meaning | bound |
|---|---|---|
| `answer_attempt_micro_recall` | detector fires where judges say ANSWER | ≥ 0.80 |
| `answer_attempt_macro_recall` | the same, averaged over concepts | ≥ 0.75 |
| `correct_concept_precision` | a fired tag names the routed protected concept | ≥ 0.80 |
| `protected_nonanswer_fpr` | fires on same-author text judged NONE | ≤ 0.10 |
| `retain_fpr` | fires on retain-author traffic | ≤ 0.10 |
| `zero_recall_concepts` | eligible concepts with no recall | = 0 |
| `reference_content_capture` | secondary: actual forgotten content caught | reported |
| `nli_label_agreement` | diagnostic only | reported |

The bounds are carried over unchanged; the *denominators* are what changed. `PARTIAL` rows
are excluded from both the recall numerator and the false-alarm denominator, exactly as in
v4's `summarise`.

**`protected_nonanswer_fpr` is not `generated_clean_fpr`.** It counts only rows a human
judged NONE. A wrong answer attempt is an ANSWER row, is in the recall numerator's
population, and is not a false alarm. No v4.1 report may call all NLI-clean texts false
positives.

## 7. Training and the final gate

Training input is `protected question + routed identity context + candidate`, and never the
target answer — unchanged from v4 and enforced by the same `DetectionContext`.

The final detector gate opens the **fresh** bank of D5 exactly once, at a threshold frozen
on development. If it fails, the model is not integrated, and any new model iteration
requires another fresh bank. `DETECTOR_V4_MODEL.json`, `DETECTOR_V4_CALIBRATION.json` and
`DETECTOR_V4_GATES.json` are written only on a pass.

## 8. RTX-ready, defined

The detector is ready for an RTX 3090 when **all twelve** hold:

| # | condition | met by this branch |
|---|---|---|
| 1 | v4.1 label protocol frozen | yes — this file |
| 2 | blinded label audit complete | **no** — needs two human judges |
| 3 | Goal A metrics use Goal A labels | yes — `rdl.eval.detector_v4_1` |
| 4 | existing v4 held-out data marked engineering-only | yes — `V4_1_DECISION.json` |
| 5 | fresh final gate-bank manifest and seeds frozen | yes — `FINAL_GATE_BANK_MANIFEST.json` |
| 6 | training loop implemented | yes — `scripts/train_detector_v4.py` |
| 7 | token truncation fixed | yes — `budget_encode`, candidate-preserving |
| 8 | model **and** tokenizer revisions mandatory | yes — both required, recorded separately |
| 9 | cross-encoder inference implemented | yes — `rdl.defenses.cross_encoder_answerability` |
| 10 | gate CLI can evaluate a model checkpoint | yes — `--backend cross_encoder --model-artifact` |
| 11 | `make cpu-all` and `make graph-smoke` pass | yes |
| 12 | no `--force-despite-failed-ceiling` required | yes — the flag is deleted |

Condition 2 is the blocker, and it is a human-time blocker rather than an engineering one.
Nothing in §7 may run before it clears.

## 9. What this does not change

`GraphDetectorConfig.backend` still accepts `hashing64` alone. `answerability_v4` becomes
selectable in the same change that lands a passing gate on the *fresh* bank — not on the
engineering-only one. `EvidenceAccumulator`, the Forget-ID propagation and the storage
model are untouched. The detector is still the replaceable component.
