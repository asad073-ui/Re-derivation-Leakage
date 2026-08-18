"""The Typer app that aggregates every CLI."""

from __future__ import annotations

import typer

from .audit_mechanism import audit_mechanism
from .bundle_graph import bundle_graph
from .calibrate_detector import calibrate_detector
from .detector_corpus import detector_corpus
from .detector_gates import detector_gates
from .detector_recall import detector_recall
from .detector_v4_1_freeze import detector_v4_1_freeze
from .detector_v4_2_bank_audit import detector_v4_2_bank_audit
from .detector_v4_2_bank_runs import detector_v4_2_plan_bank_runs
from .detector_v4_2_banks import (
    detector_v4_2_build_bank,
    detector_v4_2_final_gate,
    detector_v4_2_freeze_banks,
)
from .detector_v4_2_llm_judge import detector_v4_2_llm_judge
from .detector_v4_2_model_pins import detector_v4_2_freeze_model_pins
from .detector_v4_2_operating_point import detector_v4_2_select_operating_point
from .detector_v4_2_plan import detector_v4_2_judge_plan
from .detector_v4_2_report import detector_v4_2_label_report
from .detector_v4_3_bundle import detector_v4_3_bundle
from .detector_v4_3_gate import (
    detector_v4_3_final_gate,
    detector_v4_3_judge_smoke_fixture,
    detector_v4_3_select_operating_point,
)
from .detector_v4_3_human import (
    detector_v4_3_human_adjudicate,
    detector_v4_3_human_import,
    detector_v4_3_human_report,
    detector_v4_3_human_sample,
)
from .detector_v4_3_local_judge import detector_v4_3_local_judge
from .detector_v4_3_pins import detector_v4_3_env_check, detector_v4_3_freeze_judge_pins
from .detector_v4_3_report import detector_v4_3_label_report
from .detector_v4_3_store import detector_v4_3_build_store
from .detector_v4_4_bundle import (
    detector_v4_4_bundle,
    detector_v4_4_judge_smoke_fixture,
    detector_v4_4_panel,
    detector_v4_4_supplement,
)
from .detector_v4_4_gate import (
    detector_v4_4_audit_sample,
    detector_v4_4_final_gate,
    detector_v4_4_select_operating_point,
)
from .detector_v4_4_human import (
    detector_v4_4_human_adjudicate,
    detector_v4_4_human_import,
    detector_v4_4_human_pilot,
    detector_v4_4_human_reference_pass,
    detector_v4_4_human_sample,
)
from .detector_v4_4_judge import detector_v4_4_env_check, detector_v4_4_local_judge
from .detector_v4_4_report import detector_v4_4_label_report
from .detector_v4_data import detector_v4_build_data
from .detector_v4_gates import detector_v4_gates
from .detector_v4_label_audit import detector_v4_label_audit, detector_v4_label_report
from .detector_v4_oracle import detector_v4_oracle
from .discover_checkpoints import discover_checkpoints
from .env_check import env_check
from .finalize_graph import finalize_graph
from .freeze_graph_cohort import freeze_graph_cohort
from .make_leak_report import make_leak_report
from .make_report import make_report
from .plan_graph_run import plan_graph_run
from .report_graph import report_graph
from .rescore_day2 import rescore_day2
from .rescore_leak import rescore_leak
from .run_condition import run_condition
from .run_graph import run_graph
from .run_leak import run_leak
from .run_repro import run_repro
from .score_graph import score_graph

app = typer.Typer(
    name="rdl",
    help="Re-derivation Leakage — memory-mediated recovery of unlearned knowledge.",
    no_args_is_help=True,
    add_completion=False,
)

app.command("env-check")(env_check)
app.command("discover-checkpoints")(discover_checkpoints)
app.command("run-repro")(run_repro)
app.command("run-condition")(run_condition)
app.command("make-report")(make_report)
app.command("rescore-day2")(rescore_day2)
app.command("run-leak")(run_leak)
app.command("make-leak-report")(make_leak_report)
app.command("rescore-leak")(rescore_leak)

# graph-unlearning-v1. A separate experiment family: the commands above are frozen
# evidence for the v5 conditions and the two-agent Leak@k work.
app.command("graph-plan")(plan_graph_run)
app.command("graph-run")(run_graph)
app.command("graph-score")(score_graph)
app.command("graph-report")(report_graph)
app.command("graph-finalize")(finalize_graph)
app.command("graph-freeze-cohort")(freeze_graph_cohort)
app.command("graph-calibrate")(calibrate_detector)
# Reanalysis phases (GU-0030, GU-0031). All three read committed evidence and generate
# nothing: recall diagnoses the detection gap, the corpus freezes what a fix may be
# fitted on, and the bundle links a study's runs into one verdict.
app.command("graph-detector-recall")(detector_recall)
app.command("graph-audit-mechanism")(audit_mechanism)
app.command("graph-detector-corpus")(detector_corpus)
# The CPU go/no-go before an instance is rented (GU-0032). Exits non-zero on a
# failing gate, because a detector that cannot see the leakage makes the
# graph-versus-node-local comparison unable to mean anything.
app.command("graph-detector-gates")(detector_gates)
app.command("graph-bundle")(bundle_graph)

# Detector v4 / v4.1 (answerability). CPU commands, run in this order, none of which
# generates anything or touches a GPU:
#
#   build-data    freezes the synthetic answerability corpus, its offline answer key, and
#                 the NATURAL clean/leaking bank collected from the natural graph_flow run
#   oracle        FROZEN, and reinterpreted by GU-0037: it measures an answer-token-OVERLAP
#                 baseline, not a universal ceiling. Kept as a negative result; it no
#                 longer gates anything
#   v4-1-freeze   writes that reinterpretation, marks the v4 held-out data engineering-only
#                 and pre-registers the fresh final gate bank
#   label-audit   builds the BLINDED human annotation set — Goal A's actual label source
#   label-report  adjudicates the two judges and writes the alignment report. This is the
#                 gate the trainer refuses to start without
#   gates         the detector at an operating point chosen on development only, against
#                 the adjudicated Goal A labels. `--backend lexical|cross_encoder`
#
# `answerability_v4` is deliberately NOT a value of GraphDetectorConfig.backend yet. A
# detector reaches a study after its gate has been opened and passed, and v3's scorer is
# the reason that is a rule rather than a habit.
app.command("graph-detector-v4-build-data")(detector_v4_build_data)
app.command("graph-detector-v4-oracle")(detector_v4_oracle)
app.command("graph-detector-v4-1-freeze")(detector_v4_1_freeze)
app.command("graph-detector-v4-label-audit")(detector_v4_label_audit)
app.command("graph-detector-v4-label-report")(detector_v4_label_report)
app.command("graph-detector-v4-gates")(detector_v4_gates)

# v4.2 — the same audit, annotated by two model judges instead of two humans. The
# commands are named apart from v4.1's for the reason DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md
# E4 gives: LABEL_ALIGNMENT_REPORT.json is the human report, and an artifact whose schema
# says "human judges" must never be written from an LLM's output.
#
#   llm-judge     one judge, one pass, stateless per row. --dry-run calls nothing.
#   label-report  adjudicates the two judges and writes the MODEL label report, which
#                 carries human_grounded=false and publication_label_valid=false.
#   judge-plan    how many calls, how many tokens, how many free-tier days. Calls nothing.
# The exact encoder, tokenizer and baseline COMMITS a reportable run may use. The
# trainer required a non-empty revision and accepted whatever was typed, which pins
# the shape of the claim rather than the claim.
app.command("graph-detector-v4-2-freeze-model-pins")(detector_v4_2_freeze_model_pins)
app.command("graph-detector-v4-2-judge-plan")(detector_v4_2_judge_plan)
app.command("graph-detector-v4-2-llm-judge")(detector_v4_2_llm_judge)
app.command("graph-detector-v4-2-label-report")(detector_v4_2_label_report)

# The banks. FINAL_GATE_BANK_MANIFEST.json froze the final bank's seeds and gates but not
# its generation budget, and a bank whose size is decided at generation time can be grown
# until a gate passes. These close that gap without editing the frozen manifest.
#
#   freeze-banks  the engineering bank's manifest, and the budget the final one lacks
#   build-bank    assemble a bank, REFUSING runs that do not match the pre-registration
#   bank-audit    freeze a stratified audit sample of a bank and write the judge inputs
#   select-operating-point
#                 choose the threshold on the DEVELOPMENT partition and freeze it. The
#                 threshold used to be a `--threshold` float on the gate, which is the
#                 protocol's one number supplied by whoever ran the command.
#   final-gate    SCORE a bank once, at that frozen threshold, and record that it was
#                 opened. Refuses labels that never passed the judge gate.
app.command("graph-detector-v4-2-freeze-banks")(detector_v4_2_freeze_banks)
app.command("graph-detector-v4-2-build-bank")(detector_v4_2_build_bank)
app.command("graph-detector-v4-2-plan-bank-runs")(detector_v4_2_plan_bank_runs)
app.command("graph-detector-v4-2-bank-audit")(detector_v4_2_bank_audit)
app.command("graph-detector-v4-2-select-operating-point")(detector_v4_2_select_operating_point)
app.command("graph-detector-v4-2-final-gate")(detector_v4_2_final_gate)

# v4.3 — the protected store, and the artifacts that keep the model's input answer-free
# and population-blind. Additive: every v4.1 and v4.2 file stays byte-for-byte as frozen,
# because their hashes are quoted in DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md and editing one
# would invalidate a pre-registration rather than correct it.
#
#   build-store  PROTECTED_STORE_RUNTIME.json (answer-free, runtime-loadable),
#                DETECTOR_V4_3_CONDITIONING_INDEX.json (answer-free AND population-free)
#                and PROTECTED_STORE_EVAL_KEY.json (sealed; no defenses module reads it)
#   bundle       the derived 1,019-row pair bundle: one alias builder for both
#                populations, group-disjoint splits, and a shortcut probe that measures
#                whether population is still recoverable from the model's own inputs
app.command("graph-detector-v4-3-build-store")(detector_v4_3_build_store)
app.command("graph-detector-v4-3-bundle")(detector_v4_3_bundle)
#   local-judge  one open-weight judge, one process, one pass. --fake-model exercises
#                resume, malformed handling and injection framing without weights.
app.command("graph-detector-v4-3-local-judge")(detector_v4_3_local_judge)
#   freeze-judge-pins  the two judges' exact commit shas. The ENCODER pins stay with
#                `graph-detector-v4-2-freeze-model-pins`: a second artifact for the same
#                three repositories could disagree with the first, and "which commit
#                trained the checkpoint" would then have two answers.
#   env-check    can this box run the reportable phase, checked before weights download
#   smoke-fixture  50 rows DISJOINT from the reportable 1,019, for the GPU-1 fit smoke
#   label-report the missing middle: agreement, gates, blind disagreements, adjudication,
#                and the authority artifact the trainer refuses to run without
#   select-operating-point / final-gate
#                the callers for rdl.eval.detector_v4_3, which shipped untested against a
#                real artifact because nothing invoked it. Both go through one scoring
#                path, so the threshold is chosen under the routing the gate applies.
app.command("graph-detector-v4-3-freeze-judge-pins")(detector_v4_3_freeze_judge_pins)
app.command("graph-detector-v4-3-env-check")(detector_v4_3_env_check)
app.command("graph-detector-v4-3-judge-smoke-fixture")(detector_v4_3_judge_smoke_fixture)
app.command("graph-detector-v4-3-label-report")(detector_v4_3_label_report)
app.command("graph-detector-v4-3-select-operating-point")(detector_v4_3_select_operating_point)
app.command("graph-detector-v4-3-final-gate")(detector_v4_3_final_gate)
#   human-*      the 250-row validation. The draw never reads a detector score, and
#                adjudication refuses to run before both rater files are frozen.
app.command("graph-detector-v4-3-human-sample")(detector_v4_3_human_sample)
app.command("graph-detector-v4-3-human-import")(detector_v4_3_human_import)
app.command("graph-detector-v4-3-human-adjudicate")(detector_v4_3_human_adjudicate)
app.command("graph-detector-v4-3-human-report")(detector_v4_3_human_report)

# v4.4 — the answer-attempt repair. Additive again: every v4.3 artifact stays byte-for-byte,
# including the two failed blind passes, because a failed experiment whose evidence was
# edited afterwards stops being evidence. See GU-0049 and
# docs/graph_unlearning/DETECTOR_V4_4_ANSWER_ATTEMPT_PROTOCOL.md.
#
# The target did NOT move. It is still "does this candidate attempt to answer the routed
# protected question", correctness still lives on the separate reference axis, and the
# Goal-A false-alarm denominator was always rows judged NONE. What moved is the data (the
# audit contained almost no non-attempts), the rubric (PARTIAL/ANSWER was underdefined on
# open-ended questions), the distribution kappa is gated on, and four metric names that
# said "clean" when they meant "did not attempt".
#
#   supplement   ~714 non-attempt and partial rows, composed from independent slot banks
#                and refused if the pool fails its own diversity bounds
#   bundle       the ~1,733-row v4.4 bundle, on v4.3's conditioning index and split rule
#   panel        the frozen 600-row balanced calibration panel, 200 per intended class
#   judge-smoke-fixture  60 rows disjoint from the bundle, 20 per intended class
app.command("graph-detector-v4-4-supplement")(detector_v4_4_supplement)
app.command("graph-detector-v4-4-bundle")(detector_v4_4_bundle)
app.command("graph-detector-v4-4-panel")(detector_v4_4_panel)
app.command("graph-detector-v4-4-judge-smoke-fixture")(detector_v4_4_judge_smoke_fixture)
#   local-judge  the same two pinned models and the same one-model-per-process rule, under
#                the hierarchical prompt. An object whose answer_attempt contradicts its own
#                addresses_question/standalone_answer is MALFORMED, never repaired.
#   env-check    v4.3's hardware checks plus the v4.4 artifacts, including that the frozen
#                panel names the bundle actually on disk
app.command("graph-detector-v4-4-local-judge")(detector_v4_4_local_judge)
app.command("graph-detector-v4-4-env-check")(detector_v4_4_env_check)
#   label-report the balanced-panel gate is PRIMARY; the full-mixture kappa is reported
#                beside it as a distribution-dependent diagnostic. The kappa BOUND did not
#                move -- it is still 0.70 -- only the distribution it is evaluated on.
app.command("graph-detector-v4-4-label-report")(detector_v4_4_label_report)
#   audit-sample the score-independent fresh-bank sampler. v4.2 enriched by the NLI/ROUGE
#                `protected_clean` stratum and v4.3 measured that 336 of its 600 rows are
#                answer attempts, so it could never supply the non-attempt denominator.
#   select-operating-point / final-gate
#                the same bounds as v4.3 under names that say what they measure:
#                protected_nonattempt_fpr, not protected_clean_fpr.
app.command("graph-detector-v4-4-audit-sample")(detector_v4_4_audit_sample)
app.command("graph-detector-v4-4-select-operating-point")(detector_v4_4_select_operating_point)
app.command("graph-detector-v4-4-final-gate")(detector_v4_4_final_gate)
#   human-*      two ORDERED passes. The reference pass refuses to open until every blind
#                rater file is complete, because a rater who has seen the answer can no
#                longer report whether the text attempted one.
#   human-pilot  30-50 rows, non-reportable, before the prompt and panel are frozen.
app.command("graph-detector-v4-4-human-pilot")(detector_v4_4_human_pilot)
app.command("graph-detector-v4-4-human-sample")(detector_v4_4_human_sample)
app.command("graph-detector-v4-4-human-reference-pass")(detector_v4_4_human_reference_pass)
app.command("graph-detector-v4-4-human-import")(detector_v4_4_human_import)
app.command("graph-detector-v4-4-human-adjudicate")(detector_v4_4_human_adjudicate)


if __name__ == "__main__":
    app()
