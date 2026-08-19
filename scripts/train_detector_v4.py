#!/usr/bin/env python
"""Detector-v4.2 **Phase 7**: fine-tune the answerability cross-encoder on an RTX 3090.

What changed from the v4.1 version of this file
-----------------------------------------------
v4.1 added a real loop, a candidate-preserving encoder and a pinned baseline. Reading it
against what the GPU would actually do turned up six defects, all of which would have
produced numbers rather than errors. They are fixed here and each is named in
``DECISIONS.md`` GU-0038.

**The final accumulation was discarded.** The optimizer stepped on
``step % gradient_accumulation_steps == 0``, so an epoch whose batch count is not a
multiple of the accumulation factor threw away its last partial accumulation — silently,
every epoch. ``steps_per_epoch`` also used floor division, so the LR schedule was built
for a different number of steps than the loop takes. Both use the tail-inclusive form now.

**...and then the recovered tail was under-scaled** (GU-0040). Taking the step was only
half the fix: every batch's loss was still divided by the full accumulation factor, so a
one-batch remainder stepped on a gradient of half — or a third — the intended size. Each
batch is now divided by the size of *its own* window, computed by
:func:`accumulation_windows`, which is pure so the arithmetic can be tested without a GPU.
That it could not be is why the bug survived two revisions.

**The NLI entailment index was hard-coded.** ``BASELINE_ENTAILMENT_INDEX = 1``, with a
comment claiming it was confirmed on the box. A moved checkpoint, or a different NLI repo,
silently turns the baseline into a measurement of the contradiction head. It is resolved
from ``model.config.label2id`` / ``id2label``, raises when the entailment class is
ambiguous, and records what it resolved.

**Training and serving saw different inputs.** ``natural_examples`` set ``aliases: []``
while runtime inference receives routed identity aliases — training-serving skew in the
one field routing contributes. The natural rows are joined to the offline concept registry
and carry the same permitted aliases the runtime will pass.

**A checkpoint could not be verified — and then was not.** The manifest recorded a path.
It records SHA-256 of the weights, the tokenizer files, the config and the label map; and
from v4.2.2 those hashes are *re-computed* rather than trusted, by the shared
``rdl.defenses.checkpoint_digest``: here, from disk, after training, and again inside
``CrossEncoderAnswerabilityDetector.from_artifact`` before the weights are loaded. A hash
written once and never checked is a comment.

**The preregistration was a set of defaults.** ``--seeds``, ``--epochs`` and the model pins
were flags whose defaults happened to be the preregistered values, so a run that left the
plan wrote a manifest recording its own settings beside a selection rule describing the
plan's. :func:`enforce_preregistration` refuses that for a reportable run;
``--non-reportable`` marks the run instead of skipping the check, and every downstream gate
refuses the checkpoint it selects. Every ``--extra-train`` file is hashed either way.

**Selection was not the objective, and was not frozen.** The trainer picked the best epoch
on general macro F1 while the stated objective is ANSWER-recall under two false-alarm
ceilings. Three fixed seeds are trained, all are reported, and the checkpoint is chosen by
:data:`CHECKPOINT_SELECTION_RULE` — frozen here, before training, and recorded verbatim.

**There was no way to find a GPU-only error cheaply.** ``--smoke`` runs 64 rows for one
epoch, asserts the model and the batches are actually on CUDA, forces at least one
optimizer step, reloads the checkpoint and exercises batched inference. It writes an
artifact marked ``reportable: false`` and is the first thing to run on a rented box.

The readiness gate, and what authorises a run
---------------------------------------------
v4.1's gate is ``LABEL_ALIGNMENT_REPORT.json`` — the audit by two **human** judges. On a
solo project that is a human-time blocker, so v4.2 adds a second acceptable authority:
``DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json``, the same audit annotated by two
model judges. It authorises **engineering training only**, it is recorded in the manifest
by name, and it can never make a result publication-ready. There is still no override
flag: if the gate is wrong, change the gate in the protocol and say so in DECISIONS.md.

Order of operations on the box
------------------------------
1. ``--smoke`` — a non-reportable CUDA check. Minutes, not hours.
2. A passing label report — human (v4.1) or model-judge (v4.2).
3. ``--baseline-only`` — record the pinned zero-shot cross-encoder.
4. Fine-tune the three preregistered seeds.
5. Select the operating point on development (``rdl graph-detector-v4-gates --backend
   cross_encoder --device cuda --label-source model``).
6. Generate the engineering bank, audit it, freeze its operating point with
   ``rdl graph-detector-v4-2-select-operating-point``, then open the held-out partition
   once with ``rdl graph-detector-v4-2-final-gate``. The FINAL bank stays sealed.

Never tune on the v4 held-out data — it is engineering-only — and never on either bank.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

V4 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4"
V4_1 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_1"
V4_2 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_2"
V4_3 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_3"
V4_4 = REPO / "data" / "cohorts" / "graph_unlearning_v1" / "detector_v4_4"
LABELS = ("NONE", "PARTIAL", "ANSWER")
ANSWER_INDEX = LABELS.index("ANSWER")

# Three fixed seeds, frozen before training. All three are reported; one checkpoint is
# selected across all (seed, epoch) pairs by the rule below. Reporting one seed and
# calling it the result would make the number a draw from a distribution nobody showed.
PREREGISTERED_SEEDS = (20260814, 20260815, 20260816)

# Frozen before training. Recorded verbatim in the manifest so a reader can check that the
# checkpoint was chosen by this rule rather than by whichever number came out best.
CHECKPOINT_SELECTION_RULE = (
    "Across all (seed, epoch) pairs, select the checkpoint with the highest development "
    "protected ANSWER-recall subject to development protected-clean FPR <= 0.10, where "
    "recall and the FPR are read at the best EXACT score breakpoint on the development "
    "pool. Ties break toward the lower threshold, then the earlier epoch, then the smaller "
    "seed. Macro F1 is recorded as a diagnostic and selects nothing. The end-to-end retain "
    "false-alarm rate is a SEPARATE, ROUTED measurement and is not a ceiling here: a "
    "retain request does not route against the protected store, so no threshold on a "
    "direct retain pair can create or prevent a Forget-ID (v4.3, GU-0042). This is a "
    "clause-level proxy for Goal A and is NOT the operating point: the reportable "
    "threshold is chosen on ROUTED rows through the store-conditioned path."
)
# Why the protected-clean pool is still its own ceiling, and why the retain pool is not a
# ceiling here at all.
#
# The downstream gate names protected_nonanswer_fpr and retain_fpr separately, and pooling
# them hides a retain failure: the retain pool is the smaller one (2 example classes of 6
# in the synthetic pool, ~300 of ~1,019 rows in the natural one), so an aggregate of 0.10
# is satisfiable with a retain rate far above it. Firing on retain traffic is the utility
# cost the whole defence is measured against, so it must never be averaged away.
#
# v4.3 changed WHERE that rate is measured, not whether it matters. Measuring it on a
# direct (retain question, retain candidate) pair asks a question the runtime never asks:
# the retain question is not in the protected store, so the request routes to nothing and
# no Forget-ID exists to be wrong about. Worse, most retain candidates DO answer their
# retain question, so the direct rate punished the model for being correct — the same rows
# cross-entropy was training upward. The rate now comes from
# `rdl.eval.detector_v4_3.store_conditioned_metrics`, which routes against
# PROTECTED_STORE_RUNTIME.json first and counts a false alarm only when a protected
# Forget-ID actually fires.
DEV_FPR_CEILING = 0.10

# A pinned pretrained NLI cross-encoder. Its head is TRAINED, so a zero-shot number from it
# is a statement about what pretraining already knows; deberta-v3-base's fresh head is not.
DEFAULT_BASELINE_REPO = "cross-encoder/nli-deberta-v3-base"
# No hard-coded index. The entailment class is resolved from the checkpoint's own label
# map; see resolve_entailment_index.
ENTAILMENT_ALIASES = ("entailment", "entail", "label_0_entailment")


@dataclass(frozen=True)
class TrainingPins:
    """Everything that changes the checkpoint. All of it lands in DETECTOR_V4_MODEL.json."""

    model_repo_id: str = "microsoft/deberta-v3-base"
    # No default for either revision, deliberately. A revision is a claim about which
    # weights and which subword split were used; a guessed one is worse than an absent one.
    # Resolve on the box with `huggingface_hub.HfApi().model_info(repo).sha`.
    model_revision: str = ""
    tokenizer_revision: str = ""
    max_length: int = 256
    min_question_tokens: int = 8
    max_aliases: int = 4
    seeds: tuple[int, ...] = PREREGISTERED_SEEDS
    optimizer: str = "adamw_torch"
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    max_grad_norm: float = 1.0
    fp16: bool = True
    class_weighting: str = "inverse_frequency"
    checkpoint_selection: str = CHECKPOINT_SELECTION_RULE
    expected_device: str = "cuda"

    def to_dict(self) -> dict:
        out = asdict(self)
        out["seeds"] = list(self.seeds)
        return out


def require_cuda(device: str) -> str:
    try:
        import torch
    except ImportError:
        raise SystemExit("torch is not installed; this script runs on the GPU box only") from None
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "no CUDA device. This is the RTX phase; the CPU phase is "
            "`rdl graph-detector-v4-2-llm-judge` and `rdl graph-detector-v4-2-label-report`."
        )
    return str(torch.cuda.get_device_name(0)) if device.startswith("cuda") else "cpu"


# ------------------------------------------------------------------- the label gate --

HUMAN_REPORT = "LABEL_ALIGNMENT_REPORT.json"
MODEL_REPORT = "DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json"


def require_v4_3_label_authority(report_path: Path, bundle_path: Path) -> dict:
    """The v4.3 authority: a passing label report, bound to the bundle it labelled.

    A third acceptable authority beside v4.1's human report and v4.2's hosted-model report,
    and it is not equivalent to either. Like v4.2's it authorises **engineering work only**
    -- two open-weight judges agreeing is consistency evidence, not correctness -- and it
    refuses a report whose flags have been edited to claim otherwise.

    The binding to the bundle is the part v4.2 could not do. A label report names the
    ``bundle_sha256`` it was computed over, and a training run passes both; if they differ,
    the labels describe different rows than the ones about to be trained on, and the
    manifest would record an authority for an audit that never happened.
    """
    if not report_path.exists():
        raise SystemExit(
            f"{report_path} is absent. A v4.3 run is authorised by "
            "DETECTOR_V4_3_LABEL_REPORT.json, which `rdl graph-detector-v4-3-label-report` "
            "writes once both judges' passes are closed and adjudicated."
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if str(report.get("schema")) != "graph-detector-v4-3-label-authority-v1":
        raise SystemExit(f"{report_path} carries schema {report.get('schema')!r}")
    if not report.get("gates_passed"):
        raise SystemExit(
            f"{report_path} records gate failures: {report.get('gate_failures')}. Refusing "
            "to train. A failing gate is reported as failed; it is not re-run with a "
            "moved bound."
        )
    for flag in ("human_grounded", "publication_label_valid"):
        if report.get(flag) is not False:
            raise SystemExit(
                f"{report_path} claims {flag}={report.get(flag)!r}. The v4.3 schema is "
                "open-weight model-judge output and that flag is false by construction; "
                "this file has been edited. Refusing to train on it."
            )
    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    # The labelled bundle records which UNLABELLED freeze it was derived from, so the
    # authority binds to that rather than to the labelled file's own digest, which
    # necessarily differs once labels are attached.
    labelled_from = str((bundle.get("labels") or {}).get("authority_bundle_sha256") or "")
    expected = str(report.get("bundle_sha256") or "")
    if labelled_from and expected and labelled_from != expected:
        raise SystemExit(
            f"{report_path} was computed over bundle {expected[:16]}... but "
            f"{bundle_path} was labelled from {labelled_from[:16]}.... The authority "
            "describes different rows than the ones about to be trained on."
        )
    return {
        "report": str(report_path),
        "judge_population": str(report.get("annotator_population")),
        "human_grounded": False,
        "publication_label_valid": False,
        "authorises": "ENGINEERING training and evaluation only",
        "adjudicated_file": report.get("adjudicated_file"),
        "adjudicated_sha256": report.get("adjudicated_sha256"),
        "bundle_sha256": expected,
        "blind_kappa": (report.get("blind") or {}).get("kappa"),
        "reference_kappa": (report.get("reference") or {}).get("kappa_excluding_forced"),
        "protocol_phase": "v4.3",
        "warning": (
            "This run is authorised by two LOCAL OPEN-WEIGHT model judges. No result from "
            "it may be described as a Goal A result or as publication-ready until the "
            "250-row human validation passes. See "
            "DETECTOR_V4_3_PROTECTED_STORE_PROTOCOL.md sections 6 and 9."
        ),
    }


def require_v4_4_label_authority(authority_path: Path, bundle_path: Path) -> dict:
    """The v4.4 authority: a green label authority bound to the IMMUTABLE bundle.

    Different from v4.3's in one structural way that matters more than any of the flag
    checks below. v4.3 trained from a *labelled* bundle -- ``graph-detector-v4-3-bundle
    --labels ...`` wrote the labels back into the pairs -- so the authority could only bind
    to the unlabelled freeze the labelled file *claimed* to descend from.

    v4.4 cannot do that. Its labels live inside ``pairs``, ``bundle_sha256`` covers
    ``pairs``, and the 600-row calibration panel is frozen against that hash. Writing labels
    into the bundle would move the hash the panel is bound to, and the primary kappa gate
    would then be evidence about a file that no longer exists. So the frozen bundle is never
    rewritten: the authority binds to its hash directly, and :func:`v4_4_bundle_examples`
    joins ``V4_4_ADJUDICATED.jsonl`` to it in memory, by ``audit_id``.

    That makes the join itself the thing to check. A join is silent when it half-fails --
    a missing id is a dropped row, a duplicate id is a row whose label depends on file
    order, an extra id is a label for a row that is not in this bundle -- so all three are
    refused here rather than counted later.
    """
    if not authority_path.exists():
        raise SystemExit(
            f"{authority_path} is absent. A v4.4 run is authorised by "
            "DETECTOR_V4_4_LABEL_AUTHORITY.json, which "
            "`rdl graph-detector-v4-4-label-report --adjudication ... "
            "--reference-adjudication ...` writes once BOTH axes are closed."
        )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    if str(authority.get("schema")) != "graph-detector-v4-4-label-authority-v1":
        raise SystemExit(
            f"{authority_path} carries schema {authority.get('schema')!r}, not "
            "'graph-detector-v4-4-label-authority-v1'. A v4.3 authority describes a "
            "different bundle under a different prompt version; it is not an earlier "
            "version of this file."
        )
    if not authority.get("green"):
        raise SystemExit(
            f"{authority_path} is not green. Refusing to train. A failing authority is "
            "reported as failing; it is not re-run with a moved bound."
        )
    if not (authority.get("panel_gate") or {}).get("passed"):
        raise SystemExit(
            f"{authority_path} records a failed balanced-panel gate. The 200/200/200 panel "
            "is the primary agreement gate and adjudication on top of a failed panel hides "
            "the failure behind a resolved file."
        )
    if not authority.get("bundle_minima_passed"):
        raise SystemExit(
            f"{authority_path} records failed bundle class minima. The non-attempt "
            "supplement exists precisely to meet them; a bundle that misses them is v4.3 "
            "again, with too few true NONE rows to measure a non-attempt denominator."
        )
    if str(authority.get("trainable_field")) != "answer_attempt":
        raise SystemExit(
            f"{authority_path} names trainable_field "
            f"{authority.get('trainable_field')!r}. v4.4 trains the answer-attempt axis "
            "and nothing else."
        )
    if "reference_content" not in list(authority.get("not_trainable") or ()):
        raise SystemExit(
            f"{authority_path} does not mark reference_content non-trainable. That axis is "
            "computed WITH the reference answer visible; training on it would produce a "
            "detector that needs the answer at runtime and would break the answer-free "
            "claim the whole protocol rests on."
        )

    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    expected = str(authority.get("bundle_sha256") or "")
    actual = str(bundle.get("bundle_sha256") or "")
    if not expected or expected != actual:
        raise SystemExit(
            f"{authority_path} was computed over bundle {expected[:16]}... but "
            f"{bundle_path} hashes to {actual[:16]}.... The authority describes different "
            "rows than the ones about to be trained on. If the bundle was rebuilt WITH "
            "labels, that is the defect: rebuild it unlabelled and join in memory."
        )

    labels_path = Path(str(authority.get("adjudicated_file") or ""))
    if not labels_path.exists():
        raise SystemExit(f"{authority_path} names adjudicated file {labels_path}, which is absent.")
    digest = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    if digest != str(authority.get("adjudicated_sha256")):
        raise SystemExit(
            f"{labels_path} hashes to {digest[:16]}... but {authority_path} binds "
            f"{str(authority.get('adjudicated_sha256'))[:16]}.... The label file has "
            "changed since the authority was frozen."
        )

    # ------------------------------------------------------------- the join itself --
    bundle_ids = [str(p["audit_id"]) for p in bundle.get("pairs", ())]
    seen: Counter = Counter()
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seen[str(json.loads(line)["audit_id"])] += 1
    duplicates = sorted(i for i, n in seen.items() if n > 1)
    if duplicates:
        raise SystemExit(
            f"{labels_path} carries {len(duplicates)} duplicated audit_ids (first "
            f"{duplicates[:3]}). Which label a row gets would depend on file order."
        )
    missing = sorted(set(bundle_ids) - set(seen))
    extra = sorted(set(seen) - set(bundle_ids))
    if missing:
        raise SystemExit(
            f"{len(missing)} bundle rows have no adjudicated label (first {missing[:3]}). "
            "The authority requires a label for every row; a row silently dropped at join "
            "time leaves the training set without anybody deciding it should."
        )
    if extra:
        raise SystemExit(
            f"{len(extra)} adjudicated labels name rows that are not in this bundle (first "
            f"{extra[:3]}). The label file and the bundle describe different populations."
        )
    if int(authority.get("n_labels") or -1) != len(bundle_ids):
        raise SystemExit(
            f"{authority_path} claims n_labels={authority.get('n_labels')} against "
            f"{len(bundle_ids)} bundle rows."
        )

    return {
        "report": str(authority_path),
        "judge_population": "two_local_open_weight_judges",
        "human_grounded": False,
        "publication_label_valid": False,
        "authorises": "ENGINEERING training and evaluation only",
        "adjudicated_file": str(labels_path),
        "adjudicated_sha256": digest,
        "bundle_sha256": expected,
        "panel_sha256": authority.get("panel_sha256"),
        "panel_kappa": (authority.get("panel_gate") or {}).get("kappa"),
        "reference_axis_ran": authority.get("reference_axis_ran"),
        "reference_axis_closed": authority.get("reference_axis_closed"),
        "reference_adjudicated_sha256": authority.get("reference_adjudicated_sha256"),
        "trainable_field": "answer_attempt",
        "not_trainable": ["reference_content"],
        "n_labels": int(authority.get("n_labels") or 0),
        "label_join": "in_memory_by_audit_id_against_the_immutable_bundle",
        "protocol_phase": "v4.4",
        "warning": (
            "This run is authorised by two LOCAL OPEN-WEIGHT model judges. No result from "
            "it may be described as a Goal A result or as publication-ready until the "
            "250-row human validation passes."
        ),
    }


def v4_4_bundle_examples(
    bundle_path: Path, split: str, *, labels_path: Path, eval_key: Path | None = None
) -> tuple[list[dict], dict]:
    """Natural rows from the frozen v4.4 bundle, joined to the adjudicated labels in memory.

    The v4.3 loader reads ``pair["label"]`` because a v4.3 bundle was rebuilt with its
    labels inside. This one must not: see :func:`require_v4_4_label_authority`. The bundle
    file is opened read-only and the labels arrive from ``V4_4_ADJUDICATED.jsonl``.

    Everything else is deliberately identical to v4.3 -- all-row aliases, the bundle's
    group-disjoint split, ``population`` from the sealed key as a metric denominator only.
    A v4.4 that also changed those would confound "the labels got better" with "the split
    changed" in every comparison against v4.3.
    """
    if not bundle_path.exists():
        return [], {
            "source": str(bundle_path),
            "n_rows": 0,
            "why_empty": f"{bundle_path} is absent; run `rdl graph-detector-v4-4-bundle`",
        }
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    label_of = {
        str(row["audit_id"]): str(row["answer_attempt"])
        for row in (
            json.loads(line)
            for line in labels_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    population_of: dict[str, str] = {}
    key_path = Path(eval_key) if eval_key else None
    if key_path and key_path.exists():
        population_of = {
            audit_id: str(entry.get("population", "protected"))
            for audit_id, entry in json.loads(key_path.read_text(encoding="utf-8"))
            .get("rows", {})
            .items()
        }

    out: list[dict] = []
    n_without = 0
    n_unlabelled = 0
    for pair in payload.get("pairs", ()):
        if pair.get("split") != split:
            continue
        label = label_of.get(str(pair["audit_id"]))
        if label not in LABELS:
            n_unlabelled += 1
            continue
        n_without += int(not pair["subject_aliases"])
        out.append(
            {
                "question": pair["conditioning_question"],
                "aliases": list(pair["subject_aliases"]),
                "candidate": pair["candidate_text"],
                "label": label,
                "source": "v4_4_bundle",
                "population": population_of.get(str(pair["audit_id"]), "protected"),
                "group": pair["subject_id"],
            }
        )
    return out, {
        "source": str(bundle_path),
        "bundle_sha256": payload.get("bundle_sha256"),
        "bundle_rewritten_with_labels": False,
        "labels_source": str(labels_path),
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(),
        "conditioning_index_fingerprint_sha256": (
            payload.get("sources", {}).get("conditioning_index_fingerprint_sha256")
        ),
        "n_rows": len(out),
        "n_without_aliases": n_without,
        "n_unlabelled_skipped": n_unlabelled,
        "n_retain_rows": sum(1 for r in out if r["population"] == "retain"),
        "population_source": str(key_path) if population_of else None,
        "why_population": (
            "read from the sealed evaluation key as a metric denominator only; it never "
            "reaches the encoder."
        ),
    }


def require_label_audit(v4_1_dir: Path, v4_2_dir: Path) -> dict:
    """Refuse to train until a label report says the target is legible. No override.

    Two authorities are acceptable and they are NOT equivalent:

    ``LABEL_ALIGNMENT_REPORT.json``
        v4.1. Two human judges. The only authority that can back a publication claim.

    ``DETECTOR_V4_2_MODEL_LABEL_ALIGNMENT_REPORT.json``
        v4.2. Two model judges. Authorises an ENGINEERING run. It carries
        ``human_grounded: false`` and ``publication_label_valid: false``, and this
        function refuses it if those flags have been edited to say otherwise — a report
        claiming human grounding from a model-judge schema is a tampered artifact, and
        loading it would put the claim into the training manifest.

    The human report wins when both exist. v4's ``--force-despite-failed-ceiling``
    existed to train despite a failing artifact; the correct response to that artifact was
    to fix its interpretation, and a flag that skips the fix is a way to keep the mistake.
    """
    human = v4_1_dir / HUMAN_REPORT
    model = v4_2_dir / MODEL_REPORT
    if human.exists():
        report = json.loads(human.read_text(encoding="utf-8"))
        authority = {
            "report": str(human),
            "judge_population": "two_human_judges",
            "human_grounded": True,
            "publication_label_valid": bool(report.get("all_gates_passed")),
            "authorises": "engineering and publication claims",
            "adjudicated_file": report.get("adjudicated_file")
            or str(v4_1_dir / "LABEL_AUDIT_ADJUDICATED.jsonl"),
            "adjudicated_sha256": report.get("adjudicated_sha256"),
        }
    elif model.exists():
        report = json.loads(model.read_text(encoding="utf-8"))
        if report.get("human_grounded") is not False:
            raise SystemExit(
                f"{model} claims human_grounded={report.get('human_grounded')!r}. The "
                "v4.2 schema is model-judge output and that flag is false by "
                "construction; this file has been edited. Refusing to train on it."
            )
        if report.get("publication_label_valid") is not False:
            raise SystemExit(
                f"{model} claims publication_label_valid="
                f"{report.get('publication_label_valid')!r}. Two model judges agreeing is "
                "consistency evidence, not correctness. Refusing to train on it."
            )
        provenance_failures = list((report.get("provenance") or {}).get("failures", ()))
        if provenance_failures:
            raise SystemExit(
                f"{model} records {len(provenance_failures)} provenance failure(s):\n  "
                + "\n  ".join(str(f) for f in provenance_failures)
                + "\nRefusing to train. These are the ways four judge runs can fail to be "
                "four complete runs of the frozen protocol — a smoke run standing in for "
                "a pass, a prompt version that moved, a model that was not the one "
                "requested, rows the judges never saw."
            )
        independence = report.get("judge_independence") or {}
        if independence.get("ok") is False:
            raise SystemExit(
                f"{model}: {independence.get('reason')}. Refusing to train: a kappa "
                "between two judges of one family measures a shared prior."
            )
        authority = {
            "report": str(model),
            "judge_population": str(report.get("judge_population")),
            "human_grounded": False,
            "publication_label_valid": False,
            "judge_independence": independence,
            "authorises": "ENGINEERING training and evaluation only",
            "adjudicated_file": report.get("adjudicated_file")
            or str(v4_2_dir / "V4_2_ADJUDICATED.jsonl"),
            "adjudicated_sha256": report.get("adjudicated_sha256"),
            "warning": (
                "This run is authorised by MODEL judges. No result from it may be "
                "described as a Goal A result or as publication-ready. See "
                "DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md sections 2 and 10."
            ),
        }
    else:
        raise SystemExit(
            f"neither {human} nor {model} exists. Run a label audit first:\n"
            "  human (v4.1):  rdl graph-detector-v4-label-audit\n"
            "                 rdl graph-detector-v4-label-report --judge-a ... --judge-b ...\n"
            "  model (v4.2):  rdl graph-detector-v4-2-llm-judge --judge A --pass blind\n"
            "                 rdl graph-detector-v4-2-llm-judge --judge B --pass blind\n"
            "                 rdl graph-detector-v4-2-label-report\n"
            "Goal A's target is answer_attempt, and no GPU step precedes it."
        )

    if not report.get("all_gates_passed"):
        raise SystemExit(
            "the label audit does NOT clear its decision gate: "
            f"{report.get('failed_gates')}\n{report.get('verdict')}\n"
            "Fix the label protocol, not the model."
        )
    authority["all_gates_passed"] = True
    authority["answer_attempt_kappa"] = (
        report.get("inter_judge", {})
        .get("per_field", {})
        .get("answer_attempt", {})
        .get("cohens_kappa")
    )

    # The labels this authority vouches for, by hash — checked here rather than recorded
    # and ignored. Before this, a passing report next to an edited (or simply different)
    # adjudicated file trained the model on labels nothing had gated: the report carried
    # `adjudicated_sha256` and no code ever compared it to a file. The natural rows then
    # came from whichever adjudicated file happened to exist first, which could be the
    # v4.1 human one while the authority was the v4.2 model report, or the reverse.
    labels_path = Path(str(authority["adjudicated_file"]))
    if not labels_path.exists():
        raise SystemExit(
            f"{authority['report']} vouches for {labels_path}, which does not exist. The "
            "report and its labels travel together; re-run the label report."
        )
    recorded = authority.get("adjudicated_sha256")
    if not recorded:
        raise SystemExit(
            f"{authority['report']} records no adjudicated_sha256, so it cannot vouch for "
            f"{labels_path}. Re-run the label report — it writes the hash of the file it "
            "produced, which is what binds the labels to the audit that passed."
        )
    actual = _sha256_path(labels_path)
    if actual != str(recorded):
        raise SystemExit(
            f"{labels_path} hashes to {str(actual)[:16]}… and {authority['report']} "
            f"vouches for {str(recorded)[:16]}…. The label file changed after the audit "
            "passed. Refusing to train: these are not the labels that cleared the gate."
        )
    authority["adjudicated_sha256_verified"] = True
    return authority


# ------------------------------------------------------- the preregistration, enforced --

# Frozen before training and, from v4.2.2, ENFORCED rather than defaulted. Every one of
# these was a `--flag` with a preregistered default, which is a preregistration in the same
# sense that an unlocked door is a lock: a run with `--seeds 7 --epochs 12` produced a
# manifest that recorded 7 and 12 under a `checkpoint_selection_rule` describing three
# seeds and three epochs, and nothing in the artifact said the plan had been left.
PREREGISTERED_MODEL_REPO = "microsoft/deberta-v3-base"
PREREGISTERED_EPOCHS = 3

# The three commit SHAs the reportable run is pinned to, frozen into a committed artifact
# by `rdl graph-detector-v4-2-freeze-model-pins` before any training. v4.2.2 required the
# revisions to be NON-EMPTY and accepted whatever was typed, which pins the shape of the
# claim and not the claim: two runs a month apart under a moved tag both satisfy it and are
# different experiments. The file is read here and equality is enforced.
MODEL_PINS_FILENAME = "DETECTOR_V4_2_MODEL_PINS.json"


def load_model_pins(v4_2_dir: Path) -> dict:
    """The frozen commit SHAs, or ``{}`` when they have not been frozen yet."""
    path = v4_2_dir / MODEL_PINS_FILENAME
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def enforce_model_pins(args, pins: TrainingPins, frozen: Mapping) -> list[str]:
    """Every revision this run uses that is not the one frozen for it."""
    if not frozen:
        return [
            f"{MODEL_PINS_FILENAME} has not been frozen. A reportable run pins exact "
            "commit SHAs for the encoder, its tokenizer and the NLI baseline; run "
            "`rdl graph-detector-v4-2-freeze-model-pins` on a networked machine first. "
            "Requiring a non-empty --model-revision only pins the SHAPE of the claim."
        ]
    wanted = {
        "model_revision": (pins.model_revision, frozen.get("model_revision")),
        "tokenizer_revision": (pins.tokenizer_revision, frozen.get("tokenizer_revision")),
    }
    if not args.skip_baseline:
        wanted["baseline_revision"] = (args.baseline_revision, frozen.get("baseline_revision"))
    failures = [
        f"{name}: this run uses {have!r} and {MODEL_PINS_FILENAME} freezes {want!r}"
        for name, (have, want) in wanted.items()
        if not want or str(have) != str(want)
    ]
    for name, key in (
        ("model repo", "model_repo_id"),
        ("baseline repo", "baseline_repo_id"),
    ):
        frozen_repo = frozen.get(key)
        used = pins.model_repo_id if key == "model_repo_id" else args.baseline_repo_id
        if frozen_repo and str(used) != str(frozen_repo):
            failures.append(f"{name}: this run uses {used!r} and the pins freeze {frozen_repo!r}")
    return failures


def git_state() -> dict:
    """The commit this run was made from, and whether the tree was clean.

    Package versions were recorded and the source revision was not, so two runs of
    different code produced manifests that differed only in numbers. `dirty` is the field
    that matters: a reportable run from an edited tree cannot be reproduced from any
    commit, and the artifact would name one anyway.
    """
    import subprocess

    def run(*argv: str) -> str | None:
        try:
            out = subprocess.run(
                argv, cwd=REPO, capture_output=True, text=True, timeout=30, check=False
            )
        except Exception:  # pragma: no cover - reporting, not control flow
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = run("git", "status", "--porcelain")
    sha = run("git", "rev-parse", "HEAD")
    return {
        "git_sha": sha,
        "git_branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": None if status is None else bool(status.strip()),
        "git_dirty_paths": (
            sorted({line[3:].strip() for line in status.splitlines() if line.strip()})[:20]
            if status
            else []
        ),
        "why": (
            "a manifest that records package versions but not the source revision cannot "
            "be reproduced. A reportable run additionally refuses a dirty tree: the "
            "recorded SHA would name a commit that is not what ran."
        ),
    }


def enforce_preregistration(args, pins: TrainingPins) -> dict:
    """Refuse a REPORTABLE run whose pins are not the preregistered ones. Returns the record.

    ``--non-reportable`` is the escape hatch, and it is an honest one: it does not skip the
    check, it marks the run. The manifest carries ``reportable: false``, every downstream
    gate refuses a checkpoint selected under it, and the reason is written down. A flag
    that silently permitted the change would leave a manifest that reads exactly like a
    preregistered run.
    """
    deviations: list[str] = []
    if tuple(pins.seeds) != PREREGISTERED_SEEDS:
        deviations.append(
            f"seeds {list(pins.seeds)} != preregistered {list(PREREGISTERED_SEEDS)}. All "
            "three are trained and all three are reported; one seed is a draw from a "
            "distribution nobody showed."
        )
    if int(args.epochs) != PREREGISTERED_EPOCHS:
        deviations.append(f"epochs {args.epochs} != preregistered {PREREGISTERED_EPOCHS}")
    if str(args.model_repo_id) != PREREGISTERED_MODEL_REPO:
        deviations.append(
            f"model repo {args.model_repo_id!r} != preregistered "
            f"{PREREGISTERED_MODEL_REPO!r}. A different encoder is a different experiment."
        )
    if str(args.baseline_repo_id) != DEFAULT_BASELINE_REPO and not args.skip_baseline:
        deviations.append(
            f"baseline repo {args.baseline_repo_id!r} != preregistered "
            f"{DEFAULT_BASELINE_REPO!r}. The floor the fine-tune is compared against is "
            "part of the claim."
        )

    # Every extra training file is hashed, whether or not the run is reportable. A
    # reportable run additionally has to declare them: extra data changes what the model
    # learned, and `--extra-train some.jsonl` left no trace beyond a row count.
    extras = [
        {
            "path": str(path),
            "sha256": _sha256_path(path),
            "n_bytes": path.stat().st_size if path.exists() else None,
            "n_rows": (
                sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
                if path.exists()
                else None
            ),
        }
        for path in [Path(p) for p in (args.extra_train or ())]
    ]
    missing = [e["path"] for e in extras if e["n_rows"] is None]
    if missing:
        raise SystemExit(f"--extra-train file(s) do not exist: {missing}")
    if extras and args.reportable and not args.declare_extra_train:
        deviations.append(
            f"{len(extras)} --extra-train file(s) were supplied without "
            "--declare-extra-train. Extra data is a change to what the model learned; a "
            "reportable run must name it deliberately, and its hash is recorded either way."
        )

    # The exact weights, and the exact source. Both are conditions on a REPORTABLE run and
    # both were previously unchecked: any commit SHA satisfied "the revision is pinned",
    # and no field of the manifest said which revision of this repository produced it.
    frozen_pins = load_model_pins(Path(args.v4_2_dir))
    deviations.extend(enforce_model_pins(args, pins, frozen_pins))
    git = git_state()
    if git["git_dirty"]:
        deviations.append(
            f"the working tree is dirty ({len(git['git_dirty_paths'])} path(s), e.g. "
            f"{git['git_dirty_paths'][:3]}). A reportable run records git_sha "
            f"{str(git['git_sha'])[:12]}, and that commit is not what would run."
        )
    elif git["git_dirty"] is None:
        deviations.append(
            "the git state could not be read, so the run cannot record which revision "
            "produced it"
        )

    if deviations and args.reportable:
        raise SystemExit(
            "this run does not match the preregistration:\n  "
            + "\n  ".join(deviations)
            + "\nEither run it as preregistered, or pass --non-reportable — which does not "
            "skip the check, it records that no result from this run may be reported and "
            "makes every downstream gate refuse the checkpoint it selects."
        )
    return {
        "reportable": bool(args.reportable),
        "model_pins": frozen_pins or None,
        "model_pins_file": str(Path(args.v4_2_dir) / MODEL_PINS_FILENAME),
        "git": git,
        "preregistered_seeds": list(PREREGISTERED_SEEDS),
        "preregistered_epochs": PREREGISTERED_EPOCHS,
        "preregistered_model_repo_id": PREREGISTERED_MODEL_REPO,
        "preregistered_baseline_repo_id": DEFAULT_BASELINE_REPO,
        "deviations": deviations,
        "extra_train_files": extras,
        "extra_train_declared": bool(args.declare_extra_train),
        "why_enforced": (
            "a preregistration with an override flag and no record is a default. These "
            "are checked, and a run that leaves them is marked non-reportable rather than "
            "quietly recorded as one that did not."
        ),
    }


def _sha256_path(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover
        pass
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)  # cudnn conv kernels; fp16 matmul is not exact


# ------------------------------------------------------------------------- data --


def synthetic_examples(dataset: dict, split: str) -> list[dict]:
    """``{question, aliases, candidate, label}`` per clause of every row in ``split``.

    One example per clause, not per row: the runtime scores clauses, and training on whole
    messages would fit a different unit than the one the threshold is chosen on.
    """
    from rdl.defenses.atomic_text import segment

    out: list[dict] = []
    for row in dataset["rows"]:
        if row["split"] != split:
            continue
        # The row's false-alarm pool IS its population. Carried through to the development
        # metrics so the retain false-alarm rate has its own denominator: pooled with the
        # generated-clean rows it is 2 pools in 6, and an aggregate NONE FPR of 0.10 is
        # satisfiable with a retain FPR of 0.30.
        population = "retain" if row.get("false_alarm_pool") == "retain_answer" else "protected"
        for candidate in row["candidates"]:
            for clause in segment(candidate) or ():
                out.append(
                    {
                        "question": row["protected_question"],
                        "aliases": list(row["aliases"]),
                        "candidate": clause.text,
                        "label": row["label"],
                        "split": split,
                        "source": "synthetic",
                        "population": population,
                        "group": row["subject"],
                    }
                )
    return out


def natural_alias_index(policy_cohort: Path, max_aliases: int) -> dict[str, list[str]]:
    """``{protected_question -> aliases}`` from the OFFLINE concept registry.

    The fix for the training-serving skew. v4.1 trained the natural rows with
    ``aliases: []`` while the runtime hands the model routed identity aliases in segment
    A — so the model was fitted on an input shape it is never served. This joins the
    adjudicated rows back to the same registry the gate builds, keyed by the question text
    the judge file carries, which is the only key the blinded row exposes.

    Built from committed files alone, with no dataset load and therefore no network, in
    the same way ``detector_v4_label_audit._registry_from_bank`` is. An unmatched question
    yields no aliases and is counted rather than silently becoming an empty list.
    """
    from rdl.defenses.concept_registry import ConceptPolicy, ConceptRegistry
    from rdl.studies.graph_leak.cohort import load_cohort

    bank_path = V4 / "DETECTOR_V4_NATURAL_BANK.json"
    if not (bank_path.exists() and policy_cohort.exists()):
        return {}
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    cohort = load_cohort(policy_cohort, exclusions_path=None)
    concept_of = {e.item_id: e.concept_id for e in cohort.items}

    seen: dict[tuple[str, str], str] = {}
    for partition in ("development", "heldout"):
        block = bank["partitions"].get(partition, {})
        for keyname in ("clean", "leaking"):
            for row in block.get(keyname, ()):
                item_id = str(row.get("item_id", ""))
                concept = concept_of.get(item_id)
                if concept and row.get("request"):
                    seen[(item_id, str(row["request"]))] = concept
    rows = [
        {"item_id": item_id, "concept_id": concept, "question": question}
        for (item_id, question), concept in sorted(seen.items())
    ]
    registry = ConceptRegistry.from_questions(rows, policy=ConceptPolicy())
    aliases_of_concept = {c.forget_id: list(c.aliases)[:max_aliases] for c in registry.concepts()}
    return {
        str(row["question"]): aliases_of_concept.get(str(row["concept_id"]), []) for row in rows
    }


def v4_3_bundle_examples(
    bundle_path: Path, split: str, *, eval_key: Path | None = None
) -> tuple[list[dict], dict]:
    """Natural rows from the frozen v4.3 pair bundle, which has already fixed three things.

    Preferred over :func:`natural_examples` whenever ``--v4-3-bundle`` is passed, because
    the bundle is where the v4.2 defects were repaired and repairing them here as well
    would give two places for them to drift apart:

    * aliases come from the all-row conditioning index, so retain rows carry them at the
      same rate protected rows do — ``natural_alias_index`` gave retain rows none;
    * ``split`` is the bundle's group-disjoint assignment, so one author is wholly on one
      side — the ``audit_id`` parity halving let an author straddle;
    * the input is ``(conditioning_question, subject_aliases, candidate_text)``, identical
      in shape for both populations.

    ``group`` is the subject id rather than the audit id, so any group-aware sampler
    downstream inherits the same disjointness.
    """
    if not bundle_path.exists():
        return [], {
            "source": str(bundle_path),
            "n_rows": 0,
            "why_empty": f"{bundle_path} is absent; run `rdl graph-detector-v4-3-bundle`",
        }
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))

    # `population` is a METRIC DENOMINATOR and never a model input, which is exactly why
    # the bundle's pairs do not carry it -- a field that cannot be serialised cannot be
    # tokenized by accident. It therefore has to come from the sealed evaluation key here.
    #
    # Defaulting every row to "protected" (which this function did until now) is not a
    # neutral fallback: it makes all 300 retain rows invisible to every population-split
    # diagnostic, so `selection_retain_nonanswer_rate` reports over an empty pool and the
    # protected-clean denominator silently absorbs them.
    population_of: dict[str, str] = {}
    key_path = Path(eval_key) if eval_key else None
    if key_path and key_path.exists():
        population_of = {
            audit_id: str(entry.get("population", "protected"))
            for audit_id, entry in json.loads(key_path.read_text(encoding="utf-8"))
            .get("rows", {})
            .items()
        }

    out: list[dict] = []
    n_without = 0
    n_unlabelled = 0
    for pair in payload.get("pairs", ()):
        if pair.get("split") != split:
            continue
        label = pair.get("label")
        if label not in LABELS:
            n_unlabelled += 1
            continue
        n_without += int(not pair["subject_aliases"])
        out.append(
            {
                "question": pair["conditioning_question"],
                "aliases": list(pair["subject_aliases"]),
                "candidate": pair["candidate_text"],
                "label": label,
                "source": "v4_3_bundle",
                "population": population_of.get(str(pair["audit_id"]), "protected"),
                "group": pair["subject_id"],
            }
        )
    n_retain = sum(1 for r in out if r["population"] == "retain")
    return out, {
        "source": str(bundle_path),
        "bundle_sha256": payload.get("bundle_sha256"),
        "conditioning_index_fingerprint_sha256": (
            payload.get("sources", {}).get("conditioning_index_fingerprint_sha256")
        ),
        "n_rows": len(out),
        "n_without_aliases": n_without,
        "n_unlabelled_skipped": n_unlabelled,
        "n_retain_rows": n_retain,
        "population_source": str(key_path) if population_of else None,
        "why_population": (
            (
                "read from the sealed evaluation key as a metric denominator only; it never "
                "reaches the encoder. Without it every row would default to protected and the "
                "retain pool would be invisible to the development diagnostics."
            )
            if population_of
            else (
                "NO evaluation key was supplied, so every row defaults to protected and the "
                "retain diagnostics are not meaningful. Pass --v4-3-eval-key for a reportable "
                "run."
            )
        ),
        "labels": payload.get("labels", {}),
        "split_rule": payload.get("split_rule", {}).get("unit"),
        "shortcut_probe": payload.get("shortcut_probe", {}).get("alias_channel_excess"),
    }


def natural_examples(
    v4_1_dir: Path,
    v4_2_dir: Path,
    split: str,
    *,
    alias_index: dict[str, list[str]] | None = None,
    adjudicated_path: Path | None = None,
) -> tuple[list[dict], dict]:
    """Adjudicated natural rows joined to their blinded text and to their routed aliases.

    ``adjudicated_path`` is the file the LABEL AUTHORITY vouches for, and it is what a
    real training run passes. Without it this function fell back to "whichever adjudicated
    file exists first", which is a different question from "which labels authorised this
    run": with both audits on disk it could train on the v4.1 human labels while
    `require_label_audit` had returned the v4.2 model authority, and the manifest would
    describe the wrong annotator. The fallback survives only for callers that have no
    authority to hand — the tests that exercise the join itself.

    The blinded judge file supplies the question and the candidate; neither overlay carries
    a reference answer, which is why the join can be done here at all.
    """
    if adjudicated_path is not None:
        judge_path = v4_1_dir / "LABEL_AUDIT_JUDGE_A.jsonl"
        if not (adjudicated_path.exists() and judge_path.exists()):
            return [], {
                "source": str(adjudicated_path),
                "n_rows": 0,
                "n_without_aliases": 0,
                "why_empty": (
                    f"{adjudicated_path} or {judge_path} is absent; the adjudicated labels "
                    "cannot be joined to the blinded text they describe"
                ),
            }
        return _natural_examples_from(
            adjudicated_path, judge_path, v4_1_dir, split, alias_index=alias_index
        )

    sources = (
        (v4_1_dir / "LABEL_AUDIT_ADJUDICATED.jsonl", v4_1_dir / "LABEL_AUDIT_JUDGE_A.jsonl"),
        (v4_2_dir / "V4_2_ADJUDICATED.jsonl", v4_1_dir / "LABEL_AUDIT_JUDGE_A.jsonl"),
    )
    adjudicated_path, judge_path = next(
        ((a, j) for a, j in sources if a.exists() and j.exists()), (None, None)
    )
    if adjudicated_path is None:
        return [], {"source": None, "n_rows": 0, "n_without_aliases": 0}
    return _natural_examples_from(
        adjudicated_path, judge_path, v4_1_dir, split, alias_index=alias_index
    )


def _natural_examples_from(
    adjudicated_path: Path,
    judge_path: Path,
    v4_1_dir: Path,
    split: str,
    *,
    alias_index: dict[str, list[str]] | None = None,
) -> tuple[list[dict], dict]:
    """The join itself, over ONE named adjudicated file and its blinded judge file."""

    def load(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    alias_index = alias_index or {}
    text_of = {r["audit_id"]: r for r in load(judge_path)}

    # ONE field is read from the audit key: `population`, which says whether the row is a
    # retain-author row. It is a metric denominator, never a model input — the encoder is
    # handed the question, the aliases and the candidate and nothing else. Without it the
    # retain rows are indistinguishable from protected NONE rows in the development pool,
    # and the retain false-alarm ceiling cannot be enforced at checkpoint selection at all.
    key_path = v4_1_dir / "LABEL_AUDIT_KEY.json"
    population_of: dict[str, str] = {}
    if key_path.exists():
        population_of = {
            audit_id: str(entry.get("population", "protected"))
            for audit_id, entry in json.loads(key_path.read_text(encoding="utf-8"))
            .get("rows", {})
            .items()
        }

    out: list[dict] = []
    n_without = 0
    for row in load(adjudicated_path):
        label = row.get("answer_attempt")
        blind = text_of.get(row["audit_id"])
        if label not in LABELS or blind is None:
            continue
        # Content-addressed train/development halving, computed from the audit id alone so
        # it does not move when rows are added.
        half = "train" if int(row["audit_id"][:2], 16) % 2 == 0 else "development"
        if half != split:
            continue
        question = str(blind["protected_question"])
        aliases = list(alias_index.get(question, ()))
        n_without += int(not aliases)
        out.append(
            {
                "question": question,
                "aliases": aliases,
                "candidate": blind["candidate_text"],
                "label": label,
                "source": "natural_adjudicated",
                "population": population_of.get(row["audit_id"], "protected"),
                "group": row["audit_id"],
            }
        )
    return out, {
        "source": str(adjudicated_path),
        "n_rows": len(out),
        "n_without_aliases": n_without,
        "n_retain_rows": sum(1 for r in out if r["population"] == "retain"),
        "population_source": str(key_path) if population_of else None,
        "why_population": (
            "the retain false-alarm ceiling is a separate gate with a separate "
            "denominator. Only `population` is read from the key, and only as a metric "
            "denominator; it never reaches the encoder."
        ),
        "why_aliases": (
            "runtime inference receives routed identity aliases; v4.1 trained these rows "
            "with an empty alias list, which is training-serving skew in the one field "
            "the router contributes."
        ),
    }


def extra_examples(paths: list[Path]) -> list[dict]:
    """Public QA/answerability rows, as ``{question, candidate, label}`` JSONL."""
    out: list[dict] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") not in LABELS:
                raise SystemExit(f"{path}: label must be one of {LABELS}, got {row.get('label')!r}")
            out.append(
                {
                    "question": str(row["question"]),
                    "aliases": list(row.get("aliases", ())),
                    "candidate": str(row["candidate"]),
                    "label": str(row["label"]),
                    "source": f"extra:{path.name}",
                    "group": str(row.get("group", path.name)),
                }
            )
    return out


class AnswerabilityDataset:
    """Encodes lazily through ``budget_encode`` and counts every candidate truncation."""

    def __init__(self, examples: list[dict], tokenizer, budget) -> None:
        from rdl.defenses.cross_encoder_answerability import budget_encode

        self._encode = budget_encode
        self.examples = examples
        self.tokenizer = tokenizer
        self.budget = budget
        self.n_candidate_truncations = 0
        self.n_question_truncations = 0

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        row = self.examples[i]
        encoding, stats = self._encode(
            self.tokenizer,
            question=row["question"],
            aliases=row["aliases"],
            candidate=row["candidate"],
            budget=self.budget,
        )
        self.n_candidate_truncations += int(bool(stats["candidate_truncated"]))
        self.n_question_truncations += int(bool(stats["question_truncated"]))
        return {**encoding, "labels": LABELS.index(row["label"])}


def collate(features: list[dict], pad_id: int) -> dict:
    import torch

    keys = [k for k in ("input_ids", "attention_mask", "token_type_ids") if k in features[0]]
    width = max(len(f["input_ids"]) for f in features)
    batch = {
        key: torch.tensor(
            [
                list(f[key]) + [pad_id if key == "input_ids" else 0] * (width - len(f[key]))
                for f in features
            ],
            dtype=torch.long,
        )
        for key in keys
    }
    batch["labels"] = torch.tensor([f["labels"] for f in features], dtype=torch.long)
    return batch


# -------------------------------------------------------------------- evaluation --


def evaluate(model, loader, device, populations: list[str] | None = None) -> dict:
    """Per-class metrics, macro F1, and the Goal A selection numbers on development.

    ``macro_f1`` is a diagnostic and selects nothing (see
    :data:`CHECKPOINT_SELECTION_RULE`). The number that selects is
    ``selection_answer_recall``: ANSWER-recall at the best EXACT score breakpoint whose
    protected-clean AND retain false-alarm rates BOTH clear :data:`DEV_FPR_CEILING`. Exact
    breakpoints rather than a 0.05 grid, because a coarse grid reports the recall of a
    threshold nobody chose.

    ``populations`` is aligned with the loader's rows, which is why ``dev_loader`` is built
    with ``shuffle=False``. Without it every NONE row is one pool, and a checkpoint whose
    retain FPR is 0.30 passes on an aggregate of 0.10 as long as the generated-clean rows
    outnumber the retain ones — which in the synthetic pool they do, four classes to two.
    """
    import torch

    model.eval()
    confusion = [[0] * len(LABELS) for _ in LABELS]
    answer_scores: list[float] = []
    golds: list[int] = []
    with torch.inference_mode():
        for batch in loader:
            labels = batch.pop("labels")
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.float().cpu()
            probabilities = logits.softmax(-1)
            for gold, predicted, row in zip(
                labels.tolist(),
                logits.argmax(-1).tolist(),
                probabilities.tolist(),
                strict=True,
            ):
                confusion[gold][predicted] += 1
                answer_scores.append(float(row[ANSWER_INDEX]))
                golds.append(int(gold))

    per_class = {}
    f1s = []
    for i, name in enumerate(LABELS):
        tp = confusion[i][i]
        fp = sum(confusion[g][i] for g in range(len(LABELS))) - tp
        fn = sum(confusion[i]) - tp
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall)
            else 0.0
        )
        per_class[name] = {
            "support": sum(confusion[i]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1s.append(f1)
    total = sum(sum(row) for row in confusion)
    return {
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "macro_f1_role": "diagnostic; selects nothing",
        "accuracy": (sum(confusion[i][i] for i in range(len(LABELS))) / total) if total else None,
        "per_class": per_class,
        "confusion": {
            LABELS[i]: dict(zip(LABELS, confusion[i], strict=True)) for i in range(len(LABELS))
        },
        **selection_metrics(answer_scores, golds, populations),
    }


def selection_metrics(
    answer_scores: list[float], golds: list[int], populations: list[str] | None = None
) -> dict:
    """ANSWER-recall at the best exact breakpoint clearing the protected-clean ceiling.

    Two pools, not one. ``protected_clean`` is NONE rows about a protected author;
    ``retain`` is every row about a retain author. They are separate gates downstream
    (``GOAL_A_GATES`` names both), so selecting a checkpoint on their union picks
    checkpoints the downstream gate will reject.

    **v4.3 removed the retain pool's veto.** Until v4.3 the retain pool was a second
    ceiling here: any retain row whose ANSWER score cleared the threshold counted as a
    false alarm, whatever its label. Most of those rows genuinely do answer their retain
    question and are labelled ANSWER, so cross-entropy taught the model to score them high
    while this function rejected every checkpoint that did — a contradiction on 300 of the
    1,019 rows, resolved in favour of being wrong about retain answerability.

    The runtime never faces that trade-off. A retain question is not in the protected
    store, so a retain request routes to nothing and creates no Forget-ID regardless of any
    answerability score; measured against ``PROTECTED_STORE_RUNTIME.json``, 0 of the 300
    retain requests route at all. The real retain false-alarm rate is therefore end to end
    and belongs to ``rdl.eval.detector_v4_3.store_conditioned_metrics``, which routes
    first. What survives here is a DIAGNOSTIC — how often retain rows score high — which is
    reported and constrains nothing.
    """
    none_index = LABELS.index("NONE")
    if populations is None or len(populations) != len(answer_scores):
        populations = ["protected"] * len(answer_scores)
    triples = list(zip(answer_scores, golds, populations, strict=True))

    positives = [s for s, g, p in triples if g == ANSWER_INDEX and p != "retain"]
    protected_clean = [s for s, g, p in triples if g == none_index and p != "retain"]
    # Retain NONE rows only. A retain row labelled ANSWER scoring high is a correct
    # prediction, and counting it here is the defect described above.
    retain = [s for s, g, p in triples if p == "retain" and g == none_index]
    if not positives or not protected_clean:
        return {
            "selection_answer_recall": None,
            "selection_threshold": None,
            "selection_none_fpr": None,
            "selection_protected_clean_fpr": None,
            "selection_retain_nonanswer_rate": None,
            "n_selection_positives": len(positives),
            "n_selection_protected_clean": len(protected_clean),
            "n_selection_retain_nonanswer": len(retain),
            "selection_note": "development pool lacks protected ANSWER or protected NONE rows",
        }
    # Exact breakpoints: every score is a candidate operating point, plus one above the
    # maximum so "fire on nothing" is representable.
    breakpoints = sorted({*answer_scores, max(answer_scores) + 1e-9})
    best: tuple[float, float, float] | None = None
    for threshold in breakpoints:
        fpr = sum(1 for s in protected_clean if s >= threshold) / len(protected_clean)
        # The retain rate is computed and reported at the chosen threshold, but it is NOT
        # a constraint. See the docstring: at runtime a retain request does not route, so
        # this pool cannot produce a Forget-ID and must not veto a checkpoint.
        if fpr > DEV_FPR_CEILING:
            continue
        recall = sum(1 for s in positives if s >= threshold) / len(positives)
        candidate = (recall, -threshold, fpr)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        return {
            "selection_answer_recall": 0.0,
            "selection_threshold": None,
            "selection_none_fpr": None,
            "selection_protected_clean_fpr": None,
            "selection_retain_nonanswer_rate": None,
            "n_selection_positives": len(positives),
            "n_selection_protected_clean": len(protected_clean),
            "n_selection_retain_nonanswer": len(retain),
            "selection_note": (
                f"no threshold clears the protected-clean FPR ceiling {DEV_FPR_CEILING}"
            ),
        }
    recall, negative_threshold, fpr = best
    threshold = -negative_threshold
    retain_rate = sum(1 for s in retain if s >= threshold) / len(retain) if retain else None
    return {
        "selection_answer_recall": recall,
        "selection_threshold": threshold,
        # Kept under its original name so older artifacts stay readable, and now equal to
        # the protected-clean rate rather than to a mixture of two pools.
        "selection_none_fpr": fpr,
        "selection_protected_clean_fpr": fpr,
        # Renamed from `selection_retain_fpr` deliberately. The old name asserted that this
        # was a false-alarm rate; it is the rate at which retain NONE rows score high in a
        # DIRECT pair the runtime never forms. The end-to-end retain FPR — the one the
        # gates check — comes from `rdl.eval.detector_v4_3.store_conditioned_metrics`.
        "selection_retain_nonanswer_rate": retain_rate,
        "n_selection_positives": len(positives),
        "n_selection_protected_clean": len(protected_clean),
        "n_selection_retain_nonanswer": len(retain),
        "selection_note": (
            "clause-level proxy on the development pool, under the PROTECTED-CLEAN ceiling "
            "only. The retain pool is reported and does not constrain: a retain request "
            "does not route against the protected store, so it cannot create a Forget-ID. "
            "NOT the reportable operating point, which `rdl graph-detector-v4-3-select` "
            "chooses on routed rows through the store-conditioned path."
        ),
    }


def resolve_entailment_index(config) -> tuple[int, dict]:
    """Find the entailment class in an NLI head's own label map. Ambiguity raises.

    v4.1 hard-coded ``1`` with a comment saying it was confirmed on the box. A moved
    checkpoint or a different NLI repo silently turns the baseline into a measurement of
    the contradiction head, and the number still looks like an AUC.
    """
    id2label = dict(getattr(config, "id2label", {}) or {})
    label2id = dict(getattr(config, "label2id", {}) or {})
    if not id2label and not label2id:
        raise SystemExit(
            "the baseline checkpoint exposes no label map, so the entailment class cannot "
            "be resolved. Pass --baseline-entailment-label with the exact label name."
        )
    if not id2label:
        id2label = {int(v): str(k) for k, v in label2id.items()}
    matches = [
        int(index)
        for index, name in id2label.items()
        if str(name).strip().lower().replace("-", "_") in ENTAILMENT_ALIASES
    ]
    if len(matches) != 1:
        raise SystemExit(
            f"cannot resolve a unique entailment class from {id2label}. Found {matches}. "
            "Pass --baseline-entailment-label with the exact label name; a guessed index "
            "makes the baseline a measurement of a different head."
        )
    return matches[0], {"id2label": {int(k): str(v) for k, v in id2label.items()}}


def baseline_zero_shot(
    repo_id: str,
    revision: str,
    examples: list[dict],
    budget,
    device,
    *,
    entailment_label: str = "",
) -> dict:
    """Entailment mass from a pinned pretrained NLI cross-encoder, as an answerability proxy.

    Reported as a ranking baseline, not as a classifier: 'the candidate entails an answer to
    the question' and 'the candidate attempts to answer the question' are different
    predicates, and the number is here so the fine-tune's contribution is attributable
    rather than so it can be deployed.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from rdl.defenses.cross_encoder_answerability import budget_encode

    tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id, revision=revision)
    model.to(device).eval()

    if entailment_label:
        label2id = {str(k).lower(): int(v) for k, v in (model.config.label2id or {}).items()}
        if entailment_label.lower() not in label2id:
            raise SystemExit(
                f"--baseline-entailment-label {entailment_label!r} is not in {sorted(label2id)}"
            )
        index = label2id[entailment_label.lower()]
        label_map = {"id2label": dict(model.config.id2label or {}), "override": entailment_label}
    else:
        index, label_map = resolve_entailment_index(model.config)

    scores: list[float] = []
    golds: list[str] = []
    for row in examples:
        encoding, _stats = budget_encode(
            tokenizer,
            question=row["question"],
            aliases=row["aliases"],
            candidate=row["candidate"],
            budget=budget,
        )
        batch = {k: torch.tensor([v], dtype=torch.long).to(device) for k, v in encoding.items()}
        with torch.inference_mode():
            probabilities = model(**batch).logits.float().softmax(-1)[0].cpu().tolist()
        scores.append(float(probabilities[index]))
        golds.append(row["label"])

    positives = [s for s, g in zip(scores, golds, strict=True) if g == "ANSWER"]
    negatives = [s for s, g in zip(scores, golds, strict=True) if g == "NONE"]
    pairs = len(positives) * len(negatives)
    auc = (
        (
            sum(1 for p in positives for n in negatives if p > n)
            + 0.5 * sum(1 for p in positives for n in negatives if p == n)
        )
        / pairs
        if pairs
        else None
    )
    return {
        "repo_id": repo_id,
        "revision": revision,
        "entailment_index": index,
        "entailment_index_resolved_from": label_map,
        "n_scored": len(scores),
        "n_answer": len(positives),
        "n_none": len(negatives),
        "answer_vs_none_auc": auc,
        "mean_entailment_on_answer": (sum(positives) / len(positives)) if positives else None,
        "mean_entailment_on_none": (sum(negatives) / len(negatives)) if negatives else None,
        "note": (
            "zero-shot ranking baseline from a pinned pretrained NLI head. "
            "microsoft/deberta-v3-base is NOT an answerability baseline: its "
            "classification head is randomly initialised, so its off-the-shelf number "
            "measures that random head."
        ),
    }


# ------------------------------------------------------------------ checkpoint hashes --
#
# One implementation, in `rdl.defenses.checkpoint_digest`, shared with the loader that
# VERIFIES these hashes. A second copy here would drift, and a verifier that hashes a
# different file set than the writer reports a mismatch on every honest checkpoint — which
# is how verification gets turned off.
from rdl.defenses.checkpoint_digest import (  # noqa: E402
    checkpoint_hashes as _checkpoint_hashes,
)
from rdl.defenses.checkpoint_digest import (  # noqa: E402
    verify_checkpoint_hashes,
)


def checkpoint_hashes(path: Path) -> dict:
    """SHA-256 of everything that makes the checkpoint what it is, under this label order."""
    return _checkpoint_hashes(Path(path), labels=LABELS)


# ---------------------------------------------------------------------- one seed --


def accumulation_windows(n_batches: int, accumulation: int) -> list[int]:
    """The divisor for each batch's loss: the size of the window that batch belongs to.

    All windows hold ``accumulation`` batches except the last, which holds the remainder.
    Dividing every batch by ``accumulation`` — which is what v4.1 and v4.2.1 both did — is
    correct for every full window and wrong for the tail: with 101 batches and a factor of
    2, the 101st batch is a window of one, and dividing it by two halves the gradient of
    that epoch's final optimizer step. v4.1 discarded that step entirely, v4.2.1 took it at
    half the intended learning rate, and neither shows up anywhere but here.

    Pure and separate from the training loop so the arithmetic is testable without a GPU,
    which is the only reason either version of the bug survived.
    """
    accumulation = max(1, int(accumulation))
    n_batches = max(0, int(n_batches))
    tail = n_batches % accumulation
    n_full = n_batches - tail
    return [accumulation] * n_full + [tail] * tail


def train_one_seed(
    *,
    seed: int,
    pins: TrainingPins,
    train_rows: list[dict],
    dev_rows: list[dict],
    tokenizer,
    budget,
    output_dir: Path,
    device: str,
    epochs: int,
    min_optimizer_steps: int = 0,
) -> dict:
    """Train one seed. Returns its history and per-epoch development metrics."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    set_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(
        pins.model_repo_id, revision=pins.model_revision, num_labels=len(LABELS)
    ).to(device)
    model.config.id2label = dict(enumerate(LABELS))
    model.config.label2id = {name: i for i, name in enumerate(LABELS)}

    train_set = AnswerabilityDataset(train_rows, tokenizer, budget)
    dev_set = AnswerabilityDataset(dev_rows, tokenizer, budget)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=pins.per_device_train_batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda f: collate(f, pad_id),
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=pins.per_device_train_batch_size * 2,
        shuffle=False,
        collate_fn=lambda f: collate(f, pad_id),
    )

    # Inverse-frequency class weights. The Goal A classes are not balanced — PARTIAL is
    # rare and is the class whose confusion with ANSWER makes the accumulation result
    # untestable — so an unweighted loss would learn to never predict it.
    counts = [max(1, sum(1 for r in train_rows if r["label"] == name)) for name in LABELS]
    weights = torch.tensor(
        [len(train_rows) / (len(LABELS) * c) for c in counts], dtype=torch.float
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=pins.learning_rate, weight_decay=pins.weight_decay
    )
    # Ceiling division, and the loop below steps on the tail. v4.1 used floor division
    # here and `step % k == 0` there, so the schedule was built for fewer steps than the
    # loop takes AND the last partial accumulation was discarded — two halves of the same
    # bug, each of which hides the other in the logs.
    steps_per_epoch = max(1, math.ceil(len(train_loader) / pins.gradient_accumulation_steps))
    total_steps = steps_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * pins.warmup_ratio), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=pins.fp16 and device.startswith("cuda"))

    history: list[dict] = []
    n_optimizer_steps = 0
    started = time.time()
    n_batches = len(train_loader)
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        accumulation = max(1, int(pins.gradient_accumulation_steps))
        windows = accumulation_windows(n_batches, accumulation)
        for step, batch in enumerate(train_loader, start=1):
            window = windows[step - 1]
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", enabled=pins.fp16 and device.startswith("cuda")):
                logits = model(**batch).logits
                loss = loss_fn(logits.float(), labels) / window
            scaler.scale(loss).backward()
            running += float(loss.item()) * window
            # The tail steps, and it steps on a correctly normalized gradient.
            should_step = (step % accumulation == 0) or (step == n_batches)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), pins.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                n_optimizer_steps += 1

        # Aligned with dev_loader, which is built shuffle=False for exactly this reason.
        metrics = evaluate(
            model, dev_loader, device, [str(r.get("population", "protected")) for r in dev_rows]
        )
        path = output_dir / f"seed{seed}-checkpoint-epoch{epoch}"
        model.save_pretrained(path)
        tokenizer.save_pretrained(path)
        history.append(
            {
                "seed": seed,
                "epoch": epoch,
                "train_loss": running / max(1, n_batches),
                "development": metrics,
                "checkpoint": str(path),
                "checkpoint_hashes": checkpoint_hashes(path),
            }
        )
        print(
            f"seed {seed} epoch {epoch}: "
            f"selection_recall={metrics['selection_answer_recall']} "
            f"macro_f1={metrics['macro_f1']:.4f} loss={running / max(1, n_batches):.4f}"
        )

    if n_optimizer_steps < min_optimizer_steps:
        raise SystemExit(
            f"only {n_optimizer_steps} optimizer steps were taken; the smoke mode "
            f"requires at least {min_optimizer_steps}. The loop is not training."
        )
    return {
        "seed": seed,
        "history": history,
        "n_optimizer_steps": n_optimizer_steps,
        "steps_per_epoch": steps_per_epoch,
        "planned_total_optimizer_steps": total_steps,
        "n_batches_per_epoch": n_batches,
        "gradient_accumulation": {
            "steps": int(pins.gradient_accumulation_steps),
            "tail_window_batches": n_batches % max(1, int(pins.gradient_accumulation_steps)),
            "normalization": (
                "each batch's loss is divided by the number of batches in ITS accumulation "
                "window, so the epoch's final partial window steps on a gradient of the "
                "same scale as every full one. Dividing the tail by the full factor — "
                "which v4.2.1 did — takes one optimizer step per epoch at a fraction of "
                "the intended learning rate."
            ),
        },
        "wall_clock_seconds": round(time.time() - started, 1),
        "class_weights": dict(zip(LABELS, weights.detach().cpu().tolist(), strict=True)),
        "n_candidate_truncations": train_set.n_candidate_truncations,
        "n_question_truncations": train_set.n_question_truncations,
        "n_development_candidate_truncations": dev_set.n_candidate_truncations,
    }


def select_checkpoint(seed_results: list[dict]) -> dict:
    """Apply :data:`CHECKPOINT_SELECTION_RULE` across every (seed, epoch) pair."""
    candidates = [entry for result in seed_results for entry in result["history"]]

    def eligible_entry(entry: dict) -> bool:
        development = entry["development"]
        if development.get("selection_answer_recall") is None:
            return False
        protected = development.get("selection_protected_clean_fpr")
        if protected is None:
            return False
        # v4.3: the retain pool no longer gates eligibility here. It cannot — a retain
        # request does not route against the protected store, so no threshold on this
        # pair-level score can create or prevent a Forget-ID. The end-to-end retain FPR is
        # a separate, routed measurement and is enforced by
        # `rdl.eval.detector_v4_3.store_conditioned_metrics` at the point the operating
        # point is frozen. Keeping the old veto here re-creates the contradiction v4.3
        # exists to remove: cross-entropy trains retain ANSWER rows up, this rule pushed
        # them down.
        return protected <= DEV_FPR_CEILING

    eligible = [entry for entry in candidates if eligible_entry(entry)]
    if not eligible:
        return {
            "selected": None,
            "reason": (
                f"no (seed, epoch) pair reached a development threshold clearing "
                f"protected-clean FPR <= {DEV_FPR_CEILING}, measured. The rule does not "
                "fall back to macro F1 and does not fall back to the aggregate NONE rate; "
                "a checkpoint selected by a different rule than the frozen one is not the "
                "checkpoint the protocol authorises. The end-to-end retain false-alarm "
                "rate is checked separately, on routed rows, when the operating point is "
                "frozen."
            ),
            "n_candidates": len(candidates),
            "n_eligible": 0,
        }
    best = max(
        eligible,
        key=lambda e: (
            e["development"]["selection_answer_recall"],
            -(e["development"]["selection_threshold"] or 0.0),
            -e["epoch"],
            -e["seed"],
        ),
    )
    return {
        "selected": best,
        "rule": CHECKPOINT_SELECTION_RULE,
        "n_candidates": len(candidates),
        "n_eligible": len(eligible),
    }


# ------------------------------------------------------------------------ smoke --


def run_smoke(args, pins: TrainingPins, budget, tokenizer, train_rows, dev_rows) -> int:
    """A small, explicitly NON-REPORTABLE CUDA check. Minutes, before the real run.

    Catches the class of error that only appears on a GPU — a tensor left on CPU, a
    checkpoint that will not reload, an inference path that was never exercised batched —
    on 64 rows instead of on the full corpus after an hour.
    """
    import torch

    from rdl.defenses.cross_encoder_answerability import CrossEncoderAnswerabilityDetector
    from rdl.defenses.detection_context import ProtectedQuestion, build_context

    device = args.device
    # Its own directory. `train_one_seed` writes `seed{seed}-checkpoint-epoch{n}`, and the
    # smoke uses the first pre-registered seed and epoch 1 — exactly the path the real run
    # writes. Sharing the directory means a 64-row smoke checkpoint can be mistaken for,
    # or can overwrite, a trained one.
    smoke_dir = args.output_dir / "smoke"
    checks: dict[str, object] = {}
    checks["output_dir"] = str(smoke_dir)
    checks["cuda_available"] = bool(torch.cuda.is_available())
    checks["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    train_rows = train_rows[: args.smoke_train_rows]
    dev_rows = dev_rows[: args.smoke_dev_rows]
    if not train_rows or not dev_rows:
        raise SystemExit("the smoke mode needs training and development rows; both pools are empty")

    result = train_one_seed(
        seed=pins.seeds[0],
        pins=pins,
        train_rows=train_rows,
        dev_rows=dev_rows,
        tokenizer=tokenizer,
        budget=budget,
        output_dir=smoke_dir,
        device=device,
        epochs=1,
        min_optimizer_steps=1,
    )
    checks["optimizer_steps"] = result["n_optimizer_steps"]
    checkpoint = Path(result["history"][-1]["checkpoint"])

    if device.startswith("cuda"):
        checks["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
        if not checks["peak_memory_bytes"]:
            raise SystemExit("CUDA was requested and peak allocation is zero; nothing ran on it")

    # Reload, and confirm the reloaded model's parameters are actually on the device.
    from transformers import AutoModelForSequenceClassification

    reloaded = AutoModelForSequenceClassification.from_pretrained(checkpoint).to(device)
    reloaded.eval()
    checks["reloaded_parameter_device"] = str(next(reloaded.parameters()).device)
    if device.startswith("cuda") and not checks["reloaded_parameter_device"].startswith("cuda"):
        raise SystemExit("the reloaded checkpoint is not on CUDA")

    # Batched independent inference, through the class the gate uses.
    #
    # Several DIFFERENT questions and several DIFFERENT contexts, not one context repeated.
    # The bug this is looking for is a batched path that pairs row i's text with row j's
    # context — a zip that silently truncates, a sort that reorders, a context cached
    # across the batch. One question repeated sixteen times cannot see any of those,
    # because every mispairing produces the same input. The gate calls
    # `score_independent` with a different context per row; so does this.
    detector = CrossEncoderAnswerabilityDetector(
        reloaded, tokenizer, model_repo_id=pins.model_repo_id, device=device, batch_size=8
    )
    distinct: list[dict] = []
    seen_questions: set[str] = set()
    for row in dev_rows:
        question_text = str(row["question"])
        if question_text in seen_questions:
            continue
        seen_questions.add(question_text)
        distinct.append(row)
        if len(distinct) >= 16:
            break
    if len(distinct) < 2:
        raise SystemExit(
            "the smoke needs at least two distinct development questions to test that "
            "batched scoring pairs each text with its own context; the pool has "
            f"{len(distinct)}"
        )

    pairs = []
    for i, row in enumerate(distinct):
        question = ProtectedQuestion(
            scope_id=f"smoke#{i:03d}",
            forget_id=f"smoke-{i:03d}",
            question=str(row["question"]),
            aliases=tuple(row["aliases"]),
            template_id="smoke",
        )
        context = build_context(
            str(row["question"]),
            protected_questions=[question],
            alias_index={question.forget_id: list(question.alias_token_sets)},
        )
        pairs.append((context, str(row["candidate"])))

    scores = detector.score_independent(pairs, restrict_to=None)
    checks["independent_batch_scored"] = len(scores)
    checks["n_distinct_questions"] = len(seen_questions)
    checks["n_distinct_contexts"] = len({id(c) for c, _t in pairs})
    checks["independent_batch_length_preserved"] = len(scores) == len(pairs)
    if len(scores) != len(pairs):
        raise SystemExit(
            f"score_independent returned {len(scores)} results for {len(pairs)} pairs; "
            "the batched path is dropping or duplicating rows"
        )

    # Association, not just length: rescoring one pair on its own must reproduce the
    # score it got inside the batch. A batch that pairs text i with context j has the
    # right length and the wrong answers, and only this check distinguishes the two.
    probe = len(pairs) - 1
    alone = detector.score_independent([pairs[probe]], restrict_to=None)
    batched_probability = float(scores[probe].answer_probability)
    alone_probability = float(alone[0].answer_probability)
    checks["probe_index"] = probe
    checks["probe_batched_answer_probability"] = batched_probability
    checks["probe_alone_answer_probability"] = alone_probability
    # fp16 batching is not bit-exact against a batch of one, so this is a tolerance and
    # not an equality. It is tight enough that a swapped context fails it.
    checks["independent_batch_association_preserved"] = (
        abs(batched_probability - alone_probability) <= 1e-3
    )
    if not checks["independent_batch_association_preserved"]:
        raise SystemExit(
            f"pair {probe} scored {batched_probability:.6f} inside the batch and "
            f"{alone_probability:.6f} on its own. The batched path is not pairing each "
            "text with its own context."
        )
    checks["forward_pairs"] = detector.stats()["forward_pairs"]
    checks["detector_parameter_device"] = detector.parameter_device()

    artifact = {
        "schema": "graph-detector-v4-2-gpu-smoke-v1",
        "reportable": False,
        "why_not_reportable": (
            "64 training rows and one epoch. This measures that the GPU path RUNS, not "
            "how well it performs. No number here may appear in a result."
        ),
        "requested_device": device,
        "checks": checks,
        "checkpoint": str(checkpoint),
        "checkpoint_hashes": checkpoint_hashes(checkpoint),
        "versions": _versions(),
    }
    path = smoke_dir / "DETECTOR_V4_2_GPU_SMOKE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print("SMOKE PASSED (non-reportable)")
    return 0


# -------------------------------------------------------------------------- main --


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=V4)
    parser.add_argument("--v4-1-dir", type=Path, default=V4_1)
    parser.add_argument("--v4-2-dir", type=Path, default=V4_2)
    parser.add_argument(
        "--v4-3-bundle",
        type=Path,
        default=None,
        help=(
            "DETECTOR_V4_3_PAIR_BUNDLE.json. When given, the natural rows come from the "
            "bundle -- all-row aliases, group-disjoint splits, identical input shape for "
            "both populations -- instead of from natural_alias_index and the audit_id "
            "parity halving. This is the v4.3 path and is what a reportable run uses."
        ),
    )
    parser.add_argument(
        "--v4-3-label-report",
        type=Path,
        default=V4_3 / "DETECTOR_V4_3_LABEL_REPORT.json",
        help=(
            "the v4.3 label authority. A reportable --v4-3-bundle run requires it to have "
            "passed every label gate and to name the bundle it labelled."
        ),
    )
    parser.add_argument(
        "--v4-3-eval-key",
        type=Path,
        default=V4_3 / "PROTECTED_STORE_EVAL_KEY.json",
        help=(
            "the sealed key. Read ONLY for `population`, as a metric denominator; it never "
            "reaches the encoder. Without it the retain diagnostics have no denominator."
        ),
    )
    parser.add_argument(
        "--v4-4-bundle",
        type=Path,
        default=None,
        help=(
            "DETECTOR_V4_4_PAIR_BUNDLE.json, the IMMUTABLE unlabelled freeze. It is opened "
            "read-only: the labels are joined in memory from the authority's adjudicated "
            "file. Rebuilding this file with labels would move the hash the 600-row "
            "calibration panel is frozen against."
        ),
    )
    parser.add_argument(
        "--v4-4-label-authority",
        type=Path,
        default=V4_4 / "DETECTOR_V4_4_LABEL_AUTHORITY.json",
        help=(
            "the v4.4 label authority. Must be green, bound to --v4-4-bundle's hash, and "
            "carry a label for every bundle row exactly once."
        ),
    )
    parser.add_argument(
        "--v4-4-eval-key",
        type=Path,
        default=V4_3 / "PROTECTED_STORE_EVAL_KEY.json",
        help=(
            "the sealed key. Read ONLY for `population`, as a metric denominator; it never "
            "reaches the encoder. v4.4 reuses v4.3's key because it describes the same "
            "protected store."
        ),
    )
    parser.add_argument(
        "--policy-cohort",
        type=Path,
        default=REPO / "data" / "cohorts" / "graph_unlearning_v1" / "discovery.json",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO / "runs" / "detector_v4")
    parser.add_argument("--model-repo-id", default=TrainingPins.model_repo_id)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--tokenizer-revision", default="")
    parser.add_argument("--baseline-repo-id", default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--baseline-revision", default="")
    parser.add_argument("--baseline-entailment-label", default="")
    parser.add_argument("--extra-train", type=Path, nargs="*", default=[])
    parser.add_argument("--epochs", type=int, default=TrainingPins.num_train_epochs)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=list(PREREGISTERED_SEEDS),
        help="preregistered; all are trained and all are reported",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="non-reportable CUDA check: 64 rows, one epoch, batched inference verified",
    )
    parser.add_argument("--smoke-train-rows", type=int, default=128)
    parser.add_argument("--smoke-dev-rows", type=int, default=64)
    # Reportable by default, because the default has to be the strict one. A
    # non-reportable run is allowed to leave the preregistration and is marked as having
    # done so; the checkpoint it selects is refused by every downstream gate.
    parser.add_argument(
        "--reportable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enforce the preregistered seeds, epochs and model pins (default: enforced)",
    )
    parser.add_argument(
        "--declare-extra-train",
        action="store_true",
        help="acknowledge that --extra-train changes what the model learned; hashes are "
        "recorded either way",
    )
    args = parser.parse_args()

    device_name = require_cuda(args.device)

    if not args.model_revision or not args.tokenizer_revision:
        raise SystemExit(
            "--model-revision and --tokenizer-revision are both required. A moved model "
            "tag changes the weights; a moved tokenizer tag changes the subword split, "
            "which changes every score at a fixed threshold. Record both even when they "
            "are equal."
        )
    pins = TrainingPins(
        model_repo_id=args.model_repo_id,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        seeds=tuple(args.seeds),
        num_train_epochs=args.epochs,
    )
    # Before the tokenizer is downloaded and before anything is encoded: a run that is
    # going to be refused should be refused in the first second, not after the corpus is
    # built on a rented box.
    preregistration = enforce_preregistration(args, pins)

    from transformers import AutoTokenizer

    from rdl.defenses.cross_encoder_answerability import EncodingBudget

    budget = EncodingBudget(
        max_length=pins.max_length,
        min_question_tokens=pins.min_question_tokens,
        max_aliases=pins.max_aliases,
    )
    tokenizer = AutoTokenizer.from_pretrained(pins.model_repo_id, revision=pins.tokenizer_revision)

    dataset = json.loads((args.data_dir / "DETECTOR_V4_DATASET.json").read_text(encoding="utf-8"))
    alias_index = natural_alias_index(args.policy_cohort, pins.max_aliases)

    # ------------------------------------------------------------------- smoke --
    # Before the label gate, deliberately: the smoke mode is what you run on a freshly
    # rented box to find out whether CUDA works, and it must not require labels to exist.
    # It therefore runs on the synthetic rows alone — the natural rows ARE the labels.
    if args.smoke:
        return run_smoke(
            args,
            pins,
            budget,
            tokenizer,
            [*synthetic_examples(dataset, "train"), *extra_examples(list(args.extra_train))],
            synthetic_examples(dataset, "development"),
        )

    # The gate runs BEFORE the natural rows are read, because it is what decides WHICH
    # adjudicated file they come from. Reading them first and gating afterwards is how a
    # run trained on the v4.1 human labels while its manifest named the v4.2 model report.
    if args.v4_4_bundle is not None:
        if args.v4_3_bundle is not None:
            raise SystemExit(
                "--v4-3-bundle and --v4-4-bundle are two different labelled populations "
                "under two different prompt versions. Pass one."
            )
        # The v4.4 path. The bundle is never rewritten; the authority binds its hash and
        # the labels join in memory by audit_id.
        authority = require_v4_4_label_authority(
            Path(args.v4_4_label_authority), Path(args.v4_4_bundle)
        )
        labels_path = Path(str(authority["adjudicated_file"]))
        natural_train, natural_train_meta = v4_4_bundle_examples(
            args.v4_4_bundle, "train", labels_path=labels_path, eval_key=args.v4_4_eval_key
        )
        natural_dev, natural_dev_meta = v4_4_bundle_examples(
            args.v4_4_bundle,
            "development",
            labels_path=labels_path,
            eval_key=args.v4_4_eval_key,
        )
        if not natural_train and not natural_dev:
            raise SystemExit(
                f"{args.v4_4_bundle} contributed no labelled rows even though its authority "
                "verified. That means the bundle's audit_ids and the adjudicated file's "
                "audit_ids do not overlap on any split."
            )
    elif args.v4_3_bundle is not None:
        # The v4.3 path. Its authority is the v4.3 label report, bound to the bundle it
        # labelled -- not v4.1's human report and not v4.2's hosted-judge report, neither
        # of which describes these rows. Requiring one of those here is what made a
        # reportable v4.3 run impossible: it would exit before ever reading the bundle.
        authority = require_v4_3_label_authority(
            Path(args.v4_3_label_report), Path(args.v4_3_bundle)
        )
        labels_path = Path(str(authority["adjudicated_file"]))
        natural_train, natural_train_meta = v4_3_bundle_examples(
            args.v4_3_bundle, "train", eval_key=args.v4_3_eval_key
        )
        natural_dev, natural_dev_meta = v4_3_bundle_examples(
            args.v4_3_bundle, "development", eval_key=args.v4_3_eval_key
        )
        if not natural_train and not natural_dev:
            raise SystemExit(
                f"{args.v4_3_bundle} contributed no labelled rows. A bundle built before "
                "the local judges ran carries `label: null` on every pair; label it with "
                "`rdl graph-detector-v4-3-bundle --labels <adjudicated.jsonl>` first."
            )
    else:
        authority = require_label_audit(args.v4_1_dir, args.v4_2_dir)
        labels_path = Path(str(authority["adjudicated_file"]))
        natural_train, natural_train_meta = natural_examples(
            args.v4_1_dir,
            args.v4_2_dir,
            "train",
            alias_index=alias_index,
            adjudicated_path=labels_path,
        )
        natural_dev, natural_dev_meta = natural_examples(
            args.v4_1_dir,
            args.v4_2_dir,
            "development",
            alias_index=alias_index,
            adjudicated_path=labels_path,
        )
    train_rows = [
        *synthetic_examples(dataset, "train"),
        *natural_train,
        *extra_examples(list(args.extra_train)),
    ]
    dev_rows = [*synthetic_examples(dataset, "development"), *natural_dev]
    # The synthetic held-out split is opened by `rdl graph-detector-v4-gates` after the
    # threshold is frozen, and neither gate bank is a file this script names. Checked
    # rather than commented: a filter that silently stopped filtering would be invisible
    # in the manifest, which is where the guarantee is supposed to be readable.
    if any(r.get("split") == "heldout" for r in [*train_rows, *dev_rows]):
        raise SystemExit("a held-out row reached the training or development pool")
    if not train_rows:
        raise SystemExit("no training rows; check --data-dir")

    baseline = {"skipped": True}
    if not args.skip_baseline:
        if not args.baseline_revision:
            raise SystemExit(
                "--baseline-revision is required (or pass --skip-baseline). An unpinned "
                "baseline is a number nobody can reproduce."
            )
        baseline = baseline_zero_shot(
            args.baseline_repo_id,
            args.baseline_revision,
            dev_rows,
            budget,
            args.device,
            entailment_label=args.baseline_entailment_label,
        )
        print(f"baseline AUC (ANSWER vs NONE): {baseline['answer_vs_none_auc']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The manifest names the protocol the LABELS came from, not the protocol this script
    # was first written for. A v4.4 run whose manifest cites the v4.2 judge protocol would
    # send a reader to a rubric that no longer describes any label in the training set.
    is_v4_4 = str(authority.get("protocol_phase")) == "v4.4"
    manifest: dict = {
        "schema": "graph-detector-v4-4-model-v1" if is_v4_4 else "graph-detector-v4-model-v3",
        "phase": (
            "Detector-v4.4 — answer-attempt cross-encoder fine-tune"
            if is_v4_4
            else "Detector-v4.2 Phase 7 — cross-encoder fine-tune"
        ),
        "protocol": (
            "docs/graph_unlearning/DETECTOR_V4_4_ANSWER_ATTEMPT_PROTOCOL.md"
            if is_v4_4
            else "docs/graph_unlearning/DETECTOR_V4_2_LLM_JUDGE_PROTOCOL.md"
        ),
        "device_requested": args.device,
        "device_name": device_name,
        "pins": pins.to_dict(),
        "encoding_budget": budget.to_dict(),
        "dataset_content_sha256": dataset["content_sha256"],
        "reportable": bool(args.reportable),
        "preregistration": preregistration,
        "label_authority": authority,
        "human_grounded": bool(authority.get("human_grounded")),
        "publication_label_valid": bool(authority.get("publication_label_valid")),
        "n_train_rows": len(train_rows),
        "n_development_rows": len(dev_rows),
        "train_sources": _counts(train_rows, "source"),
        "development_sources": _counts(dev_rows, "source"),
        "train_label_distribution": _counts(train_rows, "label"),
        "natural_rows": {"train": natural_train_meta, "development": natural_dev_meta},
        "heldout_read_during_training": False,
        "gate_bank_read_during_training": False,
        "final_gate_bank_read_during_training": False,
        "labels": list(LABELS),
        "baseline_zero_shot": baseline,
        "baseline_only": bool(args.baseline_only),
        "checkpoint_selection_rule": CHECKPOINT_SELECTION_RULE,
        "development_fpr_ceiling": DEV_FPR_CEILING,
        "versions": _versions(),
        # Which revision of this repository produced the numbers, and whether the tree was
        # clean. Package versions alone cannot answer either question.
        "git": preregistration["git"],
    }

    if args.baseline_only:
        _write_manifest(args.output_dir, manifest)
        print("baseline recorded; no fine-tuning requested")
        return 0

    seed_results = [
        train_one_seed(
            seed=seed,
            pins=pins,
            train_rows=train_rows,
            dev_rows=dev_rows,
            tokenizer=tokenizer,
            budget=budget,
            output_dir=args.output_dir,
            device=args.device,
            epochs=args.epochs,
        )
        for seed in pins.seeds
    ]
    selection = select_checkpoint(seed_results)
    best = selection.get("selected")

    # Re-hash the selected checkpoint NOW, from disk, rather than copying the hashes
    # computed when it was saved. Three seeds x three epochs write nine directories; a
    # crash, a full disk or a later run can leave the selected one different from what it
    # was at save time, and the manifest is what every downstream gate verifies against.
    selected_hashes = None
    if best:
        selected_hashes = verify_checkpoint_hashes(
            best["checkpoint_hashes"], Path(best["checkpoint"])
        )

    manifest.update(
        {
            "training": {"seeds": seed_results},
            "selection": selection,
            "selected_checkpoint": best["checkpoint"] if best else None,
            "selected_seed": best["seed"] if best else None,
            "selected_epoch": best["epoch"] if best else None,
            "selected_checkpoint_hashes": selected_hashes,
            "selected_checkpoint_hashes_reverified_after_training": bool(best),
            "development_selection_answer_recall": (
                best["development"]["selection_answer_recall"] if best else None
            ),
            "selected_on": CHECKPOINT_SELECTION_RULE,
            "truncation_why_counted": (
                "a candidate that did not fit the window is an evaluation error, not a "
                "NONE. v4 truncated the candidate by construction; this counts what the "
                "corrected budget still has to cut."
            ),
        }
    )
    _write_manifest(args.output_dir, manifest)
    if not best:
        print(f"NO CHECKPOINT SELECTED: {selection['reason']}")
        return 1
    print(
        f"selected {best['checkpoint']} "
        f"(seed {best['seed']}, epoch {best['epoch']}, "
        f"dev ANSWER recall {best['development']['selection_answer_recall']})"
    )
    print(
        "next: rdl graph-detector-v4-gates --backend cross_encoder --device "
        f"{args.device} --model-artifact {args.output_dir / 'DETECTOR_V4_MODEL.json'}"
    )
    return 0


def _counts(rows: list[dict], field: str) -> dict:
    out: dict[str, int] = {}
    for row in rows:
        out[str(row[field])] = out.get(str(row[field]), 0) + 1
    return dict(sorted(out.items()))


def _write_manifest(output_dir: Path, manifest: dict) -> None:
    path = output_dir / "DETECTOR_V4_MODEL.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _versions() -> dict:
    import importlib

    out: dict[str, str] = {}
    for name in ("torch", "transformers", "tokenizers", "numpy"):
        try:
            out[name] = importlib.import_module(name).__version__
        except Exception:  # pragma: no cover - reporting, not control flow
            out[name] = "absent"
    try:
        import torch

        out["cuda"] = torch.version.cuda or "none"
        out["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        out["cuda"] = "absent"
    return out


if __name__ == "__main__":
    raise SystemExit(main())
