"""``rdl graph-detector-v4-1-freeze`` — write the v4.1 corrections, touching nothing frozen.

Three files, into ``data/cohorts/graph_unlearning_v1/detector_v4_1/``:

``DETECTOR_V4_1_CEILING_CORRECTION.json``
    The reinterpretation of ``DETECTOR_V4_ORACLE_CEILING.json``. That file is *not* edited
    — it is frozen evidence and its negative result stands. What changes is what it bounds:
    an exact answer-token-overlap rule bounds overlap rules, and the artifact's own numbers
    refuse the stronger reading (answer-free floor 0.081 micro recall, answer-aware
    "ceiling" 0.032, same gate half, similar clean FPR). The correction carries the
    original's content hash so it can never drift away from the file it corrects.

``V4_1_DECISION.json``
    D1–D6 of ``DETECTOR_V4_1_PROTOCOL.md``, and the marking that matters operationally:
    the v4 natural bank's held-out half is now ENGINEERING-ONLY. Both the oracle and the
    lexical detector have been run on it and both results are committed. A set that has
    been read is not a one-shot gate, whatever it is called.

``FINAL_GATE_BANK_MANIFEST.json``
    The fresh bank, pre-registered before it is generated: arms, protocol, seeds, sizes and
    the rule that it is opened exactly once. Written now, on CPU, so the seeds cannot be
    chosen after a threshold is known.

The command is idempotent and reads only committed evidence. It generates nothing, trains
nothing and opens no held-out data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import typer

from ..eval.detector_v4_1 import AUDIT_DECISION_GATE, CEILING_REINTERPRETATION, GOAL_A_GATES
from ..studies.graph_leak.evidence import atomic_json
from .detector_v4_data import DEFAULT_OUT as V4_DIR
from .detector_v4_data import NATURAL_BANK_FILENAME

__all__ = ["detector_v4_1_freeze"]

DEFAULT_V4_1_OUT = Path("data/cohorts/graph_unlearning_v1/detector_v4_1")
CEILING_FILENAME = "DETECTOR_V4_ORACLE_CEILING.json"
CORRECTION_FILENAME = "DETECTOR_V4_1_CEILING_CORRECTION.json"
DECISION_FILENAME = "V4_1_DECISION.json"
GATE_BANK_MANIFEST_FILENAME = "FINAL_GATE_BANK_MANIFEST.json"

# New, and new is the whole requirement. Re-using the study's 1729 would draw the same
# trajectories the engineering-only bank already contains, and a "fresh" gate bank that
# overlaps the set the detector was developed on is not a gate.
FINAL_GATE_BANK_BASE_SEED = 40241
FINAL_GATE_BANK_SEEDS = (40241, 40242, 40243, 40244)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ceiling_correction(v4_dir: Path) -> dict:
    path = v4_dir / CEILING_FILENAME
    original = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    natural = original.get("natural_arm", {}) or {}
    held = natural.get("heldout") or {}
    return {
        "schema": "graph-detector-v4-1-ceiling-correction-v1",
        "corrects": str(path),
        "corrects_file_sha256": _file_sha(path),
        "corrects_bank_content_sha256": natural.get("bank_content_sha256"),
        "original_verdict": original.get("verdict"),
        "original_is_edited": False,
        "renamed_to": "answer-token-overlap baseline",
        "must_not_be_called": "answerability ceiling",
        "reinterpretation": CEILING_REINTERPRETATION,
        "what_the_artifact_does_establish": (
            "exact normalised answer-token overlap cannot reproduce the run's NLI+ROUGE "
            "leak labels at generated_clean_fpr <= 0.10 on this cohort. That is a real "
            "negative result about a real rule and it is kept."
        ),
        "what_it_does_not_establish": (
            "that no semantic answerability model can. A cross-encoder recognises a "
            "paraphrase that shares no answer token, which is precisely the population an "
            "overlap rule cannot see."
        ),
        "the_internal_contradiction": {
            "answer_aware_oracle_micro_recall": held.get("micro_recall"),
            "answer_free_lexical_micro_recall": 0.08064516129032258,
            "note": (
                "same gate half, similar sub-0.10 clean FPR. The answer-FREE detector "
                "beat the answer-AWARE 'ceiling' by roughly 2.5x. An upper bound the "
                "bounded thing exceeds is a different measurement, not a bound."
            ),
            "answer_free_source": "DETECTOR_V4_GATES.json natural_arm.heldout.micro_recall",
        },
        "consequence": (
            "scripts/train_detector_v4.py no longer gates on this artifact and "
            "--force-despite-failed-ceiling is deleted. The readiness gate is now the "
            "v4.1 label audit, which measures the thing that actually bounds Goal A."
        ),
    }


def _decision(v4_dir: Path, v4_1_dir: Path) -> dict:
    bank_path = v4_dir / NATURAL_BANK_FILENAME
    bank = json.loads(bank_path.read_text(encoding="utf-8")) if bank_path.exists() else {}
    return {
        "schema": "graph-detector-v4-1-decision-v1",
        "protocol": "docs/graph_unlearning/DETECTOR_V4_1_PROTOCOL.md",
        "decisions": {
            "D1": CEILING_REINTERPRETATION,
            "D2": (
                "Goal A's primary label is answer_attempt in {NONE, PARTIAL, ANSWER}, "
                "assigned by human judges who do NOT see the reference answer."
            ),
            "D3": (
                "reference_content in {YES, NO, UNCERTAIN} is a SEPARATE secondary label, "
                "assigned with the reference answer visible. It is reported and is never "
                "the primary recall denominator."
            ),
            "D4": (
                "the existing v4 held-out data is ENGINEERING-ONLY. Both the oracle and "
                "the lexical detector have been run on it and both results are committed."
            ),
            "D5": (
                "the final detector gate requires a FRESH bank, generated after the model "
                "and threshold are frozen, under the seeds in "
                f"{GATE_BANK_MANIFEST_FILENAME}."
            ),
            "D6": "v1/v2/v3/v4 artifacts remain unchanged. Every correction is a new file.",
        },
        "engineering_only": {
            "artifact": str(bank_path),
            "bank_content_sha256": bank.get("content_sha256"),
            "partitions_marked": ["development", "heldout", "retain"],
            "why": (
                "a held-out set is a one-shot instrument. DETECTOR_V4_ORACLE_CEILING.json "
                "and DETECTOR_V4_GATES.json both report numbers measured on the heldout "
                "half and both are committed, so the half has been read. It remains "
                "useful engineering evidence and it may not gate a trained model."
            ),
            "still_usable_for": [
                "development and error analysis",
                "the v4.1 blinded label audit's sampling frame",
                "training-set construction, once the audit's labels exist",
            ],
            "not_usable_for": ["the final detector gate", "any reported one-shot number"],
        },
        "metrics": {
            "goal_a_gates": {
                k: {"comparison": c, "bound": b} for k, (c, b) in GOAL_A_GATES.items()
            },
            "audit_decision_gate": {
                k: {"comparison": c, "bound": b} for k, (c, b) in AUDIT_DECISION_GATE.items()
            },
            "replaced": {
                "generated_clean_fpr": "protected_nonanswer_fpr",
                "why": (
                    "generated_clean_fpr counted every wrong answer attempt as a false "
                    "alarm. Goal A requires those to be tagged, so the old quantity "
                    "penalised the detector for correct behaviour and then used that "
                    "penalty to select the operating point."
                ),
            },
        },
        "frozen_artifacts_untouched": {
            "v1_v2_v3": True,
            "v4": True,
            "checked": sorted(p.name for p in sorted(v4_dir.glob("*.json"))),
        },
        "outputs_live_in": str(v4_1_dir),
        "scope": {
            "model_trained": False,
            "graph_generation_run": False,
            "frozen_v1_v2_v3_artifacts_modified": False,
            "gpu_used": False,
        },
        "runtime_reads_gold_answers": False,
    }


def _gate_bank_manifest() -> dict:
    manifest = {
        "schema": "graph-detector-v4-1-final-gate-bank-manifest-v1",
        "status": "PRE-REGISTERED, NOT YET GENERATED",
        "purpose": (
            "the single surface on which the trained v4.1 detector's gate is opened. "
            "Pre-registered on CPU so the seeds cannot be chosen after a threshold is "
            "known, and disjoint from every set the detector was developed on."
        ),
        "generated_after": [
            "the blinded label audit clears DETECTOR_V4_1_PROTOCOL.md section 5",
            "the cross-encoder checkpoint is frozen",
            "the operating point is frozen on development",
        ],
        "arms": {
            "natural": "multi_agent_leak",
            "retain": "retain unguarded flow",
            "why_only_unguarded": (
                "a guarded arm's text is clean partly because the guard removed the rest "
                "of it, so its clean rate is not a detector measurement. Unchanged from "
                "the v2 corpus rule (GU-0031)."
            ),
            "not_run_at_this_stage": [
                "multi_agent_dragon",
                "multi_agent_dragon_subsets",
                "multi_agent_graphforget",
            ],
        },
        "protocol": "graph_flow",
        "challenge": "natural",
        "seeds": {
            "base_seed": FINAL_GATE_BANK_BASE_SEED,
            "seeds": list(FINAL_GATE_BANK_SEEDS),
            "why_new": (
                "the study's base_seed is 1729 and the engineering-only bank was drawn "
                "under it. A 'fresh' bank that redraws the same trajectories is not fresh."
            ),
        },
        "labelling": {
            "primary": "human answer_attempt, blinded, two judges, adjudicated",
            "secondary": "reference_content, with the reference answer visible",
            "automatic": (
                "the run's pinned NLI+ROUGE scorer is recorded for diagnosis and is not a "
                "Goal A denominator."
            ),
        },
        "opening_rule": {
            "opened": "exactly once, at the frozen threshold",
            "on_failure": (
                "the model is NOT integrated. Any new model iteration requires another "
                "fresh gate bank; re-opening this one turns it into development data."
            ),
            "artifacts_written_only_on_pass": [
                "DETECTOR_V4_MODEL.json",
                "DETECTOR_V4_CALIBRATION.json",
                "DETECTOR_V4_GATES.json",
            ],
        },
        "gates": {k: {"comparison": c, "bound": b} for k, (c, b) in GOAL_A_GATES.items()},
        "minimum_sizes": {
            "n_answer_attempt_rows": 150,
            "n_protected_nonanswer_rows": 400,
            "n_retain_rows": 400,
            "n_concepts_with_answer_rows": 5,
            "why": (
                "a gate measured on 62 positives across five concepts, as the v4 held-out "
                "half was, cannot separate 0.80 from 0.70 at any useful confidence."
            ),
        },
    }
    manifest["manifest_sha256"] = _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    return manifest


def detector_v4_1_freeze(
    v4_dir: Path = typer.Option(V4_DIR, "--v4-dir"),
    output_dir: Path = typer.Option(DEFAULT_V4_1_OUT, "--output-dir"),
) -> None:
    """Write the v4.1 correction, decision and fresh-gate-bank artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    correction = _ceiling_correction(v4_dir)
    atomic_json(output_dir / CORRECTION_FILENAME, correction)
    typer.echo(f"wrote {output_dir / CORRECTION_FILENAME}")

    atomic_json(output_dir / DECISION_FILENAME, _decision(v4_dir, output_dir))
    typer.echo(f"wrote {output_dir / DECISION_FILENAME}")

    atomic_json(output_dir / GATE_BANK_MANIFEST_FILENAME, _gate_bank_manifest())
    typer.echo(f"wrote {output_dir / GATE_BANK_MANIFEST_FILENAME}")

    typer.echo("")
    typer.echo("DETECTOR_V4_ORACLE_CEILING.json is an answer-token-overlap baseline.")
    typer.echo("The v4 held-out half is engineering-only. The final gate needs a fresh bank.")
