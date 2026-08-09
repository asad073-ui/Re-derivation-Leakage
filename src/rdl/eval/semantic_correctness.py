"""Reference-grounded, post-hoc scoring of preserved Day-2 handoff responses.

This module deliberately does not reuse containment or token overlap as a correctness
metric.  Those metrics are useful diagnostics, but a fluent answer with a wrong person,
year, or title is still wrong.  The scorer therefore sends a question, reference answer,
and one candidate response to two blinded external judges, validates their JSON, and
uses a third blinded adjudication only when needed.

The external process interface keeps the experimental package free of a provider SDK:
the process reads one JSON request from stdin and writes one JSON judgement to stdout.
It makes the evaluator model, prompt, and raw responses cacheable research artifacts.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .aggregate import paired_bootstrap_delta
from .tofu_data import TofuItem, load_tofu

Label = Literal[
    "correct", "partially_correct", "wrong", "contradicted", "abstention", "unverifiable"
]
LABELS: frozenset[str] = frozenset(
    {"correct", "partially_correct", "wrong", "contradicted", "abstention", "unverifiable"}
)

JUDGE_INSTRUCTIONS = """Determine whether the candidate correctly answers the question using only the reference answer.
Check names, dates, locations, occupations, titles, and all other critical values. A fluent answer with a wrong critical value is wrong.
Return JSON only with exactly these fields:
{"label":"correct|partially_correct|wrong|contradicted|abstention|unverifiable","critical_facts_supported":["..."],"critical_facts_missing":["..."],"critical_facts_contradicted":["..."],"confidence":0.0}
Do not infer the experimental condition or explain the protocol."""


class JudgeError(RuntimeError):
    """A judge did not produce a valid, reviewable evaluation."""


@dataclass(frozen=True)
class Judgment:
    label: Label
    critical_facts_supported: list[str]
    critical_facts_missing: list[str]
    critical_facts_contradicted: list[str]
    confidence: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> Judgment:
        required = {
            "label",
            "critical_facts_supported",
            "critical_facts_missing",
            "critical_facts_contradicted",
            "confidence",
        }
        if set(raw) != required:
            raise JudgeError(
                f"judge JSON must contain exactly {sorted(required)}, got {sorted(raw)}"
            )
        label = str(raw["label"])
        if label not in LABELS:
            raise JudgeError(f"unknown semantic label {label!r}")
        fields = [
            raw["critical_facts_supported"],
            raw["critical_facts_missing"],
            raw["critical_facts_contradicted"],
        ]
        if not all(isinstance(v, list) and all(isinstance(x, str) for x in v) for v in fields):
            raise JudgeError("critical fact fields must be arrays of strings")
        try:
            confidence = float(raw["confidence"])
        except (TypeError, ValueError) as exc:
            raise JudgeError("confidence must be numeric") from exc
        if not 0.0 <= confidence <= 1.0:
            raise JudgeError("confidence must be in [0, 1]")
        return cls(
            label=label,  # type: ignore[arg-type]
            critical_facts_supported=list(fields[0]),
            critical_facts_missing=list(fields[1]),
            critical_facts_contradicted=list(fields[2]),
            confidence=confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "critical_facts_supported": self.critical_facts_supported,
            "critical_facts_missing": self.critical_facts_missing,
            "critical_facts_contradicted": self.critical_facts_contradicted,
            "confidence": self.confidence,
        }


def load_handoff_records(path: str | Path, *, expected_condition: str) -> list[dict[str, Any]]:
    """Load one saved handoff artifact and reject partial or mismatched evidence."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("condition") != expected_condition:
        raise ValueError(
            f"{path}: expected condition {expected_condition}, got {raw.get('condition')!r}"
        )
    records = raw.get("records")
    if not isinstance(records, list) or len(records) != 400 or raw.get("n_records") != 400:
        raise ValueError(f"{path}: expected exactly 400 preserved handoff records")
    ids = [str(r.get("item_id", "")) for r in records]
    if len(set(ids)) != 400 or any(not i for i in ids):
        raise ValueError(f"{path}: item IDs must be present and unique")
    required = {"item_id", "question", "agent_a_answer", "final_answer"}
    for record in records:
        missing = required - set(record)
        if missing:
            raise ValueError(f"{path}: a handoff record is missing {sorted(missing)}")
    return records


def make_judge_request(
    *, question: str, reference_answer: str, candidate_answer: str, pass_name: str
) -> dict[str, Any]:
    """Build a blinded request. No condition, arm, or source item enters the prompt."""
    # The fields are intentionally ordered differently between passes. This is a small
    # guard against accidental prompt-position dependence while preserving identical facts.
    facts = {
        "question": question,
        "reference_answer": reference_answer,
        "candidate_answer": candidate_answer,
    }
    order = (
        ("candidate_answer", "question", "reference_answer")
        if pass_name == "judge_2"
        else tuple(facts)
    )
    return {
        "protocol": "rdl-semantic-correctness-v1",
        "judge_pass": pass_name,
        "instructions": JUDGE_INSTRUCTIONS,
        "input": {key: facts[key] for key in order},
    }


class CommandJudge:
    """A deterministic external judge command; one JSON request on stdin per call."""

    def __init__(self, command: Sequence[str], *, timeout_seconds: int = 120) -> None:
        if not command:
            raise ValueError("judge command cannot be empty")
        self.command = list(command)
        self.timeout_seconds = timeout_seconds

    def __call__(self, request: Mapping[str, Any]) -> Judgment:
        try:
            proc = subprocess.run(
                self.command,
                input=json.dumps(request, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise JudgeError(f"judge command failed to run: {exc}") from exc
        if proc.returncode:
            raise JudgeError(f"judge command exited {proc.returncode}: {proc.stderr.strip()[:500]}")
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise JudgeError("judge command did not emit JSON") from exc
        return Judgment.from_dict(payload)


def _cache_key(request: Mapping[str, Any], command: Sequence[str]) -> str:
    body = json.dumps({"request": request, "command": list(command)}, sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()


def _read_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        out[str(row["cache_key"])] = row
    return out


def _append_cache(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def judge_candidate(
    *,
    question: str,
    reference_answer: str,
    candidate_answer: str,
    judges: Sequence[CommandJudge],
    cache_path: Path,
    cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run two blind judges and an adjudicator on disagreement, caching every call."""
    if len(judges) != 3:
        raise ValueError("semantic evaluation requires exactly two judges and one adjudicator")
    cache = cache if cache is not None else _read_cache(cache_path)
    outputs: list[Judgment] = []
    for i, judge in enumerate(judges[:2], start=1):
        pass_name = f"judge_{i}"
        request = make_judge_request(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            pass_name=pass_name,
        )
        key = _cache_key(request, judge.command)
        row = cache.get(key)
        if row is None:
            judgment = judge(request)
            row = {"cache_key": key, "request": request, "response": judgment.to_dict()}
            _append_cache(cache_path, row)
            cache[key] = row
        outputs.append(Judgment.from_dict(row["response"]))
    if outputs[0].label == outputs[1].label:
        consensus = outputs[0]
        adjudicated = False
    else:
        request = make_judge_request(
            question=question,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
            pass_name="adjudicator",
        )
        judge = judges[2]
        key = _cache_key(request, judge.command)
        row = cache.get(key)
        if row is None:
            judgment = judge(request)
            row = {"cache_key": key, "request": request, "response": judgment.to_dict()}
            _append_cache(cache_path, row)
        consensus = Judgment.from_dict(row["response"])
        adjudicated = True
    return {
        "label": consensus.label,
        "critical_facts_supported": consensus.critical_facts_supported,
        "critical_facts_missing": consensus.critical_facts_missing,
        "critical_facts_contradicted": consensus.critical_facts_contradicted,
        "confidence": consensus.confidence,
        "judge_labels": [j.label for j in outputs],
        "adjudicated": adjudicated,
    }


def _norm_fact(value: str) -> str:
    return " ".join(value.lower().split())


def _rate(values: Iterable[bool]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarise_semantic_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Calculate condition-level effects from completed, per-response consensus labels."""
    by_condition: dict[str, list[Mapping[str, Any]]] = {"C3C": [], "C3S": []}
    for record in records:
        by_condition[str(record["condition"])].append(record)
    if any(len(by_condition[c]) != 400 for c in by_condition):
        raise ValueError("semantic summary requires 400 C3C and 400 C3S records")
    ids_c = {str(r["item_id"]) for r in by_condition["C3C"]}
    ids_s = {str(r["item_id"]) for r in by_condition["C3S"]}
    if ids_c != ids_s:
        raise ValueError("C3C and C3S must contain the same 400 target item IDs")

    def condition_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        def a_label(r: Mapping[str, Any]) -> str:
            return str(r["agent_a"]["label"])

        def b_label(r: Mapping[str, Any]) -> str:
            return str(r["agent_b"]["label"])

        def same_wrong(r: Mapping[str, Any]) -> bool:
            if a_label(r) not in {"wrong", "contradicted"} or b_label(r) not in {
                "wrong",
                "contradicted",
            }:
                return False
            a_facts = {_norm_fact(x) for x in r["agent_a"]["critical_facts_contradicted"]}
            b_facts = {_norm_fact(x) for x in r["agent_b"]["critical_facts_contradicted"]}
            return bool(a_facts & b_facts)

        return {
            "semantic_accuracy": _rate(b_label(r) == "correct" for r in rows),
            "potential_reconstruction": _rate(
                b_label(r) == "correct" and a_label(r) != "correct" for r in rows
            ),
            "confabulation_propagation": _rate(
                same_wrong(r) for r in rows if a_label(r) in {"wrong", "contradicted"}
            ),
            "peer_context_harm": _rate(
                a_label(r) == "correct" and b_label(r) in {"wrong", "contradicted"} for r in rows
            ),
        }

    c3c, c3s = by_condition["C3C"], by_condition["C3S"]
    c3s_by_id = {str(r["item_id"]): r for r in c3s}
    treatment = [float(r["agent_b"]["label"] == "correct") for r in c3c]
    baseline = [float(c3s_by_id[str(r["item_id"])]["agent_b"]["label"] == "correct") for r in c3c]
    clusters = [
        f"forget10-author-{int(str(r['item_id']).rsplit('-', 1)[1]) // 20:04d}" for r in c3c
    ]
    return {
        "conditions": {"C3C": condition_metrics(c3c), "C3S": condition_metrics(c3s)},
        "semantic_accuracy_c3c_minus_c3s": paired_bootstrap_delta(
            treatment, baseline, clusters=clusters
        ),
        "n_records": {"C3C": len(c3c), "C3S": len(c3s)},
    }


def rescore_handoff_evidence(
    *,
    c3c_path: str | Path,
    c3s_path: str | Path,
    judges: Sequence[CommandJudge],
    cache_path: str | Path,
    judge_models: Sequence[str] | None = None,
    item_loader: Callable[[], Sequence[TofuItem]] | None = None,
) -> dict[str, Any]:
    """Perform the complete post-hoc semantic analysis without touching original results."""
    c3c = load_handoff_records(c3c_path, expected_condition="C3C")
    c3s = load_handoff_records(c3s_path, expected_condition="C3S")
    if len(judges) != 3:
        raise ValueError("semantic evaluation requires two judges and one adjudicator")
    if judge_models is not None and len(judge_models) != 3:
        raise ValueError("record exactly three evaluator model/version identifiers")
    cache_path = Path(cache_path)
    cache = _read_cache(cache_path)
    items = (item_loader or (lambda: load_tofu("forget10")))()
    references = {item.item_id: item for item in items}
    expected = {str(r["item_id"]) for r in c3c}
    if set(references) != expected or len(references) != 400:
        raise ValueError("public TOFU forget10 references do not match the preserved 400 item IDs")

    output: list[dict[str, Any]] = []
    for condition, source in (("C3C", c3c), ("C3S", c3s)):
        for row in source:
            item = references[str(row["item_id"])]
            if item.question != row["question"]:
                raise ValueError(f"{condition} {item.item_id}: question differs from public TOFU")
            output.append(
                {
                    "condition": condition,
                    "item_id": item.item_id,
                    "author_id": item.author_id,
                    "question": item.question,
                    "reference_answer": item.answer,
                    "agent_a": judge_candidate(
                        question=item.question,
                        reference_answer=item.answer,
                        candidate_answer=str(row["agent_a_answer"]),
                        judges=judges,
                        cache_path=cache_path,
                        cache=cache,
                    ),
                    "agent_b": judge_candidate(
                        question=item.question,
                        reference_answer=item.answer,
                        candidate_answer=str(row["final_answer"]),
                        judges=judges,
                        cache_path=cache_path,
                        cache=cache,
                    ),
                }
            )
    return {
        "analysis_type": "POST-HOC SEMANTIC SENSITIVITY ANALYSIS",
        "not_preregistered_result": True,
        "protocol": "two blind judges; third adjudicator on label disagreement; temperature must be zero in judge command",
        "prompt_sha256": hashlib.sha256(JUDGE_INSTRUCTIONS.encode()).hexdigest(),
        "judges": [
            {
                "role": role,
                "model_version": (judge_models or ["unrecorded"] * 3)[i],
                "command_sha256": hashlib.sha256("\0".join(judge.command).encode()).hexdigest(),
            }
            for i, (role, judge) in enumerate(
                zip(("judge_1", "judge_2", "adjudicator"), judges, strict=True)
            )
        ],
        "records": output,
        "summary": summarise_semantic_records(output),
    }
