"""Frozen cohorts, exclusions, and the split discipline.

The previous 50-item pilot took a spread sample of ``forget10``, which covers all 20
forget10 authors. So a *further* 50 non-overlapping questions are new questions but not
new forgotten concepts, and they cannot serve as an untouched validation set. The
manifests encode that distinction:

``exclusions.json``   every item and concept already touched by earlier work.
``smoke.json``        four items, for the wiring check.
``engineering.json``  20 items for the 3090 discovery pass.
``discovery.json``    50 non-overlapping items. Discovery, explicitly not validation.
``validation.json``   reserved for genuinely untouched concepts. Refuses to load while
                      its concepts intersect the exclusion list, which — with the current
                      checkpoint — they always will. Confirming the result needs a new
                      preregistered unlearning checkpoint, another dataset, or a new
                      model/method combination with a frozen concept split.

Hashes over question and answer text are recorded so the loader can prove the frozen
cohort is the one that ran. The answers are hashed, never stored.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ...eval.tofu_data import TofuItem

__all__ = [
    "Cohort",
    "CohortError",
    "CohortItem",
    "assert_forget_policy_cohort",
    "assert_policy_excludes_evaluation_concepts",
    "cohort_dir",
    "load_cohort",
    "load_exclusions",
    "resolve_cohort",
    "sha256_text",
]

SCHEMA = "graph-cohort-v1"


class CohortError(ValueError):
    """Raised for any malformed, unfrozen, or split-violating cohort."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cohort_dir(root: Path, relative: str) -> Path:
    return root / relative


@dataclass(frozen=True)
class CohortItem:
    item_id: str
    concept_id: str
    index: int
    question_sha256: str | None = None
    answer_sha256: str | None = None
    usage: str = "discovery"

    @property
    def frozen(self) -> bool:
        return bool(self.question_sha256 and self.answer_sha256)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "concept_id": self.concept_id,
            "index": self.index,
            "question_sha256": self.question_sha256,
            "answer_sha256": self.answer_sha256,
            "usage": self.usage,
        }


@dataclass(frozen=True)
class Cohort:
    split: str
    dataset: str
    dataset_config: str
    dataset_revision: str | None
    items: tuple[CohortItem, ...]
    note: str = ""

    @property
    def item_ids(self) -> tuple[str, ...]:
        return tuple(i.item_id for i in self.items)

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return tuple(sorted({i.concept_id for i in self.items}))

    @property
    def frozen(self) -> bool:
        return all(i.frozen for i in self.items)

    @property
    def is_retain(self) -> bool:
        """Does this cohort hold questions the system is SUPPOSED to answer?

        Read off the split name and the dataset config rather than a hand-set flag,
        because the one thing that must never happen is a retain cohort being taken for
        a forget cohort through a manifest someone forgot to annotate. ``retain90`` and
        any split whose name contains ``retain`` answer yes.
        """
        return "retain" in self.split.lower() or "retain" in self.dataset_config.lower()

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "schema": SCHEMA,
                "split": self.split,
                "dataset": self.dataset,
                "dataset_config": self.dataset_config,
                "dataset_revision": self.dataset_revision,
                "items": [i.to_dict() for i in self.items],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "split": self.split,
            "dataset": self.dataset,
            "dataset_config": self.dataset_config,
            "dataset_revision": self.dataset_revision,
            "note": self.note,
            "n_items": len(self.items),
            "n_concepts": len(self.concept_ids),
            "hashes_frozen": self.frozen,
            "fingerprint": self.fingerprint(),
            "items": [i.to_dict() for i in self.items],
        }

    def limited(self, n: int | None) -> Cohort:
        if n is None or n >= len(self.items):
            return self
        return Cohort(
            split=self.split,
            dataset=self.dataset,
            dataset_config=self.dataset_config,
            dataset_revision=self.dataset_revision,
            items=self.items[:n],
            note=f"{self.note} [limited to {n} items]".strip(),
        )


def _read(path: Path) -> dict:
    if not path.exists():
        raise CohortError(f"cohort manifest not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise CohortError(f"{path}: top level must be an object")
    if payload.get("schema") != SCHEMA:
        raise CohortError(f"{path}: unknown schema {payload.get('schema')!r} (expected {SCHEMA})")
    return payload


def load_exclusions(path: Path) -> tuple[frozenset[str], frozenset[str]]:
    """``(excluded_item_ids, excluded_concept_ids)`` from ``exclusions.json``."""
    payload = _read(path)
    items = frozenset(str(x) for x in payload.get("item_ids", ()))
    concepts = frozenset(str(x) for x in payload.get("concept_ids", ()))
    return items, concepts


def load_cohort(
    path: Path,
    *,
    exclusions_path: Path | None = None,
    require_frozen: bool = True,
    forbid_excluded_items: bool = True,
    forbid_excluded_concepts: bool = False,
    disjoint_from: Sequence[Cohort] = (),
) -> Cohort:
    """Load one split, refusing every way it could silently be the wrong one."""
    payload = _read(path)
    if payload.get("status") == "unavailable":
        raise CohortError(
            f"{path}: this split is declared unavailable and holds no items.\n"
            f"{payload.get('note', '')}"
        )
    items = tuple(
        CohortItem(
            item_id=str(row["item_id"]),
            concept_id=str(row["concept_id"]),
            index=int(row["index"]),
            question_sha256=row.get("question_sha256"),
            answer_sha256=row.get("answer_sha256"),
            usage=str(row.get("usage", payload.get("split", "discovery"))),
        )
        for row in payload.get("items", ())
    )
    if not items:
        raise CohortError(f"{path}: cohort is empty")
    ids = [i.item_id for i in items]
    if len(set(ids)) != len(ids):
        raise CohortError(f"{path}: duplicate item ids")

    cohort = Cohort(
        split=str(payload["split"]),
        dataset=str(payload.get("dataset", "TOFU")),
        dataset_config=str(payload.get("dataset_config", "forget10")),
        dataset_revision=payload.get("dataset_revision"),
        items=items,
        note=str(payload.get("note", "")),
    )

    if require_frozen and not cohort.frozen:
        missing = [i.item_id for i in items if not i.frozen][:5]
        raise CohortError(
            f"{path}: cohort is not frozen — {missing} lack question/answer hashes. Run "
            "`rdl graph-freeze-cohort` against the pinned dataset revision before using "
            "this split for anything reportable."
        )
    # A frozen cohort over real data must name the dataset commit it was frozen against.
    # Content hashes alone would catch a changed question, but only after the download;
    # the revision is what makes the download itself reproducible.
    if require_frozen and cohort.dataset != "fixture" and not cohort.dataset_revision:
        raise CohortError(
            f"{path}: frozen cohort has no dataset_revision. Re-freeze it with\n"
            f"  rdl graph-freeze-cohort --manifest {path} --dataset-revision <sha> --write\n"
            "Recording a revision that was never passed to load_dataset is worse than "
            "recording none, and that is exactly what this check prevents."
        )

    # The exclusion list is about the real dataset. The development fixture reuses TOFU
    # item ids for eight invented questions, so applying TOFU exclusions to it would
    # block the CPU gate over a name collision rather than over shared content.
    if exclusions_path is not None and cohort.dataset != "fixture":
        excluded_items, excluded_concepts = load_exclusions(exclusions_path)
        if forbid_excluded_items:
            overlap = sorted(set(ids) & excluded_items)
            if overlap:
                raise CohortError(
                    f"{path}: {len(overlap)} items appear in the exclusion list "
                    f"(first: {overlap[:5]}). They were already used by earlier work."
                )
        if forbid_excluded_concepts:
            overlap_c = sorted(set(cohort.concept_ids) & excluded_concepts)
            if overlap_c:
                raise CohortError(
                    f"{path}: concepts {overlap_c[:5]} have already been touched, so this "
                    "split cannot serve as untouched concept-level validation. Final "
                    "confirmation needs a new preregistered unlearning checkpoint, another "
                    "dataset, or a new model/method combination with a frozen concept split."
                )

    for other in disjoint_from:
        shared_items = sorted(set(ids) & set(other.item_ids))
        if shared_items:
            raise CohortError(f"{path}: shares items {shared_items[:5]} with split '{other.split}'")
        if cohort.split == "validation":
            shared_concepts = sorted(set(cohort.concept_ids) & set(other.concept_ids))
            if shared_concepts:
                raise CohortError(
                    f"{path}: validation concepts {shared_concepts[:5]} also appear in "
                    f"split '{other.split}'. Detector thresholds selected on those "
                    "concepts would make the validation run a second discovery run."
                )
    return cohort


def assert_forget_policy_cohort(cohort: Cohort) -> Cohort:
    """The cohort that defines what was forgotten must be a FORGET cohort.

    The forget-policy cohort is what builds the runtime concept registry and the deleted
    baseline memory: every concept in it is a concept the system is required to withhold.
    Handing a retain cohort to that role registers questions the system is *supposed* to
    answer as forgotten, so the guard fires on correct behaviour, the retain-utility
    number measures over-blocking of concepts it just declared forbidden, and the
    detector's false-positive rate is computed against its own positives. Nothing
    downstream can recover from it, so it is refused here rather than reported.
    """
    if cohort.is_retain:
        raise CohortError(
            f"cohort '{cohort.split}' (dataset_config '{cohort.dataset_config}') is a "
            "RETAIN cohort and cannot be the forget-policy cohort. The concept registry "
            "and the deleted baseline memory define what the system must withhold; "
            "building them from retain questions would classify retained concepts as "
            "forgotten and make every retain-utility and false-positive number invalid.\n"
            "Pass the frozen forget cohort as the policy cohort and the retain cohort as "
            "the evaluation cohort — see `forget_policy_phase` in the launch file."
        )
    if not cohort.items:
        raise CohortError(f"forget-policy cohort '{cohort.split}' is empty")
    return cohort


def assert_policy_excludes_evaluation_concepts(policy: Cohort, evaluation: Cohort) -> None:
    """A retain evaluation cohort must share no concept with the forget policy.

    This is the check the registry bug would have had to survive. Retain90 and forget10
    are disjoint author sets by construction, so any overlap here means the wrong
    manifest was selected — and an overlapping author would be simultaneously
    'must withhold' and 'must answer', which is not a measurable condition.
    """
    if not evaluation.is_retain:
        return
    shared = sorted(set(policy.concept_ids) & set(evaluation.concept_ids))
    if shared:
        raise CohortError(
            f"retain evaluation cohort '{evaluation.split}' shares concepts {shared[:5]} "
            f"with the forget-policy cohort '{policy.split}'. Those concepts would be "
            "both forbidden and required in one run, and the resulting utility and "
            "false-positive rates would be uninterpretable."
        )


def resolve_cohort(
    cohort: Cohort, source_items: Sequence[TofuItem], *, verify_hashes: bool = True
) -> list[TofuItem]:
    """Attach real question/answer text to a frozen cohort, verifying every hash.

    A cohort is item ids and hashes; the text comes from the dataset. If the text a run
    loads does not hash to what the cohort froze, the run is on different data than the
    manifest claims and must not start.
    """
    by_id = {item.item_id: item for item in source_items}
    resolved: list[TofuItem] = []
    for entry in cohort.items:
        item = by_id.get(entry.item_id)
        if item is None:
            raise CohortError(
                f"cohort item '{entry.item_id}' is absent from the loaded dataset "
                f"({len(source_items)} items). Wrong split or a truncated load."
            )
        if verify_hashes and entry.frozen:
            if sha256_text(item.question) != entry.question_sha256:
                raise CohortError(f"{entry.item_id}: question text does not match the frozen hash")
            if sha256_text(item.answer) != entry.answer_sha256:
                raise CohortError(f"{entry.item_id}: answer text does not match the frozen hash")
        resolved.append(item)
    return resolved


def freeze_cohort(
    cohort: Cohort, source_items: Sequence[TofuItem], *, revision: str | None
) -> Cohort:
    """Fill in the content hashes from real data. Used by ``rdl graph-freeze-cohort``."""
    by_id = {item.item_id: item for item in source_items}
    items: list[CohortItem] = []
    for entry in cohort.items:
        item = by_id.get(entry.item_id)
        if item is None:
            raise CohortError(f"cannot freeze '{entry.item_id}': absent from the loaded dataset")
        items.append(
            CohortItem(
                item_id=entry.item_id,
                concept_id=entry.concept_id,
                index=entry.index,
                question_sha256=sha256_text(item.question),
                answer_sha256=sha256_text(item.answer),
                usage=entry.usage,
            )
        )
    return Cohort(
        split=cohort.split,
        dataset=cohort.dataset,
        dataset_config=cohort.dataset_config,
        dataset_revision=revision or cohort.dataset_revision,
        items=tuple(items),
        note=cohort.note,
    )
