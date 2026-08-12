"""Cohort discipline: frozen hashes, exclusions, and the validation refusal."""

from __future__ import annotations

import json

import pytest

from rdl.paths import repo_root
from rdl.studies.graph_leak.cohort import (
    CohortError,
    load_cohort,
    load_exclusions,
    resolve_cohort,
    sha256_text,
)

COHORTS = repo_root() / "data" / "cohorts" / "graph_unlearning_v1"


def test_the_pilot_exclusions_cover_every_forget10_author():
    items, concepts = load_exclusions(COHORTS / "exclusions.json")
    assert len(items) == 50
    assert len(concepts) == 20
    assert "forget10-0000" in items and "forget10-0392" in items


def test_discovery_does_not_overlap_the_pilot():
    cohort = load_cohort(
        COHORTS / "discovery.json",
        exclusions_path=COHORTS / "exclusions.json",
        require_frozen=False,
    )
    assert len(cohort.items) == 50


def test_discovery_is_labelled_discovery_not_validation():
    payload = json.loads((COHORTS / "discovery.json").read_text(encoding="utf-8"))
    assert "NOT VALIDATION" in payload["note"]
    assert all(i["usage"] == "discovery" for i in payload["items"])
    # New questions, but the SAME 20 forgotten concepts the pilot already covered.
    assert len({i["concept_id"] for i in payload["items"]}) == 20


def test_the_splits_are_pairwise_disjoint():
    sets = {}
    for split in ("smoke", "engineering", "discovery"):
        cohort = load_cohort(COHORTS / f"{split}.json", require_frozen=False)
        sets[split] = set(cohort.item_ids)
    names = sorted(sets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert not (sets[a] & sets[b]), f"{a} and {b} overlap"


def test_validation_refuses_to_load_on_this_checkpoint():
    with pytest.raises(CohortError, match="declared unavailable"):
        load_cohort(COHORTS / "validation.json", require_frozen=False)


def test_every_shipped_real_cohort_is_frozen_and_revision_pinned():
    """The state the GPU gate requires: content hashes AND the dataset commit."""
    for split in ("smoke", "engineering", "discovery", "retain_utility"):
        cohort = load_cohort(COHORTS / f"{split}.json", require_frozen=True)
        assert cohort.frozen, split
        assert cohort.dataset_revision, split
        assert all(i.question_sha256 and i.answer_sha256 for i in cohort.items), split


def test_all_real_cohorts_share_one_dataset_revision():
    revisions = {
        load_cohort(COHORTS / f"{s}.json").dataset_revision
        for s in ("smoke", "engineering", "discovery", "retain_utility")
    }
    assert len(revisions) == 1, f"cohorts frozen against different commits: {revisions}"


def test_an_unfrozen_cohort_is_refused_when_hashes_are_required(tmp_path):
    payload = json.loads((COHORTS / "smoke.json").read_text(encoding="utf-8"))
    for item in payload["items"]:
        item["question_sha256"] = None
        item["answer_sha256"] = None
    path = tmp_path / "unfrozen.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CohortError, match="not frozen"):
        load_cohort(path, require_frozen=True)


def test_a_frozen_cohort_without_a_dataset_revision_is_refused(tmp_path):
    """Content hashes catch a changed question only AFTER the download.

    The revision is what makes the download itself reproducible, so a frozen cohort
    that omits it is refused rather than trusted.
    """
    payload = json.loads((COHORTS / "smoke.json").read_text(encoding="utf-8"))
    payload["dataset_revision"] = None
    path = tmp_path / "unpinned.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CohortError, match="no dataset_revision"):
        load_cohort(path, require_frozen=True)


def test_the_fixture_cohort_is_exempt_from_the_revision_requirement():
    """It is a checked-in file, so its 'revision' is the path, and it loads."""
    cohort = load_cohort(COHORTS / "cpu_stub.json", require_frozen=True)
    assert cohort.dataset == "fixture"


def test_the_retain_cohort_is_author_balanced():
    cohort = load_cohort(COHORTS / "retain_utility.json")
    assert cohort.dataset_config == "retain90"
    # One question per author, so items and concepts are the same count.
    assert len(cohort.items) == len(cohort.concept_ids) == 45
    assert all(i.usage == "retain_utility" for i in cohort.items)


def test_the_cpu_stub_cohort_is_frozen_and_loads():
    cohort = load_cohort(COHORTS / "cpu_stub.json", require_frozen=True)
    assert cohort.frozen
    assert cohort.dataset == "fixture"
    assert len(cohort.items) == 8


def test_resolving_verifies_every_content_hash(tofu_items):
    cohort = load_cohort(COHORTS / "cpu_stub.json", require_frozen=True)
    items = resolve_cohort(cohort, tofu_items)
    assert len(items) == len(cohort.items)
    for entry, item in zip(cohort.items, items, strict=True):
        assert sha256_text(item.question) == entry.question_sha256


def test_a_changed_question_fails_resolution(tofu_items):
    cohort = load_cohort(COHORTS / "cpu_stub.json", require_frozen=True)
    tampered = list(tofu_items)
    tampered[0] = type(tampered[0])(
        item_id=tampered[0].item_id,
        question="a different question entirely",
        answer=tampered[0].answer,
        split=tampered[0].split,
        index=tampered[0].index,
    )
    with pytest.raises(CohortError, match="does not match the frozen hash"):
        resolve_cohort(cohort, tampered)


def test_a_missing_item_fails_resolution(tofu_items):
    cohort = load_cohort(COHORTS / "cpu_stub.json", require_frozen=True)
    with pytest.raises(CohortError, match="absent from the loaded dataset"):
        resolve_cohort(cohort, tofu_items[:2])


def test_cohort_fingerprint_is_stable():
    a = load_cohort(COHORTS / "cpu_stub.json")
    b = load_cohort(COHORTS / "cpu_stub.json")
    assert a.fingerprint() == b.fingerprint()
    assert a.limited(4).fingerprint() != a.fingerprint()
