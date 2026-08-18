"""What v4.4 must not change: the v4.3 record, and the sealed final-gate bank.

One table, read by both the unit test and ``scripts/v44_pipeline_dryrun.py``. Two copies of
a digest list drift, and the copy that drifts is always the one nobody ran.

Why these files are frozen
--------------------------
Two of them -- the conditioning index and the local judge pins -- have their hashes quoted
in ``DETECTOR_V4_3_PROTECTED_STORE_PROTOCOL.md``, so editing one would invalidate a
pre-registration rather than correct it.

The rest are frozen for a blunter reason. v4.4 exists *because* v4.3 failed, and its whole
argument rests on what v4.3's two blind passes actually contain: 69 and 17 ``NONE`` rows
against a gate requiring 200, 184 of 258 disagreements in the ``A=PARTIAL/B=ANSWER`` cell,
raw agreement 0.695 on open-ended questions. A failed experiment whose evidence was edited
afterwards is not evidence of anything, least of all of the diagnosis that justified the
repair.

The sealed bank is separate and stricter. Every other file here would merely be
*embarrassing* to change; the final-gate bank cannot be un-opened. It stays shut until the
detector phase and its human validation have both closed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

__all__ = [
    "MERGE_COMMIT",
    "SEALED_DIGESTS",
    "V4_3_FROZEN_DIGESTS",
    "changed_artifacts",
]

# The commit these digests were taken at: PR #52, "detector v4.3 GPU 2: the 1,019 blind
# labels, and the gate they fail".
MERGE_COMMIT = "2a1f7cecc9aa59ba2d8f4cbf1c598536099fd274"

V4_3_FROZEN_DIGESTS: dict[str, str] = {
    "detector_v4_3/DETECTOR_V4_3_PAIR_BUNDLE.json": (
        "6347af35110cbe71a6e49525f849c345ebada716d6387824a48fbca38f37d501"
    ),
    "detector_v4_3/DETECTOR_V4_3_CONDITIONING_INDEX.json": (
        "8fd1e55f519f27708c6cc86aef4a0ad5685180b4a8186f2b480cb826d3c02818"
    ),
    "detector_v4_3/DETECTOR_V4_3_LOCAL_JUDGE_PINS.json": (
        "c0116f3c5d23363edf2f5cd62bad1ab40890fbec530b2010b1c04e251a31a30b"
    ),
    "detector_v4_3/PROTECTED_STORE_RUNTIME.json": (
        "df7c9ad26e8a7410b7889dc0251cd75c72eb8726def9ce8a693ab7420f2087c3"
    ),
    "detector_v4_3/PROTECTED_STORE_EVAL_KEY.json": (
        "116abd7dddf3e974ac2d7e74ae45a99ee4a66f822684e81f3599c3841be265ed"
    ),
    "detector_v4_3/JUDGE_SMOKE_50.jsonl": (
        "d29aa7675429cd1f79c1cff89a10f1defae04bd028c95d02df9ce5e2854e632e"
    ),
    "detector_v4_3/judges/V4_3_LOCAL_JUDGE_A_BLIND.jsonl": (
        "1b7397a02fb8699e31676e082046bf6800afda01e63f8f2180b0865bcac39376"
    ),
    "detector_v4_3/judges/V4_3_LOCAL_JUDGE_B_BLIND.jsonl": (
        "c8533f98efa08e47da9977ba485801f4907d331818564ffbd1f329694dcc8f67"
    ),
    "detector_v4_3/judges/V4_3_LOCAL_JUDGE_A_BLIND_RUN.json": (
        "58630008c2c7979ab54fa6279abb9f5a6b43cb7ee62da0e4c12932f642204b84"
    ),
    "detector_v4_3/judges/V4_3_LOCAL_JUDGE_B_BLIND_RUN.json": (
        "1d54c0852898d19ef14f4b949e681389bfdfb18de6ca46dbdf9880f4e3c261c2"
    ),
    "detector_v4_3/labels/V4_3_BLIND_AGREEMENT_SUMMARY.json": (
        "6bd3bef7c2ae5da93aa7babdd1b30574327c00895311201fafd7988b07870262"
    ),
    "detector_v4_3/labels/V4_3_BLIND_DISAGREEMENTS.jsonl": (
        "ce8ef1e50a3bffd05a250374ed0fa692459a05a75c1c9951d0df7c9ff961dd12"
    ),
    "detector_v4_3/labels/V4_3_BLIND_STRATUM_BREAKDOWN.json": (
        "f927c44f8b6a950426e8e8b1801be2d821ab31e8510b9e9592bcaa317f5cb8a5"
    ),
    "detector_v4_1/LABEL_AUDIT_JUDGE_A.jsonl": (
        "8271884e6616474115ecf57904e82b84d81ace33e480473bf702c2fc98e27ae2"
    ),
    "detector_v4_1/LABEL_AUDIT_JUDGE_B.jsonl": (
        "f6e82e724223a65f3d44cc2cb788d82be1108be7d4f38ddbfd9cd9684179a67a"
    ),
    "detector_v4_1/LABEL_AUDIT_KEY.json": (
        "6fbf569090f9a5cfc943e39d1f70d41c60319b5de0a32004ff75353d44d7eba0"
    ),
    # Recovered in PR #53, not present at PR #52. The GPU-1 acceptance evidence for v4.3
    # existed only on the rented box's overlay filesystem, and GU-0048 cites its facts in
    # prose -- correct pinned commit shas, real 8-bit and 4-bit quantization, bfloat16
    # confirmed loaded, no CPU offload, zero malformed on 50 rows per judge -- while
    # nothing in the repository carried them. Stopping the instance would have destroyed
    # the only copy, and re-deriving it means re-renting a GPU.
    #
    # The per-row judgements are NOT here. They live only in `*.partial.jsonl`, which
    # `.gitignore` excludes deliberately: a committed partial is a second copy of a label
    # set that can drift from the closed one, and a smoke run must not be mistakable for a
    # pass. That rule is not overridden here. These two run records carry every fact the
    # acceptance criteria are stated in terms of; the 50 judgements themselves remain
    # non-reportable working output.
    "detector_v4_3/smoke/V4_3_SMOKE_smoke-qwen_JUDGE_A_BLIND_RUN.json": (
        "fdbe7b12b9effc1daa0d4e40087f3c86413df404af10e14e50ed0868c47ef678"
    ),
    "detector_v4_3/smoke/V4_3_SMOKE_smoke-mistral_JUDGE_B_BLIND_RUN.json": (
        "e5127f1cec99117ebb0bc83a8d3371d9cbeafac9e0e2379d10727b2b8518d315"
    ),
}

SEALED_DIGESTS: dict[str, str] = {
    "detector_v4_1/FINAL_GATE_BANK_MANIFEST.json": (
        "91784a31ab6d3132e804815475ab1e290b72e176f68f2452371abff46575df67"
    ),
    "detector_v4_2/FINAL_GATE_BANK_BUDGET.json": (
        "6795e0782a3de960a3ef27cf58e19afaab9dd2f004f36276eb1e541bfdb040a9"
    ),
}


def changed_artifacts(
    cohort_root: Path, digests: Mapping[str, str] | None = None
) -> list[tuple[str, str]]:
    """``[(relative path, what happened)]`` for every artifact that no longer matches.

    A missing file and an edited one are reported separately, because they are different
    accidents: the first is usually a bad checkout, the second is usually a tool that
    rewrote a file it should have read.
    """
    table = dict(digests if digests is not None else {**V4_3_FROZEN_DIGESTS, **SEALED_DIGESTS})
    out: list[tuple[str, str]] = []
    for relative, expected in sorted(table.items()):
        path = Path(cohort_root) / relative
        if not path.exists():
            out.append((relative, "absent"))
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            out.append((relative, f"changed: {actual[:12]} != {expected[:12]}"))
    return out
