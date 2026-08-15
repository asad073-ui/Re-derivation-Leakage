"""NETWORK. Both pinned DeBERTa-v3 tokenizers load, and they encode a real judged pair.

DeBERTa-v3's tokenizer is the DeBERTa-**v2** SentencePiece tokenizer, and `transformers`
depends on neither package required to load one: `sentencepiece` reads the model and
`protobuf` parses its proto. Neither the ``gpu`` extra nor ``requirements-gpu-ampere.txt``
declared them, so the first ``AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")``
on a freshly built GPU box raises — after the environment is installed, the repository is
cloned, the label audit is done, and the rental clock is running.

This test earned its place immediately: written with `sentencepiece` alone, it failed in CI
on the missing `protobuf`, one dependency further along than the fix that prompted it. Both
are declared now.

It loads BOTH tokenizers because they are separate pins for separate reasons: the encoder's
subword split decides every score at a fixed threshold, and the NLI baseline is a different
repository whose tokenizer the zero-shot number is read through.

Skipped, never failed, when the Hub is unreachable: a red CI job for somebody's proxy
teaches nothing. A missing `sentencepiece` is NOT a skip — that is the defect.
"""

from __future__ import annotations

import importlib.util

import pytest

pytest.importorskip("transformers", reason="the offline CPU extra omits transformers")

MODEL_REPO = "microsoft/deberta-v3-base"
BASELINE_REPO = "cross-encoder/nli-deberta-v3-base"


@pytest.mark.parametrize(
    ("module", "why"),
    [
        (
            "sentencepiece",
            "DeBERTa-v3 uses the DeBERTa-v2 SentencePiece tokenizer and transformers does "
            "not pull it in",
        ),
        (
            "google.protobuf",
            "transformers parses the SentencePiece model through protobuf in "
            "convert_slow_tokenizer, and raises ImportError without it — sentencepiece "
            "alone gets one step further and still fails",
        ),
    ],
)
def test_both_tokenizer_dependencies_are_installed(module, why):
    """Asserted separately from the load below, so an absence is unambiguous.

    Two packages, not one. The first version of this file checked `sentencepiece` only,
    and the network job then failed on protobuf — which is exactly the failure this test
    exists to move off the rented box and into CI.
    """
    assert importlib.util.find_spec(module) is not None, (
        f"{module} is not installed. {why}; add it to the gpu extra and to "
        "requirements-gpu-ampere.txt rather than discovering this on a rented box."
    )


@pytest.mark.parametrize("repo_id", [MODEL_REPO, BASELINE_REPO])
def test_the_pinned_deberta_tokenizers_load(repo_id):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
    except OSError as exc:  # pragma: no cover - network, not logic
        pytest.skip(f"{repo_id} is unreachable: {exc}")

    # A real (question, candidate) pair in the shape the cross-encoder is served, not a
    # bare string: the segment separator is part of what the tokenizer has to produce.
    encoded = tokenizer(
        "Where was Ada Vane born?",
        "She was born in Rome, according to the biography.",
        truncation=True,
        max_length=256,
    )
    assert encoded["input_ids"], f"{repo_id} produced no token ids"
    assert len(encoded["input_ids"]) > 8, f"{repo_id} produced a suspiciously short encoding"
    # Round-trips through the vocabulary rather than the byte fallback.
    assert "Rome" in tokenizer.decode(encoded["input_ids"])
