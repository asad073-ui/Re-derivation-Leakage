"""NETWORK. Both pinned local judges tokenize, through the family each one actually ships.

The sibling of ``test_deberta_tokenizers.py``, for the same class of defect one layer up.
PR #47 resolved the judge's *model* class from the checkpoint's own config -- correct, and
still true -- and left the *tokenizer* on a bare ``AutoTokenizer.from_pretrained``. On
judge B that raises::

    AutoTokenizer.from_pretrained("mistralai/Mistral-Small-3.2-24B-Instruct-2506")
      -> KeyError: <class ...configuration_mistral3.Mistral3Config>

Two independent causes, either sufficient: transformers 4.51 has no ``mistral3`` entry in
``TOKENIZER_MAPPING_NAMES``, and the repository publishes ``tekken.json`` and nothing else
-- no ``tokenizer.json``, no ``tokenizer_config.json``. Its documented tokenizer is
``mistral-common``.

Why this file exists rather than more unit tests: every v4.3 judge test runs
``--fake-model``, which loads no tokenizer, no config and no weights. That is the right
default for a CPU suite, and it means the loader had no test at all -- the first real
tokenizer load for judge B would have been on a rented GPU box, after the environment is
built, the repository cloned and the clock running.

Only tokenizers are loaded here. They are a few MB; the judges themselves are 27 and 45
GiB and have no business in CI.

Skipped, never failed, when the Hub is unreachable. A missing ``mistral-common`` is NOT a
skip -- that is the defect.
"""

from __future__ import annotations

import dataclasses
import importlib.util

import pytest

pytest.importorskip("transformers", reason="the offline CPU extra omits transformers")

from rdl.cli.detector_v4_3_local_judge import _resolve_tokenizer
from rdl.eval.detector_v4_3_judges import LOCAL_JUDGE_ROSTER, blind_prompt


def _pinned(role):
    """The roster ships EMPTY revisions on purpose; a pin file supplies them in production.

    ``from_pretrained(repo, revision="")`` is not a valid request, so this test resolves
    the repository's current sha the same way ``graph-detector-v4-3-freeze-judge-pins
    --resolve`` does. That makes the test a check of the DISPATCH, not of any one commit:
    what matters here is which tokenizer family the checkpoint routes to.
    """
    from huggingface_hub import HfApi

    pin = LOCAL_JUDGE_ROSTER[role]
    try:
        sha = HfApi().model_info(pin.repo_id).sha
    except Exception as exc:  # pragma: no cover - network, not logic
        pytest.skip(f"{pin.repo_id} is unreachable: {exc}")
    return dataclasses.replace(pin, revision=sha)


def test_mistral_common_is_installed():
    """Asserted separately from the load below, so an absence is unambiguous."""
    assert importlib.util.find_spec("mistral_common") is not None, (
        "mistral-common is not installed. Mistral-Small-3.2 publishes only tekken.json and "
        "transformers has no Mistral3Config tokenizer mapping, so AutoTokenizer raises "
        "KeyError on it; add it to the gpu extra rather than discovering this on a rented "
        "box."
    )


@pytest.mark.parametrize(
    ("role", "expected_kind"),
    [("A", "transformers"), ("B", "mistral-common")],
)
def test_each_judge_resolves_the_tokenizer_family_its_checkpoint_ships(role, expected_kind):
    """Dispatch is on the declared architecture, not on a guess about the model card.

    Judge A is a plain causal LM and keeps ``apply_chat_template``. Judge B declares
    ``Mistral3ForConditionalGeneration`` and must not be routed to ``AutoTokenizer``.
    """
    pin = _pinned(role)
    try:
        resolved = _resolve_tokenizer(pin)
    except OSError as exc:  # pragma: no cover - network, not logic
        pytest.skip(f"{pin.repo_id} is unreachable: {exc}")

    assert resolved.kind == expected_kind, (
        f"judge {role} ({pin.repo_id}) resolved to {resolved.kind!r}. Routing it to the "
        "other family is how a judge becomes unloadable on the box."
    )
    assert resolved.eos_id is not None, "generation pads with eos; it cannot be unknown"
    assert resolved.max_length > 1000, f"implausible context bound {resolved.max_length}"


@pytest.mark.parametrize("role", ["A", "B"])
def test_each_judge_encodes_a_real_blind_prompt_and_round_trips(role):
    """The actual prompt the labelling run sends, not a bare string.

    The chat template is part of what the context has to hold and part of what the model
    reads as an instruction, so it is what gets measured. A prompt of this size against a
    judge context this large is also the evidence for expecting zero truncations across
    the 1,019-row audit.
    """
    pin = _pinned(role)
    prompt = blind_prompt(
        conditioning_question="What is the theme of 'Apparitions in the Classroom'?",
        subject_aliases=["Annalise Martin"],
        candidate_text="It explores psychological terror in a school setting.",
    )
    try:
        resolved = _resolve_tokenizer(pin)
    except OSError as exc:  # pragma: no cover - network, not logic
        pytest.skip(f"{pin.repo_id} is unreachable: {exc}")

    ids = resolved.encode_chat(prompt)
    assert ids, f"judge {role} produced no token ids"
    assert len(ids) > 50, f"judge {role} produced a suspiciously short encoding"
    assert len(ids) < resolved.max_length, "a real blind prompt must fit the judge's context"

    # Round-trips through the vocabulary rather than the byte fallback. Decoding the tail
    # rather than the whole prompt keeps this independent of each family's opening markers.
    assert "psychological terror" in resolved.decode(ids[-64:])


def test_an_unpinned_revision_is_not_silently_resolved_to_main():
    """The roster ships empty revisions; the pin file supplies them.

    Not a tokenizer property, but it is checked in the one test file that reaches the Hub:
    a tokenizer fetched from a moved ``main`` is a different annotator wearing the pinned
    one's name.
    """
    for role, pin in LOCAL_JUDGE_ROSTER.items():
        assert not pin.pinned, (
            f"judge {role} carries a revision in source. Pins belong in the committed "
            "artifact that `graph-detector-v4-3-freeze-judge-pins` writes."
        )
    pinned = dataclasses.replace(LOCAL_JUDGE_ROSTER["A"], revision="a" * 40)
    assert pinned.pinned
