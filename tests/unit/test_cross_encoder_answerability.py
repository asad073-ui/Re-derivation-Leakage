"""The trained backend, exercised against a tiny in-process model.

No network, no checkpoint, no ``transformers``: the tokenizer and the model are fakes with
the two methods the detector actually uses. That is deliberate rather than expedient — a
defence that could only be tested with a downloaded model is a defence the CPU gate never
runs, and ``make cpu-all`` has to stay offline.

The file is in two halves. The first is :func:`budget_encode`, where v4's real defect lived:
it truncated the CANDIDATE, which is the only span the verdict is about. The second is the
detector's contract — the protocol, the routing fence, accumulation, and the pins without
which a checkpoint is not eligible for a study.
"""

from __future__ import annotations

import json

import pytest

from rdl.defenses.cross_encoder_answerability import (
    CrossEncoderAnswerabilityDetector,
    EncodingBudget,
    budget_encode,
    identity_context,
)
from rdl.defenses.detection_context import DetectionContext, ProtectedQuestion
from rdl.defenses.detector_protocol import ConceptDetector
from rdl.defenses.identity_router import IngressRouting


class FakeTokenizer:
    """Whitespace tokenizer with a stable vocabulary. Enough for the budget arithmetic."""

    pad_token_id = 0

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def _id(self, word: str) -> int:
        # From 100, so no word can collide with the 0/1/2 pad and separator ids the fake
        # model uses to find the candidate segment.
        return self.vocab.setdefault(word, len(self.vocab) + 100)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._id(w) for w in str(text).split()]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        back = {v: k for k, v in self.vocab.items()}
        return " ".join(back.get(int(i), "") for i in ids).strip()

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 3 if pair else 2

    def __call__(self, a, b=None, truncation=False, max_length=None, **_kwargs) -> dict:
        ids = [1, *self.encode(a)]
        if b is not None:
            ids += [2, *self.encode(b), 2]
        if truncation and max_length:
            ids = ids[:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class FakeModel:
    """A stand-in scorer with the shape of a real one: relation cue plus filled slot.

    ANSWER needs both ``relation`` and ``slot`` present; ``relation`` alone is PARTIAL and
    neither is NONE. That is the minimum structure the accumulation test needs — a
    continuation that cannot fire on its own but completes a fragment that could not
    either — and it is why the fake reads only the CANDIDATE segment. Scoring the whole
    window would make every pair fire, since the protected question names the relation.
    """

    tokenizer: FakeTokenizer | None = None

    def __init__(self, relation: str = "born", slot: str = "Paris.") -> None:
        self.relation = relation
        self.slot = slot
        self.calls = 0

    def __call__(self, **batch):
        self.calls += 1
        rows = []
        for ids in batch["input_ids"]:
            text = self._candidate(ids)
            has_relation = self.relation in text
            has_slot = self.slot in text
            if has_relation and has_slot:
                rows.append([0.0, 0.0, 6.0])
            elif has_relation:
                rows.append([0.0, 6.0, 0.0])
            else:
                rows.append([6.0, 0.0, 0.0])
        return type("Out", (), {"logits": rows})()

    def _candidate(self, ids) -> str:
        """Segment B only: ``[CLS] question [SEP] identity [SEP] candidate [SEP]``."""
        assert self.tokenizer is not None
        ids = list(ids)
        return self.tokenizer.decode(ids[ids.index(2) + 1 :] if 2 in ids else ids)


def _detector(**kwargs) -> CrossEncoderAnswerabilityDetector:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    model.tokenizer = tokenizer
    return CrossEncoderAnswerabilityDetector(
        model,
        tokenizer,
        model_repo_id="fake/model",
        model_revision="abc123",
        tokenizer_revision="abc123",
        **kwargs,
    )


def _context(question: str = "Where was Ada Vane born?") -> DetectionContext:
    protected = ProtectedQuestion(
        scope_id="author-0000#000",
        forget_id="author-0000",
        question=question,
        aliases=("Ada Vane", "Vane"),
    )
    return DetectionContext(
        request_text=question,
        routing=IngressRouting(policy_context_ids=("author-0000",)),
        protected_questions=(protected,),
    )


# =====================================================================================
# budget_encode — the v4 truncation defect
# =====================================================================================


def test_the_candidate_survives_a_budget_that_cannot_hold_everything():
    """THE regression. v4 passed ``truncation="only_second"`` on
    ``identity + "[SEP]" + candidate``, which drops the END of that sequence — the
    candidate. A verdict about evidence that was truncated away is not a verdict.
    """
    tokenizer = FakeTokenizer()
    question = " ".join(f"q{i}" for i in range(40))
    candidate = " ".join(f"c{i}" for i in range(20))
    _encoding, stats = budget_encode(
        tokenizer,
        question=question,
        aliases=["Ada Vane", "Vane", "A V", "Ms Vane"],
        candidate=candidate,
        budget=EncodingBudget(max_length=32),
    )
    assert stats["candidate_truncated"] is False
    assert stats["n_candidate_tokens_dropped"] == 0
    # The budget was spent on the other two, in the documented order.
    assert stats["n_aliases_dropped"] > 0
    assert stats["question_truncated"] is True


def test_the_question_keeps_its_floor():
    """Below the floor the question stops naming a relation, and the model is being asked
    a question nobody posed."""
    tokenizer = FakeTokenizer()
    budget = EncodingBudget(max_length=20, min_question_tokens=8)
    _encoding, stats = budget_encode(
        tokenizer,
        question=" ".join(f"q{i}" for i in range(30)),
        aliases=[],
        candidate=" ".join(f"c{i}" for i in range(30)),
        budget=budget,
    )
    # room = 20 - 3 = 17; the candidate may take 17 - 8 = 9, so 21 of its tokens are cut
    # and the cut is COUNTED rather than silently absorbed.
    assert stats["candidate_truncated"] is True
    assert stats["n_candidate_tokens_dropped"] == 21


def test_nothing_is_truncated_when_everything_fits():
    tokenizer = FakeTokenizer()
    encoding, stats = budget_encode(
        tokenizer,
        question="Where was Ada Vane born?",
        aliases=["Ada Vane"],
        candidate="Ada Vane was born in Paris.",
        budget=EncodingBudget(max_length=256),
    )
    assert stats == {
        "candidate_truncated": False,
        "n_candidate_tokens_dropped": 0,
        "question_truncated": False,
        "n_aliases_dropped": 0,
        "n_candidate_tokens": 6,
        "n_question_tokens": 5,
    }
    assert len(encoding["input_ids"]) <= 256


def test_aliases_are_dropped_whole():
    """Half a name is not an identity signal, and a router matching it matches noise."""
    tokenizer = FakeTokenizer()
    identity = identity_context(["Ada Vane", "Vane", "Ada"], max_aliases=4)
    assert identity == "Ada Vane, Vane, Ada"
    _encoding, stats = budget_encode(
        tokenizer,
        question="Where was Ada Vane born?",
        aliases=["Ada Vane", "Vane", "Ada"],
        candidate=" ".join(f"c{i}" for i in range(10)),
        budget=EncodingBudget(max_length=19, min_question_tokens=5),
    )
    assert stats["n_aliases_dropped"] >= 1


def test_the_raw_request_is_never_the_identity_context():
    """Putting the request in the window with the candidate is the failure Phase 3 exists
    to prevent: the model would learn that a forget question predicts the label."""
    assert identity_context([]) == ""
    assert identity_context(["Ada Vane", "Ada Vane"]) == "Ada Vane"


# =====================================================================================
# The detector contract
# =====================================================================================


def test_it_satisfies_the_concept_detector_protocol():
    assert isinstance(_detector(), ConceptDetector)


def test_an_unrouted_call_never_reaches_the_scorer():
    """Same rule as the lexical backend and as ``identity_router.content_forget_ids``:
    nothing is scanned on a request the router left unselected."""
    detector = _detector()
    context = DetectionContext(request_text="Who wrote Dune?", routing=IngressRouting())
    results = detector.score_batch(["Ada Vane was born in Paris."], context=context)
    assert results[0].forget_ids == ()
    assert detector.model.calls == 0
    assert detector.stats()["detector_unrouted_calls"] == 1


def test_a_routed_answer_fires_on_the_routed_concept():
    detector = _detector()
    result = detector.score_batch(["Ada Vane was born in Paris."], context=_context())[0]
    assert result.fired is True
    assert result.forget_ids == ("author-0000",)
    assert result.answer_probability > 0.9
    assert result.from_accumulated_evidence is False


def test_a_non_answer_does_not_fire():
    detector = _detector()
    result = detector.score_batch(["Ada Vane is a famous author."], context=_context())[0]
    assert result.fired is False
    assert result.forget_ids == ()


def test_a_fragment_is_partial_and_not_enforceable():
    detector = _detector()
    result = detector.score_batch(["Ada Vane was born …"], context=_context())[0]
    assert result.fired is False
    assert result.partial_probability > 0.9


def test_a_split_clue_is_tagged_and_labelled_as_accumulated():
    """The claim the propagation story rests on. A tag that required two messages must be
    counted apart from one any node-local guard would also make."""
    detector = _detector()
    results = detector.score_batch(["Ada Vane was born …", "… in Paris."], context=_context())
    assert results[0].fired is False
    assert results[1].fired is True
    assert results[1].from_accumulated_evidence is True
    assert detector.stats()["accumulated_tags"] == 1


def test_accumulation_does_not_survive_the_call():
    """A detector that remembered fragments across trajectories would make one
    trajectory's verdict depend on another's, and no report could attribute a tag."""
    detector = _detector()
    detector.score_batch(["Ada Vane was born …"], context=_context())
    second = detector.score_batch(["… in Paris."], context=_context())[0]
    assert second.fired is False


def test_the_revision_names_both_revisions_and_the_segmentation():
    detector = _detector(answer_threshold=0.4)
    revision = detector.revision
    assert "fake/model@abc123" in revision
    assert "tok=abc123" in revision
    assert "atomic-text-v1" in revision
    assert "answer=0.400" in revision


def test_re_thresholding_shares_the_model():
    detector = _detector()
    other = detector.with_thresholds(answer_threshold=0.9)
    assert other.model is detector.model
    assert other.tokenizer is detector.tokenizer
    assert other.answer_threshold == 0.9


def test_it_declares_itself_uncalibrated_and_unselectable_by_default():
    described = _detector().to_dict()
    assert described["calibrated"] is False
    assert described["receives_gold_answers"] is False
    assert described["selectable_from_graph_detector_config"] is False
    assert described["reports_answerability"] is True


def test_candidate_truncations_are_reported_in_stats():
    """A candidate that did not fit the window is an evaluation error, not a NONE, and a
    gate measured with this above zero has to say so."""
    detector = _detector(budget=EncodingBudget(max_length=12, min_question_tokens=8))
    detector.score_batch([" ".join(f"c{i}" for i in range(40))], context=_context())
    assert detector.stats()["candidate_truncations"] > 0


def test_an_artifact_without_both_revisions_is_refused(tmp_path):
    """A checkpoint whose weights or whose subword split cannot be named is not eligible."""
    artifact = tmp_path / "DETECTOR_V4_MODEL.json"
    artifact.write_text(
        json.dumps(
            {"pins": {"model_repo_id": "x", "model_revision": "abc", "tokenizer_revision": ""}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tokenizer_revision"):
        CrossEncoderAnswerabilityDetector.from_artifact(artifact)


def test_the_label_order_is_part_of_the_checkpoint():
    tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="label order"):
        CrossEncoderAnswerabilityDetector(
            FakeModel(), tokenizer, labels=("ANSWER", "PARTIAL", "NONE")
        )
