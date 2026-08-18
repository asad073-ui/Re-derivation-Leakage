"""The protected store: what a deployed GraphForget knows about what it forgot.

The store is the answer to a question v4.2 left implicit. A runtime detector is asked
"does this candidate answer a protected question?", and *which* protected questions exist
was, until now, rebuilt on the fly from a policy cohort by whichever command happened to
need it. Two commands rebuilding the same set from the same files is fine until one of
them rebuilds it slightly differently, and then the recall number and the enforcement
decision are about different sets of concepts.

So the set is an artifact. It is built once, hashed, and read by everything downstream.

**The store holds questions and policy. It never holds answers, and it never holds answer
hashes.** The no-answer rule is the same one :class:`~.detection_context.ProtectedQuestion`
already enforces. The no-*hash* rule is additional and is worth stating separately, because
a hash feels like a safe way to carry an answer and is not: the forgotten answers here are
birth cities, years, genres and multiple-choice keys, and a candidate space that small is
enumerable. A store that shipped ``answer_sha256`` would let anyone holding it confirm the
forgotten fact by guessing, which is the precise capability the system claims to have
destroyed.

Enforcement is an **allowlist**, not a denylist. :data:`RUNTIME_SCOPE_FIELDS` names every
key a runtime record may carry and :meth:`ProtectedScope.from_mapping` refuses everything
else. A denylist has to anticipate the name of the field that leaks; an allowlist only has
to anticipate the fields that are legitimate, and those are fixed by this protocol.

Three artifacts, three audiences
--------------------------------
:class:`ProtectedStore`
    ``PROTECTED_STORE_RUNTIME.json``. Answer-free. Loadable by runtime detector code. One
    record per protected *scope* -- a concept protected under two relations is two scopes,
    because the detector reports which relation it thinks a candidate answered.

:class:`ConditioningIndex`
    ``DETECTOR_V4_3_CONDITIONING_INDEX.json``. Answer-free *and* population-free. Covers
    protected and retain subjects alike under one alias builder. Its job is building model
    inputs, never enforcement -- a retain record conditions generic answerability training
    without becoming a protected runtime scope.

the evaluation key
    ``PROTECTED_STORE_EVAL_KEY.json``. Reference answers, item ids, population, strata.
    **There is deliberately no loader for it in this module**, and
    ``tests/contract/test_detector_v4_3_store_contract.py`` asserts that no module under
    ``rdl.defenses`` mentions it. The separation is structural, not conventional: runtime
    code cannot read the key because runtime code contains no code that reads the key.

Why the conditioning index is population-free
---------------------------------------------
v4.2 built natural aliases from the forget-policy cohort alone. Retain questions are not in
that cohort, so retain rows joined to an empty alias list and protected rows did not -- and
"aliases present" became a perfect linear separator for "row is protected". A model
optimising cross-entropy will take that feature, and the resulting recall number describes
a membership classifier rather than an answerability classifier.

:func:`conditioning_records_from_questions` therefore runs one extractor,
``concept_registry.extract_name_spans``, over every question regardless of where the row
came from. On the frozen 1,019-row audit that moves retain alias coverage from 0/300 to
295/300 against protected's 717/719 -- close enough that alias presence carries almost no
population signal, and what remains is measured rather than assumed by
``rdl graph-detector-v4-3-bundle``'s shortcut probe.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..logging_utils import dumps_canonical
from .concept_registry import extract_name_spans, normalise_scope_text
from .detection_context import ProtectedQuestion

__all__ = [
    "CONDITIONING_INDEX_SCHEMA",
    "CONDITIONING_RECORD_FIELDS",
    "RUNTIME_SCOPE_FIELDS",
    "RUNTIME_STORE_SCHEMA",
    "ConditioningIndex",
    "ConditioningRecord",
    "ProtectedScope",
    "ProtectedStore",
    "StoreDecision",
    "conditioning_records_from_questions",
    "store_decision",
    "subject_id_for",
]

RUNTIME_STORE_SCHEMA = "graph-detector-v4-3-protected-store-runtime-v1"
CONDITIONING_INDEX_SCHEMA = "graph-detector-v4-3-conditioning-index-v1"

# Every key a runtime store record may carry. Anything else is refused by name. See the
# module docstring for why this is an allowlist.
RUNTIME_SCOPE_FIELDS = frozenset(
    {
        "dataset_id",
        "dataset_revision",
        "forget_split",
        "policy_version",
        "forget_id",
        "scope_id",
        "subject_id",
        "question",
        "aliases",
        "relation",
        "template_id",
        "allow_refusal",
        "allow_persistent_write",
        "allow_edge_release",
        "allow_retrieval",
        "question_sha256",
        "provenance_sha256",
    }
)

# Every key a conditioning record may carry. Strictly smaller than the runtime store's:
# a conditioning record has no policy, no forget id and no dataset split, because it
# exists to build a tokenizer input and none of those may reach one.
CONDITIONING_RECORD_FIELDS = frozenset(
    {
        "conditioning_id",
        "subject_id",
        "conditioning_question",
        "subject_aliases",
        "relation",
        "template_id",
        "question_sha256",
    }
)

# Substrings that name the evaluation's answer sheet. Checked against every key of every
# mapping handed to `from_mapping`, so `answer_sha256`, `gold`, `reference_completion` and
# `option_key` are all refused with a message that says which rule they broke, rather than
# with a bare KeyError from the allowlist.
_ANSWER_BEARING = ("answer", "gold", "reference", "completion", "option", "solution")

# `answer_threshold` is detector configuration and `answer_attempt` is the name of a LABEL,
# not the label's value. Neither is a store field, so both still fall to the allowlist --
# this exemption only keeps the error message honest about which rule was broken.
_NOT_THE_ANSWER_SHEET = frozenset({"answer_threshold", "answer_attempt"})


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _refuse_answer_bearing(row: Mapping, *, what: str) -> None:
    """Refuse a mapping that carries the answer sheet, before the allowlist runs.

    Ordered before the allowlist so the error names the *reason*. "ProtectedStore refuses
    answer_sha256" is a protocol violation a reader can act on; "unknown field
    answer_sha256" reads like a typo.
    """
    guilty = sorted(
        key
        for key in row
        if any(token in str(key).lower() for token in _ANSWER_BEARING)
        and str(key).lower() not in _NOT_THE_ANSWER_SHEET
    )
    if guilty:
        raise ValueError(
            f"{what} may not carry {guilty}. The protected store is the artifact a "
            "deployed system holds, and it must not contain the forgotten content in any "
            "form -- including a hash of it. The answers here are cities, years, genres "
            "and option keys; that candidate space is small enough to enumerate, so a "
            "stored digest confirms the fact it was meant to hide. Reference answers "
            "belong in PROTECTED_STORE_EVAL_KEY.json, which no runtime module can read."
        )


def _refuse_unknown(row: Mapping, allowed: frozenset[str], *, what: str) -> None:
    unknown = sorted(set(row) - allowed)
    if unknown:
        raise ValueError(
            f"{what} carries unknown fields {unknown}. This is an allowlist: the fields a "
            f"record may have are {sorted(allowed)}. A denylist would have to anticipate "
            "the name of the field that leaks; refusing everything unrecognised does not."
        )


def subject_id_for(group_key: str) -> str:
    """A stable, non-semantic subject id for a subject/concept group key.

    Non-semantic on purpose. ``tofu-forget10-author-0007`` names the split, the dataset and
    the author's index in it; a store shipping that string tells its holder which authors
    were forgotten and in what order, which is metadata about the forget set rather than
    the policy needed to enforce it. The digest is stable across rebuilds because it is a
    pure function of the group key, so an id can be cited in a report and still resolve
    next month.
    """
    return "subj-" + hashlib.sha256(f"v4.3|{group_key}".encode()).hexdigest()[:16]


@dataclass(frozen=True)
class ProtectedScope:
    """One protected relation, as a deployed system holds it.

    ``forget_id`` is the concept enforcement acts on; ``scope_id`` is the relation the
    detector reports. They are separate for the reason
    :class:`~.detection_context.ProtectedQuestion` keeps them separate -- per-relation
    recall is unwritable once they are collapsed.
    """

    dataset_id: str
    dataset_revision: str
    forget_split: str
    policy_version: str
    forget_id: str
    scope_id: str
    subject_id: str
    question: str
    aliases: tuple[str, ...] = ()
    relation: str = ""
    template_id: str = ""
    allow_refusal: bool = True
    allow_persistent_write: bool = False
    allow_edge_release: bool = False
    allow_retrieval: bool = False
    provenance_sha256: str = ""

    @property
    def question_sha256(self) -> str:
        """A digest of the QUESTION, which is public: it is what the user asked.

        Recorded so a store can be joined to an audit row without shipping the question
        twice, and safe to ship for the reason the answer digest is not -- the question is
        already in the store in cleartext one field above.
        """
        return _sha256(self.question)

    @classmethod
    def from_mapping(cls, row: Mapping) -> ProtectedScope:
        _refuse_answer_bearing(row, what="a ProtectedScope")
        _refuse_unknown(row, RUNTIME_SCOPE_FIELDS, what="a ProtectedScope")
        for required in ("dataset_id", "forget_id", "scope_id", "question"):
            if not str(row.get(required, "")).strip():
                raise ValueError(f"a ProtectedScope requires a non-empty {required!r}")
        scope = cls(
            dataset_id=str(row["dataset_id"]),
            dataset_revision=str(row.get("dataset_revision", "")),
            forget_split=str(row.get("forget_split", "")),
            policy_version=str(row.get("policy_version", "")),
            forget_id=str(row["forget_id"]),
            scope_id=str(row["scope_id"]),
            subject_id=str(row.get("subject_id", "")),
            question=str(row["question"]),
            aliases=tuple(str(a) for a in row.get("aliases", ())),
            relation=str(row.get("relation", "")),
            template_id=str(row.get("template_id", "")),
            allow_refusal=bool(row.get("allow_refusal", True)),
            allow_persistent_write=bool(row.get("allow_persistent_write", False)),
            allow_edge_release=bool(row.get("allow_edge_release", False)),
            allow_retrieval=bool(row.get("allow_retrieval", False)),
            provenance_sha256=str(row.get("provenance_sha256", "")),
        )
        recorded = str(row.get("question_sha256", ""))
        if recorded and recorded != scope.question_sha256:
            raise ValueError(
                f"scope {scope.scope_id!r} records question_sha256 {recorded[:16]}... but "
                f"its question hashes to {scope.question_sha256[:16]}.... The store has "
                "been edited away from the questions it claims to describe."
            )
        return scope

    def to_protected_question(self) -> ProtectedQuestion:
        """The conditioning object the detector actually takes.

        Deliberately lossy: policy actions, dataset identity and provenance do not cross
        into the detector, because none of them is an input to "does this text answer this
        question?".
        """
        return ProtectedQuestion(
            scope_id=self.scope_id,
            forget_id=self.forget_id,
            question=self.question,
            aliases=self.aliases,
            relation=self.relation,
            template_id=self.template_id,
        )

    def to_dict(self) -> dict:
        return {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "forget_split": self.forget_split,
            "policy_version": self.policy_version,
            "forget_id": self.forget_id,
            "scope_id": self.scope_id,
            "subject_id": self.subject_id,
            "question": self.question,
            "aliases": list(self.aliases),
            "relation": self.relation,
            "template_id": self.template_id,
            "allow_refusal": self.allow_refusal,
            "allow_persistent_write": self.allow_persistent_write,
            "allow_edge_release": self.allow_edge_release,
            "allow_retrieval": self.allow_retrieval,
            "question_sha256": self.question_sha256,
            "provenance_sha256": self.provenance_sha256,
        }


class ProtectedStore:
    """The runtime protected set: every scope a request may be routed against."""

    schema = RUNTIME_STORE_SCHEMA

    def __init__(self, scopes: Iterable[ProtectedScope] = ()) -> None:
        self._scopes: tuple[ProtectedScope, ...] = tuple(scopes)
        seen: set[str] = set()
        for scope in self._scopes:
            if scope.scope_id in seen:
                raise ValueError(
                    f"duplicate scope_id {scope.scope_id!r}. Scope ids key the detector's "
                    "per-relation report; two rows under one id make that report ambiguous."
                )
            seen.add(scope.scope_id)

    def __len__(self) -> int:
        return len(self._scopes)

    def __iter__(self):
        return iter(self._scopes)

    def scopes(self) -> tuple[ProtectedScope, ...]:
        return self._scopes

    def forget_ids(self) -> tuple[str, ...]:
        return tuple(sorted({s.forget_id for s in self._scopes}))

    def subject_ids(self) -> tuple[str, ...]:
        return tuple(sorted({s.subject_id for s in self._scopes if s.subject_id}))

    def protected_questions(self) -> tuple[ProtectedQuestion, ...]:
        return tuple(s.to_protected_question() for s in self._scopes)

    def alias_index(self) -> dict[str, tuple[frozenset[str], ...]]:
        """``{forget_id -> alias token sets}``, the shape ``route_request`` takes.

        Built from the store rather than from a registry rebuilt beside it, so the set a
        request can route to and the set the detector can score against are the same set
        by construction.
        """
        index: dict[str, list[frozenset[str]]] = {}
        for scope in self._scopes:
            bucket = index.setdefault(scope.forget_id, [])
            for alias in scope.aliases:
                tokens = frozenset(normalise_scope_text(alias).split())
                if tokens and tokens not in bucket:
                    bucket.append(tokens)
        return {fid: tuple(sets) for fid, sets in index.items()}

    def fingerprint(self) -> str:
        return hashlib.sha256(
            dumps_canonical([s.to_dict() for s in self._scopes]).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "n_scopes": len(self._scopes),
            "n_forget_ids": len(self.forget_ids()),
            "fingerprint_sha256": self.fingerprint(),
            "carries_answers": False,
            "carries_answer_hashes": False,
            "why": (
                "the protected set a deployed GraphForget holds: questions, safe aliases "
                "and policy actions. Reference answers and answer digests live in "
                "PROTECTED_STORE_EVAL_KEY.json, which no runtime module can read."
            ),
            "scopes": [s.to_dict() for s in self._scopes],
        }

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_canonical(self.to_dict()) + "\n", encoding="utf-8")
        return self.fingerprint()

    @classmethod
    def load(cls, path: Path) -> ProtectedStore:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = str(payload.get("schema", ""))
        if schema != RUNTIME_STORE_SCHEMA:
            raise ValueError(
                f"{path} carries schema {schema!r}, not {RUNTIME_STORE_SCHEMA!r}. Refusing "
                "to guess: a store of an unknown schema may name its fields differently, "
                "and a silently empty alias list is an unrouted request."
            )
        store = cls(ProtectedScope.from_mapping(row) for row in payload.get("scopes", ()))
        recorded = str(payload.get("fingerprint_sha256", ""))
        if recorded and recorded != store.fingerprint():
            raise ValueError(
                f"{path} records fingerprint {recorded[:16]}... but its scopes hash to "
                f"{store.fingerprint()[:16]}.... The file has been edited since it was "
                "frozen, so any result citing that fingerprint describes a different store."
            )
        return store


@dataclass(frozen=True)
class ConditioningRecord:
    """One question a model input may be built from. Protected or retain -- unmarked.

    There is no ``population`` field and no ``is_protected`` field, and that absence is the
    point of the class. If the record cannot express which population it came from, no
    downstream input builder can accidentally serialise it.
    """

    conditioning_id: str
    subject_id: str
    conditioning_question: str
    subject_aliases: tuple[str, ...] = ()
    relation: str = ""
    template_id: str = ""

    @property
    def question_sha256(self) -> str:
        return _sha256(self.conditioning_question)

    @classmethod
    def from_mapping(cls, row: Mapping) -> ConditioningRecord:
        _refuse_answer_bearing(row, what="a ConditioningRecord")
        _refuse_unknown(row, CONDITIONING_RECORD_FIELDS, what="a ConditioningRecord")
        return cls(
            conditioning_id=str(row["conditioning_id"]),
            subject_id=str(row.get("subject_id", "")),
            conditioning_question=str(row["conditioning_question"]),
            subject_aliases=tuple(str(a) for a in row.get("subject_aliases", ())),
            relation=str(row.get("relation", "")),
            template_id=str(row.get("template_id", "")),
        )

    def to_dict(self) -> dict:
        return {
            "conditioning_id": self.conditioning_id,
            "subject_id": self.subject_id,
            "conditioning_question": self.conditioning_question,
            "subject_aliases": list(self.subject_aliases),
            "relation": self.relation,
            "template_id": self.template_id,
            "question_sha256": self.question_sha256,
        }


class ConditioningIndex:
    """Every subject the bundle conditions on, under one alias builder.

    Keyed by question text because that is the only join key a *blinded* audit row
    exposes: the judge file carries the question and the candidate and nothing else, which
    is what makes joining aliases to it possible without opening the key.
    """

    schema = CONDITIONING_INDEX_SCHEMA

    def __init__(self, records: Iterable[ConditioningRecord] = ()) -> None:
        self._records = tuple(records)
        self._by_question = {r.conditioning_question: r for r in self._records}

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def records(self) -> tuple[ConditioningRecord, ...]:
        return self._records

    def for_question(self, question: str) -> ConditioningRecord | None:
        return self._by_question.get(question)

    def coverage(self) -> dict:
        """How many records got aliases -- the number that would have exposed v4.2's shortcut.

        Reported rather than asserted, because the honest failure mode is not "some record
        has no aliases" but "the records with no aliases are all in one population". The
        bundle builder computes that split; this is the pooled figure.
        """
        with_aliases = sum(1 for r in self._records if r.subject_aliases)
        return {
            "n_records": len(self._records),
            "n_with_aliases": with_aliases,
            "n_without_aliases": len(self._records) - with_aliases,
            "alias_coverage": (with_aliases / len(self._records)) if self._records else None,
        }

    def fingerprint(self) -> str:
        return hashlib.sha256(
            dumps_canonical([r.to_dict() for r in self._records]).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "fingerprint_sha256": self.fingerprint(),
            "carries_population": False,
            "carries_answers": False,
            "why": (
                "one alias builder for protected and retain subjects alike. v4.2 built "
                "aliases from the forget cohort only, so 'has aliases' separated the "
                "populations perfectly and a model could score well without ever "
                "learning answerability."
            ),
            **self.coverage(),
            "records": [r.to_dict() for r in self._records],
        }

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(dumps_canonical(self.to_dict()) + "\n", encoding="utf-8")
        return self.fingerprint()

    @classmethod
    def load(cls, path: Path) -> ConditioningIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        schema = str(payload.get("schema", ""))
        if schema != CONDITIONING_INDEX_SCHEMA:
            raise ValueError(
                f"{path} carries schema {schema!r}, not {CONDITIONING_INDEX_SCHEMA!r}."
            )
        return cls(ConditioningRecord.from_mapping(r) for r in payload.get("records", ()))


def conditioning_records_from_questions(
    questions: Mapping[str, str],
    *,
    max_aliases: int = 4,
) -> ConditioningIndex:
    """Build the index from ``{conditioning_id -> question}`` and nothing else.

    The signature is the guarantee. There is no ``population`` parameter, no
    ``is_protected`` parameter and no cohort to join against, so the extraction cannot be
    conditioned on which population a question came from even by mistake -- the same code
    path runs for all of them because there is only one code path.

    ``subject_id`` here is derived from the extracted spans, which is the best a purely
    text-derived index can do. The *bundle* overrides it with an authoritative group where
    one exists; see ``rdl graph-detector-v4-3-bundle``.
    """
    records: list[ConditioningRecord] = []
    for conditioning_id, question in sorted(questions.items()):
        spans = extract_name_spans(question)[:max_aliases]
        group = normalise_scope_text(spans[0]) if spans else f"unnamed|{conditioning_id}"
        records.append(
            ConditioningRecord(
                conditioning_id=str(conditioning_id),
                subject_id=subject_id_for(group),
                conditioning_question=str(question),
                subject_aliases=tuple(spans),
                relation="",
                template_id="extract_name_spans/v1",
            )
        )
    return ConditioningIndex(records)


@dataclass(frozen=True)
class StoreDecision:
    """What the store says about one candidate, after routing.

    Two scores, never combined. A weighted sum of "this text completes the forgotten fact"
    and "this text carries a clue toward it" would let two clues outvote a disclosure, and
    the two call for different actions anyway: an ANSWER creates a Forget-ID now, a PARTIAL
    only adds to a bounded, request-local accumulator.
    """

    answer_score: float
    partial_score: float
    fired_forget_ids: tuple[str, ...]
    fired_scope_ids: tuple[str, ...]
    partial_scope_ids: tuple[str, ...]
    n_routed_scopes: int

    @property
    def fires(self) -> bool:
        return bool(self.fired_forget_ids)

    def to_dict(self) -> dict:
        return {
            "protected_answer_score": self.answer_score,
            "protected_partial_score": self.partial_score,
            "fired_forget_ids": list(self.fired_forget_ids),
            "fired_scope_ids": list(self.fired_scope_ids),
            "partial_scope_ids": list(self.partial_scope_ids),
            "n_routed_scopes": self.n_routed_scopes,
            "fires": self.fires,
        }


def store_decision(
    scored: Sequence[tuple[ProtectedQuestion, float, float]],
    *,
    tau_answer: float,
    tau_partial: float,
) -> StoreDecision:
    """``max`` over the routed scopes, per class, plus the ids that cleared each threshold.

    ``scored`` is ``(question, P(ANSWER), P(PARTIAL))`` for scopes the ROUTER licensed --
    this function does not route and cannot widen the set it is given. An empty ``scored``
    is the unrouted case and returns zeros with no Forget-ID: when the store has nothing to
    say about a request, the honest protected score is not a low score, it is no score, and
    a caller that treated 0.0 as evidence of safety would be reading a number that was
    never computed. :attr:`StoreDecision.n_routed_scopes` is how a caller tells the two
    apart.
    """
    if not scored:
        return StoreDecision(0.0, 0.0, (), (), (), 0)
    answer = max(a for _q, a, _p in scored)
    partial = max(p for _q, _a, p in scored)
    fired = tuple(sorted(q.scope_id for q, a, _p in scored if a >= tau_answer))
    partials = tuple(
        sorted(q.scope_id for q, a, p in scored if p >= tau_partial and a < tau_answer)
    )
    forget_ids = tuple(sorted({q.forget_id for q, a, _p in scored if a >= tau_answer}))
    return StoreDecision(
        answer_score=float(answer),
        partial_score=float(partial),
        fired_forget_ids=forget_ids,
        fired_scope_ids=fired,
        partial_scope_ids=partials,
        n_routed_scopes=len(scored),
    )
