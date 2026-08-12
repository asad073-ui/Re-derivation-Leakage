"""Cross-arm reuse of identical generations, with the audit trail that makes it safe.

Wraps ``runtime.ResponseCache`` and records, per reuse, the arms that shared the entry.
That record is the evidence for the claim a reviewer will want: reuse happened only
where the complete serialized prompt, the decoding parameters and the seed were
byte-identical, so no arm was served another arm's conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...runtime.response_cache import ResponseCache

__all__ = ["ResponseBank"]


@dataclass
class ResponseBank:
    cache: ResponseCache = field(default_factory=ResponseCache)
    # cache key -> the arms that used it
    _arms_by_key: dict[str, set[str]] = field(default_factory=dict)

    def note(self, key: str, arm: str) -> None:
        self._arms_by_key.setdefault(key, set()).add(arm)

    def shared_keys(self) -> int:
        return sum(1 for arms in self._arms_by_key.values() if len(arms) > 1)

    def to_dict(self) -> dict:
        return {
            **self.cache.stats(),
            "keys_shared_across_arms": self.shared_keys(),
            "distinct_keys": len(self._arms_by_key),
            "reuse_rule": (
                "an entry is reused only when backend, model revision, tokenizer revision, "
                "system message, complete prompt, every decoding parameter and the seed are "
                "byte-identical"
            ),
        }
