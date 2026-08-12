"""Safe refusals and sanitization certificates.

The only output a sanitizer may certify is one it *constructed itself* from a fixed
template. There is no "rewrite the answer to remove the forgotten part" path here, on
purpose: certifying a model-generated rewrite would need a verifier at least as strong
as the thing being verified, and a defence whose guarantee rests on an unverified
rewrite is a claim, not a mechanism.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["SAFE_REFUSAL", "SanitizationCertificate", "Sanitizer"]

# Deliberately identical to the repo's existing abstention text so the offline scorer,
# the abstention detectors and the graph defence all agree on what a refusal looks like.
SAFE_REFUSAL = "I don't know."
QUARANTINE_NOTICE = "[withheld: content matched a forgotten scope]"


@dataclass(frozen=True)
class SanitizationCertificate:
    method: str
    verified: bool
    detector_version: str
    forget_ids: tuple[str, ...]
    output_sha256: str
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "verified": self.verified,
            "detector_version": self.detector_version,
            "forget_ids": list(self.forget_ids),
            "output_sha256": self.output_sha256,
            "note": self.note,
        }


class Sanitizer:
    version = "sanitizer-v1"

    def __init__(self, *, refusal_text: str = SAFE_REFUSAL) -> None:
        self.refusal_text = refusal_text

    def safe_refusal(
        self, forget_ids: Sequence[str], *, detector_version: str
    ) -> tuple[str, SanitizationCertificate]:
        text = self.refusal_text
        return text, SanitizationCertificate(
            method="fixed_safe_refusal",
            verified=True,
            detector_version=detector_version,
            forget_ids=tuple(sorted(forget_ids)),
            output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            note=(
                "constructed from a fixed template; contains no model-generated content, "
                "so it carries no forgotten information. Its provenance retains the "
                "Forget IDs so descendants still inherit the scope."
            ),
        )

    def quarantine(
        self, forget_ids: Sequence[str], *, detector_version: str
    ) -> tuple[str, SanitizationCertificate]:
        text = QUARANTINE_NOTICE
        return text, SanitizationCertificate(
            method="quarantine_notice",
            verified=True,
            detector_version=detector_version,
            forget_ids=tuple(sorted(forget_ids)),
            output_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            note="payload withheld from the consumer and retained in the evidence trace",
        )
