"""An audit bundle: which key, which blinded inputs, which rows — named explicitly.

The v4.2 report read four filenames off a directory: ``LABEL_AUDIT_KEY.json``,
``LABEL_AUDIT_JUDGE_{A,B}.jsonl`` and ``LABEL_AUDIT_REFERENCE_PASS_{A,B}.jsonl``. Those are
the **v4.1 human audit's** names. ``rdl graph-detector-v4-2-bank-audit`` — the command that
exists to label a fresh bank — writes ``BANK_AUDIT_KEY.json`` and ``BANK_AUDIT_JUDGE_*``,
so the report could not read its own pipeline's output at all: pointed at the bank audit's
directory it reported "LABEL_AUDIT_KEY.json is absent", and pointed at the v4.1 directory
it silently produced a report about the 1,019-row training audit while the operator
believed it described the new bank.

A bundle is the fix. Every audit writes one manifest that names its own files, and every
consumer takes ``--audit-manifest`` and reads the names out of it. The v4.1 layout is still
resolvable by convention, because those files are committed and their names are frozen —
but it is one named legacy case rather than the only thing the readers can see.

What a bundle carries
---------------------
``key_path``            the offline key. No judge reads it; the report reads it for strata.
``blind_inputs``        ``{judge: path}`` — the blinded pass files, per judge.
``reference_inputs``    ``{judge: path}`` — the reference-pass files, per judge.
``bank_content_sha256`` the bank these rows were drawn from, when there is one. A v4.1
                        audit has none, and ``None`` is the honest value.
``audit_ids``           every id the key knows, so a consumer can check coverage without
                        re-reading four files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import typer

from ..eval.detector_v4_2 import JUDGES

__all__ = ["AuditBundle", "load_bundle"]

# The v4.1 human audit's frozen filenames. Named here as one legacy layout rather than
# hard-coded in every reader.
LEGACY_KEY_FILENAME = "LABEL_AUDIT_KEY.json"
LEGACY_BLIND_FILENAME = "LABEL_AUDIT_JUDGE_{judge}.jsonl"
LEGACY_REFERENCE_FILENAME = "LABEL_AUDIT_REFERENCE_PASS_{judge}.jsonl"
LEGACY_BUNDLE_ID = "v4.1-label-audit"

BUNDLE_SCHEMAS = (
    "graph-detector-v4-2-bank-audit-manifest-v1",
    "graph-detector-v4-2-bank-audit-manifest-v2",
)


@dataclass(frozen=True)
class AuditBundle:
    """The files one audit produced, resolved once and passed around as one object."""

    bundle_id: str
    manifest_path: Path | None
    key_path: Path
    blind_inputs: dict[str, Path]
    reference_inputs: dict[str, Path]
    bank_path: str | None = None
    bank_content_sha256: str | None = None
    manifest: dict = field(default_factory=dict)

    def input_for(self, *, judge: str, pass_name: str) -> Path:
        table = self.blind_inputs if pass_name == "blind" else self.reference_inputs
        if judge not in table:
            raise typer.BadParameter(
                f"the audit bundle {self.bundle_id!r} names no {pass_name} input for judge "
                f"{judge!r}; it has {sorted(table)}"
            )
        return table[judge]

    def key_rows(self) -> dict[str, dict]:
        if not self.key_path.exists():
            raise typer.BadParameter(
                f"{self.key_path} is absent. The key carries the stratum and the "
                "population every per-stratum number in the report is computed over; "
                "without it the report would describe one undifferentiated pool."
            )
        payload = json.loads(self.key_path.read_text(encoding="utf-8"))
        return dict(payload.get("rows", {}))

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "manifest": str(self.manifest_path) if self.manifest_path else None,
            "key": str(self.key_path),
            "blind_inputs": {j: str(p) for j, p in sorted(self.blind_inputs.items())},
            "reference_inputs": {j: str(p) for j, p in sorted(self.reference_inputs.items())},
            "bank_path": self.bank_path,
            "bank_content_sha256": self.bank_content_sha256,
            "resolved_from": (
                "an audit manifest" if self.manifest_path else "the frozen v4.1 filenames"
            ),
        }


def _resolve(base: Path, value: object, *, what: str, manifest_path: Path) -> Path:
    if not value:
        raise typer.BadParameter(
            f"{manifest_path} names no {what}. A bundle that does not name its own files "
            "cannot be read without guessing, and guessing is how a report about a "
            "1,019-row training audit ends up describing a fresh bank."
        )
    path = Path(str(value))
    return path if path.is_absolute() else (base / path if not path.exists() else path)


def load_bundle(
    *,
    audit_dir: Path,
    manifest: Path | None = None,
    require_reference: bool = True,
) -> AuditBundle:
    """Resolve an audit bundle from a manifest, or from the frozen v4.1 layout.

    Passing ``manifest`` is the explicit path and is what every v4.2 bank audit uses.
    Falling back to ``audit_dir`` resolves the v4.1 human audit's frozen filenames, which
    is the only layout that predates bundles.
    """
    if manifest is not None:
        if not manifest.exists():
            raise typer.BadParameter(f"{manifest} is absent")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        schema = str(payload.get("schema", ""))
        if schema not in BUNDLE_SCHEMAS:
            raise typer.BadParameter(
                f"{manifest} carries schema {schema!r}, and this reader knows "
                f"{list(BUNDLE_SCHEMAS)}. An audit manifest of an unknown schema may name "
                "its files under different keys; refusing to guess."
            )
        bundle = payload.get("bundle") or {}
        base = manifest.parent
        blind = {
            role: _resolve(
                base,
                (bundle.get("blind") or {}).get(role),
                what=f"blind input {role}",
                manifest_path=manifest,
            )
            for role in JUDGES
        }
        reference = {
            role: _resolve(
                base,
                (bundle.get("reference") or {}).get(role),
                what=f"reference input {role}",
                manifest_path=manifest,
            )
            for role in JUDGES
        }
        # `require_reference` is about whether a CONSUMER needs the reference pass, not
        # about which names the bundle knows. Dropping absent paths here would make
        # `input_for` raise "no reference input for judge A" when the honest answer is
        # "that file has not been written yet", which is a different problem.
        del require_reference
        return AuditBundle(
            bundle_id=str(payload.get("bundle_id") or payload.get("bank_id") or manifest.stem),
            manifest_path=manifest,
            key_path=_resolve(base, bundle.get("key"), what="key", manifest_path=manifest),
            blind_inputs=blind,
            reference_inputs=reference,
            bank_path=str(payload.get("bank_path")) if payload.get("bank_path") else None,
            bank_content_sha256=(
                str(payload.get("bank_content_sha256"))
                if payload.get("bank_content_sha256")
                else None
            ),
            manifest=payload,
        )

    key_path = audit_dir / LEGACY_KEY_FILENAME
    if not key_path.exists():
        raise typer.BadParameter(
            f"{key_path} is absent and no --audit-manifest was given. Either point "
            "--audit-dir at the v4.1 label audit, or pass --audit-manifest "
            "BANK_AUDIT_MANIFEST.json for a bank audit. These are different audits over "
            "different rows and the report must be told which one it is reading."
        )
    return AuditBundle(
        bundle_id=LEGACY_BUNDLE_ID,
        manifest_path=None,
        key_path=key_path,
        blind_inputs={
            role: audit_dir / LEGACY_BLIND_FILENAME.format(judge=role) for role in JUDGES
        },
        reference_inputs={
            role: audit_dir / LEGACY_REFERENCE_FILENAME.format(judge=role) for role in JUDGES
        },
        bank_path=None,
        bank_content_sha256=None,
        manifest={},
    )
