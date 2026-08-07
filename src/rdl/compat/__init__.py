"""Compatibility shims for version skew inside `third_party/open-unlearning`.

Design invariant 1 says we never fork, vendor, or patch open-unlearning. Nothing in
this package edits a file under `third_party/`. What it does instead is restore, at
runtime and only for the eval subprocess, a behaviour that upstream's own pinned
dependencies used to provide and silently stopped providing.

Every shim here must be:

  * **narrow** — one named upstream defect, not a general "make it work" layer;
  * **recorded** — the run report says which shims were active, so a number produced
    under one is never mistaken for a number produced without;
  * **loud** — if a shim cannot be applied it raises. A shim that silently no-ops is
    worse than no shim, because the run continues and the report still claims it.
"""

from __future__ import annotations

__all__ = ["fp32_logits"]
