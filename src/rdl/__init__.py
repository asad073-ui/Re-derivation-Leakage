"""rdl — Re-derivation Leakage.

Measuring memory-mediated recovery of unlearned knowledge in multi-agent systems.

Design invariants enforced across this package (see REPO_SPEC 0):

1. `third_party/open-unlearning` is a pinned submodule. We call it; we never patch it.
   Any metric we invent lives in `rdl.eval` and is computed on top of their outputs.
2. Every model call goes through `rdl.models.loader`. No bare
   `AutoModelForCausalLM.from_pretrained` anywhere else.
3. Everything except the actual forward pass is testable on CPU with no network,
   via `rdl.models.stub.StubLM`.
4. The transcript is a typed event log, not a string.
5. Determinism by default: greedy, batch_size=1, seeds in three libraries.
6. Results are append-only and content-addressed.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
