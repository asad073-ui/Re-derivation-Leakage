# Related work / novelty audit

Kept current. The purpose is to know, at any moment, exactly which sentence of the
contribution is still standing.

---

## The claim we are defending

> A memory-level defence whose guarantees are stated over **node identity and derivation
> edges** cannot bound recovery of content that re-enters the store through a *fresh,
> unparented* node. We measure this, and we certify it per node.

Two halves, and they are load-bearing separately:

1. **the measurement** — system-level recovery in a multi-agent shared-memory setting,
   reported per surface, with the persistent store as primary;
2. **the certificate** — a machine-checkable, per-node artifact showing the recovering
   node satisfies the defence's own invariants.

Half 2 is the part nothing in the literature reports.

---

## Adjacent lines of work, and what each does *not* cover

### Machine unlearning for LLMs (TOFU, MUSE, NPO, GradDiff, RMU, …)

Evaluate a **single model in isolation**: forget quality, model utility, truth ratio,
extraction strength. All of it defined over one model's output distribution.

**Gap:** none of it says anything about a *system* built from such models, and in
particular none of it covers content re-entering a shared store. Our C0 reproduces this
literature's numbers precisely so that the system-level result cannot be dismissed as a
broken unlearning setup.

### Relearning / recovery attacks on unlearned models

Show that unlearned content can be recovered by finetuning, by adversarial prompting, or
by in-context reintroduction.

**Gap:** the recovery vector is an *attacker* acting on the model. Ours is the system's
**own default write path**, operating benignly, with every defence functioning correctly.
That distinction is the contribution — and it is also the one most likely to be missed by
a reviewer skimming, so it needs to be in the abstract's first three sentences.

### RAG and memory-augmented generation

Studies leakage from retrieval corpora, and defends with retrieval-time filtering.

**Gap:** the laundered node is **written, not retrieved**. A retrieval-time guardrail
cannot see it — the node has a fresh id and clean content-provenance by construction.
This is exactly why `writepolicy: sanitized` (write-time enforcement) is the Phase-2
method rather than a better retrieval filter.

### Agent memory frameworks (Letta/MemGPT, mem0, LangGraph, …)

Provide persistence, and increasingly provide deletion APIs.

**Gap:** the deletion APIs are id-scoped. Our `framework_default` write policy mirrors
their documented default assistant-turn write. **The leak comes from a defensible
default, not from a policy we designed to leak** — see `src/rdl/agents/writer.py` for the
per-framework citations. R7 in `05_risks.md`: verify these citations against the current
release before submission; these projects move fast.

### Provenance / lineage tracking in data systems

Well-developed. Would in principle solve this.

**Gap:** it requires provenance at write time, which is precisely what a parametric
answer does not have. There is nothing to point at. Noting this is what stops
"just track provenance" from being an obvious rebuttal — the honest answer is that you
cannot track a lineage that does not exist, so the intervention has to be content-level
(`SemanticBlocklist`), not lineage-level.

---

## What would make this not novel

Update this list as the audit continues. Any hit here is a reason to re-scope, and it is
better to find it in week one.

- [ ] A paper reporting **system-level** recovery of unlearned content in a multi-agent
      shared-memory setting.
- [ ] A paper reporting recovery **through a node that satisfies the deletion mechanism's
      own invariants** — i.e. anything equivalent to `laundering_rate`.
- [ ] A formal result showing id-scoped deletion invariants do not imply
      non-recoverability. *(If this exists, the contribution becomes the empirical
      measurement plus the certificate artifact, and the framing must change from
      "we show" to "we measure what was already known to be possible".)*
- [ ] An agent-memory framework whose **default** write path already records
      content-level provenance or applies a write-time content check. *(This would
      undercut the "defensible default" argument for that framework specifically; the
      response is to re-cite against a framework where it still holds, and to say so.)*

---

## Positioning if the primary result does not hold

Pre-committed in `00_preregistration.md` §5, restated here so the fallback is a decision
rather than an improvisation:

- **`C3 − C1 < 10` points, with a demonstrated non-zero residual in B** → the honest
  finding is a **negative result**: id-scoped deletion is adequate in this setting. That
  is publishable as a measurement paper with the certificate machinery as the artifact.
  Workshop, not ICLR.
- **C3 large but driven by routing** (fails under `always_delegate`) → the finding is
  about **utility collapse under unlearning**, which is a different and smaller paper.
- **C2 large, C3 ≈ 0** → the finding is "heterogeneous unlearning leaks", which is a
  configuration bug rather than a mechanism. Day-5 kill criterion.
