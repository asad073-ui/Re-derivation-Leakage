# Re-derivation Leakage

**Memory-mediated recovery of unlearned knowledge in multi-agent systems.**

> A shared memory implementing an ID blocklist enforced at retrieval, plus deletion over
> the derivation closure, can still end an episode with the forgotten content sitting in
> its persistent store, inside a node that **satisfies both of those guarantees**.

The claim is about a defence's *scope*, not about any unlearning method being weak. Both
invariants quantify over node identity and derivation edges. A node written from an
agent's parametric answer has a fresh id and an empty `parent_ids`, so it lies outside
the scope of both, whatever its content says.

**Scope, stated up front.** SBU addresses backflow through a parameter-side pathway as
well as a memory-side one. This repo substitutes NPO for the parameter method and
implements only the memory pathway, in a *strengthened* form (see ADR-0014/0015). So the
result speaks to that pathway and not to SBU end to end; generalising is future work, and
`docs/00b_preregistration_v2.md` §0 says so.

We do not assert that. We **certify** it, per node, in a form a reviewer can re-verify
offline.

---

## The trace, in seven events

```
turn 0  UserQuery      "Who was X's mentor?"
turn 0  Retrieval      returned=[]  blocked=[m1]        <- I1 doing its job, correctly
turn 1  AgentAnswer    A: "I don't know."   abstained=True
turn 1  Delegation     A -> B  (A abstained)
turn 1  AgentAnswer    B: "<the forgotten fact f>"      <- from B's weights, no context
turn 1  FinalAnswer    "<f>"
turn 1  MemoryWrite    m2, content=f, parent_ids=[]     <- fresh id, no derivation edges
```

`m2` is now in the persistent store, and the next episode will retrieve it.

- **I1 holds** — `m1` was never returned.
- **I2 holds** — nothing was derived from a deleted node.
- **`m2` has no derivation path to any blocked node.**

The system's own invariants declare `m2` clean. It contains `f`.

That is the paper, and it is pinned down offline by
[`tests/unit/test_invariants.py::test_laundered_node_passes_both_invariants`](tests/unit/test_invariants.py)
— **the golden test.** If it ever goes red, the central mechanism argument is wrong, and
you find out on a laptop in week one rather than in September.

---

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[cpu,dev]"

make cpu-all          # Windows: .\tasks.ps1 cpu-all
```

`cpu-all` = ruff + black + mypy + 363 tests, **offline, no GPU, no HF token.** It must be
green before anything touches Colab.

Then:

```bash
python -m rdl.cli env-check                 # first thing in every session
python -m rdl.cli discover-checkpoints      # RUN THIS FIRST ON DAY 1
python -m rdl.cli run-repro --target full   # Days 1-2 sanity gate
```

---

## The condition grid

| | Agent A | Agent B | Write-back | Purpose |
|---|---|---|---|---|
| **C0** | NPO forget10 | — | — | sanity; **empty store** |
| **C1** | NPO forget10 | same checkpoint | disabled | no-write floor |
| **C1W** | NPO forget10 | — | framework_default | **single-agent baseline** |
| **C2** | NPO forget10 | `full` (not unlearned) | framework_default | ceiling / mechanism |
| **C3** | NPO forget10 | **same checkpoint** | framework_default | redundancy control |
| **C3D** | NPO forget10 | **independent NPO forget10** | framework_default | **ensemble treatment** |
| **C3C** | NPO forget10 | independent + A's answer | framework_default | **compositional treatment** |

**The primary estimand is `C3D − C1W`.** Multi-agent write-back minus *single-agent*
write-back — not minus a condition where writing is disabled, which would measure "we
turned writing on". `rdl make-report` applies the gate and **exits non-zero when it
fails**. See [`docs/00b_preregistration_v2.md`](docs/00b_preregistration_v2.md).

---

## The headline metric

`laundering_rate` — of forget-set items recovered from the **persistent store**, the
fraction whose supporting node carries

```
inv1_satisfied and inv2_satisfied and path_to_any_blocked_node is None
```

Not containment. Containment says "the forgotten fact came back", which several papers
already report. Laundering says "it came back *through a path the defence certifies as
safe*", which nothing in the literature reports. Every row in the results table ships
with its `InvariantCertificate` witness.

**Read it against `n_recovered`.** The denominator is items recovered, not all items — a
method that recovers nothing gets an *undefined* rate reported as 0.0 with an explicit
note.

---

## Layout

```
src/rdl/
  memory/       node, store, exact index, blocklists, derivation DAG,
                invariants.py  <- the core artifact
  models/       loader.py (the ONLY place a model is loaded), stub.py (no torch, no network)
  agents/       abstention (3 detectors), delegation (incl. the always_delegate control),
                writer.py (framework_default, with citations)
  orchestrator/ typed events, transcript, episode loop
  eval/         containment, laundering, controls, open-unlearning bridge
  cli/          env-check, discover-checkpoints, run-repro, run-condition, make-report

configs/        env / models / agents / memory / writepolicy / conditions
docs/           00_preregistration.md is v1, FROZEN; 00b_preregistration_v2.md is
                ACTIVE; 04_decisions.md is the ADR log
tests/          unit (fast) | contract (StubLM) | integration (network)
third_party/    open-unlearning, pinned submodule — called, never patched
```

---

## Design invariants

1. **Never fork or vendor `open-unlearning`.** It enters as a pinned submodule. Every
   metric we report must be producible by *their* code on *their* configs.
2. **Every model call goes through `models/loader.py`.** One place decides dtype,
   attention implementation, padding side, and chat template — which is what makes the
   T4-vs-3090-vs-H100 split survivable.
3. **Everything except the forward pass is testable on CPU with no network**, via
   `StubLM`. If a module cannot be tested with `StubLM`, it is coupled wrong.
4. **The transcript is a typed event log, not a string.** Concatenated chat strings make
   `containment(..., surface="any_memory_write")` unimplementable and the claims
   unfalsifiable.
5. **Determinism by default.** Greedy, `batch_size=1`, seeds in three libraries, exact
   retrieval. Batched generation with left-padding changes greedy outputs under fp16.
6. **Results are append-only and content-addressed.**
   `run_id = <UTC timestamp>-<git sha>-<config hash>`.

---

## Hardware

| device | cc | bf16 | FA2 | use |
|---|---|---|---|---|
| Colab T4 | 7.5 | no | no | **eval only** — fp16 + sdpa, never `flash-attn`, never training |
| RTX 3090 | 8.6 | yes | yes | training permitted |
| H100 | 9.0 | yes | yes | training permitted |

`hardware.assert_training_allowed` **refuses** to start training without bf16 unless
`--allow-fp16-training` is passed. Gradient-ascent objectives (GA, NPO) push loss upward
without bound and fp16 `GradScaler` NaNs on them — silent corruption, not a crash.

---

## Where to look first

| question | file |
|---|---|
| What is actually being claimed? | [`docs/00b_preregistration_v2.md`](docs/00b_preregistration_v2.md) (active) |
| What was claimed first, and what changed | [`docs/00_preregistration.md`](docs/00_preregistration.md) (v1, frozen) + §0 of v2 |
| Why is it built this way? | [`docs/04_decisions.md`](docs/04_decisions.md) |
| What could kill it? | [`docs/05_risks.md`](docs/05_risks.md) |
| What are the Day 1–2 numbers? | [`docs/02_repro_targets.md`](docs/02_repro_targets.md) |
| The core artifact | [`src/rdl/memory/invariants.py`](src/rdl/memory/invariants.py) |
| The golden test | [`tests/unit/test_invariants.py`](tests/unit/test_invariants.py) |

---

## The known threat to measurability

**Measurable iff the agents retain residual knowledge of the forget set.** A
perfectly-unlearned agent has nothing to launder.

A real NPO/forget10 checkpoint is not perfectly unlearned — that is why its
`forget_truth_ratio` is 0.70 and not the retain model's 0.63. **The residual is the
effect size.**

Pre-registered in `00b_preregistration_v2.md` §5 and pinned in
`tests/contract/test_condition_c3_stub.py::test_c3_is_unmeasurable_when_b_is_perfectly_unlearned`,
so that a zero on Colab can be **diagnosed** rather than guessed at. A near-zero result is
ambiguous between "no residual" and "broken pipeline"; both agents' individual forgetting
numbers must be reported before anything is concluded.

---

## Licence

MIT — keeps us compatible with `open-unlearning`.
