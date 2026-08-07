# Phase 0 protocol — exact condition definitions

The authoritative statement of what each arm does. The *criteria* live in
[`00b_preregistration_v2.md`](00b_preregistration_v2.md);
[`00_preregistration.md`](00_preregistration.md) is v1, frozen and superseded. This file
describes the mechanics and may be clarified (with a dated ADR) if a description turns
out to be ambiguous.

**The grid changed on 2026-08-07.** C3 was one checkpoint loaded into both agent slots,
so agent B reproduced agent A byte for byte, and the baseline C1 had write-back disabled,
so its store recall was structurally zero. See ADR-0017 and ADR-0018.

---

## Shared setup

1. **Ingest.** Every forget10 item is written to the shared `MemoryStore` as one node,
   `source_kind="ingest"`, content = `"{question} {answer}"`.
   **Except C0**, whose `memory.ingest_forget_set` is false: a "bare checkpoint" arm that
   can retrieve the ground-truth answers out of memory is measuring the fixture, not the
   checkpoint (ADR-0023).
2. **Delete via the SBU memory pathway.** For each ingested node:
   `store.delete(node_id, blocklist)` — which marks it deleted, prunes its derivation
   closure (decrement once per unique node; mark outdated when refcount hits zero or all
   parents are dead), and adds its id to the `IDBlocklist`.
3. **Snapshot.** `store.snapshot()` before any episode, so the between-condition
   comparison is not confounded by store initialisation. A **further snapshot is taken
   after every episode**, and each episode is scored against its own — scoring against
   the final store credits episode *i* with what episode *i+n* wrote (ADR-0020).
4. **Order.** Episode order is permuted per seed. Greedy decoding over a fixed order
   makes five seeds five copies of one number (ADR-0019).

After step 2, **both invariants hold**, and they continue to hold throughout every
condition. That is the point: the leak is not a failure to implement SBU.

Deleted nodes remain in the index as tombstones — see ADR-0003.

---

## Episode loop

Per question, `orchestrator.loop.run_episode` emits:

```
turn 0  UserQuery
turn 0  Retrieval      agent A, under the blocklist
turn 1  AgentAnswer    agent A
turn 1  Delegation     if the delegation policy fires
turn 1  Retrieval      agent B, under the SAME blocklist
turn 1  AgentAnswer    agent B
turn 1  FinalAnswer    B's answer if B did not abstain, else A's
turn 1  MemoryWrite    if the write policy fires
```

Agent B sees the shared store under the **same** blocklist. Giving the delegate
privileged access would make the leak an artefact of our plumbing.

Episodes run against **one persistent store, not reset between questions.**
Cross-episode persistence is the mechanism under test.

---

## The conditions

### C0 — sanity

Agent A alone. **Empty store**, no blocklist, no delegation, no write-back. Must behave
like the bare NPO/forget10 checkpoint. If C0 does not reproduce the Days 1–2 numbers,
nothing downstream means anything.

### C1 — no-write floor

Two agents, both NPO/forget10. Blocklist enforced. **`writepolicy: disabled`** —
enforced by a cross-field validator in `config.py`.

Expected: retrieval blocked every time, nothing persisted,
`SysRecall@5(persistent store)` **structurally zero**. That is the problem with using it
as the baseline: subtracting a structural zero measures "we enabled writing", which is
true by construction.

### C1W — single-agent write-back. **THE BASELINE.**

Agent A alone, blocklist enforced, write-back on. One unlearned model recovering whatever
its own residual knowledge allows and persisting it — parametric-to-memory backflow by a
single agent, which is the phenomenon SBU already names.

**The primary estimand is measured against this arm**, not against C1. Whatever the
multi-agent arms show has to beat one agent with a memory.

### C2 — ceiling / mechanism demo. **NOT THE TEST.**

Agent B is `_full`, never unlearned. It simply knows the answers, so containment
approaches ceiling and any "> 15 point" bar over C1's zero floor is met trivially.

What C2 legitimately establishes: the write path transports content from an agent's
parametric memory into the shared persistent store, through a node that both invariants
certify as clean. Nothing more.

A note on what shows up here and is easy to misread: the **first** write has no parents
(nothing was retrievable). **Later** writes may cite *earlier laundered nodes*, since
those accumulate and are retrievable. That is not a provenance leak — the resulting
derivation subtree is entirely disconnected from the blocked ids, which is exactly why
every node in it still certifies as clean.

### C3 — redundancy control. **NOT THE TEST.**

Agent B is agent A's checkpoint, loaded a second time. Greedy decoding, same question,
same retrieval context, same weights — so B returns exactly what A returned. This is one
model queried twice.

What C3 legitimately establishes: how much of any apparent multi-agent effect is
explained by asking the same model twice and writing both answers. If `C3D − C1W` is no
larger than `C3 − C1W`, there is no multi-agent finding and this arm is what proves it.

### C3D — two independently unlearned agents. **THE TREATMENT.**

Agent A: forget10 removed at lr1e-05 / beta0.1. Agent B: forget10 removed by a *separate*
NPO run at lr2e-05 / beta0.5. Same forget set, different optimisation trajectory, so the
two residuals are not identical by construction. Blocklist enforced, write-back on.

B answers the question **in isolation** — C3D is the ensemble arm: two independent draws,
with no reasoning passed between them.

### C3C — compositional re-derivation

C3D plus one change: agent A's answer is handed to agent B as prompt material
(`EpisodePolicies.pass_primary_answer_to_secondary`). A's turn is **never** written as a
memory node and never becomes a `parent_id` — recording an agent's utterance as a node
would fabricate exactly the provenance edge whose absence is the finding.

`C3C − C3D` is the single-variable contrast that licenses the word "re-derivation". If it
is ≈ 0, the effect is ensembling and the word comes out of the paper.

---

Everything hinges on both agents' residual knowledge — see
[`00b_preregistration_v2.md`](00b_preregistration_v2.md) §5 and R2 in
[`05_risks.md`](05_risks.md). A result of ≈ 0 is **ambiguous** and must not be reported as
a null result until both residuals have been measured directly (§3.5).

---

## Control arms

Run automatically by `run_condition.py` unless `--no-controls` is passed. It must never
be passed for a reported result.

All four are computed by `eval.controls.compute_controls`, whose output goes into
`condition_report.json` under `control_reports` and is read by the gate. It was
documented here and never called — only the `always_delegate` arm actually ran.

| control | what it rules out |
|---|---|
| `always_delegate` | that the effect tracks agent A's utility collapse rather than forgetting |
| retain-set question set | that the containment metric fires on content that was never unlearned |
| `delegation_rate(forget)` vs `(retain)` | that routing is not selectively triggered by forgetting |
| agent A alone vs. the system, on retain | that the effect is plain ensemble benefit |

---

## Running it

```bash
# one condition
python -m rdl.cli run-condition --condition configs/conditions/C3D.yaml --seeds 5

# the whole grid, then the report + gate (exits non-zero when the gate fails)
bash scripts/03_run_phase0_grid.sh
```

Offline dry-run with stub models, useful for checking plumbing without a GPU:

```bash
python -m rdl.cli run-condition --condition configs/conditions/C3D.yaml --seeds 2 \
  --allow-fixture \
  --set env.name=local_cpu \
  --set models.tofu_llama32_1b_npo_forget10.kind=stub \
  --set models.tofu_llama32_1b_npo_forget10.repo_id=null \
  --set models.tofu_llama32_1b_npo_forget10_indep.kind=stub \
  --set models.tofu_llama32_1b_npo_forget10_indep.repo_id=null
```

`--allow-fixture` is required and is not a convenience: without it the run loads the real
400-item TOFU split. The eight-item fixture is a plumbing check, its report is stamped
`is_real_data: false`, and `make-report` treats that as a blocker (ADR-0024).

---

## Outputs

`results/<run_id>/` where `run_id = <UTC timestamp>-<git sha>-<config hash>`:

- `condition_report.json` — aggregated metrics, per-seed detail, the full resolved config,
  the hardware profile, and the control arm
- `transcripts_seed<N>.jsonl` — one typed event per line
- one line appended to the committed `results/manifest.jsonl`

Nothing overwrites. `run_dir()` refuses to reuse an existing directory.
