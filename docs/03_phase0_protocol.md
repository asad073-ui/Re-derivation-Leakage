# Phase 0 protocol — exact condition definitions

The authoritative statement of what C0–C3 do. The *criteria* live in
[`00_preregistration.md`](00_preregistration.md), which is frozen; this file describes the
mechanics and may be clarified (with a dated ADR) if a description turns out to be
ambiguous.

---

## Shared setup, identical across all four conditions

1. **Ingest.** Every forget10 item is written to the shared `MemoryStore` as one node,
   `source_kind="ingest"`, content = `"{question} {answer}"`.
2. **Delete via the SBU memory pathway.** For each ingested node:
   `store.delete(node_id, blocklist)` — which marks it deleted, prunes its derivation
   closure (decrement once per unique node; mark outdated when refcount hits zero or all
   parents are dead), and adds its id to the `IDBlocklist`.
3. **Snapshot.** `store.snapshot()` before any episode. C1/C2/C3 all restore from the
   same snapshot, so the between-condition comparison is not confounded by store
   initialisation.

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

Agent A alone. No blocklist, no delegation, no write-back. Must behave like the bare
NPO/forget10 checkpoint. If C0 does not reproduce the Days 1–2 numbers, nothing
downstream means anything.

### C1 — leakage floor

Two agents, both NPO/forget10. Blocklist enforced. **`writepolicy: disabled`** —
enforced by a cross-field validator in `config.py`, because enabling write-back here
would silently destroy the baseline the primary gate subtracts.

Expected: retrieval blocked every time, both agents abstain, nothing persisted,
`SysRecall@5(persistent store) ≈ 0`.

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

### C3 — the condition that decides the paper

Two agents, both NPO/forget10, **same forget set**. Blocklist enforced. Write-back on
the framework default.

Everything hinges on agent B's residual knowledge — see
[`00_preregistration.md`](00_preregistration.md) §6 and R2 in
[`05_risks.md`](05_risks.md). A C3 of ≈ 0 is **ambiguous** and must not be reported as a
null result until B's residual has been measured directly.

---

## Control arms

Run automatically by `run_condition.py` unless `--no-controls` is passed. It must never
be passed for a reported result.

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
python -m rdl.cli run-condition --condition configs/conditions/C3.yaml --seeds 5

# the whole grid, then the report
bash scripts/03_run_phase0_grid.sh
python -m rdl.cli make-report
```

Offline dry-run with stub models, useful for checking plumbing without a GPU:

```bash
python -m rdl.cli run-condition --condition configs/conditions/C3.yaml --seeds 2 \
  --set env.name=local_cpu \
  --set models.tofu_llama32_1b_npo_forget10.kind=stub \
  --set models.tofu_llama32_1b_npo_forget10.repo_id=null
```

---

## Outputs

`results/<run_id>/` where `run_id = <UTC timestamp>-<git sha>-<config hash>`:

- `condition_report.json` — aggregated metrics, per-seed detail, the full resolved config,
  the hardware profile, and the control arm
- `transcripts_seed<N>.jsonl` — one typed event per line
- one line appended to the committed `results/manifest.jsonl`

Nothing overwrites. `run_dir()` refuses to reuse an existing directory.
