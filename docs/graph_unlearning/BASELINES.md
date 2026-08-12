# Baselines, stated honestly

## DRAGON

DRAGON was published at ICLR 2026 and is the right inference-time baseline for this
work. Its public repository, however, is a code skeleton: the required detector and
guard checkpoints are not released and the WMDP implementation is incomplete. The paper
is usable; the released code cannot currently provide a complete faithful reproduction.

Therefore this repository implements and reports a

> **DRAGON-style node-local baseline**

and never calls it a reproduction. The repository is not copied into this codebase.

### What our implementation does

`src/rdl/defenses/dragon_style.py`.

* Runs at **every agent's complete incoming context**. This is the *stronger* of the two
  possible generalisations to a graph and is what the main comparison uses. The weaker
  `apply_at: external_prompt_only` variant (guard only the query entering the graph) is
  available and reported as an additional ablation.
* Scores the concatenated node input as one string. That means it **does** catch clues
  that happen to be co-located in one context — we did not weaken it to manufacture a
  win.
* Also guards the released final answer, which is one more model boundary. Generous to
  the baseline on purpose.
* **`guard_action: guard_prompt` is the default and the primary baseline** — a guard
  instruction is appended to the system message and the node generates normally. This is
  what DRAGON describes: detect, then modify the inference context.
* `guard_action: refuse` makes the node emit a fixed safe refusal instead of generating.
  That is a strictly stronger intervention than anything DRAGON performs, so it ships as
  a **separate** arm (`dragon_style_refuse` / `multi_agent_dragon_refuse`) and is
  reported as a **strong upper bound on node-local guarding**, never as a faithful
  reproduction. It was briefly the default; calling it "DRAGON" overstated the baseline
  (DECISIONS.md GU-0017). The two bracket the real baseline: the prompt guard is the
  weaker bound — on a checkpoint that ignores the instruction it is a no-op — and the
  refusal is the stronger.
* `implementation: sft_checkpoint` **raises**, rather than silently substituting a
  template for weights that were never released.

### What it does not do, because DRAGON does not

* inherit Forget IDs from provenance — each call is independent;
* score subsets of the input separately from the whole;
* guard persistent writes, retrieval, or later-episode readback.

Note what is **not** on that list: *joint evidence over several parents*. The guard
scores the node's complete incoming context as one string, so two clues that arrive
together at one node are both in what it scores and it does catch them. See "Where the
real difference lies" below, and GU-0026.

`DragonStyleDefense` strips inherited scopes from any envelope it touches
(`_untagged`), so the baseline cannot accidentally benefit from GraphForget's
bookkeeping.

### The matched-subset variant

`dragon_style_subsets` / `multi_agent_dragon_subsets` (GU-0026). Same node-local guard,
given **GraphForget's exact subset battery**: each parent message alone, the parents
together, the parents plus retrieved memory, and the whole context. Still inherits
nothing, still guards no edge, write or retrieval.

It exists because a difference between `multi_agent_dragon` and `multi_agent_graphforget`
otherwise mixes two causes — how finely each side chops up one node's input, and Forget-ID
propagation across surfaces. Only the second is the contribution. This arm holds the first
constant, and it is what makes the propagation claim falsifiable: if GraphForget's
reduction largely disappears against it, the mechanism doing the work was subset scoring
and the paper has to say so.

It is **not** the primary baseline. DRAGON as published scores one context, and a baseline
that does more than the paper describes is not the paper.

## Shared detector

**Both guarded arms are handed the same `SemanticConceptDetector` instance.**
`studies/graph_leak/arms.py` constructs one detector and passes it to both;
`tests/defenses/test_concept_registry.py::test_dragon_and_graphforget_share_one_detector_object`
asserts object identity. Without this, "our method leaks less" would be equally
consistent with "our detector is better", and that is the single most likely reviewer
objection to the whole study.

The config additionally refuses to load a `dragon_style` defence with
`propagate_scope: true` — the baseline must not be given the mechanism under test.

## Capability comparison

| Capability | DRAGON-style baseline | GraphForget |
|---|---|---|
| Incoming prompt detection | yes | yes |
| Every agent input | yes (adapted) | yes |
| Semantic edge inspection | no persistent state | yes |
| Forget-ID propagation | no | yes |
| Split clues arriving together at one node | **yes** — whole-context scoring | yes |
| Split clues a diluted whole-context score misses | no (yes with `score_subsets`) | yes, subset scoring |
| Memory-write protection | not its mechanism | yes |
| Retrieval protection | limited | yes, with rescan of untagged nodes |
| Later-episode causal protection | not its focus | yes |
| Provenance certificates | no graph-wide certificate | yes |

## Where the real difference lies

Being precise, because the "split clues" story is easy to overstate. Since our DRAGON
implementation scores the whole concatenated context, it already catches two clues that
arrive together at one node. The genuine differences are:

1. **Subset scoring.** A long query dilutes the embedding of the whole context; scoring
   the parent messages as their own subset recovers the signal. Demonstrated in
   `tests/defenses/test_split_clues.py::test_subset_scoring_can_beat_whole_context_scoring`.
2. **Inheritance across hops.** When a downstream node re-expresses tainted content in
   its own words, the concatenation no longer scores — but the scope was inherited, so
   the guard still fires. `test_scope_propagation.py::test_scope_survives_three_hops_of_paraphrase`.
3. **Persistent writes.** The node-local baseline writes its content, including its
   refusals, to memory unguarded. `test_known_graph_leak.py::test_the_node_local_baseline_leaves_persistent_memory_open`.
4. **Retrieval and later episodes.** Nothing in a node-local guard stops a later episode
   from retrieving what an earlier one wrote.

### The claim we do not make

> ~~DRAGON cannot detect split clues because it is node-local.~~

This is **false** and must not appear in the paper (GU-0026). Our DRAGON-style baseline
runs at `apply_at: every_agent_input` and scores the node's complete incoming context —
query, all parent messages, retrieved memory — as one string. Clues that arrive together
at one node are in that string. Node-locality is a statement about *where* the guard runs,
not about how much evidence any one call receives, and a node that joins two parents
receives both.

The defensible claim is narrower:

> GraphForget adds persistent Forget-ID propagation and enforcement across edges, memory
> writes, retrieval and final release, while the DRAGON-style baseline guards
> model-input boundaries.

A genuine cross-call accumulation claim would need state accumulated over calls **no
single one of which receives the complete evidence** — for example clues that reach the
sink through different episodes or different memory writes. The current diamond topology
does not produce that: every split clue rejoins inside one node's input. Until a topology
that does exists, the propagation claim rests on the surfaces, and the matched-subset arm
is what isolates it.

### The three counters, and what each is worth

| Counter | Meaning | What it licenses |
|---|---|---|
| `node_local_visible_hits` | the whole-context view fired | nothing — the baseline catches these |
| `subset_only_hits` | a strict subset fired and the whole-context view did not | subset scoring beats whole-context scoring; `dragon_style_subsets` also gets this |
| `accumulated_only_hits` | a **combination of parents** fired while no individual input, no query, **and not the whole-context view** did | the only one that counts as "reconstructed from several parents and missed by the matched baseline" |

`accumulated_only` previously omitted the `and not the whole-context view` clause, so it
counted cases the baseline catches and the claim was unfalsifiable by construction. The
query is excluded from all three explicitly: when the user asks about a forgotten concept,
every combination containing the query fires, and counting that would inflate the result
with cases a node-local guard catches trivially.

## The protocol the comparison runs under

Both baselines are compared under **both** protocols, separately (PROTOCOL_v1 §2b).

Under `end_to_end_safety` the request gate is part of the defence, and since the forget
question is one of the detector's own prototypes, both guarded arms refuse at the root
and score identically clean. That comparison is real — it says the deployed system does
not release the information — but it says nothing about the graph, because no mechanism
downstream of the root was ever reached.

Under `graph_flow` no arm inspects the question, so the guarded arms must actually
contain what the graph carries. **This is the protocol the GraphForget claim rests on**,
and it is where the capability table below becomes testable rather than notional.

## Other baselines and ablations

| Name | What it isolates |
|---|---|
| `single_agent` | Everything the graph adds on top of one agent. |
| `multi_agent_control` | The peer-message wrapper, holding content concept-disjoint. |
| `multi_agent_dragon_refuse` | The upper bound on node-local guarding. |
| `multi_agent_dragon_subsets` | Forget-ID propagation and multi-surface enforcement, with subset scoring held constant. |
| `edge_cut` | What deleting communication buys — leakage falls, collaboration falls. An **ablation**, not a method. |
| graphforget with `propagate_forget_ids: false` | Semantic detection only. |
| graphforget with `semantic_detection: false` | Provenance only. |
| graphforget with `rescan_untagged_memory: false` | Whether tags alone suffice. |
| dragon with `apply_at: external_prompt_only` | The weaker node-local reading. |
