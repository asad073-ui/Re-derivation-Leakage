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
* `guard_action: refuse` is the default: the node emits a fixed safe refusal instead of
  generating. Deterministic and model-independent. A prompt-only guard on a checkpoint
  that ignores instructions is a no-op, and a baseline that is a no-op is not a baseline.
  `guard_prompt` (append a reasoning-guard instruction, then generate) is available for
  instruction-following checkpoints and is the faithful "template reasoning" mode.
* `implementation: sft_checkpoint` **raises**, rather than silently substituting a
  template for weights that were never released.

### What it does not do, because DRAGON does not

* inherit Forget IDs from provenance — each call is independent;
* score subsets of the input separately from the whole;
* guard persistent writes, retrieval, or later-episode readback.

`DragonStyleDefense` strips inherited scopes from any envelope it touches
(`_untagged`), so the baseline cannot accidentally benefit from GraphForget's
bookkeeping.

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
| Split clues from several parents | partly — whole-context scoring | explicitly tested, subset scoring |
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

`accumulated_only` counts only cases where a **combination** of parent messages fires
and neither any individual input **nor the query on its own** does. The query is excluded
explicitly: when the user asks about a forgotten concept, every combination containing
the query fires, and counting that as "reconstructed from several parents" would inflate
the split-clue result with cases a node-local guard catches trivially.

## Other baselines and ablations

| Name | What it isolates |
|---|---|
| `single_agent` | Everything the graph adds on top of one agent. |
| `multi_agent_control` | The peer-message wrapper, holding content concept-disjoint. |
| `edge_cut` | What deleting communication buys — leakage falls, collaboration falls. An **ablation**, not a method. |
| graphforget with `propagate_forget_ids: false` | Semantic detection only. |
| graphforget with `semantic_detection: false` | Provenance only. |
| graphforget with `rescan_untagged_memory: false` | Whether tags alone suffice. |
| dragon with `apply_at: external_prompt_only` | The weaker node-local reading. |
