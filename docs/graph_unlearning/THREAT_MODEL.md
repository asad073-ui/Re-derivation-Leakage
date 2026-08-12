# Threat model

## The system under study

A directed acyclic graph of agents that all reference **one** unlearned checkpoint. They
exchange messages along fixed edges, retrieve from and write to a shared persistent
memory, and may call tools. The memory has already been subjected to the SBU deletion:
the forget content was ingested, deleted through the memory pathway, and its ids
blocklisted. Both SBU invariants hold at the start of every episode. Anything recovered
afterwards is therefore a **re-derivation**, not a failure to delete.

## What we defend against

| Path | Mechanism |
|---|---|
| Direct re-emission | An agent restates the forgotten fact from weights. |
| Paraphrase laundering | A downstream agent re-expresses tainted content so content-only detection no longer fires. |
| Split reconstruction | Two agents each emit an individually-innocuous clue; a join node combines them. |
| Write laundering | An agent's parametric answer is written as a new memory node with an empty parent list, so no id points back at the deleted content. |
| Retrieval re-entry | A later episode retrieves what an earlier one wrote. |
| Tag stripping | The scope tag on a stored node is missing or removed. |
| Tool re-entry | A tool response carries the content back into the graph. |

## What we do not defend against, and say so

* **The model's own parametric knowledge.** GraphForget contains content; it does not
  make the checkpoint forget. `raw_message_leak` is reported precisely so this limit is
  visible in every table.
* **A detector that never fires.** Every guarantee is conditional on detection. The
  detector's measured recall and FPR are reported next to every result, and an
  uncalibrated run is stamped diagnostic.
* **Adversarial encodings designed against this detector.** Steganographic or
  cipher-encoded restatements are out of scope for v1.
* **A malicious agent implementation.** Agents are assumed to be the same checkpoint
  behind the same harness; we do not model an agent that bypasses the envelope layer.
* **Cyclic topologies.** DAGs only in v1: a ring needs an explicit round counter before
  provenance closure is even well defined.

## Assumptions

1. Every logical agent shares one physical checkpoint. They are **not** independently
   unlearned models, and the manifest records this so no report can claim otherwise.
2. Persistent writes are staged and become visible only after the episode
   (`visibility: after_episode`). Otherwise agent-to-agent edge flow and memory-mediated
   flow are mixed in one number and neither is attributable. `immediate` is a separate,
   separately-reported ablation.
3. The store resets to the same post-deletion snapshot before every item
   (`store_scope: per_item`), which is what the paired item-level bootstrap assumes.
4. The controlled challenges' injected content is constructed by the harness from the
   gold answer. It measures containment when the content is definitely present; it says
   nothing about how often a natural system leaks.

## The runtime never holds the forgotten answers

The concept registry contains ids, aliases, question text and question paraphrases —
never answers. `ConceptRegistry.from_questions` raises if handed one. A defence that had
to keep the gold answers in memory to recognise them would preserve exactly the
information the system claims to have forgotten, and a reviewer would be right to say so.

The gold answers exist in two places only: the offline evaluator, which never runs in the
same process as generation; and the controlled-challenge injector, which sets
`uses_gold_answers: true` in the run manifest and in the report header.
