# Final Day-2 evidence audit

The GPU evidence and the post-hoc semantic analysis were rechecked against their saved
400-item C3C and C3S handoff artifacts. This addendum corrects the interpretation of one
secondary diagnostic; it does not modify raw evidence, semantic labels, or the frozen
preregistered verdict.

## Verified evidence

- C3C and C3S each contain exactly 400 unique target items.
- All C3C handoff texts and hashes equal their saved Agent-A answers.
- Each C3S target's saved Agent-A answer byte-matches its C3C counterpart, while the
  text B actually received is a complete cross-author derangement: all 400 distinct
  sources are `(target + 20) mod 400`, with no fixed points or same-author pairs.
- Both independent primary judges find lower B semantic accuracy in C3C than C3S
  (C3C-C3S: -5.25 and -2.00 percentage points). The adjudicated estimate is -3.25
  points with a 95% author-clustered interval of [-6.0, -0.5]. Potential reconstruction
  is likewise lower in C3C (-4.75 points, interval [-7.5, -2.0]).
- Raw token-F1 corroborates a communication effect without relying on semantic judges:
  B-to-target-A overlap is 0.4468 in C3C versus 0.3260 against that same unseen target-A
  answer in C3S (difference +0.1208; 93 versus 9 cases at F1 >= 0.6).

## Correct interpretation

The evidence supports that B receives and is influenced by A's same-item response, but
does **not** support correct re-derivation of forgotten facts. The preregistered routing
result remains negative because abstention routing did not selectively activate.

The previously named `same_false_claim_propagation` row is retained as a received-handoff
co-occurrence diagnostic only. It compares B with a same-question handoff in C3C but a
different-question handoff in C3S, so its 0.378 versus 0.030 values are **not** a causal
target-A transmission rate. A future CPU-only rescore must compare the identical target-A
answer with B in both arms, using mixed condition-blind batches and a third distinct judge.
