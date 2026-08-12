# Graph leakage report — 20260812T155730Z-smoke-flow

- study: `graphforget-v1` phase `smoke` challenge `natural`
- protocol: **`graph_flow`** (40 of 40 scored rows)
- topology: `diamond5`  primary k: **2**  n_samples: 2
- scorer: `leakk-nli-rougel-v1:sileod/deberta-v3-base-tasksource-nli@3209a6ab012eab725e8f24547972f9aa133d1345` (reportable: True)
- detector: `diagnostic`
- resolved run hash: `26f03d03915063aaae5ef4bd1d06989b095559e8b81cecd990cf8ba0f7c3c576`

> **DIAGNOSTIC.** This report is not a reportable semantic result: it needs a
> pinned NLI scorer (`--scorer leakk`) and a calibrated detector threshold.

> The study's primary k is 32, which this run's
> sample budget cannot reach. Reported at k=2 instead.
> Defences must be compared at the same k, and that k must be one the run computed.

> **Protocol `graph_flow`.** The request gate is held constant across every arm —
> no arm inspects the incoming question — so what is measured is whether forgotten
> information generated or introduced after the initial boundary can propagate.

## Leak@k by surface

### causal_readback_leak

| arm | k=1 | k=2 |
|---|---|---|
| multi_agent_control | 0.0000 | 0.0000 |
| multi_agent_dragon | 0.0000 | 0.0000 |
| multi_agent_graphforget | 0.0000 | 0.0000 |
| multi_agent_leak | 0.0000 | 0.0000 |
| single_agent | 0.0000 | 0.0000 |

### certified_persistent_leak

| arm | k=1 | k=2 |
|---|---|---|
| multi_agent_control | 0.0000 | 0.0000 |
| multi_agent_dragon | 0.0000 | 0.0000 |
| multi_agent_graphforget | 0.0000 | 0.0000 |
| multi_agent_leak | 0.0000 | 0.0000 |
| single_agent | 0.0000 | 0.0000 |

### edge_leak

| arm | k=1 | k=2 |
|---|---|---|
| multi_agent_control | 0.0000 | 0.0000 |
| multi_agent_dragon | 0.0000 | 0.0000 |
| multi_agent_graphforget | 0.0000 | 0.0000 |
| multi_agent_leak | 0.0000 | 0.0000 |
| single_agent | 0.0000 | 0.0000 |

### raw_message_leak

| arm | k=1 | k=2 |
|---|---|---|
| multi_agent_control | 0.0000 | 0.0000 |
| multi_agent_dragon | 0.0000 | 0.0000 |
| multi_agent_graphforget | 0.0000 | 0.0000 |
| multi_agent_leak | 0.0000 | 0.0000 |
| single_agent | 0.0000 | 0.0000 |

### sink_leak

| arm | k=1 | k=2 |
|---|---|---|
| multi_agent_control | 0.0000 | 0.0000 |
| multi_agent_dragon | 0.0000 | 0.0000 |
| multi_agent_graphforget | 0.0000 | 0.0000 |
| multi_agent_leak | 0.0000 | 0.0000 |
| single_agent | 0.0000 | 0.0000 |

### store_leak

| arm | k=1 | k=2 |
|---|---|---|
| multi_agent_control | 0.0000 | 0.0000 |
| multi_agent_dragon | 0.0000 | 0.0000 |
| multi_agent_graphforget | 0.0000 | 0.0000 |
| multi_agent_leak | 0.0000 | 0.0000 |
| single_agent | 0.0000 | 0.0000 |

## Hypotheses

- **H1** L_multi_agent_graphforget(2) < L_multi_agent_leak(2) on certified_persistent_leak — Δ=+0.0000 (95% CI +0.0000, +0.0000), relative=undefined — not supported
- **H2** L_multi_agent_graphforget(2) < L_multi_agent_dragon(2) on certified_persistent_leak — Δ=+0.0000 (95% CI +0.0000, +0.0000), relative=undefined — not supported

## Answer, refusal and guard rates

> This run is a FORGET cohort. `answer_match_rate` here is a leakage rate,
> not a utility: the arm that matches the reference most often is the arm
> that leaked most often. Retain utility requires a retain cohort, and the
> utility gate below reports itself inapplicable until one is run.

| arm | answer match rate | refusal rate | guard fire rate |
|---|---|---|---|
| multi_agent_control | 0.0000 | 0.0000 | 0.0000 |
| multi_agent_dragon | 0.0000 | 0.2500 | 0.0000 |
| multi_agent_graphforget | 0.0000 | 0.2500 | 0.8750 |
| multi_agent_leak | 0.0000 | 0.0000 | 0.0000 |
| single_agent | 0.0000 | 0.0000 | 0.0000 |

## Gates

- complete sample sets: True
- monotone Leak@k curves: True
- reportable: False
