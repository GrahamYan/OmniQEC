# Fig.3 confirmation protocol (pre-registered 2026-10-13)

## Why a separate stage

The Phase B search (task = `cnot_dense`, Fig.2 v3.7.1 settings) ranks codes on 25k shots
per basis with the 100-error racing stop in both the search and the built-in final stage,
i.e. a CI half-width of about 20% per code. The Fig.3 claim at a point is a 10–20% gap
between two specific codes under the gate task, so it needs its own measurement. This is
the usual division of labour for a search on a stochastic objective: the search uses a
cheap estimate, the few elites are re-measured independently with high statistics, and the
re-measurement is what gets reported (code-discovery papers such as Bravyi et al. 2024
screen with algebraic parameters and report circuit-level Monte Carlo for a handful of
codes with error bars; evolutionary / RL code searches re-evaluate the selected codes;
experimental papers report logical error per cycle from a stated number of shots and
failures). Search-stage numbers never appear as final values.

## What is measured

Per budget point N, the pool (`hpc/fig3_confirm_pool.py`):

| role | code |
|---|---|
| `memory_champion` | the Fig.2 memory champion of the point (lowest final-stage plpr across the Fig.2 arms) |
| `gate_champion` | the Phase B gate-task champion (lowest task plpr across the three arms, final-stage rankings) |
| `gate_arm_champion:<arm>` | each arm's champion when it is of a (w, k) class not already in the pool |
| `phaseb_extra` | the best Phase B code of a (w, k) class not yet in the pool (`--phaseb-extra 1`), so every point measures at least one lighter / heavier class next to the champions for panel c |

Circuits (pilot pipeline `hpc/cnot_pilot.py`, Fig.2's SEC / noise / telescoping decoder,
T = 12 rounds, two blocks):

* `idle`: 12 rounds, no gate (the Phase B idle baseline; equals Fig.2's memory circuit per block);
* `r1`: one transversal CNOT after each of rounds 1..11 (the Phase B objective);
* optional `r6` (CNOT after round 6) and `r2` (after rounds 2, 4, 6, 8, 10) at the points
  shown in panel c.

`idle` and `r1` run in one job so every code gets the same synthesised SEC schedule and the
same shard seeds in both circuits.

## Metric

Fig.2's per-logical-qubit-per-round convention applied to the task circuit (user decision
10-12; Ruiz et al. 2025 "logical error per cycle per logical qubit"):
block = 1 − (1 − p_anyX)(1 − p_anyZ), p = 1 − (1 − block)^(1/(2k·T)); the added error per
CNOT per logical qubit is ε = (p_gate − p_idle)·T / n_gates.

## Statistics and stopping rule

* Shard bootstrap (2000 draws; 625 shots per shard in every stage, so shards are
  exchangeable) gives every CI and the pairwise order probabilities
  P_gate = P[p_gate(G) < p_gate(M)] and P_idle = P[p_idle(M) < p_idle(G)] for the memory
  champion M and each gate-role code G.
* Shots per basis are added in stages, 50k → 100k → 200k (`RESUME=1` keeps the finished
  shards; the new shards draw fresh seeds). After each stage `hpc/fig3_confirm_analyze.py`
  gives one of:
  * **different**: P_gate ≥ 0.99 and P_idle ≥ 0.99 — the gate-optimal code is not the
    memory-optimal one;
  * **same**: P_gate ≤ 0.01 — the memory champion also wins the gate task;
  * **extend**: otherwise, below the cap — run the next stage;
  * **tie_at_cap**: otherwise at 200k — reported as a crossover point with the gap and its CI.
* The thresholds and the cap are fixed here before any confirmation data exists; they are
  the pilot's Go criteria (brief 2026-10-11 §1) with the cap added.

## Cost

Measured decode cost on the gate circuit: 0.3 CPU-s/shot for the b144 w4/w5 codes, 1.4 for
w6, 0.6–2.4 CPU-s/shot for the b220 finalists. One stage of the main job (idle + r1, both
bases) for 4 codes is therefore 0.3–4 node-hours at 50k shots, double per stage; the full
ladder at b220 is about 10–15 node-hours (≈2k SU), well inside an 8 h job per stage.

## Files

* `hpc/fig3_confirm_pool.py` — pool builder (stdlib; runs on a login node; `--fig2-only`
  carries a point's Fig.2 finalists from another run tree).
* `hpc/nscc_cnot_pilot.pbs` (via a site-specific submit script that is not part of the
  open-source tree) — runs `main` (and `r6`/`r2`) for a stage; outputs under
  `<SCRATCH>/confirm/b<N>_*`.
* `hpc/fig3_confirm_analyze.py` — report (md + json) and the verdict above.
* `docs/fig3_confirm_pool_b<N>.json`, `docs/fig2_b<N>_final_slow_run_config.json` — the
  pool and the Fig.2 final-stage run config of each point (the pilot reads seed, noise,
  SEC policy and decoder settings from the config).
