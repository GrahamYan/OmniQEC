# omniqec_agentic

LLM-driven stabilizer-code discovery framework — the "New Fig.2 Experiment"
architecture. Start at `docs/04_architecture.md` — the as-built snapshot of
what actually runs. `docs/01_code_family_design.md`, `docs/02_sec_design.md`,
`docs/03_workflow_design.md` and `docs/03a_agent_prompts.md` are the frozen
design specs behind it; their bodies are never rewritten, only amended by
appended implementation-revision sections. Authority order: **code > `docs/04` > design
specs**. This README is an orientation map written during development, not a spec; the
settings used in the paper are in `docs/final_protocol_20261106.md` and
`docs/paper_alignment_*.md`, which take precedence where this map differs.

## What it does

One campaign = one **total-qubit budget point**. An LLM proposes
`construct(rng)` programs over a mixed search space of four code-family
grammars; every distinct code is proxy-scored fast, and promoted candidates
are measured at circuit level: a ranked Left-Right syndrome-extraction
circuit (SBB), noisy-circuit sampling, and the beam-search decoder, against a
resident bivariate-bicycle (BB) baseline under the identical protocol. The
goal is codes that beat BB on logical error rate per logical qubit per round
(paper A4/A5) with CI separation.

## Code families (01)

| family | genotype |
|---|---|
| `bb` | bivariate bicycle: `ell, m`, monomial lists `A`, `B` |
| `hgp` | hypergraph product of two classical seed matrices `H1`, `H2` (`H1 == H2` allowed) |
| `lp_abelian` | lifted product over an abelian group algebra (exponent tuples, base matrices up to 2x2) |
| `lp_nonabelian` | lifted product over the non-abelian group catalog (dihedral / A4 / S4 / metacyclic / direct products, order <= 72) |

Clifford deformation (`sector_h`) is a config switch, default off.

## Measurement protocol (02)

- **SEC = Left-Right Circuit** (`sec/`): minimal edge colorings of the four
  sector subgraphs, two windows + staggered Z cycles, unconditional validity;
  candidate schedules ranked by GF2 Δ=1 elimination → τA → extended-matrix
  ISD probes (`d_circ_ext` evidence channel). Pinned segment-by-segment
  against the SBB Zenodo golden circuits — gate/noise placement plus
  detector/observable counts, ignoring golden's empty no-op noise
  instructions (`tests/golden/`).
- **Noise profiles** (`verification/noise.py`): `uniform` (CX), `si1000`
  (CZ + window-level H frames), `helios` (native RZZ Pauli vector), each with
  a versioned lowering in `sec/stim_builder.py` (02 §6.3, §9.2).
- **Sampling** = noisy-circuit sampling (`sampling_source="noisy_circuit"`);
  decoders consume faithful undecomposed DEM matrices.
- **Decoders**: beam search B=64 (paper beam64 hyperparameters; deterministic)
  as primary, BP+LSD (paper config, `lsd_method` pinned) as the trusted
  fallback; BPOSD/RelayBP remain options. `qualify-decoders` measures
  agreement/throughput on the versioned set in `docs/qualification/`.

## Workflow (03)

```text
workflow/     LangGraph campaign graph (the ONLY langgraph import):
              intake -> [orchestrator_turn -> execute_plan -> reconcile]* ->
              final_slow -> report; SqliteSaver checkpoints, --resume with
              protocol-snapshot validation
orchestrator/ policy layer (framework-free): the agentic tool-loop
              Orchestrator (registry-validated JSON tool protocol, TurnPlan
              schema with enforced citations, prompt-as-code hashing, a
              4-slot format-repair budget that a clean tool round refills by
              one, all inside a hard cap of tool_budget + 2 LLM calls per
              turn, then fallback), the deterministic FallbackPolicy,
              the per-turn Reflector (reflector.py), and manual.py, which
              renders <run_dir>/manual/{static,dynamic,skills} per run
evaluation/   framework-free evaluator library: pipeline / pool / dem_cache /
              sampling (anchor-p racing, 03 §8) / evidence (d_circ_ext,
              degraded_*, verified_circuit_distance) / writeback / feedback /
              baseline / worker
tools/        policy-agent tool suite (read/draft/policy scopes; docstrings
              render the manual's tool catalog)
memory/       campaign memory: archive + learning layer — two-tier Tanner
              fingerprints & known-code table, insight ledger with a
              deterministic PROSPECTIVE backtester (the only judge), UCB1
              strategy bandit, structural coverage (coverage.py)
sec/          LRC synthesis, ranking, audits, profile lowerings
construction/ family grammars + group catalog
deformation/  local-Clifford sector deformation (sector_h) + mode dispatch
verification/ seam (circuit.py), decoders, distance, noise, A4/A5 LER (ler.py)
llm/          completion-only provider adapters; --provider takes deepseek /
              openrouter / claude / openai (plus dry_run, the FallbackPolicy
              path); one adapter module each — deepseek.py, openrouter.py,
              anthropic_claude.py, openai_compat.py;
              retry, usage & reasoning stores
```

Search-phase sampling races candidates at the profile's anchor point
(uniform 0.005 / si1000 1e-3 / helios scale=1) against hall-of-fame and
baseline CI thresholds on the plpr scale; the shard stream is pre-registered
so adaptive depth never changes the RNG protocol. Search, racing and the
beats-BB decision all run at one campaign-fixed round count T_c (fairness
protocol); only `final_slow` re-measures — the top-K, plus the resident BB
baseline at the budget points that carry one (a BB fits exactly at 144 and
288 total qubits; other budgets run un-anchored) — at T=d rounds, and at the
same campaign-fixed p, since a campaign protocol accepts exactly one noise
point (`--p` is a single scalar; the former campaign-level multi-p sweep was
removed).

## CLI

```text
run-campaign         budget-point campaign through the workflow graph
run-agentic-cycle    one fast+slow cycle (no graph)
run-funsearch        fast loop only
slow-verify-seed / slow-verify-queue   direct slow verification
qualify-decoders     decoder qualification harness (03 §10)
export-stim-seed / export-stim-queue   Stim circuit + DEM export
build-dem-smoke, check-slow-deps, make-slow-aware-context,
analyze-runs, calibrate-slope-offset
```

Example dry run (no API key needed):

```bash
python -m omniqec_agentic.cli run-campaign --budget 144 --out runs/b144 \
  --turns 2 --racing --slow-workers 0
```

## Reproducibility invariants

- Seeds are index-derived everywhere; `shot_shards`/racing streams are part
  of the measurement protocol; the beam decoder is deterministic (bit-exact
  serial-vs-pool parity); ldpc's LSD reproduces statistically only (its
  cluster tie-breaking is address-nondeterministic).
- `sec_policy_id = "lrc_ranked_v1"` versions every LER: values are not
  comparable across SEC policy versions.
- `slope_offset` is currently `None` (uncalibrated under LRC, 03 D8):
  `d_circ_ler_estimate` stays unavailable until `calibrate-slope-offset`
  re-runs on final_slow survivors.
- Campaign runs snapshot the frozen protocol + resolved dependency versions;
  `--resume` refuses on protocol mismatch.

Learning is externalized into verified artifacts (03 §6): hypotheses enter
the ledger only via tools, become "supported" only through the prospective
backtester (post-proposal evidence, N>=20, bootstrap CI), and reach prompts
only from the manual's dynamic section. Hypothesis tags on construct programs
(`# exploits: I-3; tests: H-12`) drive bandit credit assignment.

Historical documents (`PROGRESS.md`, `E6_*`) are archived; the ver1-era
postsearch/sinter paths, the hook surrogate, and slow.py are removed
(03 §12, D11/D12).
