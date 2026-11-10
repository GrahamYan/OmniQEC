> **[ARCHIVED 2026-08-15]** 本文档描述重构前的旧系统状态，已被 `omniqec_agentic/docs/01_code_family_design.md` / `02_sec_design.md` / `03_workflow_design.md`（规范源）取代，仅作历史记录保留。

# omniqec_agentic Status

This file records the current design state. Detailed development transcripts and one-off experiment logs should stay under `runs/` or outside the package.

## HARD LESSONS — do not repeat (2026-06-14)

These cost real time during the budget sweep. Read before changing the slow stage.

### L1: fast_Q does NOT predict measured LER — NEVER promote top-N by Q for the sweep
`Q = k*d^2/n` rewards high rate (high k). The lowest-LER codes are LOW-k, high-d
(few logical qubits, each better protected). Measured across completed points, the
LER champion ranked by fast_Q was: B=80 #65/100, B=98 #87/118, B=120 #21/148,
**B=144 #102/204**. A `--promote-top N` (N=6 or even 24) slow stage ranked by Q
would MISS the champion entirely. **The sweep MUST use `--slow-all-archive`**
(measure every archive cell) to find champions. top-N-by-Q is only acceptable when
the objective IS Q, which it is not. (A top-N that ranked by a true LER predictor
would be fine, but we don't have one cheaper than measuring.)

### L2: the slow cost is decode-at-shots on LOW-LER codes; ISD is negligible
Per-candidate runtime at B=240 ([[120,*,*]], 100k shots): decode ~18000-45000
CPU-s, ISD ~300-1500 CPU-s (~3-5%). Good (low-LER) codes never hit
`max_errors` early-stop, so they run the FULL shot count — that is the entire cost.
Levers that work: fewer total codes measured (fewer generations; smaller archive),
NOT disabling ISD, NOT reducing shots (low-LER codes need the shots to see any
failures at all — at 1e-6 LER, 100k shots gives only ~3 failures). Precise champion
LER comes from the post-search confirmation protocol (high shots), not the search.

### L3: shared machine — co-tenant launcher killed us via broad `pkill python`
Two big points were SIGKILLed (exit 137) at the same second, twice. Root cause was
NOT resources (account has `memory.max=max`, 251G free; 80 workers ≈ 80 cores, no
thread leak). It was the user's own `_exp_pub.py` launcher running `pkill python`,
which killed all the account's python incl. our sweep (proof: its 13 procs started
2s after our death). Fixed user-side. Lesson: on shared boxes, confirm co-tenant
job launchers don't broad-kill; diagnose simultaneous exit-137 as external kill,
not OOM, when no memory limit exists.

### L4: slow job pool rebuilt the circuit/DEM PER SHARD — finer shards were SLOWER
The fine-grained pool's old job = one (decoder, basis, shard); each job rebuilt
`build_memory_experiment_pair` (stim circuit + DEM) from scratch. So per candidate
the circuit was rebuilt shards × 3 × 2 times. For big codes (n=120, DEM build tens
of seconds) raising shot-shards from 8 to 64 made things SLOWER, not faster — the
8× rebuild explosion swamped the better core-fill (observed: 240 gen1 stalled at
0/60 after 3h on 64 shards, vs 45/56 at 3h on 8 shards). Lesson: more shards only
helps if the per-shard setup is cheap.
Fix (2026-06-16, `run_shard_request`): job granularity is now one SHARD; it builds
the pair ONCE and decodes all decoders×bases on that shard. Builds per candidate
dropped from shards×6 to shards (6×), and the 8-shard path also got 6× fewer
builds. **Verified bit-identical**: each (decoder, basis, shard) keeps its exact
original seed (`base_seed + decoder*100 + basis`, then `_shard_seed`), so
failures/shots match the old per-job path exactly (micro + end-to-end check).
Determinism iron rule preserved.

### L5: the slow bottleneck is DECODE VOLUME, not the build — don't over-optimize build
After L4's per-shard build-once, raising shot_shards to 80 was STILL slow for big
codes (240 gen1 stalled). Root cause finally measured: a single n=120 code at 100k
shots takes ~5.8 min wall on 80 cores, of which the circuit/DEM **build is only
~30s (~10%)** — the rest is decode (100k shots × 3 decoders × 2 bases). So build
optimization (L4) gives ~10%, not the hoped-for speedup. A big point with
all-archive + 100k shots is ~20-30h regardless of shard/build tricks, because the
decode work is irreducible (low-LER codes need the full shot count, L2).
Real speed levers: more machines (points are independent — parallelize across
boxes), fewer shots (breaks low-LER measurement), top-N (misses champions, L1).
Architecture (SUPERSEDED by L6 — the active path was actually build-once-PER-SHARD
in an interleaved pool, not one-candidate-at-a-time + pickle; this paragraph
described an intended design that never became the active code, which caused real
confusion). The corrected, now-active design is in L6: build once per CANDIDATE +
pickle + per-(decoder,basis,shard) decode jobs, interleaved. shot_shards=8 (=
completed points, 口径 consistent); since decode dominates, shard count barely
moves total time.

### L6: build-once-PER-SHARD bundled 6 decoders into one 1h job → big codes timed out
(2026-06-16, supersedes the L4/L5 description of the active path.) The code that
actually ran the sweep was `run_shard_request` (mode="shard"): build the DEM once
PER SHARD, then decode ALL 6 (decoder×basis) on that shard inside ONE
timeout-bounded subprocess. For big codes (B=240, n≈120, d up to 14) that 6-decoder
bundle ran ~5400s > the 3600s job timeout → the shard was killed → **17/41 codes in
240 gen1 timed out and lost all LER data** (quality hole in the ranking) while their
dead jobs idled cores (the "load full but not really" tail).
Root cause was the BUNDLING, proven by elimination: 220 used the SAME 8 shards, SAME
3600s timeout, SAME big codes and had **0 timeouts in 13 gens** — its OLD job
granularity was one (decoder,basis,shard) ≈ 900s, well under the cap. Jobs are
single-threaded (`nlwp=1`, no OMP/MKL) on a 128-core box, so workers 40→80 does NOT
change a single job's wall time → **not the worker count**; and shard count was
identical → **not the shards**. Only the 6× bundling (the L4 "build once per shard"
optimization) crossed the fixed 1h cap.
Fix (`run_build_dem_request` + rewritten `run_shard_dem_request` + rewritten
`_evaluate_queue_with_jobs`): build the DEM ONCE PER CANDIDATE → pickle exact
float64 probs → decode jobs split per (decoder×basis×shard), each LOADS the pickle
and runs one explicit `unit` (global decoder/basis index passed in so the seed
`_shard_seed(base_seed + decoder*100 + basis, shard, shards)` is unchanged). Each
decode job ≈900s ≪ 3600s → **timeouts gone**; builds dropped to 1/candidate (vs 8
per-shard, vs 48 old). Interleaved pool kept (build→decode submitted dynamically as
each build lands), so cores stay full and codes still report `[slow]` one-by-one.
**Verified bit-identical 18/18** (new pickle/units path == old `run_shard_request`
baseline, 3 shards × 6 decoder/basis) on both Mac (ARM) and server (x86); end-to-end
orchestration smoke (ok+plpr+distance, invalid spec skipped, temp pickle cleaned).
workers=80, timeout=3600 unchanged.
shot-shards 8 → 32 (2026-06-17, protocol change, user-decided): real_test on the 4
codes that timed out in 240 gen1, full 100k shots, showed relaybp ≈ 50–57 min/shard
at 8 shards on n=120 (≈0.24 s/shot, runs the full slice because relaybp is the BEST
decoder there → no early-stop) — dangerously close to the 60 min job cap and a long
gen tail. 32 shards → relaybp ≈ 12.5 min/shard (4.8× cap margin) + more pieces to
fill the cores freed by the faster lsd/bposd. pickle-load overhead is negligible
(192 decode jobs/candidate × ~1 s load vs ~43000 s decode, <0.5%). shots stay 100k
(halving them would widen the LER CI ~√2 and break the BB144 CI-separation goal —
the low-LER codes we want need the full shots). 32 is statistically equivalent to the
completed points' 8 (finer partition of the same 100k shots), only exact values
differ; determinism holds at fixed (seed, shards=32).
#3 (ISD split) — NOT done, measured unnecessary: the distance job runs ISD ON by
default (`--dem-isd-distance` default-true; exact-circuit OFF) but ISD on n=120 is
only ~90 s (not the 22 min once feared), dwarfed by decode (~900 s) and hidden under
it — not a long pole. The per-decoder split also makes the timeout graceful: if
relaybp ever overruns, only its shard is lost, lsd/bposd still give the code an LER
(the candidate is no longer lost wholesale like the old bundle).

### L7: niche competition is Q_eff-driven → great-LER low-Q codes get evicted; LER hall-of-fame
(2026-06-18.) MAP-Elites niche competition (`evaluate_and_record`) keeps the higher
`_record_q_eff` (k·d_eff²/n) per cell — it does NOT look at measured LER. Parents
(`parent_records`) come from the same Q-ranked archive. So a low-Q/great-LER code
(observed: B=200 g2 [[96,2,12]] k=2 plpr 1.25e-6) gets evicted by a same-niche
higher-Q code at g3 and vanishes from both the live "best" and the parent pool — the
search abandons a good direction. This is L1 ("fast_Q ≠ LER") at the archive layer:
the search optimizes Q, the goal is LER.
Fix — **LER hall-of-fame** (`ArchiveMemory.hall_of_fame`, additive; MAP-Elites Q-churn
untouched as the exploration engine): top-N (16) NON-degraded records by measured plpr,
snapshotted each gen at the end of `apply_slow_evidence` (refs kept alive past niche
eviction), excludes baseline (blinding). Wired into: (1) `parent_records` reserves the
top-3 by plpr (reason="hall_of_fame") so good codes keep being mutated even after
eviction — the rest stay Q/family-diverse (explore); (2) the heartbeat champion is
`hall_of_fame[0]` → MONOTONIC (the per-gen archive min churns and is NOT the headline),
plus a k≥4 line as the fair-vs-BB144 lens, and ★NEW-BEST fires only on a true all-time
plpr improvement; (3) `hall_of_fame.json` artifact per gen. Also `_parent_record` now
exposes `slow_per_logical_per_round` so PARENTS render their LER in the prompt (was
Q_eff) — "LER is the only adjuster" (iron rule #3), verified no forbidden words. Global
record of ALL good codes stays on disk (slow_results) — the hall-of-fame is the small
exploit cache, not a global store. Verified end-to-end on a real 2-gen cycle.
Deploy-path gotcha: `archive.py` lives in `omniqec_agentic/memory/`; `scp`-ing it to
`omniqec_agentic/` lands a top-level orphan that is never imported (the real file stays
stale). Always scp subdir files with the full `.../memory/archive.py` path.

### Sweep operational notes
- Budget = TOTAL qubits (data + one ancilla per check row); BB72=144, BB144=288.
- Completed/correct config (all points): `--slow-all-archive --promote-top 6`,
  five families, 100k/50k/20k shots by budget tier, 8 shot-shards, 40 workers/lane,
  dual-lane (~92 load, safe). Big points (>=160) gens=60, mid 80, small 60.
- B=144 all-archive (204 codes) took ~14.5h; big points ~15-30h each.
- Killing the sweep: `pkill -9 -f` with a `[.]`-bracketed pattern to avoid the SSH
  session self-matching; verify after (interrupted nohup launches may have landed).

## v2 Draft Freeze (2026-06-11)

Five construction families (gb/lp/gb3/2bga/glp, plugin registry, per-campaign
selection), reconcile loop, fine-grained slow job pool, benchmark blinding,
calibrated slope offset (0.958), and full weight neutrality: check weight and
qubit degree appear NOWHERE as prompts, gates, penalties, or ordering terms —
the circuit-level LER measurement is the only mechanism that prices heavy
checks. Validated end-to-end with real DeepSeek (4 local campaigns, 400
proposals, 100% format-parse rate across all five families; best blind find
[[72,8,9]] gb3, raw Q=9.0, plpr=2.3e-5 @ 2k shots, no hook degradation).

## Current State

`omniqec_agentic` is a standalone agentic CSS-code search package anchored to local `omniqec_framework ver1`.

Implemented:

- ver1-style `construct(rng)` sandbox.
- `gb` and `lp` construction support.
- Structural CSS checks with `N_data <= N_budget` gating.
- ver1-compatible fast proxy score `Q = k*d^2/n`.
- MAP-Elites archive, duplicate tracking, failure signatures, parent selection, and prompt memory.
- DeepSeek and dry-run proposal providers.
- GPT and Claude provider adapters in the multi-model harness.
- ver1-aligned FunSearch loop with batched code-block proposals.
- Proposal refill when an LLM returns fewer parseable code blocks than requested.
- One-command `run-agentic-cycle` pipeline.
- Periodic slow promotion with `--promote-every`.
- Configurable fast promotion queue size with `--promote-top`.
- All-archive slow verification path with `--slow-all-archive`.
- Queue-level slow status distinguishes empty queues, no decoder-attempt candidates, decoder-attempt candidates with no ok result, and valid slow results.
- Summary artifacts record requested, parsed, missing, and refill-attempt proposal counts.
- Circuit-level DEM generation.
- BPOSD and RelayBP observable decoding.
- Slow LER rankings and slow-aware prompt context.
- Distance evidence artifacts with exact code-distance certification, optional exact circuit-distance search, and optional LER-slope estimates.
- Slow-calibrated hook surrogate artifact and prompt feedback. Calibration uses only exact circuit distance, LER-slope distance, or DEM-ISD positive-degradation labels. Hook-risk numbers reach prompt-facing parent records ONLY after calibration: the default weights are a hand-set prior penalizing high w+q, and surfacing them unevidenced would steer the model against high-weight checks, contradicting the hardware profile's "weight is a free parameter". w/q carry no penalty anywhere in scoring (Q_eff/LER); they appear only as recorded metrics and as last-resort archive tiebreakers.
- Optional DEM-ISD circuit-distance upper-bound evidence via `--dem-isd-distance`.
- Post-search sinter LER sweeps with LSD, BPOSD, and BeliefFind adapters.
- Post-search Stim circuit and undecomposed DEM export for seed and queue candidates.
- Hard-timeout subprocess isolation for slow verification candidates.
- `schema_version: 3` on slow verification artifacts.
- `analyze-runs` leaderboard with ver1-style `bestQ`, `beatBB`, verified, yield, and accepted metrics.
- Unified ranking module (`ranking.py`): `d_eff` (most-trustworthy distance, capped at the verifiable regime via `d_cap`), `Q_eff = k*d_eff^2/n`, Wilson confidence intervals, CI-aware lexicographic slow ranking, ver1-style `d_proxy/sqrt(n)` trust filter.
- Reconcile channel: slow candidate evidence is written back into fast archive records by `code_hash` (`ArchiveMemory.apply_slow_evidence`); degraded elites are quarantined from the parent pool and promotion queues but stay archived; prompt tables show `Q_eff` and `HOOK-DEGRADED` / `slow-verified` annotations.
- LSD (BP+LSD) observable decoder in the slow loop, first in the default decoder order (ver1's d_circ_ler protocol decoder).
- Resident BB [[72,12,6]] baseline injected into non-empty slow queues (hash-deduplicated, `is_baseline` flagged, never a parent); `analyze-runs` adds `beats_baseline_ler` (CI-separated).
- `slope_offset` is configurable in `VerificationConfig`; `calibrate-slope-offset` CLI re-fits it on the BB reference under the local protocol. Current default 0.958 was calibrated 2026-06-10 (LSD, p in {3e-3,4.5e-3,7e-3}, 50k shots/basis, fitted slope 3.958 on BB d_circ=6); ver1's 1.622 belongs to ver1's sinter protocol.
- DEM-ISD upper bound is wired into the reconcile loop: a bound below `d_proxy` proves degradation (`degraded_isd`), lowers `d_eff`, and quarantines the elite. Default-on in `run-agentic-cycle`; `dem_isd_max_mechanisms` default 4000 admits n<=72 memory DEMs (~25s/candidate at 120 iters; BB72 sanity check returns ub=6=d_circ, no false positive).
- Sandbox wall-clock timeout (`timeout_s`, default 10s, SIGALRM/main-thread): a runaway construct program no longer hangs the fast loop.
- Provider API calls retry transient failures with exponential backoff (`LLMConfig.max_retries`/`retry_backoff_s`); auth/config errors fail fast.
- RelayBP runs sequentially by protocol (`RelayBpDecoderConfig.parallel = False` default): its internal Rust threads consume RNG in scheduling-dependent order, so the parallel flag selects a *different* (individually reproducible) random stream. Sequential is the only choice that keeps results identical across worker counts and machines; the job pool supplies the parallelism. Worker subprocesses also cap BLAS/OMP threads to 1.
- Fine-grained slow job pool: with `--slow-workers N` (0 = auto cpu_count-1) the slow stage flattens all candidates into (candidate, p, decoder, basis, shot-shard) decode jobs plus per-candidate distance jobs, each an isolated hard-timeout subprocess consumed by one global work-stealing pool. `--slow-shot-shards K` splits shots for full multi-core saturation; K is a protocol parameter (K=1 reproduces the serial path bit-for-bit; results are worker-count-independent for fixed K). RelayBP's internal Rust threads are disabled when the pool is wide to avoid oversubscription. Per-job timeouts mean one stuck decoder no longer discards the candidate's other results.
- Multi-RNG-seed proposal evaluation: `--rng-seeds-per-program K` (default 8) runs each proposed program under K seeds and lets every distinct code compete for its own niche (ver1 PROXY_SEEDS breadth); duplicate codes short-circuit by hash before the proxy runs, so deterministic programs cost one evaluation.
- Benchmark blinding: the proposal prompt no longer names a baseline to beat (ver1's "BASELINE to beat: BB [[72,12,6]]" line is removed), and injected `is_baseline` rows are filtered from every prompt-facing slow-feedback surface (best_results, guidance, parents, distance-evidence list). The BB comparison is measurement-side only (rankings.json, analyze `beats_baseline_ler`). An organic archive elite that happens to be the BB code remains visible as ordinary search state.

- Family plugin registry (`construction/registry.py`): a family = one registered plugin bundling validate/build/prompt-contract/size-formula; sandbox, builder, and prompt assembler dispatch through it (gb/lp behavior verified line-identical post-refactor). `--families` selects per-campaign subsets; off-family proposals are gated out before the proxy (`family_not_allowed`). `--llm-temperature` exposes the sampling temperature (vary across replicas for exploration diversity).
- `gb3` (three-variable bicycle on the Z_ell x Z_m x Z_r torus): r=1 reduces bit-for-bit to gb (BB [[72,12,6]] reproduced), CSS orthogonality structural + asserted, full fast/slow/reconcile pipeline verified.

- `2bga` (non-abelian two-block group-algebra, ported from ver1 `gb_nonabelian` with review): permutation-group catalog D3..D36 / A4 / S4 / A5 (A5 is an addition over ver1; ver1's unused-and-dubious `quaternion_like` helper was deliberately NOT ported), canonical sorted element indexing, spec = group name + element index lists. Verified: ver1 A4 anchor [[24,5]] reproduced exactly; CSS orthogonality across all group types; group orders and dihedral relations self-checked; full fast/slow/reconcile pipeline.

- `glp` (general lifted product over F_2[G], ported from ver1 `gb_lifted` WITH two corrections found in review and confirmed computationally: (1) ver1 Kronecker-multiplied lifted matrices, scrambling the column ordering between HX and HZ and breaking CSS orthogonality outside its two degenerate self-test cases — e.g. 32 commutation violations on a 2x2 base over D4; fixed by taking Kronecker products at the protograph level before lifting. (2) ver1 lifted both blocks with the left regular representation, which needs commuting entries; fixed by lifting the A-side with L and the B-side with R, provably orthogonal for any group/shapes. In ver1 the bug failed safe — candidates were silently rejected — so the family's claimed search space was empty outside the degenerate slices.) Anchors verified: C1+rep3 -> HGP toric [[18,2]]; 1x1 over A4 bit-identical to 2bga ([[24,5]]); glp over C4 reproduces the lp family's [[72,8,8]] invariants; orthogonality across random shapes/groups including ver1's failure cases.

Deferred:

- Hardware-specific atom/ion connectivity branches beyond the current shared all-to-all/ASC-coloring path.
- Open CSS construction.
- Superconducting targets.
- Full ver1 certification parity beyond the current bounded code MILP, optional Stim circuit search, and DEM-ISD upper-bound path, including BZ/MILP variants.
- Sinter RelayBP unless the local `relay_bp` package exposes `SinterRelayBP`; current QEC environment does not.
- ver1 post-search plotting/batch plotting tools beyond the current `analyze-runs`, sinter sweep, and Stim export commands.

## Design Decisions

### D001: ver1 Is The Baseline

The new package may reimplement or port behavior, but normal runtime code does not import historical versions. `check-ver1-baseline` is the explicit developer comparison path.

### D002: Candidate Identity Is Source Plus RNG

The genotype is the generated `construct(rng)` program and controlled seed. Family labels are construction metadata, not fixed search boundaries.

### D003: Structural Checks Are Hard Gates

Structural checks validate CSS legality, connectivity, positive `k`, and `N_data <= N_budget`. They do not replace ver1-compatible fast score `Q`.

### D004: Initial Families Are `gb` And `lp`

`gb_multivariate` remains planned until the baseline fast/slow pipeline is stable under the same verifier.

### D005: Atom And Ion Are Explicit Targets

Run config records `atom` or `ion`. Current circuit scheduling has not yet
diverged between these hardware targets, and the verification protocol is
unified across platforms (no per-platform scoring, by design). The prompt
carries a one-line hardware introduction (`config.HARDWARE_PROMPT_PROFILES`):
the platform connectivity statement only. Check weight is deliberately never
mentioned (project decision, 2026-06-11): the platforms impose no weight
limit, the model chooses weights freely, and the circuit-level LER
measurement naturally prices heavy checks (deeper extraction, more fault
locations, hook propagation) — the feedback loop is the regulator, not prompt
text. Local validation supports this: with no weight statement enforced, the
model's proposals self-distributed over w=6-8 (max 12 at ~1%), and the
measured LER decided the winners. The connectivity claims are identical
across the two targets so prompt flavor can never confound atom-vs-ion
comparisons.

### D006: Slow Promotion Has Two Evidence Lanes

Slow promotion records:

- LER performance ranking under LSD/BPOSD/RelayBP (LSD first by default).
- Exact CSS code-distance evidence from a bounded package-local MILP backend.
- Optional exact circuit-distance evidence from Stim search when explicitly enabled.
- DEM-ISD circuit-distance upper bound (default-on in `run-agentic-cycle`):
  a bound below `d_proxy` is a constructive degradation proof, sets
  `degraded_isd`, lowers `d_eff`, and quarantines the elite. This is the
  automatic degradation signal in single-p runs.
- Optional `d_circ_ler_estimate/degraded_estimate` evidence using ver1-style LER-slope semantics when enough p-points exist.
- Hook-risk surrogate predictions calibrated only when exact circuit or LER-slope distance labels exist.

The two are not collapsed into one score.
With the current default single `p=0.005` protocol, LER is still circuit-level
sampling, while LER-slope distance estimates are normally unavailable because a
single point has no slope.
Each slow candidate runs behind a subprocess hard-timeout boundary so a stuck
decoder/circuit call does not block the full campaign.
The analyzer keeps `verified_non_degraded` separate from `circuit_ler_ok` because
single-p LER is not an exact/certified distance result.
The current exact code-distance backend certifies the static CSS code distance,
not circuit hook degradation. Exact circuit distance is available behind
`--exact-circuit-distance` and can be slow, so default runs keep it off.

### D007: Reconcile Without Score Collapse

Slow evidence corrects the fast tier through hard rules instead of ver1's soft
FOM multipliers:

- `d_eff = min(d_proxy, verified circuit distance)`, capped at `d_cap`
  (default 8, ver1's D_CERT_MAX idea); niche competition, parent selection,
  and queue ordering use `Q_eff = k*d_eff^2/n`.
- Slow-verified candidates rank above unverified ones by measured
  `per_logical_per_round`; overlapping Wilson intervals are statistical ties
  that fall back to `Q_eff`, then calibrated hook risk.
- Degraded candidates are quarantined from parents/promotion but stay archived
  as audit trail and hook-surrogate labels.
- ver1's trust/depth/wq multipliers are deliberately not adopted (hand-tuned
  constants tied to ver1's protocol); the hook surrogate stays prompt guidance
  and a tiebreaker, never a score multiplier.

## Differences From ver1

Kept:

- `construct(rng)` candidate contract.
- `gb` and `lp` construction semantics.
- Fast proxy score `Q = k*d^2/n`.
- Archive niche idea.
- Slow promotion of fast elites to circuit-level verification.

Different:

- Standalone package boundary.
- Explicit `.env` handling for DeepSeek credentials.
- `N_data <= N_budget` is a hard archive eligibility rule.
- Structural checks are explicit hard gates.
- Batch proposal support is available.
- Provider registry supports DeepSeek, GPT, Claude, and dry-run; GPT/Claude remote calls are not validated without API keys.
- Batch proposal refill is an agentic-runner robustness layer; ver1 expected the generation call itself to produce usable candidates.
- Slow promotion writes explicit artifacts rather than mutating fast archive rows.
- The fast loop writes both a top-N `promotion_queue` and an `all_archive_queue`; ver1 only had the implicit fast-to-slow promotion flow.
- Slow verification uses a package-local subprocess worker for timeout isolation instead of ver1 `qec_core/na_verify.py`.
- Analyzer output adds `ler_ok` beside ver1-style `verif` so current single-p circuit-level LER does not masquerade as distance verification.
- Exact code distance is implemented with a bounded package-local SciPy MILP backend instead of ver1's full certification stack.
- Exact circuit distance is available through optional Stim search, but is disabled by default and protected by the slow subprocess timeout.
- Slow hook surrogate is stored as `hook_surrogate.json` and used as prompt guidance only after trustworthy exact/slope labels or DEM-ISD positive-degradation labels exist; single-p LER is deliberately not used as a degradation label.
- Sinter LER sweeps are post-search commands, not part of the default agentic loop. They use the same circuit builder and undecomposed DEM semantics as slow verification.
- Sinter RelayBP is environment-dependent and currently unavailable through `relay_bp.SinterRelayBP`; direct RelayBP DEM sampling remains available in slow verification.
- Full ver1 certification parity still excludes BZ/MILP variants, plotting, and the larger batch figure tool suite.

## Main Verification Commands

```bash
/opt/homebrew/anaconda3/envs/QEC/bin/python -m py_compile $(find omniqec_agentic -name '*.py' | sort)
/opt/homebrew/anaconda3/envs/QEC/bin/python -m omniqec_agentic.cli check-ver1-baseline --n-budget 72 --proxy-iters 120
```

Recommended full-loop smoke command:

```bash
/opt/homebrew/anaconda3/envs/QEC/bin/python -m omniqec_agentic.cli run-agentic-cycle \
  --provider deepseek \
  --run-id n72_atom_main \
  --n-budget 72 \
  --hardware atom \
  --out runs/omniqec_agentic/n72_atom_main \
  --generations 3 \
  --promote-every 1 \
  --proposal-count 4 \
  --promote-top 6 \
  --slow-all-archive \
  --proxy-iters 120 \
  --slow-p-values 0.005 \
  --slow-shots 1000 \
  --slow-batch-size 200 \
  --slow-max-errors 1000 \
  --slow-timeout-s 600
```

## Next Work

- Improve proposal quality guidance for connected Tanner graphs and avoiding `d_proxy=2`.
- Decide whether `gb_multivariate` is stable enough to enter the same construction and verification path.
- Add real atom/ion connectivity constraints once the physical rules are specified.
