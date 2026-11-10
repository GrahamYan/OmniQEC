# OmniQEC final protocols (2026-11-06)

This is the protocol of the released code for the two search experiments of the paper -- Fig.2 (quantum memory)
and Fig.3 (logical CNOT). Everything below is enforced by the code: the launcher presets set the keys, the protocol
snapshot of every run freezes them, and the validator checks a finished run against them.

**Revision 2026-11-10 (paper alignment of the measurement and the final stage).** The search-stage early stop and
the final stage are now exactly the paper's (Appendix D3, Table VII), under new ids so that no run of the earlier
rules can be mistaken for one of these: early stop `paper_d3_wilson10x_ref_maxerr30_v1` (was `pilot10x_maxerr30_v3`)
and final stage `paper_top5_cumulative_v1` (snapshot key `final_stage`; was 2 structural + 4 runtime finalists at
25 000 shots followed by the separate Top-5 extension tool). The final stage now runs inside the campaign: one
`run-campaign` produces the paper's final ranking and champion (`final_slow/final_ranking.json`). Paper passage by
paper passage, with every reading of an ambiguous sentence: `docs/paper_alignment_measurement.md`.

**Revision 2026-11-10 (paper revision, `docs/paper_alignment_measurement.md` §8).** The reported LER of the BB reference
codes is based on 50 000 (BB72) and 100 000 (BB144) shots per basis in both tasks (the initial confirmation plus a
top-up to that total; `config.REFERENCE_CODE_SHOTS_PER_BASIS`), no longer on the Table VII limit of their budgets. A
run under an exact check-weight contract (Fig. 4 c) takes the final stage `paper_fig4c_top6_confirm_v1`: the six
leading distinct codes by search-stage pLER, 25 000 shots per basis each, no top-up, the three best reported.

## 1. Common settings (search, final stage, extension)

| Item | Setting |
|---|---|
| Noise | circuit-level `uniform_paper_v1`, p = 0.005, no idle noise |
| Syndrome-extraction circuit | `twobasis_nonint_v1` (both bases every round, X half then Z half, per-basis layer-order search certified by an ISD probe) |
| Decoder | `telescoping_cpu` (vendored telescoping decoder, CPU stages S3-A/B/C; an undecoded shot counts as a logical error) |
| Bases | X and Z experiments, equal shots per basis |
| Rounds | T = 12 for every code (`fixed_Tc`), in the search and in the final stage |
| Metric | pLPR = 1 - (1 - P)^(1 / (b k T)), P = 1 - (1 - q_X)(1 - q_Z); b = 1 (memory), b = 2 (CNOT, two code blocks); 95 % Wilson interval per basis mapped through the same formula |
| Code families | bb, hgp, lp_abelian, lp_nonabelian; budget N counts data qubits plus one ancilla per check |
| Seed programs | five fixed hand-written construction programs (`examples/seed_programs.py`), evaluated in the first turn only (protocol key `seed_programs = first_turn`; paper appendix D1), under the same filters as LLM programs and outside the LLM program budget |
| LLM search | 30 turns, 4 generations per turn, 24 programs per generation, 40 seeds per program in the fast loop, strict turns (a turn without a complete proposal batch is retried from the checkpoint) |
| Check weight | fast-loop cap w <= 8 (`verification.max_check_weight = 8`: a code with a heavier X or Z check is rejected by the construction-time checks before distance estimation; paper Methods and appendix B2/D1), plus the slow-loop gate w <= 8 as a redundant guard |
| Early stop (slow loop) | `paper_d3_wilson10x_ref_maxerr30_v1` (Appendix D3): 2 000 shots per basis first, then 250-shot batches; after the 2 000 and after every batch the candidate stops when its Wilson lower pLER bound exceeds (strictly) ten times the upper bound of the best measured code (the hall-of-fame row with the lowest estimate) AND ten times the upper bound of the run's applicable BB reference code (BB72 at N = 144, BB144 at N >= 288; at the other budgets the best measured code only), or once both bases hold 30 failures, or at 25 000 shots per basis; from turn 0. The reference code itself is measured without the 10x comparison |
| Shots | search: at most 25 000 per basis per candidate; final stage: see "Final stage" below |
| Final stage (both figures) | `paper_top5_cumulative_v1` (Appendix D3, Table VII): the five leading canonically distinct codes by search-stage pLER (ties: smaller upper bound) get an initial confirmation of 25 000 fresh shots per basis (seed namespace `final_confirmation_v1`, 128 shards), then further fresh shots up to the cumulative per-basis limit of Table VII for the run's task and N (namespace `final_topup_v1/p<i>`, parts of at most 900 000 / 150 000 shots, ~300 shots per shard, at least 128 shards); shots and failures are added per basis and the pooled pLER ranks the finalists, rank 1 is the champion. The applicable BB reference code gets the same initial confirmation and a top-up to its paper total (BB72: 50 000, BB144: 100 000 shots per basis, both tasks), namespace `final_topup_v1/reference/p<i>`; a finalist that is the reference code itself is measured once, the reference row pooling the prefix that reaches the reference total. Per-candidate resume in every part |
| Models | Claude `claude-opus-5`, GPT `gpt-5.6-sol` (both through CLIProxyAPI), DeepSeek `deepseek/deepseek-v4-pro` (OpenRouter); temperature 1.0 / 1.0 / 0.45; output cap 64 000 tokens; default reasoning setting |
| Reasoning-saturation rule | see below |
| Iteration review | the reflector runs at the end of every turn; a failed call is retried (3 attempts, 60 s / 300 s backoff, 30 min wall budget; protocol key `reflection_policy`) and a review that still fails is recorded in `reflection.json`, the turn audit and the campaign audit summary |

**Reasoning-saturation rule.** Every arm starts a figure at the model's default reasoning setting. If turn 0 of a
budget point cannot be completed because the reasoning fills the output -- a generation whose three completion
attempts all return an empty answer after the full 64 000 output tokens -- the arm restarts every budget point of that
figure from scratch with the reasoning level `medium` (`claude-opus-5(medium)`). The rule looks only at whether the
model produced an answer, never at the codes or their measurements. The runs it replaces are kept
(`aborted_reasoning/`), the switch is recorded in the arm's flag file and in `reasoning_fallback.json` of every
restarted run, and the validator refuses a default-level run of an arm that has switched. The fallback is defined for
the Claude arm; a saturated start of another arm stops the point for a human decision.

## 2. Fig.2 -- memory (`PRESET=fig2_final`)

| Item | Setting |
|---|---|
| Structural profile | `telescoping_structural_v3_8` = v3.7 + two machine rules |
| Variant generations | the last 2 of the 4 generations of a turn only make parameter variants of the hall of fame and the budget point's reference code (`exploit_generations = 2`) |
| Endpoint family rules | the two structural endpoints come from different families when the front offers two; a family whose last 3 endpoint measurements were all >= 10x the best measured code loses its endpoint slots |
| Slow queue per turn | 2 structural (max-k and max-d_search endpoints of the (k, d_search, -w) front) + 4 runtime |
| Final stage | the common final stage above; the resident BB reference code (BB72 at N = 144, BB144 at N >= 288) is measured to its paper total (50 000 / 100 000 shots per basis) |

## 3. Fig.3 -- logical CNOT (`PRESET=fig3_final`)

| Item | Setting |
|---|---|
| Task | `cnot_dense`: two blocks of the code with a transversal logical CNOT after every QEC round but the last, decoded jointly |
| Structural profile | `telescoping_structural_v3_7_cnot_v2` = v3.7 + a third structural slot for the front's lightest cell + the not-competitive prompt block (ratio 2, at most 12 rows) + measured/twin feedback to the generator |
| Slow queue per turn | 3 structural (max-k, max-d_search, lightest cell) + 4 runtime |
| Final stage | the common final stage above (Table VII CNOT column) |

Differences between the two searches that are not part of the task itself: the slow-queue composition (2 + 4 versus
3 + 4 with the lightest-cell slot), the generator feedback (variant generations and endpoint family rules versus the
not-competitive block and twin feedback), and the Claude reasoning level wherever the reasoning-saturation rule
switches one figure and not the other.

## 4. Top-5 extension (LEGACY, `hpc/top5_extension.py`)

Superseded on 2026-11-10 by the campaign's own final stage (`paper_top5_cumulative_v1`, §1). The tool refuses a
run that carries that final stage; it remains only for runs whose final stage was the earlier 2 structural +
4 runtime shortlist. Its rule is not the paper's (Top-5 among those six by final pLPR, an adaptive target); the
caps file is Table VII and is pinned to `config.FINAL_CUMULATIVE_SHOT_CAPS` by the tests. The rule as it was:

1. Top-5: the five best measurable non-reference finalists of a run by final pLPR (ties: smaller upper interval end).
2. A Top-5 code with fewer than 30 failures in a basis is extended once. With f = min(F_X, F_Z) of its final stage
   (25 000 shots per basis), the target total per basis is the cap if f = 0, otherwise
   min(cap, 50 000 * ceil(45 * 25 000 / f / 50 000)); the shots already measured count toward it.
3. Caps (per basis, final stage included; `hpc/top5_extension_caps.json`):

   | N | 60 | 98 | 144 | 180 | 220 | 260 | 288 | 320 | 360 |
   |---|---|---|---|---|---|---|---|---|---|
   | Fig.2 (memory) | 175 000 | 175 000 | 175 000 | 175 000 | 275 000 | 525 000 | 525 000 | 875 000 | 1 725 000 |
   | Fig.3 (CNOT) | 75 000 | 75 000 | 75 000 | 75 000 | 125 000 | 175 000 | 325 000 | 525 000 | 875 000 |

4. Each part of an extension has its own seed namespace (`top5_extension_v1/<key>`) and is measured by
   `hpc/fixed_code_measure.py` under the run's own protocol snapshot (T = 12); failures and shots are added per basis.
5. The champion of a run is the Top-5 code with the lowest pooled pLPR (ties: smaller upper interval end). There is no
   second round; a code still below 30 failures at its cap is reported as measured.

## 5. How to run

```bash
# one budget point of one arm (ARM=claude|gpt|deepseek)
PRESET=fig2_final ARM=claude POINTS=180 qsub -v PRESET,ARM,POINTS omniqec_agentic/hpc/nscc_claude_grid_v36.pbs
PRESET=fig3_final ARM=claude POINTS=180 qsub -v PRESET,ARM,POINTS omniqec_agentic/hpc/nscc_claude_grid_v36.pbs

# validation of a finished run (the launcher runs it itself with VALIDATE_FORMAL=1)
python omniqec_agentic/hpc/validate_fig2_campaign.py --profile fig2_final --run-dir RUN --turns 30 --p 0.005 \
    --provider PROVIDER --model MODEL --decoders telescoping_cpu --reasoning-fallback-flag FLAG
python omniqec_agentic/hpc/validate_fig2_campaign.py --profile fig3_final --run-dir RUN --turns 30 --p 0.005 \
    --provider PROVIDER --model MODEL --decoders telescoping_cpu --reasoning-fallback-flag FLAG

# the final ranking and champion of a finished run (written by the campaign itself)
cat RUN/final_slow/final_ranking.csv
```

`FLAG` is `RUNS/reasoning_fallback_<arm prefix>_<RUN_TAG>.json` (it need not exist); `MODEL` is the run's
`required_model`.

## 6. Earlier versions

Every earlier experiment version is tagged in the repository (`git tag -l 'exp/*' -n1`); the map from data set to code
version is `runs_export/paper_data_20261028/code/CODE_INDEX.md`. The validator profiles `v3_7` and `v3_8` check the
as-run Fig.2 data (T = max(12, d_search)).
