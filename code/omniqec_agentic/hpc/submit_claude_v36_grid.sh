#!/usr/bin/env bash
# Submit one independent afterany chain per Claude v3.6 budget point
# (telescoping_structural_v3_6: ONE decoder telescoping_cpu -- the Mitten
# paper's open-source telescoping decoder (vendored at
# third_party/telescoping-decoder) in its CPU configuration: S1/S2 GPU stages
# off, S3-A/B/C on the GARI system of the two-basis DEM, S4 Gurobi off, an
# undecoded shot (NC) counts as a logical error, no BPOSD -- v3.4 geometry,
# paper SI C noise uniform_paper_v1 at p=0.005 with no idle noise, SEC policy
# twobasis_nonint_v1 with multi-seed certification (sec_cert_seeds=3, refine
# top-4 x 2000; code default, asserted by the launcher's V37_DEFAULTS_OK
# self-check together with the validator's FORMAL_SEC_POLICY -- there is no
# flag to pass here; strict runs refuse lrc_ranked_v1), and the BUDGET POLICY
# (config.budget_policy): points <= 288 run 20k/200k shots per basis, points
# >= 300 run 5k/50k -- shots only, no per-stage time limits -- applied by
# run-campaign from --budget, so neither this submitter nor the launcher
# passes shots. There is NO FINAL_SHOTS_OVERRIDE any more. The launcher
# prebuilds the telescoping S3 kernels once per job (XDG_CACHE_HOME under
# runs/.cache on the shared filesystem) before any worker starts.
#
# Every submission needs PROJ=<PROJECT>, the PBS project the links are charged to
# (no default; the examples below leave it out for brevity).
#
# Formal grid (13 points, DESCENDING = big codes first; every point is an
# independent afterany chain, so the order only sets submission/queue order):
#   PROJ=<PROJECT> bash ~/OmniQEC/omniqec_agentic/hpc/submit_claude_v36_grid.sh
# Recommended staging:
# wave 1 = the two largest points with long chains, gated on their first
# audited turn (peak_rss, s/turn, no FATAL_STOP); wave 2 = the rest. The
# duplicate check refuses a point that already has an active chain under the
# same RUN_TAG (and exits before submitting anything), so a later wave must
# name only the points not yet submitted:
#   POINTS="360 288" CHAIN_LINKS=8 bash .../submit_claude_v36_grid.sh
#   POINTS="260 240 220 200" CHAIN_LINKS=6 bash .../submit_claude_v36_grid.sh
#   POINTS="180 160 140 120 98 80 60" CHAIN_LINKS=4 bash .../submit_claude_v36_grid.sh
# DeepSeek arm on NSCC through OpenRouter (2026-09-10; ~/.config/omniqec/
# openrouter.env holds the key; jobs omni_deepseek_v36, dirs fig2_deepseek_*):
#   ARM=deepseek POINTS="144 98 60" CHAIN_LINKS=4 bash .../submit_claude_v36_grid.sh
# NCPUS=64 (2026-09-04): submit the chain on 64-core q2 nodes (-l
# select=1:ncpus=64:mem=100gb, routed by `normal`) -- half the SU rate for a
# point whose wall clock is dominated by LLM calls and the 4-process build
# wave, i.e. every point that has not started yet. A point that already
# froze slow_workers=127 must stay at NCPUS=128 (the launcher refuses to
# resume it otherwise; this script checks first):
#   POINTS="260 240 220 200 180 160 140 120 98 80 60" NCPUS=64 CHAIN_LINKS=6 bash .../submit_claude_v36_grid.sh
# Every link is one 24 h x NCPUS-core job (3072 SU at 128 cores, 1536 at 64
# if fully used); a link that
# finds campaign_report.json exits at once, so spare links cost nothing. A
# chain that ends before 60 turns is extended WITHOUT a gap while it is still
# alive: EXTEND=1 appends CHAIN_LINKS links after the highest-numbered live
# link of every named point (a point with no live link gets a fresh chain),
# so the next link is already queued when the last one hits its walltime:
#   EXTEND=1 NCPUS=64 CHAIN_LINKS=4 POINTS="200 180 160 140 120 80" bash .../submit_claude_v36_grid.sh
# Without EXTEND=1 a point with a live chain is refused (the old behaviour:
# re-run the same POINTS after its last link has left the queue).
#
# Two-turn b144 canary (the policy's real 20k/200k shots and the 128-shard
# final stage; freezes decoder=telescoping_cpu / telescoping / budget_policy
# in the snapshot; re-measures s/shot for telescoping_cpu on NSCC cores --
# locally BB144 ~0.07 and Mitten200 ~11-18 CPU-s/shot). CHAIN_LINKS=2: the second
# afterany link is free when the first finishes (campaign_report.json =>
# exits at once) and resumes from the checkpoint if the first link is killed
# by walltime. The canary's run directory must NOT be rsynced over by a
# later deployment (worker subprocesses import code at spawn):
#   POINTS=144 TURNS=2 RUN_TAG=v37_canary \
#     VALIDATE_FORMAL=0 CHAIN_LINKS=2 bash \
#     ~/OmniQEC/omniqec_agentic/hpc/submit_claude_v36_grid.sh

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/OmniQEC}"
PBS_SCRIPT="${PBS_SCRIPT:-$REPO_ROOT/omniqec_agentic/hpc/nscc_claude_grid_v36.pbs}"
# Each point gets an independent chain; descending order only controls the
# submission/display order, not execution dependencies between points.
POINTS="${POINTS:-360 288 260 240 220 200 180 160 140 120 98 80 60}"
CHAIN_LINKS="${CHAIN_LINKS:-4}"
# 2026-11-06: PRESET=fig2_final | fig3_final submits the final protocols (docs/final_protocol_20261106.md);
# the launcher sets and checks the protocol keys, its run tag defaults to f2final / f3final.
# 2026-11-10 (paper alignment): ABLATION=no_slow | no_feedback (Table VIII) or EXACT_CHECK_WEIGHT=w (Fig. 4 c,
# PRESET=fig3_final) run on a preset; both are forwarded to the launcher, need VALIDATE_FORMAL=0, and the run tag
# defaults to the preset's tag plus _no_slow / _no_feedback / _w<w> (the launcher's rule). Example, the w = 5 search:
#   PRESET=fig3_final EXACT_CHECK_WEIGHT=5 VALIDATE_FORMAL=0 POINTS="98 144 180" bash submit_claude_v36_grid.sh
PRESET="${PRESET:-}"
PRESET_VARIANT=""
if [[ -n "${ABLATION:-}" && "${ABLATION:-}" != "full" ]]; then PRESET_VARIANT="_$ABLATION"; fi
if [[ -n "${EXACT_CHECK_WEIGHT:-}" ]]; then PRESET_VARIANT="${PRESET_VARIANT}_w$EXACT_CHECK_WEIGHT"; fi
case "$PRESET" in
  "") RUN_TAG="${RUN_TAG:-v37}" ;;
  fig2_final) RUN_TAG="${RUN_TAG:-f2final$PRESET_VARIANT}" ;;
  fig3_final) RUN_TAG="${RUN_TAG:-f3final$PRESET_VARIANT}" ;;
  *) echo "PRESET must be fig2_final or fig3_final (got $PRESET)" >&2; exit 2 ;;
esac
TURNS="${TURNS:-30}"
VALIDATE_FORMAL="${VALIDATE_FORMAL:-1}"
NCPUS="${NCPUS:-128}"
EXTEND="${EXTEND:-0}"
USER_OUT_ROOT="${OUT_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/runs}"
# ARM=claude (default): the Claude subscription arm. ARM=deepseek (2026-09-10):
# the DeepSeek arm through OpenRouter -- same launcher, same per-point
# independent chains (pay-per-token has no one-campaign-at-a-time limit), run
# dirs runs/fig2_deepseek_b<N>_<tag>, jobs named omni_deepseek_v36, read with
# deepseek_v36_status.sh. The ChatGPT arm keeps its serial submitter.
ARM="${ARM:-claude}"
case "$ARM" in
  claude)   ARM_PREFIX="claudesub"; ARM_LABEL="Claude" ;;
  deepseek) ARM_PREFIX="deepseek";  ARM_LABEL="DeepSeek" ;;
  *) echo "ARM must be claude or deepseek (the ChatGPT arm goes through submit_gpt_v36_serial.sh; got ARM=$ARM)" >&2; exit 2 ;;
esac
JOB_NAME="omni_${ARM}_v36"

# SU account. The launcher's #PBS -P is a placeholder; -P "$PROJ" on the qsub
# line overrides it. PROJ=<PROJECT> is required (no account-derived default) --
# the same convention as submit_gpt_v36_serial.sh.
PROJ="${PROJ:-}"
[[ -n "$PROJ" ]] || { echo "set PROJ=<PROJECT> (the PBS project to charge)" >&2; exit 2; }
[[ "$PROJ" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid PROJ=$PROJ" >&2; exit 2; }

[[ -f "$PBS_SCRIPT" ]] || { echo "missing PBS launcher: $PBS_SCRIPT" >&2; exit 2; }
command -v qsub >/dev/null || { echo "qsub is unavailable; run this on an NSCC login node" >&2; exit 2; }
command -v qstat >/dev/null || { echo "qstat is unavailable; run this on an NSCC login node" >&2; exit 2; }
[[ "$CHAIN_LINKS" =~ ^[1-9][0-9]*$ ]] || { echo "CHAIN_LINKS must be a positive integer" >&2; exit 2; }
[[ "$TURNS" =~ ^[1-9][0-9]*$ ]] || { echo "TURNS must be a positive integer" >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid RUN_TAG=$RUN_TAG" >&2; exit 2; }
[[ "$NCPUS" == "128" || "$NCPUS" == "64" ]] || { echo "NCPUS must be 128 (q4 whole node) or 64 (q2 node)" >&2; exit 2; }
[[ "$VALIDATE_FORMAL" == "0" || "$VALIDATE_FORMAL" == "1" ]] || { echo "VALIDATE_FORMAL must be 0 or 1" >&2; exit 2; }
[[ "$EXTEND" == "0" || "$EXTEND" == "1" ]] || { echo "EXTEND must be 0 or 1" >&2; exit 2; }
if [[ -n "${FINAL_SHOTS_OVERRIDE:-}" ]]; then
  echo "FINAL_SHOTS_OVERRIDE is not a v3.6 knob: shots per basis come from config.budget_policy(--budget)" >&2
  exit 2
fi
if (( VALIDATE_FORMAL )) && [[ -n "$PRESET" ]] && { [[ "$RUN_TAG" == "v37" ]] || (( TURNS != 30 )); }; then
  echo "PRESET=$PRESET with VALIDATE_FORMAL=1 requires TURNS=30 and a RUN_TAG other than v37" >&2
  exit 2
fi
if (( VALIDATE_FORMAL )) && [[ -z "$PRESET" ]] && { [[ "$RUN_TAG" != "v37" ]] || (( TURNS != 30 )); }; then
  echo "formal submission requires RUN_TAG=v37 and TURNS=30" >&2
  echo "for a canary set VALIDATE_FORMAL=0 explicitly" >&2
  exit 2
fi

# Plain strings, not `declare -A`: the login nodes have bash 4 but the script
# is also exercised by the test-suite on macOS bash 3.2.
seen_points=" "
n_points=0
for point in $POINTS; do
  [[ "$point" =~ ^[1-9][0-9]*$ ]] || { echo "invalid point: $point" >&2; exit 2; }
  [[ "$seen_points" != *" $point "* ]] || { echo "duplicate point in POINTS: $point" >&2; exit 2; }
  seen_points+="$point "
  n_points=$(( n_points + 1 ))
done
(( n_points > 0 )) || { echo "POINTS is empty" >&2; exit 2; }

# Normalize qstat -f's folded records into one key/value per line.
job_info() {
  qstat -f "$1" 2>/dev/null | sed 's/^[[:space:]]*//' \
    | awk 'NR>1 && $0 !~ / = / {printf "%s", $0; next} {if (NR>1) printf "\n"; printf "%s", $0} END {print ""}'
}

# Every live (queued, held or running) link under this RUN_TAG that carries
# one of our points. Without EXTEND=1 any such link is a duplicate and the
# whole submission is refused; with EXTEND=1 the new links of that point are
# chained after its highest-numbered live link (links are submitted in
# order, so the highest id is the chain's tail) and start the moment it ends.
active_jobs="$(qstat -u "$USER" 2>/dev/null | awk '$0 ~ /omni_/ {print $1}')"
duplicates=()
live_tails=" "   # "point=jid " pairs; the highest-numbered live link per point
for jid in $active_jobs; do
  info="$(job_info "$jid")"
  job_name="$(sed -n 's/^Job_Name = //p' <<<"$info")"
  [[ "$job_name" == "$JOB_NAME" ]] || continue
  vars="$(sed -n 's/^Variable_List = //p' <<<"$info")"
  existing_tag="$(grep -oE '(^|,)RUN_TAG=[^,]+' <<<"$vars" | head -1 | sed -E 's/(^|,)RUN_TAG=//' || true)"
  [[ -n "$existing_tag" ]] || existing_tag="v37"
  [[ "$existing_tag" == "$RUN_TAG" ]] || continue
  for point in $POINTS; do
    if grep -Eq "(^|,)POINTS=${point}(,|$)" <<<"$vars"; then
      duplicates+=("$jid:$RUN_TAG:b$point")
      current="$(sed -n "s/.* ${point}=\([^ ]*\) .*/\1/p" <<<"$live_tails")"
      if [[ -z "$current" ]]; then
        live_tails+="${point}=${jid} "
      elif [[ "${jid%%.*}" =~ ^[0-9]+$ && "${current%%.*}" =~ ^[0-9]+$ ]] && (( ${jid%%.*} > ${current%%.*} )); then
        live_tails="${live_tails/ ${point}=${current} / ${point}=${jid} }"
      fi
    fi
  done
done
if (( ${#duplicates[@]} > 0 )) && (( ! EXTEND )); then
  echo "refusing duplicate $ARM_LABEL v3.6 submission; active/held jobs already target this RUN_TAG and point:" >&2
  printf '  %s\n' "${duplicates[@]}" >&2
  echo "to lengthen a live chain without a gap: EXTEND=1 CHAIN_LINKS=<n> NCPUS=<its ncpus> POINTS=\"<points>\" bash $0" >&2
  exit 3
fi
extend_after() {  # <point> -> jid of its live tail link, or ""
  sed -n "s/.* $1=\([^ ]*\) .*/\1/p" <<<"$live_tails"
}


# A point's worker count used to be frozen in its protocol snapshot (--resume
# refused any other value). Since 2026-09-05 slow_workers is a runtime key:
# a point may move between 64- and 128-core links, the launcher records the
# override. The check below is now a NOTE so the operator sees the change.
frozen_workers() {  # <run dir> -> slow_workers from protocol_snapshot.json, or ""
  [[ -f "$1/protocol_snapshot.json" ]] || { echo ""; return; }
  python - "$1/protocol_snapshot.json" <<'PYSNAP' 2>/dev/null || echo ""
import json, sys
print((json.load(open(sys.argv[1])) or {}).get("slow_workers", ""))
PYSNAP
}
mismatched=()
for point in $POINTS; do
  frozen="$(frozen_workers "$OUT_ROOT/fig2_${ARM_PREFIX}_b${point}_${RUN_TAG}")"
  if [[ -n "$frozen" && "$frozen" != "$(( NCPUS - 1 ))" ]]; then
    mismatched+=("b$point: last slow_workers=$frozen -> now $(( NCPUS - 1 )) workers")
  fi
done
if (( ${#mismatched[@]} > 0 )); then
  echo "note: NCPUS=$NCPUS differs from the worker count these points last ran with (runtime override on resume):"
  printf '  %s\n' "${mismatched[@]}"
fi

common_vars="RUN_TAG=$RUN_TAG,TURNS=$TURNS,VALIDATE_FORMAL=$VALIDATE_FORMAL,REPO_ROOT=$REPO_ROOT"
# The launcher defaults to ARM=claude; only the DeepSeek arm needs the variable.
[[ "$ARM" == "claude" ]] || common_vars="ARM=$ARM,$common_vars"
for optional_name in PRESET OUT_ROOT CLAUDE_MODEL DEEPSEEK_MODEL OPENROUTER_ENV_FILE CLIPROXY_DIR QUOTA_POLL_S WALL_RESERVE_S WALL_S OMNIQEC_WORKER_RSS_BYTES OMNIQEC_TOTAL_MEM_BYTES OMNIQEC_SEC_WORKERS OMNIQEC_FAST_WORKERS ABLATION EXACT_CHECK_WEIGHT CAMPAIGN_ID OMNIQEC_SHARD_SPLIT REFERENCE_CODES_JSON; do
  optional_value="${!optional_name:-}"
  [[ "$optional_name" != "OUT_ROOT" ]] || optional_value="$USER_OUT_ROOT"
  if [[ -n "$optional_value" ]]; then
    [[ "$optional_value" != *,* ]] || { echo "$optional_name must not contain a comma" >&2; exit 2; }
    common_vars+=",${optional_name}=${optional_value}"
  fi
done

if (( ! VALIDATE_FORMAL )); then
  echo "!!! PROTOCOL VARIANT / CANARY submission: RUN_TAG=$RUN_TAG TURNS=$TURNS !!!"
fi
if (( EXTEND )); then
  echo "EXTEND=1: appending $CHAIN_LINKS links per point after each point's live tail link (ncpus=$NCPUS)"
else
  echo "submitting $n_points independent $ARM_LABEL chains × $CHAIN_LINKS links (ARM=$ARM, ncpus=$NCPUS, -P $PROJ)"
fi
echo "launcher=$PBS_SCRIPT"

all_ids=()
for point in $POINTS; do
  vars="POINTS=$point,$common_vars"
  previous=""
  if (( EXTEND )); then
    previous="$(extend_after "$point")"
    if [[ -n "$previous" ]]; then
      echo "b$point: extending the live chain, first new link depends on afterany:$previous"
    else
      echo "b$point: no live chain under RUN_TAG=$RUN_TAG — submitting a fresh chain"
    fi
  fi
  point_ids=()
  for (( link=1; link<=CHAIN_LINKS; link++ )); do
    if [[ -z "$previous" ]]; then
      jid="$(qsub -P "$PROJ" -N "$JOB_NAME" -l "select=1:ncpus=$NCPUS:mem=100gb" -v "$vars" "$PBS_SCRIPT")"
    else
      jid="$(qsub -P "$PROJ" -N "$JOB_NAME" -l "select=1:ncpus=$NCPUS:mem=100gb" -W "depend=afterany:$previous" -v "$vars" "$PBS_SCRIPT")"
    fi
    [[ -n "$jid" ]] || { echo "qsub returned an empty job id for b$point link $link" >&2; exit 4; }
    point_ids+=("$jid")
    all_ids+=("$jid")
    previous="$jid"
  done
  echo "b$point: ${point_ids[*]}"
done

echo "all job IDs (${#all_ids[@]}):"
printf '  %s\n' "${all_ids[@]}"
