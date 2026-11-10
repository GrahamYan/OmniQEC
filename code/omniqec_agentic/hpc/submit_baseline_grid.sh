#!/bin/bash
# Submit the search baselines without an LLM (paper appendix D4 / D5): one chain of LINKS jobs per (method, task, point).
#
#   PROJECT=<PBS project id> REPO_ROOT=<deployed tree> OUT_ROOT=<run root> BO_VENV=<venv with the BO packages> \
#     [QUEUE=normal] [CONDA_ENV=omniqec] [MODULES=<mod1:mod2>] [LINKS=4] [HOURS=24] [TURNS=30] [SEED_ID=r01] \
#     [RUN_TAG=bl1] [AFTER=<job id>] [DRY_RUN=1] submit_baseline_grid.sh SPEC ...
#
#   The paper's runs: bo:memory:N and bo:cnot_dense:N for the BO baseline, random:memory:N and random:cnot_dense:N
#   for N in 98 180 260 for the Random configuration; TURNS = 30 iterations (the default).
#
#   SPEC = METHOD:TASK:POINT[:NCPUS[:HOURS[:LINKS]]]   e.g.  bo:memory:360:128:18:6  random:cnot_dense:98
#   (NCPUS 64 by default; HOURS / LINKS per spec override the environment's.) PBS reserves NCPUS x HOURS SU per
#   running link against the project balance, so shorter links let more chains run at once on a small balance.
#
# Each link is nscc_baseline_grid.pbs with -W depend=afterany on the previous link: a link resumes the checkpoint,
# and a link that finds the point complete (campaign_report.json) or stopped (FATAL_STOP) exits at once. The job ids
# are appended to $OUT_ROOT/jobs.tsv (spec, link, job id). AFTER extends an existing chain: the first new link waits
# (afterany) for that job, and the new links are numbered after the chain's links already in jobs.tsv.
set -euo pipefail

: "${PROJECT:?PROJECT (the PBS project to charge) is required}"
: "${REPO_ROOT:?REPO_ROOT is required}"
: "${OUT_ROOT:?OUT_ROOT is required}"
BO_VENV="${BO_VENV:-}"
LINKS="${LINKS:-4}"
HOURS="${HOURS:-24}"
SEED_ID="${SEED_ID:-r01}"
RUN_TAG="${RUN_TAG:-bl1}"
DRY_RUN="${DRY_RUN:-0}"
AFTER="${AFTER:-}"
QUEUE="${QUEUE:-normal}"
CONDA_ENV="${CONDA_ENV:-}"
MODULES="${MODULES:-}"
[[ "$QUEUE" =~ ^[A-Za-z0-9_.@-]+$ ]] || { echo "invalid QUEUE=$QUEUE" >&2; exit 2; }
[[ "$CONDA_ENV$MODULES" != *,* ]] || { echo "CONDA_ENV / MODULES must not contain a comma (use ':' between modules)" >&2; exit 2; }
SCRIPT="$REPO_ROOT/omniqec_agentic/hpc/nscc_baseline_grid.pbs"
[[ -f "$SCRIPT" ]] || { echo "missing $SCRIPT" >&2; exit 2; }
[[ "$LINKS" =~ ^[1-9][0-9]*$ ]] || { echo "LINKS must be a positive integer" >&2; exit 2; }
(( $# > 0 )) || { echo "no SPEC given" >&2; exit 2; }
mkdir -p "$OUT_ROOT"

for spec in "$@"; do
  IFS=: read -r method task point ncpus hours links <<< "$spec"
  ncpus="${ncpus:-64}"
  hours="${hours:-$HOURS}"
  links="${links:-$LINKS}"
  [[ "$hours" =~ ^[1-9][0-9]?$ ]] && (( hours <= 24 )) || { echo "HOURS must be 1..24 in $spec" >&2; exit 2; }
  [[ "$links" =~ ^[1-9][0-9]*$ ]] || { echo "LINKS must be a positive integer in $spec" >&2; exit 2; }
  case "$method" in random|bo) ;; *) echo "bad method in $spec" >&2; exit 2 ;; esac
  case "$task" in memory) tag=m ;; cnot_dense) tag=c ;; *) echo "bad task in $spec" >&2; exit 2 ;; esac
  [[ "$point" =~ ^[0-9]+$ ]] || { echo "bad point in $spec" >&2; exit 2; }
  case "$ncpus" in 128) mem=100gb ;; 64) mem=110gb ;; *) echo "NCPUS must be 64 or 128 in $spec" >&2; exit 2 ;; esac
  [[ "$method" != "bo" || -n "$BO_VENV" ]] || { echo "bo needs BO_VENV" >&2; exit 2; }
  name="bl${method:0:2}${tag}${point}"
  vars="METHOD=$method,TASK=$task,POINT=$point,SEED_ID=$SEED_ID,RUN_TAG=$RUN_TAG,REPO_ROOT=$REPO_ROOT,OUT_ROOT=$OUT_ROOT"
  vars="$vars,WALL_S=$(( hours * 3600 ))"
  # TURNS lowers the search on resume (only lowering is accepted; the finalists come from the first TURNS turns).
  [[ -n "${TURNS:-}" ]] && vars="$vars,TURNS=$TURNS"
  [[ "$method" == "bo" ]] && vars="$vars,BO_VENV=$BO_VENV"
  [[ -n "$CONDA_ENV" ]] && vars="$vars,CONDA_ENV=$CONDA_ENV"
  [[ -n "$MODULES" ]] && vars="$vars,MODULES=${MODULES// /:}"
  prev="$AFTER"
  first=1
  if [[ -n "$AFTER" && -f "$OUT_ROOT/jobs.tsv" ]]; then
    first=$(( $(awk -F'\t' -v s="$spec" '$1 == s' "$OUT_ROOT/jobs.tsv" | wc -l) + 1 ))
  fi
  for (( link = first; link < first + links; link++ )); do
    dep=()
    [[ -n "$prev" ]] && dep=(-W "depend=afterany:$prev")
    cmd=(qsub -P "$PROJECT" -q "$QUEUE" -N "$name" -l "select=1:ncpus=$ncpus:mem=$mem" -l "walltime=$(printf '%02d' "$hours"):00:00"
         ${dep[@]+"${dep[@]}"} -v "$vars" "$SCRIPT")
    if [[ "$DRY_RUN" == "1" ]]; then
      echo "${cmd[*]}"; prev="DRY$link"; continue
    fi
    jid="$("${cmd[@]}")"
    printf '%s\t%s\t%s\n' "$spec" "$link" "$jid" >> "$OUT_ROOT/jobs.tsv"
    echo "$spec link $link -> $jid"
    prev="$jid"
  done
done
