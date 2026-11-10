#!/bin/bash
# Split the PENDING keys of a fixed-code plan into chunks and submit one PBS job per chunk.
#
#   PROJECT=<PROJECT> REPO_ROOT=<TREE> PLAN=.../fig4_ab_plan.json OUT_ROOT=/scratch/.../fig4_ab \
#   PROTOCOL_ROOTS=$HOME/fixed_code/protocols CHUNK_SIZE=20 WALLTIME=24:00:00 bash fixed_code_submit.sh
#
# DRY_RUN=1 prints the qsub commands instead of running them. Re-running after some jobs
# finished re-chunks only what is still pending (finished keys are never re-measured).
# Optional: WORKERS, SHOT_SHARDS, DECODE_SHOT_BUDGET_S, XDG_CACHE_HOME, CONDA_ENV, QUEUE (normal), NAME (omni_fixed).
set -u
: "${PROJECT:?set PROJECT=<NSCC project id>}" "${REPO_ROOT:?}" "${PLAN:?}" "${OUT_ROOT:?}" "${PROTOCOL_ROOTS:?}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
WALLTIME="${WALLTIME:-24:00:00}"
QUEUE="${QUEUE:-normal}"
NAME="${NAME:-omni_fixed}"
PBS="$REPO_ROOT/omniqec_agentic/hpc/nscc_fixed_code_measure.pbs"
[ -f "$PBS" ] || { echo "missing $PBS" >&2; exit 2; }
KEYS_DIR="$OUT_ROOT/_keys/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$KEYS_DIR"
cd "$REPO_ROOT" || exit 2
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
python -m omniqec_agentic.hpc.fixed_code_measure --plan "$PLAN" --out-root "$OUT_ROOT" --list-keys --pending > "$KEYS_DIR/pending.txt" || exit 3
TOTAL=$(grep -c . "$KEYS_DIR/pending.txt" || true)
echo "pending keys: $TOTAL (chunks of $CHUNK_SIZE)"
[ "$TOTAL" -gt 0 ] || exit 0
split -l "$CHUNK_SIZE" -a 3 "$KEYS_DIR/pending.txt" "$KEYS_DIR/chunk_"
EXTRA="OUT_ROOT=$OUT_ROOT,PROTOCOL_ROOTS=$PROTOCOL_ROOTS,REPO_ROOT=$REPO_ROOT,PLAN=$PLAN"
for v in WORKERS SHOT_SHARDS DECODE_SHOT_BUDGET_S XDG_CACHE_HOME CONDA_ENV; do
  if [ -n "${!v:-}" ]; then EXTRA="$EXTRA,$v=${!v}"; fi
done
for chunk in "$KEYS_DIR"/chunk_*; do
  cmd=(qsub -P "$PROJECT" -q "$QUEUE" -N "$NAME" -l "walltime=$WALLTIME" -v "$EXTRA,KEYS_FILE=$chunk" "$PBS")
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '%q ' "${cmd[@]}"; echo
  else
    "${cmd[@]}"
  fi
done
echo "keys files under $KEYS_DIR"
