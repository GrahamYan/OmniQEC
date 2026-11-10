#!/usr/bin/env bash
# Submit the ChatGPT (GPT) arm of the v3.6 Fig.2 grid as ONE SERIAL afterany
# chain (telescoping_structural_v3_6, the same launcher, protocol, geometry,
# budget_policy shots, telescoping_cpu decoder and FATAL_STOP branches as the
# Claude arm -- nscc_claude_grid_v36.pbs with ARM=gpt).
#
# WHY SERIAL. The ChatGPT subscription behind the GPT arm sustains one or two
# campaigns at a time (its usage window would stall thirteen). So instead of
# thirteen independent chains (submit_claude_v36_grid.sh), every link of a
# chain carries a whole point LIST, colon-joined (POINTS=360:288:...:60 -- a
# qsub -v value must not contain spaces; the launcher normalises it). A link
# walks its list left to right (big codes first), resumes the first unfinished
# point from its checkpoint and only moves on once that point is complete --
# one chain = one campaign at a time = one 128-core node.
#
# TWO CHAINS = TWO CAMPAIGNS AT A TIME. Run this script once per chain with
# DISJOINT point lists (the user's 2026-09-04 cap: at most two GPT campaigns
# in parallel). Both lists descending, alternated so the two chains carry
# similar work:
#   POINTS="360 260 220 180 140 98 60" CHAIN_LINKS=20 bash .../submit_gpt_v36_serial.sh
#   POINTS="288 240 200 160 120 80"    CHAIN_LINKS=15 bash .../submit_gpt_v36_serial.sh
# A point may live in ONE chain only: a submission whose list overlaps a live
# chain's list is refused (two links on the same point would share its
# checkpoint), and EXTEND=1 appends to the live chain with the IDENTICAL list.
# FATAL_POLICY=skip (default here): a FATAL_STOP on one point is logged and
# SKIPPED, the marker stays for a human, the link walks on; delete the marker
# after fixing the cause and the next link goes back to that point first.
#
# Every submission needs PROJ=<PROJECT>, the PBS project the links are charged to
# (no default; the examples below leave it out for brevity).
#   PROJ=<PROJECT> bash ~/OmniQEC/omniqec_agentic/hpc/submit_gpt_v36_serial.sh # 13 points, 20 links, -P <PROJECT>
#   CHAIN_LINKS=10 bash .../submit_gpt_v36_serial.sh                           # shorter chain
#   EXTEND=1 CHAIN_LINKS=10 bash .../submit_gpt_v36_serial.sh                  # append 10 links after the live chain's LAST link (no gap)
#   QUOTA_MAX_WAIT_S=7200 bash .../submit_gpt_v36_serial.sh                    # cap the SU burnt idling in a ChatGPT usage window
#   POINTS=144 TURNS=2 RUN_TAG=v37_canary VALIDATE_FORMAL=0 CHAIN_LINKS=1 \
#     bash .../submit_gpt_v36_serial.sh                                        # two-turn canary (its run dir must NOT be rsynced over)
#   NCPUS=64 POINTS="260 220 180 140 98 60" AFTER=<jobid> CHAIN_LINKS=15 \
#     bash .../submit_gpt_v36_serial.sh                                        # 64-core q2 continuation chained after a live chain's last link
#
# NCPUS=64 (2026-09-04): half the SU rate for points that have not started
# (a turn is mostly serial LLM calls plus the 4-process build wave; the
# decode wave fills a node for minutes). A point that froze slow_workers=127
# must stay on NCPUS=128 (checked below). AFTER=<jobid> makes the first new
# link depend on afterany:<jobid>, so a 64-core continuation over the rest of
# a list starts only when the 128-core chain over its first point is done --
# still at most one campaign per chain.
#
# Every link is one 24 h x 128-core job (3072 SU if fully used); a link that
# finds every point complete exits at once. The queue cap is ~100 jobs per
# user: with the Claude arm's 68 links queued, keep this chain <= 30 links and
# grow it with EXTEND=1 as links leave the queue. The whole GPT arm costs
# about as much SU as the Claude arm (decoding dominates, not the LLM):
# check the project's SU balance before submitting.
#
# A live chain with the same list under the same RUN_TAG is refused unless
# EXTEND=1: two links running the same list would both open the first
# unfinished point's checkpoint (flock only catches the worst case). EXTEND=1
# chains the new links after that chain's highest-numbered live link, so
# nothing overlaps. A live chain with a DIFFERENT but overlapping list is
# always refused; a disjoint list becomes an independent parallel chain.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/OmniQEC}"
PBS_SCRIPT="${PBS_SCRIPT:-$REPO_ROOT/omniqec_agentic/hpc/nscc_claude_grid_v36.pbs}"
# The whole grid, DESCENDING = run order inside every link.
POINTS="${POINTS:-360 288 260 240 220 200 180 160 140 120 98 80 60}"
CHAIN_LINKS="${CHAIN_LINKS:-20}"
# 2026-11-06: PRESET=fig2_final | fig3_final submits the final protocols (docs/final_protocol_20261106.md);
# the launcher sets and checks the protocol keys, its run tag defaults to f2final / f3final.
PRESET="${PRESET:-}"
case "$PRESET" in
  "") RUN_TAG="${RUN_TAG:-v37}" ;;
  fig2_final) RUN_TAG="${RUN_TAG:-f2final}" ;;
  fig3_final) RUN_TAG="${RUN_TAG:-f3final}" ;;
  *) echo "PRESET must be fig2_final or fig3_final (got $PRESET)" >&2; exit 2 ;;
esac
TURNS="${TURNS:-30}"
VALIDATE_FORMAL="${VALIDATE_FORMAL:-1}"
PROJ="${PROJ:-}"
[[ -n "$PROJ" ]] || { echo "set PROJ=<PROJECT> (the PBS project to charge)" >&2; exit 2; }
EXTEND="${EXTEND:-0}"
FATAL_POLICY="${FATAL_POLICY:-skip}"
NCPUS="${NCPUS:-128}"
AFTER="${AFTER:-}"
USER_OUT_ROOT="${OUT_ROOT:-}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/runs}"
ARM_PREFIX="gptsub"
JOB_NAME="omni_gpt_v36"

[[ -f "$PBS_SCRIPT" ]] || { echo "launcher not found: $PBS_SCRIPT" >&2; exit 2; }
command -v qsub >/dev/null || { echo "qsub is unavailable; run this on an NSCC login node" >&2; exit 2; }
command -v qstat >/dev/null || { echo "qstat is unavailable; run this on an NSCC login node" >&2; exit 2; }
[[ "$CHAIN_LINKS" =~ ^[1-9][0-9]*$ ]] || { echo "CHAIN_LINKS must be a positive integer" >&2; exit 2; }
[[ "$TURNS" =~ ^[1-9][0-9]*$ ]] || { echo "TURNS must be a positive integer" >&2; exit 2; }
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid RUN_TAG=$RUN_TAG" >&2; exit 2; }
[[ "$VALIDATE_FORMAL" == "0" || "$VALIDATE_FORMAL" == "1" ]] || { echo "VALIDATE_FORMAL must be 0 or 1" >&2; exit 2; }
[[ "$EXTEND" == "0" || "$EXTEND" == "1" ]] || { echo "EXTEND must be 0 or 1" >&2; exit 2; }
[[ "$PROJ" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid PROJ=$PROJ" >&2; exit 2; }
[[ "$FATAL_POLICY" == "stop" || "$FATAL_POLICY" == "skip" ]] || { echo "FATAL_POLICY must be stop or skip" >&2; exit 2; }
[[ "$NCPUS" == "128" || "$NCPUS" == "64" ]] || { echo "NCPUS must be 128 (q4 whole node) or 64 (q2 node)" >&2; exit 2; }
[[ -z "$AFTER" || "$AFTER" =~ ^[0-9]+(\.[A-Za-z0-9_.-]+)?$ ]] || { echo "AFTER must be a PBS job id" >&2; exit 2; }
if (( VALIDATE_FORMAL )) && [[ -n "$PRESET" ]] && { [[ "$RUN_TAG" == "v37" ]] || (( TURNS != 30 )); }; then
  echo "PRESET=$PRESET with VALIDATE_FORMAL=1 requires TURNS=30 and a RUN_TAG other than v37" >&2
  exit 2
fi
if (( VALIDATE_FORMAL )) && [[ -z "$PRESET" ]] && { [[ "$RUN_TAG" != "v37" ]] || (( TURNS != 30 )); }; then
  echo "formal submission requires RUN_TAG=v37 and TURNS=30" >&2
  echo "for a canary set VALIDATE_FORMAL=0 explicitly" >&2
  exit 2
fi

points=()
for point in $POINTS; do
  [[ "$point" =~ ^[1-9][0-9]*$ ]] || { echo "invalid budget point: $point" >&2; exit 2; }
  for seen in "${points[@]:-}"; do
    [[ "$seen" != "$point" ]] || { echo "duplicate point in POINTS: $point" >&2; exit 2; }
  done
  points+=("$point")
done
(( ${#points[@]} > 0 )) || { echo "POINTS is empty" >&2; exit 2; }
POINTS_JOINED="$(IFS=:; echo "${points[*]}")"

# Normalize qstat -f's folded records into one key/value per line.
job_info() {
  qstat -f "$1" 2>/dev/null | sed 's/^[[:space:]]*//' \
    | awk 'NR>1 && $0 !~ / = / {printf "%s", $0; next} {if (NR>1) printf "\n"; printf "%s", $0} END {print ""}'
}

# Every live (queued, held or running) link under this RUN_TAG, with its
# point list: same list -> the chain this submission may extend; a different
# list sharing a point -> a conflict (one point, one chain).
same_ids=()
conflicts=()
for jid in $(qstat -u "$USER" 2>/dev/null | awk '$0 ~ /omni_gpt/ {print $1}'); do
  info="$(job_info "$jid")"
  job_name="$(sed -n 's/^Job_Name = //p' <<<"$info")"
  [[ "$job_name" == "$JOB_NAME" ]] || continue
  vars="$(sed -n 's/^Variable_List = //p' <<<"$info")"
  existing_tag="$(grep -oE '(^|,)RUN_TAG=[^,]+' <<<"$vars" | head -1 | sed -E 's/(^|,)RUN_TAG=//' || true)"
  [[ -n "$existing_tag" ]] || existing_tag="v37"
  [[ "$existing_tag" == "$RUN_TAG" ]] || continue
  existing_points="$(grep -oE '(^|,)POINTS=[0-9: ]+' <<<"$vars" | head -1 | sed -E 's/(^|,)POINTS=//' || true)"
  if [[ "$existing_points" == "$POINTS_JOINED" ]]; then
    same_ids+=("$jid")
    continue
  fi
  for point in "${points[@]}"; do
    if [[ ":${existing_points//[ ]/:}:" == *":$point:"* ]]; then
      conflicts+=("$jid:b$point")
      break
    fi
  done
done

if (( ${#conflicts[@]} > 0 )); then
  echo "refusing: these live $JOB_NAME links (RUN_TAG=$RUN_TAG) carry a DIFFERENT point list that shares a point with [${points[*]}] — a point may live in one chain only:" >&2
  printf '  %s\n' "${conflicts[@]}" >&2
  exit 3
fi

previous=""
if (( ${#same_ids[@]} > 0 )); then
  if (( ! EXTEND )); then
    echo "refusing a second chain over the same list [${points[*]}]; live $JOB_NAME links already run it under RUN_TAG=$RUN_TAG:" >&2
    printf '  %s\n' "${same_ids[@]}" >&2
    echo "to lengthen that chain without a gap: EXTEND=1 CHAIN_LINKS=<n> POINTS=\"${points[*]}\" bash $0" >&2
    exit 3
  fi
  # The last link is the highest-numbered one (links are submitted in order).
  previous="$(printf '%s\n' "${same_ids[@]}" | sort -t. -k1,1n | tail -1)"
  echo "extending the live chain over [${points[*]}]: new links depend on afterany:$previous"
elif (( EXTEND )); then
  echo "EXTEND=1 but no live $JOB_NAME chain over [${points[*]}] under RUN_TAG=$RUN_TAG — submitting a fresh chain"
fi


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

if [[ -n "$AFTER" ]]; then
  if [[ -n "$previous" ]]; then
    echo "AFTER=$AFTER conflicts with EXTEND=1 (which already chains after $previous)" >&2
    exit 2
  fi
  previous="$AFTER"
  echo "first link depends on afterany:$AFTER"
fi

common_vars="ARM=gpt,POINTS=$POINTS_JOINED,FATAL_POLICY=$FATAL_POLICY,RUN_TAG=$RUN_TAG,TURNS=$TURNS,VALIDATE_FORMAL=$VALIDATE_FORMAL,REPO_ROOT=$REPO_ROOT"
for optional_name in PRESET OUT_ROOT GPT_MODEL GPT_ENDPOINT_FILE CLIPROXY_DIR QUOTA_POLL_S QUOTA_MAX_WAIT_S WALL_RESERVE_S WALL_S STRICT_RETRY_MAX OMNIQEC_WORKER_RSS_BYTES OMNIQEC_TOTAL_MEM_BYTES OMNIQEC_SEC_WORKERS OMNIQEC_FAST_WORKERS OMNIQEC_SHARD_SPLIT REFERENCE_CODES_JSON; do
  optional_value="${!optional_name:-}"
  [[ "$optional_name" != "OUT_ROOT" ]] || optional_value="$USER_OUT_ROOT"
  if [[ -n "$optional_value" ]]; then
    [[ "$optional_value" != *,* && "$optional_value" != *" "* ]] || { echo "$optional_name must not contain a comma or a space" >&2; exit 2; }
    common_vars+=",${optional_name}=${optional_value}"
  fi
done

if (( ! VALIDATE_FORMAL )); then
  echo "!!! PROTOCOL VARIANT / CANARY submission: RUN_TAG=$RUN_TAG TURNS=$TURNS !!!"
fi
echo "submitting ONE serial chain × $CHAIN_LINKS links over ${#points[@]} points [${points[*]}] (ARM=gpt, FATAL_POLICY=$FATAL_POLICY, ncpus=$NCPUS, -P $PROJ)"
echo "launcher=$PBS_SCRIPT"

ids=()
for (( link=1; link<=CHAIN_LINKS; link++ )); do
  if [[ -z "$previous" ]]; then
    jid="$(qsub -P "$PROJ" -N "$JOB_NAME" -l "select=1:ncpus=$NCPUS:mem=100gb" -v "$common_vars" "$PBS_SCRIPT")"
  else
    jid="$(qsub -P "$PROJ" -N "$JOB_NAME" -l "select=1:ncpus=$NCPUS:mem=100gb" -W "depend=afterany:$previous" -v "$common_vars" "$PBS_SCRIPT")"
  fi
  [[ -n "$jid" ]] || { echo "qsub returned an empty job id for link $link" >&2; exit 4; }
  ids+=("$jid")
  previous="$jid"
done

echo "chain: ${ids[*]}"
echo "status: bash $REPO_ROOT/omniqec_agentic/hpc/gpt_v36_status.sh"
echo "log:    tail -40 $REPO_ROOT/runs/logs/gpt_grid_${RUN_TAG}.${ids[0]}.log   (after the first link starts)"
