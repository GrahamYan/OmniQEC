#!/bin/bash
# One status view for the v3.6 subscription grids. ARM=claude (default):
# runs/fig2_claudesub_b<N>_v37, runs/logs/claude_grid_v37.*, jobs named
# omni_claude_v36. ARM=deepseek (or deepseek_v36_status.sh): the DeepSeek arm's
# runs/fig2_deepseek_b<N>_v37, runs/logs/deepseek_grid_v37.*, omni_deepseek_v36
# (2026-09-10, OpenRouter). ARM=gpt (or the gpt_v36_status.sh wrapper): the ChatGPT
# arm's runs/fig2_gptsub_b<N>_v37, runs/logs/gpt_grid_v37.*, omni_gpt_v36 --
# the same launcher, protocol and checks, so the same view (a serial chain
# shows as one job walking the point list; its point column says 串行).
# Protocol: telescoping_structural_v3_7 = ONE decoder telescoping_cpu (the Mitten
# paper's telescoping decoder in its CPU configuration: S1/S2 off, S3-A/B/C
# on the GARI system of the two-basis DEM, S4 off, NC = logical error, no
# BPOSD) + v3.4 geometry + paper SI C noise uniform_paper_v1 + SEC policy
# twobasis_nonint_v1 with multi-seed certification + the budget policy
# (<=288: 20k/200k shots; >=300: 5k/50k; shots only, no per-stage time
# limits), all frozen in protocol_snapshot.json (keys decoder / telescoping /
# budget_policy / sec_policy); the launcher's V37_DEFAULTS_OK self-check and
# the validator (FORMAL_DECODER, TELESCOPING_PROTOCOL_FIELDS, budget_policy,
# FORMAL_SEC_POLICY) pin them, this view shows the decoder, the telescoping
# preset, the policy shots and the noise key).
#
#   bash omniqec_agentic/hpc/claude_v36_status.sh
#   bash omniqec_agentic/hpc/gpt_v36_status.sh          # = ARM=gpt bash claude_v36_status.sh
#
# Three answers per point: is it running (queue state, elapsed, CPU
# utilization from PBS accounting = cput / (walltime x ncpus)), how far it
# got (audited turns, avg wall time per turn, naive ETA, decode_infeasible
# candidates summed over turns, peak worker RSS seen in the shard records,
# the frozen decoder / shots / noise profile), and whether the finished point
# VALIDATED.
# The status views of the older launchers (_v33 / _v34 / _v35 dirs, BPOSD
# history) were historical deployment files and are not in the open-source tree.
set -uo pipefail

REPO="${REPO:-$HOME/OmniQEC}"
LOG_DIR="$REPO/runs/logs"
RUN_TAG="${RUN_TAG:-v37}"
[[ "$RUN_TAG" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid RUN_TAG=$RUN_TAG" >&2; exit 2; }
ARM="${ARM:-claude}"
case "$ARM" in
  claude) ARM_PREFIX="claudesub"; ARM_LABEL="Claude" ;;
  gpt)    ARM_PREFIX="gptsub";    ARM_LABEL="ChatGPT" ;;
  deepseek) ARM_PREFIX="deepseek"; ARM_LABEL="DeepSeek (OpenRouter)" ;;
  *) echo "ARM must be claude, gpt or deepseek (got ARM=$ARM)" >&2; exit 2 ;;
esac
JOB_NAME="omni_${ARM}_v36"
LOG_PREFIX="${ARM}_grid_${RUN_TAG}"

hms_to_s() { # HH:MM:SS (hours may exceed 99) -> seconds
  local IFS=:
  # shellcheck disable=SC2086
  set -- $1
  echo $(( 10#${1:-0}*3600 + 10#${2:-0}*60 + 10#${3:-0} ))
}

echo "=============== ${ARM_LABEL} v3.6 grid [$RUN_TAG]  $(date '+%F %H:%M') ==============="

echo
echo "── 作业：状态 / 已跑时长 / CPU 利用率 ──"
printf '  %-20s %-8s %-3s %-11s %-9s %s\n' "JobID" "point" "S" "elapsed" "cpu_util" "node"
JOBS=$(qstat -u "$USER" 2>/dev/null | awk '$0 ~ /omni_/ {print $1}')
shown_jobs=0
for jid in $JOBS; do
  # qstat -f folds long lines; join continuations (lines without " = ").
  info=$(qstat -f "$jid" 2>/dev/null | sed 's/^[[:space:]]*//' \
    | awk 'NR>1 && $0 !~ / = / {printf "%s", $0; next} {if (NR>1) printf "\n"; printf "%s", $0} END {print ""}')
  job_name=$(sed -n 's/^Job_Name = //p' <<<"$info")
  [[ "$job_name" == "$JOB_NAME" ]] || continue
  shown_jobs=$((shown_jobs + 1))
  state=$(sed -n 's/^job_state = //p' <<<"$info")
  point=$(grep -oE 'POINTS=[0-9:]+' <<<"$info" | head -1 | cut -d= -f2)
  if [[ -z "$point" ]]; then
    point="默认表"
  elif [[ "$point" == *:* ]]; then
    # a serial chain link carries the whole list (colon-joined); it walks
    # them left to right, so the run-dir table below says which one is live.
    point="串行$(( $(tr -cd ':' <<<"$point" | wc -c) + 1 ))点"
  else
    point="b$point"
  fi
  wall=$(sed -n 's/^resources_used\.walltime = //p' <<<"$info")
  cput=$(sed -n 's/^resources_used\.cput = //p' <<<"$info")
  ncpus=$(sed -n 's/^resources_used\.ncpus = //p' <<<"$info")
  node=$(sed -n 's/^exec_host = //p' <<<"$info" | cut -d/ -f1)
  util="-"
  if [[ -n "$wall" && -n "$cput" && -n "${ncpus:-}" && "${ncpus:-0}" != "0" ]]; then
    ws=$(hms_to_s "$wall"); cs=$(hms_to_s "$cput")
    (( ws > 0 )) && util="$(( cs * 100 / (ws * ncpus) ))%"
  fi
  printf '  %-20s %-8s %-3s %-11s %-9s %s\n' "$jid" "$point" "${state:-?}" "${wall:--}" "$util" "${node:--}"
done
(( shown_jobs > 0 )) || echo "  (队列中没有 $JOB_NAME 作业)"
echo "  (H = 接力链待命，不耗 SU；cpu_util 是开跑以来的累计平均值。"
echo "   v3.6 slow decode shards: search 16 / final 128 —— 搜索期 4候选 × X/Z × 16 = 128 jobs，"
echo "   终局每基 128 shards；shots 按预算点策略（<=288: 20k/200k，>=300: 5k/50k，只有 shots、无逐级时限），"
echo "   watchdog = 下限 + 模型：max(1800s, 8s/shot × shots × max(1, DEM列数/60000), shots × telescoping_cpu 最坏一枪模型)；"
echo "   LLM/build启动与distance/straggler尾部仍会拉低累计平均值。解码波次长期低于80%才需排查。"
echo "   decode_infeasible = 只因 watchdog 超时而不完整的候选，是终态不是停止（v3.6 终局可重试一次）；数量见下表。"
echo "   decoder_unsupported = telescoping 无法为该候选的 DEM 建系统（电路距离 2：同一探测器特征、不同逻辑效应），v3.6.1 起也是候选终态，不再是 measurement_blocked。"
echo "   decoder = 快照冻结的解码器；v3.6 正式值是 telescoping_cpu（Mitten 论文 telescoping decoder 的 CPU 配置：S1/S2 关、"
echo "   S3-A/B/C 跑 GARI 两基 DEM、S4 关，未解出的 shot（NC）计为逻辑错误；快照 telescoping 键 = 全部字段，preset telescoping_cpu_v1）。"
echo "   noise = 快照冻结的噪声 profile；正式值是 uniform_paper_v1（论文 SI C，无 idle 噪声）。)"

echo
echo "── 进度：轮数 / 平均轮速 / ETA ──"
found=0
for d in "$REPO"/runs/fig2_"${ARM_PREFIX}"_b*_"$RUN_TAG"; do
  [[ -d "$d" ]] || continue
  found=1
  name=$(basename "$d")
  done_n=$(grep -l '"status": "complete"' "$d"/turns/turn_*/turn_audit.json 2>/dev/null | wc -l | tr -d ' ')
  started=$(ls -d "$d"/turns/turn_* 2>/dev/null | wc -l | tr -d ' ')
  target_n=30
  shards="?"
  final_shards="?"
  noise_profile="?"
  decoder="?"
  preset="?"
  policy_shots="?"
  if [[ -f "$d/protocol_snapshot.json" ]]; then
    read -r target_n shards final_shards noise_profile decoder policy_shots preset < <(python - "$d/protocol_snapshot.json" <<'PY' 2>/dev/null || echo "30 ? ? ? ? ? ?"
import json, sys
p = json.load(open(sys.argv[1]))
pol = p.get("budget_policy") or {}
shots = f"{pol.get('search_shots', '?')}/{pol.get('final_shots', '?')}" if pol else "?"
tele = p.get("telescoping") or {}
print(int(p.get("max_turns") or 30), p.get("shot_shards", "?"), p.get("final_shot_shards", "?"), p.get("noise_profile") or "?", p.get("decoder") or "?", shots, tele.get("preset") or "?")
PY
)
  fi
  # v3.6 geometry (unchanged from v3.4): search shards must be 16, final
  # shards 128; the frozen noise profile must be uniform_paper_v1, the frozen
  # decoder telescoping_cpu and the frozen telescoping blob's preset
  # telescoping_cpu_v1 (a _v35 snapshot has no decoder / telescoping /
  # budget_policy key at all and shows as '?').
  shard_mark=""
  [[ "$shards" == "16" ]] || shard_mark=" ⚠shards=$shards"
  [[ "$final_shards" == "128" ]] || shard_mark="$shard_mark ⚠final_shards=$final_shards"
  [[ "$noise_profile" == "uniform_paper_v1" ]] || shard_mark="$shard_mark ⚠noise=$noise_profile"
  [[ "$decoder" == "telescoping_cpu" ]] || shard_mark="$shard_mark ⚠decoder=$decoder"
  [[ "$preset" == "telescoping_cpu_v1" ]] || shard_mark="$shard_mark ⚠telescoping=$preset"
  # decode_infeasible candidates summed over every turn's slow summary
  # (turns/turn_XXXX/cycle/slow/slow_summary.json). Terminal, not a stop.
  infeasible=$(python - "$d" <<'PY' 2>/dev/null || echo "?"
import glob, json, os, sys
total = 0
for path in sorted(glob.glob(os.path.join(sys.argv[1], "turns", "turn_*", "cycle", "slow", "slow_summary.json"))):
    try:
        total += int((json.load(open(path)) or {}).get("decode_infeasible_candidate_count") or 0)
    except Exception:
        pass
print(total)
PY
)
  unsupported=$(python - "$d" <<'PY' 2>/dev/null || echo "?"
import glob, json, os, sys
total = 0
for path in sorted(glob.glob(os.path.join(sys.argv[1], "turns", "turn_*", "cycle", "slow", "slow_summary.json"))):
    try:
        total += int((json.load(open(path)) or {}).get("decoder_unsupported_candidate_count") or 0)
    except Exception:
        pass
print(total)
PY
)
  # v3.6 (M4): peak worker RSS over every shard record that carries
  # job_execution.ru_maxrss (bytes), reported in GiB; '?' when none yet.
  peak_rss=$(python - "$d" <<'PY' 2>/dev/null || echo "?"
import glob, json, os, sys
peak = 0
for path in glob.glob(os.path.join(sys.argv[1], "turns", "turn_*", "cycle", "slow", "candidates", "*.json")) + glob.glob(os.path.join(sys.argv[1], "final_slow", "T*", "candidates", "*.json")):
    try:
        payload = json.load(open(path)) or {}
    except Exception:
        continue
    for result in payload.get("results") or []:
        for blob in ((result or {}).get("basis") or {}).values():
            for value in (blob or {}).get("shard_max_rss_bytes") or []:
                if isinstance(value, (int, float)) and value > peak:
                    peak = int(value)
print("?" if peak == 0 else f"{peak / 2**30:.2f}GiB")
PY
)
  extra=""
  if (( done_n > 0 )); then
    first_dir=$(ls -d "$d"/turns/turn_* 2>/dev/null | head -1)
    first_t=$(stat -c %Y "$first_dir" 2>/dev/null || echo 0)
    last_t=0
    for a in $(grep -l '"status": "complete"' "$d"/turns/turn_*/turn_audit.json 2>/dev/null); do
      t=$(stat -c %Y "$a" 2>/dev/null || echo 0)
      (( t > last_t )) && last_t=$t
    done
    # first_t = end of turn 0 (its audit is the last file written into the
    # turn dir), last_t = end of the newest audited turn: the window spans
    # done_n-1 turns, not done_n. Dividing by done_n understated the rate by
    # half at 2 audited turns (2026-09-04 estimate bug).
    if (( done_n > 1 && first_t > 0 && last_t > first_t )); then
      rate_min=$(( (last_t - first_t) / 60 / (done_n - 1) ))
      eta_h=$(( (target_n - done_n) * rate_min / 60 ))
      extra="   ~${rate_min}min/轮, 剩余ETA ~${eta_h}h"
    fi
  fi
  status=""
  [[ -e "$d/campaign_report.json" ]] && status="  ✅完成"
  if [[ -e "$d/FATAL_STOP.json" ]]; then
    fatal_reason=$(python - "$d/FATAL_STOP.json" <<'PY' 2>/dev/null || echo unknown
import json, sys
print((json.load(open(sys.argv[1])) or {}).get("reason") or "unknown")
PY
)
    status="$status 🛑FATAL_STOP(${fatal_reason})"
  fi
  bn="${name#fig2_${ARM_PREFIX}_}"; bn="${bn%_${RUN_TAG}}"
  if grep -qs "\[point\] ${bn} VALIDATED" "$LOG_DIR"/"${LOG_PREFIX}".*.log 2>/dev/null; then
    status="$status ✅VALIDATED"
  fi
  echo "  $name: 审计通过 ${done_n}/${target_n} (已开始 ${started}) decoder=${decoder} telescoping=${preset} shots=${policy_shots} shards=${shards}/final ${final_shards} noise=${noise_profile}${shard_mark} decode_infeasible=${infeasible} decoder_unsupported=${unsupported} peak_rss=${peak_rss}${extra}${status}"
done
(( found )) || echo "  (还没有任何 _${RUN_TAG} run 目录)"

echo
echo "── 持久停止标记（存在时 afterany 后继不会触碰 checkpoint/Generator）──"
fatal_found=0
for marker in "$REPO"/runs/fig2_"${ARM_PREFIX}"_b*_"$RUN_TAG"/FATAL_STOP.json; do
  [[ -f "$marker" ]] || continue
  fatal_found=1
  python - "$marker" <<'PY' 2>/dev/null || { echo "  $marker (无法解析)"; continue; }
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
x = json.load(open(p))
print(f"  {p.parent.name}: reason={x.get('reason')} at={x.get('created_at_utc')} log={x.get('attempt_log')}")
PY
done
if (( ! fatal_found )); then
  echo "  (无)"
else
  echo "  仅在检查并修复根因后，人工删除对应 FATAL_STOP.json 才会允许续跑。"
  [[ "$ARM" == "gpt" ]] && echo "  （串行链 FATAL_POLICY=skip：带标记的点被跳过、链继续走后面的点；删掉标记后下一节会先回头续跑它。）"
fi

echo
echo "── 最近异常（各点最新日志里的错误行，无输出 = 干净）──"
for f in $(ls -t "$LOG_DIR"/"${LOG_PREFIX}".*.log 2>/dev/null | head -8); do
  hits=$(grep -Ei 'Traceback|RuntimeError|ValueError|measurement_blocked|MEASUREMENT BLOCKED|FATAL_STOP|non-quota reason|重试烧完|CLIProxyAPI (died|重启失败)' "$f" 2>/dev/null | tail -2)
  [[ -n "$hits" ]] && { echo "  $(basename "$f"):"; sed 's/^/    /' <<<"$hits"; }
done

echo
echo "── quota/接力最新动态 ──"
for f in $(ls -t "$LOG_DIR"/"${LOG_PREFIX}".*.log 2>/dev/null | head -8); do
  line=$(grep -E '^\[stats\]|^\[quota\]|^\[point\]' "$f" 2>/dev/null | tail -1)
  [[ -n "$line" ]] && echo "  $(basename "$f"): $line"
done
echo
