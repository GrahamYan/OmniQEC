"""Where did a campaign turn's wall time go? (v3.6 utilisation diagnosis)

    python omniqec_agentic/hpc/turn_timeline.py runs/fig2_claudesub_b360_v36            # latest turn with an audit
    python omniqec_agentic/hpc/turn_timeline.py runs/fig2_claudesub_b360_v36 --turn 0
    python omniqec_agentic/hpc/turn_timeline.py runs/fig2_claudesub_b360_v36 --all      # one summary line per turn

The campaign writes no phase timers, so this reconstructs the timeline from
artifact mtimes (each milestone file is written when its phase ends) plus the
runtime_s fields the slow queue does record. Read the phase windows as
"nothing else was running on the node except this phase":

  orchestrator  turn start -> plan.json               (planner LLM calls, tool loop)
  generation    plan.json -> last generator response  (proposal LLM calls)
  fast          last generator response -> fast/status.json (screening, promotion)
  slow          fast done -> slow_summary.json         (build + SEC + distance + decode shards)
  wrap-up       slow done -> turn_audit.json           (ranking, reflection, audit, checkpoint)

Inside `slow`, each candidate JSON carries per basis `runtime_s` = the SUM of
its shard runtimes (CPU-ish, not wall) plus `shard_runtime_s` per shard; the
candidate's decode WALL is the slowest shard of its slower basis (X and Z
shards are submitted together). `runtime_s` at candidate level = prepare_s +
the basis sums; the build job (stim + DEM + GARI + SEC synthesis/certification,
ONE process per candidate) records no timer, so its wall is what is left of
the slow phase once the decode wall is subtracted. Only the decode shards can
occupy ~127 workers; the build wave occupies at most 4.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys


def _mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def _load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh) or {}
    except Exception:
        return {}


def _hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 3600:d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def _turn_dirs(run_dir: str) -> list[str]:
    return sorted(glob.glob(os.path.join(run_dir, "turns", "turn_*")))


def _turn_start(run_dir: str, turn_dir: str) -> float | None:
    """Previous turn's audit mtime is the truest start; turn 0 falls back to
    the earliest file in its own directory."""
    dirs = _turn_dirs(run_dir)
    idx = dirs.index(turn_dir) if turn_dir in dirs else -1
    if idx > 0:
        prev_audit = os.path.join(dirs[idx - 1], "turn_audit.json")
        t = _mtime(prev_audit)
        if t:
            return t
    earliest = None
    for root, _dirs, files in os.walk(turn_dir):
        for f in files:
            t = _mtime(os.path.join(root, f))
            if t and (earliest is None or t < earliest):
                earliest = t
    return earliest


def _workers(run_dir: str) -> int:
    """slow_workers frozen in the protocol snapshot (127 on a 128-core NSCC
    node, 63 on 64 cores, CPUS-1 on a training Pod); 127 when unknown."""
    snap = _load(os.path.join(run_dir, "protocol_snapshot.json")) or {}
    try:
        n = int(snap.get("slow_workers") or 0)
    except (TypeError, ValueError):
        n = 0
    return n if n > 0 else 127


def analyse(run_dir: str, turn_dir: str) -> dict:
    start = _turn_start(run_dir, turn_dir)
    if start is None:
        return {"turn": os.path.basename(turn_dir), "error": "no files"}
    plan = _mtime(os.path.join(turn_dir, "plan.json"))
    orch_calls = sorted(glob.glob(os.path.join(turn_dir, "orchestrator", "llm_response_*.txt")))
    gen_files = sorted(
        (p for p in glob.glob(os.path.join(turn_dir, "cycle", "fast", "generations", "**", "*"), recursive=True) if os.path.isfile(p)),
        key=lambda p: _mtime(p) or 0,
    )
    gen_last = _mtime(gen_files[-1]) if gen_files else None
    fast_done = _mtime(os.path.join(turn_dir, "cycle", "fast", "funsearch_summary.json")) or _mtime(os.path.join(turn_dir, "cycle", "fast", "status.json"))
    slow_summary_path = os.path.join(turn_dir, "cycle", "slow", "slow_summary.json")
    slow_done = _mtime(slow_summary_path)
    slow_summary = _load(slow_summary_path)
    audit = _mtime(os.path.join(turn_dir, "turn_audit.json"))
    end = audit or slow_done or fast_done or gen_last or plan or start

    candidates = []
    for cpath in sorted(glob.glob(os.path.join(turn_dir, "cycle", "slow", "candidates", "*.json"))):
        c = _load(cpath)
        done = _mtime(cpath) or 0
        total = float(c.get("runtime_s") or 0)
        decode_cpu = 0.0   # sum of shard runtimes over both bases (core-seconds)
        decode_wall = 0.0  # slowest shard (X and Z shards run together)
        shards = None
        shots = None
        for r in c.get("results") or []:
            for basis, blob in ((r or {}).get("basis") or {}).items():
                blob = blob or {}
                basis_sum = float(blob.get("runtime_s") or 0)
                decode_cpu += basis_sum
                per_shard = [float(x) for x in (blob.get("shard_runtime_s") or []) if x is not None]
                if per_shard:
                    decode_wall = max(decode_wall, max(per_shard))
                elif blob.get("shard_count"):
                    decode_wall = max(decode_wall, basis_sum / max(1, int(blob["shard_count"])))
                shards = shards or blob.get("shard_count")
                shots = shots or blob.get("shots")
        dist = float((((c.get("distance_evidence") or {}).get("exact_distance") or {}).get("code") or {}).get("runtime_s") or 0)
        candidates.append({
            "id": str(c.get("program_id") or os.path.basename(cpath))[:28],
            "status": c.get("status"),
            "done_at": done,
            "runtime_s": total,
            "decode_cpu_s": decode_cpu,
            "decode_wall_s": decode_wall,
            "prepare_s": max(0.0, total - decode_cpu),
            "distance_s": dist,
            "shards": shards,
            "shots": shots,
            "s_per_shot": (decode_cpu / (2 * float(shots))) if shots else None,
        })

    phases = []
    def add(name, a, b):
        if a is not None and b is not None and b >= a:
            phases.append((name, a, b))
    add("orchestrator (planner LLM + tools)", start, plan)
    add("generation (proposal LLM calls)", plan, gen_last)
    add("fast screening / promotion", gen_last, fast_done)
    add("slow queue (build+SEC+distance+decode)", fast_done, slow_done)
    add("wrap-up (ranking, reflection, audit)", slow_done, audit)

    turn_wall = end - start
    decode_wall = sum(c["decode_wall_s"] for c in candidates)  # upper bound (candidates' waves one after another)
    decode_wall_parallel = max((c["decode_wall_s"] for c in candidates), default=0.0)  # lower bound (all waves together)
    decode_core_s = sum(c["decode_cpu_s"] for c in candidates)
    return {
        "turn": os.path.basename(turn_dir),
        "workers": _workers(run_dir),
        "start": start,
        "end": end,
        "turn_wall_s": turn_wall,
        "phases": phases,
        "orchestrator_calls": len(orch_calls),
        "generation_files": len(gen_files),
        "candidates": candidates,
        "slow_runtime_s": float(slow_summary.get("runtime_s") or 0),
        "decode_wall_serial_s": decode_wall,
        "decode_wall_parallel_s": decode_wall_parallel,
        "decode_core_s": decode_core_s,
        "complete": bool(audit),
    }


def print_turn(info: dict) -> None:
    if "error" in info:
        print(f"{info['turn']}: {info['error']}")
        return
    print(f"=== {info['turn']}  wall {_hms(info['turn_wall_s'])}  ({dt.datetime.fromtimestamp(info['start']):%m-%d %H:%M} -> {dt.datetime.fromtimestamp(info['end']):%m-%d %H:%M}){'' if info['complete'] else '  [turn not audited yet: partial]'}")
    print(f"  {'phase':44s} {'wall':>9s} {'share':>6s}")
    for name, a, b in info["phases"]:
        w = b - a
        print(f"  {name:44s} {_hms(w):>9s} {100 * w / max(1, info['turn_wall_s']):5.0f}%")
    print(f"  orchestrator LLM calls: {info['orchestrator_calls']}   generator response files: {info['generation_files']}")
    if info["candidates"]:
        print(f"  slow candidates ({len(info['candidates'])}; slow_summary.runtime_s={_hms(info['slow_runtime_s'])}):")
        print(f"    {'candidate':28s} {'status':10s} {'decode wall':>11s} {'decode core-min':>15s} {'s/shot':>7s} {'prepare':>8s} {'shards':>6s} {'shots':>7s}")
        for c in info["candidates"]:
            sps = f"{c['s_per_shot']:.2f}" if c["s_per_shot"] is not None else "?"
            print(f"    {c['id']:28s} {str(c['status'])[:10]:10s} {_hms(c['decode_wall_s']):>11s} {c['decode_cpu_s'] / 60:15.0f} {sps:>7s} {_hms(c['prepare_s']):>8s} {str(c['shards']):>6s} {str(c['shots']):>7s}")
        lo, hi = info["decode_wall_parallel_s"], info["decode_wall_serial_s"]
        slow = next((b - a for n, a, b in info["phases"] if n.startswith("slow")), 0.0)
        workers = info.get("workers", 127)
        node_core_min = (workers + 1) * info["turn_wall_s"] / 60
        print(f"  decode wall (the only phase that can fill ~{workers} workers): {_hms(lo)}-{_hms(hi)} of the slow phase's {_hms(slow)}; the remaining ~{_hms(max(0.0, slow - lo))} is the build wave (SEC synthesis/certification, <=4 processes)")
        print(f"  decode core-minutes {info['decode_core_s'] / 60:.0f} vs allocated core-minutes {node_core_min:.0f} ({workers + 1} cores over the turn) -> decode alone = {100 * info['decode_core_s'] / 60 / max(1, node_core_min):.0f}% utilisation; add ~4 cores x build wave for the rest")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--turn", type=int, default=None, help="turn index (default: latest audited turn)")
    ap.add_argument("--all", action="store_true", help="one summary line per turn")
    args = ap.parse_args()
    dirs = _turn_dirs(args.run_dir)
    if not dirs:
        print(f"no turns under {args.run_dir}", file=sys.stderr)
        return 2
    if args.all:
        print(f"{'turn':10s} {'wall':>9s} {'orch':>9s} {'gen':>9s} {'fast':>9s} {'slow':>9s} {'wrap':>9s} {'decode(lo-hi)':>19s}")
        for d in dirs:
            info = analyse(args.run_dir, d)
            if "error" in info:
                continue
            ph = {n.split(" ")[0]: b - a for n, a, b in info["phases"]}
            print(f"{info['turn']:10s} {_hms(info['turn_wall_s']):>9s} {_hms(ph.get('orchestrator', 0)):>9s} {_hms(ph.get('generation', 0)):>9s} {_hms(ph.get('fast', 0)):>9s} {_hms(ph.get('slow', 0)):>9s} {_hms(ph.get('wrap-up', 0)):>9s} {_hms(info['decode_wall_parallel_s']) + '-' + _hms(info['decode_wall_serial_s']):>19s}{'' if info['complete'] else '  (partial)'}")
        return 0
    if args.turn is None:
        audited = [d for d in dirs if os.path.exists(os.path.join(d, "turn_audit.json"))]
        target = audited[-1] if audited else dirs[-1]
    else:
        target = os.path.join(args.run_dir, "turns", f"turn_{args.turn:04d}")
    print_turn(analyse(args.run_dir, target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
