"""Slow-queue rule of the search baselines (2026-11-06; plan B §4-§5, plan A §5-§6).

The pool is cumulative: every code the run's fast loop has seen that is legal for the slow loop -- constructed,
structurally valid, k >= 1, d_search >= the promotion floor, compiled max check weight within the run's gate -- and
has not been nominated before. Codes are identified as the Full search identifies them: code_hash plus the pynauty
canonical form, falling back to code_hash when the canonical form is unavailable (the Full search's 60 s rule,
memory/fingerprints.py). The run-level structural ledger store (structural_ledger.jsonl, appended by the campaign
graph after every turn) holds exactly these codes with their specs, and the current turn's cycle ledger adds this
turn's ones.

``random`` takes the turn's slots uniformly without replacement from the pool. ``bo`` does the same for its first
``init_random_turns`` turns (same seed key, so a random and a bo run of one seed key nominate the same codes there)
and from then on asks the Gaussian-process model (baselines/bo.py) for the batch, training on every measurement
of the run so far -- the resident reference code included, which is a public observation (plan B §8).

A nominated code is never nominated again, whatever its measurement outcome (plan B §6: a failed or timed-out
measurement still holds its slot). Nominations are read back from the slow queues of the turns before this one,
so a turn retried from its checkpoint draws the same batch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from omniqec_agentic import ranking

SELECTION_IDS = {"random": "uniform_pool_v1", "bo": "bo_pool_v1"}


def baseline_seed(key: str) -> int:
    """A 63-bit integer seed derived from a text key (stable across machines and Python versions)."""

    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") >> 1


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _turn_dir(run_dir: Path, turn: int) -> Path:
    return Path(run_dir) / "turns" / f"turn_{int(turn):04d}"


def nominated_rows(run_dir: Path, *, before_turn: int) -> list[dict[str, Any]]:
    """Every slow-queue row of the turns before ``before_turn`` (the codes this run has nominated)."""

    rows: list[dict[str, Any]] = []
    for turn in range(int(before_turn)):
        rows.extend(_jsonl(_turn_dir(run_dir, turn) / "cycle" / "slow_queue.jsonl"))
    return rows


def _identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("code_hash") or ""), str(row.get("canonical_hash") or "")


def _int_or_none(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def reference_identity(resident_baseline: tuple[dict[str, Any], str] | None) -> tuple[str, str] | None:
    """(code_hash, canonical_hash) of the budget point's resident reference code, None where there is none."""

    if resident_baseline is None:
        return None
    from omniqec_agentic.construction.build import build_matrices
    from omniqec_agentic.memory.fingerprints import canonical_code_hash
    from omniqec_agentic.verification.structural import code_hash

    hx, hz = build_matrices(dict(resident_baseline[0]))
    return code_hash(hx, hz), str(canonical_code_hash(hx, hz) or "")


def candidate_pool(
    *,
    run_dir: Path,
    cycle_ledger_path: str | None,
    nominated: list[dict[str, Any]],
    measured_hashes: set[str] | None,
    gate: int | None,
    reference: tuple[str, str] | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """The legal, distinct, never-nominated codes of the run so far, in code_hash order, and why the others left."""

    blocked_hash = {str(item) for item in (measured_hashes or set())}
    blocked_canonical: set[str] = set()
    for row in nominated:
        code_hash, canonical = _identity(row)
        if code_hash:
            blocked_hash.add(code_hash)
        if canonical:
            blocked_canonical.add(canonical)
    reference_hash, reference_canonical = reference or ("", "")
    rows = _jsonl(Path(run_dir) / "structural_ledger.jsonl")
    if cycle_ledger_path:
        rows += _jsonl(Path(cycle_ledger_path))
    counts = {
        "rows_seen": 0,
        "distinct_codes": 0,
        "no_spec": 0,
        "below_floor": 0,
        "over_weight_gate": 0,
        "nominated_or_measured": 0,
        "twin_of_nominated": 0,
        "reference_code": 0,
        "twin_in_pool": 0,
    }
    seen_hash: set[str] = set()
    pool_canonical: set[str] = set()
    pool: list[dict[str, Any]] = []
    # Earliest row per code_hash wins (the store's own reader rule); the pool's twin dedup keeps the
    # earliest-seen realisation of a canonical class.
    for row in rows:
        code_hash, canonical = _identity(row)
        if not code_hash or code_hash in seen_hash:
            continue
        seen_hash.add(code_hash)
        counts["rows_seen"] += 1
        if not isinstance(row.get("spec"), dict) or not str(row.get("construct_source") or "").strip():
            counts["no_spec"] += 1
            continue
        k = _int_or_none(row.get("k"))
        d_search = _int_or_none(row.get("d_search") if row.get("d_search") is not None else row.get("d_proxy"))
        if k is None or k < 1 or d_search is None or d_search < ranking.MIN_PROMOTION_DISTANCE:
            counts["below_floor"] += 1
            continue
        weight = _int_or_none(row.get("max_check_weight"))
        if gate is not None and weight is not None and weight > int(gate):
            counts["over_weight_gate"] += 1
            continue
        if reference_hash and (code_hash == reference_hash or (canonical and canonical == reference_canonical)):
            counts["reference_code"] += 1
            continue
        if code_hash in blocked_hash:
            counts["nominated_or_measured"] += 1
            continue
        if canonical and canonical in blocked_canonical:
            counts["twin_of_nominated"] += 1
            continue
        if canonical and canonical in pool_canonical:
            counts["twin_in_pool"] += 1
            continue
        if canonical:
            pool_canonical.add(canonical)
        pool.append(row)
    pool.sort(key=lambda row: str(row.get("code_hash")))
    counts["distinct_codes"] = len(pool)
    return pool, counts


def _counts(basis: dict[str, Any]) -> tuple[int, int] | None:
    failures, shots = _int_or_none(basis.get("failures")), _int_or_none(basis.get("shots"))
    if failures is None or shots is None or shots <= 0:
        return None
    return failures, shots


def measured_observations(run_dir: Path, *, before_turn: int) -> list[dict[str, Any]]:
    """Per-basis failures and shots of every successful slow measurement of the turns before ``before_turn``.

    One row per code (its largest-shot measurement; a code is measured once in the search). Read from each turn's
    slow_results.jsonl and the candidate file it points to; a path that no longer resolves (a run directory that
    moved) is looked up by file name in the turn's candidates directory."""

    by_hash: dict[str, dict[str, Any]] = {}
    for turn in range(int(before_turn)):
        slow_dir = _turn_dir(run_dir, turn) / "cycle" / "slow"
        for row in _jsonl(slow_dir / "slow_results.jsonl"):
            if str(row.get("status") or "") != "ok" or row.get("per_logical_per_round") is None:
                continue
            path = Path(str(row.get("path") or ""))
            if not path.is_file():
                path = slow_dir / "candidates" / path.name
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            results = payload.get("results") or []
            if not results:
                continue
            basis = (results[0] or {}).get("basis") or {}
            x, z = _counts(basis.get("X") or {}), _counts(basis.get("Z") or {})
            spec = payload.get("spec")
            code_hash = str(row.get("code_hash") or "")
            if x is None or z is None or not isinstance(spec, dict) or not code_hash:
                continue
            rounds = _int_or_none((results[0] or {}).get("rounds"))
            item = {
                "code_hash": code_hash,
                "spec": spec,
                "k": _int_or_none(row.get("k")),
                "rounds": rounds,
                "fail_x": x[0],
                "shots_x": x[1],
                "fail_z": z[0],
                "shots_z": z[1],
                "is_baseline": bool(row.get("is_baseline")),
                "turn": turn,
            }
            held = by_hash.get(code_hash)
            if held is None or item["shots_x"] + item["shots_z"] > held["shots_x"] + held["shots_z"]:
                by_hash[code_hash] = item
    return [by_hash[key] for key in sorted(by_hash)]


def _uniform_picks(pool: list[dict[str, Any]], *, q: int, seed: int) -> list[dict[str, Any]]:
    if not pool or q <= 0:
        return []
    rng = np.random.Generator(np.random.PCG64(seed))
    order = rng.choice(len(pool), size=min(int(q), len(pool)), replace=False)
    return [pool[int(index)] for index in order]


def _bo_picks(
    *,
    pool: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    q: int,
    seed: int,
    state_dir: Path,
    turn: int,
    blocks: int,
    config: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from omniqec_agentic.baselines import bo
    from omniqec_agentic.construction.build import build_matrices

    observed = []
    for item in observations:
        if item.get("k") is None or item.get("rounds") is None:
            continue
        hx, hz = build_matrices(dict(item["spec"]))
        observed.append(
            bo.Observation(
                code_hash=str(item["code_hash"]),
                hx=hx,
                hz=hz,
                k=int(item["k"]),
                rounds=int(item["rounds"]),
                blocks=int(blocks),
                fail_x=int(item["fail_x"]),
                shots_x=int(item["shots_x"]),
                fail_z=int(item["fail_z"]),
                shots_z=int(item["shots_z"]),
            )
        )
    items = []
    for row in pool:
        hx, hz = build_matrices(dict(row["spec"]))
        items.append(bo.PoolItem(code_hash=str(row["code_hash"]), hx=hx, hz=hz))
    kwargs: dict[str, Any] = {}
    if config:
        kwargs["config"] = config
    chosen, diagnostics = bo.select_batch(
        observations=observed, pool=items, q=int(q), seed=int(seed), state_dir=Path(state_dir), turn=int(turn), **kwargs
    )
    by_hash = {str(row["code_hash"]): row for row in pool}
    picks = [by_hash[code_hash] for code_hash in chosen if code_hash in by_hash]
    if len(picks) != len(chosen) or len(set(chosen)) != len(chosen):
        raise RuntimeError("the BO selector returned codes outside the pool or a repeated code")
    return picks, diagnostics


def build_baseline_slow_queue(
    *,
    context: dict[str, Any],
    cycle_ledger_path: str | None,
    out_dir: Path,
    measured_hashes: set[str] | None,
    gate: int | None,
    cap: int,
    resident_baseline: tuple[dict[str, Any], str] | None,
) -> tuple[str, dict[str, Any]]:
    """Write this turn's slow queue (``cap`` codes at most) and its composition record; returns (path, record)."""

    method = str(context["method"])
    if method not in SELECTION_IDS:
        raise ValueError(f"unknown baseline method {method!r}")
    run_dir = Path(str(context["run_dir"]))
    turn = int(context["turn"])
    seed_key = str(context["seed_key"])
    init_turns = int(context.get("init_random_turns") or 0)
    nominated = nominated_rows(run_dir, before_turn=turn)
    pool, counts = candidate_pool(
        run_dir=run_dir,
        cycle_ledger_path=cycle_ledger_path,
        nominated=nominated,
        measured_hashes=measured_hashes,
        gate=gate,
        reference=reference_identity(resident_baseline),
    )
    select_key = f"{seed_key}|uniform_select|turn={turn}"
    diagnostics: dict[str, Any] | None = None
    observation_count = None
    if method == "random" or turn < init_turns:
        rule = "uniform"
        channel = "baseline_random" if method == "random" else "baseline_bo_init"
        picks = _uniform_picks(pool, q=cap, seed=baseline_seed(select_key))
    else:
        rule = "bo"
        channel = "baseline_bo"
        select_key = f"{seed_key}|bo_select|turn={turn}"
        observations = measured_observations(run_dir, before_turn=turn)
        observation_count = len(observations)
        picks, diagnostics = _bo_picks(
            pool=pool,
            observations=observations,
            q=cap,
            seed=baseline_seed(select_key),
            state_dir=run_dir / "baseline_bo",
            turn=turn,
            blocks=int(context.get("blocks") or 1),
            config=context.get("bo") or None,
        )
    selected = []
    for row in picks:
        item = dict(row)
        item["selection_channels"] = [channel]
        selected.append(item)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "slow_queue.jsonl"
    out_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    composition = {
        "schema": "slow_queue_composition",
        "search_method": method,
        "selection_id": SELECTION_IDS[method],
        "rule": rule,
        "turn": turn,
        "cap": int(cap),
        "selected_count": len(selected),
        "unused_slots": max(0, int(cap) - len(selected)),
        "select_key": select_key,
        "pool": counts,
        "previously_nominated": len(nominated),
        "weight_gate": {"max_check_weight": None if gate is None else int(gate)},
        "observations": observation_count,
        "selected": [
            {
                "code_hash": str(row.get("code_hash") or "")[:12],
                "family": row.get("family"),
                "n": row.get("n"),
                "k": row.get("k"),
                "d_search": row.get("d_search"),
                "max_check_weight": row.get("max_check_weight"),
                "selection_channels": row.get("selection_channels"),
            }
            for row in selected
        ],
    }
    if diagnostics is not None:
        composition["bo"] = diagnostics
    (out_dir / "slow_queue_composition.json").write_text(
        json.dumps(composition, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return str(out_path), composition
