"""Decoder qualification harness (03 §10, v3.6 docs/06 §1.3) — standalone CLI,
not in the graph.

Runs the versioned qualification set (docs/qualification/) through the SAME
evaluation pipeline as campaigns (two-basis SEC circuits, noisy-circuit
sampling, the subprocess job pool and its worker), one run per noise profile
and per-code shot budget, then

  acceptance         (v3.6, the reference decoder = the frozen telescoping
                      decoder, config.TelescopingDecoderConfig defaults): per-code
                      per-basis block-LER ceilings and plpr orderings declared
                      in the set file (BB144 X/Z <= 0.5%, Mitten200 <= 50%,
                      BB72 worse than BB144), plus a same-seed determinism
                      re-run through the same worker path;
  primary criterion  (candidate vs reference, when they differ) pairwise
                     agreement >= 0.9 on code pairs the REFERENCE separates
                     (disjoint per-logical-block CIs) at every p-level of a
                     profile (>= 2 levels when the profile has them);
  diagnostics        Spearman rho on log10(plpr) (global and per family),
                     per-family log-LER bias, measured throughput;
  extras             beam determinism check (02 §11.1) and the num_results=32
                     list-decoding throughput measurement (only when beam is
                     among the decoders).
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from omniqec_agentic.config import (
    BposdDecoderConfig,
    DecoderName,
    ExactDistanceConfig,
    TelescopingDecoderConfig,
    VerificationConfig,
)
from omniqec_agentic.evaluation import evaluate_promotion_queue
from omniqec_agentic.verification.noise import NoiseSweepConfig

QUALIFICATION_SCHEMA_VERSION = 2
DEFAULT_SET_PATH = Path(__file__).parent / "docs" / "qualification" / "qualification_set_v1.json"
# v3.6: the formal protocol is ONE fixed point (uniform_paper_v1, p=0.005);
# the acceptance harness runs exactly that point. si1000 keeps two levels for
# the historical cross-decoder agreement criterion.
DEFAULT_P_LEVELS: dict[str, tuple[float, ...]] = {
    "uniform": (0.005,),
    "si1000": (1.0e-3, 2.0e-3),
}
AGREEMENT_THRESHOLD = 0.9
# The reference decoder of telescoping_structural_v3_6 (docs/06 §1, user
# decision 2026-09-03) is the frozen telescoping decoder
# (config.TelescopingDecoderConfig defaults: telescoping_cpu_v1, S1/S2/S4 off,
# S3-A/B/C on the GARI system with the package tables, nc_policy fail).
# Qualification measures the same decoder the campaign runs rather than a
# tuning of its own.
TELESCOPING_REFERENCE = TelescopingDecoderConfig()
REFERENCE_DECODER = DecoderName.TELESCOPING.value
# Historical BPOSD reference (product_sum / 50 / osd_cs / 10); only used when
# bposd (a dev-only decoder since v3.6) is among the decoders.
BPOSD_REFERENCE = BposdDecoderConfig()
DEFAULT_SHOTS = 1000
DEFAULT_DETERMINISM_SHOTS = 200


def _noise_sweep(profile_key: str, values: tuple[float, ...]) -> NoiseSweepConfig:
    if profile_key == "uniform":
        return NoiseSweepConfig.uniform(values)
    if profile_key == "si1000":
        return NoiseSweepConfig.si1000(values)
    if profile_key == "helios":
        return NoiseSweepConfig.helios(values)
    raise ValueError(f"unknown profile key: {profile_key}")


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx <= 0 or dy <= 0:
        return None
    return num / (dx * dy)


def _pairwise_agreement(
    reference_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Agreement on code pairs the reference separates with disjoint CIs."""

    codes = sorted(set(reference_rows) & set(candidate_rows))
    separated = 0
    agreed = 0
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = reference_rows[codes[i]], reference_rows[codes[j]]
            ci_a, ci_b = a.get("per_logical_block_ci"), b.get("per_logical_block_ci")
            if not ci_a or not ci_b:
                continue
            if ci_a[1] < ci_b[0] or ci_b[1] < ci_a[0]:  # disjoint
                separated += 1
                ref_order = a["per_logical_per_round"] < b["per_logical_per_round"]
                ca = candidate_rows[codes[i]].get("per_logical_per_round")
                cb = candidate_rows[codes[j]].get("per_logical_per_round")
                if ca is not None and cb is not None and (ca < cb) == ref_order:
                    agreed += 1
    return {
        "separated_pairs": separated,
        "agreed_pairs": agreed,
        "agreement": (agreed / separated) if separated else None,
        "codes_compared": len(codes),
    }


def _rows_by_code(rows: list[dict[str, Any]], *, decoder: str, p: float) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("status") != "ok" or row.get("per_logical_per_round") is None:
            continue
        if str(row.get("decoder")) != decoder:
            continue
        if abs(float(row.get("noise_value") or 0.0) - float(p)) > 1e-15:
            continue
        out[str(row.get("program_id"))] = row
    return out


def _beam_determinism_and_throughput(
    spec: dict[str, Any],
    *,
    p: float,
    shots_determinism: int,
    shots_throughput: int,
    seed: int,
) -> dict[str, Any]:
    """Beam determinism check + num_results=1 vs 32 throughput (02 §11.1)."""

    from omniqec_agentic.config import DEFAULT_SEC_POLICY, BeamDecoderConfig
    from omniqec_agentic.construction import build_compiled_code
    from omniqec_agentic.verification.circuit import build_memory_experiment_pair
    from omniqec_agentic.verification.decoders import BeamObservableDecoder
    from omniqec_agentic.verification.dem import dem_to_matrices

    compiled = build_compiled_code(spec)
    pair = build_memory_experiment_pair(
        np.asarray(compiled.hx), np.asarray(compiled.hz), p=float(p), rounds=None,
        sector_sizes=compiled.sector_sizes,
        sec_policy=DEFAULT_SEC_POLICY,
        layer_blocks=compiled.layer_blocks,
    )
    matrices = dem_to_matrices(pair.x_memory.detector_error_model)
    sampler = pair.x_memory.circuit.compile_detector_sampler(seed=int(seed))
    n_shots = max(int(shots_determinism), int(shots_throughput))
    dets, obs = sampler.sample(n_shots, separate_observables=True)
    dets8 = np.asarray(dets, dtype=np.uint8)
    obs8 = np.asarray(obs, dtype=np.uint8)

    det_slice = dets8[: int(shots_determinism)]
    pred_a = BeamObservableDecoder(matrices).decode_observables_batch(det_slice)
    pred_b = BeamObservableDecoder(matrices).decode_observables_batch(det_slice)
    deterministic = bool(np.array_equal(pred_a, pred_b))

    result: dict[str, Any] = {
        "schema": "beam_determinism_and_throughput",
        "shots_determinism": int(shots_determinism),
        "deterministic": deterministic,
        "throughput": {},
    }
    thr_slice = dets8[: int(shots_throughput)]
    thr_obs = obs8[: int(shots_throughput)]
    for num_results in (1, 32):
        decoder = BeamObservableDecoder(
            matrices, BeamDecoderConfig(num_results=num_results)
        )
        start = time.perf_counter()
        pred = decoder.decode_observables_batch(thr_slice)
        elapsed = time.perf_counter() - start
        failures = int(np.any(pred != thr_obs, axis=1).sum())
        result["throughput"][f"num_results_{num_results}"] = {
            "shots": int(shots_throughput),
            "runtime_s": elapsed,
            "shots_per_s": (int(shots_throughput) / elapsed) if elapsed > 0 else None,
            "failures": failures,
            "block_ler": failures / int(shots_throughput),
        }
    return result


def _qualification_verification(
    noise: NoiseSweepConfig, *, decoder_names: tuple[str, ...], shots: int
) -> VerificationConfig:
    """The campaign's VerificationConfig with the reference decoder configs
    pinned: the frozen telescoping decoder (v3.6) and, when requested, the historical
    BPOSD tuple."""

    verification = VerificationConfig(
        noise=noise,
        p_values=noise.values,
        decoders=tuple(DecoderName(name) for name in decoder_names),
        shots=int(shots),
        max_errors=None,
        circuit_timeout_s=3600,
        exact_distance=ExactDistanceConfig(enabled=False, code_milp=False),
    )
    decoder_build = dataclasses.replace(verification.decoder_build, telescoping=TELESCOPING_REFERENCE)
    if "bposd" in decoder_names:
        decoder_build = dataclasses.replace(decoder_build, bposd=BPOSD_REFERENCE)
    return dataclasses.replace(verification, decoder_build=decoder_build)


def _run_queue(
    codes: list[dict[str, Any]],
    *,
    out_dir: Path,
    queue_name: str,
    verification: VerificationConfig,
    workers: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One evaluate_promotion_queue run (the campaign's worker path) over
    ``codes``; returns the summary and its ranking rows."""

    queue_path = out_dir / f"queue_{queue_name}.jsonl"
    queue_path.write_text(
        "\n".join(
            json.dumps({"program_id": c["program_id"], "spec": c["spec"]}) for c in codes
        )
        + "\n",
        encoding="utf-8",
    )
    summary = evaluate_promotion_queue(
        queue_path=queue_path,
        out_dir=out_dir / f"run_{queue_name}",
        verification=verification,
        include_baseline=False,
        workers=int(workers),
        seed=int(seed),
        stage="search",
    )
    rankings = json.loads(
        Path(summary["artifact_paths"]["rankings"]).read_text(encoding="utf-8")
    )["rankings"]
    return summary, rankings


def _code_shots(code: dict[str, Any], default: int) -> int:
    """Per-code shot budget (v3.6 set rows may carry ``shots``: Mitten rows
    run 100 shots/basis, BB rows the harness default)."""

    value = code.get("shots")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and int(value) > 0:
        return int(value)
    return int(default)


def _acceptance_report(
    codes: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    *,
    plpr_order: list[list[str]],
    p: float,
    decoder: str,
) -> dict[str, Any]:
    """v3.6 acceptance (docs/06 §1.3 (a)-(c)): per-code basis-LER ceilings
    (``acceptance.max_basis_ler`` in the set row) and the set's plpr orderings
    (each pair [better, worse]: plpr(better) < plpr(worse))."""

    by_code = _rows_by_code(rows, decoder=decoder, p=p)
    checks: list[dict[str, Any]] = []
    for code in codes:
        program_id = str(code["program_id"])
        row = by_code.get(program_id)
        entry: dict[str, Any] = {
            "program_id": program_id,
            "measured": row is not None,
            "ler_x": None if row is None else row.get("ler_x"),
            "ler_z": None if row is None else row.get("ler_z"),
            "shots_x": None if row is None else row.get("shots_x"),
            "shots_z": None if row is None else row.get("shots_z"),
            "per_logical_per_round": None if row is None else row.get("per_logical_per_round"),
            "status": None if row is None else row.get("status"),
        }
        acceptance = code.get("acceptance") if isinstance(code.get("acceptance"), dict) else {}
        ceiling = acceptance.get("max_basis_ler")
        if ceiling is not None:
            entry["max_basis_ler"] = float(ceiling)
            lers = [entry["ler_x"], entry["ler_z"]]
            entry["passed"] = row is not None and all(
                value is not None and float(value) <= float(ceiling) for value in lers
            )
        else:
            entry["passed"] = None
        checks.append(entry)
    orderings: list[dict[str, Any]] = []
    for pair in plpr_order:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        better, worse = str(pair[0]), str(pair[1])
        left = by_code.get(better)
        right = by_code.get(worse)
        ok = (
            left is not None
            and right is not None
            and left.get("per_logical_per_round") is not None
            and right.get("per_logical_per_round") is not None
            and float(left["per_logical_per_round"]) < float(right["per_logical_per_round"])
        )
        orderings.append(
            {
                "better": better,
                "worse": worse,
                "plpr_better": None if left is None else left.get("per_logical_per_round"),
                "plpr_worse": None if right is None else right.get("per_logical_per_round"),
                "passed": bool(ok),
            }
        )
    gated = [c["passed"] for c in checks if c["passed"] is not None]
    return {
        "decoder": decoder,
        "p": float(p),
        "codes": checks,
        "plpr_order": orderings,
        "passed": all(gated) and all(item["passed"] for item in orderings),
    }


def _determinism_report(
    codes: list[dict[str, Any]],
    *,
    out_dir: Path,
    profile_key: str,
    values: tuple[float, ...],
    decoder_names: tuple[str, ...],
    shots: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """v3.6 acceptance (d): two same-seed runs through the same worker path
    must agree shot for shot; the pool exposes per-basis failure counts and
    per-shard statuses, which are compared here for every code and decoder."""

    noise = _noise_sweep(profile_key, values)
    verification = _qualification_verification(noise, decoder_names=decoder_names, shots=shots)
    runs = []
    for label in ("a", "b"):
        _summary, rows = _run_queue(
            codes,
            out_dir=out_dir,
            queue_name=f"determinism_{label}",
            verification=verification,
            workers=workers,
            seed=seed,
        )
        runs.append(
            {
                (str(row.get("program_id")), str(row.get("decoder")), float(row.get("noise_value") or 0.0)): (
                    row.get("failures_x"),
                    row.get("failures_z"),
                    row.get("shots_x"),
                    row.get("shots_z"),
                    row.get("status"),
                )
                for row in rows
            }
        )
    keys = sorted(set(runs[0]) | set(runs[1]))
    mismatches = [
        {"program_id": key[0], "decoder": key[1], "p": key[2], "a": runs[0].get(key), "b": runs[1].get(key)}
        for key in keys
        if runs[0].get(key) != runs[1].get(key)
    ]
    return {
        "schema": "same_seed_determinism",
        "shots": int(shots),
        "seed": int(seed),
        "codes": [str(c["program_id"]) for c in codes],
        "compared": len(keys),
        "deterministic": not mismatches,
        "mismatches": mismatches,
    }


def run_decoder_qualification(
    *,
    set_path: str | Path | None = None,
    out_dir: str | Path,
    candidates: tuple[str, ...] = (REFERENCE_DECODER,),
    references: tuple[str, ...] = (REFERENCE_DECODER,),
    profiles: tuple[str, ...] = ("uniform",),
    p_levels: dict[str, tuple[float, ...]] | None = None,
    shots: int = DEFAULT_SHOTS,
    seed: int = 0,
    workers: int = 0,
    limit: int | None = None,
    determinism_check: bool = True,
    beam32_throughput: bool = True,
    determinism_shots: int = DEFAULT_DETERMINISM_SHOTS,
) -> dict[str, Any]:
    set_file = Path(set_path) if set_path is not None else DEFAULT_SET_PATH
    set_data = json.loads(set_file.read_text(encoding="utf-8"))
    codes = set_data.get("codes") or []
    if limit is not None:
        codes = codes[: max(0, int(limit))]
    if not codes:
        raise ValueError(f"qualification set is empty: {set_file}")
    p_levels = dict(p_levels or DEFAULT_P_LEVELS)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    decoder_names = tuple(dict.fromkeys(list(references) + list(candidates)))
    reference = references[0]
    start = time.perf_counter()
    profile_results: dict[str, Any] = {}
    for profile_key in profiles:
        values = tuple(float(v) for v in p_levels.get(profile_key, ()))
        if not values:
            raise ValueError(f"profile '{profile_key}' needs >= 1 p-level")
        noise = _noise_sweep(profile_key, values)
        # One pool run per distinct shot budget (v3.6: Mitten rows carry
        # their own smaller budget), all through the campaign's worker path.
        groups: dict[int, list[dict[str, Any]]] = {}
        for code in codes:
            groups.setdefault(_code_shots(code, shots), []).append(code)
        rows: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []
        for group_shots, group_codes in sorted(groups.items(), key=lambda item: -item[0]):
            verification = _qualification_verification(
                noise, decoder_names=decoder_names, shots=group_shots
            )
            name = profile_key if len(groups) == 1 else f"{profile_key}_s{group_shots}"
            summary, group_rows = _run_queue(
                group_codes,
                out_dir=out_path,
                queue_name=name,
                verification=verification,
                workers=workers,
                seed=seed,
            )
            summaries.append(summary)
            rows.extend(group_rows)
        profile_results[profile_key] = {
            "summaries": summaries,
            "rows": rows,
            "p_values": values,
            "shots_by_group": {str(k): [c["program_id"] for c in v] for k, v in groups.items()},
        }

    # ---- statistics ----------------------------------------------------
    family_of = {c["program_id"]: c.get("family") for c in codes}
    candidates_report: dict[str, Any] = {}
    for candidate in candidates:
        if candidate == reference:
            continue
        per_profile: dict[str, Any] = {}
        for profile_key, bundle in profile_results.items():
            levels = []
            for p in bundle["p_values"]:
                ref_rows = _rows_by_code(bundle["rows"], decoder=reference, p=p)
                cand_rows = _rows_by_code(bundle["rows"], decoder=candidate, p=p)
                agreement = _pairwise_agreement(ref_rows, cand_rows)
                shared = sorted(set(ref_rows) & set(cand_rows))
                log_ref = [math.log10(ref_rows[c]["per_logical_per_round"]) for c in shared]
                log_cand = [math.log10(cand_rows[c]["per_logical_per_round"]) for c in shared]
                rho = _spearman(log_ref, log_cand)
                family_rho: dict[str, float | None] = {}
                family_bias: dict[str, float] = {}
                for family in sorted({family_of.get(c) for c in shared} - {None}):
                    idx = [i for i, c in enumerate(shared) if family_of.get(c) == family]
                    family_rho[family] = _spearman(
                        [log_ref[i] for i in idx], [log_cand[i] for i in idx]
                    )
                    if idx:
                        family_bias[family] = sum(
                            log_cand[i] - log_ref[i] for i in idx
                        ) / len(idx)
                levels.append(
                    {
                        "p": p,
                        **agreement,
                        "spearman_log_plpr": rho,
                        "family_spearman": family_rho,
                        "family_log10_bias": family_bias,
                        "criterion_met": (
                            agreement["agreement"] is not None
                            and agreement["agreement"] >= AGREEMENT_THRESHOLD
                        ),
                    }
                )
            # >= 2 levels when the profile has them (03 §10); the fixed-p
            # v3.6 protocol has exactly one, which must then hold.
            needed = min(2, len(levels))
            met = sum(1 for lv in levels if lv["criterion_met"])
            per_profile[profile_key] = {
                "levels": levels,
                "levels_meeting_criterion": met,
                "levels_required": needed,
                "profile_pass": met >= needed and needed > 0,
            }
        candidates_report[candidate] = {
            "reference": reference,
            "profiles": per_profile,
            "qualified": all(v["profile_pass"] for v in per_profile.values()),
        }

    # ---- v3.6 acceptance (reference decoder, uniform anchor point) ----------
    acceptance = None
    if "uniform" in profile_results:
        bundle = profile_results["uniform"]
        acceptance = _acceptance_report(
            codes,
            bundle["rows"],
            plpr_order=list(set_data.get("plpr_order") or []),
            p=float(bundle["p_values"][0]),
            decoder=reference,
        )

    # throughput per decoder from measured rows (decode+sampling runtime)
    throughput: dict[str, Any] = {}
    for decoder in decoder_names:
        total_shots = 0
        total_runtime = 0.0
        for bundle in profile_results.values():
            for row in bundle["rows"]:
                if str(row.get("decoder")) != decoder or row.get("status") != "ok":
                    continue
                total_shots += int(row.get("total_shots") or 0)
                total_runtime += float(row.get("runtime_s") or 0.0)
        throughput[decoder] = {
            "total_shots": total_shots,
            "total_runtime_s": total_runtime,
            "shots_per_s": (total_shots / total_runtime) if total_runtime > 0 else None,
        }

    # ---- same-seed determinism through the worker path (v3.6 (d)) ----------
    determinism = None
    if determinism_check:
        anchors = [c for c in codes if c.get("anchor")] or codes[:1]
        profile_key = "uniform" if "uniform" in profile_results else profiles[0]
        determinism = _determinism_report(
            anchors,
            out_dir=out_path,
            profile_key=profile_key,
            values=tuple(float(v) for v in p_levels.get(profile_key, (0.005,))),
            decoder_names=decoder_names,
            shots=max(1, min(int(determinism_shots), int(shots))),
            workers=workers,
            seed=int(seed) + 7,
        )

    beam_extras = None
    if (determinism_check or beam32_throughput) and "beam" in decoder_names:
        anchor = next(
            (c for c in codes if c.get("anchor")), codes[0]
        )
        beam_extras = _beam_determinism_and_throughput(
            anchor["spec"],
            p=float(p_levels.get("uniform", (0.003,))[0]),
            shots_determinism=1000 if determinism_check else 0,
            shots_throughput=200 if beam32_throughput else 0,
            seed=int(seed) + 999,
        )
        beam_extras["code"] = anchor["program_id"]

    report = {
        "schema": "decoder_qualification_report",
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "set_path": str(set_file),
        "set_version": set_data.get("version"),
        "code_count": len(codes),
        "shots": int(shots),
        "seed": int(seed),
        "references": list(references),
        "candidates": list(candidates),
        "reference_decoder_config": TELESCOPING_REFERENCE.to_dict()
        if reference == REFERENCE_DECODER
        else (BPOSD_REFERENCE.to_dict() if reference == "bposd" else None),
        "profiles": {k: list(v["p_values"]) for k, v in profile_results.items()},
        "shots_by_group": {k: v["shots_by_group"] for k, v in profile_results.items()},
        "agreement_threshold": AGREEMENT_THRESHOLD,
        "candidates_report": candidates_report,
        "acceptance": acceptance,
        "determinism": determinism,
        "throughput": throughput,
        "beam_extras": beam_extras,
        "run_summaries": {
            k: [{"status": s["status"], "out_dir": s["out_dir"]} for s in v["summaries"]]
            for k, v in profile_results.items()
        },
        "runtime_s": time.perf_counter() - start,
    }
    report_path = out_path / "qualification_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_path / "qualification_report.md").write_text(
        _report_markdown(report), encoding="utf-8"
    )
    report["artifact_paths"] = {
        "report": str(report_path),
        "report_md": str(out_path / "qualification_report.md"),
    }
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Decoder qualification report",
        "",
        f"- set: `{report['set_path']}` (version {report['set_version']}, "
        f"{report['code_count']} codes)",
        f"- shots/basis: {report['shots']} (per-code overrides: {report.get('shots_by_group')}), "
        f"seed: {report['seed']}",
        f"- reference: {', '.join(report['references'])}; "
        f"candidates: {', '.join(report['candidates'])}",
        f"- agreement threshold: {report['agreement_threshold']} on reference-CI-"
        "separated pairs, every p-level of a profile (>= 2 when available)",
        "",
    ]
    acceptance = report.get("acceptance")
    if acceptance:
        lines.append(
            f"## Acceptance ({acceptance['decoder']} @ p={acceptance['p']:g}) — "
            f"{'PASS' if acceptance['passed'] else 'FAIL'}"
        )
        lines.append("")
        lines.append("| code | status | LER X | LER Z | shots X/Z | plpr | ceiling | pass |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for entry in acceptance["codes"]:
            lines.append(
                f"| {entry['program_id']} | {_fmt(entry['status'])} | {_fmt(entry['ler_x'])} "
                f"| {_fmt(entry['ler_z'])} | {_fmt(entry['shots_x'])}/{_fmt(entry['shots_z'])} "
                f"| {_fmt(entry['per_logical_per_round'])} | {_fmt(entry.get('max_basis_ler'))} "
                f"| {'-' if entry['passed'] is None else ('yes' if entry['passed'] else 'no')} |"
            )
        for item in acceptance["plpr_order"]:
            lines.append(
                f"- order {item['better']} < {item['worse']}: "
                f"{_fmt(item['plpr_better'])} vs {_fmt(item['plpr_worse'])} -> "
                f"{'yes' if item['passed'] else 'no'}"
            )
        lines.append("")
    determinism = report.get("determinism")
    if determinism:
        lines.append(
            f"## Same-seed determinism ({determinism['shots']} shots, seed {determinism['seed']}, "
            f"codes {', '.join(determinism['codes'])}): "
            f"{'PASS' if determinism['deterministic'] else 'FAIL'}"
        )
        for item in determinism.get("mismatches") or []:
            lines.append(f"- mismatch {item}")
        lines.append("")
    for candidate, cand in (report.get("candidates_report") or {}).items():
        lines.append(f"## {candidate} vs {cand['reference']} — "
                     f"{'QUALIFIED' if cand['qualified'] else 'NOT QUALIFIED'}")
        lines.append("")
        lines.append("| profile | p | separated | agreed | agreement | rho(log plpr) | pass |")
        lines.append("|---|---|---|---|---|---|---|")
        for profile_key, prof in cand["profiles"].items():
            for level in prof["levels"]:
                agreement = level["agreement"]
                rho = level["spearman_log_plpr"]
                lines.append(
                    f"| {profile_key} | {level['p']:g} | {level['separated_pairs']} "
                    f"| {level['agreed_pairs']} "
                    f"| {'-' if agreement is None else f'{agreement:.3f}'} "
                    f"| {'-' if rho is None else f'{rho:.3f}'} "
                    f"| {'yes' if level['criterion_met'] else 'no'} |"
                )
        lines.append("")
        for profile_key, prof in cand["profiles"].items():
            biases = prof["levels"][0].get("family_log10_bias") or {}
            if biases:
                pretty = ", ".join(f"{fam}: {bias:+.3f}" for fam, bias in biases.items())
                lines.append(f"- per-family log10 bias ({profile_key}, first p): {pretty}")
        lines.append("")
    lines.append("## Throughput (decode incl. sampling, measured rows)")
    lines.append("")
    lines.append("| decoder | total shots | runtime s | shots/s |")
    lines.append("|---|---|---|---|")
    for decoder, thr in (report.get("throughput") or {}).items():
        sps = thr["shots_per_s"]
        lines.append(
            f"| {decoder} | {thr['total_shots']} | {thr['total_runtime_s']:.1f} "
            f"| {'-' if sps is None else f'{sps:.1f}'} |"
        )
    extras = report.get("beam_extras")
    if extras:
        lines += [
            "",
            "## Beam extras",
            "",
            f"- determinism ({extras['shots_determinism']} shots, code "
            f"{extras['code']}): {'PASS' if extras['deterministic'] else 'FAIL'}",
        ]
        for key, thr in (extras.get("throughput") or {}).items():
            lines.append(
                f"- {key}: {thr['shots_per_s']:.1f} shots/s, "
                f"block LER {thr['block_ler']:.4f} ({thr['shots']} shots)"
            )
    lines.append("")
    return "\n".join(lines)
