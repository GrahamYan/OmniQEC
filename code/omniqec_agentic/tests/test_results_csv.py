from __future__ import annotations

import csv
import json

from omniqec_agentic.config import DecoderName, ExactDistanceConfig, VerificationConfig
from omniqec_agentic.results_csv import RESULTS_CSV_COLUMNS
from omniqec_agentic.evaluation import evaluate_promotion_queue
from omniqec_agentic.verification.noise import NoiseSweepConfig


# Lin-Pryadko D6 [[24,8,3]] (arXiv:2306.16400 Table III) in exponent tuples.
LP_SPEC = {
    "family": "lp_nonabelian",
    "group": "D6",
    "A": [[[[0, 0], [4, 0]]]],
    "B": [[[[0, 0], [2, 1], [3, 0], [4, 0], [4, 1], [1, 0]]]],
}


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def test_slow_queue_automatically_writes_metrics_only_csv(tmp_path) -> None:
    queue = tmp_path / "queue.jsonl"
    queue.write_text(
        json.dumps({"program_id": "csv_fixture", "spec": LP_SPEC}) + "\n",
        encoding="utf-8",
    )
    noise = NoiseSweepConfig.si1000((1e-3,))
    verification = VerificationConfig(
        noise=noise,
        p_values=noise.values,
        decoders=(DecoderName.LSD,),
        shots=20,
        rounds=5,
        max_errors=20,
        circuit_timeout_s=0,
        exact_distance=ExactDistanceConfig(enabled=False, code_milp=False),
    )
    output = tmp_path / "slow"
    summary = evaluate_promotion_queue(
        queue_path=queue,
        out_dir=output,
        verification=verification,
        hardware="ion",
        include_baseline=False,
        workers=1,
    )
    csv_path = output / "slow_results.csv"
    fields, rows = _read_csv(csv_path)
    assert summary["artifact_paths"]["results_csv"] == str(csv_path)
    assert fields == list(RESULTS_CSV_COLUMNS)
    assert "spec_json" not in fields
    assert "n_checks_x" in fields
    assert "n_checks_z" in fields
    assert "total_budget" in fields
    assert len(rows) == 1
    assert rows[0]["program_id"] == "csv_fixture"
    assert rows[0]["sampling_source"] == "noisy_circuit"
    assert rows[0]["noise_profile"] == "si1000_modern_v1"
    assert rows[0]["family"] == "lp_nonabelian"
    # Pure-CSS fixture: per-basis proxy distances are populated (the old btlp
    # fixture was non-CSS and nulled them).
    assert rows[0]["d_x_proxy"] != ""
    assert rows[0]["d_z_proxy"] != ""
    assert rows[0]["shots_x"] == "20"
    assert rows[0]["shots_z"] == "20"
    assert rows[0]["rounds"] == "5"
    assert rows[0]["block_ler"] != ""
    assert rows[0]["per_logical_per_round"] != ""
    assert rows[0]["per_logical_block_ci_low"] != ""
    assert rows[0]["per_logical_block_ci_high"] != ""
    assert rows[0]["ler_definition"] == "paper_A4_A5_single_T"
    assert "combined_ler" not in fields
    assert "per_round_ler" not in fields
    assert "ci_low" not in fields
    assert "ci_high" not in fields
