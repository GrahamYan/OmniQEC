from __future__ import annotations

import math

import pytest

from omniqec_agentic.verification.ler import combine_basis_ler_results
from omniqec_agentic.evaluation.pool import run_worker_request as _run_worker_request
from omniqec_agentic.evaluation.sampling import merge_shard_results as _merge_shard_results
from omniqec_agentic.analyze import _beats_baseline_ler_count


def _paper_a5(block_ler: float, *, k: int, rounds: int) -> float:
    return 1.0 - (1.0 - block_ler) ** (1.0 / (k * rounds))


@pytest.mark.parametrize("k", [1, 2, 4])
def test_a4_a5_exact_for_fixed_five_rounds(k: int) -> None:
    px = 0.031
    pz = 0.017
    expected_block = 1.0 - (1.0 - px) * (1.0 - pz)
    result = combine_basis_ler_results(
        {
            "X": {"logical_error_rate": px},
            "Z": {"logical_error_rate": pz},
        },
        rounds=5,
        k=k,
    )

    assert result["block_ler"] == pytest.approx(expected_block)
    assert result["per_logical_per_round"] == pytest.approx(
        _paper_a5(expected_block, k=k, rounds=5)
    )
    assert result["ler_definition"] == "paper_A4_A5_single_T"
    assert "logical_error_rate" not in result
    assert "per_round_logical_error_rate" not in result


def test_low_error_limit_is_block_over_k_rounds() -> None:
    block_ler = 1e-8
    result = combine_basis_ler_results(
        {"X": {"logical_error_rate": block_ler}},
        rounds=5,
        k=4,
    )

    approximation = block_ler / (4 * 5)
    assert result["per_logical_per_round"] == pytest.approx(
        approximation,
        rel=1e-6,
        abs=1e-16,
    )


def test_wilson_basis_bounds_are_combined_then_transformed_by_e5() -> None:
    result = combine_basis_ler_results(
        {
            "X": {"shots": 100_000, "failures": 120, "logical_error_rate": 0.0012},
            "Z": {"shots": 100_000, "failures": 80, "logical_error_rate": 0.0008},
        },
        rounds=5,
        k=2,
    )

    low, high = result["per_logical_block_ci"]
    point = 1.0 - (1.0 - result["block_ler"]) ** 0.5
    assert 0.0 <= low <= point <= high <= 1.0
    plpr_low, plpr_high = result["per_logical_per_round_ci"]
    plpr_point = result["per_logical_per_round"]
    assert 0.0 <= plpr_low <= plpr_point <= plpr_high <= 1.0
    assert result["total_shots"] == 200_000
    assert result["total_failures"] == 200


def test_high_block_error_is_still_defined_under_a5() -> None:
    # A5's compound root is total on [0, 1]: no fidelity-domain failure mode.
    result = combine_basis_ler_results(
        {"X": {"logical_error_rate": 0.75}},
        rounds=5,
        k=1,
    )

    assert result["block_ler"] == pytest.approx(0.75)
    assert result["per_logical_per_round"] == pytest.approx(1.0 - 0.25 ** (1.0 / 5.0))
    assert "failure_reason" not in result


def test_zero_block_error_has_zero_per_logical_per_round() -> None:
    result = combine_basis_ler_results(
        {"X": {"logical_error_rate": 0.0}},
        rounds=5,
        k=2,
    )

    assert math.isclose(result["block_ler"], 0.0)
    assert math.isclose(result["per_logical_per_round"], 0.0)


@pytest.mark.parametrize("timeout_s", [None, 0.0])
def test_parallel_worker_nonpositive_timeout_means_no_timeout(
    timeout_s,
    monkeypatch,
) -> None:
    observed = {}

    class _Process:
        returncode = 0

        def communicate(self, payload, *, timeout):
            observed["payload"] = payload
            observed["timeout"] = timeout
            return '{"status":"ok"}', ""

    monkeypatch.setattr(
        "omniqec_agentic.evaluation.pool.subprocess.Popen",
        lambda *args, **kwargs: _Process(),
    )
    parsed, metadata = _run_worker_request(
        {"mode": "fixture"},
        timeout_s=timeout_s,
    )

    assert observed["timeout"] is None
    assert parsed == {"status": "ok"}
    assert metadata["timeout"] is False


def test_reference_win_requires_separated_confidence_intervals() -> None:
    reference = [{
        "status": "ok",
        "per_logical_per_round": 0.02,
        "per_logical_block_ci": [0.09, 0.11],
    }]
    candidates = [
        {
            "status": "ok",
            "per_logical_per_round": 0.01,
            "per_logical_block_ci": [0.05, 0.08],
        },
        {
            "status": "ok",
            "per_logical_per_round": 0.005,
            "per_logical_block_ci": [0.08, 0.10],
        },
        {
            "status": "ok",
            "per_logical_per_round": 0.001,
        },
    ]

    assert _beats_baseline_ler_count(candidates, reference) == 1


def test_shard_merge_sums_requested_and_actual_shots() -> None:
    shard = lambda shots, failures: {
        "status": "ok",
        "shots": shots,
        "failures": failures,
        "runtime_s": 0.1,
        "metrics": {
            "requested_shots": shots,
            "actual_shots": shots,
            "sampling_complete": True,
            "stopped_by_max_errors": False,
        },
    }

    merged = _merge_shard_results(
        [shard(5_000, 8), shard(5_000, 7), shard(5_000, 9), shard(5_000, 6)],
        decoder="lsd",
    )

    assert merged["shots"] == 20_000
    assert merged["failures"] == 30
    assert merged["metrics"]["requested_shots"] == 20_000
    assert merged["metrics"]["actual_shots"] == 20_000
    assert merged["metrics"]["sampling_complete"] is True
    assert merged["metrics"]["stopped_by_max_errors"] is False
