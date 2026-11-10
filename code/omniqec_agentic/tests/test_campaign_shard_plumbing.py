from __future__ import annotations

from types import SimpleNamespace

from omniqec_agentic.config import RacingConfig, VerificationConfig


def test_execute_plan_passes_frozen_shards_to_search_slow_stage(
    monkeypatch, tmp_path
) -> None:
    import omniqec_agentic.campaign as campaign
    import omniqec_agentic.workflow.graph as graph

    captured: dict[str, object] = {}

    def fake_cycle(*args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(status="ok", artifact_paths={})

    monkeypatch.setattr(campaign, "run_agentic_cycle", fake_cycle)
    monkeypatch.setattr(graph, "_audit_turn_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        graph,
        "_verification_from_protocol",
        lambda *args, **kwargs: VerificationConfig(),
    )
    state = {
        "run_id": "campaign_b144",
        "run_dir": str(tmp_path),
        "turn": 0,
        "plan": {
            "provider": "dry_run",
            "generations": 1,
            "proposal_count": 1,
            "rng_seeds_per_program": 1,
        },
        "protocol": {
            "proposal_count": 1,
            "generations_per_turn": 1,
            "feedback_mode": "both",
            "racing": RacingConfig().to_dict(),
            "slow_workers": 127,
            "shot_shards": 16,
            "strict_turns": False,
        },
        "spec": {
            "qubit_budget": 144,
            "families": ["bb"],
        },
        "structural_proxy_elites": [],
        "unmeasurable_hashes": [],
        "baseline_summary": {},
        "budget": {},
    }

    graph.execute_plan_node(state)

    assert captured["slow_workers"] == 127
    assert captured["slow_shot_shards"] == 16


def test_final_slow_uses_the_frozen_final_shards(monkeypatch, tmp_path) -> None:
    """v3.4: the final stage partitions its 200k shots into 128 shards (its
    own protocol key), while the search stage keeps 16."""

    import omniqec_agentic.workflow.graph as graph

    captured: dict[str, object] = {}

    def fake_evaluate(**kwargs):
        captured.update(kwargs)
        return {"status": "ok", "artifact_paths": {}}

    monkeypatch.setattr(graph, "evaluate_promotion_queue", fake_evaluate)
    monkeypatch.setattr(graph, "_validate_all_turn_audits", lambda state: {})
    monkeypatch.setattr(
        graph,
        "_verification_from_protocol",
        lambda *args, **kwargs: VerificationConfig(),
    )
    monkeypatch.setattr(
        graph,
        "_final_entries",
        lambda state, structural: (
            [
                {
                    "program_id": "candidate",
                    "spec": {"family": "bb", "ell": 6, "m": 1},
                    "code_hash": "abc",
                    "selection_channels": ["structural_high_k"],
                }
            ],
            {"entries": 1},
        ),
    )
    monkeypatch.setattr(
        graph,
        "_audit_final_measurements",
        lambda state, runs: {"enabled": False, "status": "complete", "errors": []},
    )
    state = {
        "run_dir": str(tmp_path),
        "protocol": {
            "verification": VerificationConfig().to_dict(),
            "rounds_policy": {
                "final": "max_Tc_d",
                "final_rounds": 12,
                "final_rng_namespace": "final_confirmation_v1",
            },
            "final_p_values": [0.005],
            "final_shots": 40_000,
            "slow_workers": 127,
            "shot_shards": 16,
            "final_shot_shards": 128,
            "strict_turns": False,
        },
        "spec": {"qubit_budget": 144, "families": ["bb"]},
        "baseline_summary": {},
    }

    result = graph.final_slow_node(state)

    assert result["final_slow"]["status"] == "ok"
    assert captured["workers"] == 127
    assert captured["shot_shards"] == 128

