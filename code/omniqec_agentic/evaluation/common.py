"""Shared plumbing for the evaluation package (schema version, small helpers,
protocol deserialization). Mechanically extracted from slow.py (03 §7)."""

from __future__ import annotations

from typing import Any

from omniqec_agentic.config import (
    BeamDecoderConfig,
    BposdDecoderConfig,
    TelescopingDecoderConfig,
    DecoderBuildConfig,
    DecoderName,
    ExactDistanceConfig,
    HardwareTarget,
    LsdDecoderConfig,
    RelayBpDecoderConfig,
    SecPolicyConfig,
    StructuralSearchConfig,
    VerificationConfig,
)
from omniqec_agentic.sandbox import run_construct_source
from omniqec_agentic.verification.noise import NoiseSweepConfig

SLOW_SCHEMA_VERSION = 7


def decoder_name(decoder: DecoderName | str) -> DecoderName:
    return decoder if isinstance(decoder, DecoderName) else DecoderName(str(decoder))


def noise_result_fields(noise_point: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "noise_point": noise_point.to_dict(),
        "noise_axis": noise_point.axis,
        "noise_value": float(noise_point.value),
    }
    data[noise_point.axis] = float(noise_point.value)
    return data


def hardware_value(hardware: HardwareTarget | str | None) -> str | None:
    if hardware is None:
        return None
    return hardware.value if isinstance(hardware, HardwareTarget) else str(hardware)


def safe_file_stem(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:120] or "candidate"


def tail(text: str | None, *, limit: int = 4000) -> str:
    value = text or ""
    return value[-limit:]


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def resolve_deformation_mode(
    explicit: str | None,
    record: dict[str, Any] | None = None,
) -> str | None:
    value = explicit
    if value is None and isinstance(record, dict):
        value = record.get("deformation_mode") or record.get("deformation")
    if value is None:
        return None
    normalized = str(value)
    if normalized not in {"none", "sector_h"}:
        raise ValueError(f"unsupported deformation_mode: {normalized}")
    return normalized


def record_to_slow_payload_input(record: dict[str, Any], *, program_id: str) -> dict[str, Any]:
    spec = record.get("spec")
    if isinstance(spec, dict):
        return {"spec": spec}
    source = record.get("construct_source")
    if isinstance(source, str) and source.strip():
        try:
            constructed = run_construct_source(
                source,
                program_id=program_id,
                seed=int(record.get("rng_seed") or 0),
            ).constructed
        except Exception as exc:
            return {"error": f"construct_source:{exc}"}
        return {"spec": constructed.spec}
    return {"error": "promotion record has neither spec nor construct_source"}


def verification_from_dict(data: dict[str, Any] | None) -> VerificationConfig:
    raw = data or {}
    legacy_timeout = float(
        raw.get("circuit_timeout_s", VerificationConfig.circuit_timeout_s)
    )

    def typed_timeout(name: str, default: float) -> float:
        # Pre-v3.3 artifacts serialized only circuit_timeout_s. Preserve their
        # exact execution semantics, while an entirely absent config receives
        # the new typed v3.3 defaults.
        if name in raw:
            return float(raw[name])
        if "circuit_timeout_s" in raw:
            return legacy_timeout
        return float(default)

    legacy_values = tuple(float(p) for p in raw.get("p_values", VerificationConfig.p_values))
    # VerificationConfig.to_dict always writes ``noise``; an artifact without
    # it predates the noise-profile architecture and was lowered with the
    # idle-noise uniform rule that is now ``uniform_legacy_v1``. Read it back
    # as what it measured (the pre-v3.5 meaning of this fallback), not as the
    # current v3.5 paper profile -- a fresh config gets the paper default from
    # the dataclass, never from this branch.
    noise = (
        NoiseSweepConfig.from_dict(raw.get("noise"))
        if isinstance(raw.get("noise"), dict)
        else NoiseSweepConfig.uniform_legacy(legacy_values)
    )
    decoder_build_raw = raw.get("decoder_build") if isinstance(raw.get("decoder_build"), dict) else {}
    exact_distance_raw = raw.get("exact_distance") if isinstance(raw.get("exact_distance"), dict) else {}
    beam_raw = decoder_build_raw.get("beam") if isinstance(decoder_build_raw.get("beam"), dict) else {}
    lsd_raw = decoder_build_raw.get("lsd") if isinstance(decoder_build_raw.get("lsd"), dict) else {}
    bposd_raw = decoder_build_raw.get("bposd") if isinstance(decoder_build_raw.get("bposd"), dict) else {}
    relaybp_raw = decoder_build_raw.get("relaybp") if isinstance(decoder_build_raw.get("relaybp"), dict) else {}
    relaybp_kwargs = dict(relaybp_raw)
    if "gamma_dist_interval" in relaybp_kwargs:
        relaybp_kwargs["gamma_dist_interval"] = tuple(relaybp_kwargs["gamma_dist_interval"])
    # v3.6: the telescoping block; a pre-v3.6 artifact has none and receives
    # the dataclass defaults (its decoders list names bposd, so it never runs).
    telescoping_raw = (
        decoder_build_raw.get("telescoping")
        if isinstance(decoder_build_raw.get("telescoping"), dict)
        else None
    )
    structural_raw = dict(raw.get("structural") or {})
    if "proxy_seeds" in structural_raw:
        structural_raw["proxy_seeds"] = tuple(int(s) for s in structural_raw["proxy_seeds"])
    return VerificationConfig(
        proxy_iters=int(raw.get("proxy_iters", VerificationConfig.proxy_iters)),
        proxy_n_min=raw.get("proxy_n_min"),
        proxy_n_max=raw.get("proxy_n_max"),
        decoders=tuple(DecoderName(item) for item in raw.get("decoders", [decoder.value for decoder in VerificationConfig.decoders])),
        p_values=tuple(noise.values),
        noise=noise,
        shots=int(raw.get("shots", VerificationConfig.shots)),
        rounds=raw.get("rounds"),
        bases=tuple(raw.get("bases", VerificationConfig.bases)),
        max_errors=raw.get("max_errors"),
        promote_top=int(raw.get("promote_top", VerificationConfig.promote_top)),
        circuit_timeout_s=legacy_timeout,
        build_timeout_s=typed_timeout(
            "build_timeout_s", VerificationConfig.build_timeout_s
        ),
        decode_shard_timeout_s=typed_timeout(
            "decode_shard_timeout_s", VerificationConfig.decode_shard_timeout_s
        ),
        # v3.4 fields: pre-v3.4 artifacts lack them and receive the dataclass
        # defaults (per-shot budget 8 s, one transient replay).
        decode_shot_budget_s=float(
            raw.get("decode_shot_budget_s", VerificationConfig.decode_shot_budget_s)
        ),
        # Runtime ceiling of the final-stage shard watchdog (2026-09-24); the
        # snapshot never carries it, so this only serves hand-written dicts.
        final_decode_shard_timeout_cap_s=(
            float(raw["final_decode_shard_timeout_cap_s"])
            if raw.get("final_decode_shard_timeout_cap_s") is not None
            else None
        ),
        search_decode_shard_timeout_cap_s=(
            float(raw["search_decode_shard_timeout_cap_s"])
            if raw.get("search_decode_shard_timeout_cap_s") is not None
            else None
        ),
        distance_timeout_s=typed_timeout(
            "distance_timeout_s", VerificationConfig.distance_timeout_s
        ),
        timeout_retries=int(
            raw.get("timeout_retries", VerificationConfig.timeout_retries)
        ),
        transient_retries=int(
            raw.get("transient_retries", VerificationConfig.transient_retries)
        ),
        d_cap_enabled=bool(raw.get("d_cap_enabled", VerificationConfig.d_cap_enabled)),
        d_cap=int(raw.get("d_cap", VerificationConfig.d_cap)),
        dsqrtn_floor=float(raw.get("dsqrtn_floor", VerificationConfig.dsqrtn_floor)),
        max_check_weight=(
            None
            if raw.get("max_check_weight", VerificationConfig.max_check_weight) is None
            else int(raw.get("max_check_weight", VerificationConfig.max_check_weight))
        ),
        exact_check_weight=(
            None if raw.get("exact_check_weight") is None else int(raw["exact_check_weight"])
        ),
        ler_ci_z=float(raw.get("ler_ci_z", VerificationConfig.ler_ci_z)),
        slope_offset=(
            None
            if raw.get("slope_offset", VerificationConfig.slope_offset) is None
            else float(raw.get("slope_offset", VerificationConfig.slope_offset))
        ),
        decoder_build=DecoderBuildConfig(
            # beam is the PRIMARY decoder — dropping it here silently reverted
            # any non-default beam config at every subprocess boundary.
            beam=BeamDecoderConfig(**beam_raw),
            lsd=LsdDecoderConfig(**lsd_raw),
            bposd=BposdDecoderConfig(**bposd_raw),
            relaybp=RelayBpDecoderConfig(**relaybp_kwargs),
            telescoping=TelescopingDecoderConfig.from_dict(telescoping_raw),
        ),
        exact_distance=ExactDistanceConfig(**exact_distance_raw),
        structural=StructuralSearchConfig(**structural_raw),
        # v3.5 SEC policy. ABSENT (a v3.4 artifact) -> the default twobasis
        # policy; such an artifact is refused by the snapshot/resume check
        # (its protocol snapshot has no sec_policy_id key), never silently
        # re-measured under the new circuit.
        sec_policy=SecPolicyConfig.from_dict(
            raw.get("sec_policy") if isinstance(raw.get("sec_policy"), dict) else None
        ),
        # Fig.3 Phase B: absent (every Fig.2 artifact) -> the memory task.
        task=str(raw.get("task") or "memory"),
        # 2026-10-15: absent (every Fig.2 artifact) -> max_Tc_d.
        rounds_policy=str(raw.get("rounds_policy") or "max_Tc_d"),
    )
