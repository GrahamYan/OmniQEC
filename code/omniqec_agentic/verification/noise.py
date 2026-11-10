"""Versioned noise profiles and provenance-aware sweep points for Experiment 2A."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


SI1000_REFERENCE_P = 1.0e-3
HELIOS_PRIMARY_SCALE = 1.0

SI1000_SOURCE = "arXiv:2108.10457v2"
HELIOS_PAPER_SOURCE = "arXiv:2511.05465v1"
HELIOS_EMULATOR_SOURCE = "Helios-1E_2025-11-18"

HELIOS_RZZ_ORDER = (
    "IX", "IY", "IZ", "XI", "XX", "XY", "XZ", "YI",
    "YX", "YY", "YZ", "ZI", "ZX", "ZY", "ZZ",
)
HELIOS_RZZ_PAULI = (
    4.5e-5, 4.5e-5, 19.0e-5, 5.8e-5, 0.06e-5,
    0.06e-5, 5.8e-5, 5.8e-5, 0.06e-5, 0.06e-5,
    5.8e-5, 19.0e-5, 4.5e-5, 4.5e-5, 5.9e-5,
)
HELIOS_RZZ_TOTAL = float(sum(HELIOS_RZZ_PAULI))
SI1000_STANDARD_MEASUREMENT_FACTOR = 5.0
SI1000_ENHANCED_MEASUREMENT_FACTOR = 10.0
HELIOS_ENHANCED_SHARPENING_GAMMA = 2.0


class NoiseProfile(str, Enum):
    """Versioned noise profiles (02 §6.1).

    UNIFORM_LEGACY  uniform circuit-level depolarizing with the SBB-golden
                    idle rule (X/Z flip after reset, selective idle
                    DEPOLARIZE1, final DEPOLARIZE1 on data); kept only for
                    golden parity / structural audits / v3.4 history.
    UNIFORM_PAPER   OmniQEC paper SI C b "Noise model" (v3.5 default):
                    DEPOLARIZE1(p) after every reset, DEPOLARIZE2(p) after
                    every CX, bit-flip p before every ancilla measurement,
                    basis-conjugate flip p before the final data readout;
                    NO idle noise.
    """

    UNIFORM_LEGACY = "uniform_legacy_v1"
    UNIFORM_PAPER = "uniform_paper_v1"
    # 2026-09-15 brute-force study: "Pauli-label-uniform, no-ordinary-idle".
    # On the uniform (CX-native) timeline: flip p before every measurement
    # (incl. the final data readout) and after every reset, PAULI_CHANNEL_1
    # (p,p,p) on BOTH qubits of every CX (X, Y and Z each with probability p),
    # DEPOLARIZE1(p) "resonator idle" on every qubit not being measured/reset
    # while some qubit is, NO ordinary idle noise. The timeline has no 1q
    # Clifford layers, so the 1q-gate rule of the model never fires.
    PAULI_LABEL_UNIFORM = "pauli_label_uniform_v1"
    SI1000 = "si1000_modern_v1"
    SI1000_MEASUREMENT_X2 = "si1000_measurement_x2_v2"
    HELIOS = "helios_calibration_informed_pauli_v1"
    HELIOS_SYMMETRIZED = "helios_symmetrized_v1"
    HELIOS_PAULI_SHARPENED = "helios_pauli_sharpened_gamma2_v2"


# The "uniform family" (one physical rate p on the LRC timeline): anything that
# asks "is this a uniform profile?" must test membership here, never the
# legacy member alone.
UNIFORM_PROFILES: frozenset[NoiseProfile] = frozenset(
    {NoiseProfile.UNIFORM_LEGACY, NoiseProfile.UNIFORM_PAPER}
)

UNIFORM_PAPER_SOURCE_KIND = "paper_protocol_parameter"
UNIFORM_PAPER_SOURCE = "OmniQEC paper SI C uniform v1"
UNIFORM_LEGACY_SOURCE_KIND = "legacy_protocol_parameter"
UNIFORM_LEGACY_SOURCE = "OmniQEC uniform legacy v1"
_UNIFORM_SOURCE_DEFAULTS = {
    NoiseProfile.UNIFORM_PAPER: (UNIFORM_PAPER_SOURCE_KIND, UNIFORM_PAPER_SOURCE),
    NoiseProfile.UNIFORM_LEGACY: (UNIFORM_LEGACY_SOURCE_KIND, UNIFORM_LEGACY_SOURCE),
}


class RzzOperandOrientation(str, Enum):
    CONTROL_FIRST = "control_first"
    TARGET_FIRST = "target_first"
    SWAP_AVERAGE = "swap_average"


@dataclass(frozen=True)
class SI1000NoiseConfig:
    p: float
    version: str = "modern"
    noisy_boundaries: bool = True
    measurement_factor: float = SI1000_STANDARD_MEASUREMENT_FACTOR

    def __post_init__(self) -> None:
        p = float(self.p)
        factor = float(self.measurement_factor)
        if factor < 0 or not 0 <= p or factor * p > 1:
            raise ValueError(
                f"SI1000 requires 0 <= p and measurement_factor*p={factor:g}p <= 1"
            )
        if self.version != "modern":
            raise ValueError("only the modern SI1000 interpretation is supported")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HeliosNoiseConfig:
    scale: float = 1.0
    p1: float = 2.5e-5
    p_init: float = 5.0e-4
    p_meas: float = 1.0e-6
    orientation: RzzOperandOrientation = RzzOperandOrientation.CONTROL_FIRST
    symmetrized: bool = False
    approximate_disjoint_errors: float = 1.0e-3
    pauli_sharpening_gamma: float = 1.0

    def __post_init__(self) -> None:
        if float(self.scale) < 0:
            raise ValueError("Helios scale must be non-negative")
        if float(self.pauli_sharpening_gamma) <= 0:
            raise ValueError("Helios pauli_sharpening_gamma must be positive")
        for name, value in (("p1", self.p1), ("p_init", self.p_init), ("p_meas", self.p_meas)):
            scaled = float(self.scale) * float(value)
            if not 0 <= scaled <= 1:
                raise ValueError(f"Helios {name} probability is outside [0,1]")
        if self.symmetrized and self.orientation != RzzOperandOrientation.CONTROL_FIRST:
            raise ValueError("Helios symmetrized control only accepts control_first orientation")
        probs = helios_rzz_probs(self)
        if probs and max(probs) > float(self.approximate_disjoint_errors):
            raise ValueError(
                "Helios PAULI_CHANNEL_2 component exceeds approximate_disjoint_errors threshold"
            )
        if sum(probs) > 1:
            raise ValueError("Helios RZZ Pauli probabilities sum to more than one")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["orientation"] = self.orientation.value
        return data


def helios_rzz_probs(config: HeliosNoiseConfig) -> tuple[float, ...]:
    raw = dict(zip(HELIOS_RZZ_ORDER, HELIOS_RZZ_PAULI, strict=True))
    if config.symmetrized:
        values = [HELIOS_RZZ_TOTAL / len(HELIOS_RZZ_ORDER)] * len(HELIOS_RZZ_ORDER)
    elif config.orientation == RzzOperandOrientation.CONTROL_FIRST:
        values = [raw[label] for label in HELIOS_RZZ_ORDER]
    elif config.orientation == RzzOperandOrientation.TARGET_FIRST:
        values = [raw[label[::-1]] for label in HELIOS_RZZ_ORDER]
    else:
        values = [(raw[label] + raw[label[::-1]]) / 2 for label in HELIOS_RZZ_ORDER]
    gamma = float(config.pauli_sharpening_gamma)
    if gamma != 1.0 and values:
        total = float(sum(values))
        powered = [float(value) ** gamma for value in values]
        denominator = float(sum(powered))
        values = [total * value / denominator for value in powered]
    return tuple(float(config.scale) * value for value in values)


@dataclass(frozen=True)
class NoisePoint:
    profile: NoiseProfile
    value: float
    axis: str
    source_kind: str
    source_reference: str
    orientation: RzzOperandOrientation = RzzOperandOrientation.CONTROL_FIRST
    approximate_disjoint_errors: float = 1.0e-3
    si1000_measurement_factor: float = SI1000_STANDARD_MEASUREMENT_FACTOR
    helios_pauli_sharpening_gamma: float = 1.0

    def si1000_config(self) -> SI1000NoiseConfig:
        if self.profile not in {NoiseProfile.SI1000, NoiseProfile.SI1000_MEASUREMENT_X2}:
            raise ValueError("noise point is not SI1000")
        return SI1000NoiseConfig(
            p=float(self.value),
            measurement_factor=float(self.si1000_measurement_factor),
        )

    def helios_config(self) -> HeliosNoiseConfig:
        if self.profile not in {
            NoiseProfile.HELIOS,
            NoiseProfile.HELIOS_SYMMETRIZED,
            NoiseProfile.HELIOS_PAULI_SHARPENED,
        }:
            raise ValueError("noise point is not Helios")
        return HeliosNoiseConfig(
            scale=float(self.value),
            orientation=self.orientation,
            symmetrized=self.profile == NoiseProfile.HELIOS_SYMMETRIZED,
            approximate_disjoint_errors=float(self.approximate_disjoint_errors),
            pauli_sharpening_gamma=float(self.helios_pauli_sharpening_gamma),
        )

    @property
    def config_hash(self) -> str:
        encoded = json.dumps(self._canonical_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _canonical_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "axis": self.axis,
            "value": float(self.value),
            "source_kind": self.source_kind,
            "source_reference": self.source_reference,
            "orientation": self.orientation.value,
            "approximate_disjoint_errors": float(self.approximate_disjoint_errors),
            "si1000_measurement_factor": float(self.si1000_measurement_factor),
            "helios_pauli_sharpening_gamma": float(self.helios_pauli_sharpening_gamma),
        }

    def to_dict(self) -> dict[str, Any]:
        data = self._canonical_dict()
        data["config_hash"] = self.config_hash
        return data


@dataclass(frozen=True)
class NoiseSweepConfig:
    profile: NoiseProfile = NoiseProfile.UNIFORM_PAPER
    values: tuple[float, ...] = (0.005,)
    source_kind: str = UNIFORM_PAPER_SOURCE_KIND
    source_reference: str = UNIFORM_PAPER_SOURCE
    orientation: RzzOperandOrientation = RzzOperandOrientation.CONTROL_FIRST
    approximate_disjoint_errors: float = 1.0e-3
    si1000_measurement_factor: float = SI1000_STANDARD_MEASUREMENT_FACTOR
    helios_pauli_sharpening_gamma: float = 1.0

    def __post_init__(self) -> None:
        if not self.values:
            raise ValueError("noise sweep requires at least one point")
        for value in self.values:
            self._validate_value(float(value))

    @property
    def axis(self) -> str:
        return "scale" if self.profile in {
            NoiseProfile.HELIOS,
            NoiseProfile.HELIOS_SYMMETRIZED,
            NoiseProfile.HELIOS_PAULI_SHARPENED,
        } else "p"

    def _validate_value(self, value: float) -> None:
        point = NoisePoint(
            profile=self.profile,
            value=value,
            axis=self.axis,
            source_kind=self.source_kind,
            source_reference=self.source_reference,
            orientation=self.orientation,
            approximate_disjoint_errors=self.approximate_disjoint_errors,
            si1000_measurement_factor=self.si1000_measurement_factor,
            helios_pauli_sharpening_gamma=self.helios_pauli_sharpening_gamma,
        )
        if self.profile in {NoiseProfile.SI1000, NoiseProfile.SI1000_MEASUREMENT_X2}:
            point.si1000_config()
        elif self.profile in {
            NoiseProfile.HELIOS,
            NoiseProfile.HELIOS_SYMMETRIZED,
            NoiseProfile.HELIOS_PAULI_SHARPENED,
        }:
            point.helios_config()
        elif not 0 <= value <= 1:
            raise ValueError("uniform p must be in [0,1]")

    def point(self, index: int) -> NoisePoint:
        return NoisePoint(
            profile=self.profile,
            value=float(self.values[int(index)]),
            axis=self.axis,
            source_kind=self.source_kind,
            source_reference=self.source_reference,
            orientation=self.orientation,
            approximate_disjoint_errors=float(self.approximate_disjoint_errors),
            si1000_measurement_factor=float(self.si1000_measurement_factor),
            helios_pauli_sharpening_gamma=float(self.helios_pauli_sharpening_gamma),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "axis": self.axis,
            "values": [float(value) for value in self.values],
            "source_kind": self.source_kind,
            "source_reference": self.source_reference,
            "orientation": self.orientation.value,
            "approximate_disjoint_errors": float(self.approximate_disjoint_errors),
            "si1000_measurement_factor": float(self.si1000_measurement_factor),
            "helios_pauli_sharpening_gamma": float(self.helios_pauli_sharpening_gamma),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "NoiseSweepConfig":
        raw = dict(data or {})
        profile = NoiseProfile(str(raw.get("profile", NoiseProfile.UNIFORM_PAPER.value)))
        default_kind, default_reference = _UNIFORM_SOURCE_DEFAULTS.get(
            profile, (UNIFORM_PAPER_SOURCE_KIND, UNIFORM_PAPER_SOURCE)
        )
        return cls(
            profile=profile,
            values=tuple(float(value) for value in raw.get("values", (0.005,))),
            source_kind=str(raw.get("source_kind", default_kind)),
            source_reference=str(raw.get("source_reference", default_reference)),
            orientation=RzzOperandOrientation(
                str(raw.get("orientation", RzzOperandOrientation.CONTROL_FIRST.value))
            ),
            approximate_disjoint_errors=float(raw.get("approximate_disjoint_errors", 1.0e-3)),
            si1000_measurement_factor=float(
                raw.get("si1000_measurement_factor", SI1000_STANDARD_MEASUREMENT_FACTOR)
            ),
            helios_pauli_sharpening_gamma=float(
                raw.get("helios_pauli_sharpening_gamma", 1.0)
            ),
        )

    @classmethod
    def uniform(cls, values: tuple[float, ...] = (0.005,)) -> "NoiseSweepConfig":
        """The current uniform protocol: OmniQEC paper SI C b (v3.5)."""
        return cls(
            profile=NoiseProfile.UNIFORM_PAPER,
            values=tuple(values),
            source_kind=UNIFORM_PAPER_SOURCE_KIND,
            source_reference=UNIFORM_PAPER_SOURCE,
        )

    @classmethod
    def pauli_label_uniform(cls, values: tuple[float, ...] = (0.001,)) -> "NoiseSweepConfig":
        """Pauli-label-uniform, no-ordinary-idle (brute-force study, 2026-09-15)."""
        return cls(
            profile=NoiseProfile.PAULI_LABEL_UNIFORM,
            values=tuple(float(v) for v in values),
            source_kind="user_supplied_model_parameter",
            source_reference="pauli-label-uniform / no-ordinary-idle, p=0.001 (user spec 2026-09-15)",
        )

    @classmethod
    def uniform_legacy(cls, values: tuple[float, ...] = (0.005,)) -> "NoiseSweepConfig":
        """The v3.4-and-earlier uniform lowering (idle rule); history/golden only."""
        return cls(
            profile=NoiseProfile.UNIFORM_LEGACY,
            values=tuple(values),
            source_kind=UNIFORM_LEGACY_SOURCE_KIND,
            source_reference=UNIFORM_LEGACY_SOURCE,
        )

    @classmethod
    def si1000(
        cls,
        values: tuple[float, ...] = (SI1000_REFERENCE_P,),
        *,
        source_kind: str = "published_reference_point",
    ) -> "NoiseSweepConfig":
        return cls(
            profile=NoiseProfile.SI1000,
            values=tuple(values),
            source_kind=source_kind,
            source_reference=SI1000_SOURCE,
            si1000_measurement_factor=SI1000_STANDARD_MEASUREMENT_FACTOR,
        )

    @classmethod
    def si1000_measurement_enhanced(
        cls,
        values: tuple[float, ...] = (SI1000_REFERENCE_P,),
    ) -> "NoiseSweepConfig":
        return cls(
            profile=NoiseProfile.SI1000_MEASUREMENT_X2,
            values=tuple(values),
            source_kind="derived_contrast_profile",
            source_reference=f"{SI1000_SOURCE}; measurement error 5p -> 10p",
            si1000_measurement_factor=SI1000_ENHANCED_MEASUREMENT_FACTOR,
        )

    @classmethod
    def helios(
        cls,
        values: tuple[float, ...] = (HELIOS_PRIMARY_SCALE,),
        *,
        symmetrized: bool = False,
        orientation: RzzOperandOrientation = RzzOperandOrientation.CONTROL_FIRST,
        source_kind: str = "published_calibration",
    ) -> "NoiseSweepConfig":
        return cls(
            profile=NoiseProfile.HELIOS_SYMMETRIZED if symmetrized else NoiseProfile.HELIOS,
            values=tuple(values),
            source_kind=source_kind,
            source_reference=f"{HELIOS_PAPER_SOURCE}; {HELIOS_EMULATOR_SOURCE}",
            orientation=orientation,
        )

    @classmethod
    def helios_bias_enhanced(
        cls,
        values: tuple[float, ...] = (HELIOS_PRIMARY_SCALE,),
        *,
        orientation: RzzOperandOrientation = RzzOperandOrientation.CONTROL_FIRST,
    ) -> "NoiseSweepConfig":
        return cls(
            profile=NoiseProfile.HELIOS_PAULI_SHARPENED,
            values=tuple(values),
            source_kind="derived_contrast_profile",
            source_reference=(
                f"{HELIOS_PAPER_SOURCE}; {HELIOS_EMULATOR_SOURCE}; "
                "RZZ Pauli distribution sharpened at fixed total probability"
            ),
            orientation=orientation,
            helios_pauli_sharpening_gamma=HELIOS_ENHANCED_SHARPENING_GAMMA,
        )


def noise_sweep_from_legacy_p_values(values: tuple[float, ...]) -> NoiseSweepConfig:
    """``--p/--p-values`` aliases mean "the current uniform protocol" (paper)."""
    return NoiseSweepConfig.uniform(tuple(float(value) for value in values))
