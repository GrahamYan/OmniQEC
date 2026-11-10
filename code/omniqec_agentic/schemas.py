"""Serializable data contracts for the agentic QEC discovery loop.

The primary genotype follows the ver1 contract: an LLM proposes a Python
`construct(rng)` program. The program is executed with a controlled RNG seed to
produce a concrete code spec. Family labels are properties of that produced
spec, not the identity of the candidate itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import DecoderName, HardwareTarget
from .ranking import is_degraded, is_flagged_degraded

BinaryMatrix = list[list[int]]


@dataclass(frozen=True)
class CandidateProgram:
    """LLM-authored ver1-style genotype."""

    program_id: str
    source: str
    rng_seed: int = 0
    n_budget: int | None = None
    parent_program_ids: list[str] = field(default_factory=list)
    source_agent: str = "proposal_agent"
    rationale: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    schema: str = "candidate_program"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConstructedSpec:
    """Concrete spec returned by executing `construct(rng)`."""

    program_id: str
    spec: dict[str, Any]
    family: str
    rng_seed: int = 0
    normalized: bool = False
    warnings: list[str] = field(default_factory=list)
    schema: str = "constructed_spec"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CompiledCode:
    """CSS ancestor plus an optional restricted local-Clifford deformation.

    ``hx``/``hz`` deliberately remain the CSS ancestor matrices so existing
    distance routines stay usable.  ``h_mask`` records data qubits whose X/Z
    axes are exchanged in the physical stabilizer code.
    """

    program_id: str
    n_budget: int
    hx: BinaryMatrix
    hz: BinaryMatrix
    family: str
    deformation: str = "none"
    h_mask: list[int] = field(default_factory=list)
    sector_sizes: list[int] = field(default_factory=list)
    construction_metadata: dict[str, Any] = field(default_factory=dict)
    # v3.5 SEC (twobasis_nonint_v1): JSON-safe block structure of hx/hz for the
    # structured layer colouring (sec/layers.py). Schema "layer_blocks_v1":
    # {"schema", "block_size": g, "X"/"Z": {"n_check_blocks", "n_data_blocks",
    # "terms": [[cb, db, perm], ...]}} where term [cb, db, perm] asserts
    # H_W[cb*g + perm[t], db*g + t] == 1 for every t in range(g) (perm maps
    # data-LOCAL column -> check-LOCAL row); terms XOR to exactly H_W and
    # never overlap. None for hgp / raw-matrix paths (fallback colouring).
    # Deliberately NOT part of code_hash or compiled_code_metadata.
    layer_blocks: dict[str, Any] | None = None
    code_hash: str | None = None
    schema: str = "compiled_code"

    @property
    def n_data(self) -> int:
        if self.hx and self.hx[0]:
            return len(self.hx[0])
        if self.hz and self.hz[0]:
            return len(self.hz[0])
        return 0

    @property
    def is_css(self) -> bool:
        return not any(self.h_mask)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["n_data"] = self.n_data
        return data


@dataclass(frozen=True)
class FastResult:
    """Proxy-tier record.

    ``valid`` is a compatibility alias for ``runtime_screen_passed`` (the
    MAP-niche / acquisition channel). This layer does not know whether a
    candidate is structurally eligible — that also needs `StructuralResult`
    and a reconstructable inheritance payload — so it deliberately does NOT
    name a field `runtime_eligible` or `structural_eligible`; the archive
    layer decides both (repair plan §4.1).
    """

    program_id: str
    valid: bool
    status: str
    n_budget: int
    n_data: int | None = None
    k: int | None = None
    # Read-only compatibility alias of d_search, written only once all proxy
    # seeds and both sectors have completed; equal to d_search by construction.
    d_proxy: int | None = None
    d_search: int | None = None
    q_score: float | None = None
    family: str | None = None
    max_check_weight: int | None = None
    max_qubit_degree: int | None = None
    code_hash: str | None = None
    proxy_valid: bool = False
    trust_filter_passed: bool = False
    runtime_screen_passed: bool = False
    below_min_promotion_distance: bool = False
    failure_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    schema: str = "fast_result"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StructuralResult:
    """Strict structural checks recorded alongside the ver1-style fast metric."""

    program_id: str
    valid: bool
    status: str
    n_budget: int | None = None
    n_data: int | None = None
    n_checks_x: int | None = None
    n_checks_z: int | None = None
    rank_hx: int | None = None
    rank_hz: int | None = None
    k: int | None = None
    css_violations: int | None = None
    commutation_violations: int | None = None
    stabilizer_rank: int | None = None
    is_css: bool | None = None
    deformation: str = "none"
    max_check_weight: int | None = None
    max_qubit_degree: int | None = None
    connected: bool | None = None
    code_hash: str | None = None
    # v3.6 (docs/06 §3 B3): exact pynauty canonical-form identity, None when
    # pynauty is unavailable. Persisted into structural ledger / promotion /
    # slow rows; with code_hash it is the ONLY near-duplicate gate.
    canonical_hash: str | None = None
    failure_reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    schema: str = "structural_result"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SlowResult:
    program_id: str
    hardware: HardwareTarget
    decoder: DecoderName
    status: str
    p: float
    shots: int
    rounds: int
    block_ler: float | None = None
    per_logical_per_round: float | None = None
    per_logical_block_ci: list[float] | None = None
    d_circ_ler_estimate: int | None = None
    degraded_estimate: bool | None = None
    exact_d_circ_status: str = "not_available"
    runtime_s: float | None = None
    failure_reason: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    schema: str = "slow_result"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hardware"] = self.hardware.value
        data["decoder"] = self.decoder.value
        return data


@dataclass(frozen=True)
class ArchiveRecord:
    program: CandidateProgram
    constructed: ConstructedSpec | None = None
    structural: StructuralResult | None = None
    fast: FastResult | None = None
    slow_results: list[SlowResult] = field(default_factory=list)
    archive_cell: str | None = None
    parent_eligible: bool = False
    # Reconcile channel: slow verification evidence written back by code_hash
    # (paper A4/A5 LER, d_circ evidence, degraded flags, hook risk).
    # The record stays frozen; this dict's contents are updated in place.
    slow_evidence: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    schema: str = "archive_record"

    def to_dict(self) -> dict[str, Any]:
        return {
            "program": self.program.to_dict(),
            "constructed": self.constructed.to_dict() if self.constructed is not None else None,
            "structural": self.structural.to_dict() if self.structural is not None else None,
            "fast": self.fast.to_dict() if self.fast is not None else None,
            "slow_results": [result.to_dict() for result in self.slow_results],
            "archive_cell": self.archive_cell,
            # Serialized value always reflects the runtime semantics: degraded
            # quarantine is decided dynamically from slow_evidence, so a record
            # downgraded after creation must not serialize as parent-eligible.
            # v3.6 (M2): `degraded` is the HARD quarantine (schedule_catastrophic
            # only); `flagged_degraded` keeps every degraded_* audit flag.
            "parent_eligible": bool(self.parent_eligible and not is_degraded(self.slow_evidence)),
            "degraded": is_degraded(self.slow_evidence),
            "flagged_degraded": is_flagged_degraded(self.slow_evidence),
            "slow_evidence": dict(self.slow_evidence),
            "notes": list(self.notes),
            "schema": self.schema,
        }


def family_from_spec(spec: dict[str, Any]) -> str:
    """Extract the family label produced by a construct program."""

    return str(spec.get("family") or "")


def normalize_source_text(source: str) -> str:
    """Normalize Python source strings that may be JSON-escaped twice."""

    text = str(source or "")
    candidates = [text]
    if "\\n" in text or '\\"' in text or "\\'" in text:
        try:
            decoded = text.encode("utf-8").decode("unicode_escape")
        except UnicodeError:
            decoded = ""
        if decoded and decoded != text:
            candidates.insert(0, decoded)

    for candidate in candidates:
        cleaned = candidate.strip() + "\n"
        if "def construct" in cleaned:
            return cleaned
    return text.strip() + "\n"
