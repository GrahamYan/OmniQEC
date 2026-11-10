"""LRCSchedule — the timing law of the left-right circuit.

Timing law (paper frame, steps 1..t per steady round, t = t1 + t2 + 2),
pinned against SBB Fig. 3 AND reverse-engineered step-by-step from the
Zenodo golden circuit HGP625_LRC (record 10.5281/zenodo.18853601):

  step 1        : X-ancilla prep (|+>)      || LZ wrap gates (color c at
                                               step t1+3+c > t of PREVIOUS round)
  steps 2..t1+1 : window 1 — LX color c at step 1+c  ||  RZ color c at step 1+c
  step t1+2     : Z-ancilla measure (M)     || RX color 1
  step t1+3     : Z-ancilla prep (|0>)      || RX color 2
  steps ..t1+t2+1 : RX color c at step t1+1+c  ||  LZ color c at step t1+3+c
  step t        : X-ancilla measure (MX)    || LZ gates (continuing)

The Z measurement cycle is staggered: prep(round m) -> LZ (end of round m,
wrapping into step 1 of m+1) -> RZ (window 1 of m+1) -> measure (step t1+2 of
m+1). Hence every Z-check CNOT precedes every X-check CNOT of the same round
— the circuit is non-interleaved and therefore valid for any CSS code and any
partition (SBB Sec. III.A; validity is unconditional).

Boundary rules (also from the golden circuit):
  - prologue: after global prep, the first Z cycle's LZ colors run in
    |C(LZ)| dedicated steps (everything else idles); round 1 then starts at
    window 1 and has no separate step-1 (X prep merged into global prep);
  - epilogue: the final round's step-t segment also carries the final data
    measurement, the truncated final Z-ancilla measurement, the
    reconstruction detectors, and the observables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

from .coloring import Edge, minimal_edge_coloring, permuted_coloring
from .graph import SectorGraphs

# Subgraph slots in canonical order (also the permutation-stream slot ids).
SLOTS = ("LX", "RX", "LZ", "RZ")


@dataclass(frozen=True)
class LRCSchedule:
    """A fully specified left-right schedule for one candidate code."""

    n_data: int
    left: int
    m_x: int
    m_z: int
    t1: int
    t2: int
    layers: dict[str, list[list[Edge]]]  # slot -> color classes (post-permutation)
    candidate_index: int = 0  # 0 = canonical

    @property
    def t(self) -> int:
        return self.t1 + self.t2 + 2

    @property
    def depth(self) -> int:
        return self.t

    @property
    def source(self) -> str:
        """Schedule provenance tag (compat with the old ScheduleInfo field)."""

        return "lrc_ranked_v1"

    # -- per-round step maps -------------------------------------------------
    # Steps are 1-based within the paper frame. LZ steps may exceed t: they
    # wrap into the NEXT round's step (s - t). The builder handles wrapping.

    def window_step(self, slot: str, color_index: int) -> int:
        """Paper-frame step at which color class `color_index` (0-based) fires."""

        c = color_index + 1
        if slot in ("LX", "RZ"):
            return 1 + c
        if slot == "RX":
            return self.t1 + 1 + c
        if slot == "LZ":
            return self.t1 + 3 + c
        raise ValueError(f"unknown slot {slot}")

    def gates_at_step(self, step: int) -> Iterator[tuple[str, Edge]]:
        """All CNOTs firing at paper-frame `step`, including LZ wrap gates
        (which belong to the previous round's Z cycle)."""

        for slot in SLOTS:
            for color_index, group in enumerate(self.layers[slot]):
                s = self.window_step(slot, color_index)
                if s == step or (slot == "LZ" and s == step + self.t):
                    for edge in group:
                        yield slot, edge

    def to_metadata(self) -> dict:
        return {
            "sec_policy_id": "lrc_ranked_v1",
            "candidate_index": int(self.candidate_index),
            "t1": self.t1,
            "t2": self.t2,
            "depth": self.t,
            "color_counts": {slot: len(self.layers[slot]) for slot in SLOTS},
        }


def build_canonical_schedule(
    sectors: SectorGraphs, *, candidate_index: int = 0
) -> LRCSchedule:
    """Canonical minimal-coloring LRC; candidate_index > 0 permutes colors."""

    base = {
        "LX": minimal_edge_coloring(sectors.lx),
        "RX": minimal_edge_coloring(sectors.rx),
        "LZ": minimal_edge_coloring(sectors.lz),
        "RZ": minimal_edge_coloring(sectors.rz),
    }
    layers = {
        slot: permuted_coloring(base[slot], candidate=candidate_index, slot=slot_id)
        for slot_id, slot in enumerate(SLOTS)
    }
    t1 = max(len(layers["LX"]), len(layers["RZ"]))
    t2 = max(len(layers["LZ"]), len(layers["RX"]))
    return LRCSchedule(
        n_data=sectors.n_data,
        left=sectors.left,
        m_x=sectors.m_x,
        m_z=sectors.m_z,
        t1=t1,
        t2=t2,
        layers=layers,
        candidate_index=candidate_index,
    )
