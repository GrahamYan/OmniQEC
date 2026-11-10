"""Syndrome-extraction circuit synthesis (design doc 02; SBB arXiv:2603.05481).

sec_policy_id versions the whole synthesis policy: schedule law, ranking rules,
and builder conventions. LER results across different policy ids are not
comparable (design doc 02 §8).

  twobasis_nonint_v1  (v3.5 default) both bases measured every round, no
                      interleaving, cheap intra-basis order search + ISD
                      certification (02 §9.4)
  lrc_ranked_v1       historical Left-Right Circuit policy (selectable)
"""

SEC_POLICY_ID = "twobasis_nonint_v1"
LRC_SEC_POLICY_ID = "lrc_ranked_v1"

from .graph import SectorGraphs, split_sectors
from .coloring import minimal_edge_coloring, permuted_coloring
from .schedule import LRCSchedule, build_canonical_schedule
from .schedule_rank import RankedSchedule, rank_schedules
from .stim_builder import build_lrc_memory_circuit, build_lrc_memory_pair
from .audits import audit_twobasis_static, run_all_audits
from .layers import BlockStructure, build_basis_layers, build_layer_sets
from .twobasis_rank import TwoBasisSchedule, rank_twobasis_schedules
from .twobasis_builder import build_twobasis_memory_circuit, build_twobasis_memory_pair

__all__ = [
    "SEC_POLICY_ID",
    "LRC_SEC_POLICY_ID",
    "SectorGraphs",
    "split_sectors",
    "minimal_edge_coloring",
    "permuted_coloring",
    "LRCSchedule",
    "build_canonical_schedule",
    "RankedSchedule",
    "rank_schedules",
    "build_lrc_memory_circuit",
    "build_lrc_memory_pair",
    "run_all_audits",
    "audit_twobasis_static",
    "BlockStructure",
    "build_basis_layers",
    "build_layer_sets",
    "TwoBasisSchedule",
    "rank_twobasis_schedules",
    "build_twobasis_memory_circuit",
    "build_twobasis_memory_pair",
]
