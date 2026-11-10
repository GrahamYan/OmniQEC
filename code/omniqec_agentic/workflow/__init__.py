"""Workflow layer (03 §2): the only package that imports langgraph.

Swapping frameworks means rewriting this thin layer — orchestrator/,
evaluation/, sec/, construction/, verification/ stay framework-free."""

from .graph import build_campaign_graph, run_campaign
from .state import CampaignSpec, CampaignState, build_protocol_snapshot

__all__ = [
    "CampaignSpec",
    "CampaignState",
    "build_campaign_graph",
    "build_protocol_snapshot",
    "run_campaign",
]
