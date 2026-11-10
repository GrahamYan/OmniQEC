#!/usr/bin/env bash
# Status view of the DeepSeek arm on NSCC (2026-09-10, OpenRouter): the
# claude_v36_status.sh logic with ARM=deepseek, i.e. runs/fig2_deepseek_b<N>_<tag>,
# runs/logs/deepseek_grid_<tag>.* and jobs named omni_deepseek_v36.
#
#   bash omniqec_agentic/hpc/deepseek_v36_status.sh            # RUN_TAG=v36
#   REPO=<deployed tree> bash omniqec_agentic/hpc/deepseek_v36_status.sh
ARM=deepseek exec bash "$(dirname "${BASH_SOURCE[0]}")/claude_v36_status.sh" "$@"
