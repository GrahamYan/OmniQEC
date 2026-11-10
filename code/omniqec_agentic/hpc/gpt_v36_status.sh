#!/bin/bash
# The v3.6 status view for the ChatGPT (GPT) arm: claude_v36_status.sh with
# ARM=gpt, i.e. runs/fig2_gptsub_b<N>_<tag>, runs/logs/gpt_grid_<tag>.* and
# the omni_gpt_v36 job name. Same launcher, same protocol, same checks.
#
#   bash omniqec_agentic/hpc/gpt_v36_status.sh
#   RUN_TAG=v37_canary bash omniqec_agentic/hpc/gpt_v36_status.sh
ARM=gpt exec bash "$(dirname "${BASH_SOURCE[0]}")/claude_v36_status.sh" "$@"
