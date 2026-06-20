#!/bin/bash
# Shared paths/vars sourced by the workspace scripts.
COMMON_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "${COMMON_DIR}/.." && pwd )"
ISAACLAB_DIR="${REPO_DIR}/_isaaclab"
ISAACLAB_TAG="v2.1.0"                       # pinned for Isaac Sim 4.5.0 compatibility
ISAACSIM_BASE_IMAGE="nvcr.io/nvidia/isaac-sim:4.5.0"
ISAACLAB_CONTAINER="isaac-lab-base"         # container name created by container.py (base profile)
RSL_RL_DIR="scripts/reinforcement_learning/rsl_rl"   # train.py / play.py live here
