#!/bin/bash
# Headless rsl_rl PPO training inside the Isaac Lab container.
# Usage: ./scripts/train.sh [--task <id>] [--num_envs <n>] [extra train.py args...]
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_common.sh"

TASK="Isaac-Reach-Franka-v0"
NUM_ENVS="256"                              # laptop-safe (6 GB); raise on the workstation
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --num_envs) NUM_ENVS="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

docker exec "${ISAACLAB_CONTAINER}" bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p ${RSL_RL_DIR}/train.py \
     --task ${TASK} --num_envs ${NUM_ENVS} --headless ${EXTRA[*]}"
