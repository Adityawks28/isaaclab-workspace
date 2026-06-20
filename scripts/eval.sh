#!/bin/bash
# Watch a trained policy in the Isaac Lab GUI (X11), inside the container.
# Usage: ./scripts/eval.sh [--task <id>] [--num_envs <n>] [--checkpoint <path>]
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_common.sh"

TASK="Isaac-Reach-Franka-v0"
NUM_ENVS="16"
CKPT=""                                     # empty -> rsl_rl loads the latest run
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --num_envs) NUM_ENVS="$2"; shift 2;;
    --checkpoint) CKPT="--checkpoint $2"; shift 2;;
    *) shift;;
  esac
done

xhost +local:root >/dev/null 2>&1 || true
docker exec -e DISPLAY="${DISPLAY:-:1}" "${ISAACLAB_CONTAINER}" bash -lc \
  "cd /workspace/isaaclab && ./isaaclab.sh -p ${RSL_RL_DIR}/play.py \
     --task ${TASK} --num_envs ${NUM_ENVS} ${CKPT}"
