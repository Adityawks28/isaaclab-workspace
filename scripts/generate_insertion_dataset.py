"""STRUCTURAL SKELETON (box-only). Rolls out the scripted oracle in the insertion env and
records LeRobot demos. Mirrors scripts/vla_play.py's AppLauncher-first structure. Fix every
# VERIFY against the real obs keys (recorded during the Task 2 box smoke)."""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Insertion-Franka-Tactile-v0")
parser.add_argument("--episodes", type=int, default=150)
parser.add_argument("--repo_id", default="adityawks28/insertion_tactile_v1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np

from insertion_tactile.oracle import plan_insertion
from insertion_tactile.recorder import assemble_frame, write_lerobot_dataset
from isaaclab_tasks.utils import parse_env_cfg


def socket_pose_from_obs(obs):           # VERIFY: pull socket xyz from the env obs/state
    return np.zeros(3)


def step_action_from_waypoint(wp, obs):  # VERIFY: map Waypoint -> env IK + force action tensor
    raise NotImplementedError


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    frames, ep = [], 0
    obs, _ = env.reset()
    while ep < args.episodes:
        wps = plan_insertion(socket_pose_from_obs(obs), approach_height=0.1,
                             insert_depth=0.04, insert_force=5.0, n_steps=50)
        for wp in wps:
            action = step_action_from_waypoint(wp, obs)          # VERIFY
            obs, _, term, trunc, _ = env.step(action)
            fr = assemble_frame(
                rgb_wrist=obs["policy"]["wrist_cam"],            # VERIFY obs keys (Task 2)
                rgb_scene=obs["policy"]["scene_cam"],            # VERIFY
                tactile_force=obs["policy"]["tactile_left"],     # VERIFY (concat L/R if desired)
                proprio=obs["policy"]["proprio"],                # VERIFY
                action=action.cpu().numpy().reshape(-1)[:7],
                language="insert the connector into the socket",
            )
            fr["episode_index"] = ep
            frames.append(fr)
            if term or trunc:
                break
        obs, _ = env.reset()
        ep += 1
    write_lerobot_dataset(frames, repo_id=args.repo_id, fps=10)
    env.close()
    app.close()


if __name__ == "__main__":
    main()
