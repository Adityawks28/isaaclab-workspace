"""Milestone 3a check: confirm the camera-enabled Lift env renders an RGB image.

Loads Isaac-Lift-Cube-Franka-Camera-v0 with cameras on, steps a few times, and
prints the wrist-camera RGB shape. Run: ./isaaclab.sh -p vla_camera_test.py --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.headless = True


def log(*a):
    print("VLA3A:", *a, flush=True)


app = AppLauncher(args).app
log("app launched")

import gymnasium as gym
import torch

import lift_franka  # noqa: F401  (registers the camera task)
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-Lift-Cube-Franka-Camera-v0"

try:
    log("registered?", TASK in gym.registry)
    cfg = parse_env_cfg(TASK, num_envs=args.num_envs)
    log("cfg parsed; making env")
    env = gym.make(TASK, cfg=cfg)
    log("env made; resetting")
    obs, _ = env.reset()
    log("reset done; stepping")
    act_dim = env.unwrapped.action_manager.total_action_dim
    for i in range(6):
        actions = torch.zeros((args.num_envs, act_dim), device=env.unwrapped.device)
        obs, _, term, trunc, _ = env.step(actions)
    log("stepped 6x; reading camera")
    cam = env.unwrapped.scene["wrist_cam"]
    rgb = cam.data.output["rgb"]
    log("CAMERA_OK rgb shape", tuple(rgb.shape), "dtype", str(rgb.dtype),
        "min", float(rgb.min()), "max", float(rgb.max()))
    env.close()
except Exception as e:  # noqa: BLE001
    import traceback
    log("FAILED:", repr(e))
    traceback.print_exc()

app.close()
log("done")
