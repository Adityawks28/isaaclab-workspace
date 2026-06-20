"""Milestone 3a scaffold: run a pretrained VLA (Octo) in the Isaac Lab Lift env.

This is a STRUCTURAL SKELETON, not a tested script. It lays out the 3a inference
pipeline so it can be fleshed out once the GPU is free and Octo is installed. Every
VLA/camera-specific line is marked `# VERIFY`, confirm against the real APIs before
relying on it. Design: docs/superpowers/specs/2026-06-20-milestone3-vla-design.md

Pipeline (one control step):
    rendered camera image + language instruction
        -> VLA (Octo) -> end-effector action (delta pose + gripper)
        -> Isaac Lab IK action space -> PhysX step -> next image

Prereqs (do once, when the GPU is free):
    # inside the container, in a separate venv (Octo is JAX; keep it off the sim python)
    pip install "octo @ git+https://github.com/octo-models/octo"   # VERIFY install path
    # weights download on first model load (network, a few hundred MB)

Run (target, once implemented):
   ./isaaclab.sh -p scripts/vla_play.py \
        --task Isaac-Lift-Cube-Franka-IK-Rel-v0 --num_envs 1 --enable_cameras
"""

from __future__ import annotations

import argparse

# --- 1. Launch the Isaac Sim app FIRST (Isaac Lab requires AppLauncher before imports) ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Lift-Cube-Franka-IK-Rel-v0")  # IK = EE-delta action space, matches VLA output
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--instruction", default="pick up the cube")
parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True  # a VLA needs rendered images

app = AppLauncher(args).app  # boots Isaac Sim; everything below imports the sim stack

import gymnasium as gym
import torch

from isaaclab_tasks.utils import parse_env_cfg


def add_wrist_camera(env_cfg):
    """VERIFY: attach a camera sensor and expose its RGB as an observation.

    The stock Lift env has no camera. Add a `CameraCfg`/`TiledCameraCfg` (e.g. mounted on
    panda_hand looking at the workspace, ~224x224 to match the VLA's expected input) and a
    matching observation term so each step yields an RGB frame. See Isaac Lab camera tutorials.
    """
    # env_cfg.scene.wrist_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand/cam",...)
    # env_cfg.observations.policy.image = ObsTerm(func=mdp.image, params={...})
    return env_cfg  # TODO


def load_vla():
    """VERIFY: load pretrained Octo and return a callable (image, text) -> action.

    Octo is JAX-based; load the released checkpoint and wrap its sample_actions API.
    Returns a function mapping (HxWx3 uint8 image, instruction str) -> action vector
    in the VLA's convention (EE delta xyz + delta rot + gripper).
    """
    # from octo.model.octo_model import OctoModel
    # model = OctoModel.load_pretrained("hf://rail-berkeley/octo-small-1.5")  # VERIFY id
    # task = model.create_tasks(texts=[instruction])
    # return lambda img: model.sample_actions(img, task,...)
    raise NotImplementedError("install + wire Octo (see module docstring)")  # TODO


def vla_action_to_env(vla_action: torch.Tensor) -> torch.Tensor:
    """VERIFY: map the VLA's action convention to the IK env's action tensor.

    Octo/OpenVLA emit (dx, dy, dz, droll, dpitch, dyaw, gripper) in their own frame/scale.
    The IK-Rel env expects a delta EE pose + a binary gripper; align frames, scale, and
    clip. Getting this mapping right is most of the 3a engineering.
    """
    return vla_action  # TODO


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    env_cfg = add_wrist_camera(env_cfg)
    env = gym.make(args.task, cfg=env_cfg)

    vla = load_vla()
    obs, _ = env.reset()
    for _ in range(args.steps):
        image = obs["policy"]["image"]            # VERIFY: the camera obs key
        vla_action = vla(image, args.instruction)  # image + language -> action
        action = vla_action_to_env(vla_action)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            obs, _ = env.reset()

    env.close()
    app.close()


if __name__ == "__main__":
    main()
