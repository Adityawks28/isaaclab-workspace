"""Generate insertion+tactile demos in Isaac Lab and serialize them to the LeRobot contract.

Box-only (needs Isaac Sim). Drives a scripted pick-and-place state machine (the demonstration
"oracle") in Isaac-Insertion-Franka-Tactile-v0: the Franka grasps the peg (DexCube) and moves it
to the commanded goal pose (the "socket"). Each step we read the wrist/scene cameras and the two
fingertip contact-force sensors directly from the scene, and assemble a LeRobot frame with the
exact contract keys (see insertion_tactile.recorder.assemble_frame):

  observation.images.wrist  (224,224,3) uint8
  observation.images.scene  (224,224,3) uint8
  observation.state         proprio (env policy obs)
  observation.tactile       6-dim contact force = [left_xyz, right_xyz]
  action                    7-dim = [ee_pos(3), ee_euler(3), gripper(1)]
  task                      language string

lerobot is not installed in the Isaac image (it lives in the robot-research / Plan B repo), so we
save the contract arrays as .npz + meta.json here and only call write_lerobot_dataset() if lerobot
happens to be importable. Optionally records an MP4 of the rollout (--video).
"""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Insertion-Franka-Tactile-v0")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max_steps", type=int, default=220, help="Max control steps per episode.")
parser.add_argument("--out", default="/workspace/datasets/insertion_tactile_v1")
parser.add_argument("--repo_id", default="adityawks28/insertion_tactile_v1")
parser.add_argument("--video", action="store_true", help="Record an MP4 of the first episodes.")
parser.add_argument("--video_length", type=int, default=400)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--zoom", action="store_true", help="Zoom the render camera onto the arm + peg.")
parser.add_argument("--debug_tactile", action="store_true", help="Draw contact-force arrows at the fingertips.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True

app = AppLauncher(args).app

import json
import os
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import torch
import warp as wp

import isaaclab_tasks  # noqa: F401
import insertion_tactile

insertion_tactile.register()

from insertion_tactile.recorder import assemble_frame
from isaaclab.assets.rigid_object.rigid_object_data import RigidObjectData
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

wp.init()

LANGUAGE = "insert the peg into the socket"


# ----------------------------------------------------------------------------------------------
# Pick-and-place state machine (demonstration oracle) — warp kernel, mirrors Isaac Lab's lift SM.
# ----------------------------------------------------------------------------------------------
class GripperState:
    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class SmState:
    REST = wp.constant(0)
    APPROACH_ABOVE = wp.constant(1)
    APPROACH = wp.constant(2)
    GRASP = wp.constant(3)
    MOVE_TO_SOCKET = wp.constant(4)


class SmWait:
    REST = wp.constant(0.2)
    APPROACH_ABOVE = wp.constant(0.5)
    APPROACH = wp.constant(0.6)
    GRASP = wp.constant(0.3)
    MOVE_TO_SOCKET = wp.constant(1.0)


@wp.func
def close_enough(a: wp.vec3, b: wp.vec3, threshold: float) -> bool:
    return wp.length(a - b) < threshold


@wp.kernel
def infer_sm(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
    threshold: float,
):
    tid = wp.tid()
    state = sm_state[tid]
    if state == SmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper[tid] = GripperState.OPEN
        if sm_wait[tid] >= SmWait.REST:
            sm_state[tid] = SmState.APPROACH_ABOVE
            sm_wait[tid] = 0.0
    elif state == SmState.APPROACH_ABOVE:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper[tid] = GripperState.OPEN
        if close_enough(wp.transform_get_translation(ee_pose[tid]),
                        wp.transform_get_translation(des_ee_pose[tid]), threshold):
            if sm_wait[tid] >= SmWait.APPROACH:
                sm_state[tid] = SmState.APPROACH
                sm_wait[tid] = 0.0
    elif state == SmState.APPROACH:
        des_ee_pose[tid] = object_pose[tid]
        gripper[tid] = GripperState.OPEN
        if close_enough(wp.transform_get_translation(ee_pose[tid]),
                        wp.transform_get_translation(des_ee_pose[tid]), threshold):
            if sm_wait[tid] >= SmWait.APPROACH:
                sm_state[tid] = SmState.GRASP
                sm_wait[tid] = 0.0
    elif state == SmState.GRASP:
        des_ee_pose[tid] = object_pose[tid]
        gripper[tid] = GripperState.CLOSE
        if sm_wait[tid] >= SmWait.GRASP:
            sm_state[tid] = SmState.MOVE_TO_SOCKET
            sm_wait[tid] = 0.0
    elif state == SmState.MOVE_TO_SOCKET:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper[tid] = GripperState.CLOSE
        if close_enough(wp.transform_get_translation(ee_pose[tid]),
                        wp.transform_get_translation(des_ee_pose[tid]), threshold):
            if sm_wait[tid] >= SmWait.MOVE_TO_SOCKET:
                sm_state[tid] = SmState.MOVE_TO_SOCKET
                sm_wait[tid] = 0.0
    sm_wait[tid] = sm_wait[tid] + dt[tid]


class PickPlaceSm:
    def __init__(self, dt, num_envs, device, threshold=0.01):
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.threshold = threshold
        self.sm_dt = torch.full((num_envs,), self.dt, device=device)
        self.sm_state = torch.zeros((num_envs,), dtype=torch.int32, device=device)
        self.sm_wait = torch.zeros((num_envs,), device=device)
        self.des_ee_pose = torch.zeros((num_envs, 7), device=device)
        self.des_gripper = torch.zeros((num_envs,), device=device)
        self.offset = torch.zeros((num_envs, 7), device=device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_wp = wp.from_torch(self.sm_wait, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_wp = wp.from_torch(self.des_gripper, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, ids: Sequence[int] = None):
        if ids is None:
            ids = slice(None)
        self.sm_state[ids] = 0
        self.sm_wait[ids] = 0.0

    def compute(self, ee_pose, object_pose, des_object_pose):
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        wp.launch(
            infer_sm,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp, self.sm_state_wp, self.sm_wait_wp,
                wp.from_torch(ee_pose.contiguous(), wp.transform),
                wp.from_torch(object_pose.contiguous(), wp.transform),
                wp.from_torch(des_object_pose.contiguous(), wp.transform),
                self.des_ee_pose_wp, self.des_gripper_wp, self.offset_wp, self.threshold,
            ],
            device=self.device,
        )
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]  # back to (w,x,y,z)
        return torch.cat([des_ee_pose, self.des_gripper.unsqueeze(-1)], dim=-1)


# ----------------------------------------------------------------------------------------------
def to_np(t):
    return t.detach().cpu().numpy()


def action_to_contract(action_row: torch.Tensor) -> np.ndarray:
    """env action (pos3, quat_wxyz4, gripper1) -> contract 7 = [pos3, euler3, gripper1]."""
    pos = action_row[0:3]
    quat = action_row[3:7].unsqueeze(0)
    r, p, y = euler_xyz_from_quat(quat)
    euler = torch.stack([r[0], p[0], y[0]])
    grip = action_row[7:8]
    return to_np(torch.cat([pos, euler, grip])).astype(np.float32)


def main():
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    if args.zoom:
        # close 3/4 view framing the Franka hand + peg + socket region (world coords, 1 env at origin)
        env_cfg.viewer.eye = (0.95, 0.75, 0.6)
        env_cfg.viewer.lookat = (0.45, 0.0, 0.18)
    if args.debug_tactile:
        # draw contact-force arrows at the fingertips so the tactile is visible in the clip
        env_cfg.scene.tactile_left.debug_vis = True
        env_cfg.scene.tactile_right.debug_vis = True
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.video:
        os.makedirs("/workspace/videos", exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env, video_folder="/workspace/videos", step_trigger=lambda s: s == 0,
            video_length=args.video_length, name_prefix="insertion", disable_logger=True,
        )

    obs, _ = env.reset()
    base = env.unwrapped
    actions = torch.zeros(base.action_space.shape, device=base.device)
    actions[:, 3] = 1.0
    des_orn = torch.zeros((base.num_envs, 4), device=base.device)
    des_orn[:, 1] = 1.0
    sm = PickPlaceSm(env_cfg.sim.dt * env_cfg.decimation, base.num_envs, base.device, threshold=0.01)

    os.makedirs(args.out, exist_ok=True)
    episodes_written, total_frames = 0, 0
    ep_frames: list[dict] = []
    step_in_ep = 0

    def flush_episode(idx):
        nonlocal total_frames
        if not ep_frames:
            return
        data = {
            "observation.images.wrist": np.stack([f["observation.images.wrist"] for f in ep_frames]),
            "observation.images.scene": np.stack([f["observation.images.scene"] for f in ep_frames]),
            "observation.state": np.stack([f["observation.state"] for f in ep_frames]),
            "observation.tactile": np.stack([f["observation.tactile"] for f in ep_frames]),
            "action": np.stack([f["action"] for f in ep_frames]),
        }
        np.savez_compressed(os.path.join(args.out, f"episode_{idx:04d}.npz"), task=LANGUAGE, **data)
        total_frames += len(ep_frames)

    while episodes_written < args.episodes:
        with torch.inference_mode():
            obs = env.step(actions)[0]
            dones = base.termination_manager.dones if hasattr(base, "termination_manager") else None

            # --- read sensors directly from the scene ---
            scene = base.scene
            wrist = to_np(scene["wrist_cam"].data.output["rgb"][0]).astype(np.uint8)
            scn = to_np(scene["scene_cam"].data.output["rgb"][0]).astype(np.uint8)
            tl = to_np(scene["tactile_left"].data.net_forces_w[0, 0]).astype(np.float32)
            tr = to_np(scene["tactile_right"].data.net_forces_w[0, 0]).astype(np.float32)
            tactile = np.concatenate([tl, tr]).astype(np.float32)  # 6-dim
            proprio = to_np(obs["policy"][0]).astype(np.float32)

            # --- SM oracle: compute next action from ee/object/goal ---
            ee = scene["ee_frame"]
            tcp_pos = ee.data.target_pos_w[..., 0, :].clone() - scene.env_origins
            tcp_quat = ee.data.target_quat_w[..., 0, :].clone()
            obj: RigidObjectData = scene["object"].data
            obj_pos = obj.root_pos_w - scene.env_origins
            goal_pos = base.command_manager.get_command("object_pose")[..., :3]
            actions = sm.compute(
                torch.cat([tcp_pos, tcp_quat], dim=-1),
                torch.cat([obj_pos, des_orn], dim=-1),
                torch.cat([goal_pos, des_orn], dim=-1),
            )

            fr = assemble_frame(
                rgb_wrist=wrist, rgb_scene=scn, tactile_force=tactile,
                proprio=proprio, action=action_to_contract(actions[0]), language=LANGUAGE,
            )
            ep_frames.append(fr)
            step_in_ep += 1

            done = bool(dones[0]) if dones is not None else False
            if done or step_in_ep >= args.max_steps:
                flush_episode(episodes_written)
                episodes_written += 1
                ep_frames = []
                step_in_ep = 0
                sm.reset_idx()
                env.reset()

    env.close()

    # dataset metadata
    meta = {
        "repo_id": args.repo_id, "fps": 10, "episodes": episodes_written, "frames": total_frames,
        "language": LANGUAGE,
        "features": {
            "observation.images.wrist": {"dtype": "uint8", "shape": [224, 224, 3]},
            "observation.images.scene": {"dtype": "uint8", "shape": [224, 224, 3]},
            "observation.state": {"dtype": "float32"},
            "observation.tactile": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [7]},
        },
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # try proper LeRobot packaging if available (otherwise the npz + meta are the deliverable here)
    try:
        from insertion_tactile.recorder import write_lerobot_dataset  # noqa: F401
        import lerobot  # noqa: F401

        lerobot_ok = True
    except Exception as e:
        lerobot_ok = False
        with open(os.path.join(args.out, "LEROBOT_STATUS.txt"), "w") as f:
            f.write(f"lerobot not packaged in Isaac image: {e!r}\n")
            f.write("npz frames follow the LeRobot contract keys; run write_lerobot_dataset() in the Plan B (robot-research) container.\n")

    with open(os.path.join(args.out, "DONE.txt"), "w") as f:
        f.write(f"episodes={episodes_written} frames={total_frames} lerobot_packaged={lerobot_ok}\n")


if __name__ == "__main__":
    main()
    app.close()
