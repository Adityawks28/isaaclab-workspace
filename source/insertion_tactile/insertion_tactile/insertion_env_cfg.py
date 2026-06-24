"""STRUCTURAL SKELETON (box-only). Franka peg-insertion env with contact-force tactile.

Every Isaac-specific line is # VERIFY against Isaac Lab `main` on Isaac Sim 5.1. Base it on the
Isaac Lab Factory insertion task (isaaclab_tasks ... manipulation.factory) where possible. This
mirrors the repo convention for un-CI-able sim code (see scripts/vla_play.py)."""
from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg            # VERIFY import path on Lab main
from isaaclab.sensors import CameraCfg, ContactSensorCfg  # VERIFY
from isaaclab.utils import configclass                    # VERIFY


@configclass
class FrankaInsertionTactileEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # --- scene: Franka + Factory peg/socket assets; IK-relative action space ---
        # self.scene.peg = RigidObjectCfg(...)             # VERIFY (Factory assets)
        # self.scene.socket = RigidObjectCfg(...)          # VERIFY
        # --- dual contact-force tactile (v1): one ContactSensor per fingertip ---
        # self.scene.tactile_left = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger", ...)   # VERIFY
        # self.scene.tactile_right = ContactSensorCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger", ...)  # VERIFY
        # --- cameras: wrist + scene RGB, ~224x224 to match the VLA front-end ---
        # self.scene.wrist_cam = CameraCfg(
        #     prim_path="{ENV_REGEX_NS}/Robot/panda_hand/cam", ...)     # VERIFY
        # self.scene.scene_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/front_cam", ...)  # VERIFY
        # --- observation terms must expose: wrist_cam, scene_cam, tactile_left/right, proprio ---
        self.decimation = 2           # VERIFY
        self.episode_length_s = 5.0   # VERIFY
