"""Franka peg-insertion env with contact-force tactile (v1) + wrist/scene RGB cameras.

Strategy: subclass Isaac Lab's *proven* Franka IK-Abs Lift env and ADD sensors to the scene:
  - two ContactSensors (one per fingertip) -> contact-force tactile (v1),
  - wrist + scene PinholeCameras (224x224 RGB) -> VLA front-end.
The manipulated DexCube plays the role of the "peg"; the commanded goal pose plays the role of
the "socket" target. Sensors live in the scene so the recorder reads RGB + contact force directly
(env.unwrapped.scene["wrist_cam"], ["tactile_left"], ...), which avoids depending on exact obs-term
APIs and keeps the env bootable. Verified on Isaac Sim 5.1 + Isaac Lab main.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg, ContactSensorCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.ik_abs_env_cfg import FrankaCubeLiftEnvCfg


@configclass
class FrankaInsertionTactileEnvCfg(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # build the proven Franka IK-Abs lift scene first
        super().__post_init__()

        # ContactSensors on an articulation require contact reporting on the robot prim.
        self.scene.robot.spawn.activate_contact_sensors = True

        # --- dual contact-force tactile (v1): one ContactSensor per fingertip ---
        self.scene.tactile_left = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )
        self.scene.tactile_right = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger",
            update_period=0.0,
            history_length=1,
            debug_vis=False,
        )

        # --- cameras: wrist (on the hand) + scene, 224x224 RGB to match the VLA front-end ---
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 10.0)),
            offset=CameraCfg.OffsetCfg(pos=(0.07, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        )
        self.scene.scene_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/scene_cam",
            update_period=0.0,
            height=224,
            width=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 20.0)),
            offset=CameraCfg.OffsetCfg(pos=(1.4, 0.0, 0.9), rot=(0.35, 0.0, 0.0, 0.94), convention="world"),
        )

        # cameras add render cost; keep demo episodes a touch longer than lift
        self.decimation = 2
        self.episode_length_s = 6.0
