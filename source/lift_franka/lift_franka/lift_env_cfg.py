"""Custom Franka Lift env: built-in pick-and-place with a higher lift threshold.

The built-in task counts the cube as "lifted" once it clears 0.04 m. We raise
that to 0.08 m so the policy must lift the cube visibly higher before the gate
opens. Because three reward terms gate on ``minimal_height`` (the lift bonus and
both goal-tracking terms), we update all three so the staging stays consistent —
otherwise goal-tracking would unlock at the old height while the lift bonus
required the new one. See docs/learning/09-staged-sparse-reward.html (the gate
1[h > h_min]) and 11-lift-task-isaaclab.html.

Verified against Isaac Lab v2.1.0:
  - base cfg: ...manipulation.lift.config.franka.joint_pos_env_cfg:FrankaCubeLiftEnvCfg
  - reward terms: lifting_object, object_goal_tracking, object_goal_tracking_fine_grained
"""

from isaaclab.utils import configclass

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import (
    CurriculumCfg as BaseLiftCurriculumCfg,
)

from . import curriculums as custom_curr

# The new lift threshold (metres). Built-in default is 0.04.
NEW_MINIMAL_HEIGHT = 0.08


@configclass
class FrankaCubeLiftCustomEnvCfg(FrankaCubeLiftEnvCfg):
    """Identical to the built-in Franka Lift, but the cube must be lifted higher.

    Editing the staging threshold and re-training is the core learning loop for
    this milestone: a higher gate makes the lift sub-task harder and the reward
    landscape sparser, so we expect slower convergence and a clearer separation
    between "reached the cube" and "actually lifted it".
    """

    def __post_init__(self):
        super().__post_init__()
        # Raise the gate consistently across every term that references it.
        self.rewards.lifting_object.params["minimal_height"] = NEW_MINIMAL_HEIGHT
        self.rewards.object_goal_tracking.params["minimal_height"] = NEW_MINIMAL_HEIGHT
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = NEW_MINIMAL_HEIGHT

        # Closer camera for eval/recording. The built-in default (eye 7.5,7.5,7.5)
        # frames the whole scene and renders the arm tiny; this zooms onto the
        # workspace (robot base at the origin, cube ~[0.5, 0, 0.05], goal in front).
        # Viewer-only: affects rendering, not training, physics, or the reward gate.
        self.viewer.eye = (1.8, 1.8, 1.2)
        self.viewer.lookat = (0.35, 0.0, 0.15)


@configclass
class LiftRecedingGoalCurriculumCfg(BaseLiftCurriculumCfg):
    """Built-in penalty curriculum (action_rate, joint_vel) + a receding goal.

    Two staged widenings of the object-pose goal range. Step thresholds are in
    env steps; one rsl_rl iteration ≈ num_steps_per_env (24) env steps, so for a
    4000-iter run (~96k steps): easy until ~1040 iters, mid until ~2290, then full.
    """

    goal_range_mid = CurrTerm(
        func=custom_curr.modify_command_pose_range,
        params={
            "command_name": "object_pose",
            "pos_x": (0.42, 0.58),
            "pos_y": (-0.16, 0.16),
            "pos_z": (0.12, 0.32),
            "num_steps": 25000,
        },
    )
    goal_range_full = CurrTerm(
        func=custom_curr.modify_command_pose_range,
        params={
            "command_name": "object_pose",
            "pos_x": (0.4, 0.6),
            "pos_y": (-0.25, 0.25),
            "pos_z": (0.25, 0.5),  # the built-in full range
            "num_steps": 55000,
        },
    )


@configclass
class FrankaCubeLiftCurriculumEnvCfg(FrankaCubeLiftEnvCfg):
    """Stock 0.04 gate + a receding-goal curriculum (the winning gate from the sweep).

    Targets the carry-to-goal failure the sweep/ablation exposed: the policy lifts
    but rarely carries. We start the goal close and low so a small lift already earns
    goal-tracking reward (coupling lift->carry early), then widen the goal range in
    stages to the full task. See lift_franka/curriculums.py and Ch. 10.
    """

    def __post_init__(self):
        super().__post_init__()
        # Stage 0 (easy): goal close to the cube's start [0.5, 0, ~0.05] and low,
        # so a small lift + tiny carry already scores goal-tracking reward.
        self.commands.object_pose.ranges.pos_x = (0.45, 0.55)
        self.commands.object_pose.ranges.pos_y = (-0.08, 0.08)
        self.commands.object_pose.ranges.pos_z = (0.08, 0.16)
        # Widen the goal range in stages as the policy improves.
        self.curriculum = LiftRecedingGoalCurriculumCfg()
        # Zoomed camera for eval/recording (same as the custom task).
        self.viewer.eye = (1.8, 1.8, 1.2)
        self.viewer.lookat = (0.35, 0.0, 0.15)
