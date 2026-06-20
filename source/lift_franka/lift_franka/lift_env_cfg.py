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

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    FrankaCubeLiftEnvCfg,
)

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
