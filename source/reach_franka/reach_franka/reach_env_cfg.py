"""Custom Franka Reach env: built-in Reach with a heavier end-effector tracking reward."""

from isaaclab.utils import configclass

# Built-in Franka Reach joint-position env cfg (verified against Isaac Lab v2.1.0).
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (
    FrankaReachEnvCfg,
)


@configclass
class FrankaReachCustomEnvCfg(FrankaReachEnvCfg):
    """Identical to the built-in task, but we boost position-tracking reward.

    Editing reward weights here and re-training is the core learning loop.
    The base weight is negative (a distance penalty), so multiplying keeps the
    sign and makes the end-effector tracking term count for more.
    """

    def __post_init__(self):
        super().__post_init__()
        self.rewards.end_effector_position_tracking.weight *= 2.0
