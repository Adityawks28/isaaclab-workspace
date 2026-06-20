"""Custom curriculum term: progressively widen the goal-pose command range.

A "receding goal" curriculum for the Lift task (docs/learning/10-exploration.html).
The sweep + ablation showed the policy learns to lift but rarely *carries* the cube
to the goal, and that more iterations don't help, a structural/exploration problem,
not a duration one. So we make the carry easy first: start with the goal close to the
object and low (a small lift is enough to score goal-tracking reward), then widen the
command range in stages as the policy improves, until it covers the full task.

This works because UniformPoseCommand reads ``self.cfg.ranges.*`` *live* at each
resample (isaaclab/envs/mdp/commands/pose_command.py), so mutating the ranges here
takes effect on the next goal resampling, no env rebuild needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_command_pose_range(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    command_name: str,
    pos_x: tuple[float, float],
    pos_y: tuple[float, float],
    pos_z: tuple[float, float],
    num_steps: int,
):
    """Once training passes ``num_steps`` env steps, widen the goal command range.

    Args:
        command_name: name of the pose command term (e.g. "object_pose").
        pos_x/pos_y/pos_z: the new sampling ranges to apply (in the command frame).
        num_steps: env-step threshold after which the new ranges take effect.
    """
    if env.common_step_counter > num_steps:
        ranges = env.command_manager.get_term(command_name).cfg.ranges
        ranges.pos_x = tuple(pos_x)
        ranges.pos_y = tuple(pos_y)
        ranges.pos_z = tuple(pos_z)
