"""Pure-Python scripted insertion oracle. No Isaac imports -> unit-testable.

Produces a top-down insertion trajectory of target EE positions + target contact forces.
The Isaac actuation glue (scripts/generate_insertion_dataset.py) maps each Waypoint to the
env's IK + force action."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Waypoint:
    target_pos: np.ndarray   # [3] xyz; top-down orientation assumed fixed
    target_force: float      # desired downward contact force (N); 0 before contact
    phase: str               # "approach" | "insert"


def plan_insertion(socket_pos, approach_height: float, insert_depth: float,
                   insert_force: float, n_steps: int) -> list[Waypoint]:
    socket_pos = np.asarray(socket_pos, dtype=float)
    z_top = socket_pos[2]
    z_start = z_top + approach_height
    z_end = z_top - insert_depth
    wps = []
    for i in range(n_steps):
        z = z_start + (z_end - z_start) * (i / (n_steps - 1))
        inserting = z < z_top - 1e-9
        wps.append(Waypoint(
            target_pos=np.array([socket_pos[0], socket_pos[1], z]),
            target_force=insert_force if inserting else 0.0,
            phase="insert" if inserting else "approach",
        ))
    return wps
