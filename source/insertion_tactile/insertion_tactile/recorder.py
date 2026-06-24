"""Serializes insertion demos to a LeRobot dataset matching contract #1.

assemble_frame() is pure-python (unit-testable); write_lerobot_dataset() lazy-imports lerobot
and runs on the box. Feature keys MUST match robot-research/src/tactvla/data/lerobot_loader.py."""
import numpy as np

ACTION_DIM = 7  # 6 pose + 1 force


def assemble_frame(rgb_wrist, rgb_scene, tactile_force, proprio, action, language: str) -> dict:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != ACTION_DIM:
        raise ValueError(f"action must have {ACTION_DIM} dims, got {action.shape[-1]}")
    return {
        "observation.images.wrist": np.asarray(rgb_wrist, np.uint8),
        "observation.images.scene": np.asarray(rgb_scene, np.uint8),
        "observation.state": np.asarray(proprio, np.float32),
        "observation.tactile": np.asarray(tactile_force, np.float32),
        "action": action,
        "task": language,
    }


def _features_from(frame: dict) -> dict:
    feats = {}
    for k, v in frame.items():
        if k in ("task", "episode_index"):
            continue
        arr = np.asarray(v)
        feats[k] = {"dtype": str(arr.dtype), "shape": list(arr.shape)}
    return feats


def write_lerobot_dataset(frames: list[dict], repo_id: str, fps: int = 10):
    """Box-only. Writes frames to a LeRobot v2 dataset. Lazy-imports lerobot."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # lazy  # VERIFY API
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=_features_from(frames[0]))
    for fr in frames:
        ds.add_frame(fr)            # VERIFY add_frame / save_episode API on installed lerobot
    ds.consolidate()               # VERIFY
    return ds
