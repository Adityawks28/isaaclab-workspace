# Insertion + Tactile Data Factory (Plan A, `isaaclab-workspace`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an Isaac Lab task that records contact-rich **connector/peg insertion** demonstrations — RGB + **contact-force tactile (v1)** + proprio + force-aware actions — and writes them as a **LeRobot dataset** for the `robot-research` VLA stack (Plan B) to train on.

**Architecture:** A new external task plugin `source/insertion_tactile/` (same gymnasium-entry-point pattern as `lift_franka`/`reach_franka`). A **scripted oracle** drives insertion; a **recorder** serializes each step to LeRobot format. Two layers are split deliberately: Isaac-dependent glue (env cfg, sensors, actuation) lives in skeletons marked `# VERIFY` (run on the GPU box, mirroring `scripts/vla_play.py`), while the **oracle waypoint math** and **LeRobot frame assembly** are pure-Python modules with real unit tests (no Isaac needed).

**Tech Stack:** Isaac Sim 5.1 + Isaac Lab `main` (Docker), Python 3.10, `gymnasium`, `lerobot`, numpy, pytest.

## Global Constraints

- **Runs on the 16 GB Ubuntu box only** for anything importing `isaaclab`/`isaacsim`. The Mac dev session can only test the Isaac-independent modules (oracle math, frame assembly).
- **Fresh Isaac Sim 5.1 + Isaac Lab `main`** container, patterned on the existing `docker/container.sh` (Isaac Sim 4.5.0) but with new version pins. **Do not modify** the existing 4.5.0 setup — add alongside.
- **Never edit Isaac Lab source** — register tasks via the gymnasium entry point, like the existing tasks.
- **Tactile is v1 = contact-force** (force-torque field per finger, no gel image). Gel images are v2 (future).
- **Dataset contract (#1) — exact LeRobot feature keys** (must match `robot-research`'s `lerobot_loader.py`):
  `observation.images.wrist`, `observation.images.scene`, `observation.state` (proprio),
  `observation.tactile` (v1 force vector), `action` (`6 pose + 1 force = 7`), and a `task` string.
- **TDD for the testable modules; commit after every passing step.**

---

### Task 0: Isaac Sim 5.1 container (box-only smoke gate)

**Files:**
- Create: `docker/Dockerfile.tactile` (FROM `nvcr.io/nvidia/isaac-sim:5.1.0`, Isaac Lab `main`)
- Create: `docker/container_tactile.sh` (copy of `container.sh` with the new image/name)

**Interfaces:**
- Produces: a container in which `python -c "import isaaclab"` works and a headless sim can spawn a Franka. Manual gate; not CI.

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# docker/Dockerfile.tactile — fresh Isaac Sim 5.1 + Isaac Lab main for the tactile data factory.
# Built on the 16GB box; does NOT touch the existing 4.5.0 setup.
FROM nvcr.io/nvidia/isaac-sim:5.1.0
ARG ISAACLAB_REF=main
RUN git clone https://github.com/isaac-sim/IsaacLab.git /isaaclab \
 && cd /isaaclab && git checkout ${ISAACLAB_REF} \
 && ln -s /isaac-sim _isaac_sim && ./isaaclab.sh --install
ENV ACCEPT_EULA=Y PRIVACY_CONSENT=Y OMNI_KIT_ALLOW_ROOT=1
```

- [ ] **Step 2: Box smoke (manual)**

Run on the Ubuntu box:
```
docker build -f docker/Dockerfile.tactile -t isaac-tactile:5.1 .
docker run --rm --gpus all isaac-tactile:5.1 \
  /isaaclab/isaaclab.sh -p -c "import isaaclab; print('isaaclab OK')"
```
Expected: prints `isaaclab OK`. **NO-GO** → check the Isaac Lab `main` install against Isaac Sim 5.1
release notes; pin a compatible Isaac Lab tag instead of `main`.

- [ ] **Step 3: Commit**

```bash
git add docker/Dockerfile.tactile docker/container_tactile.sh
git commit -m "feat: Isaac Sim 5.1 + Isaac Lab main container for tactile data factory"
```

---

### Task 1: `insertion_tactile` plugin scaffold (box-verified)

**Files:**
- Create: `source/insertion_tactile/pyproject.toml`
- Create: `source/insertion_tactile/insertion_tactile/__init__.py`

**Interfaces:**
- Produces: gym id `Isaac-Insertion-Franka-Tactile-v0` registered via the `gymnasium.envs` entry point
  (string entry points only — no heavy imports at registration, per the `lift_franka` pattern).

- [ ] **Step 1: pyproject (mirror lift_franka)**

```toml
[project]
name = "insertion_tactile"
version = "0.1.0"
description = "Connector/peg insertion task with contact-force tactile, records LeRobot demos"
requires-python = ">=3.10"

[project.entry-points."gymnasium.envs"]
__root__ = "insertion_tactile:register"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: register() — string entry points only**

```python
# source/insertion_tactile/insertion_tactile/__init__.py
"""Registers Isaac-Insertion-Franka-Tactile-v0 as a gymnasium plugin.
Mirrors lift_franka: register() touches no isaaclab code (string entry points only),
resolved lazily at gym.make after the Isaac Sim app boots."""


def register():
    from gymnasium.envs.registration import register as gym_register, registry
    if "Isaac-Insertion-Franka-Tactile-v0" in registry:
        return
    gym_register(
        id="Isaac-Insertion-Franka-Tactile-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point":
                "insertion_tactile.insertion_env_cfg:FrankaInsertionTactileEnvCfg",
        },
    )
```

- [ ] **Step 3: Box smoke** — `./isaaclab.sh -p -c "import gymnasium, insertion_tactile; insertion_tactile.register(); print('registered')"` after `pip install -e source/insertion_tactile`.

- [ ] **Step 4: Commit**

```bash
git add source/insertion_tactile/pyproject.toml source/insertion_tactile/insertion_tactile/__init__.py
git commit -m "feat: insertion_tactile plugin scaffold + gym registration"
```

---

### Task 2: Insertion env-cfg skeleton — Franka + dual contact-force sensors + cameras (box)

**Files:**
- Create: `source/insertion_tactile/insertion_tactile/insertion_env_cfg.py`

**Interfaces:**
- Produces: `FrankaInsertionTactileEnvCfg` (a `ManagerBasedRLEnvCfg`) referenced by `register()`. Adds a
  socket + peg, two `ContactSensorCfg` on the gripper fingers, and wrist/scene `CameraCfg`s.
- This is an Isaac-dependent **skeleton** (marked `# VERIFY`, like `scripts/vla_play.py`); it is wired on
  the box, not tested on the Mac.

- [ ] **Step 1: Write the skeleton**

```python
# source/insertion_tactile/insertion_tactile/insertion_env_cfg.py
"""STRUCTURAL SKELETON (box-only). Franka peg-insertion env with contact-force tactile.
Every Isaac-specific line is # VERIFY against Isaac Lab main on Isaac Sim 5.1.
Base it on the Isaac Lab Factory insertion task (manipulation.factory) where possible."""
from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg          # VERIFY import path on Lab main
from isaaclab.sensors import ContactSensorCfg, CameraCfg  # VERIFY
from isaaclab.utils import configclass                   # VERIFY


@configclass
class FrankaInsertionTactileEnvCfg(ManagerBasedRLEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # --- scene: reuse a Factory peg/socket asset; add Franka with an IK-rel action space ---
        # self.scene.peg = RigidObjectCfg(...)            # VERIFY (Factory assets)
        # self.scene.socket = RigidObjectCfg(...)         # VERIFY
        # --- dual contact-force tactile (v1): one ContactSensor per fingertip ---
        # self.scene.tactile_left  = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_leftfinger",  ...)
        # self.scene.tactile_right = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_rightfinger", ...)
        # --- cameras: wrist + scene RGB, ~224x224 to match the VLA front-end ---
        # self.scene.wrist_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/Robot/panda_hand/cam", ...)
        # self.scene.scene_cam = CameraCfg(prim_path="{ENV_REGEX_NS}/front_cam", ...)
        self.decimation = 2          # VERIFY
        self.episode_length_s = 5.0  # VERIFY
```

- [ ] **Step 2: Box smoke** — `gym.make("Isaac-Insertion-Franka-Tactile-v0", ...)` boots and `env.reset()` returns
  obs containing the camera + contact-sensor keys. Record the actual obs keys for Task 4.

- [ ] **Step 3: Commit**

```bash
git add source/insertion_tactile/insertion_tactile/insertion_env_cfg.py
git commit -m "feat: insertion env-cfg skeleton (Franka + dual contact-force + cameras)"
```

---

### Task 3: Scripted oracle waypoint planner (PURE PYTHON — TDD, testable on the Mac)

**Files:**
- Create: `source/insertion_tactile/insertion_tactile/oracle.py`
- Test: `source/insertion_tactile/tests/test_oracle.py`

**Interfaces:**
- Produces: `Waypoint` dataclass `(target_pos: np.ndarray[3], target_force: float, phase: str)`;
  `plan_insertion(socket_pos, approach_height, insert_depth, insert_force, n_steps) -> list[Waypoint]`.
  Top-down insertion: approach above socket (force 0) → descend → insert to `insert_depth` below the
  socket top with `insert_force`. Pure numpy; no Isaac.

- [ ] **Step 1: Write the failing test**

```python
# source/insertion_tactile/tests/test_oracle.py
import numpy as np
from insertion_tactile.oracle import plan_insertion, Waypoint


def test_starts_above_socket_no_force():
    wps = plan_insertion(np.zeros(3), approach_height=0.1, insert_depth=0.04,
                         insert_force=5.0, n_steps=20)
    assert isinstance(wps[0], Waypoint)
    assert wps[0].target_pos[2] == 0.1          # approach height above socket z=0
    assert wps[0].target_force == 0.0


def test_ends_inserted_with_force():
    wps = plan_insertion(np.zeros(3), approach_height=0.1, insert_depth=0.04,
                         insert_force=5.0, n_steps=20)
    assert np.isclose(wps[-1].target_pos[2], -0.04)   # insert_depth below socket top
    assert wps[-1].target_force == 5.0


def test_descent_is_monotonic():
    wps = plan_insertion(np.zeros(3), approach_height=0.1, insert_depth=0.04,
                         insert_force=5.0, n_steps=20)
    zs = [w.target_pos[2] for w in wps]
    assert all(zs[i] >= zs[i + 1] - 1e-9 for i in range(len(zs) - 1))


def test_force_only_during_insertion():
    wps = plan_insertion(np.zeros(3), approach_height=0.1, insert_depth=0.04,
                         insert_force=5.0, n_steps=20)
    # any waypoint at or below socket top (z<=0) is the insertion phase -> force on
    for w in wps:
        if w.target_pos[2] < -1e-9:
            assert w.target_force == 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest source/insertion_tactile/tests/test_oracle.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'insertion_tactile.oracle'`

- [ ] **Step 3: Write minimal implementation**

```python
# source/insertion_tactile/insertion_tactile/oracle.py
"""Pure-Python scripted insertion oracle. No Isaac imports -> unit-testable.
Produces a top-down insertion trajectory of target EE positions + target contact forces.
The Isaac actuation glue (Task 5) maps each Waypoint to the env's IK + force action."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Waypoint:
    target_pos: np.ndarray   # [3] xyz, top-down orientation assumed fixed
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest source/insertion_tactile/tests/test_oracle.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add source/insertion_tactile/insertion_tactile/oracle.py source/insertion_tactile/tests/test_oracle.py
git commit -m "feat: scripted insertion oracle waypoint planner (pure python, tested)"
```

---

### Task 4: LeRobot recorder — frame assembly (TDD) + dataset writer (box)

**Files:**
- Create: `source/insertion_tactile/insertion_tactile/recorder.py`
- Test: `source/insertion_tactile/tests/test_recorder.py`

**Interfaces:**
- Produces: `assemble_frame(rgb_wrist, rgb_scene, tactile_force, proprio, action, language) -> dict`
  returning a LeRobot frame with the **exact contract #1 keys**; raises `ValueError` on a wrong-length
  action (`!= 7`). Plus `write_lerobot_dataset(frames, repo_id, fps)` (box-only; lazy-imports `lerobot`).

- [ ] **Step 1: Write the failing test**

```python
# source/insertion_tactile/tests/test_recorder.py
import numpy as np
import pytest
from insertion_tactile.recorder import assemble_frame


def _frame():
    return assemble_frame(
        rgb_wrist=np.zeros((224, 224, 3), np.uint8),
        rgb_scene=np.zeros((224, 224, 3), np.uint8),
        tactile_force=np.zeros(6, np.float32),
        proprio=np.zeros(14, np.float32),
        action=np.zeros(7, np.float32),
        language="insert the connector into the socket",
    )


def test_frame_has_contract_keys():
    f = _frame()
    for k in ("observation.images.wrist", "observation.images.scene",
              "observation.state", "observation.tactile", "action", "task"):
        assert k in f


def test_action_length_validated():
    with pytest.raises(ValueError):
        assemble_frame(np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1, 3), np.uint8),
                       np.zeros(6, np.float32), np.zeros(14, np.float32),
                       np.zeros(5, np.float32), "x")   # action len 5 != 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest source/insertion_tactile/tests/test_recorder.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'insertion_tactile.recorder'`

- [ ] **Step 3: Write minimal implementation**

```python
# source/insertion_tactile/insertion_tactile/recorder.py
"""Serializes insertion demos to a LeRobot dataset matching contract #1.
assemble_frame() is pure-python (testable); write_lerobot_dataset() lazy-imports lerobot
and runs on the box."""
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


def write_lerobot_dataset(frames: list[dict], repo_id: str, fps: int = 10):
    """Box-only. Writes frames (grouped into episodes by 'episode_index' if present)
    to a LeRobot v2 dataset. Lazy-imports lerobot."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # lazy  # VERIFY API
    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=_features_from(frames[0]))
    for fr in frames:
        ds.add_frame(fr)            # VERIFY add_frame/save_episode API on installed lerobot
    ds.consolidate()               # VERIFY
    return ds


def _features_from(frame: dict) -> dict:
    feats = {}
    for k, v in frame.items():
        if k == "task":
            continue
        feats[k] = {"dtype": str(np.asarray(v).dtype), "shape": list(np.asarray(v).shape)}
    return feats
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest source/insertion_tactile/tests/test_recorder.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add source/insertion_tactile/insertion_tactile/recorder.py source/insertion_tactile/tests/test_recorder.py
git commit -m "feat: LeRobot recorder — frame assembly (tested) + dataset writer"
```

---

### Task 5: Dataset-generation script (box-only) — wire oracle → env → recorder

**Files:**
- Create: `scripts/generate_insertion_dataset.py`

**Interfaces:**
- Consumes: `plan_insertion` (Task 3), `assemble_frame`/`write_lerobot_dataset` (Task 4), the env (Task 2).
- Produces: a LeRobot dataset on disk. Isaac-dependent **skeleton** (`# VERIFY`), run on the box.

- [ ] **Step 1: Write the skeleton**

```python
# scripts/generate_insertion_dataset.py
"""STRUCTURAL SKELETON (box-only). Rolls out the scripted oracle in the insertion env and
records LeRobot demos. Mirrors scripts/vla_play.py's AppLauncher-first structure."""
from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Insertion-Franka-Tactile-v0")
parser.add_argument("--episodes", type=int, default=150)
parser.add_argument("--repo_id", default="adityawks28/insertion_tactile_v1")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np

from insertion_tactile.oracle import plan_insertion
from insertion_tactile.recorder import assemble_frame, write_lerobot_dataset
from isaaclab_tasks.utils import parse_env_cfg


def socket_pose_from_obs(obs):           # VERIFY: pull socket xyz from the env obs/state
    return np.zeros(3)


def step_action_from_waypoint(wp, obs):  # VERIFY: map Waypoint -> env IK + force action tensor
    raise NotImplementedError


def main():
    env_cfg = parse_env_cfg(args.task, num_envs=1)
    env = gym.make(args.task, cfg=env_cfg)
    frames, ep = [], 0
    obs, _ = env.reset()
    while ep < args.episodes:
        wps = plan_insertion(socket_pose_from_obs(obs), approach_height=0.1,
                             insert_depth=0.04, insert_force=5.0, n_steps=50)
        for wp in wps:
            action = step_action_from_waypoint(wp, obs)        # VERIFY
            obs, _, term, trunc, _ = env.step(action)
            fr = assemble_frame(
                rgb_wrist=obs["policy"]["wrist_cam"],           # VERIFY obs keys (from Task 2)
                rgb_scene=obs["policy"]["scene_cam"],           # VERIFY
                tactile_force=obs["policy"]["tactile_left"],    # VERIFY (concat L/R if desired)
                proprio=obs["policy"]["proprio"],               # VERIFY
                action=action.cpu().numpy().reshape(-1)[:7],
                language="insert the connector into the socket",
            )
            fr["episode_index"] = ep
            frames.append(fr)
            if term or trunc:
                break
        obs, _ = env.reset(); ep += 1
    write_lerobot_dataset(frames, repo_id=args.repo_id, fps=10)
    env.close(); app.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Box run** — `./isaaclab.sh -p scripts/generate_insertion_dataset.py --episodes 150`.
  Produces the LeRobot dataset. Fix every `# VERIFY` against the real obs keys recorded in Task 2.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_insertion_dataset.py
git commit -m "feat: oracle->env->recorder dataset generation script (box skeleton)"
```

---

### Task 6: Cross-repo validation (box) — `robot-research` loads the dataset

**Files:** none new — uses `robot-research/src/tactvla/data/lerobot_loader.py`.

- [ ] **Step 1: Box validation**

```bash
# in the robot-research CUDA container, with the generated dataset mounted:
python -c "
from tactvla.data.lerobot_loader import load_lerobot_episodes
eps = load_lerobot_episodes('adityawks28/insertion_tactile_v1', tactile_key='observation.tactile')
print('episodes:', len(eps), 'first ep steps:', len(eps[0]))
print('action dim:', eps[0][0].action.shape, 'tactile dim:', eps[0][0].tactile.shape)
"
```
Expected: prints episode counts; `action dim = (7,)`, `tactile dim = (6,)`. This closes the loop:
Plan A's dataset feeds Plan B's training, replacing the synthetic data.

- [ ] **Step 2: (no commit — validation only)**

---

## Self-Review

**Spec coverage (§5):** Dockerized Isaac 5.1 → Task 0. Insertion task → Tasks 1-2. Dual contact-force
tactile (v1) → Task 2. Scripted oracle → Task 3. LeRobot recorder + contract #1 keys → Task 4. Dataset
generation → Task 5. Feeds Plan B (cross-repo contract) → Task 6. Gel-image tactile (v2) and
closed-loop are out of scope (future), consistent with the spec.

**Placeholder scan:** Isaac-dependent tasks (0,1,2,5) are explicit `# VERIFY` skeletons run on the box —
this matches the repo's existing convention (`scripts/vla_play.py`) for code that can't be CI'd without
the GPU. They are not vague placeholders: each names the exact APIs/keys to confirm. The testable
modules (Tasks 3,4) have complete code + real tests.

**Type/key consistency:** contract #1 keys (`observation.images.wrist`, `observation.images.scene`,
`observation.state`, `observation.tactile`, `action`, `task`) are identical in Task 4 (writer) and
Task 6 (`robot-research` loader, `tactile_key="observation.tactile"`). `action` length 7 (6 pose + 1
force) is consistent with Plan B's `pose_dim=6, force_dim=1`. `plan_insertion` signature matches between
Task 3 and Task 5.
