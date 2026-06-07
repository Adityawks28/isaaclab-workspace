# Isaac Lab Workspace — Franka Reach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up a fully-Dockerized Isaac Lab workspace and train + watch a Franka arm on the Reach task, then fork it into an editable custom task — as a guided learning project.

**Architecture:** Approach A — clone Isaac Lab (pinned **v2.1.0**) and use its official Docker tooling to build an image **FROM** `nvcr.io/nvidia/isaac-sim:4.5.0`, reusing the warm `~/docker/isaac-sim` caches. Our repo (`isaaclab-workspace`) is a clean GitHub-shippable wrapper: setup/train/eval scripts + an external custom task. A private, gitignored learning layer sits alongside.

**Tech Stack:** Docker + NVIDIA Container Toolkit, Isaac Sim 4.5.0, Isaac Lab v2.1.0, Python 3.10, PyTorch, rsl_rl (PPO), TensorBoard. Hardware: laptop RTX 4050 (6 GB) for light runs / GUI eval; workstation for heavy headless training; Tailscale links them.

---

## A note on "tests" for this plan

This project is infrastructure- and ML-heavy, so most verification is **smoke-test commands with expected output** (build the image, list registered tasks, confirm training emits a reward and a checkpoint) rather than `pytest` units. The one piece of pure Python — the custom task registration (Task 6) — gets a real importable smoke test. Every task still ends in a commit.

**Commit style (project rule):** one-line [Conventional Commits](https://www.conventionalcommits.org) messages (`feat:`, `fix:`, `docs:`, `chore:`), committed under the user's personal account. **No `Co-Authored-By` / attribution trailers, ever.**

**Version caveat:** Isaac Lab import paths and script locations shift between releases. Commands below target **v2.1.0**. Where a path is version-sensitive it is flagged "VERIFY"; confirm against the cloned tag before relying on it.

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/setup.sh` | Clone+pin Isaac Lab into `_isaaclab/`, build the Docker image, install our custom task. One-shot per machine. |
| `scripts/train.sh` | Wrapper → launch headless rsl_rl training for a given `--task` and `--num_envs`. |
| `scripts/eval.sh` | Wrapper → launch GUI play of a checkpoint for a given `--task`. |
| `scripts/_common.sh` | Shared paths/vars sourced by the other scripts (repo root, Isaac Lab dir, container name). |
| `source/reach_franka/` | Our external Isaac Lab task: registers `Isaac-Reach-Franka-Custom-v0`, subclasses the built-in Franka Reach cfg, tweaks one reward term. Pip-installable. |
| `requirements.txt` | Extra host-side Python deps (minimal; Isaac Lab brings the heavy ones). |
| `README.md` | Public setup/run instructions. |
| `docker/container.sh` | Existing plain-Isaac-Sim launcher (already present; untouched here). |
| `LEARNING.md`, `notes/`, `CLAUDE.md` | **Private** (gitignored) learning layer. |
| `_isaaclab/`, `outputs/` | Gitignored: Isaac Lab clone + generated checkpoints/logs. |

---

## Task 1: Repo scaffold

**Files:**
- Create: `scripts/.gitkeep`, `source/.gitkeep`, `requirements.txt`, `README.md`

- [ ] **Step 1: Create the directory skeleton**

```bash
cd ~/isaaclab-workspace
mkdir -p scripts source
touch scripts/.gitkeep source/.gitkeep
```

- [ ] **Step 2: Write `requirements.txt`** (host-side extras only)

```text
# Isaac Lab + its RL libs are installed inside the Docker image, not here.
# This file holds only light host-side tooling for the custom task code.
tensorboard
```

- [ ] **Step 3: Write a minimal `README.md` stub** (expanded in Task 9)

```markdown
# isaaclab-workspace

A fully-Dockerized [Isaac Lab](https://github.com/isaac-sim/IsaacLab) workspace
for robot reinforcement learning. First milestone: Franka **Reach**.

Built on Isaac Sim 4.5.0 + Isaac Lab v2.1.0. See `docs/superpowers/specs/` for the design.

## Quick start
```bash
./scripts/setup.sh      # clone + pin Isaac Lab, build the image
./scripts/train.sh      # headless training
./scripts/eval.sh       # watch the trained policy (GUI)
```
```

- [ ] **Step 4: Verify the tree**

Run: `cd ~/isaaclab-workspace && git status --short`
Expected: untracked `README.md`, `requirements.txt`, `scripts/`, `source/`.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore: scaffold scripts, source, requirements, README"
```

---

## Task 2: `scripts/setup.sh` — clone Isaac Lab + build the image

**Files:**
- Create: `scripts/_common.sh`, `scripts/setup.sh`

- [ ] **Step 1: Write `scripts/_common.sh`** (shared vars)

```bash
#!/bin/bash
# Shared paths/vars for the workspace scripts.
COMMON_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$( cd "${COMMON_DIR}/.." && pwd )"
ISAACLAB_DIR="${REPO_DIR}/_isaaclab"
ISAACLAB_TAG="v2.1.0"                 # pinned for Isaac Sim 4.5.0 compatibility
ISAACSIM_BASE_IMAGE="nvcr.io/nvidia/isaac-sim:4.5.0"
```

- [ ] **Step 2: Write `scripts/setup.sh`**

```bash
#!/bin/bash
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_common.sh"

# 1. Clone Isaac Lab pinned to a 4.5-compatible tag (idempotent).
if [ ! -d "${ISAACLAB_DIR}/.git" ]; then
    echo "Cloning Isaac Lab ${ISAACLAB_TAG}..."
    git clone --depth 1 --branch "${ISAACLAB_TAG}" \
        https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
else
    echo "Isaac Lab already cloned at ${ISAACLAB_DIR}."
fi

# 2. Confirm the base image is present (pull if missing).
if ! docker image inspect "${ISAACSIM_BASE_IMAGE}" >/dev/null 2>&1; then
    echo "Pulling base image ${ISAACSIM_BASE_IMAGE} (large)..."
    docker pull "${ISAACSIM_BASE_IMAGE}"
fi

# 3. Build the Isaac Lab image via its official Docker tooling.
#    VERIFY: container.py path/flags against the cloned tag.
echo "Building Isaac Lab image (first build is slow)..."
python3 "${ISAACLAB_DIR}/docker/container.py" start

echo "Setup complete. Enter with: python3 ${ISAACLAB_DIR}/docker/container.py enter"
```

- [ ] **Step 3: Make scripts executable**

```bash
chmod +x scripts/setup.sh scripts/_common.sh
```

- [ ] **Step 4: VERIFY Isaac Lab's container tooling interface before first run**

Run: `git clone --depth 1 --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git _isaaclab && ls _isaaclab/docker/`
Expected: contains `container.py` and `.env.base` (or similar). Confirm `container.py start` is the build+launch verb and that `.env.base` references `nvcr.io/nvidia/isaac-sim:4.5.0`. If the base image var differs, set it in `_isaaclab/docker/.env.base` before building.

- [ ] **Step 5: Run setup end-to-end**

Run: `./scripts/setup.sh`
Expected: Isaac Lab clones; image builds; finishes with the "Setup complete" line. (First build pulls/builds many GB — minutes to tens of minutes.)

- [ ] **Step 6: Commit**

```bash
git add scripts/_common.sh scripts/setup.sh
git commit -m "feat: add setup.sh to clone Isaac Lab and build the Docker image"
```

---

## Task 3: Smoke-test Isaac Lab + confirm Franka Reach is registered

**Files:** none created — this task verifies the environment and records findings.

- [ ] **Step 1: Enter the Isaac Lab container**

Run: `python3 _isaaclab/docker/container.py enter`
Expected: a shell inside the `isaac-lab-base` container, at `/workspace/isaaclab`.

- [ ] **Step 2: Confirm Isaac Lab imports and Reach is registered**

Run (inside container):
```bash
./isaaclab.sh -p -c "import gymnasium as gym; import isaaclab_tasks; \
print([e for e in gym.registry if 'Reach-Franka' in e])"
```
Expected: a non-empty list containing `Isaac-Reach-Franka-v0`.
VERIFY: in v2.1.0 the tasks package is `isaaclab_tasks`; if import fails, find the correct name with `./isaaclab.sh -p -c "import pkgutil; print([m.name for m in pkgutil.iter_modules() if 'task' in m.name])"` and record it in `notes/`.

- [ ] **Step 3: Record the confirmed import paths**

Write the confirmed package name, train/play script paths, and Franka Reach env-cfg path into `notes/00-isaaclab-paths.md` (created in Task 8; for now keep them in the task output / scratch).

- [ ] **Step 4: Commit** (nothing to commit — verification only)

Skip commit; proceed to Task 4. (Findings get committed via `notes/` only if you choose to ship them — by default `notes/` is gitignored.)

---

## Task 4: `scripts/train.sh` — headless training of built-in Reach

**Files:**
- Create: `scripts/train.sh`

- [ ] **Step 1: Write `scripts/train.sh`**

```bash
#!/bin/bash
# Launch headless rsl_rl PPO training inside the Isaac Lab container.
# Usage: ./scripts/train.sh [--task <id>] [--num_envs <n>] [extra args...]
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_common.sh"

TASK="Isaac-Reach-Franka-v0"
NUM_ENVS="256"                        # laptop-safe default (6 GB); raise on workstation
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --num_envs) NUM_ENVS="$2"; shift 2;;
    *) EXTRA+=("$1"); shift;;
  esac
done

# VERIFY: rsl_rl train script path for the pinned tag.
TRAIN_PY="scripts/reinforcement_learning/rsl_rl/train.py"
python3 "${ISAACLAB_DIR}/docker/container.py" enter <<EOF
./isaaclab.sh -p ${TRAIN_PY} --task ${TASK} --num_envs ${NUM_ENVS} --headless ${EXTRA[@]}
EOF
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/train.sh
```

- [ ] **Step 3: VERIFY the train script path inside the container**

Run: `python3 _isaaclab/docker/container.py enter` then `ls scripts/reinforcement_learning/rsl_rl/`
Expected: `train.py` and `play.py` present. If the path differs in v2.1.0, update `TRAIN_PY` in `train.sh` and the `eval.sh` equivalent in Task 5.

- [ ] **Step 4: Run a short training smoke test**

Run: `./scripts/train.sh --num_envs 256 --max_iterations 10`
Expected: training starts, prints per-iteration mean reward, and writes a run dir under `_isaaclab/logs/rsl_rl/reach_franka/<timestamp>/` containing a `.pt` checkpoint. (VERIFY log dir; mirror it to `outputs/` in Task 7 wiring if desired.)

- [ ] **Step 5: Commit**

```bash
git add scripts/train.sh
git commit -m "feat: add train.sh for headless rsl_rl Reach training"
```

---

## Task 5: `scripts/eval.sh` — GUI playback of a checkpoint

**Files:**
- Create: `scripts/eval.sh`

- [ ] **Step 1: Write `scripts/eval.sh`**

```bash
#!/bin/bash
# Watch a trained policy in the Isaac Lab GUI (X11).
# Usage: ./scripts/eval.sh [--task <id>] [--num_envs <n>] [--checkpoint <path>]
set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_common.sh"

TASK="Isaac-Reach-Franka-v0"
NUM_ENVS="16"
CKPT=""                               # empty → rsl_rl picks the latest run
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task) TASK="$2"; shift 2;;
    --num_envs) NUM_ENVS="$2"; shift 2;;
    --checkpoint) CKPT="--checkpoint $2"; shift 2;;
    *) shift;;
  esac
done

xhost +local:root >/dev/null 2>&1 || true
PLAY_PY="scripts/reinforcement_learning/rsl_rl/play.py"
python3 "${ISAACLAB_DIR}/docker/container.py" enter <<EOF
./isaaclab.sh -p ${PLAY_PY} --task ${TASK} --num_envs ${NUM_ENVS} ${CKPT}
EOF
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/eval.sh
```

- [ ] **Step 3: Run eval on the smoke-trained checkpoint**

Run: `./scripts/eval.sh --num_envs 16`
Expected: Isaac Lab GUI window opens; 16 Franka arms move their end-effectors toward target markers. (After only 10 training iterations motion will be poor — that's fine; we're verifying the eval path, not policy quality.)

- [ ] **Step 4: Commit**

```bash
git add scripts/eval.sh
git commit -m "feat: add eval.sh for GUI playback of trained policies"
```

---

## Task 6: Custom external task `source/reach_franka`

**Files:**
- Create: `source/reach_franka/pyproject.toml`
- Create: `source/reach_franka/reach_franka/__init__.py`
- Create: `source/reach_franka/reach_franka/reach_env_cfg.py`

- [ ] **Step 1: Write `source/reach_franka/pyproject.toml`** (editable-installable package)

```toml
[project]
name = "reach_franka"
version = "0.1.0"
description = "Custom Franka Reach task for isaaclab-workspace"
requires-python = ">=3.10"

[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
```

- [ ] **Step 2: Write `source/reach_franka/reach_franka/reach_env_cfg.py`**

Subclass the built-in Franka Reach cfg and change one reward weight so behavior visibly differs.
VERIFY the import path against v2.1.0 (Step 5).

```python
"""Custom Franka Reach env: built-in Reach with a heavier end-effector tracking reward."""
from isaaclab.utils import configclass
# VERIFY path: built-in Franka Reach joint-position env cfg.
from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import (
    FrankaReachEnvCfg,
)


@configclass
class FrankaReachCustomEnvCfg(FrankaReachEnvCfg):
    """Identical to the built-in task, but we boost position-tracking reward.

    Editing reward weights here and re-training is the core learning loop.
    """

    def __post_init__(self):
        super().__post_init__()
        # VERIFY the reward term name in v2.1.0 (e.g. end_effector_position_tracking).
        self.rewards.end_effector_position_tracking.weight *= 2.0
```

- [ ] **Step 3: Write `source/reach_franka/reach_franka/__init__.py`** (registers the gym id on import)

```python
"""Registers Isaac-Reach-Franka-Custom-v0 when this package is imported."""
import gymnasium as gym

from .reach_env_cfg import FrankaReachCustomEnvCfg

gym.register(
    id="Isaac-Reach-Franka-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",  # VERIFY entry point for v2.1.0
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": FrankaReachCustomEnvCfg,
        # Reuse the built-in Franka rsl_rl PPO runner cfg. VERIFY this import path.
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.manager_based.manipulation.reach.config.franka."
            "agents.rsl_rl_ppo_cfg:ReachPPORunnerCfg"
        ),
    },
)
```

- [ ] **Step 4: Install the package into the container**

Add to `scripts/setup.sh` (after the build step), then re-run setup:
```bash
# 4. Install our custom task into the container's python (editable).
python3 "${ISAACLAB_DIR}/docker/container.py" enter <<'EOF'
./isaaclab.sh -p -m pip install -e /workspace/isaaclab-workspace/source/reach_franka
EOF
```
VERIFY the mounted path of our repo inside the container (Isaac Lab's container mounts the Isaac Lab repo at `/workspace/isaaclab`; our repo must also be mounted — if it is not, add a bind mount of `${REPO_DIR}:/workspace/isaaclab-workspace` via `_isaaclab/docker/.env.base` `DOCKER_VOLUMES` or an override, and record the working mount in `notes/`).

- [ ] **Step 5: VERIFY all four flagged import paths**

Run inside the container:
```bash
./isaaclab.sh -p -c "from isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg import FrankaReachEnvCfg; \
c=FrankaReachEnvCfg(); print([t for t in dir(c.rewards) if 'track' in t])"
```
Expected: prints the reward term names; confirm `end_effector_position_tracking` (or correct it in `reach_env_cfg.py`). Fix any of the four VERIFY paths that error, then re-install (Step 4).

- [ ] **Step 6: Smoke-test the registration**

Run inside the container:
```bash
./isaaclab.sh -p -c "import gymnasium as gym, reach_franka; \
assert 'Isaac-Reach-Franka-Custom-v0' in gym.registry, 'not registered'; \
print('OK: custom task registered')"
```
Expected: `OK: custom task registered`.

- [ ] **Step 7: Commit**

```bash
git add source/reach_franka scripts/setup.sh
git commit -m "feat: add custom Isaac-Reach-Franka-Custom-v0 external task"
```

---

## Task 7: Train + eval the custom task; run the reward experiment

**Files:** none created (uses Task 4/5 scripts with the custom task id).

- [ ] **Step 1: Train the custom task (short run to confirm it trains)**

Run: `./scripts/train.sh --task Isaac-Reach-Franka-Custom-v0 --num_envs 256 --max_iterations 50`
Expected: training runs against the custom task; mean reward climbs; checkpoint written under a `reach_franka` log dir.

- [ ] **Step 2: Watch it**

Run: `./scripts/eval.sh --task Isaac-Reach-Franka-Custom-v0 --num_envs 16`
Expected: GUI opens; arms track targets. With 50 iterations tracking is rough but directional.

- [ ] **Step 3: The learning experiment — change the reward, compare**

In `source/reach_franka/reach_franka/reach_env_cfg.py`, change the `*= 2.0` multiplier to `*= 0.5`, re-install (`setup.sh` Step 4 block, or the single pip line), retrain Step 1, and watch Step 2. Note in `LEARNING.md` how the weaker tracking weight changes learned behavior / reward curve.

- [ ] **Step 4: Restore a sensible default and commit**

Set the multiplier back to `2.0` (or whatever you found best).
```bash
git add source/reach_franka/reach_franka/reach_env_cfg.py
git commit -m "feat: tune end-effector tracking reward weight for custom Reach"
```

---

## Task 8: Private learning scaffold

**Files:**
- Create (all gitignored): `LEARNING.md`, `CLAUDE.md`, `notes/00-isaaclab-paths.md`, `notes/01-isaac-sim-vs-lab.md`

- [ ] **Step 1: Confirm these paths are gitignored**

Run: `git check-ignore LEARNING.md CLAUDE.md notes/`
Expected: all three echoed back (meaning ignored). If not, fix `.gitignore`.

- [ ] **Step 2: Write `LEARNING.md`** (roadmap + dated log)

```markdown
# Learning log — Isaac Lab + RL

## Roadmap
- [x] Isaac Sim vs Kit vs Lab
- [ ] The RL loop (env ↔ agent, episodes, rollouts)
- [ ] Observation / action / reward (task anatomy)
- [ ] Reading TensorBoard reward curves
- [ ] Reward shaping & credit assignment
- [ ] Parallel envs & on-policy PPO

## Log
### 2026-06-07
- Stood up Isaac Lab v2.1.0 in Docker on Isaac Sim 4.5.0.
- (fill in as we go)
```

- [ ] **Step 3: Write `CLAUDE.md`** (teaching context — auto-loads for the assistant)

```markdown
# Project context for the assistant

This is a guided **learning** project: train a Franka Reach policy in Isaac Lab
while teaching me Isaac Sim + RL. Act as research/coding/study partner.

## How to teach me
- Explain the *why*, tied to what's on screen — not just the *what*.
- Before running something, ask me to predict the outcome, then we test it.
- Two depths: terse inline explanation + optional deeper `notes/` entry.
- I drive the pace ("go deeper" / "just build").
- When unsure, dig into Isaac Lab source/docs together; record findings in `notes/`.

## Where we are
- Pinned: Isaac Sim 4.5.0, Isaac Lab v2.1.0. See docs/superpowers/specs/ + plans/.
```

- [ ] **Step 4: Write `notes/00-isaaclab-paths.md`** with the verified import/script paths from Tasks 3 & 6.

```markdown
# Verified Isaac Lab v2.1.0 paths (this machine)
- Tasks package: isaaclab_tasks            # update if VERIFY found otherwise
- Train script:  scripts/reinforcement_learning/rsl_rl/train.py
- Play script:   scripts/reinforcement_learning/rsl_rl/play.py
- Franka Reach cfg: isaaclab_tasks.manager_based.manipulation.reach.config.franka.joint_pos_env_cfg
- Reward term: end_effector_position_tracking
- Our repo mount in container: /workspace/isaaclab-workspace
```

- [ ] **Step 5: Verify nothing private is staged**

Run: `git status --short`
Expected: `LEARNING.md`, `CLAUDE.md`, `notes/` do **not** appear. If any appear, stop and fix `.gitignore`.

- [ ] **Step 6: No commit** (these are intentionally untracked/private).

---

## Task 9: Finalize README and push

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Expand `README.md`** with real setup/run/sync instructions

```markdown
# isaaclab-workspace

Fully-Dockerized [Isaac Lab](https://github.com/isaac-sim/IsaacLab) workspace for
robot reinforcement learning. First milestone: Franka **Reach**. Built on
Isaac Sim 4.5.0 + Isaac Lab v2.1.0.

## Prerequisites
- NVIDIA driver + Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- `docker login nvcr.io` with an NGC API key (to pull the Isaac Sim base image)
- `git`, `git-lfs`

## Setup
```bash
git clone https://github.com/Adityawks28/isaaclab-workspace.git
cd isaaclab-workspace
./scripts/setup.sh          # clone+pin Isaac Lab, build the image (first run: slow)
```

## Train & evaluate
```bash
./scripts/train.sh                                   # headless built-in Reach (256 envs)
./scripts/eval.sh                                    # watch it (GUI)
./scripts/train.sh --task Isaac-Reach-Franka-Custom-v0   # our custom task
```

## Laptop ↔ workstation
Train headless on the workstation, watch on the laptop:
```bash
# on workstation
./scripts/train.sh --num_envs 4096
# on laptop (over Tailscale)
rsync -av workstation:isaaclab-workspace/_isaaclab/logs/ ./_isaaclab/logs/
./scripts/eval.sh
```

## Design
See `docs/superpowers/specs/2026-06-07-manipulation-rl-reach-design.md`.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: expand README with setup, train/eval, and sync instructions"
```

- [ ] **Step 3: Push to the personal GitHub repo**

Run: `git push`
Expected: pushes to `Adityawks28/isaaclab-workspace` (personal account — NOT the shared CINAPSLab org). Verify nothing gitignored was pushed: `git ls-files | grep -E 'LEARNING|CLAUDE|notes/|_isaaclab/|outputs/'` returns nothing.

---

## Self-Review

- **Spec coverage:** Repo layout/ship-private split → Tasks 1, 8, 9 + `.gitignore` (already committed). Docker + Isaac Lab setup (Approach A) → Task 2. Reach train→eval flow → Tasks 4, 5. Fork-and-customize → Tasks 6, 7. Learning scaffold → Task 8. Laptop↔workstation/Tailscale → README (Task 9) + train/eval `--num_envs` flag. Success criteria 1–5 all map to verification steps in Tasks 4–8. ✓
- **Placeholders:** "VERIFY" markers are deliberate, scoped verification steps for version-sensitive external paths (each has a concrete command to resolve it), not vague TODOs. No "add error handling"-style gaps. ✓
- **Type/name consistency:** `_common.sh` vars (`ISAACLAB_DIR`, `ISAACSIM_BASE_IMAGE`, `REPO_DIR`) are reused verbatim in `setup.sh`/`train.sh`/`eval.sh`; task id `Isaac-Reach-Franka-Custom-v0` and reward term `end_effector_position_tracking` match across Tasks 6–8. ✓

---

**Sources (version pinning):**
- [Isaac Lab Releases](https://github.com/isaac-sim/IsaacLab/releases)
- [Isaac Lab Installation docs](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
- [Isaac Sim 4.5.0 Requirements](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/requirements.html)
