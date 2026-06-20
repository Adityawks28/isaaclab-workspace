<div align="center">

# 🦾 isaaclab-workspace

**A fully-Dockerized [Isaac Lab](https://github.com/isaac-sim/IsaacLab) workspace for learning robot reinforcement learning — from Franka _Reach_ to contact-rich _pick-and-place_.**

![Isaac Sim](https://img.shields.io/badge/Isaac%20Sim-4.5.0-76B900?logo=nvidia&logoColor=white)
![Isaac Lab](https://img.shields.io/badge/Isaac%20Lab-v2.1.0-1f6feb)
![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![RL](https://img.shields.io/badge/RL-PPO%20·%20rsl__rl-EE4C2C?logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/runtime-Docker-2496ED?logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/Reach-converged-2ea043) ![Status](https://img.shields.io/badge/Lift-in%20progress-d29922)

[**Quick start**](#-quick-start) · [**Tasks**](#-tasks) · [**Results**](#-results--demos) · [**How it works**](#-how-it-works) · [**Roadmap**](#-roadmap)

<br/>

<table>
<tr>
<td align="center"><img src="docs/media/reach.gif" width="380" alt="Franka Reach"/><br/><b>Reach</b> · ✅ converged</td>
<td align="center"><img src="docs/media/lift.gif" width="380" alt="Franka Lift"/><br/><b>Lift</b> · 🚧 in progress</td>
</tr>
</table>

</div>

---

## Overview

A clean, reproducible workspace for training and watching reinforcement-learning policies on a
[Franka Emika Panda](https://www.franka.de/) arm in [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim).
Everything runs in Docker (no local install), built **FROM** the official Isaac Sim image with
Isaac Lab + RL libraries layered on top. Custom tasks live in this repo as pip-installable
[external tasks](https://isaac-sim.github.io/IsaacLab/) — Isaac Lab itself is never edited.

It doubles as a **guided learning project**: a private, gitignored layer of notes and interactive
HTML lessons sits alongside the shippable code (it stays on the laptop, never on GitHub).

## ✨ Highlights

- 🐳 **Zero local install** — one script clones + pins Isaac Lab and builds the image, reusing warm caches.
- 🧩 **Custom tasks as plugins** — registered via a gymnasium entry point; stock `train.py`/`play.py` see them with no edits to Isaac Lab.
- 🔬 **Experiment-driven** — reward-weight, lift-gate, curriculum, and grasp-friction sweeps, each with an interactive HTML analysis dashboard.
- 💻 **Laptop-friendly** — laptop-safe `--num_envs` defaults (6 GB GPU); headless training, GUI eval, and a laptop↔workstation sync flow over Tailscale.
- 📱 **Semi-autonomous** — drive the whole loop from your phone via Claude Code Remote Control (monitor + approve), GPU never leaves the machine.

## 🎬 Results / demos

Trained policies, rendered headlessly (`play.py --video`). Click to play.

| Milestone | Demo | Status |
|---|---|---|
| **1 · Reach** | [▶ `docs/media/01-reach-franka.mp4`](docs/media/01-reach-franka.mp4) | ✅ **converged** — the end-effector tracks randomly-placed target poses (≈ 0.12 m error) |
| **2 · Lift** (pick-and-place) | [▶ `docs/media/02-lift-franka-wip.mp4`](docs/media/02-lift-franka-wip.mp4) | 🚧 **in progress** — reliably reaches + intermittently lifts; the pipeline is end-to-end |

> All training here is **laptop-budget** (RTX 4050, 6 GB → 128 envs). Reach converges easily;
> contact-rich lifting is sample-hungry and plateaus on the laptop — controlled sweeps trace the
> cause to **grasp reliability**, with workstation-scale parallelism as the next lever.

## 🚀 Quick start

```bash
# Prereqs: NVIDIA driver + Docker + NVIDIA Container Toolkit, and `docker login nvcr.io`
./scripts/setup.sh      # clone + pin Isaac Lab, build the image, install the custom tasks
./scripts/train.sh      # headless PPO training (defaults to Reach)
./scripts/eval.sh       # watch the trained policy in the Isaac Sim GUI
```

Train a specific task and env count:

```bash
./scripts/train.sh --task Isaac-Lift-Cube-Franka-Custom-v0 --num_envs 128
./scripts/eval.sh  --task Isaac-Lift-Cube-Franka-Custom-v0 --num_envs 16
```

## 🧩 Tasks

Custom external tasks registered by this repo (in addition to all of Isaac Lab's built-ins):

| Gym ID | What it is |
|---|---|
| `Isaac-Reach-Franka-Custom-v0` | Franka Reach with a tuned end-effector tracking reward |
| `Isaac-Lift-Cube-Franka-Custom-v0` | Pick-and-place with a raised lift gate (harder lift) |
| `Isaac-Lift-Cube-Franka-Curriculum-v0` | Lift + a **receding-goal curriculum** (easy goal → full task) |
| `Isaac-Lift-Cube-Franka-Grasp-v0` | Lift + a **high-friction cube** (targets grasp reliability) |

## 🗂️ Project structure

```
isaaclab-workspace/
├── scripts/
│   ├── setup.sh                 # clone+pin Isaac Lab, build image, install tasks
│   ├── train.sh / eval.sh       # headless train / GUI eval wrappers
│   ├── build_sweep_dashboard.py # parse training logs → interactive HTML analysis
│   └── auto*.sh                 # semi-autonomous Remote Control supervisor
├── source/
│   ├── reach_franka/            # custom Reach task (pip-installable plugin)
│   └── lift_franka/             # custom Lift tasks: Custom / Curriculum / Grasp
├── docs/
│   ├── media/                   # demo videos
│   └── superpowers/             # design spec + implementation plan
├── _isaaclab/                   # pinned Isaac Lab clone        (gitignored)
└── outputs/                     # checkpoints, logs, videos     (gitignored)
```

## 🛠️ How it works

**Approach A — official Isaac Lab Docker tooling.** `setup.sh` clones Isaac Lab (pinned **v2.1.0**)
and builds an image **FROM** `nvcr.io/nvidia/isaac-sim:4.5.0` with Isaac Lab + rsl_rl baked in,
then `pip install -e` our task packages into the container. Each task registers its gym ID through a
[gymnasium plugin entry point](https://gymnasium.farama.org/), so Isaac Lab's stock training and
play scripts discover it **without any edits to the Isaac Lab source**.

```
Omniverse Kit  →  Isaac Sim (PhysX, robots)  →  Isaac Lab (RL env)  →  our task  →  PPO (rsl_rl)
```

The three-layer software stack, the RL theory (MDP, PPO, GAE), reward shaping, and the
pick-and-place physics are documented as interactive lessons in the private learning layer.

## 📊 Experiments

Controlled studies (fixed seed + budget, weight-independent success metrics), each rendered to a
self-contained interactive dashboard by `scripts/build_sweep_dashboard.py`:

- **Reward-weight sweep** (Reach) — weights set *priorities*, not absolute quality; compare on the honest metric.
- **Lift-gate sweep** (0.04 / 0.06 / 0.08) — only the easy gate sustains lifting.
- **Iteration ablation** — more training plateaus; the bottleneck is structural, not duration.
- **Receding-goal curriculum** & **grasp-friction** — isolating *why* carry-to-goal is hard.

## 📱 Semi-autonomous mode (monitor + approve from your phone)

Run Claude Code so it works the plan autonomously on this machine while you check in and approve
actions from your phone. The session stays **local** (your GPU + Docker never leave the machine);
only the chat/approval UI flows to your phone via
[Remote Control](https://code.claude.com/docs/en/remote-control).

```bash
./scripts/auto.sh                 # supervisor in tmux: keeps the session alive
tmux attach -t claude-auto        # watch locally; detach with Ctrl-b then d
```

The supervisor relaunches the session after crashes, reboots, network drops, and **usage-limit
resets**. The session self-paces through the plan (`scripts/auto-prompt.md` is the standing
kickoff), generating HTML learning notes as it goes. See `scripts/claude-auto.service` for a
boot-survival user service.

## 🗺️ Roadmap

- [x] **Milestone 1 — Reach** · converges, end-effector tracking ≈ 0.12 m
- [ ] **Milestone 2 — Pick-and-place** · pipeline works; reliable grasp-and-carry needs more parallel envs
- [ ] **Milestone 3 — VLA** · run a pretrained vision-language-action model in sim; sim-as-data

## 🙏 Built on

[Isaac Sim](https://developer.nvidia.com/isaac-sim) · [Isaac Lab](https://github.com/isaac-sim/IsaacLab) · [rsl_rl](https://github.com/leggedrobotics/rsl_rl) · [gymnasium](https://gymnasium.farama.org/)

<div align="center"><sub>Design spec &amp; implementation plan in <code>docs/superpowers/</code>.</sub></div>
