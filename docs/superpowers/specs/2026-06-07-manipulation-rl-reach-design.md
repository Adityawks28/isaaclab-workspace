# Manipulation RL — Franka Reach (Isaac Lab) — Design

**Date:** 2026-06-07
**Local dir:** `~/robot-manipulation-lab`  ·  **GitHub repo:** `isaaclab-franka-manipulation`
**Status:** Approved design → ready for implementation planning

## Purpose

A guided, fully-Dockerized reinforcement-learning project that trains a Franka
arm on the **Reach** task in **Isaac Lab**, while doubling as a structured
learning vehicle for Isaac Sim and RL. The project ships a clean, reproducible
research repo to GitHub; the personal learning layer stays private and local.

Primary goal (user-stated): **learn Isaac Lab + RL by building**, with the
assistant acting as research, coding, and study partner.

## Constraints & context

- **Runtime:** fully Docker, no local install. Isaac Sim **4.5.0**
  (`nvcr.io/nvidia/isaac-sim:4.5.0`). Local 5.1.0 source build was deleted.
- **GPU:** laptop = RTX 4050 Laptop, **6 GB VRAM** (below Isaac Sim's stated
  minimum) → reduced `--num_envs` locally. A beefier **workstation** is
  available for heavy training.
- **Network fabric:** Tailscale (MagicDNS) connects laptop ↔ workstation from
  any network.
- **Existing assets:** warm caches in `~/docker/isaac-sim/`; launcher at
  `docker/container.sh` (plain Isaac Sim / GUI).

## Approach

**Approach A — official Isaac Lab Docker tooling.** Clone the Isaac Lab repo
(pinned to a version compatible with Isaac Sim 4.5), and use its
`docker/container.py`, which builds an image **FROM** the 4.5.0 base with Isaac
Lab + RL libraries (rsl_rl / skrl / rl_games) baked in, reusing the warm caches.
Chosen because tutorials/docs/community all assume it, so learning transfers
1:1, and it is the most maintained, reproducible, portable path. Our repo is a
clean GitHub-shippable wrapper around it.

## Section 1 — Repo layout & ship/private split

```
robot-manipulation-lab/                 ← git repo (pushed to GitHub)
├── README.md              ship   Overview + setup/run instructions
├── .gitignore             ship   Enforces the ship/private split
├── requirements.txt       ship   Extra Python deps for our task code
├── docker/
│   └── container.sh        ship   Existing launcher (plain Isaac Sim / GUI)
├── source/
│   └── reach_franka/       ship   OUR Reach task (Isaac Lab external task):
│                                  env cfg, observations, rewards, agent cfg
├── scripts/
│   ├── setup.sh            ship   Clone+pin Isaac Lab, build image
│   ├── train.sh            ship   Launch headless training
│   └── eval.sh             ship   Watch trained policy (GUI)
├── docs/superpowers/specs/ ship   This design doc
│
├── _isaaclab/              ignore Cloned Isaac Lab repo (big, pinned)
├── notes/                  PRIVATE Study notes & learning logs (gitignored)
├── LEARNING.md             PRIVATE Guided learning-workflow doc (gitignored)
├── CLAUDE.md               PRIVATE Teaching context for the assistant (gitignored)
└── outputs/                ignore Checkpoints, TensorBoard logs, videos
```

`.gitignore` is the mechanism for the split. A GitHub cloner gets a runnable
project; the learning layer never leaves the laptop. `CLAUDE.md` being
gitignored means it does not auto-load on the workstation clone — acceptable,
since the workstation only trains.

`source/reach_franka/` follows Isaac Lab's **external-task** pattern so our task
code lives in our repo, separate from the Isaac Lab clone.

## Section 2 — Docker + Isaac Lab setup

`scripts/setup.sh` orchestrates:
1. `git clone` Isaac Lab into `_isaaclab/`, pinned to a 4.5-compatible tag
   (exact tag locked + verified at implementation time).
2. Isaac Lab's `docker/container.py start` builds an image FROM
   `isaac-sim:4.5.0` with Isaac Lab + RL libs, reusing warm caches.

**Train vs. eval — the one real difference:**

| | Command | Display | Where |
|---|---|---|---|
| Train | `./scripts/train.sh` (→ `--headless`) | none | laptop or workstation |
| Eval  | `./scripts/eval.sh` | GUI (X11) | wherever you want to watch |

Headless = no window, no X11; just GPU + checkpoints to `outputs/`. Runs on any
machine including a headless server.

### Workstation setup

One-time prereqs (machine-level, not scriptable): NVIDIA driver + Docker +
NVIDIA Container Toolkit; `docker login nvcr.io` with NGC API key (to pull the
~22 GB base image); `git` + `git-lfs`.

Then: `git clone <repo>` → `./scripts/setup.sh` → `./scripts/train.sh`.

## Section 3 — The Reach train→eval flow

**Task (`Isaac-Reach-Franka`):** move the end-effector to a randomly placed
target pose that resets each episode. Continuous tracking, no grasping — the
simplest task that exercises the full RL machinery.

**The three RL pieces** (every Isaac Lab task is defined by these):

| Piece | For Reach | Concept |
|---|---|---|
| Observations | joint angles + velocities + target pose + last action | the state |
| Actions | 7 joint position targets | the action space |
| Reward | mostly `−distance(EE, target)` + small smoothness penalties | reward shaping |

Agent = PPO (rsl_rl). Many arms train in parallel in one GPU sim.

**Flow:**
1. **Run built-in** `Isaac-Reach-Franka-v0` headless → validate pipeline,
   watch reward curve in TensorBoard.
2. **Watch it** via `eval.sh` (GUI, ~16 arms).
3. **Fork it** into `source/reach_franka/` as `Isaac-Reach-Franka-Custom-v0` →
   edit reward/observations, retrain, compare. This is where the learning
   happens.

On 6 GB GPU: reduced `--num_envs` (e.g. 256–512); workstation runs full count.

Exact observation layout, reward terms, and script paths confirmed against the
pinned Isaac Lab version at implementation time (they shift between releases).

## Section 4 — The learning scaffold (private)

| File/dir | Job |
|---|---|
| `LEARNING.md` | Roadmap of concepts + dated progress log |
| `notes/` | Per-concept deep dives (`01-isaac-sim-vs-lab.md`, `02-obs-action-reward.md`, `03-ppo-intuition.md`, …) |
| `CLAUDE.md` | Teaching context: level, goals, learning style, where we left off |

**Method — the build is the curriculum.** Just-in-time teaching tied to the
thing on screen:

```
Build step                  → Concept
set up Isaac Lab            → Isaac Sim vs Kit vs Lab
run built-in Reach         → the RL loop (env ↔ agent, episodes, rollouts)
read the env config        → observation / action / reward
watch TensorBoard          → what "learning" looks like
fork & edit reward         → reward shaping & credit assignment
tune --num_envs            → parallel envs, on-policy PPO, sample efficiency
```

**Operating principles** (recorded in `CLAUDE.md`): explain the "why";
active-recall predictions before running; two depths (inline + optional
`notes/` entry); user drives pace; research unknowns together and record
findings in `notes/`.

All three stay gitignored — private to the laptop, never on GitHub.

## Dev loop (laptop ↔ workstation, over Tailscale)

```
LAPTOP                              WORKSTATION
edit source/ (VS Code / Remote-SSH)
git commit + push  ───────────────► git pull
                                    ./scripts/train.sh (headless, GPU)
                                    writes outputs/checkpoint
      ◄──────── rsync outputs/ ──────
./scripts/eval.sh (GUI, local)
   watch → iterate → repeat
```

- **Git** = code spine + GitHub ship.
- **rsync** = checkpoints back to laptop (gitignored, too big/frequent for git).
- **Remote-SSH** = fast inner-loop edits without commit spam.
- **Tailscale** = makes all of the above work from any network; also unlocks
  optional Isaac Sim WebRTC livestream from the workstation for heavier tasks.

## Out of scope (YAGNI for v1)

- Lift / drawer / stacking tasks (future milestones after Reach works).
- Custom robots beyond Franka.
- WebRTC streaming setup (door left open, not built now).
- Sim-to-real transfer.

## Success criteria

1. Built-in Reach trains headless on both machines; reward curve converges.
2. Trained policy visibly tracks targets in the GUI on the laptop.
3. Forked custom task trains and shows behavior change from a reward edit.
4. Repo clones clean on the workstation and reproduces the setup.
5. Learning layer (`LEARNING.md` + `notes/`) captures the concepts covered.
```
