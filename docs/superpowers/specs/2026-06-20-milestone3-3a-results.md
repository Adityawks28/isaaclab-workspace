# Milestone 3a: results on the laptop (RTX 4050, 6 GB)

**Date:** 2026-06-20
**Status:** Vision front-end verified; VLA model gated by hardware/architecture.

## What works (verified)

The vision-observation front-end a VLA needs runs on the 6 GB laptop:

- A new task, `Isaac-Lift-Cube-Franka-Camera-v0`, adds a 224x224 wrist RGB camera to
  the Lift scene (`source/lift_franka/lift_franka/lift_env_cfg.py`).
- `scripts/vla_camera_test.py` loads it with `--enable_cameras`, steps the sim, and
  reads the camera. Verified output:

  ```
  CAMERA_OK rgb shape (1, 224, 224, 3) dtype torch.uint8 min 1.0 max 250.0
  ```

So each control step renders a real RGB frame (not blank) that a policy or VLA can
consume, grabbed via `env.unwrapped.scene["wrist_cam"].data.output["rgb"]`. Challenge
#1 from the design doc (vision observations) is solved on this hardware.

## The real blocker: model + process architecture, not the camera

Running the VLA *model* itself is where the laptop stops being enough, for two
reasons that compound:

1. **Process/runtime split.** Octo is JAX; Isaac Sim ships its own pinned CPython with
   PyTorch and a fixed numpy. Installing JAX into the Isaac Sim Python risks breaking
   that environment, and JAX + Isaac Sim do not naturally co-reside. The clean design
   is **two processes**: the sim process renders and steps; a separate VLA inference
   process (its own venv) takes the image + instruction and returns an action; they
   talk over a socket/queue. That is real engineering, not a one-file script.

2. **6 GB VRAM.** Even with the model resident, cameras-on Isaac Sim plus a VLA on a
   single 6 GB GPU is at or past the budget (the hardware table in the design doc).

## Probe: can the VLA stack even install here?

Isolated-venv install test (container system Python, separate from the sim Python):

> Result: the Isaac Sim container has **no standalone `python3`** on PATH. The only
> interpreter is the pinned Isaac Sim Python, reached through `./isaaclab.sh -p`. So
> there is no clean place inside the container to isolate a JAX/Octo venv: a JAX
> install would land in the pinned Sim Python and risk it. This concretely confirms
> the conclusion above, the VLA model belongs in its **own process or host**, not
> wedged into the Sim runtime.

## Honest conclusion for 3a

The laptop-achievable half of 3a, the **vision pipeline**, is done and verified. The
VLA model half needs either a two-process server architecture or, more simply, the
workstation/cloud where VRAM and a clean JAX environment are available. The scaffold
(`scripts/vla_play.py`) and this camera task are the runnable starting point for that
next tier. This matches the design doc's hardware reality: Octo inference in sim is
"tight" on 6 GB, and the integration plumbing is the actual cost.

## Next steps (workstation/cloud)

1. Stand up an Octo inference process (its own JAX venv), load `octo-small`.
2. Bridge: sim sends the 224x224 RGB + "pick up the cube" to the Octo process; it
   returns a 7-DoF EE delta + gripper; map that to `Isaac-Lift-Cube-Franka-IK-Rel-v0`.
3. Run zero-shot, record behavior honestly (expect a large sim-to-real domain gap).
4. 3b: generate sim demonstrations from the RL experts, fine-tune Octo, re-evaluate.
