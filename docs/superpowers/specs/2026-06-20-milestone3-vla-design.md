# Milestone 3: Vision-Language-Action (VLA) in Isaac Sim

**Date:** 2026-06-20
**Status:** Design (Milestones 1-2 built; this scopes the VLA milestone)

## Purpose

Milestones 1-2 trained **per-task RL** policies (Reach, Lift) from hand-designed
rewards. They hit a ceiling: a new reward per task, no generalization, no language,
and, for contact-rich lifting on a 6 GB laptop, a grasp-reliability plateau that
more envs/iterations/curriculum/friction did **not** break.

A **Vision-Language-Action (VLA)** model is the next paradigm: a large transformer
that maps **camera image(s) + a language instruction → robot actions**, trained by
imitation on broad robot data. Milestone 3 is about *learning that paradigm hands-on*
in our existing Isaac Lab workspace, not about beating our RL policies overnight.

## VLA landscape & the choice for our hardware

| Model | Params | Inference VRAM | Notes |
|---|---|---|---|
| **Octo** | **93 M** | runs alongside sim on a modest GPU | diffusion action head, smooth trajectories; ~50 ms/step on a 4090 |
| OpenVLA | 7 B | bf16 ≈ 16 GB · INT8 ≈ 7 GB · INT4 ≈ 3.5 GB (8-12% worse) | Llama-2 backbone; tokenized actions |
| π0 / RT-2 | large | A100-class / not open | frontier, out of scope here |

**Hardware reality (this project):**

| Activity | Needs | Laptop (RTX 4050, 6 GB) | Workstation | Cloud |
|---|---|---|---|---|
| Run **Octo** inference in sim | model (~hundreds of MB) **+** Isaac Sim (~3-4 GB) | ✅ tight but feasible | ✅ | ✅ |
| Run **OpenVLA** inference | INT4 ≈ 3.5 GB **+** sim | ✗ together OOMs 6 GB | ~ | ✅ |
| **Fine-tune** Octo (≈200 demos) | ~2 h on a 4090 | slow / borderline | ✅ | ✅ |
| Fine-tune OpenVLA (LoRA) | A100-class, 4-8 h | ✗ | ~ | ✅ |

**Decision: Octo for the laptop path** (small enough to co-reside with Isaac Sim);
**OpenVLA for the workstation/cloud path.** This mirrors the Milestone 1-2 split, 
laptop validates the pipeline, heavier compute lives elsewhere.

## The three integration challenges (be honest up front)

1. **Vision observations.** Our Lift env uses *state* obs (joint/object pose). A VLA
 needs **camera images**, so the env must render a camera (`enable_cameras`, add a
 `Camera`/`TiledCamera` sensor), heavier on VRAM and the main new plumbing.
2. **Action-space mapping.** VLAs output their own action convention (e.g. 7-DoF
 end-effector delta + gripper, from Open X-Embodiment). That must be mapped to the
 Franka's action space, our **IK action variant** (`Isaac-Lift-Cube-Franka-IK-Rel-v0`)
 is the natural target, since it already acts in EE-delta space.
3. **Domain gap → zero-shot will likely fail.** Octo/OpenVLA were trained on *real*
 robot images and embodiments, not our sim Franka and camera. Run zero-shot, the
 behavior will probably be poor. **That is expected**, the value of 3a is learning
 the interface, which then motivates fine-tuning on sim data (3b).

## Staged plan

- **3a, Inference pipeline (laptop, Octo).** Add a camera to the Lift scene; write a
 loop that feeds the rendered image + an instruction ("pick up the cube") to Octo and
 applies its actions through the IK action space. Goal: a working *pipeline* and an
 honest look at zero-shot behavior. Deliverable: a video + notes on the domain gap.
- **3b, Sim as a data factory + fine-tune.** Use our RL experts (or scripted graspers)
 to generate demonstrations in sim, convert to the VLA's data format, and fine-tune
 Octo on them. Goal: a VLA that actually performs the sim task, closing the loop
 between RL (data engine) and imitation (generalist policy).
- **3c, Language-conditioned multi-task (stretch).** Multiple instructions / objects;
 the real payoff of the VLA paradigm. Workstation/cloud territory.

## What's doable now vs. later

- **GPU-free now:** this design; the integration scaffold (`source/vla_franka/`, marked
 VERIFY against Octo's real API); the camera-obs cfg sketch.
- **Needs GPU + network (after the Lift run frees the GPU):** install Octo (JAX) into a
 Python env, download pretrained weights, enable cameras, run 3a inference.

## Success criteria

1. A camera-enabled Lift env renders an image observation each step.
2. Octo loads and produces actions from (image + instruction) in our env (3a).
3. Honest write-up of zero-shot behavior + the domain gap (3a).
4. A sim-demonstration dataset + a fine-tuned Octo that improves on zero-shot (3b).

## References

- [OpenVLA (GitHub)](https://github.com/openvla/openvla) · [OpenVLA paper](https://arxiv.org/html/2406.09246v3)
- [Octo, generalist robot policy](https://octo-models.github.io/)
- [Open-weight robot models overview (OpenVLA, π0, Octo, RT-X)](https://robocloud-dashboard.vercel.app/learn/blog/open-weight-robot-models)
- [VLA models, benchmarks & GPU requirements](https://robocloud-dashboard.vercel.app/learn/blog/vla-models-robotics-2025)
- Our Ch. 12 lesson (`docs/learning/12-toward-general-manipulation-vla.html`), the conceptual background.
