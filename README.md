# isaaclab-workspace

A fully-Dockerized [Isaac Lab](https://github.com/isaac-sim/IsaacLab) workspace for
robot reinforcement learning. First milestone: Franka **Reach**.

Built on **Isaac Sim 4.5.0** + **Isaac Lab v2.1.0**. See `docs/superpowers/specs/` for
the design and `docs/superpowers/plans/` for the implementation plan.

## Quick start

```bash
./scripts/setup.sh      # clone + pin Isaac Lab, build the Docker image
./scripts/train.sh      # headless training
./scripts/eval.sh       # watch the trained policy (GUI)
```

> Full setup, training, and laptop↔workstation instructions are filled in as the
> scripts land (see the implementation plan).
