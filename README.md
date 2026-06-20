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

## Semi-autonomous mode (monitor + approve from your phone)

Run Claude Code so it works the plan autonomously on this machine while you check in
and approve actions from your phone. The session stays **local** (your GPU + Docker
never leave the machine); only the chat/approval UI flows to your phone via
[Remote Control](https://code.claude.com/docs/en/remote-control). Requires Claude Code
≥ 2.1.110 and a claude.ai Pro/Max login (not an API key).

**One-time setup**
1. Phone: install the Claude app (iOS/Android), sign in with the **same account**,
   allow notifications. (`/mobile` in Claude Code shows a QR.)
2. In a Claude session run `/config` and enable **Push when Claude decides** and
   **Push when actions required**.
3. Permissions live in `.claude/settings.local.json` (already created): edits and safe
   project commands auto-run; risky ones (`git push`, `rm`, image rebuilds, `setup.sh`)
   prompt you — those prompts show up on your phone.

**Run it**
```bash
./scripts/auto.sh                 # supervisor in tmux: keeps the session alive
tmux attach -t claude-auto        # watch locally; detach with Ctrl-b then d
```
The supervisor relaunches the session after crashes, reboots, network drops, and
**usage-limit resets** (it backs off and resumes via `--continue`). The session itself
self-paces through the plan (`scripts/auto-prompt.md` is the standing kickoff),
generating HTML learning notes under `docs/learning/` as it goes. On a usage limit it
schedules its next tick just past the reset instead of stopping.

For boot survival, install the optional user service (see `scripts/claude-auto.service`).
