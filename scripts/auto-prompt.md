# Autonomous loop — standing kickoff prompt

You are running semi-autonomously on a local machine. I (the human) monitor from my
phone via Remote Control and approve permission prompts there. Drive yourself with the
**`/loop` skill in self-paced mode** (no fixed interval): do one unit of work, then
schedule your own next tick.

## On the very first tick, load context
1. Read the design spec: `docs/superpowers/specs/2026-06-07-manipulation-rl-reach-design.md`
   — especially the **hardware spec** (laptop RTX 4050, 6 GB VRAM → keep `--num_envs`
   laptop-safe ~256; workstation does heavy runs; Tailscale links them) and the
   **main purpose** (Dockerized Isaac Lab RL, first milestone Franka Reach).
2. Read the implementation plan: `docs/superpowers/plans/2026-06-07-isaaclab-workspace-reach.md`.
3. Read `LEARNING.md` so new learning artifacts build on what's already there.

## Each tick (one small, reviewable unit)
1. Pick the **next unchecked task** in the implementation plan.
2. Execute it following the `executing-plans` discipline: TDD where it applies, keep
   changes small, and **commit at each checkpoint** with a clear message.
3. Write/refresh an **HTML learning artifact** in `docs/learning/` (one file per
   concept or task, plain self-contained HTML) explaining *what* you did and *why* it
   matters for the RL goal — written to teach, not just to log. Append a matching
   short entry to `LEARNING.md`.
4. Check the plan box, commit, and report a one-line status.

## Guardrails
- Stay laptop-safe: never raise `--num_envs` past the 6 GB budget unless explicitly on
  the workstation. Heavy/headless training is fine; GUI eval is laptop-side.
- Routine commands (`train.sh`, `eval.sh`, `docker exec`, git status/diff/add/commit,
  rsync, tensorboard) are pre-approved. Anything risky (`git push`, `rm`, image
  rebuilds, `setup.sh`) will prompt me on my phone — pause for that approval.
- Don't start an `ultraplan` session; it would disconnect Remote Control.

## Usage-limit handling (important)
If a tick can't proceed because the usage limit is hit, **do not stop the workflow**.
Use `ScheduleWakeup` to set the next wake-up to **just past the reset time** (reset +
~2 min buffer) with the same loop prompt, so work resumes automatically once limits
reset. Between ticks, prefer a wake-up that stays within the prompt-cache window when
you're actively iterating, and a longer one when genuinely idle.

When the plan is fully complete, stop the loop and send a final summary.
