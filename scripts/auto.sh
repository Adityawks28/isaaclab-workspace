#!/usr/bin/env bash
# Supervisor for the semi-autonomous Claude Code session.
#
# Keeps a local, Remote-Controllable Claude session alive across crashes, reboots,
# network drops, and usage-limit resets. The session runs locally (your GPU + Docker
# stay on this machine); you monitor and approve permission prompts from your phone
# via the Claude app / claude.ai/code.
#
#   ./scripts/auto.sh            # launches inside a tmux session named "claude-auto"
#   ./scripts/auto.sh --inner    # run the loop directly (no tmux wrapper)
#   tmux attach -t claude-auto   # watch it locally
#
# Env overrides:
#   CLAUDE_AUTO_PROJECT   project dir to run in (default: repo root)
#   CLAUDE_AUTO_BACKOFF   seconds to wait between relaunches (default: 300)
set -uo pipefail

PROJECT_DIR="${CLAUDE_AUTO_PROJECT:-/home/cll3/isaaclab-workspace}"
SESSION="claude-auto"
BACKOFF="${CLAUDE_AUTO_BACKOFF:-300}"
NAME="Franka Reach auto"
PROMPT_FILE="${PROJECT_DIR}/scripts/auto-prompt.md"
STATE_DIR="${PROJECT_DIR}/.claude"
STARTED_FLAG="${STATE_DIR}/.auto-started"

# Re-exec inside tmux so the session survives logout / SSH disconnect.
if [[ -z "${TMUX:-}" && "${1:-}" != "--inner" ]]; then
  if ! command -v tmux >/dev/null; then
    echo "tmux not found. Install it (sudo apt install tmux) or run: $0 --inner" >&2
    exit 1
  fi
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "Session '$SESSION' already running. Attach with: tmux attach -t $SESSION"
    exit 0
  fi
  echo "Launching supervisor in tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
  exec tmux new-session -s "$SESSION" "$0" --inner
fi

cd "$PROJECT_DIR" || { echo "Cannot cd to $PROJECT_DIR" >&2; exit 1; }
mkdir -p "$STATE_DIR"

if [[ ! -f "$PROMPT_FILE" ]]; then
  echo "Missing kickoff prompt: $PROMPT_FILE" >&2
  exit 1
fi

echo "[auto] supervisor up. project=$PROJECT_DIR backoff=${BACKOFF}s"
while true; do
  if [[ -f "$STARTED_FLAG" ]]; then
    echo "[auto] $(date '+%F %T') resuming session (--continue)…"
    start=$(date +%s)
    claude --continue --remote-control --name "$NAME" --permission-mode acceptEdits
    elapsed=$(( $(date +%s) - start ))
    # If --continue dies almost immediately, there's probably no conversation to
    # resume — drop the flag so the next iteration starts a fresh, seeded session.
    if (( elapsed < 10 )); then
      echo "[auto] resume exited after ${elapsed}s; will start fresh next time."
      rm -f "$STARTED_FLAG"
    fi
  else
    echo "[auto] $(date '+%F %T') starting fresh autonomous session…"
    touch "$STARTED_FLAG"
    claude --remote-control --name "$NAME" --permission-mode acceptEdits "$(cat "$PROMPT_FILE")"
  fi
  echo "[auto] $(date '+%F %T') session ended; backing off ${BACKOFF}s then relaunching."
  sleep "$BACKOFF"
done
