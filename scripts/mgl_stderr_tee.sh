#!/bin/bash
# Prism WrapperCommand helper: run java while appending stderr to a file.
# Paths with spaces must come from the state file, not WrapperCommand args.
set -euo pipefail
STATE="${XDG_CACHE_HOME:-$HOME/.cache}/mgl_stderr_tee.state"
LOG=""
if [[ -f "$STATE" ]]; then
  while IFS=$'\t' read -r key val; do
    case "$key" in
      LOG_FILE) LOG="$val";;
    esac
  done < "$STATE"
fi
[[ -n "$LOG" ]] || LOG="${TMPDIR:-/tmp}/mgl-stderr.log"
mkdir -p "$(dirname "$LOG")"
# Keep a fresh file per launch.
: > "$LOG"
echo "MGL_STDERR_TEE log=$LOG pid=$$" >> "$LOG"
exec "$@" 2> >(tee -a "$LOG" >&2)
