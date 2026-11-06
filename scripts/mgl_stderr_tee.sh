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
# Force Iris perf capture even if Prism cached an older Env=0.
export MGL_PERF_SUMMARY="${MGL_PERF_SUMMARY:-1}"
export MGL_PERF_SUMMARY_EVERY="${MGL_PERF_SUMMARY_EVERY:-15}"
export MTL_HUD_ENABLED="${MTL_HUD_ENABLED:-1}"
# A/B: first-use color attachments without blend become DontCare (see
# mglRenderPassPlanLoadStore). Default off in code; enable for Iris bandwidth.
export MGL_ENABLE_DONTCARE_LOAD="${MGL_ENABLE_DONTCARE_LOAD:-1}"
echo "MGL_STDERR_TEE env MGL_PERF_SUMMARY=$MGL_PERF_SUMMARY EVERY=$MGL_PERF_SUMMARY_EVERY DONTCARE_LOAD=$MGL_ENABLE_DONTCARE_LOAD" >> "$LOG"
exec "$@" 2> >(tee -a "$LOG" >&2)
