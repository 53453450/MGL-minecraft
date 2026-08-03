#!/bin/bash
# Generic apitrace + MGL capture controller for PrismLauncher.
#
# Why generic: the previous build/apitrace_captures scripts hard-coded one
# instance ("1.21.11(2)") and one trace path. This script takes everything
# as arguments and works for any instance / any trace.
#
# Call chain:
#   LWJGL (opengl.libname = apitrace OpenGL.framework wrapper)
#     -> apitrace records GL calls
#     -> forwards to TRACE_LIBGL = libmgl.dylib  -> Metal
#
# Subcommands:
#   enable  [--instance NAME|PATH] [--trace PATH] [--apittrace DIR] [--mgl-dir DIR]
#   disable [--instance NAME|PATH]
#   capture [--trace PATH] [--apittrace DIR] [--mgl-dir DIR] [--trace-internal] -- <java> <args...>
#
# `enable`  wires WrapperCommand into the instance's instance.cfg.
# `capture` is what Prism execs as that WrapperCommand.
# `disable` clears WrapperCommand afterwards.
#
# Paths with spaces (Prism instance dirs, "Application Support") are passed
# via a state file, NEVER via the WrapperCommand string, because Prism
# splits WrapperCommand on whitespace and would mangle quoted args.

set -euo pipefail

APITRACE_DEFAULT="/Users/fterward/apitrace/build"
MGL_DIR_DEFAULT="$HOME/Library/Application Support/PrismLauncher/mgl"
PRISM_INSTANCES_DEFAULT="$HOME/Library/Application Support/PrismLauncher/instances"
STATE_FILE="${XDG_CACHE_HOME:-$HOME/.cache}/mgl_apitrace_capture.state"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "$0")"
# repo root = parent of scripts/ (used only for default trace location)
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

die() { echo "ERROR: $*" >&2; exit 1; }

# Resolve an instance spec to an absolute instance.cfg path.
resolve_instance() {
  local spec="$1"
  if [[ -f "$spec" && "$(basename "$spec")" == instance.cfg ]]; then
    echo "$spec"; return
  fi
  if [[ -f "$spec/instance.cfg" ]]; then
    echo "$spec/instance.cfg"; return
  fi
  local p="$PRISM_INSTANCES_DEFAULT/$spec/instance.cfg"
  [[ -f "$p" ]] || die "instance not found: $spec (looked for $p)"
  echo "$p"
}

# Set/replace a key=value line in instance.cfg (create under [General] if absent).
set_cfg_line() {
  local cfg="$1" key="$2" full="$3"
  python3 - "$cfg" "$key" "$full" <<'PY'
import re, sys
from pathlib import Path
cfg_p, key, full = sys.argv[1:4]
text = Path(cfg_p).read_text()
pat = re.compile(rf'^{re.escape(key)}=.*$', re.M)
if pat.search(text):
    text = pat.sub(full, text, count=1)
elif "[General]\n" in text:
    text = text.replace("[General]\n", f"[General]\n{full}\n", 1)
else:
    text = text + "\n" + full + "\n"
Path(cfg_p).write_text(text)
PY
}

write_state() {
  local trace="$1" apittrace="$2" mgl_dir="$3" internal="$4"
  mkdir -p "$(dirname "$STATE_FILE")"
  {
    echo "TRACE_FILE='$trace'"
    echo "APITRACE='$apittrace'"
    echo "MGL_DIR='$mgl_dir'"
    echo "TRACE_INTERNAL='$internal'"
  } > "$STATE_FILE"
}

cmd_enable() {
  local instance_spec="" trace="" apittrace="$APITRACE_DEFAULT" mgl_dir="$MGL_DIR_DEFAULT" internal=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --instance)  instance_spec="$2"; shift 2;;
      --trace)     trace="$2"; shift 2;;
      --apittrace) apittrace="$2"; shift 2;;
      --mgl-dir)   mgl_dir="$2"; shift 2;;
      --trace-internal) internal=1; shift;;
      *) die "unknown arg: $1";;
    esac
  done
  [[ -n "$instance_spec" ]] || die "enable requires --instance NAME|PATH"
  local cfg; cfg="$(resolve_instance "$instance_spec")"

  [[ -n "$trace" ]] || trace="$REPO_ROOT/build/apitrace_captures/capture_$(date +%Y%m%d_%H%M%S).trace"
  # normalize to absolute
  trace="$(cd "$(dirname "$trace")" 2>/dev/null && pwd)/$(basename "$trace")" \
    || die "bad trace path: $trace"

  write_state "$trace" "$apittrace" "$mgl_dir" "$internal"

  # WrapperCommand carries NO path args (avoids whitespace/quote issues);
  # capture (re)reads everything from STATE_FILE.
  set_cfg_line "$cfg" "WrapperCommand" "WrapperCommand=$SCRIPT_PATH capture"
  set_cfg_line "$cfg" "OverrideCommands" "OverrideCommands=true"

  echo "CAPTURE MODE ON"
  echo "  instance : $cfg"
  echo "  trace    : $trace"
  echo "  mgl-dir  : $mgl_dir"
  echo "  internal : $([[ $internal -eq 1 ]] && echo on || echo off)"
  echo "  launch the instance, play, then quit normally."
  echo "  afterwards run: $SCRIPT_PATH disable --instance '$instance_spec'"
}

cmd_disable() {
  local instance_spec=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --instance) instance_spec="$2"; shift 2;;
      *) die "unknown arg: $1";;
    esac
  done
  [[ -n "$instance_spec" ]] || die "disable requires --instance NAME|PATH"
  local cfg; cfg="$(resolve_instance "$instance_spec")"
  set_cfg_line "$cfg" "WrapperCommand" "WrapperCommand="
  rm -f "$STATE_FILE"
  echo "CAPTURE MODE OFF — WrapperCommand cleared: $cfg"
}

cmd_capture() {
  local trace="" apittrace="$APITRACE_DEFAULT" mgl_dir="$MGL_DIR_DEFAULT" internal=0
  # Parse our own leading flags until we hit the java binary.
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --trace)          trace="$2"; shift 2;;
      --apittrace)      apittrace="$2"; shift 2;;
      --mgl-dir)        mgl_dir="$2"; shift 2;;
      --trace-internal) internal=1; shift;;
      --) shift; break;;
      -*) die "unknown flag: $1";;
      *) break;;   # first non-flag = java binary
    esac
  done

  # Fill in from state file (set by `enable`) for anything not passed explicitly.
  if [[ -f "$STATE_FILE" ]]; then
    # shellcheck disable=SC1090
    source "$STATE_FILE"
    [[ -z "$trace" ]]     && trace="$TRACE_FILE"
    [[ -z "$apittrace" ]] && : # APITRACE default already set
    [[ -z "$mgl_dir" ]]   && mgl_dir="$MGL_DIR"
    [[ $internal -eq 0 && -n "${TRACE_INTERNAL:-}" ]] && internal="$TRACE_INTERNAL"
    [[ -n "${APITRACE:-}" ]] && apittrace="$APITRACE"
  fi

  if [[ -z "$trace" ]]; then
    # No --trace and no state file (e.g. WrapperCommand was set to just this
    # script path): auto-generate a timestamped trace so capture still works.
    trace="$REPO_ROOT/build/apitrace_captures/capture_$(date +%Y%m%d_%H%M%S).trace"
    echo "  (no --trace and no state file; auto trace -> $trace)" >&2
  fi
  [[ $# -ge 1 ]]    || die "capture expects the java command after the flags"

  local MGL="$mgl_dir/libmgl.dylib"
  local MGL_ES="$mgl_dir/libmgl_es.dylib"
  local GLFW="$mgl_dir/libglfw.dylib"
  local APITRACE_GL="$apittrace/wrappers/OpenGL.framework/Versions/A/OpenGL"
  [[ -f "$APITRACE_GL" ]] || die "apitrace OpenGL wrapper missing: $APITRACE_GL"
  [[ -f "$MGL" ]]         || die "MGL missing: $MGL"

  mkdir -p "$(dirname "$trace")"
  rm -f "$trace"

  export TRACE_LIBGL="$MGL"
  export TRACE_FILE="$trace"
  export MTL_HUD_ENABLED="${MTL_HUD_ENABLED:-1}"
  # CRITICAL: never point DYLD_FRAMEWORK_PATH at apitrace wrappers — on macOS
  # it hijacks even absolute /System/.../OpenGL loads -> "symbol lookup recursion".
  unset DYLD_FRAMEWORK_PATH
  unset DYLD_LIBRARY_PATH

  # Optional parallel MGL-internal trace (Y-flip / RT write marks).
  if [[ $internal -eq 1 ]]; then
    export MGL_TRACE_LOG=1
    export MGL_TRACE_LEVEL=3
    export MGL_TRACE_RT_WRITE_MARKS=1
    export MGL_TRACE_FILE="$(dirname "$trace")/mgl-trace-$$.log"
  fi

  # Rewrite LWJGL lib paths so capture works even if JvmArgs still point at libmgl.
  local args=() have_gl=0 have_glfw=0 have_gles=0
  for a in "$@"; do
    case "$a" in
      -Dorg.lwjgl.opengl.libname=*)   args+=("-Dorg.lwjgl.opengl.libname=${APITRACE_GL}"); have_gl=1;;
      -Dorg.lwjgl.glfw.libname=*)     args+=("-Dorg.lwjgl.glfw.libname=${GLFW}");           have_glfw=1;;
      -Dorg.lwjgl.opengles.libname=*) args+=("-Dorg.lwjgl.opengles.libname=${MGL_ES}");     have_gles=1;;
      *) args+=("$a");;
    esac
  done
  if [[ $have_gl -eq 0 || $have_glfw -eq 0 || $have_gles -eq 0 ]]; then
    local java_bin="${args[0]}" rest=("${args[@]:1}") inject=()
    [[ $have_gl -eq 0 ]]   && inject+=("-Dorg.lwjgl.opengl.libname=${APITRACE_GL}")
    [[ $have_glfw -eq 0 ]] && inject+=("-Dorg.lwjgl.glfw.libname=${GLFW}")
    [[ $have_gles -eq 0 ]] && inject+=("-Dorg.lwjgl.opengles.libname=${MGL_ES}")
    args=("$java_bin" "${inject[@]}" "${rest[@]}")
  fi

  echo "CAPTURE MODE ON (wrapper)"
  echo "  TRACE_LIBGL=$TRACE_LIBGL"
  echo "  TRACE_FILE=$TRACE_FILE"
  if [[ $internal -eq 1 ]]; then echo "  MGL_TRACE_FILE=$MGL_TRACE_FILE"; fi
  echo "  opengl.libname=$APITRACE_GL"
  echo "  pid=$$ -> exec java"
  exec "${args[@]}"
}

case "${1:-}" in
  enable)  shift; cmd_enable "$@";;
  disable) shift; cmd_disable "$@";;
  capture) shift; cmd_capture "$@";;
  "") die "usage: $0 {enable|disable|capture} [options]";;
  *) # Invoked as a Prism WrapperCommand without the 'capture' subcommand
     # (e.g. WrapperCommand = this script path). Treat args as <java> <args...>.
     cmd_capture "$@";;
esac
