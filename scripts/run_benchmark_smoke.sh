#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/build/mgl_benchmark"
DO_BUILD=1

if [[ "${1:-}" == "--no-build" ]]; then
  DO_BUILD=0
elif [[ $# -gt 0 ]]; then
  echo "unknown argument: $1" >&2
  exit 2
fi

cd "$PROJECT_DIR"
if [[ "$DO_BUILD" -eq 1 ]]; then
  make bench >/dev/null
fi

if [[ ! -x "$BINARY" ]]; then
  echo "benchmark binary not found: $BINARY" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d /tmp/mgl-benchmark-smoke.XXXXXX)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

export DYLD_LIBRARY_PATH="$PROJECT_DIR/build${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

"$BINARY" --list | rg -qx 'minecraft-cpu'
"$BINARY" --test dispatch --json "$TMP_DIR/dispatch.json" >/dev/null
"$BINARY" --test pipeline --json "$TMP_DIR/pipeline.json" >/dev/null
"$BINARY" --test minecraft-cpu --frames 2 --warmup 1 \
  --json "$TMP_DIR/minecraft.json" >/dev/null

plutil -convert binary1 -o /dev/null -- "$TMP_DIR/dispatch.json"
plutil -convert binary1 -o /dev/null -- "$TMP_DIR/pipeline.json"
plutil -convert binary1 -o /dev/null -- "$TMP_DIR/minecraft.json"
rg -q '"test": "Minecraft CPU 1.21"' "$TMP_DIR/minecraft.json"
rg -q '"metric": "P95 Frame ms"' "$TMP_DIR/minecraft.json"

echo "benchmark smoke: PASS"
