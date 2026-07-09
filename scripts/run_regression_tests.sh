#!/usr/bin/env bash
#
# run_regression_tests.sh — one-command regression runner for MGL draw pipeline.
#
# Stage 0.1 of docs/RENDERER_EVOLUTION_TODO.md.
#
# Usage:
#   scripts/run_regression_tests.sh           # build + run + compare vs golden
#   scripts/run_regression_tests.sh --update  # rebuild golden images from current output
#   scripts/run_regression_tests.sh --no-build# skip build step
#
# Exit code: 0 if all PASS, 1 if any FAIL or build error.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_DIR/build/test_regression"
GOLDEN_DIR="$PROJECT_DIR/MGL_Golden_Images"

DO_UPDATE=0
DO_BUILD=1

for arg in "$@"; do
  case "$arg" in
    --update|-u) DO_UPDATE=1 ;;
    --no-build)  DO_BUILD=0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

cd "$PROJECT_DIR"

# --- Build ---
if [[ "$DO_BUILD" -eq 1 ]]; then
  echo "==> Building libmgl (make lib)..."
  make lib >/dev/null 2>&1 || { echo "make lib FAILED" >&2; exit 1; }
  echo "==> Building regression suite (make test-regression)..."
  make test-regression >/dev/null 2>&1 || { echo "make test-regression FAILED" >&2; exit 1; }
fi

if [[ ! -x "$BINARY" ]]; then
  echo "ERROR: binary not found at $BINARY" >&2
  exit 1
fi

# --- Run ---
GLFW_LIB="$(brew --prefix glfw 2>/dev/null)/lib"
export DYLD_LIBRARY_PATH="$PROJECT_DIR/build:${GLFW_LIB}"

if [[ "$DO_UPDATE" -eq 1 ]]; then
  echo "==> Updating golden images in $GOLDEN_DIR/ ..."
  "$BINARY" --update --golden-dir "$GOLDEN_DIR"
  rc=$?
else
  echo "==> Running regression suite (comparing vs $GOLDEN_DIR/) ..."
  "$BINARY" --golden-dir "$GOLDEN_DIR"
  rc=$?
fi

if [[ $rc -eq 0 ]]; then
  echo "✅ All regression tests PASS"
else
  echo "❌ Some regression tests FAILED (exit $rc)"
fi

exit $rc
