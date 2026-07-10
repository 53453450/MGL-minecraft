#!/usr/bin/env bash
#
# stress_loop.sh — run the regression suite N times to catch nondeterminism.
#
# Stage 5.2 acceptance ("multiple runs pixel-stable, no flaky"). The suite
# already does byte-exact golden comparison; this script just loops it and
# reports which runs (if any) diverged. Catches the class of bug parallel
# command recording is most likely to introduce (races surfacing as
# nondeterministic output).
#
# Usage:
#   scripts/stress_loop.sh [N]        # default N=1000
#   scripts/stress_loop.sh 50         # quick smoke
#   scripts/stress_loop.sh 1000 --no-build
#   scripts/stress_loop.sh 200 --asan # ASan+UBSan instrumented build
#
# Pass --no-build to reuse the already-built binary (required for speed —
# with build each run is dominated by compile time).
# Pass --asan to auto-copy config.mk.example → config.mk, rebuild with
# sanitizers, run N iterations, then restore config.mk.
#
# Exit code: 0 if no run failed, 1 if any run failed.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

N=1000
NO_BUILD=0
ASAN=0
for arg in "$@"; do
  case "$arg" in
    --no-build) NO_BUILD=1 ;;
    --asan)     ASAN=1 ;;
    *)
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        N="$arg"
      else
        echo "unknown arg: $arg" >&2; exit 2
      fi
      ;;
  esac
done

RUNNER="$PROJECT_DIR/scripts/run_regression_tests.sh"

# --- ASan mode: instrumented build + env + cleanup ---
CONFIG_MK="$PROJECT_DIR/config.mk"
CLEANUP_CONFIG=0

if [[ "$ASAN" -eq 1 ]]; then
  echo "==> ASan mode: enabling AddressSanitizer + UBSan"
  if [[ ! -f "$PROJECT_DIR/config.mk.example" ]]; then
    echo "ERROR: config.mk.example not found" >&2; exit 1
  fi
  cp "$PROJECT_DIR/config.mk.example" "$CONFIG_MK"
  CLEANUP_CONFIG=1
  # Force rebuild even if --no-build was passed — sanitized binary required.
  NO_BUILD=0
  # ASan runtime options: don't abort on first error (we want flaky detection),
  # allow MGL's signal handler, disable leak detection (Metal frameworks leak).
  export ASAN_OPTIONS="${ASAN_OPTIONS:-}detect_leaks=0:abort_on_error=0:allow_user_segv_handler=1"
  # Clean build to ensure everything is re-instrumented.
  echo "==> Cleaning and rebuilding with sanitizers..."
  (cd "$PROJECT_DIR" && make clean) >/dev/null 2>&1
fi

# Ensure config.mk is restored on exit (ASan mode only).
cleanup() {
  if [[ "$CLEANUP_CONFIG" -eq 1 && -f "$CONFIG_MK" ]]; then
    rm -f "$CONFIG_MK"
    echo "==> Cleaned up config.mk"
  fi
}
trap cleanup EXIT

# First run: build (unless --no-build) and establish the baseline.
FIRST_ARGS=""
[[ "$NO_BUILD" -eq 1 ]] && FIRST_ARGS="--no-build"

echo "==> stress_loop: $N iterations"
"$RUNNER" $FIRST_ARGS >/tmp/stress_run_1.log 2>&1 || {
  echo "❌ run 1 FAILED — not entering loop:" >&2
  cat /tmp/stress_run_1.log >&2
  exit 1
}

fail_count=0
failed_runs=()
for i in $(seq 2 "$N"); do
  if "$RUNNER" --no-build >/tmp/stress_run_$i.log 2>&1; then
    printf "\r  run %d/%d OK    " "$i" "$N"
  else
    printf "\r  run %d/%d FAIL  " "$i" "$N"
    fail_count=$((fail_count + 1))
    failed_runs+=("$i")
  fi
done
echo ""
echo "========================================"
echo "  stress_loop: $N runs, $fail_count failed"
if [[ $fail_count -gt 0 ]]; then
  echo "  failed runs: ${failed_runs[*]}"
  echo "  (see /tmp/stress_run_<n>.log)"
fi
echo "========================================"

exit $([[ $fail_count -eq 0 ]] && echo 0 || echo 1)
