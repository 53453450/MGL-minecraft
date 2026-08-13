#!/bin/bash
# P3 baseline recorder: snapshot the full local gate output before/after a P3
# batch so A/B parity can be compared per sub-batch.  Logs land in
# build/p3_baseline/ (not tracked).  Usage: bash scripts/record_p3_baseline.sh <label>
set -u

label=${1:-baseline}
out_dir=${PWD}/build/p3_baseline
mkdir -p "$out_dir"

run() {
    local name="$1"; shift
    printf '==== %s ====\n' "$name" | tee -a "$out_dir/${label}_00_summary.log"
    "$@" > "$out_dir/${label}_${name}.log" 2>&1
    local rc=$?
    printf '  -> %s exit=%d\n' "$name" "$rc" | tee -a "$out_dir/${label}_00_summary.log"
}

run lib make -j4 lib
run test-mglair make test-mglair
run test-mglair-gtest make test-mglair-gtest
run test-metalcpp make test-metalcpp
run test-regression make test-regression
run regression-mc0 env DYLD_LIBRARY_PATH=${PWD}/build MGL_USE_METALCPP=0 \
    ${PWD}/build/test_regression --golden-dir ${PWD}/MGL_Golden_Images
run regression-mc1 env DYLD_LIBRARY_PATH=${PWD}/build MGL_USE_METALCPP=1 \
    ${PWD}/build/test_regression --golden-dir ${PWD}/MGL_Golden_Images

printf 'P3 baseline %s done: %s\n' "$label" "$out_dir" | tee -a "$out_dir/${label}_00_summary.log"