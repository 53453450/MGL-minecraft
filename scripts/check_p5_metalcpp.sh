#!/bin/bash
# P5 hard gate for the single-path Metal-cpp renderer.
set -u

failures=0
fail() {
    printf 'check-p5-metalcpp FAIL: %s\n' "$1"
    failures=$((failures + 1))
}

gate_hits=$(rg -n 'MGL_USE_METALCPP' MGL/src MGL/include || true)
if [ -n "$gate_hits" ]; then
    fail "production sources still reference MGL_USE_METALCPP"
    printf '%s\n' "$gate_hits" | sed 's/^/    /'
fi

fallback_getter_hits=$(rg -n \
    'mglRenderCppRenderEncoderOwnerGetCurrentForFallback' \
    MGL/src MGL/include test_legacy_compat || true)
if [ -n "$fallback_getter_hits" ]; then
    fail "borrowed render-encoder fallback getter still exists"
    printf '%s\n' "$fallback_getter_hits" | sed 's/^/    /'
fi

queue_fallback_hits=$(rg -n '\[_device newCommandQueue' \
    MGL/src/MGLRenderer+Lifecycle.m \
    MGL/src/MGLRenderer+GPURecovery.m || true)
if [ -n "$queue_fallback_hits" ]; then
    fail "Objective-C command-queue fallback still exists"
    printf '%s\n' "$queue_fallback_hits" | sed 's/^/    /'
fi

macro_files=$(rg -l \
    '^#define (NS_PRIVATE_IMPLEMENTATION|CA_PRIVATE_IMPLEMENTATION|MTL_PRIVATE_IMPLEMENTATION)' \
    MGL --glob '*.{c,cc,cpp,cxx,h,m,mm}' | sort || true)
if [ "$macro_files" != "MGL/src/mgl_render_cpp.cpp" ]; then
    fail "Metal-cpp implementation macros are not owned only by mgl_render_cpp.cpp"
    printf '%s\n' "$macro_files" | sed 's/^/    /'
fi

abi_metal_hits=$(perl -0pe 's{/\*.*?\*/}{}gs; s{//[^\n]*}{}g' \
    MGL/src/mgl_render_cpp.h | \
    rg -n 'id[[:space:]]*<MTL|MTL::' || true)
if [ -n "$abi_metal_hits" ]; then
    fail "public C ABI exposes Objective-C or Metal-cpp types"
    printf '%s\n' "$abi_metal_hits" | sed 's/^/    /'
fi

if [ "$failures" -ne 0 ]; then
    printf 'check-p5-metalcpp: %d violation(s)\n' "$failures"
    exit 1
fi

printf 'P5_SINGLE_PATH_GATE_OK\n'
