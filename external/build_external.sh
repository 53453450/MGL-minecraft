#!/bin/bash
# Prepare MGL's external build dependencies.  metal-cpp is header-only, so the
# script fetches it when missing and builds only the repository-local modified
# GLFW checkout.
set -euo pipefail

# Run from any cwd: the script resolves its own directory.
cd "$(dirname "$0")"

if [[ ! -d metal-cpp ]]; then
    git clone --depth 1 https://github.com/apple/metal-cpp.git metal-cpp
fi

if [[ ! -d glfw ]]; then
    printf 'error: external/glfw is missing; refusing to clone an upstream GLFW\n' >&2
    exit 1
fi

SDKROOT=$(xcrun --show-sdk-path)
export SDKROOT

# GLFW keeps its own thin facades in glfw/src/{MGLContext,MGLRenderer}.h.
# This is the repository-local modified checkout; no git fetch/pull is run.
cd glfw
mkdir -p build
cd build
if [[ ! -f CMakeCache.txt ]]; then
    cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
fi
# Prefer cmake --build so source edits rebuild incrementally instead of the
# top-level Makefile treating libglfw3.a as always up to date.
cmake --build . --target glfw -j "$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"
cd ../..
