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
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j 4 glfw
cd ../..
