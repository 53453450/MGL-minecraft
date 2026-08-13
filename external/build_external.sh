#!/bin/bash
# Build the remaining external dependencies.  The old GLSL->SPIR-V->MSL toolchain
# (SPIRV-Tools / SPIRV-Cross / SPIRV-Headers / glslang) is no longer built or
# linked by MGL; only GLFW and ezxml are still needed.  See the top-level
# Makefile for the aux-shader metallib generation (Apple SDK only).
set -e

SDKROOT=$(xcrun --show-sdk-path)
export SDKROOT

# GLFW keeps its own thin facades in glfw/src/{MGLContext,MGLRenderer}.h.
cd glfw
mkdir -p build
cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5
make -j 4 glfw
cd ../..