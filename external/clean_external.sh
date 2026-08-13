#!/bin/bash
# Clean the remaining external dependency build outputs.  The old
# GLSL->SPIR-V->MSL toolchain (SPIRV-Tools / SPIRV-Cross / SPIRV-Headers /
# glslang) is no longer built or linked by MGL.
set -e

cd glfw
cd build
make clean
cd ../..