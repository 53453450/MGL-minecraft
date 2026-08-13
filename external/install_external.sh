#!/bin/bash
# Install the remaining external dependencies into the system.  The old
# GLSL->SPIR-V->MSL toolchain (SPIRV-Tools / SPIRV-Cross / SPIRV-Headers /
# glslang) is no longer built, linked, or installed by MGL.  GLFW is
# typically consumed from its local build directory instead.
set -e

cd glfw
cd build
sudo make install
cd ../..