#!/bin/bash
# Update the remaining external dependencies.  The old GLSL->SPIR-V->MSL
# toolchain (SPIRV-Tools / SPIRV-Cross / SPIRV-Headers / glslang) is no longer
# built or linked by MGL.
set -e

cd glfw
git pull
cd ..