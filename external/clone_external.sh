#!/bin/bash
# Clone the remaining external dependencies.  The old GLSL->SPIR-V->MSL
# toolchain (SPIRV-Tools / SPIRV-Cross / SPIRV-Headers / glslang) is no longer
# built or linked by MGL and must NOT be cloned.
set -e

git clone https://github.com/KhronosGroup/OpenGL-Registry.git --depth 1
git clone https://github.com/lxfontes/ezxml.git --depth 1
git clone https://github.com/glfw/glfw.git --depth 1