#!/bin/bash
# Clean build outputs produced from the repository-local modified GLFW.
# metal-cpp is header-only and has no build output to remove.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/glfw"
cd build
make clean
cd ../..
