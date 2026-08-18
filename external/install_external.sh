#!/bin/bash
# Install the repository-local modified GLFW build when a system installation
# is explicitly required.  Normal MGL builds consume the local build directly;
# header-only metal-cpp requires no installation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/glfw"
cd build
sudo make install
cd ../..
