#!/bin/bash
# Update an independently cloned Apple metal-cpp checkout.  GLFW is the
# repository-local modified checkout and must not be pulled from, or compared
# with, upstream.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

METAL_CPP_DIR="metal-cpp"
METAL_CPP_URL="https://github.com/apple/metal-cpp.git"

if [[ ! -d "$METAL_CPP_DIR" ]]; then
    git clone --depth 1 "$METAL_CPP_URL" "$METAL_CPP_DIR"
elif [[ -e "$METAL_CPP_DIR/.git" ]]; then
    git -C "$METAL_CPP_DIR" pull --ff-only
else
    printf 'using vendored metal-cpp snapshot; no update performed\n'
fi

if [[ ! -d glfw ]]; then
    printf 'error: external/glfw is missing; the local modified checkout is required\n' >&2
    exit 1
fi
printf 'leaving local glfw checkout unchanged (no remote pull)\n'
