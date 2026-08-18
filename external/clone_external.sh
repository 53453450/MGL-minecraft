#!/bin/bash
# Fetch missing source dependencies, including Apple's header-only metal-cpp.
# GLFW is intentionally excluded: external/glfw is the repository-local
# modified checkout and must never be replaced by an upstream clone.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

clone_if_missing() {
    local name="$1"
    local url="$2"

    if [[ -e "$name" ]]; then
        [[ -d "$name" ]] || {
            printf 'error: %s exists but is not a directory\n' "$name" >&2
            return 1
        }
        printf 'using existing %s\n' "$name"
        return 0
    fi

    git clone --depth 1 "$url" "$name"
}

clone_if_missing OpenGL-Registry https://github.com/KhronosGroup/OpenGL-Registry.git
clone_if_missing ezxml https://github.com/lxfontes/ezxml.git
clone_if_missing metal-cpp https://github.com/apple/metal-cpp.git

if [[ ! -d glfw ]]; then
    printf 'error: external/glfw is missing; it is a required local modified checkout and is not cloned from upstream\n' >&2
    exit 1
fi
printf 'using local glfw checkout (no remote fetch)\n'
