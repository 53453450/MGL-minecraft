#!/bin/bash
# Clone or update external/OpenGL-Registry to the commit in MGL/generated/registry.lock.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOCK="$ROOT/MGL/generated/registry.lock"
DEST="$ROOT/external/OpenGL-Registry"

url=$(sed -n 's/^url=//p' "$LOCK" | head -n1)
commit=$(sed -n 's/^commit=//p' "$LOCK" | head -n1)
if [[ -z "$url" || -z "$commit" ]]; then
    echo "error: $LOCK missing url= or commit=" >&2
    exit 1
fi

if [[ ! -d "$DEST/.git" ]]; then
    rm -rf "$DEST"
    git clone "$url" "$DEST"
fi

git -C "$DEST" fetch --depth 1 origin "$commit" 2>/dev/null || git -C "$DEST" fetch origin "$commit"
git -C "$DEST" checkout --detach "$commit"
test -f "$DEST/xml/gl.xml"
echo "OpenGL-Registry at $commit"
