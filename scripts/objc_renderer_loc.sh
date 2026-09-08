#!/usr/bin/env bash
#
# objc_renderer_loc.sh — measure ObjC renderer category thickness.
#
# Part of docs/OBJC_CATEGORY_DISMANTLE_TODO.md Batch O0.2.
# Target: MGLRenderer*.m total ≤ 8–12k LOC (thin platform shell).
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$PROJECT_DIR/MGL/src"

cd "$SRC"

echo "==> MGLRenderer*.m line counts"
wc -l MGLRenderer*.m | sort -n

echo
echo "==> Related thick ObjC (draw encode / shell; informational)"
wc -l mgl_draw_encode.m MGLPlatformRendererShell.m MGLRenderPassManager.m \
  MGLPipelineCache.m 2>/dev/null || true

echo
TOTAL=$(wc -l MGLRenderer*.m | tail -1 | awk '{print $1}')
echo "MGLRenderer*.m total: ${TOTAL}"
echo "Target from OBJC_CATEGORY_DISMANTLE_TODO: ≤ 8–12k (platform shell + thin materialization)"
if [[ "$TOTAL" -gt 12000 ]]; then
  echo "Status: ABOVE target (still thick; continue O1–O6 sinks)"
elif [[ "$TOTAL" -gt 8000 ]]; then
  echo "Status: within upper band (8–12k)"
else
  echo "Status: at or below lower target (≤8k)"
fi
