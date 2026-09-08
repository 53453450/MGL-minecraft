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
echo "==> Draw cluster (audit A1/A3: measure cluster, not DrawSupport alone)"
wc -l MGLRenderer+DrawSupport.m MGLRenderer+DrawStageHost.m MGLRenderer+Draw.m \
  MGLRenderer+Tessellation.m mgl_draw_metal_port.m mgl_draw_gs_metal.cpp \
  2>/dev/null || true
DRAW_CLUSTER=$(wc -l MGLRenderer+DrawSupport.m MGLRenderer+DrawStageHost.m \
  MGLRenderer+Draw.m MGLRenderer+Tessellation.m mgl_draw_metal_port.m \
  mgl_draw_gs_metal.cpp 2>/dev/null | tail -1 | awk '{print $1}')
echo "Draw cluster total: ${DRAW_CLUSTER}"

echo
echo "==> Batch cluster (A3 / O2.5 target Batch*.m < 600)"
wc -l MGLRenderer+Batch.m MGLRenderer+BatchReplay.m 2>/dev/null || true
BATCH_CLUSTER=$(wc -l MGLRenderer+Batch.m MGLRenderer+BatchReplay.m 2>/dev/null | tail -1 | awk '{print $1}')
echo "Batch cluster total: ${BATCH_CLUSTER}"
echo "(informational) batch domain C/C++ + diag/encode split:"
wc -l mgl_batch_replay.cpp mgl_batch_restore.c mgl_batch_path.c mgl_batch_hazard.c \
  mgl_batch_issue.c mgl_batch_rt_mark.c \
  mgl_batch_replay_trace.m mgl_batch_issue_encode.m mgl_batch_icb_mdi_encode.m \
  mgl_batch_rt_mark_port.m 2>/dev/null || true

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
