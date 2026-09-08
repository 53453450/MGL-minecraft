#!/usr/bin/env bash
#
# objc_renderer_loc.sh — measure ObjC renderer category thickness.
#
# Part of docs/OBJC_CATEGORY_DISMANTLE_TODO.md Batch O0.2.
# Target: MGLRenderer*.m total ≤ 8–12k LOC (thin platform shell).
#
# Track B (metrics honesty): Batch cluster MUST include same-category
# encode ports (mgl_batch_*_encode.m), replay_trace, and sibling batch
# encode ports — not only MGLRenderer+Batch*.m. O2.5 literal (Batch*.m
# < 600) is necessary but not sufficient for ObjC cleanup.
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
echo "==> Batch cluster (Track B: categories + encode ports + trace)"
echo "    O2.5 literal gate: MGLRenderer+Batch*.m only (< 600)"
wc -l MGLRenderer+Batch.m MGLRenderer+BatchReplay.m 2>/dev/null || true
BATCH_LITERAL=$(wc -l MGLRenderer+Batch.m MGLRenderer+BatchReplay.m 2>/dev/null | tail -1 | awk '{print $1}')
echo "Batch*.m literal total: ${BATCH_LITERAL}"

echo
echo "    Honest Batch ObjC cluster (includes encode / trace / sibling ports):"
# Glob encode ports + explicit siblings so new mgl_batch_*_encode.m are counted.
BATCH_OBJC_FILES=(
  MGLRenderer+Batch.m
  MGLRenderer+BatchReplay.m
  mgl_batch_replay_trace.m
  mgl_batch_rt_mark_port.m
)
for f in mgl_batch_*_encode.m; do
  [[ -f "$f" ]] && BATCH_OBJC_FILES+=("$f")
done
wc -l "${BATCH_OBJC_FILES[@]}" 2>/dev/null || true
BATCH_CLUSTER=$(wc -l "${BATCH_OBJC_FILES[@]}" 2>/dev/null | tail -1 | awk '{print $1}')
echo "Batch ObjC cluster total (honest): ${BATCH_CLUSTER}"
echo "NOTE: O2.5 literal passed iff Batch*.m < 600; ObjC cleanup NOT done while"
echo "      encode/trace still hold multi-k ObjC in this cluster."

echo
echo "(informational) batch domain C/C++ plans (not in ObjC cluster):"
wc -l mgl_batch_replay.cpp mgl_batch_mtl_encode.cpp mgl_batch_restore.c \
  mgl_batch_path.c mgl_batch_hazard.c mgl_batch_issue.c mgl_batch_rt_mark.c \
  2>/dev/null || true

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
