/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Extension hook registration + thin hazard/arena facades (Phase 6).
 */

#include "mgl_extension_hooks.h"
#include "glm_context.h"
#include "draw_command.h"

#include <string.h>

enum { kMGLMaxExtStates = 32, kMGLMaxGeomStages = 16 };

typedef struct {
    uint32_t ext_id;
    MGLExtensionStateVTable vt;
} MGLExtStateSlot;

static MGLExtStateSlot s_ext_states[kMGLMaxExtStates];
static size_t s_ext_state_count;
static MGLGeometryPipelineStage s_geom_stages[kMGLMaxGeomStages];
static size_t s_geom_stage_count;

void mglRegisterExtensionState(GLMContext ctx, uint32_t ext_id,
                               const MGLExtensionStateVTable *vt)
{
    (void)ctx;
    if (!vt || s_ext_state_count >= kMGLMaxExtStates) {
        return;
    }
    for (size_t i = 0; i < s_ext_state_count; i++) {
        if (s_ext_states[i].ext_id == ext_id) {
            s_ext_states[i].vt = *vt;
            return;
        }
    }
    s_ext_states[s_ext_state_count].ext_id = ext_id;
    s_ext_states[s_ext_state_count].vt = *vt;
    s_ext_state_count++;
}

void mglRegisterGeometryStage(GLMContext ctx,
                              const MGLGeometryPipelineStage *st)
{
    (void)ctx;
    if (!st || !st->name || s_geom_stage_count >= kMGLMaxGeomStages) {
        return;
    }
    for (size_t i = 0; i < s_geom_stage_count; i++) {
        if (s_geom_stages[i].name &&
            strcmp(s_geom_stages[i].name, st->name) == 0) {
            s_geom_stages[i] = *st;
            return;
        }
    }
    s_geom_stages[s_geom_stage_count++] = *st;
}

size_t mglGeometryStageCount(void)
{
    return s_geom_stage_count;
}

const MGLGeometryPipelineStage *mglGeometryStageAt(size_t index)
{
    if (index >= s_geom_stage_count) {
        return NULL;
    }
    return &s_geom_stages[index];
}

int mglPendingDrawsReferenceBuffer(GLMContext ctx, struct Buffer *buf)
{
    return mglPendingDrawsReadBufferRange(ctx, buf, 0, -1) ? 1 : 0;
}

void *mglFrameArenaAlloc(GLMContext ctx, size_t n)
{
    if (!ctx || !ctx->batch_arena || !ctx->batch_arena->enabled || n == 0) {
        return NULL;
    }
    return mglBatchArenaAlloc(ctx->batch_arena, n);
}
