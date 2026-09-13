/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_tess_compute_ops.c — see the header.  The NSData keep-alive became CFData:
 * CFDataCreate returns +1, which is exactly what mglRendererTemporariesAdd
 * consumes (the Objective-C version had the array retain the NSData and ARC
 * release the strong local at scope exit, so the plan's byte pointer stayed
 * valid because the array still held it - same here).
 */

#include <stdio.h>

#include <CoreFoundation/CoreFoundation.h>

#include "mgl_tess_compute_ops.h"
#include "mgl_renderer_ports.h"   /* state areas + keep-alive set */
#include "mgl_buffer_slots.h"     /* kMGLPointSizeBufferIndex */
#include "mgl_draw_tess.h"        /* mglTessFillPointSizeParams */
#include "mgl_types_program.h"    /* _MAX_SHADER_TYPES */

bool mglTessAppendComputeBytesOp(MGLRenderComputeExecutionPlan *plan,
                                 void *temporaries, const void *bytes,
                                 size_t length, size_t index)
{
    if (!plan || !temporaries || !bytes || length == 0u ||
        length > UINT32_MAX) {
        return false;
    }
    if (plan->binding_op_count >= MGL_RENDER_COMPUTE_EXECUTION_MAX_OPS) {
        fprintf(stderr, "MGL TESS ERROR: compute bytes-binding overflow (%u)\n",
                (unsigned)plan->binding_op_count);
        return false;
    }
    CFDataRef storage = CFDataCreate(NULL, (const UInt8 *)bytes,
                                     (CFIndex)length);
    if (!storage) {
        return false;
    }
    mglRendererTemporariesAdd(temporaries, (void *)storage);
    plan->binding_ops[plan->binding_op_count++] =
        (MGLRenderComputeBindingOp){
            .kind = 1u,
            .index = (uint32_t)index,
            .offset = 0u,
            .buffer = NULL,
            .bytes = (void *)CFDataGetBytePtr(storage),
            .length = (uint32_t)length,
        };
    /* The set holds its own reference now, like the NSMutableArray did. */
    CFRelease(storage);
    return true;
}

void mglTessBindPointSizeParamsToComputeEncoder(
    void *renderer, Program *program, int stage,
    MGLRenderComputeExecutionPlan *plan, void *temporaries)
{
    if (!plan || !program || stage < 0 || stage >= _MAX_SHADER_TYPES) {
        return;
    }
    if (!program->uses_point_size_params) {
        return;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMState *state = areas.core && areas.core->activeState
        ? areas.core->activeState
        : (areas.ctx ? areas.ctx->active_state : NULL);

    float pointSizeParams[2] = {0.f, 0.f};
    mglTessFillPointSizeParams(
        state && state->var.point_size > 0.0f ? state->var.point_size : 0.0f,
        state && state->caps.program_point_size ? 1 : 0,
        pointSizeParams);
    (void)mglTessAppendComputeBytesOp(plan, temporaries, pointSizeParams,
                                      sizeof(pointSizeParams),
                                      kMGLPointSizeBufferIndex);
}
