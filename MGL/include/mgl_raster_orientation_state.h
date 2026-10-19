/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_raster_orientation_state.h -- the raster orientation plan of the
 * current GL state (see mgl_raster_orientation_plan.h).
 *
 * mglRasterTargetStoresGLOrder is the single place that decides whether the
 * current draw target stores GL row order: every user framebuffer does, and
 * only the window-system target keeps Metal row order
 * (docs/TODO_FBO_GL_ROW_ORDER_2026-10-18.md).
 */

#ifndef MGL_RASTER_ORIENTATION_STATE_H
#define MGL_RASTER_ORIENTATION_STATE_H

#include "glm_context.h"
#include "mgl_raster_orientation_plan.h"

static inline int mglRasterTargetStoresGLOrder(const GLMState *state)
{
    /* A user framebuffer's attachments; the default framebuffer is the
     * window-system target. */
    return state && state->framebuffer != NULL;
}

static inline MGLRasterOrientationPlan
mglRasterOrientationForState(const GLMState *state)
{
    MGLRasterOrientationInput in;
    in.clip_origin = (uint32_t)state->var.clip_origin;
    in.target_gl_order = mglRasterTargetStoresGLOrder(state);
    MGLRasterOrientationPlan plan;
    (void)mglRasterOrientationPlanBuild(&in, &plan);
    return plan;
}

#endif /* MGL_RASTER_ORIENTATION_STATE_H */
