/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_raster_orientation_plan.c -- see mgl_raster_orientation_plan.h.
 */

#include "mgl_raster_orientation_plan.h"

#include <stddef.h>
#include <stdlib.h>

#define MGL_RO_GL_UPPER_LEFT 0x8CA2u

int mglFBOGLOrderEnabled(void)
{
    static int cached = -1;
    if (cached < 0) {
        const char *v = getenv("MGL_FBO_GL_ORDER");
        cached = (v && v[0] == '0') ? 0 : 1;
    }
    return cached;
}

int mglRasterOrientationPlanBuild(const MGLRasterOrientationInput *in,
                                  MGLRasterOrientationPlan *out)
{
    if (!in || !out) {
        return -1;
    }
    const int upper_left = in->clip_origin == MGL_RO_GL_UPPER_LEFT;
    const int gl_order = in->target_gl_order != 0;

    out->viewport_from_top = !gl_order;
    out->scissor_from_top = !gl_order && !upper_left;
    out->negative_viewport_height = upper_left != gl_order;
    out->invert_winding = upper_left != gl_order;
    out->fragcoord_from_top = !upper_left && !gl_order;
    return 0;
}

int32_t mglRasterPlanScissorY(const MGLRasterOrientationPlan *plan, int32_t y,
                              int32_t h, uint32_t pass_h)
{
    if (!plan || !plan->scissor_from_top) {
        return y;
    }
    const int32_t metal_y = (int32_t)pass_h - (y + h);
    return metal_y < 0 ? 0 : metal_y;
}

void mglRasterPlanViewportY(const MGLRasterOrientationPlan *plan, double *y,
                            double *h, uint32_t pass_h)
{
    if (!plan || !y || !h) {
        return;
    }
    if (plan->viewport_from_top) {
        const double metal_y = (double)pass_h - (*y + *h);
        *y = metal_y < 0.0 ? 0.0 : metal_y;
    }
    if (plan->negative_viewport_height && *h != 0.0) {
        *y = *y + *h;
        *h = -*h;
    }
}
