/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_raster_orientation_plan.h -- which way the rasterizer's Y runs for a
 * draw, as plain values (pure C, no Metal, no renderer state).
 *
 * GL window Y grows upward, Metal's texture rows grow downward.  Rendering
 * into a window-system target keeps Metal row order (row 0 = top); rendering
 * into an FBO attachment is meant to store GL row order (row 0 = bottom), which
 * takes a negated Y in the vertex stage, an inverted winding and viewport /
 * scissor rectangles placed by GL's y.  ARB_clip_control's UPPER_LEFT origin
 * is a second, independent Y inversion (negative viewport height plus an
 * inverted winding).  This plan folds both inputs into the decisions the
 * encoder state code and the vertex stage need, so they are made in one place
 * (docs/TODO_FBO_GL_ROW_ORDER_2026-10-18.md).
 *
 * clip_origin takes the GL enum values (GL_LOWER_LEFT 0x8CA1,
 * GL_UPPER_LEFT 0x8CA2).
 */

#ifndef MGL_RASTER_ORIENTATION_PLAN_H
#define MGL_RASTER_ORIENTATION_PLAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLRasterOrientationInput {
    uint32_t clip_origin;  /* GL_LOWER_LEFT or GL_UPPER_LEFT */
    int target_gl_order;   /* the draw target stores GL row order (an FBO
                              attachment under the redesign); 0 = window-system
                              target in Metal row order */
} MGLRasterOrientationInput;

typedef struct MGLRasterOrientationPlan {
    int scissor_from_top;          /* scissor y = pass_h - (y + h) */
    int viewport_from_top;         /* viewport y = pass_h - (y + h), clamped at 0 */
    int negative_viewport_height;  /* origin moves to the far edge, height negates */
    int invert_winding;            /* flip the Metal front-face winding */
    float vs_y_scale;              /* factor the vertex stage applies to
                                      gl_Position.y: -1 for GL-order targets */
} MGLRasterOrientationPlan;

/* Plan a draw's orientation.  Returns 0; -1 for a NULL argument. */
int mglRasterOrientationPlanBuild(const MGLRasterOrientationInput *in,
                                  MGLRasterOrientationPlan *out);

/* Metal scissor Y for a GL scissor box (y, h) in a pass of height pass_h. */
int32_t mglRasterPlanScissorY(const MGLRasterOrientationPlan *plan, int32_t y,
                              int32_t h, uint32_t pass_h);

/* Metal viewport origin Y and height for a GL viewport (y, h), in place. */
void mglRasterPlanViewportY(const MGLRasterOrientationPlan *plan, double *y,
                            double *h, uint32_t pass_h);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RASTER_ORIENTATION_PLAN_H */
