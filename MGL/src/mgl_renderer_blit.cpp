/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_blit.h"

extern "C" void mglRendererObjCBlitFramebuffer(
    GLMContext context, int src_x0, int src_y0, int src_x1, int src_y1,
    int dst_x0, int dst_y0, int dst_x1, int dst_y1, unsigned int mask,
    unsigned int filter);

extern "C" void mglRenderBlitFramebuffer(
    GLMContext context,
    int32_t src_x0, int32_t src_y0, int32_t src_x1, int32_t src_y1,
    int32_t dst_x0, int32_t dst_y0, int32_t dst_x1, int32_t dst_y1,
    uint32_t mask, uint32_t filter)
{
    if (!context) {
        return;
    }
    mglRendererObjCBlitFramebuffer(
        context, src_x0, src_y0, src_x1, src_y1, dst_x0, dst_y0, dst_x1, dst_y1,
        mask, filter);
}
