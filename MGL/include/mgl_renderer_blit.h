/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Framebuffer blit orchestration (C++ home).
 */

#ifndef MGL_RENDERER_BLIT_H
#define MGL_RENDERER_BLIT_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

void mglRenderBlitFramebuffer(
    GLMContext context,
    int32_t src_x0, int32_t src_y0, int32_t src_x1, int32_t src_y1,
    int32_t dst_x0, int32_t dst_y0, int32_t dst_x1, int32_t dst_y1,
    uint32_t mask, uint32_t filter);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_BLIT_H */
