/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_clear_buffer_ops.h - C home of -mtlClearBuffer:type:mask: and the two
 * default-draw-buffer texture creators (P0-1, log 184).
 */

#ifndef MGL_CLEAR_BUFFER_OPS_H
#define MGL_CLEAR_BUFFER_OPS_H

#include "glm_context.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -newDrawBuffer:isDepthStencil: (drawable-sized). */
void *mglRendererNewDrawBuffer(void *renderer, uint32_t pixelFormat,
                               int depthStencil);

/* -newDrawBufferWithCustomSize:isDepthStencil:customSize:.  Both return the new
 * Metal texture (+1) or NULL. */
void *mglRendererNewDrawBufferWithCustomSize(uint32_t pixelFormat,
                                             int depthStencil, uint64_t width,
                                             uint64_t height);

/* -mtlClearBuffer:type:mask:. */
void mglRendererMTLClearBuffer(void *renderer, GLMContext glm_ctx,
                               unsigned int type, unsigned int mask);

#ifdef __cplusplus
}
#endif

#endif /* MGL_CLEAR_BUFFER_OPS_H */
