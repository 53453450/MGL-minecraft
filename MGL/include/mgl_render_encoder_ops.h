/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_render_encoder_ops.h - C homes of -newRenderEncoderLockedWithReason: and
 * the five attachment/load-store middle layers it drives (P0-1, log 186).
 */

#ifndef MGL_RENDER_ENCODER_OPS_H
#define MGL_RENDER_ENCODER_OPS_H

#include "glm_context.h"

#include <stdint.h>

struct Framebuffer_t;

/* -checkDrawBufferSize: */
int mglRenderPassCheckDrawBufferSize(void *renderer, unsigned int index);

/* -configureDefaultFramebufferAttachmentsLocked */
int mglRenderPassConfigureDefaultFramebufferAttachments(void *renderer);

/* -ensureTransientDepthForDefaultFramebufferLocked */
void mglRenderPassEnsureTransientDepthForDefaultFramebuffer(void *renderer);

/* -configureUserFBOLoadStoreActionsLocked:fboColorClearMask:
 *  fboColorAttachment0ClearMask: */
void mglRenderPassConfigureUserFBOLoadStoreActions(
    void *renderer, unsigned int *outFboColorClearCount,
    unsigned int *outFboColorClearMask,
    unsigned int *outFboColorAttachment0ClearMask);

/* -configureDefaultFramebufferLoadStoreActionsLocked */
void mglRenderPassConfigureDefaultFramebufferLoadStoreActions(void *renderer);

/* -logRenderPassClearResolveLocked:... */
void mglRenderPassLogClearResolve(
    void *renderer, uint64_t renderEncoderCall, int traceRenderEncoder,
    unsigned int fboColorClearCount, unsigned int fboColorClearMask,
    unsigned int fboColorAttachment0ClearMask,
    unsigned int fboDepthClearMaskBefore,
    unsigned int fboStencilClearMaskBefore, unsigned int defaultClearMask,
    struct Framebuffer_t *fbo);

/* -newRenderEncoderLockedWithReason: */
int mglRenderPassNewRenderEncoderLockedWithReason(void *renderer,
                                                  uint32_t reason);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDER_ENCODER_OPS_H */
