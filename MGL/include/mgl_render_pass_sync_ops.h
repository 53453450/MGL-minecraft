/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_render_pass_sync_ops.h - C home of the render-pass sync / close leaf
 * methods of MGLRenderer+RenderPass.m (P0-1, log 191):
 *
 *   -currentRenderPassUsesTexture:
 *   -currentRenderPassMatchesCurrentFramebuffer
 *   -endRenderPassIfFramebufferChangedForNonDraw:
 *   -synchronizeRenderPassForTextureReadback:reason:
 *   -syncRenderPassStateForContext:
 *   -rotateRenderEncoderForCurrentFramebufferLocked
 *
 * Six methods whose only Objective-C was the renderer receiver (and, for the
 * readback sync, the @try/@catch around the commit): with them in C, five port
 * wrappers in mgl_renderer_ports.h retire together.
 */

#ifndef MGL_RENDER_PASS_SYNC_OPS_H
#define MGL_RENDER_PASS_SYNC_OPS_H

#include "glm_context.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -currentRenderPassUsesTexture:.  Returns 1 when the current render pass
 * references `texture`. */
int mglRenderPassCurrentRenderPassUsesTexture(void *renderer, void *texture);

/* -currentRenderPassMatchesCurrentFramebuffer.  The FBO-match cache lives in
 * the pass manager, so a cached answer short-circuits the full comparison. */
int mglRenderPassMatchesCurrentFramebuffer(void *renderer);

/* -endRenderPassIfFramebufferChangedForNonDraw:.  Closes a stale render pass
 * when the encoder's FBO no longer matches the context FBO. */
void mglRenderPassEndIfFramebufferChangedForNonDraw(void *renderer,
                                                    uint64_t process_call);

/* -synchronizeRenderPassForTextureReadback:reason:.  Returns 1 when there was
 * nothing to do or the wait succeeded, 0 on failure (the caller skips the
 * readback). */
int mglRenderPassSynchronizeForTextureReadback(void *renderer, void *texture,
                                               const char *reason);

/* -syncRenderPassStateForContext:.  Returns 1 when the encoder matches (or was
 * rotated to match) the context's FBO. */
int mglRenderPassSyncRenderPassStateForContext(void *renderer,
                                               GLMContext glm_ctx);

/* -rotateRenderEncoderForCurrentFramebufferLocked.  Returns 1 when a fresh
 * encoder was created. */
int mglRenderPassRotateRenderEncoderForCurrentFramebufferLocked(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDER_PASS_SYNC_OPS_H */
