/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_command_state.h — the render pass manager's command state, C-visible.
 *
 * The record used to be defined in the Objective-C MGLRenderPassManager.h, so
 * every C caller that needed one field (the render encoder owner, the render
 * pass state owner, the trace-replay identity, the framebuffer name) went
 * through its own shim port.  It is C state now: the manager keeps it as its
 * ivar, one port hands out the address, and C reads the fields directly.  That
 * retired five wrapper ports.
 */

#ifndef MGL_COMMAND_STATE_H
#define MGL_COMMAND_STATE_H

#include "glm_context.h"
#include "mgl_render.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MGLCommandState_t {
    void *_Nullable renderPassIdentityOwner;
    void *_Nullable renderPassStateOwner;
    Framebuffer *_Nullable renderPassFramebuffer;
    GLuint renderPassFramebufferName;
    GLenum renderPassDrawBuffer;
    GLsizei renderPassDrawBufferCount;
    GLenum renderPassDrawBuffers[MAX_COLOR_ATTACHMENTS];
    uint64_t traceReplayFlushId;
    uint32_t traceReplayBatchIndex;
    GLuint dontCareFrameGeneration;
    void *_Nullable currentCommandBufferOwner;
    void *_Nullable detachedCommandBufferSubmission;
    void *_Nullable mdiArgsScratchOwner;
    void *_Nullable currentRenderEncoderOwner;
    uint8_t currentDrawUsesRTSampledCopy;
    void *_Nullable pendingEventOwner;
    /* Cache for currentRenderPassMatchesCurrentFramebuffer.
     * lastFboMatchFboName == 0 means "invalid cache, recompute".
     * Valid only for non-default FBOs (fbo != NULL && fboName != 0);
     * the default-framebuffer path is never cached because its inputs
     * (drawable, depth/stencil caps, _drawBuffers) change independently
     * of fbo_attachment_generation.
     * Invalidated on encoder install/clear, descriptor install, and
     * render-pass identity update/clear — all signals that the render
     * pass configuration may have changed. */
    GLuint lastFboMatchFboName;
    uint64_t lastFboMatchFboGeneration;
    uint8_t lastFboMatchResult;
    GLMContext _Nullable runtimeContext;
} MGLCommandState;

#ifdef __cplusplus
}
#endif

#endif /* MGL_COMMAND_STATE_H */
