/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_render_pass_sync_ops.c - the render-pass sync / close leaves (P0-1,
 * log 191), moved verbatim from MGLRenderer+RenderPass.m:
 *
 *   -currentRenderPassUsesTexture:
 *   -currentRenderPassMatchesCurrentFramebuffer
 *   -endRenderPassIfFramebufferChangedForNonDraw:
 *   -synchronizeRenderPassForTextureReadback:reason:
 *   -syncRenderPassStateForContext:
 *   -rotateRenderEncoderForCurrentFramebufferLocked
 *
 * Three translation notes, all of them the usual ones:
 *   * the attachment-texture helpers of the .m (mglRenderPassAttachmentTextureFor
 *     and its color/depth/stencil wrappers) come back as mglRs* twins;
 *   * `[self endRenderEncoding]` is METAL_LOCK + mglRendererEndRenderEncodingLocked,
 *     so a C caller links straight to the C entry (this is what lets the port go);
 *   * the readback sync's @try/@catch becomes the shell's guarded call with an
 *     exception-reason buffer, so the original log line is replayed verbatim
 *     (rule 58 (b)).
 */

#include "mgl_render_pass_sync_ops.h"

#include "mgl_attachment_binding.h"   /* mglRendererBindFramebufferAttachmentTextures */
#include "mgl_frame_activity.h"       /* MGL_PERF_INC, MGL_ENC_REASON_* */
#include "mgl_gpu_recovery.h"         /* mglPlatformShellGuardedCallCtxReason */
#include "mgl_render_pass_manager.h"  /* mglPassManagerSetFboMatchCacheResult */
#include "mgl_render_encoder_ops.h"  /* mglRenderPassNewRenderEncoderLockedWithReason */
#include "mgl_render_pass_manager_ops.h"
#include "mgl_renderer_ports.h"
#include "mgl_trace_log.h"


#include "mgl_render.h"

#include <stdio.h>

/* Declared next to their definitions in Objective-C headers a .c file cannot
 * include. */
extern Framebuffer *mglRendererGetValidatedFramebuffer(GLMContext ctx,
                                                       const char *where);

/* Declared in the Objective-C MGLRenderer+RenderPass_Private.h (the same
 * restatement mgl_render_pass_manager_ops.c makes). */
extern void mglLogRenderPassLifecycle(const char *tag, uint64_t call,
                                      GLMContext ctx, void *commandBufferOwner,
                                      void *renderEncoderOwner,
                                      void *renderPassStateOwner,
                                      void *drawable,
                                      Framebuffer *renderPassFramebuffer,
                                      uint32_t renderPassFramebufferName,
                                      uint32_t renderPassDrawBuffer,
                                      uint32_t renderPassDrawBufferCount);

/* MGL_STATE() from MGLRenderer_Private.h, in C. */
static GLMState *mglRsState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

/* The .m's mglRenderPassGetPersistentAttachmentState, in C. */
static int mglRsPersistentAttachment(const MGLCommandState *commandState,
                                     uint32_t attachmentKind,
                                     uint64_t colorIndex,
                                     MGLRenderPassAttachmentState *attachmentOut)
{
    if (!commandState || !attachmentOut || !commandState->renderPassStateOwner) {
        return 0;
    }
    MGLRenderPassState state = {0};
    if (mglRenderGetRenderPassStateOwner(commandState->renderPassStateOwner,
                                         &state) != 0) {
        return 0;
    }
    switch (mglRenderPassAttachmentClass(attachmentKind)) {
    case 1:
        if (!mglRenderPassColorAttachmentIndexValid((uint32_t)colorIndex,
                                                    MAX_COLOR_ATTACHMENTS)) {
            return 0;
        }
        *attachmentOut = state.color[colorIndex].attachment;
        return 1;
    case 2:
        *attachmentOut = state.depth.attachment;
        return 1;
    case 3:
        *attachmentOut = state.stencil.attachment;
        return 1;
    default:
        return 0;
    }
}

/* The .m's mglRenderPassColorTextureFor / …DepthTextureFor / …StencilTextureFor. */
static void *mglRsAttachmentTexture(const MGLCommandState *commandState,
                                    uint32_t attachmentKind,
                                    uint64_t colorIndex)
{
    MGLRenderPassAttachmentState attachment = {0};
    if (!mglRsPersistentAttachment(commandState, attachmentKind, colorIndex,
                                   &attachment)) {
        return NULL;
    }
    return attachment.texture;
}

/* -currentRenderPassUsesTexture: */
int mglRenderPassCurrentRenderPassUsesTexture(void *renderer, void *texture)
{
    if (!renderer) {
        return 0;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *commandState = areas.command;
    if (!texture || !commandState ||
        mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) != 1) {
        return 0;
    }
    if (!commandState->renderPassStateOwner) {
        return 0;
    }

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (mglRsAttachmentTexture(commandState,
                                   MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                                   (uint64_t)i) == texture) {
            return 1;
        }
    }
    if (mglRsAttachmentTexture(commandState,
                               MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u) ==
            texture ||
        mglRsAttachmentTexture(commandState,
                               MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u) ==
            texture) {
        return 1;
    }

    return 0;
}

/* -currentRenderPassMatchesCurrentFramebuffer */
int mglRenderPassMatchesCurrentFramebuffer(void *renderer)
{
    if (!renderer) {
        return 1;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *commandState = areas.command;
    if (!ctx || !commandState || !commandState->renderPassStateOwner) {
        return 1;
    }

    Framebuffer *fbo = mglRsState(&areas)->framebuffer;
    GLuint fboName = fbo ? fbo->name : 0u;

    if (fbo != NULL && fboName != 0u) {
        MGLRenderFboMatchCacheState cache = {0};
        if (commandState->renderPassIdentityOwner &&
            mglRenderGetFboMatchCache(commandState->renderPassIdentityOwner,
                                      &cache) == 0 &&
            cache.fbo_name == fboName &&
            cache.generation == fbo->fbo_attachment_generation) {
            return cache.result != 0;
        }
    }

    int result =
        mglRenderPassMatchesFramebufferImpl(renderer, fbo, fboName) != 0;

    /* store cache for non-default FBOs only. */
    if (fbo != NULL && fboName != 0u) {
        mglPassManagerSetFboMatchCacheResult(areas.render_pass_manager, result,
                                             fboName,
                                             fbo->fbo_attachment_generation);
    }

    return result;
}

/* -endRenderPassIfFramebufferChangedForNonDraw: */
void mglRenderPassEndIfFramebufferChangedForNonDraw(void *renderer,
                                                    uint64_t processCall)
{
    if (!renderer) {
        return;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *commandState = areas.command;
    if (!ctx || !commandState ||
        mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) != 1) {
        return;
    }

    if (mglRenderPassMatchesCurrentFramebuffer(renderer)) {
        return;
    }

    static uint64_t s_nonDrawFboMismatchCount = 0;
    uint64_t hit = ++s_nonDrawFboMismatchCount;
    if (mglTraceLogIsEnabled() && (hit <= 32ull || (hit % 256ull) == 0ull)) {
        Framebuffer *fbo = mglRsState(&areas)->framebuffer;
        GLuint fboName = fbo ? fbo->name : 0u;
        mglTraceLog("RENDERPASS_NON_DRAW_MISMATCH processCall=%llu hit=%llu "
                    "ctxFbo=%u(%p) ctxDrawBuf=0x%x rpFbo=%u(%p) rpDrawBuf=0x%x",
                    (unsigned long long)processCall, (unsigned long long)hit,
                    (unsigned)fboName, (void *)fbo,
                    (unsigned)mglRsState(&areas)->draw_buffer,
                    (unsigned)commandState->renderPassFramebufferName,
                    commandState->renderPassFramebuffer,
                    (unsigned)commandState->renderPassDrawBuffer);
        mglLogRenderPassLifecycle(
            "non-draw-mismatch-before-end", hit, ctx,
            commandState->currentCommandBufferOwner,
            commandState->currentRenderEncoderOwner,
            commandState->renderPassStateOwner, areas.drawable,
            commandState->renderPassFramebuffer,
            commandState->renderPassFramebufferName,
            commandState->renderPassDrawBuffer,
            commandState->renderPassDrawBufferCount);
    }

    mglRendererEndRenderEncodingLocked(renderer);
    mglMarkRendererDirtyBits(ctx->active_state,
                             DIRTY_FBO | DIRTY_PROGRAM | DIRTY_RENDER_STATE);
}

/* The @try of the readback sync: commit the detached command buffer and wait
 * for it.  A throwing commit is reported by the shell with its exception
 * reason, and the caller replays the original log line. */
typedef struct MglRsCommitCtx_t {
    void *command_buffer;
} MglRsCommitCtx;

static int mglRsCommitBody(void *renderer, void *rawCtx)
{
    MglRsCommitCtx *ctx = (MglRsCommitCtx *)rawCtx;
    mglRendererCommitCommandBufferWithAGXRecovery(renderer, ctx->command_buffer);
    if (mglRenderWaitCommandBuffer(ctx->command_buffer) != 0) {
        fprintf(stderr, "MGL ERROR: Metal-cpp render-pass wait failed\n");
    }
    return 1;
}

/* -synchronizeRenderPassForTextureReadback:reason: */
int mglRenderPassSynchronizeForTextureReadback(void *renderer, void *texture,
                                               const char *reason)
{
    if (!renderer) {
        return 0;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *commandState = areas.command;

    if (!mglRenderPassCurrentRenderPassUsesTexture(renderer, texture)) {
        return 1;
    }

    mglRendererEndRenderEncodingLocked(renderer);

    MGLRenderCommandBufferState commandStateSnapshot = {0};
    if (!commandState ||
        !mglRenderCommandBufferOwnerHasState(
            commandState->currentCommandBufferOwner, &commandStateSnapshot)) {
        return mglRenderPassNewCommandBufferLocked(renderer) != 0;
    }

    if (commandStateSnapshot.status != MGLCommandBufferStatusNotEnqueued) {
        return mglRenderPassNewCommandBufferLocked(renderer) != 0;
    }

    void *commandBufferToCommit =
        mglPassManagerDetachCurrentCommandBufferForSubmission(
            areas.render_pass_manager);

    char exceptionReason[256] = {0};
    MglRsCommitCtx commitCtx = { commandBufferToCommit };
    if (!mglPlatformShellGuardedCallCtxReason(
            renderer, "render pass texture readback sync", mglRsCommitBody,
            &commitCtx, exceptionReason, sizeof(exceptionReason))) {
        fprintf(stderr,
                "MGL ERROR: failed to synchronize render pass for texture readback (%s): %s\n",
                reason ? reason : "texture_readback",
                exceptionReason[0] ? exceptionReason : "(null)");
        mglRendererRecordGPUError(renderer);
        (void)mglRenderPassNewCommandBufferLocked(renderer);
        return 0;
    }

    MGLRenderCommandBufferState committedState = {0};
    (void)mglRenderGetCommandBufferState(commandBufferToCommit, &committedState);
    if (committedState.has_error) {
        fprintf(stderr,
                "MGL ERROR: render pass texture readback sync failed (%s): %s\n",
                reason ? reason : "texture_readback",
                mglRenderCommandBufferErrorDescription(&committedState));
        mglRendererRecordGPUError(renderer);
        (void)mglRenderPassNewCommandBufferLocked(renderer);
        return 0;
    }

    return mglRenderPassNewCommandBufferLocked(renderer) != 0;
}

/* -syncRenderPassStateForContext: */
int mglRenderPassSyncRenderPassStateForContext(void *renderer,
                                               GLMContext glm_ctx)
{
    if (!renderer) {
        return 0;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMState *state =
        glm_ctx ? glm_ctx->active_state : mglRsState(&areas);
    Framebuffer *framebuffer = mglRendererGetValidatedFramebuffer(
        glm_ctx, "processGLState.dirtyFBO");
    int framebufferBindingDirty =
        framebuffer && (framebuffer->dirty_bits & DIRTY_FBO_BINDING);
    if (areas.command &&
        mglRenderEncoderOwnerHasCurrent(
            areas.command->currentRenderEncoderOwner) == 1 &&
        !framebufferBindingDirty &&
        mglRenderPassMatchesCurrentFramebuffer(renderer)) {
        state->dirty_bits &= ~DIRTY_FBO;
        return 1;
    }

    if (framebuffer && framebufferBindingDirty) {
        RETURN_FALSE_ON_FAILURE(
            mglRendererBindFramebufferAttachmentTextures(renderer));
        framebuffer = mglRendererGetValidatedFramebuffer(
            glm_ctx, "processGLState.dirtyFBO.afterBind");
        if (framebuffer) {
            framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
        }
    }

    /* instrumentation: an FBO change forced a real encoder rotation
     * (the "already matches" fast path above returned early without counting).
     * newRenderEncoderLocked also bumps g_mglEncoderCreationsSinceSwap, so
     * fboRot <= new always holds; new-minus-fboRot is non-FBO creation.
     *
     * RenderPass Manager: encoder open/close is owned by the RenderPass Manager
     * facade (rotateRenderEncoderForCurrentFramebufferLocked), not by this Sync
     * unit directly.  The Sync layer only decides that a rotation is needed and
     * delegates the lifecycle transition. */
    RETURN_FALSE_ON_FAILURE(
        mglRenderPassRotateRenderEncoderForCurrentFramebufferLocked(renderer));
    return 1;
}

/* -rotateRenderEncoderForCurrentFramebufferLocked */
int mglRenderPassRotateRenderEncoderForCurrentFramebufferLocked(void *renderer)
{
    if (!renderer) {
        return 0;
    }
    MGL_PERF_INC(g_mglEncoderFBORotationsSinceSwap);
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext glm_ctx = areas.ctx;
    GLuint fbo_name = 0u;
    if (glm_ctx && glm_ctx->active_state && glm_ctx->active_state->framebuffer) {
        fbo_name = glm_ctx->active_state->framebuffer->name;
    }
    if (fbo_name == 0u) {
        MGL_PERF_INC(g_mglEncoderFboRotDefaultSinceSwap);
    } else {
        MGL_PERF_INC(g_mglEncoderFboRotNamedSinceSwap);
    }
    mglRendererEndRenderEncodingLocked(renderer);
    if (!mglRenderPassNewRenderEncoderLockedWithReason(renderer,
                                                       MGL_ENC_REASON_FBO)) {
        fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
        return 0;
    }
    return 1;
}
