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
#include "mgl_air_loader.h"         /* MGLRenderDepthStencilDescriptorState */
#include "mgl_trace_log.h"           /* kMGLDiagnosticStateLogs */
#include "mgl_vertex_layout.h"       /* mglBindingSet*IfNeeded */
#include "mgl_state_log.h"           /* mglLogRenderStateRepair */
#include "mgl_texture_compat.h"    /* mglTraceTextureName / …Label */
#include "pixel_utils.h"           /* MGLPixelFormatInvalid */
#include "mgl_sync.h"              /* mglLoadActionName / mglStoreActionName */
#include "mgl_readback_policy.h"
#include "mgl_byte_hash.h"
#include "mgl_renderer_backend.h"  /* mglRendererGetProgramBindingCount */
#include "mgl_state_compat.h"      /* mglMTLCompareFunctionForGL, mglLogRenderStateRepair */
#include "mgl_render_pass_manager_ops.h"
#include "mgl_renderer_ports.h"
#include "mgl_trace_log.h"


#include "mgl_render.h"

#include <stdio.h>

/* Declared in Objective-C headers a .c file cannot include. */
extern int mglEnvFlagEnabled(const char *name);
extern GLuint mglRendererSafeFramebufferName(GLMContext ctx);
extern uint32_t mglMaybeInvertMTLWinding(uint32_t winding, bool inverted);
extern int mglFramebufferLooksLikeGLSampledCopyRenderTarget(GLMContext ctx,
                                                           Framebuffer *fb,
                                                           Texture **color_out,
                                                           Texture **depth_out);
extern Program *mglResolveProgramFromState(GLMContext ctx);
extern Texture *findTexture(GLMContext ctx, GLuint texture);
extern int mglRendererObjectPointerLikelyValid(const void *pointer);
extern int mglRendererPointerInHashTable(const void *table, const void *pointer);
extern int mglPointerRangeIsReadable(const void *pointer, size_t length);

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


/* === The two encoder-state blocks (log 192) ==============================
 * -updateCurrentRenderEncoder and -updateViewportAndScissorLocked moved
 * verbatim from MGLRenderer+RenderPass.m (327 + 342 lines).  Everything they
 * touch was already C: the depth-stencil value state goes through a new areas
 * bridge (pipeline_cache_depth_stencil_state_for_value_state), the layer facts
 * through one combined port (mglRendererLayerMetricsPort), and the render-pass
 * attachment/action helpers through the mglRs* twins below.
 */

/* MGLRenderer+RenderPass.m's mglRenderPassTextureInfo(). */
static MGLRenderTextureInfo mglRsTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

/* mglRenderPassColorTextureFor / …DepthTextureFor / …StencilTextureFor. */
static void *mglRsColorTextureFor(const MGLCommandState *commandState,
                                  uint64_t colorIndex)
{
    return mglRsAttachmentTexture(commandState,
                                  MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                                  colorIndex);
}

static void *mglRsDepthTextureFor(const MGLCommandState *commandState)
{
    return mglRsAttachmentTexture(commandState,
                                  MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u);
}

static void *mglRsStencilTextureFor(const MGLCommandState *commandState)
{
    return mglRsAttachmentTexture(commandState,
                                  MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u);
}

/* The .m's mglRenderPassGetPersistentState, in C (mglRsPersistentAttachment
 * above already inlines the same test). */
static int mglRsPersistentState(const MGLCommandState *commandState,
                                MGLRenderPassState *stateOut)
{
    return commandState && stateOut && commandState->renderPassStateOwner &&
           mglRenderGetRenderPassStateOwner(commandState->renderPassStateOwner,
                                            stateOut) == 0;
}

/* mglRenderPassRenderTargetSizeFor / …WidthFor / …HeightFor. */
static int mglRsRenderTargetSizeFor(const MGLCommandState *commandState,
                                    uint64_t *widthOut, uint64_t *heightOut)
{
    MGLRenderPassState state = {0};
    if (!mglRsPersistentState(commandState, &state)) {
        return 0;
    }
    if (widthOut) *widthOut = state.render_target_width;
    if (heightOut) *heightOut = state.render_target_height;
    return 1;
}

static uint64_t mglRsRenderTargetWidthFor(const MGLCommandState *commandState)
{
    uint64_t width = 0;
    if (mglRsRenderTargetSizeFor(commandState, &width, NULL)) {
        return width;
    }
    return 0;
}

static uint64_t mglRsRenderTargetHeightFor(const MGLCommandState *commandState)
{
    uint64_t height = 0;
    if (mglRsRenderTargetSizeFor(commandState, NULL, &height)) {
        return height;
    }
    return 0;
}

/* mglRenderPassActionsFor (load/store action of one attachment). */
static int mglRsActionsFor(const MGLCommandState *commandState,
                           uint32_t attachmentKind, uint64_t colorIndex,
                           uint32_t *loadActionOut, uint32_t *storeActionOut,
                           uint64_t *storeActionOptionsOut)
{
    MGLRenderPassAttachmentState attachment = {0};
    if (mglRsPersistentAttachment(commandState, attachmentKind, colorIndex,
                                  &attachment)) {
        if (loadActionOut) *loadActionOut = (uint32_t)attachment.load_action;
        if (storeActionOut) *storeActionOut = (uint32_t)attachment.store_action;
        if (storeActionOptionsOut) {
            *storeActionOptionsOut = attachment.store_action_options;
        }
        return 1;
    }
    return 0;
}

static uint32_t mglRsLoadActionFor(const MGLCommandState *commandState,
                                   uint32_t attachmentKind, uint64_t colorIndex,
                                   uint32_t fallback)
{
    uint32_t action = 0u;
    if (mglRsActionsFor(commandState, attachmentKind, colorIndex, &action, NULL,
                        NULL)) {
        return action;
    }
    return fallback;
}

static uint32_t mglRsStoreActionFor(const MGLCommandState *commandState,
                                    uint32_t attachmentKind,
                                    uint64_t colorIndex, uint32_t fallback)
{
    uint32_t action = 0u;
    if (mglRsActionsFor(commandState, attachmentKind, colorIndex, NULL, &action,
                        NULL)) {
        return action;
    }
    return fallback;
}

/* mglRenderPassSetPersistentDimensions. */
static void mglRsSetPersistentDimensions(const MGLCommandState *commandState,
                                         uint64_t width, uint64_t height)
{
    if (commandState && commandState->renderPassStateOwner) {
        mglRenderSetRenderPassStateDimensions(commandState->renderPassStateOwner,
                                              width, height);
    }
}

/* kMGLDiagnosticStateLogs is in the C-safe mgl_trace_log.h; this one is the
 * Objective-C header's kMGLVerboseFrameLoopLogs. */
static const int kMglRsVerboseFrameLoopLogs = 0;

/* MGLRenderer+RenderPass.m's mglShouldTraceCall, in C. */
static bool mglRsShouldTraceCall(uint64_t count)
{
    if (!kMGLDiagnosticStateLogs) {
        return false;
    }
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}

/* The scissor rectangle value the .m's MGLScissorRectValue restated for C
 * (same shape as mgl_blit_color_paths.c's twin). */
typedef struct MGLScissorRectValue_t {
    uint64_t x;
    uint64_t y;
    uint64_t width;
    uint64_t height;
} MGLScissorRectValue;

static void mglRsUpdateViewportAndScissor(void *renderer);

/* -updateCurrentRenderEncoder */
void mglRenderPassUpdateCurrentRenderEncoder(void *renderer)
{
    if (!renderer) {
        return;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *state = mglRsState(&areas);
    MGLCommandState *commandState = areas.command;
    const MGLPipelineCacheState *cacheState = areas.pipeline_cache;
    MGLTessellationState *tessellation = areas.tessellation;
    /* areas.binding_state_owner is the ADDRESS of the owner slot (rule 59). */
    void *bindingOwner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;
    int hasConfiguredRenderPass =
        commandState->renderPassStateOwner != NULL;
    int passHasDepthAttachment =
        (hasConfiguredRenderPass &&
         mglRsDepthTextureFor(commandState) != NULL);
    int passHasStencilAttachment =
        (hasConfiguredRenderPass &&
         mglRsStencilTextureFor(commandState) != NULL);
    int useDepthState = mglRenderUseDepthState(
                             state->caps.depth_test ? 1 : 0,
                             passHasDepthAttachment ? 1 : 0) != 0;
    int useStencilState = mglRenderUseStencilState(
                               state->caps.stencil_test ? 1 : 0,
                               passHasStencilAttachment ? 1 : 0) != 0;

    if (state->caps.depth_test && !passHasDepthAttachment) {
        static uint64_t s_missingDepthAttachmentCount = 0;
        uint64_t hit = ++s_missingDepthAttachmentCount;
        if (hit <= 32 || (hit % 256) == 0) {
            fprintf(stderr, "MGL WARNING: depth test/write requested without depth attachment, disabling depth for this pass hit=%llu fbo=%u drawBuf=0x%x\n",
                  (unsigned long long)hit,
                  mglRendererSafeFramebufferName(ctx),
                  state->draw_buffer);
        }
    }

    if (state->caps.stencil_test && !passHasStencilAttachment) {
        static uint64_t s_missingStencilAttachmentCount = 0;
        uint64_t hit = ++s_missingStencilAttachmentCount;
        if (hit <= 32 || (hit % 256) == 0) {
            fprintf(stderr, "MGL WARNING: stencil test requested without stencil attachment, disabling stencil for this pass hit=%llu fbo=%u drawBuf=0x%x\n",
                  (unsigned long long)hit,
                  mglRendererSafeFramebufferName(ctx),
                  state->draw_buffer);
        }
    }

    if (useDepthState || useStencilState)
    {
        MGLRenderDepthStencilDescriptorState dsDesc = {0};
        /* MTLDepthStencilDescriptor initializes depth comparison to Always.
         * Preserve that default for stencil-only passes; leaving the value
         * zero would map to Never and reject every fragment before stencil. */
        dsDesc.depth_compare_function = MGLCompareFunctionAlways;

        if (useDepthState)
        {
            uint32_t depthFunc = mglRenderRepairDepthFunc(
                (uint32_t)state->var.depth_func);
            if (depthFunc != (uint32_t)state->var.depth_func) {
                mglLogRenderStateRepair("depth_func", state->var.depth_func,
                                        (GLenum)depthFunc);
                state->var.depth_func = (GLenum)depthFunc;
                mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
            }

            dsDesc.depth_compare_function = (uint32_t)
                mglMTLCompareFunctionForGL(state->var.depth_func,
                                           MGLCompareFunctionLess,
                                           "depth");
            dsDesc.depth_write_enabled = mglRenderDepthWriteEnabled(
                state->var.depth_writemask ? 1 : 0, 0);
        }

        /* GL_RASTERIZER_DISCARD / VS capture: no fragment is produced, so
         * depth/stencil writes must not mutate attachments (color masks are
         * cleared separately in the pipeline descriptor path). */
        const int suppressDepthStencilWrites =
            mglRenderSuppressDepthStencilWrites(
                state->caps.rasterizer_discard ? 1 : 0,
                tessellation->tessVertexCaptureActive ? 1 : 0,
                tessellation->cullDistanceCaptureActive ? 1 : 0) != 0;
        if (suppressDepthStencilWrites) {
            dsDesc.depth_write_enabled = mglRenderDepthWriteEnabled(
                state->var.depth_writemask ? 1 : 0, 1);
        }

        if (useStencilState)
        {
            if (mglTraceLogIsEnabled()) {
                mglTraceLog("STENCIL_STATE fbo=%u func=0x%x back=0x%x ref=%u backRef=%u readMask=0x%x backReadMask=0x%x writeMask=0x%x attachment=%p layered=%d",
                            (unsigned)mglRendererSafeFramebufferName(ctx),
                            (unsigned)state->var.stencil_func,
                            (unsigned)state->var.stencil_back_func,
                            (unsigned)state->var.stencil_ref,
                            (unsigned)state->var.stencil_back_ref,
                            (unsigned)state->var.stencil_value_mask,
                            (unsigned)state->var.stencil_back_value_mask,
                            (unsigned)state->var.stencil_writemask,
                            mglRsStencilTextureFor(commandState),
                            (int)(state->framebuffer ? state->framebuffer->stencil.layered : 0));
            }
            {
                uint32_t stencilFunc = mglRenderRepairStencilFunc(
                    (uint32_t)state->var.stencil_func);
                if (stencilFunc != (uint32_t)state->var.stencil_func) {
                    mglLogRenderStateRepair("stencil_func", state->var.stencil_func,
                                            (GLenum)stencilFunc);
                    state->var.stencil_func = (GLenum)stencilFunc;
                    mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
                }

                dsDesc.front.present = 1u;
                dsDesc.front.compare_function = (uint32_t)
                    mglMTLCompareFunctionForGL(state->var.stencil_func,
                                               MGLCompareFunctionAlways,
                                               "front-stencil");
                if (mglEnvFlagEnabled("MGL_FORCE_STENCIL_ALWAYS")) {
                    dsDesc.front.compare_function = MGLCompareFunctionAlways;
                }
                uint32_t failOp = 0u, depthFailOp = 0u, passOp = 0u;
                (void)mglRenderStencilOpFromGL((uint32_t)state->var.stencil_fail,
                                               &failOp);
                (void)mglRenderStencilOpFromGL(
                    (uint32_t)state->var.stencil_pass_depth_fail, &depthFailOp);
                (void)mglRenderStencilOpFromGL(
                    (uint32_t)state->var.stencil_pass_depth_pass, &passOp);
                dsDesc.front.stencil_failure_operation = failOp;
                dsDesc.front.depth_failure_operation = depthFailOp;
                dsDesc.front.depth_stencil_pass_operation = passOp;
                dsDesc.front.write_mask = mglRenderStencilWriteMask(
                    suppressDepthStencilWrites ? 1 : 0,
                    (uint32_t)state->var.stencil_writemask);
                dsDesc.front.read_mask = state->var.stencil_value_mask;
            }

            {
                uint32_t stencilBack = mglRenderRepairStencilFunc(
                    (uint32_t)state->var.stencil_back_func);
                if (stencilBack != (uint32_t)state->var.stencil_back_func) {
                    mglLogRenderStateRepair("stencil_back_func",
                                            state->var.stencil_back_func,
                                            (GLenum)stencilBack);
                    state->var.stencil_back_func = (GLenum)stencilBack;
                    mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
                }

                dsDesc.back.present = 1u;
                dsDesc.back.compare_function = (uint32_t)
                    mglMTLCompareFunctionForGL(state->var.stencil_back_func,
                                               MGLCompareFunctionAlways,
                                               "back-stencil");
                if (mglEnvFlagEnabled("MGL_FORCE_STENCIL_ALWAYS")) {
                    dsDesc.back.compare_function = MGLCompareFunctionAlways;
                }
                uint32_t backFail = 0u, backDepthFail = 0u, backPass = 0u;
                (void)mglRenderStencilOpFromGL(
                    (uint32_t)state->var.stencil_back_fail, &backFail);
                (void)mglRenderStencilOpFromGL(
                    (uint32_t)state->var.stencil_back_pass_depth_fail,
                    &backDepthFail);
                (void)mglRenderStencilOpFromGL(
                    (uint32_t)state->var.stencil_back_pass_depth_pass,
                    &backPass);
                dsDesc.back.stencil_failure_operation = backFail;
                dsDesc.back.depth_failure_operation = backDepthFail;
                dsDesc.back.depth_stencil_pass_operation = backPass;
                dsDesc.back.write_mask = mglRenderStencilWriteMask(
                    suppressDepthStencilWrites ? 1 : 0,
                    (uint32_t)state->var.stencil_back_writemask);
                dsDesc.back.read_mask = state->var.stencil_back_value_mask;
            }
        }

        void *dsState = NULL;
        if (areas.pipeline_cache_depth_stencil_state_for_value_state) {
            areas.pipeline_cache_depth_stencil_state_for_value_state(
                areas.pipeline_cache_object, &dsDesc, &dsState);
        }

        if (mglRenderBindingSetDepthStencilIfNeededForOwner(
                bindingOwner,
                commandState->currentRenderEncoderOwner,
                dsState) > 0) {
        } else {
            MGL_PERF_INC(g_mglDepthStencilStateSkipsSinceSwap);
        }
        if (useStencilState) {
            mglRenderSetStencilReferenceValuesForOwner(
                commandState->currentRenderEncoderOwner,
                (uint32_t)state->var.stencil_ref,
                (uint32_t)state->var.stencil_back_ref);
        }
    }
    else
    {
        MGLRenderDepthStencilDescriptorState disabledDSDesc = {0};
        disabledDSDesc.depth_compare_function = MGLCompareFunctionAlways;
        disabledDSDesc.depth_write_enabled = 0u;

        void *disabledDSState = NULL;
        if (areas.pipeline_cache_depth_stencil_state_for_value_state) {
            areas.pipeline_cache_depth_stencil_state_for_value_state(
                areas.pipeline_cache_object, &disabledDSDesc,
                &disabledDSState);
        }
        if (disabledDSState) {
            if (mglRenderBindingSetDepthStencilIfNeededForOwner(
                    bindingOwner,
                    commandState->currentRenderEncoderOwner,
                    disabledDSState) > 0) {
            } else {
                MGL_PERF_INC(g_mglDepthStencilStateSkipsSinceSwap);
            }
        }
    }

    {
        float bcRed   = state->var.blend_color[0];
        float bcGreen = state->var.blend_color[1];
        float bcBlue  = state->var.blend_color[2];
        float bcAlpha = state->var.blend_color[3];
        mglRenderBindingSetBlendColorIfNeededForOwner(
            bindingOwner,
            commandState->currentRenderEncoderOwner,
            bcRed, bcGreen, bcBlue, bcAlpha);
    }

    /* GL_SAMPLE_MASK: Metal does not expose a per-draw sample mask setter on
     * MTLRenderCommandEncoder.  Sample coverage in Metal is controlled via
     * alpha-to-coverage and shader-side [[sample_mask]], neither of which
     * maps cleanly to GL_SAMPLE_MASK.  This remains a known limitation. */

    mglRsUpdateViewportAndScissor(renderer);

    if (!mglRenderFrontFaceValid((uint32_t)state->var.front_face)) {
        uint32_t repaired = mglRenderFrontFaceOrCCW(
            (uint32_t)state->var.front_face);
        mglLogRenderStateRepair("front_face", state->var.front_face,
                                (GLenum)repaired);
        state->var.front_face = (GLenum)repaired;
        mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
    }

    int rtSampledCopyDraw = commandState->currentDrawUsesRTSampledCopy;
    int defaultFramebufferSampledPass =
        mglRenderSkipCullForSampledPass(
            state->framebuffer ? 1 : 0, state->caps.depth_test ? 1 : 0,
            mglRendererGetProgramBindingCount(ctx, _FRAGMENT_SHADER,
                                              _SAMPLED_IMAGE_RES) > 0
                ? 1
                : 0,
            rtSampledCopyDraw ? 1 : 0) != 0 &&
        !rtSampledCopyDraw;

    uint32_t cull_mode = mglRenderCullModeFromGL(
        (state->caps.cull_face && !defaultFramebufferSampledPass &&
         !rtSampledCopyDraw)
            ? 1
            : 0,
        (uint32_t)state->var.cull_face_mode);
    mglRenderBindingSetCullIfNeededForOwner(
        bindingOwner,
        commandState->currentRenderEncoderOwner, cull_mode);
    uint32_t _winding =
        mglMaybeInvertMTLWinding(mglMTLWindingForGL(state->var.front_face),
                                 !mglRenderClipOriginIsLowerLeft(
                                     (uint32_t)state->var.clip_origin));
    mglRenderBindingSetWindingIfNeededForOwner(
        bindingOwner,
        commandState->currentRenderEncoderOwner,
        (uint32_t)_winding);

    if (state->caps.cull_face && defaultFramebufferSampledPass) {
        static uint64_t s_defaultSampledCullBypassCount = 0;
        uint64_t hit = ++s_defaultSampledCullBypassCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            mglTraceLog("MGL TRACE default sampled pass cull bypass hit=%llu program=%u drawBuf=0x%x",
                  (unsigned long long)hit,
                  (unsigned)(ctx ? state->program_name : 0u),
                  (unsigned)(ctx ? state->draw_buffer : 0u));
        }
    }
    if (state->caps.cull_face && rtSampledCopyDraw) {
        static uint64_t s_rtSampledCopyCullBypassCount = 0;
        uint64_t hit = ++s_rtSampledCopyCullBypassCount;
        if (hit <= 64ull || (hit % 256ull) == 0ull) {
            mglTraceLog("RT_SAMPLE_COPY_CULL_BYPASS hit=%llu program=%u pipelineProgram=%u fbo=%u rpFbo=%u depth(test=%d write=%d func=0x%x) blend=%d cullFace=0x%x frontFace=0x%x",
                        (unsigned long long)hit,
                        (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
                        (unsigned)cacheState->pipelineProgramName,
                        (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
                        (unsigned)commandState->renderPassFramebufferName,
                        (ctx && state->caps.depth_test) ? 1 : 0,
                        (ctx && state->var.depth_writemask) ? 1 : 0,
                        (unsigned)(ctx ? state->var.depth_func : 0u),
                        (ctx && state->caps.blend) ? 1 : 0,
                        (unsigned)(ctx ? state->var.cull_face_mode : 0u),
                        (unsigned)(ctx ? state->var.front_face : 0u));
        }
    }

    if (state->caps.depth_clamp)
    {
        mglRenderSetDepthClipModeForOwner(
            commandState->currentRenderEncoderOwner,
            mglRenderDepthClipMode(state->caps.depth_clamp ? 1 : 0));
    }

    if (mglRenderPolygonOffsetEnabled(
            state->caps.polygon_offset_fill ? 1 : 0,
            state->caps.polygon_offset_line ? 1 : 0,
            state->caps.polygon_offset_point ? 1 : 0))
    {
        float _bias = state->var.polygon_offset_units;
        float _slope = state->var.polygon_offset_factor;
        float _clamp = 0.0f;
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            bindingOwner,
            commandState->currentRenderEncoderOwner,
            _bias, _clamp, _slope);
    }
    else
    {
        mglRenderBindingSetDepthBiasIfNeededForOwner(
            bindingOwner,
            commandState->currentRenderEncoderOwner,
            0.0f, 0.0f, 0.0f);
    }

    uint32_t triangleFillMode = mglRenderTriangleFillMode(
        (uint32_t)state->var.polygon_mode);
    if (!mglRenderPolygonModeValid((uint32_t)state->var.polygon_mode)) {
        uint32_t repaired = mglRenderPolygonModeOrFill(
            (uint32_t)state->var.polygon_mode);
        mglLogRenderStateRepair("polygon_mode", state->var.polygon_mode,
                                (GLenum)repaired);
        state->var.polygon_mode = (GLenum)repaired;
        mglMarkStateDirtyBits(state, DIRTY_RENDER_STATE);
    }
    mglBindingSetTriangleFillModeIfNeeded(renderer, triangleFillMode);
}

/* -updateViewportAndScissorLocked */
void mglRenderPassUpdateViewportAndScissorLocked(void *renderer)
{
    if (!renderer) {
        return;
    }
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *state = mglRsState(&areas);
    MGLCommandState *commandState = areas.command;
    void *bindingOwner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;
    mglRsUpdateViewportAndScissor(renderer);
}

static void mglRsUpdateViewportAndScissor(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    GLMState *state = mglRsState(&areas);
    MGLCommandState *commandState = areas.command;
    void *bindingOwner =
        areas.binding_state_owner ? *areas.binding_state_owner : NULL;
    // Metal validates viewport/scissor strictly against the active render pass dimensions.
    // Always derive pass size from the current attachments first (not from window drawable fallback).
    {
        static uint64_t s_encoderStateUpdateCount = 0;
        bool traceEncoderState = kMGLDiagnosticStateLogs || mglRsShouldTraceCall(++s_encoderStateUpdateCount);

        uint64_t passWidth = 0;
        uint64_t passHeight = 0;
        void *passTexture = NULL;

        /* The C++ owner is the authoritative configured-pass signal. */
        int hasConfiguredRenderPass =
            commandState->renderPassStateOwner != NULL;
        if (hasConfiguredRenderPass) {
            passWidth = mglRsRenderTargetWidthFor(commandState);
            passHeight = mglRsRenderTargetHeightFor(commandState);

            if (passWidth == 0 || passHeight == 0) {
                for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
                    void *candidate = mglRsColorTextureFor(commandState, i);
                    if (candidate) {
                        passTexture = candidate;
                        break;
                    }
                }

                if (!passTexture) {
                    passTexture = mglRsDepthTextureFor(commandState);
                }
                if (!passTexture) {
                    passTexture = mglRsStencilTextureFor(commandState);
                }

                if (passTexture) {
                    passWidth = mglRsTextureInfo(passTexture).width;
                    passHeight = mglRsTextureInfo(passTexture).height;
                    mglRsSetPersistentDimensions(
                        commandState, passWidth, passHeight);
                    if (kMglRsVerboseFrameLoopLogs) {
                        fprintf(stderr, "MGL INFO: Resolved render pass size from attachment %lux%lu (rtw/rth were unset)\n",
                              (unsigned long)passWidth, (unsigned long)passHeight);
                    }
                }
            }
        }

        void *drawableTexture = areas.drawable
            ? mglRendererDrawableTexturePort(renderer)
            : NULL;
        if ((passWidth == 0 || passHeight == 0) && drawableTexture) {
            passWidth = mglRsTextureInfo(drawableTexture).width;
            passHeight = mglRsTextureInfo(drawableTexture).height;
            if (traceEncoderState) {
                fprintf(stderr, "MGL WARNING: Falling back to drawable size for encoder state: %lux%lu\n",
                      (unsigned long)passWidth, (unsigned long)passHeight);
            }
        }

        MGLRendererLayerMetricsValue layerMetrics = {0};
        int hasLayer = mglRendererLayerMetricsPort(renderer, &layerMetrics) != 0;
        if ((passWidth == 0 || passHeight == 0) && hasLayer) {
            if (layerMetrics.drawable_width > 0 &&
                layerMetrics.drawable_height > 0) {
                passWidth = (uint64_t)layerMetrics.drawable_width;
                passHeight = (uint64_t)layerMetrics.drawable_height;
            } else if (layerMetrics.frame_width > 0 &&
                       layerMetrics.frame_height > 0) {
                passWidth = (uint64_t)layerMetrics.frame_width;
                passHeight = (uint64_t)layerMetrics.frame_height;
            }
            if (traceEncoderState) {
                fprintf(stderr, "MGL WARNING: Falling back to layer size for encoder state: %lux%lu\n",
                      (unsigned long)passWidth, (unsigned long)passHeight);
            }
        }

        if (passWidth > 0 && passHeight > 0) {
            GLint rawSx = 0;
            GLint rawSy = 0;
            GLint rawSw = (GLint)passWidth;
            GLint rawSh = (GLint)passHeight;

            GLint sx = 0;
            GLint sy = 0;
            GLint sw = (GLint)passWidth;
            GLint sh = (GLint)passHeight;

            if (state->caps.scissor_test) {
                rawSx = (GLint)state->var.scissor_box[0];
                rawSy = (GLint)state->var.scissor_box[1];
                rawSw = (GLint)state->var.scissor_box[2];
                rawSh = (GLint)state->var.scissor_box[3];

                sx = rawSx;
                sy = rawSy;
                sw = rawSw;
                sh = rawSh;
                mglRenderClampScissorRect(&sx, &sy, &sw, &sh,
                                          (uint32_t)passWidth,
                                          (uint32_t)passHeight);
            }

            GLint metalSy = mglRenderMetalScissorY(
                sx, sh, (uint32_t)passHeight,
                (uint32_t)state->var.clip_origin);

	            if (traceEncoderState) {
                fprintf(stderr, "MGL SCISSOR apply pass=%lux%lu scissorEnabled=%d origin=0x%x raw=(%d,%d,%d,%d) glResolved=(%d,%d,%d,%d) metal=(%d,%d,%d,%d)\n",
                      (unsigned long)passWidth, (unsigned long)passHeight,
                      state->caps.scissor_test ? 1 : 0,
                      state->var.clip_origin,
                      rawSx, rawSy, rawSw, rawSh,
                      sx, sy, sw, sh,
                      sx, metalSy, sw, sh);
            }

            MGLScissorRectValue rect;
            rect.x = (uint64_t)sx;
            rect.y = (uint64_t)metalSy;
            rect.width = (uint64_t)sw;
            rect.height = (uint64_t)sh;
            mglBindingSetScissorRectIfNeeded(renderer, rect.x, rect.y, rect.width, rect.height);

            GLdouble rawVx = (GLdouble)state->viewport[0];
            GLdouble rawVy = (GLdouble)state->viewport[1];
            GLdouble rawVw = (GLdouble)state->viewport[2];
            GLdouble rawVh = (GLdouble)state->viewport[3];

            GLdouble vx = rawVx;
            GLdouble vy = rawVy;
            GLdouble vw = rawVw;
            GLdouble vh = rawVh;
            mglRenderClampViewport(&vx, &vy, &vw, &vh, (uint32_t)passWidth,
                                   (uint32_t)passHeight);
            GLdouble metalVy = mglRenderMetalViewportY(vy, vh,
                                                       (uint32_t)passHeight);

            Texture *guiRTColor = NULL;
            Texture *guiRTDepth = NULL;
            int guiRTPass =
                mglTraceLogIsEnabled() &&
                mglFramebufferLooksLikeGLSampledCopyRenderTarget(ctx,
                                                                 state->framebuffer,
                                                                 &guiRTColor,
                                                                 &guiRTDepth);
            if (guiRTPass) {
                static uint64_t s_guiRTEncoderStateLogCount = 0;
                uint64_t hit = ++s_guiRTEncoderStateLogCount;
                if (hit <= 128ull || (hit % 256ull) == 0ull) {
                    Program *program = mglResolveProgramFromState(ctx);
                    void *c0 = mglRsColorTextureFor(commandState, 0);
                    void *d0 = mglRsDepthTextureFor(commandState);
                    mglTraceLog("RT_SAMPLE_COPY_ENCODER hit=%llu fbo=%u rpFbo=%u program=%u rtTex=%u label=\"%s\" depthTex=%u depthLabel=\"%s\" "
                          "pass=%lux%lu c0=%p fmt=%lu depth=%p fmt=%lu "
                          "loadStore(c=%s/%s d=%s/%s) clipOrigin=0x%x "
                          "scissor(en=%d raw=%d,%d,%d,%d metal=%d,%d,%d,%d) "
                          "viewport(raw=%.1f,%.1f,%.1f,%.1f metal=%.1f,%.1f,%.1f,%.1f) "
                          "depth(test=%d write=%d func=0x%x) blend=%d cull=%d levels=%u mips=%u mipmapped=%u",
                          (unsigned long long)hit,
                          state->framebuffer ? (unsigned)state->framebuffer->name : 0u,
                          (unsigned)commandState->renderPassFramebufferName,
                          program ? (unsigned)program->name : (unsigned)state->program_name,
                          (unsigned)mglTraceTextureName((Texture *)guiRTColor),
                          mglTraceTextureLabel((Texture *)guiRTColor),
                          (unsigned)mglTraceTextureName((Texture *)guiRTDepth),
                          mglTraceTextureLabel((Texture *)guiRTDepth),
                          (unsigned long)passWidth,
                          (unsigned long)passHeight,
                          c0,
                          (unsigned long)(c0 ? mglRsTextureInfo(c0).pixel_format : MGLPixelFormatInvalid),
                          d0,
                          (unsigned long)(d0 ? mglRsTextureInfo(d0).pixel_format : MGLPixelFormatInvalid),
                          mglLoadActionName(mglRsLoadActionFor(commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLLoadActionDontCare)),
                          mglStoreActionName(mglRsStoreActionFor(commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, MGLStoreActionDontCare)),
                          mglLoadActionName(mglRsLoadActionFor(commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLLoadActionDontCare)),
                          mglStoreActionName(mglRsStoreActionFor(commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, MGLStoreActionDontCare)),
                          state->var.clip_origin,
                          state->caps.scissor_test ? 1 : 0,
                          rawSx, rawSy, rawSw, rawSh,
                          sx, metalSy, sw, sh,
                          rawVx, rawVy, rawVw, rawVh,
                          vx, metalVy, vw, vh,
                          state->caps.depth_test ? 1 : 0,
                          state->var.depth_writemask ? 1 : 0,
                          (unsigned)state->var.depth_func,
                          state->caps.blend ? 1 : 0,
                          state->caps.cull_face ? 1 : 0,
                          guiRTColor ? (unsigned)guiRTColor->num_levels : 0u,
                          guiRTColor ? (unsigned)guiRTColor->mipmap_levels : 0u,
                          guiRTColor ? (unsigned)guiRTColor->mipmapped : 0u);
                }
            }

            int viewportWasClamped = (vx != rawVx || vy != rawVy || vw != rawVw || vh != rawVh);
            int viewportOriginConverted = (metalVy != vy);
            if (traceEncoderState) {
                mglTraceLog("MGL VIEWPORT apply pass=%lux%lu origin=0x%x raw=(%.3f,%.3f,%.3f,%.3f) resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
                              (unsigned long)passWidth, (unsigned long)passHeight,
                              state->var.clip_origin,
                              rawVx, rawVy, rawVw, rawVh,
                              vx, vy, vw, vh,
                              vx, metalVy, vw, vh);
            }

            if (kMGLDiagnosticStateLogs && (viewportWasClamped || viewportOriginConverted)) {
                static uint64_t s_viewportClampDetailCount = 0;
                uint64_t clampHit = ++s_viewportClampDetailCount;
                int logClampDetail = (clampHit <= 80ull || (clampHit % 120ull) == 0ull);

                if (logClampDetail) {
                    Framebuffer *debugFbo = state->framebuffer;
                    int debugFboValid = (debugFbo != NULL &&
                                          mglRendererObjectPointerLikelyValid(debugFbo) &&
                                          mglRendererPointerInHashTable(&state->framebuffer_table, debugFbo) &&
                                          mglPointerRangeIsReadable(debugFbo, sizeof(*debugFbo)));
                    void *rpColor0 = mglRsColorTextureFor(commandState, 0);
                    void *rpDepth = mglRsDepthTextureFor(commandState);
                    void *drawableTexture =
                        areas.drawable ? mglRendererDrawableTexturePort(renderer) : NULL;

                    mglTraceLog("MGL VIEWPORT CLAMP DETAIL hit=%llu fbo=%p valid=%d fboName=%u drawBuffer=0x%x pass=%lux%lu "
                                  "rpColor0=%p(%lux%lu) rpDepth=%p(%lux%lu) drawable=%p(%lux%lu) raw=(%.3f,%.3f,%.3f,%.3f) "
                                  "resolved=(%.3f,%.3f,%.3f,%.3f) metal=(%.3f,%.3f,%.3f,%.3f)",
                                  (unsigned long long)clampHit,
                                  debugFbo,
                                  debugFboValid ? 1 : 0,
                                  (debugFboValid ? debugFbo->name : 0),
                                  state->draw_buffer,
                                  (unsigned long)passWidth,
                                  (unsigned long)passHeight,
                                  rpColor0,
                                  (unsigned long)(rpColor0 ? mglRsTextureInfo(rpColor0).width : 0),
                                  (unsigned long)(rpColor0 ? mglRsTextureInfo(rpColor0).height : 0),
                                  rpDepth,
                                  (unsigned long)(rpDepth ? mglRsTextureInfo(rpDepth).width : 0),
                                  (unsigned long)(rpDepth ? mglRsTextureInfo(rpDepth).height : 0),
                                  drawableTexture,
                                  (unsigned long)(drawableTexture ? mglRsTextureInfo(drawableTexture).width : 0),
                                  (unsigned long)(drawableTexture ? mglRsTextureInfo(drawableTexture).height : 0),
                                  rawVx, rawVy, rawVw, rawVh,
                                  vx, vy, vw, vh,
                                  vx, metalVy, vw, vh);

                    if (debugFboValid) {
                        for (int attIndex = 0; attIndex < MAX_COLOR_ATTACHMENTS; attIndex++) {
                            FBOAttachment *attachment = &debugFbo->color_attachments[attIndex];
                            if (attachment->texture == 0 && attachment->buf.tex == NULL && attachment->buf.rbo == NULL) {
                                continue;
                            }

                            Texture *attachmentTexture = NULL;
                            if (mglRenderTargetIsRenderbuffer((uint32_t)attachment->textarget)) {
                                attachmentTexture = attachment->buf.rbo ? attachment->buf.rbo->tex : NULL;
                            } else {
                                attachmentTexture = attachment->buf.tex;
                                if (!attachmentTexture && attachment->texture != 0) {
                                    attachmentTexture = findTexture(ctx, attachment->texture);
                                }
                            }

                            void *attachmentMtl = (attachmentTexture && attachmentTexture->mtl_data)
                                ? (void *)(attachmentTexture->mtl_data)
                                : NULL;
                            void *rpAttachment = mglRsColorTextureFor(commandState, attIndex);

                            mglTraceLog("MGL VIEWPORT CLAMP FBO att=%d name=%u textarget=0x%x level=%d layer=%d tex=%p "
                                          "texName=%u texTarget=0x%x texSize=%ux%ux%u mtl=%p(%lux%lu) rpTex=%p(%lux%lu)",
                                          attIndex,
                                          attachment->texture,
                                          attachment->textarget,
                                          attachment->level,
                                          attachment->layer,
                                          attachmentTexture,
                                          attachmentTexture ? attachmentTexture->name : 0,
                                          attachmentTexture ? attachmentTexture->target : 0,
                                          attachmentTexture ? attachmentTexture->width : 0,
                                          attachmentTexture ? attachmentTexture->height : 0,
                                          attachmentTexture ? attachmentTexture->depth : 0,
                                          attachmentMtl,
                                          (unsigned long)(attachmentMtl ? mglRsTextureInfo(attachmentMtl).width : 0),
                                          (unsigned long)(attachmentMtl ? mglRsTextureInfo(attachmentMtl).height : 0),
                                          rpAttachment,
                                          (unsigned long)(rpAttachment ? mglRsTextureInfo(rpAttachment).width : 0),
                                          (unsigned long)(rpAttachment ? mglRsTextureInfo(rpAttachment).height : 0));
                        }
                    }
                }
            }

            /* gl_ViewportIndex: when glViewportIndexedf* set any slot
             * beyond 0, bind the whole 16-entry viewport array (Metal
             * selects per vertex via viewport_array_index).  Slot 0 uses
             * the resolved/clamped rectangle computed above. */
            if (state->viewport_array_set) {
                double viewports[MGL_MAX_VIEWPORTS * 6];
                viewports[0] = vx;
                viewports[1] = metalVy;
                viewports[2] = vw;
                viewports[3] = vh;
                viewports[4] = state->var.depth_range[0];
                viewports[5] = state->var.depth_range[1];
                for (int vi = 1; vi < MGL_MAX_VIEWPORTS; vi++) {
                    GLdouble avx = state->viewport_array[vi][0];
                    GLdouble avy = state->viewport_array[vi][1];
                    GLdouble avw = state->viewport_array[vi][2];
                    GLdouble avh = state->viewport_array[vi][3];
                    GLdouble metalAvy = (GLdouble)passHeight - (avy + avh);
                    if (metalAvy < 0.0) metalAvy = 0.0;
                    viewports[vi * 6 + 0] = avx;
                    viewports[vi * 6 + 1] = metalAvy;
                    viewports[vi * 6 + 2] = avw;
                    viewports[vi * 6 + 3] = avh;
                    viewports[vi * 6 + 4] = state->var.depth_range[0];
                    viewports[vi * 6 + 5] = state->var.depth_range[1];
                }
                mglRenderBindingSetViewportsForOwner(
                    bindingOwner,
                    commandState->currentRenderEncoderOwner,
                    viewports, (uint64_t)MGL_MAX_VIEWPORTS);
            } else {

                double viewports[MGL_MAX_VIEWPORTS * 6];
                for (int vi = 0; vi < MGL_MAX_VIEWPORTS; vi++) {
                    viewports[vi * 6 + 0] = vx;
                    viewports[vi * 6 + 1] = metalVy;
                    viewports[vi * 6 + 2] = vw;
                    viewports[vi * 6 + 3] = vh;
                    viewports[vi * 6 + 4] = state->var.depth_range[0];
                    viewports[vi * 6 + 5] = state->var.depth_range[1];
                }
                mglRenderBindingSetViewportsForOwner(
                    bindingOwner,
                    commandState->currentRenderEncoderOwner,
                    viewports, (uint64_t)MGL_MAX_VIEWPORTS);
            }
        } else {
            if (traceEncoderState) {
                fprintf(stderr, "MGL WARNING: updateCurrentRenderEncoder could not resolve pass size; using raw GL viewport\n");
            }
            mglBindingSetViewportIfNeeded(renderer, state->viewport[0], state->viewport[1], state->viewport[2], state->viewport[3], state->var.depth_range[0], state->var.depth_range[1]);
        }
    }
}
