/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_render_pass_manager_ops.c — C twins of the MGLRenderPassManager methods
 * that only touch the command state.  The manager keeps calling its own
 * methods; C callers use these.
 */

#include "mgl_render_pass_manager_ops.h"
#include "mgl_attachment_binding.h"   /* mglRendererBindFramebufferAttachmentTextures */
#include "mgl_buffer_map.h"           /* map / dirty base buffers */
#include "mgl_texture_bind.h"          /* mglRendererBindMTLTexture */
#include "mgl_draw_buffer.h"          /* Metal draw-buffer mapping */
#include "mgl_sync.h"                 /* MGLMetalAttachmentSubresource */
#include "mgl_renderer_backend.h"     /* default draw buffer attachments */
#include "mgl_batch_issue.h"          /* mglBatchBindActiveTexturesToMTL */
#include "mgl_stage_encode_drivers.h" /* stage encode bind drivers */
#include "mgl_frame_activity.h"     /* MGL_ENC_REASON_* */

#include "mgl_renderer_ports.h"
#include "mgl_binding_state_ops.h"      /* mglBindingInvalidateLastBoundState */
#include "mgl_trace_strategy.h"         /* mglClearFragmentTraceBindingsForRenderer */
#include "mgl_blit_sampled_copy.h"      /* GLSampled copies refresh */

#include <stdio.h>

/* Defined in MGLRenderer.m; declared in the Objective-C
 * MGLRenderer+RenderPass_Private.h. */
extern void mglLogRenderPassLifecycle(const char *tag, uint64_t call,
                                      GLMContext ctx, void *commandBufferOwner,
                                      void *renderEncoderOwner,
                                      void *renderPassStateOwner, void *drawable,
                                      Framebuffer *renderPassFramebuffer,
                                      GLuint renderPassFramebufferName,
                                      GLenum renderPassDrawBuffer,
                                      GLsizei renderPassDrawBufferCount);

/* Shell-provided drawable accessor (private ivar) and guarded call. */
extern void *mglPlatformShellDrawable(void *renderer);
extern int mglPlatformShellGuardedCall(void *renderer, const char *what,
                                       int (*body)(void *));

/* Local twin of the manager's file-static helper of the same name. */
static void mglRenderPassManagerSyncRuntimeOwners(MGLCommandState *state)
{
    GLMContext context = state ? state->runtimeContext : NULL;
    if (!context) {
        return;
    }
    mglRenderAttachRuntimeOwners(context, state->currentCommandBufferOwner,
                                 state->currentRenderEncoderOwner,
                                 state->renderPassStateOwner);
}

void mglRenderPassManagerEndCurrentRenderEncoder(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs || !cs->currentRenderEncoderOwner ||
        mglRenderEncoderOwnerHasCurrent(cs->currentRenderEncoderOwner) != 1) {
        return;
    }
    (void)mglRenderEndRenderEncoderOwner(cs->currentRenderEncoderOwner);
}

void mglRenderPassManagerClearCurrentRenderEncoder(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return;
    }
    /* encoder ended - invalidate FBO match cache. */
    mglRenderClearFboMatchCache(cs->renderPassIdentityOwner);
    mglRenderDestroyRenderEncoderOwner(&cs->currentRenderEncoderOwner);
    mglRenderPassManagerSyncRuntimeOwners(cs);
}

void mglRenderPassManagerDiscardCurrentCommandBuffer(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return;
    }
    mglRenderDiscardCommandBufferOwnerCurrent(cs->currentCommandBufferOwner);
    mglRenderDestroyMDIScratchOwner(&cs->mdiArgsScratchOwner);
    mglRenderPassManagerSyncRuntimeOwners(cs);
}

int mglRenderPassManagerCommitCommandBufferTransaction(
    void *renderer, void *commandBuffer, void *recoveryOwner,
    int waitForCompletion, MGLRenderCommandBufferTransaction *result)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return -1;
    }
    int transactionResult = mglRenderCommitCommandBufferTransaction(
        cs->currentCommandBufferOwner, &cs->detachedCommandBufferSubmission,
        commandBuffer, recoveryOwner, waitForCompletion ? 1u : 0u, result);
    mglRenderPassManagerSyncRuntimeOwners(cs);
    return transactionResult;
}

void mglRenderPassManagerReleaseDetachedCommandBufferIfOwned(
    void *renderer, void *commandBuffer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs || !cs->detachedCommandBufferSubmission) {
        return;
    }
    if (commandBuffer &&
        mglRenderCommandBufferSubmissionMatchesBuffer(
            cs->detachedCommandBufferSubmission, commandBuffer) != 1) {
        return;
    }
    mglRenderDestroyCommandBufferSubmission(&cs->detachedCommandBufferSubmission);
}

/* Local twins of the manager's file-static helpers (same bodies). */
static void mglRenderPassManagerSyncIdentityView(
    MGLCommandState *commandState, const MGLRenderPassIdentityState *identity)
{
    commandState->renderPassFramebuffer = (Framebuffer *)identity->framebuffer;
    commandState->renderPassFramebufferName = identity->framebuffer_name;
    commandState->renderPassDrawBuffer = identity->draw_buffer;
    commandState->renderPassDrawBufferCount = (GLsizei)identity->draw_buffer_count;
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; ++index) {
        commandState->renderPassDrawBuffers[index] = identity->draw_buffers[index];
    }
}

static void mglRenderPassManagerStoreIdentity(
    MGLCommandState *commandState, const MGLRenderPassIdentityState *identity)
{
    if (!commandState->renderPassIdentityOwner &&
        mglRenderCreateRenderPassIdentityOwner(
            &commandState->renderPassIdentityOwner) != 0) {
        commandState->renderPassIdentityOwner = NULL;
    }
    if (commandState->renderPassIdentityOwner &&
        mglRenderUpdateRenderPassIdentity(commandState->renderPassIdentityOwner,
                                          identity) != 0) {
        mglRenderDestroyRenderPassIdentityOwner(
            &commandState->renderPassIdentityOwner);
    }
    mglRenderPassManagerSyncIdentityView(commandState, identity);
}

void mglRenderPassManagerClearRenderPassIdentity(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs) {
        return;
    }
    /* render pass ended - invalidate FBO match cache. */
    mglRenderClearFboMatchCache(cs->renderPassIdentityOwner);
    MGLRenderPassIdentityState identity = {0};
    for (uint32_t index = 0; index < MAX_COLOR_ATTACHMENTS; index++) {
        identity.draw_buffers[index] = (GLenum)mglRenderEmptyDrawBuffer();
    }
    mglRenderPassManagerStoreIdentity(cs, &identity);
}

/* Former @try body of -[MGLRenderer endRenderEncodingLocked]. */
static int mglRendererEndRenderEncodingGuardedBody(void *renderer)
{
    mglRenderPassManagerEndCurrentRenderEncoder(renderer);
    mglRenderPassManagerClearCurrentRenderEncoder(renderer);
    mglClearFragmentTraceBindingsForRenderer(renderer, "end_render_encoding");
    mglRenderPassManagerClearRenderPassIdentity(renderer);
    return 1;
}

/* Body of -[MGLRenderer endRenderEncodingLocked] (P0-1).  The @try/@catch is the
 * shell guard; when it fails the same three cleanups run as the old @catch. */
void mglRendererEndRenderEncodingLocked(void *renderer)
{
    mglBindingInvalidateLastBoundState(renderer);

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs ||
        mglRenderEncoderOwnerHasCurrent(cs->currentRenderEncoderOwner) != 1) {
        return;
    }

    /* An active render encoder means work was encoded into the current command
     * buffer, so the flush must not skip the commit. */
    if (areas.batching) {
        areas.batching->currentCommandBufferHasWork = 1u;
    }

    Framebuffer *endedFramebuffer = cs->renderPassFramebuffer;

    static uint64_t s_renderPassEndLogCount = 0;
    uint64_t hit = ++s_renderPassEndLogCount;
    if (hit <= 128ull || (hit % 1024ull) == 0ull) {
        mglLogRenderPassLifecycle("end", hit, areas.ctx,
                                  cs->currentCommandBufferOwner,
                                  cs->currentRenderEncoderOwner,
                                  cs->renderPassStateOwner,
                                  mglPlatformShellDrawable(renderer),
                                  cs->renderPassFramebuffer,
                                  cs->renderPassFramebufferName,
                                  cs->renderPassDrawBuffer,
                                  cs->renderPassDrawBufferCount);
    }

    if (!mglPlatformShellGuardedCall(renderer, "end render encoding",
                                     mglRendererEndRenderEncodingGuardedBody)) {
        fprintf(stderr,
                "MGL ERROR: Exception ending render encoder - ignoring\n");
        mglRenderPassManagerClearCurrentRenderEncoder(renderer);
        mglClearFragmentTraceBindingsForRenderer(renderer,
                                                 "end_render_encoding_exception");
        mglRenderPassManagerClearRenderPassIdentity(renderer);
    }

    /* A later batch may sample this render target before the command buffer is
     * submitted, so refresh its GL-visible copy immediately. */
    if (endedFramebuffer) {
        mglBlitUpdateGLSampledCopiesForEndedRenderPassFramebuffer(
            renderer, endedFramebuffer, "end_render_pass");
    }
}

/* Declared in MGLRenderer+Draw_Private.h; a real C function in the .m. */
extern Framebuffer *mglRendererGetValidatedFramebuffer(GLMContext ctx,
                                                      const char *where);

/* === dirty state domain processing (P0-1, log 167) ====================== */

/* MGL_STATE() from MGLRenderer_Private.h, in C (same twin as mgl_tess_dispatch.c). */
static GLMState *mglPdState(const MGLRendererStateAreas *areas)
{
    if (areas->core && areas->core->activeState) {
        return areas->core->activeState;
    }
    return areas->ctx ? areas->ctx->active_state : NULL;
}

/* -processDirtyStateDomainsLocked:work: */
bool mglRenderPassProcessDirtyStateDomains(void *renderer, int draw_command,
                                           MGLResourceSyncWork *work)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    int fboBindingDirty = 0;
    if ((mglPdState(&areas)->dirty_bits & (DIRTY_STATE | DIRTY_FBO)) ==
        (DIRTY_STATE | DIRTY_FBO)) {
        Framebuffer *framebuffer =
            mglRendererGetValidatedFramebuffer(ctx, "processGLState.dirtyStateFBO");
        if (framebuffer && (framebuffer->dirty_bits & DIRTY_FBO_BINDING)) {
            fboBindingDirty = 1;
        }
    }
    MGLDirtyDomainPlan plan = {0};
    if (mglRenderPlanDirtyDomains(
            mglPdState(&areas)->dirty_bits, draw_command ? 1 : 0,
            areas.pipeline_cache->pipelineState != NULL ? 1 : 0, fboBindingDirty,
            &plan) != 0) {
        return false;
    }

    bool deferredBufferMapForPipelineBuild = plan.defer_buffer_map;
    if (plan.has_dirty)
    {
        if (plan.sync_render_pass)
        {
            RETURN_FALSE_ON_FAILURE(mglRendererSyncRenderPassStateForContextPort(renderer, ctx));
        }

        if (plan.bind_fbo_attachments)
        {
            RETURN_FALSE_ON_FAILURE(mglRendererBindFramebufferAttachmentTextures(renderer));
            Framebuffer *framebuffer = mglRendererGetValidatedFramebuffer(
                ctx, "processGLState.dirtyStateFBO.afterBind");
            if (framebuffer) {
                framebuffer->dirty_bits &= ~DIRTY_FBO_BINDING;
            }
        }

        if (mglPdState(&areas)->dirty_bits & DIRTY_STATE)
        {
            mglPdState(&areas)->dirty_bits &= ~DIRTY_STATE;
        }

        if (plan.remap_buffers)
        {
            if (plan.defer_buffer_map) {
                static uint64_t s_deferredMapCount = 0;
                s_deferredMapCount++;
                if (s_deferredMapCount <= 16 || (s_deferredMapCount % 1000ull) == 0ull) {
                    mglTraceLog("MGL DRAW SKIP: pipelineState is nil (deferring buffer mapping, occurrence=%llu)",
                                  (unsigned long long)s_deferredMapCount);
                }
            } else {
                RETURN_FALSE_ON_FAILURE(mglRendererMapBuffersToMTL(renderer));
                if (work) work->mappedBuffers = true;
            }

            mglPdState(&areas)->dirty_bits &= ~DIRTY_BUFFER_BASE_STATE;
        }

        if (plan.bind_textures)
        {
            RETURN_FALSE_ON_FAILURE(mglBatchBindActiveTexturesToMTL(renderer, ctx));
            if (work) work->boundActiveTextures = true;

            mglPdState(&areas)->dirty_bits &= ~(DIRTY_TEX | DIRTY_TEX_PARAM | DIRTY_TEX_BINDING | DIRTY_SAMPLER);
        }

        if (plan.vao_path)
        {
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(renderer, &mglPdState(&areas)->vertex_buffer_map_list));
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(renderer, &mglPdState(&areas)->fragment_buffer_map_list));
            if (work) work->updatedBaseLists = true;

            if (mglRenderEncoderOwnerHasCurrent(
                    areas.command->currentRenderEncoderOwner) != 1) {
                RETURN_FALSE_ON_FAILURE(
                    mglRendererNewRenderEncoderLockedWithReasonPort(renderer, MGL_ENC_REASON_VAO));
            }

            mglRendererUpdateCurrentRenderEncoderPort(renderer);

            mglPdState(&areas)->dirty_bits &= ~DIRTY_RENDER_STATE;
        }
        else if (plan.buffer_path)
        {
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(renderer, &mglPdState(&areas)->vertex_buffer_map_list));
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(renderer, &mglPdState(&areas)->fragment_buffer_map_list));
            if (work) work->updatedBaseLists = true;

            mglPdState(&areas)->dirty_bits &= ~DIRTY_BUFFER;
        }
        else if (plan.render_state_path)
        {
            if (mglRenderEncoderOwnerHasCurrent(
                    areas.command->currentRenderEncoderOwner) != 1)
            {
                RETURN_FALSE_ON_FAILURE(
                    mglRendererNewRenderEncoderLockedWithReasonPort(renderer, MGL_ENC_REASON_RS));
            }

            mglRendererUpdateCurrentRenderEncoderPort(renderer);

            mglPdState(&areas)->dirty_bits &= ~DIRTY_RENDER_STATE;
        }

        if (plan.sync_pipeline)
        {
            RETURN_FALSE_ON_FAILURE(mglRendererSyncPipelineStateWithDeferredBufferMapPort(renderer, deferredBufferMapForPipelineBuild));
        }

        mglPdState(&areas)->dirty_bits = 0;
    }
    else
    {
        MGLEncodeContext encCtx = {
            .render_encoder_owner = areas.command->currentRenderEncoderOwner,
        };

        if( mglRendererCheckForDirtyBufferData(renderer, &mglPdState(&areas)->vertex_buffer_map_list))
        {
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(renderer, &mglPdState(&areas)->vertex_buffer_map_list));

            RETURN_FALSE_ON_FAILURE(mglStageEncodeBindVertexBuffers(renderer, &encCtx));
        }

        if( mglRendererCheckForDirtyBufferData(renderer, &mglPdState(&areas)->fragment_buffer_map_list))
        {
            RETURN_FALSE_ON_FAILURE(mglRendererUpdateDirtyBaseBufferList(renderer, &mglPdState(&areas)->fragment_buffer_map_list));

            RETURN_FALSE_ON_FAILURE(mglStageEncodeBindFragmentBuffers(renderer, &encCtx));
        }
    }
    return true;
}

/* -ensureRasterEncoderForDraw (P0-1, log 168).  Replaces
 * mglRendererEnsureRasterEncoderForDrawPort. */
int mglRenderPassEnsureRasterEncoderForDraw(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    if (mglRenderEncoderOwnerHasCurrent(
            areas.command->currentRenderEncoderOwner) == 1) {
        return 1;
    }
    (void)mglRendererNewRenderEncoderLockedWithReasonPort(renderer, MGL_ENC_REASON_DRAW);
    if (mglRenderEncoderOwnerHasCurrent(
            areas.command->currentRenderEncoderOwner) != 1) {
        return 0;
    }
    if (!areas.pipeline_cache->pipelineState) {
        return 0;
    }

    uint32_t rpColor0Format = 0u;
    uint32_t rpDepthFormat = 0u;
    uint32_t rpStencilFormat = 0u;
    MGLRenderPassAttachmentState colorAttachment = {0};
    MGLRenderPassAttachmentState depthAttachment = {0};
    MGLRenderPassAttachmentState stencilAttachment = {0};
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        areas.command->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0, &colorAttachment);
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        areas.command->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0, &depthAttachment);
    (void)mglRenderGetRenderPassAttachmentStateOwner(
        areas.command->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0, &stencilAttachment);
    void *rpColor0 = colorAttachment.texture;
    void *rpDepth = depthAttachment.texture;
    void *rpStencil = stencilAttachment.texture;
    MGLRenderTextureInfo textureInfo = {0};
    if (rpColor0 && mglRenderGetTextureInfo(
            rpColor0, &textureInfo) == 0) {
        rpColor0Format = textureInfo.pixel_format;
    }
    if (rpDepth && mglRenderGetTextureInfo(
            rpDepth, &textureInfo) == 0) {
        rpDepthFormat = textureInfo.pixel_format;
    }
    if (rpStencil && mglRenderGetTextureInfo(
            rpStencil, &textureInfo) == 0) {
        rpStencilFormat = textureInfo.pixel_format;
    }

    const int colorMismatch =
        (areas.pipeline_cache->pipelineColor0Format != 0u &&
         rpColor0Format != 0u &&
         areas.pipeline_cache->pipelineColor0Format != rpColor0Format);
    const int depthMismatch =
        (areas.pipeline_cache->pipelineDepthFormat != rpDepthFormat);
    const int stencilMismatch =
        (areas.pipeline_cache->pipelineStencilFormat != rpStencilFormat);
    if (colorMismatch || depthMismatch || stencilMismatch) {
        return 0;
    }
    if (mglRenderSetRenderPipelineStateForOwner(
            areas.command->currentRenderEncoderOwner,
            areas.pipeline_cache->pipelineState) != 0) {
        return 0;
    }
    mglRenderBindingSetPipelineState((areas.binding_state_owner ? *areas.binding_state_owner : NULL),
                                     areas.pipeline_cache->pipelineState);
    MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
    return 1;
}

/* ---- render-pass attachment snapshots (C twins of the .m file statics) ----
 * -mglRenderPassMatchesFramebufferImpl:name: moved here whole (log 169);
 * these five helpers moved with it.  The .m keeps its own copies for the
 * call sites that stay behind. */
static MGLRenderPassIdentityState mglRenderPassIdentitySnapshot(
    const MGLCommandState *commandState)
{
    MGLRenderPassIdentityState identity = {0};
    if (commandState && commandState->renderPassIdentityOwner &&
        mglRenderGetRenderPassIdentity(
            commandState->renderPassIdentityOwner, &identity) == 0) {
        return identity;
    }
    return identity;
}
static void *mglRenderPassDefaultDrawBufferAttachment(
    MGLRendererBackendHandle *backend, GLuint drawBufferIndex,
    MGLRendererBackendDefaultDrawBufferAttachmentKind kind)
{
    return 
        mglRendererBackendGetDefaultDrawBufferAttachment(
            backend, drawBufferIndex, kind);
}
static bool mglRenderPassGetPersistentState(
    const MGLCommandState *commandState,
    MGLRenderPassState *stateOut)
{
    return commandState && stateOut && commandState->renderPassStateOwner &&
           mglRenderGetRenderPassStateOwner(
               commandState->renderPassStateOwner, stateOut) == 0;
}
static const MGLRenderPassAttachmentState *
mglRenderPassAttachmentStateFromSnapshot(
    const MGLRenderPassState *state,
    uint32_t attachmentKind,
    size_t colorIndex)
{
    if (!state) return NULL;
    switch (mglRenderPassAttachmentClass(attachmentKind)) {
        case 1:
            return mglRenderPassColorAttachmentIndexValid(
                       (uint32_t)colorIndex, MAX_COLOR_ATTACHMENTS)
                ? &state->color[colorIndex].attachment : NULL;
        case 2:
            return &state->depth.attachment;
        case 3:
            return &state->stencil.attachment;
        default:
            return NULL;
    }
}
static void *mglRenderPassTextureFromSnapshot(
    const MGLRenderPassState *state,
    uint32_t attachmentKind,
    size_t colorIndex)
{
    const MGLRenderPassAttachmentState *attachment =
        mglRenderPassAttachmentStateFromSnapshot(
            state, attachmentKind, colorIndex);
    return attachment && attachment->texture
        ? attachment->texture : NULL;
}

/* -mglRenderPassMatchesFramebufferImpl:name: is C now (log 169).  Its three
 * Objective-C targets were already C (mglRendererBindMTLTexture,
 * mglRendererAttachmentTextureFor, mglRendererDrawableTexturePort), and
 * `_renderPassManager->state` is `areas.command`, so the move was mechanical. */
int mglRenderPassMatchesFramebufferImpl(void *renderer, void *framebuffer,
                                        unsigned int framebuffer_name)
{
    Framebuffer *fbo = (Framebuffer *)framebuffer;
    const GLuint fboName = framebuffer_name;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLRenderPassState passState = {0};
    bool hasPassState =
        mglRenderPassGetPersistentState(areas.command, &passState);
    if (!ctx || !hasPassState) {
        return 1;
    }
    MGLRenderPassIdentityState identity =
        mglRenderPassIdentitySnapshot(areas.command);
    if (identity.framebuffer != fbo ||
        identity.framebuffer_name != fboName ||
        identity.draw_buffer != mglPdState(&areas)->draw_buffer ||
        identity.draw_buffer_count != (uint32_t)mglMetalDrawBufferCount(ctx)) {
        return 0;
    }
    for (uint32_t i = 0; i < identity.draw_buffer_count; ++i) {
        if (identity.draw_buffers[i] != mglMetalDrawBufferAt(ctx, i)) {
            return 0;
        }
    }

    if (!fbo) {
        GLuint mgl_drawbuffer = mglDefaultDrawBufferIndexForGL(mglPdState(&areas)->draw_buffer);
        void * expectedColor0 = NULL;
        void * actualColor0 = mglRenderPassTextureFromSnapshot(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);

        if (mglRenderDefaultDrawBufferIsFront(mgl_drawbuffer)) {
            /* _drawable ? [self mglDrawableTexture] : nil is the same value:
             * -mglDrawableTexture is self.drawable.texture, which is nil when
             * there is no drawable. */
            expectedColor0 = mglRendererDrawableTexturePort(renderer);
        } else if (mglRenderDefaultDrawBufferIsOffscreen(
                       mgl_drawbuffer, _MAX_DRAW_BUFFERS)) {
            expectedColor0 = mglRenderPassDefaultDrawBufferAttachment(
                areas.backend, mgl_drawbuffer,
                MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR);
        }

        void * expectedDepth = NULL;
        void * expectedStencil = NULL;
        int defaultPassNeedsDepth = 0;
        int defaultPassNeedsStencil = 0;
        if (mgl_drawbuffer < _MAX_DRAW_BUFFERS) {
            void * cachedDepth =
                mglRenderPassDefaultDrawBufferAttachment(
                    areas.backend, mgl_drawbuffer,
                    MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH);
            void * cachedStencil =
                mglRenderPassDefaultDrawBufferAttachment(
                    areas.backend, mgl_drawbuffer,
                    MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_STENCIL);
            defaultPassNeedsDepth = mglPdState(&areas)->caps.depth_test ||
                                    cachedDepth != NULL;
            defaultPassNeedsStencil = mglPdState(&areas)->caps.stencil_test ||
                                      ctx->stencil_format.format ||
                                      cachedStencil != NULL;
            expectedDepth = defaultPassNeedsDepth ? cachedDepth : NULL;
            expectedStencil = defaultPassNeedsStencil ? cachedStencil : NULL;
        }

        /* The comparison (and "a required attachment that is missing never
         * matches") is the same rule the user-FBO half uses; it lives in the
         * plan. */
        const MGLRenderPassSubresource no_sub = {0u, 0u, 0u};
        MGLRenderPassAttachmentMatchEntry entries[3];
        mglRenderPassFillMatchEntry(
            &entries[0], actualColor0,
            expectedColor0, 0, 0, no_sub, no_sub);
        mglRenderPassFillMatchEntry(
            &entries[1],
            mglRenderPassTextureFromSnapshot(
                &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0),
            expectedDepth,
            mglPdState(&areas)->caps.depth_test ? 1 : 0, 0, no_sub, no_sub);
        mglRenderPassFillMatchEntry(
            &entries[2],
            mglRenderPassTextureFromSnapshot(
                &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0),
            expectedStencil,
            (mglPdState(&areas)->caps.stencil_test || ctx->stencil_format.format) ? 1
                                                                             : 0,
            0, no_sub, no_sub);
        MGLRenderPassAttachmentMatchInput match = {0};
        match.identity_ok = 1;
        match.entries = entries;
        match.entry_count = 3;
        return mglRenderPassAttachmentsMatch(&match) ? 1 : 0;
    }

    for (GLuint i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        GLuint attachmentIndex = 0u;
        GLuint colorSlot = mglMetalColorSlotForDrawBuffer(ctx, i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        int drawSlotPresent =
            mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, i),
                                                  &attachmentIndex) &&
            attachmentIndex < MAX_COLOR_ATTACHMENTS &&
            ((fbo->color_attachment_bitfield >> attachmentIndex) & 1u) != 0u;
        FBOAttachment *attachment = drawSlotPresent ? &fbo->color_attachments[attachmentIndex] : NULL;
        Texture *tex = drawSlotPresent ? mglRendererAttachmentTextureFor(ctx, attachment) : NULL;
        void * expected = NULL;

        if (tex) {
            tex->is_render_target = 1;
            if (!tex->mtl_data) {
                if (!mglRendererBindMTLTexture(renderer, tex)) {
                    return 0;
                }
            }
            expected = (tex->mtl_data);
        }

        void * actual = mglRenderPassTextureFromSnapshot(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot);
        MGLRenderPassSubresource actualSub = {0u, 0u, 0u};
        MGLRenderPassSubresource expectedSub = {0u, 0u, 0u};
        int compareSubresource = 0;
        if (attachment && actual) {
            const MGLRenderPassAttachmentState *snapshot =
                mglRenderPassAttachmentStateFromSnapshot(
                    &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                    colorSlot);
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(attachment);
            compareSubresource = snapshot ? 1 : 0;
            if (snapshot) {
                actualSub = (MGLRenderPassSubresource){snapshot->level,
                                                       snapshot->slice,
                                                       snapshot->depth_plane};
            }
            expectedSub = (MGLRenderPassSubresource){subresource.level,
                                                     subresource.slice,
                                                     subresource.depthPlane};
        }
        MGLRenderPassAttachmentMatchEntry slotEntry;
        mglRenderPassFillMatchEntry(&slotEntry,
                                    actual,
                                    expected, 0,
                                    compareSubresource, actualSub, expectedSub);
        MGLRenderPassAttachmentMatchInput slotMatch = {0};
        slotMatch.identity_ok = 1;
        slotMatch.entries = &slotEntry;
        slotMatch.entry_count = 1;
        if (!mglRenderPassAttachmentsMatch(&slotMatch)) {
            return 0;
        }

        void * nextColor = NULL;
        if (i + 1u < MAX_COLOR_ATTACHMENTS) {
            nextColor = mglRenderPassTextureFromSnapshot(
                &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                i + 1u);
        }
        if (mglRenderStopColorAttachmentScan(
                i + 1u, (uint32_t)MAX_COLOR_ATTACHMENTS,
                mglRenderDrawBufferIsNone(
                    (uint32_t)mglMetalDrawBufferAt(ctx, i + 1u)),
                nextColor ? 1 : 0)) {
            break;
        }
    }

    void * expectedDepth = NULL;
    if (fbo->depth.texture) {
        Texture *depthTex = mglRendererAttachmentTextureFor(ctx, &fbo->depth);
        if (depthTex && !depthTex->mtl_data) {
            depthTex->is_render_target = 1;
            if (!mglRendererBindMTLTexture(renderer, depthTex)) {
                return 0;
            }
        }
        expectedDepth = depthTex ? (depthTex->mtl_data) : NULL;
    }
    void * actualDepth = mglRenderPassTextureFromSnapshot(
        &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    {
        MGLRenderPassSubresource actualSub = {0u, 0u, 0u};
        MGLRenderPassSubresource expectedSub = {0u, 0u, 0u};
        int compareSubresource = 0;
        if (fbo->depth.texture && expectedDepth) {
            const MGLRenderPassAttachmentState *snapshot =
                mglRenderPassAttachmentStateFromSnapshot(
                    &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->depth);
            compareSubresource = snapshot ? 1 : 0;
            if (snapshot) {
                actualSub = (MGLRenderPassSubresource){snapshot->level,
                                                       snapshot->slice,
                                                       snapshot->depth_plane};
            }
            expectedSub = (MGLRenderPassSubresource){subresource.level,
                                                     subresource.slice,
                                                     subresource.depthPlane};
        }
        MGLRenderPassAttachmentMatchEntry entry;
        mglRenderPassFillMatchEntry(&entry,
                                    actualDepth,
                                    expectedDepth, 0,
                                    compareSubresource, actualSub, expectedSub);
        MGLRenderPassAttachmentMatchInput entryMatch = {0};
        entryMatch.identity_ok = 1;
        entryMatch.entries = &entry;
        entryMatch.entry_count = 1;
        if (!mglRenderPassAttachmentsMatch(&entryMatch)) {
            return 0;
        }
    }

    void * expectedStencil = NULL;
    if (fbo->stencil.texture) {
        Texture *stencilTex = mglRendererAttachmentTextureFor(ctx, &fbo->stencil);
        if (stencilTex && !stencilTex->mtl_data) {
            stencilTex->is_render_target = 1;
            if (!mglRendererBindMTLTexture(renderer, stencilTex)) {
                return 0;
            }
        }
        expectedStencil = stencilTex ? (stencilTex->mtl_data) : NULL;
    }
    void * actualStencil = mglRenderPassTextureFromSnapshot(
        &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
    {
        MGLRenderPassSubresource actualSub = {0u, 0u, 0u};
        MGLRenderPassSubresource expectedSub = {0u, 0u, 0u};
        int compareSubresource = 0;
        if (fbo->stencil.texture && expectedStencil) {
            const MGLRenderPassAttachmentState *snapshot =
                mglRenderPassAttachmentStateFromSnapshot(
                    &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->stencil);
            compareSubresource = snapshot ? 1 : 0;
            if (snapshot) {
                actualSub = (MGLRenderPassSubresource){snapshot->level,
                                                       snapshot->slice,
                                                       snapshot->depth_plane};
            }
            expectedSub = (MGLRenderPassSubresource){subresource.level,
                                                     subresource.slice,
                                                     subresource.depthPlane};
        }
        MGLRenderPassAttachmentMatchEntry entry;
        mglRenderPassFillMatchEntry(&entry,
                                    actualStencil,
                                    expectedStencil, 0,
                                    compareSubresource, actualSub, expectedSub);
        MGLRenderPassAttachmentMatchInput entryMatch = {0};
        entryMatch.identity_ok = 1;
        entryMatch.entries = &entry;
        entryMatch.entry_count = 1;
        if (!mglRenderPassAttachmentsMatch(&entryMatch)) {
            return 0;
        }
    }

    return 1;
}
