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
#include "mgl_air_loader.h"        /* MGLRenderPipelineDescriptorState */
#include "mgl_byte_hash.h"         /* mglHashStepU64 */
#include "mgl_draw_tess.h"         /* mglTessRasterGLMode */
#include "mgl_vertex_layout.h"     /* mglRendererGenerateVertexDescriptorState */
#include "mgl_render_pass_manager.h" /* pass-manager transaction entries */
#include "mgl_render_pass_clear.h"   /* mglRenderPassPlanClearValues */
#include "mgl_trace_log.h"         /* mglTraceLog, kMGLDiagnosticStateLogs */
#include "mgl_gpu_recovery.h"      /* mglRendererRecordGPUError */
#include "mgl_pso_format_class.h"  /* mglRenderDefaultColorPixelFormat */
#include "mgl_render.h"           /* attachment kinds, MS plane adjust */
#include "mgl_texture_compat.h"   /* mglMetalTextureLevelDimension */

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

/* === user-FBO attachment configuration (log 170) ======================== */

/* MGLRenderer+RenderPass_Private.h (Objective-C) declares this one with `id`;
 * the C twin restates the pointer-typed ABI. */
extern void *mglApplySRGBStateToRenderTarget(void *texture, GLMContext ctx);

/* The _mgl* sample-loop ivars travel through the shell forwarders, exactly as
 * in mgl_ms_sample_loop.c. */
extern int mglPlatformShellMSSampleInLoop(void *renderer);
extern int mglPlatformShellMSSamplePlaneOffset(void *renderer);

/* C twins of the .m statics the moved methods used. */
static bool mglPdGetPersistentAttachmentState(
    const MGLCommandState *commandState, uint32_t attachmentKind,
    size_t colorIndex, MGLRenderPassAttachmentState *attachmentOut)
{
    if (!attachmentOut) return false;
    MGLRenderPassState state = {0};
    if (!mglRenderPassGetPersistentState(commandState, &state)) return false;
    const MGLRenderPassAttachmentState *attachment =
        mglRenderPassAttachmentStateFromSnapshot(&state, attachmentKind,
                                                 colorIndex);
    if (!attachment) return false;
    *attachmentOut = *attachment;
    return true;
}

static void *mglPdAttachmentTextureFor(const MGLCommandState *commandState,
                                       uint32_t attachmentKind,
                                       size_t colorIndex)
{
    MGLRenderPassAttachmentState attachment = {0};
    if (mglPdGetPersistentAttachmentState(commandState, attachmentKind,
                                          colorIndex, &attachment)) {
        return attachment.texture;
    }
    return NULL;
}

static bool mglPdRenderTargetSizeFor(const MGLCommandState *commandState,
                                     uint64_t *widthOut, uint64_t *heightOut)
{
    MGLRenderPassState state = {0};
    if (!mglRenderPassGetPersistentState(commandState, &state)) return false;
    if (widthOut) *widthOut = state.render_target_width;
    if (heightOut) *heightOut = state.render_target_height;
    return true;
}

static uint64_t mglPdRenderTargetWidthFor(const MGLCommandState *commandState)
{
    uint64_t width = 0;
    if (mglPdRenderTargetSizeFor(commandState, &width, NULL)) return width;
    return 0;
}

static uint64_t mglPdRenderTargetHeightFor(const MGLCommandState *commandState)
{
    uint64_t height = 0;
    if (mglPdRenderTargetSizeFor(commandState, NULL, &height)) return height;
    return 0;
}

static void mglPdSetPersistentAttachment(const MGLCommandState *commandState,
                                         uint32_t attachmentKind,
                                         size_t colorIndex, void *texture,
                                         uint64_t level, uint64_t slice,
                                         uint64_t depthPlane, int layered)
{
    if (commandState && commandState->renderPassStateOwner) {
        (void)mglRenderSetRenderPassStateAttachmentTexture(
            commandState->renderPassStateOwner, attachmentKind,
            (uint32_t)colorIndex, texture, level, slice, depthPlane,
            layered ? 1u : 0u);
    }
}

static void mglPdSetPersistentDimensions(const MGLCommandState *commandState,
                                         uint64_t width, uint64_t height)
{
    if (commandState && commandState->renderPassStateOwner) {
        (void)mglRenderSetRenderPassStateDimensions(
            commandState->renderPassStateOwner, width, height);
    }
}

static uint64_t mglPdMin(uint64_t a, uint64_t b) { return a < b ? a : b; }

/* -configureUserFBOAttachmentsLocked. */
bool mglRenderPassConfigureUserFBOAttachments(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *commandState = areas.command;
    Framebuffer *fbo = mglPdState(&areas)->framebuffer;

    const GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
    for (int i = 0; i < drawBufferCount; i++) {
        GLuint attachmentIndex = 0u;
        const GLuint colorSlot =
            mglMetalColorSlotForDrawBuffer(ctx, (GLuint)i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        if (mglMetalResolveFboDrawAttachmentIndex(
                ctx, mglMetalDrawBufferAt(ctx, (GLuint)i), &attachmentIndex) &&
            attachmentIndex < MAX_COLOR_ATTACHMENTS &&
            (fbo->color_attachment_bitfield & (1u << attachmentIndex)) &&
            fbo->color_attachments[attachmentIndex].texture) {
            Texture *tex = mglRendererAttachmentTextureFor(
                ctx, &fbo->color_attachments[attachmentIndex]);
            if (!tex) {
                continue;
            }

            /* Ensure attachment textures are created with RenderTarget usage. */
            tex->is_render_target = 1;
            if (!mglRendererBindMTLTexture(renderer, tex)) {
                fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
                return false;
            }
            if (!tex->mtl_data) {
                continue;
            }

            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(
                    &fbo->color_attachments[attachmentIndex]);
            const int32_t msOffset =
                mglPlatformShellMSSamplePlaneOffset(renderer);
            if (mglRenderMSSamplePlaneAdjust(
                    mglPlatformShellMSSampleInLoop(renderer) ? 1 : 0,
                    (uint32_t)tex->target, msOffset)) {
                subresource.slice += (uint32_t)msOffset;
            }
            mglPdSetPersistentAttachment(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot,
                mglApplySRGBStateToRenderTarget(tex->mtl_data, ctx),
                subresource.level, subresource.slice, subresource.depthPlane,
                fbo->color_attachments[attachmentIndex].layered ? 1 : 0);

            if (mglRenderTextureTargetIsMSOr2DArray((uint32_t)tex->target)) {
                void *rpTex = mglPdAttachmentTextureFor(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                    (size_t)colorSlot);
                (void)rpTex;
            }

            /* Keep render pass dimensions aligned with attached color targets.
             * Some FBO paths use textures (not renderbuffers), and Metal still
             * requires scissor/viewport to be bounded by the attachment
             * dimensions. */
            const uint64_t attWidth = mglMetalTextureLevelDimension(
                (size_t)tex->width, subresource.level);
            const uint64_t attHeight = mglMetalTextureLevelDimension(
                (size_t)tex->height, subresource.level);
            if (attWidth > 0 && attHeight > 0) {
                if (mglPdRenderTargetWidthFor(commandState) == 0 ||
                    mglPdRenderTargetHeightFor(commandState) == 0) {
                    mglPdSetPersistentDimensions(commandState, attWidth,
                                                 attHeight);
                } else if (mglPdRenderTargetWidthFor(commandState) != attWidth ||
                           mglPdRenderTargetHeightFor(commandState) !=
                               attHeight) {
                    const uint64_t oldWidth =
                        mglPdRenderTargetWidthFor(commandState);
                    const uint64_t oldHeight =
                        mglPdRenderTargetHeightFor(commandState);
                    mglPdSetPersistentDimensions(
                        commandState,
                        mglPdMin(mglPdRenderTargetWidthFor(commandState),
                                 attWidth),
                        mglPdMin(mglPdRenderTargetHeightFor(commandState),
                                 attHeight));
                    fprintf(stderr,
                            "MGL WARNING: FBO color attachment size mismatch "
                            "slot=%d old=%lux%lu new=%lux%lu resolved=%lux%lu\n",
                            i, (unsigned long)oldWidth, (unsigned long)oldHeight,
                            (unsigned long)attWidth, (unsigned long)attHeight,
                            (unsigned long)mglPdRenderTargetWidthFor(commandState),
                            (unsigned long)mglPdRenderTargetHeightFor(commandState));
                }
            }
        }
    }

    /* depth attachment */
    if (fbo->depth.texture) {
        Texture *tex = mglRendererAttachmentTextureFor(ctx, &fbo->depth);
        if (tex) {
            tex->is_render_target = 1;
            if (!mglRendererBindMTLTexture(renderer, tex)) {
                fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
                return false;
            }
        }
        if (tex && tex->mtl_data) {
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->depth);
            mglPdSetPersistentAttachment(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                tex->mtl_data, subresource.level, subresource.slice,
                subresource.depthPlane, fbo->depth.layered ? 1 : 0);
        }
    }

    /* stencil attachment */
    if (fbo->stencil.texture) {
        Texture *tex = mglRendererAttachmentTextureFor(ctx, &fbo->stencil);
        if (tex) {
            tex->is_render_target = 1;
            if (!mglRendererBindMTLTexture(renderer, tex)) {
                fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
                return false;
            }
        }
        if (tex && tex->mtl_data) {
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->stencil);
            mglPdSetPersistentAttachment(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                tex->mtl_data, subresource.level, subresource.slice,
                subresource.depthPlane, fbo->stencil.layered ? 1 : 0);
        }
    }
    return true;
}

/* === render-pass descriptor finalization (log 171) ====================== */

/* Defined in MGLRenderer.m; declared in the Objective-C
 * MGLRenderer+RenderPass_Private.h. */
extern GLuint mglRendererSafeFramebufferName(GLMContext ctx);

/* C twins of the .m statics the moved method used. */
static void *mglPdDepthTextureFor(const MGLCommandState *commandState)
{
    return mglPdAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u);
}

static void *mglPdStencilTextureFor(const MGLCommandState *commandState)
{
    return mglPdAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u);
}

static MGLRenderTextureInfo mglPdTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

static bool mglPdActionsFor(const MGLCommandState *commandState,
                            uint32_t attachmentKind, size_t colorIndex,
                            uint32_t *loadActionOut, uint32_t *storeActionOut,
                            uint64_t *storeActionOptionsOut)
{
    MGLRenderPassAttachmentState attachment = {0};
    if (mglPdGetPersistentAttachmentState(commandState, attachmentKind,
                                          colorIndex, &attachment)) {
        if (loadActionOut) *loadActionOut = attachment.load_action;
        if (storeActionOut) *storeActionOut = attachment.store_action;
        if (storeActionOptionsOut) {
            *storeActionOptionsOut = attachment.store_action_options;
        }
        return true;
    }
    return false;
}

static uint32_t mglPdLoadActionFor(const MGLCommandState *commandState,
                                   uint32_t attachmentKind, size_t colorIndex,
                                   uint32_t fallback)
{
    uint32_t action = 0u;
    if (mglPdActionsFor(commandState, attachmentKind, colorIndex, &action, NULL,
                        NULL)) {
        return action;
    }
    return fallback;
}

static uint32_t mglPdStoreActionFor(const MGLCommandState *commandState,
                                    uint32_t attachmentKind, size_t colorIndex,
                                    uint32_t fallback)
{
    uint32_t action = 0u;
    if (mglPdActionsFor(commandState, attachmentKind, colorIndex, NULL, &action,
                        NULL)) {
        return action;
    }
    return fallback;
}

static void mglPdSetPersistentActions(const MGLCommandState *commandState,
                                      uint32_t attachmentKind,
                                      size_t colorIndex, uint32_t loadAction,
                                      uint32_t storeAction)
{
    if (!commandState) return;
    MGLRenderPassAttachmentState state = {0};
    if (!mglPdGetPersistentAttachmentState(commandState, attachmentKind,
                                           colorIndex, &state)) {
        return;
    }
    if (commandState->renderPassStateOwner) {
        (void)mglRenderSetRenderPassStateAttachmentActions(
            commandState->renderPassStateOwner, attachmentKind,
            (uint32_t)colorIndex, loadAction, storeAction,
            state.store_action_options);
    }
}

static void mglPdSetPersistentStoreAction(const MGLCommandState *commandState,
                                          uint32_t attachmentKind,
                                          size_t colorIndex,
                                          uint32_t storeAction)
{
    uint32_t loadAction = (uint32_t)MGLLoadActionDontCare;
    MGLRenderPassAttachmentState state = {0};
    if (mglPdGetPersistentAttachmentState(commandState, attachmentKind,
                                          colorIndex, &state)) {
        loadAction = (uint32_t)state.load_action;
    } else {
        return;
    }
    mglPdSetPersistentActions(commandState, attachmentKind, colorIndex,
                              loadAction, storeAction);
}

static MGLRendererBackendHandle *mglPdBackend(GLMContext ctx)
{
    return ctx ? (MGLRendererBackendHandle *)ctx->renderer_backend : NULL;
}

static void *mglPdFallbackRenderTarget(GLMContext ctx)
{
    return mglRendererBackendGetFallbackRenderTargetTexture(mglPdBackend(ctx));
}

/* The .m twin returns (__bridge_transfer id): ARC consumes the +1 the creator
 * hands back.  C has no ARC, so the reference is left to the backend's own
 * ownership - the texture is reusable and the context creates at most one. */
static void *mglPdCreateTexture(
    const MGLRenderTextureDescriptorState *descriptor)
{
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(descriptor, NULL, &texture) == 0 &&
        texture) {
        return texture;
    }
    return NULL;
}

static uint64_t mglPdMax(uint64_t a, uint64_t b) { return a > b ? a : b; }

static void *mglPdFallbackRenderTargetForSize(GLMContext ctx, uint64_t width,
                                              uint64_t height,
                                              uint64_t layerCount,
                                              uint64_t sampleCount)
{
    width = mglPdMax(width, 1u);
    height = mglPdMax(height, 1u);
    sampleCount = mglPdMax(sampleCount, 1u);
    const int layered = layerCount > 0u;
    const uint64_t arrayLength = layered ? mglPdMax(layerCount, 1u) : 1u;
    const uint32_t textureType =
        layered ? (uint32_t)MGLTextureType2DArray : (uint32_t)MGLTextureType2D;
    void *texture = mglPdFallbackRenderTarget(ctx);
    MGLRenderTextureInfo info = mglPdTextureInfo(texture);
    if (texture && info.width == width && info.height == height &&
        info.array_length == arrayLength && info.texture_type == textureType &&
        info.sample_count == sampleCount) {
        return texture;
    }

    MGLRenderTextureDescriptorState desc = {0};
    desc.texture_type = textureType;
    desc.pixel_format = mglRenderDefaultColorPixelFormat();
    desc.width = width;
    desc.height = height;
    desc.depth = 1;
    desc.mipmap_level_count = 1;
    desc.sample_count = sampleCount;
    desc.array_length = arrayLength;
    desc.usage = MGLTextureUsageRenderTarget | MGLTextureUsageShaderRead;
    desc.storage_mode = MGLStorageModeShared;
    void *replacement = mglPdCreateTexture(&desc);
    if (!replacement ||
        mglRendererBackendSetFallbackRenderTargetTexture(mglPdBackend(ctx),
                                                         replacement) != 0) {
        return NULL;
    }
    return mglPdFallbackRenderTarget(ctx);
}

/* MGLRenderer+RenderPass_Private.h: `static const BOOL
 * kMGLVerboseFrameLoopLogs = NO;` - kept so the branch below stays verbatim. */
static const int kMglPdVerboseFrameLoopLogs = 0;

/* -finalizeRenderPassDescriptorLocked:traceRenderEncoder:. */
bool mglRenderPassFinalizeRenderPassDescriptor(void *renderer,
                                               uint64_t renderEncoderCall,
                                               int traceRenderEncoder)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *commandState = areas.command;

    mglPdSetPersistentStoreAction(commandState,
                                 MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                                 MGLStoreActionStore);

    if (kMGLDiagnosticStateLogs && traceRenderEncoder) {
        void *c0Tex = mglPdAttachmentTextureFor(
            commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
        void *dTex = mglPdDepthTextureFor(commandState);
        void *sTex = mglPdStencilTextureFor(commandState);
        mglTraceLog(
            "MGL TRACE renderpass.attach call=%llu fbo=%u drawBuf=0x%x rt=%lux%lu "
            "c0=%p fmt=%lu usage=0x%lx size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s stencil=%p fmt=%lu size=%lux%lu la/sa=%s/%s",
            (unsigned long long)renderEncoderCall,
            (unsigned)mglRendererSafeFramebufferName(ctx),
            (unsigned)mglPdState(&areas)->draw_buffer,
            (unsigned long)mglPdRenderTargetWidthFor(commandState),
            (unsigned long)mglPdRenderTargetHeightFor(commandState), c0Tex,
            (unsigned long)(c0Tex ? mglPdTextureInfo(c0Tex).pixel_format
                                  : mglRenderInvalidPixelFormat()),
            (unsigned long)(c0Tex ? mglPdTextureInfo(c0Tex).usage : 0),
            (unsigned long)(c0Tex ? mglPdTextureInfo(c0Tex).width : 0),
            (unsigned long)(c0Tex ? mglPdTextureInfo(c0Tex).height : 0),
            mglLoadActionName(mglPdLoadActionFor(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                MGLLoadActionDontCare)),
            mglStoreActionName(mglPdStoreActionFor(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                MGLStoreActionDontCare)),
            dTex,
            (unsigned long)(dTex ? mglPdTextureInfo(dTex).pixel_format
                                 : mglRenderInvalidPixelFormat()),
            (unsigned long)(dTex ? mglPdTextureInfo(dTex).width : 0),
            (unsigned long)(dTex ? mglPdTextureInfo(dTex).height : 0),
            mglLoadActionName(mglPdLoadActionFor(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                MGLLoadActionDontCare)),
            mglStoreActionName(mglPdStoreActionFor(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                MGLStoreActionDontCare)),
            sTex,
            (unsigned long)(sTex ? mglPdTextureInfo(sTex).pixel_format
                                 : mglRenderInvalidPixelFormat()),
            (unsigned long)(sTex ? mglPdTextureInfo(sTex).width : 0),
            (unsigned long)(sTex ? mglPdTextureInfo(sTex).height : 0),
            mglLoadActionName(mglPdLoadActionFor(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                MGLLoadActionDontCare)),
            mglStoreActionName(mglPdStoreActionFor(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                MGLStoreActionDontCare)));
    }

    /* create a render encoder from the renderpass descriptor
     * CRITICAL SAFETY: Validate inputs before creating render encoder */
    const int hasRenderPassState =
        commandState->renderPassStateOwner != NULL;
    if (!hasRenderPassState) {
        fprintf(stderr,
                "MGL ERROR: Cannot create render encoder - state owner is NULL\n");
        mglRendererRecordGPUError(renderer);
        return false;
    }

    /* Metal debug layer crashes if render pass has no output attachment.
     * Provide a tiny fallback color attachment for targetless/invalid passes. */
    bool hasOutputAttachment = false;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (mglPdAttachmentTextureFor(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                (size_t)i)) {
            hasOutputAttachment = true;
            break;
        }
    }
    if (!hasOutputAttachment &&
        (mglPdDepthTextureFor(commandState) ||
         mglPdStencilTextureFor(commandState))) {
        hasOutputAttachment = true;
    }

    if (!hasOutputAttachment) {
        Framebuffer *fbo = mglPdState(&areas)->framebuffer;
        const uint64_t fallbackWidth =
            fbo && fbo->default_width > 0 ? (uint64_t)fbo->default_width : 1u;
        const uint64_t fallbackHeight =
            fbo && fbo->default_height > 0 ? (uint64_t)fbo->default_height : 1u;
        const uint64_t fallbackLayers =
            fbo && fbo->default_layers > 0 ? (uint64_t)fbo->default_layers : 0u;
        const uint64_t fallbackSamples =
            fbo && fbo->default_samples > 0 ? (uint64_t)fbo->default_samples : 1u;
        void *fallbackRenderTarget = mglPdFallbackRenderTargetForSize(
            ctx, fallbackWidth, fallbackHeight, fallbackLayers,
            fallbackSamples);

        if (fallbackRenderTarget) {
            fprintf(stderr,
                    "MGL WARNING: Render pass had no attachments; binding %lux%lu fallback color target\n",
                    (unsigned long)fallbackWidth, (unsigned long)fallbackHeight);
            mglPdSetPersistentAttachment(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                fallbackRenderTarget, 0, 0, 0,
                fallbackLayers > 0u ? 1 : 0);
            mglPdSetPersistentActions(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                MGLLoadActionLoad, MGLStoreActionStore);
            mglPdSetPersistentDimensions(commandState, fallbackWidth,
                                         fallbackHeight);
        } else {
            fprintf(stderr,
                    "MGL ERROR: Failed to allocate fallback render target texture\n");
            mglRendererRecordGPUError(renderer);
            return false;
        }
    }

    /* Final guard: Metal will assert if a color attachment texture is missing
     * RenderTarget usage. */
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        void *attTex = mglPdAttachmentTextureFor(
            commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, (size_t)i);
        if (attTex &&
            (mglPdTextureInfo(attTex).usage & MGLTextureUsageRenderTarget) == 0) {
            fprintf(stderr,
                    "MGL WARNING: colorAttachment[%d] usage=0x%lx lacks RenderTarget; clearing attachment to avoid Metal assert\n",
                    i, (unsigned long)mglPdTextureInfo(attTex).usage);
            uint64_t clearLevel = 0u, clearSlice = 0u, clearDepthPlane = 0u;
            (void)mglRenderGetRenderPassAttachmentSubresourceOwner(
                commandState->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, (uint32_t)i,
                &clearLevel, &clearSlice, &clearDepthPlane);
            mglPdSetPersistentAttachment(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                (size_t)i, NULL, clearLevel, clearSlice, clearDepthPlane, 0);
        }
    }

    /* Default-framebuffer paths expect color attachment 0 specifically.
     * FBO draw-buffer mappings may intentionally leave slot 0 as GL_NONE. */
    if (!mglPdState(&areas)->framebuffer &&
        !mglPdAttachmentTextureFor(commandState,
                                   MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                                   0)) {
        for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) {
            if (mglPdAttachmentTextureFor(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                    (size_t)i)) {
                fprintf(stderr,
                        "MGL WARNING: colorAttachment[0] missing; remapping colorAttachment[%d] -> [0]\n",
                        i);
                uint64_t srcLevel = 0u, srcSlice = 0u, srcDepthPlane = 0u;
                (void)mglRenderGetRenderPassAttachmentSubresourceOwner(
                    commandState->renderPassStateOwner,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, (uint32_t)i,
                    &srcLevel, &srcSlice, &srcDepthPlane);
                mglPdSetPersistentAttachment(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    mglPdAttachmentTextureFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                        (size_t)i),
                    srcLevel, srcSlice, srcDepthPlane, 0);
                mglPdSetPersistentActions(
                    commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                    mglPdLoadActionFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                        (size_t)i, MGLLoadActionLoad),
                    mglPdStoreActionFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                        (size_t)i, MGLStoreActionStore));
                break;
            }
        }
    }

    /* Ultimate slot-0 fallback to keep draw path alive and avoid black frame. */
    if (!hasOutputAttachment &&
        !mglPdAttachmentTextureFor(commandState,
                                   MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                                   0)) {
        Framebuffer *fbo = mglPdState(&areas)->framebuffer;
        const uint64_t fallbackWidth =
            fbo && fbo->default_width > 0 ? (uint64_t)fbo->default_width : 1u;
        const uint64_t fallbackHeight =
            fbo && fbo->default_height > 0 ? (uint64_t)fbo->default_height : 1u;
        const uint64_t fallbackLayers =
            fbo && fbo->default_layers > 0 ? (uint64_t)fbo->default_layers : 0u;
        const uint64_t fallbackSamples =
            fbo && fbo->default_samples > 0 ? (uint64_t)fbo->default_samples : 1u;
        void *fallbackRenderTarget = mglPdFallbackRenderTargetForSize(
            ctx, fallbackWidth, fallbackHeight, fallbackLayers,
            fallbackSamples);
        if (fallbackRenderTarget) {
            fprintf(stderr,
                    "MGL WARNING: colorAttachment[0] unavailable; binding %lux%lu fallback\n",
                    (unsigned long)fallbackWidth, (unsigned long)fallbackHeight);
            mglPdSetPersistentAttachment(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                fallbackRenderTarget, 0, 0, 0,
                fallbackLayers > 0u ? 1 : 0);
            mglPdSetPersistentActions(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0,
                MGLLoadActionLoad, MGLStoreActionStore);
            mglPdSetPersistentDimensions(commandState, fallbackWidth,
                                         fallbackHeight);
        } else {
            fprintf(stderr,
                    "MGL ERROR: Unable to allocate fallback colorAttachment[0] texture\n");
            mglRendererRecordGPUError(renderer);
            return false;
        }
    }

    /* Ensure renderTargetWidth/Height are always coherent with the active
     * attachments. */
    {
        void *sizeTex = mglPdAttachmentTextureFor(
            commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
        if (!sizeTex) {
            for (int i = 1; i < MAX_COLOR_ATTACHMENTS; i++) {
                if (mglPdAttachmentTextureFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                        (size_t)i)) {
                    sizeTex = mglPdAttachmentTextureFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                        (size_t)i);
                    break;
                }
            }
        }
        if (!sizeTex) {
            sizeTex = mglPdDepthTextureFor(commandState);
        }
        if (!sizeTex) {
            sizeTex = mglPdStencilTextureFor(commandState);
        }

        if (sizeTex) {
            const uint64_t texWidth = mglPdTextureInfo(sizeTex).width;
            const uint64_t texHeight = mglPdTextureInfo(sizeTex).height;
            if (mglPdRenderTargetWidthFor(commandState) == 0 ||
                mglPdRenderTargetHeightFor(commandState) == 0 ||
                mglPdRenderTargetWidthFor(commandState) > texWidth ||
                mglPdRenderTargetHeightFor(commandState) > texHeight) {
                if (kMglPdVerboseFrameLoopLogs) {
                    fprintf(stderr,
                            "MGL INFO: Normalizing renderTarget size from %lux%lu to %lux%lu\n",
                            (unsigned long)mglPdRenderTargetWidthFor(commandState),
                            (unsigned long)mglPdRenderTargetHeightFor(commandState),
                            (unsigned long)texWidth, (unsigned long)texHeight);
                }
                mglPdSetPersistentDimensions(commandState, texWidth, texHeight);
            }
        }
    }
    return true;
}

/* === pipeline descriptor state (log 172) =============================== */

/* pixel_utils.c defines this one; the only declaration lives in the
 * Objective-C MGLRenderer+RenderPass_Private.h. */
extern uint32_t mtlPixelFormatForGLTex(Texture *gl_tex);

/* The discard-stub factory stays in Objective-C (dispatch_once + blocks), so
 * the .m exposes this C-callable bridge. */
extern void *mglRenderPassDiscardStubFragmentFunction(uint32_t valueClass);

static void *mglPdColorTextureFor(const MGLCommandState *commandState,
                                 size_t colorIndex)
{
    return mglPdAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorIndex);
}

/* Twin of the .m static: MGLStubFSValueClass is the enum
 * mglRenderMetalPixelFormatValueClass() returns. */
static uint32_t mglPdStubFSValueClass(uint32_t fmt)
{
    return mglRenderMetalPixelFormatValueClass(fmt);
}

/* Twins of the .m geometry-passthrough helpers. */
static uint32_t mglPdGeometryPassthroughLayerStride(GLMContext ctx)
{
    if (!ctx || !ctx->active_state || !ctx->active_state->framebuffer) {
        return 1u;
    }
    Framebuffer *fbo = ctx->active_state->framebuffer;
    for (GLuint i = 0u; i < MAX_COLOR_ATTACHMENTS; i++) {
        const FBOAttachment *attachment = &fbo->color_attachments[i];
        const uint32_t stride = mglRenderMSAAArrayLayerStride(
            attachment->layered ? 1 : 0, (uint32_t)attachment->textarget);
        if (stride > 1u) {
            return stride;
        }
    }
    const uint32_t depthStride = mglRenderMSAAArrayLayerStride(
        fbo->depth.layered ? 1 : 0, (uint32_t)fbo->depth.textarget);
    if (depthStride > 1u) {
        return depthStride;
    }
    return mglRenderMSAAArrayLayerStride(fbo->stencil.layered ? 1 : 0,
                                         (uint32_t)fbo->stencil.textarget);
}

static uint64_t mglPdGeometryPassthroughCacheKey(const Program *program,
                                                 uint32_t layerStride)
{
    uint64_t hash = 1469598103934665603ull;
    hash = mglHashStepU64(
        hash, program ? program->pipeline_cache_instance_id : 0u);
    hash = mglHashStepU64(hash,
                          program ? program->pipeline_cache_generation : 0u);
    return mglHashStepU64(hash, layerStride);
}

/* MGLRenderer+RenderPass_Private.h: `static const BOOL
 * kMGLVerbosePipelineLogs = NO;` - kept so the branches stay verbatim. */
static const int kMglPdVerbosePipelineLogs = 0;

/* -generatePipelineDescriptorState:vertexFunction:fragmentFunction:. */
int mglRenderPassGeneratePipelineDescriptorState(
    void *renderer, void *state, MGLRenderPassPipelineFunctions *functions_out)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLTessellationState *tess = areas.tessellation;
    MGLGeometryState *geom = areas.geometry;
    MGLCommandState *commandState = areas.command;
    GLMState *glState = mglPdState(&areas);
    MGLRenderPipelineDescriptorState *desc =
        (MGLRenderPipelineDescriptorState *)state;

    if (!ctx) {
        fprintf(stderr, "MGL PIPELINE DESC fail: context is NULL\n");
        return 0;
    }
    if (!desc || !functions_out) {
        fprintf(stderr, "MGL PIPELINE DESC fail: bad out args\n");
        return 0;
    }
    functions_out->vertex_function = NULL;
    functions_out->fragment_function = NULL;

    const int nativeTES = tess->nativeTESActive;
    const int tessVertexCapture = tess->tessVertexCaptureActive;
    const int cullDistanceCapture = tess->cullDistanceCaptureActive;
    const int geometryExpansion = geom->expansionActive;
    const int tessCompute = tess->tessComputeActive;
    const int tessVertexRender = tessCompute && tess->tessVertexRenderActive;
    const int tessVertex = tessVertexRender;
    const int vertexStage = nativeTES ? _TESS_EVALUATION_SHADER : _VERTEX_SHADER;
    Program *vertexProgram = nativeTES ? tess->nativeTESProgram
        : tessVertex ? tess->tessComputeProgram
        : mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    Program *fragmentProgram = (tessVertexCapture || cullDistanceCapture)
        ? NULL : mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    const GLuint renderProgramKey = mglCurrentRenderProgramKey(ctx);
    const GLuint vertexProgramName = vertexProgram ? vertexProgram->name : 0u;
    const GLuint fragmentProgramName =
        fragmentProgram ? fragmentProgram->name : 0u;
    const int rasterizerDiscard =
        (tessVertexCapture || cullDistanceCapture ||
         glState->caps.rasterizer_discard) ? 1 : 0;

    if (!vertexProgram || (!fragmentProgram && !rasterizerDiscard)) {
        fprintf(stderr,
                "MGL PIPELINE DESC fail: missing stage program key=%u vs=%p fs=%p current=%u pipeline=%u\n",
                (unsigned)renderProgramKey, (void *)vertexProgram,
                (void *)fragmentProgram, (unsigned)glState->program_name,
                (unsigned)glState->var.program_pipeline_binding);
        return 0;
    }

    if (kMglPdVerbosePipelineLogs) {
        fprintf(stderr,
                "MGL PIPELINE DESC begin key=%u vsProgram=%u fsProgram=%u\n",
                (unsigned)renderProgramKey, (unsigned)vertexProgramName,
                (unsigned)fragmentProgramName);
    }

    if (!mglRendererBindMTLProgramPort(renderer, vertexProgram)) {
        fprintf(stderr,
                "MGL PIPELINE DESC fail: bindMTLProgram failed for VS program=%u\n",
                (unsigned)vertexProgramName);
        return 0;
    }
    if (fragmentProgram && fragmentProgram != vertexProgram &&
        !mglRendererBindMTLProgramPort(renderer, fragmentProgram)) {
        fprintf(stderr,
                "MGL PIPELINE DESC fail: bindMTLProgram failed for FS program=%u\n",
                (unsigned)fragmentProgramName);
        return 0;
    }

    Shader *vertex_shader = vertexProgram->shader_slots[vertexStage];
    Shader *fragment_shader =
        fragmentProgram ? fragmentProgram->shader_slots[_FRAGMENT_SHADER] : NULL;
    if (!vertex_shader || (!fragment_shader && !rasterizerDiscard)) {
        fprintf(stderr,
                "MGL PIPELINE DESC fail: missing shaders key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)\n",
                (unsigned)renderProgramKey, (unsigned)vertexProgramName,
                (unsigned)fragmentProgramName, (void *)vertex_shader,
                (void *)fragment_shader);
        return 0;
    }

    void *geometryPassthroughFunction = NULL;
    if (geometryExpansion && geom->program) {
        const uint32_t layerStride =
            mglPdGeometryPassthroughLayerStride(ctx);
        const uint64_t passthroughKey =
            mglPdGeometryPassthroughCacheKey(geom->program, layerStride);
        (void)mglRendererBackendGetPassthroughFunction(
            areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY,
            passthroughKey, &geometryPassthroughFunction);
    }
    void *tessPassthroughFunction = NULL;
    if (tessCompute && tess->tessComputeProgram) {
        /* A TES-vertex program is its own vertex function (the expanded
         * stream rasterizes directly); only the record-passthrough of the
         * compute expansion uses the generated slot-28 reader. */
        if (tess->tessVertexRenderActive) {
            tessPassthroughFunction =
                tess->tessComputeProgram->modules[_TESS_EVALUATION_SHADER]
                    .mtl_function;
        } else {
            (void)mglRendererBackendGetPassthroughFunction(
                areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION,
                tess->tessComputeProgram->pipeline_cache_instance_id,
                &tessPassthroughFunction);
        }
    }
    void *vertexFunctionPtr = geometryExpansion
        ? geometryPassthroughFunction
        : tessCompute
        ? tessPassthroughFunction
        : cullDistanceCapture
        ? vertexProgram->modules[_VERTEX_SHADER].mtl_cull_capture_function
        : tessVertexCapture
        ? vertexProgram->modules[_VERTEX_SHADER].mtl_tess_capture_function
        : vertexProgram->modules[vertexStage].mtl_function;
    void *vertexFunction = vertexFunctionPtr;

    void *fragmentFunction =
        fragmentProgram
            ? fragmentProgram->modules[_FRAGMENT_SHADER].mtl_function
            : NULL;
    /* Tess/cull VS capture also needs a real FS: AGX drops vertex device-
     * buffer stores when Metal rasterization is off (same as
     * GL_RASTERIZER_DISCARD). Stub FS + cleared color masks below. */
    if (!fragmentFunction && rasterizerDiscard) {
        /* Metal validates the stub FS output against color attachment 0's
         * format: float4 stubs are rejected by integer-format targets.
         * Resolve the format early (read-only; the FBO walk below re-binds
         * the same textures) and pick the matching zero-return variant. */
        uint32_t stubColor0 = mglRenderInvalidPixelFormat();
        if (glState->framebuffer) {
            Framebuffer *stubFbo = glState->framebuffer;
            for (int i = 0; i < glState->max_color_attachments; i++) {
                if (!stubFbo->color_attachments[i].texture) {
                    if ((stubFbo->color_attachment_bitfield >> (i + 1)) == 0) {
                        break;
                    }
                    continue;
                }
                Texture *stubTex = mglRendererAttachmentTextureFor(
                    ctx, &stubFbo->color_attachments[i]);
                if (stubTex && stubTex->mtl_data) {
                    stubColor0 = mtlPixelFormatForGLTex(stubTex);
                    if (!mglRenderPixelFormatIsInvalid(stubColor0)) {
                        break;
                    }
                }
            }
        } else if (commandState && mglPdColorTextureFor(commandState, 0)) {
            stubColor0 = mglPdTextureInfo(
                mglPdColorTextureFor(commandState, 0)).pixel_format;
        } else if (mglRendererDrawableTexturePort(renderer)) {
            stubColor0 = mglPdTextureInfo(
                mglRendererDrawableTexturePort(renderer)).pixel_format;
        } else {
            stubColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        fragmentFunction = mglRenderPassDiscardStubFragmentFunction(
            mglPdStubFSValueClass(stubColor0));
    }
    if (kMglPdVerbosePipelineLogs) {
        fprintf(stderr, "MGL PIPELINE DESC vs=%p fs=%p\n", vertexFunction,
                fragmentFunction);
    }
    if (!mglRenderPipelineFunctionsReady(vertexFunction ? 1 : 0,
                                         fragmentFunction ? 1 : 0,
                                         rasterizerDiscard)) {
        fprintf(stderr,
                "MGL PIPELINE DESC fail: missing MTLFunction key=%u vsProgram=%u fsProgram=%u (vs=%p fs=%p)\n",
                (unsigned)renderProgramKey, (unsigned)vertexProgramName,
                (unsigned)fragmentProgramName, vertexFunction,
                fragmentFunction);
        return 0;
    }

    memset(desc, 0, sizeof(*desc));
    desc->vertex_program_instance = vertexProgram->pipeline_cache_instance_id;
    desc->vertex_program_generation = vertexProgram->pipeline_cache_generation;
    desc->fragment_program_instance =
        fragmentProgram ? fragmentProgram->pipeline_cache_instance_id : 0u;
    desc->fragment_program_generation =
        fragmentProgram ? fragmentProgram->pipeline_cache_generation : 0u;
    desc->color_count = MAX_COLOR_ATTACHMENTS;
    desc->rasterization_enabled = 1;

    desc->max_tessellation_factor = mglRenderMaxTessellationFactor();

    {
        /* Metal requires the pipeline's primitive topology class to match
         * the drawn primitive type: an unspecified-class pipeline silently
         * drops point draws.  A compute-routed geometry expansion always
         * gets its output class explicitly.  For ordinary draws only the
         * point case is forced: leaving triangles on the historical
         * unspecified value keeps programs that write gl_PointSize while
         * drawing triangles linkable (Metal rejects a triangle-class
         * pipeline whose vertex function writes point size).
         *
         * Exception: VS writing [[render_target_array_index]] (gl_Layer)
         * requires an explicit topology.  Real AGX often tolerates
         * Unspecified; Apple Paravirtual rejects with CompilerError. */
        const int needsExplicitTopology = mglRenderNeedsExplicitTopology(
            geometryExpansion ? 1 : 0, areas.core->lastDrawPrimitiveMode,
            vertexProgram ? mglRenderVSWritesLayer(vertexProgram) : 0);
        if (needsExplicitTopology) {
            /* A geometry expansion emits the geometry shader's output
             * primitive type, not the GL draw mode: a layout(points) geometry
             * shader drawing GL_PATCHES still rasterizes points.  Classifying
             * by the draw mode gave it MTLPrimitiveTopologyClassTriangle, and
             * Metal refuses to build a pipeline whose vertex function writes
             * [[point_size]] against a triangle class ("Vertex shader writes
             * point size but inputPrimitiveTopology is
             * MTLPrimitiveTopologyClassTriangle").  The geometry pass then
             * produced nothing and the draw rendered all zeroes
             * (tessellation_shader_point_mode.point_rendering), while the
             * sibling points_verification case only passed because its
             * geometry shader happens to emit triangle strips. */
            GLenum topologyMode = (GLenum)areas.core->lastDrawPrimitiveMode;
            if (geometryExpansion && geom->program) {
                switch (geom->program->geometry_output_type) {
                case GL_POINTS:
                    topologyMode = GL_POINTS;
                    break;
                case GL_LINE_STRIP:
                    topologyMode = GL_LINES;
                    break;
                default:
                    topologyMode = GL_TRIANGLES;
                    break;
                }
            }
            desc->input_primitive_topology =
                mglRenderPrimitiveTopologyClass((uint32_t)topologyMode);
        }
        /* isolines / point_mode rasterize the expanded point / line stream,
         * either with the TES acting as its own vertex function (TES-vertex)
         * or through the generated record-passthrough vertex function (TES
         * compute).  Both write gl_PointSize under point_mode, and Metal
         * refuses to link such a vertex function against a triangle topology
         * class.  Force the class from the tessellation raster mode whenever
         * a TES compute draw is being submitted. */
        if (tessCompute && tess->tessComputeProgram) {
            desc->input_primitive_topology =
                mglRenderPrimitiveTopologyClass(
                    (uint32_t)mglTessRasterGLMode(tess->tessComputeProgram));
        }
    }

    if (nativeTES) {
        desc->tessellation_partition_mode =
            mglRenderTessPartitionMode(vertexProgram->tess_gen_spacing);
        desc->max_tessellation_factor = mglRenderMaxTessellationFactor();
        desc->tessellation_factor_scale_enabled = 0;
        desc->tessellation_factor_format =
            (uint32_t)MGLTessellationFactorFormatHalf;

        desc->tessellation_control_point_index_type =
            mglRenderTessControlPointIndexType(tess->tessIndexedDraw ? 1 : 0);
        desc->tessellation_factor_step_function =
            (uint32_t)MGLTessellationFactorStepFunctionPerPatch;
        desc->tessellation_output_winding_order =
            mglRenderTessOutputWinding(vertexProgram->tess_gen_vertex_order);
    }

    /* AGX drops vertex texture/SSBO stores when Metal rasterization is
     * disabled - including tess/cull VS capture draws. Keep rasterization
     * on (real FS or discard stub above); color write masks cleared below. */
    desc->rasterization_enabled = mglRenderRasterizationEnabled(
        rasterizerDiscard, fragmentFunction ? 1 : 0);

    /* Attachment formats: FBO attachment -> pass/drawable/context fallback. */
    if (glState->framebuffer) {
        Framebuffer *fbo = glState->framebuffer;

        for (int i = 0; i < glState->max_color_attachments; i++) {
            if (fbo->color_attachments[i].texture) {
                Texture *tex = mglRendererAttachmentTextureFor(
                    ctx, &fbo->color_attachments[i]);
                if (tex && !mglRendererBindMTLTexture(renderer, tex)) {
                    fprintf(stderr,
                            "MGL PIPELINE DESC fail: bindMTLTexture failed for color attachment %d tex=%u\n",
                            i, tex->name);
                    return 0;
                }
                if (tex && tex->mtl_data) {
                    desc->color_format[i] =
                        (uint32_t)mtlPixelFormatForGLTex(tex);
                } else {
                    desc->color_format[i] = mglRenderInvalidPixelFormat();
                }
            }

            if (mglRenderColorAttachmentBitfieldDone(
                    (uint32_t)fbo->color_attachment_bitfield, i)) {
                break;
            }
        }

        if (fbo->depth.texture) {
            Texture *tex = mglRendererAttachmentTextureFor(ctx, &fbo->depth);
            if (tex && !mglRendererBindMTLTexture(renderer, tex)) {
                fprintf(stderr,
                        "MGL PIPELINE DESC fail: bindMTLTexture failed for depth tex=%u\n",
                        tex->name);
                return 0;
            }
            if (tex && tex->mtl_data) {
                const uint32_t rawDepth =
                    (uint32_t)mtlPixelFormatForGLTex(tex);
                const uint32_t depthFormat =
                    mglRenderDepthFormatOrFallback(rawDepth);
                if (mglRenderPixelFormatIsInvalid(rawDepth)) {
                    fprintf(stderr,
                            "MGL ERROR: Invalid depth texture format, falling back to Depth32Float\n");
                }
                desc->depth_format = depthFormat;
            } else {
                desc->depth_format = mglRenderInvalidPixelFormat();
            }
        }

        if (fbo->stencil.texture) {
            Texture *tex = mglRendererAttachmentTextureFor(ctx, &fbo->stencil);
            if (tex && !mglRendererBindMTLTexture(renderer, tex)) {
                fprintf(stderr,
                        "MGL PIPELINE DESC fail: bindMTLTexture failed for stencil tex=%u\n",
                        tex->name);
                return 0;
            }
            if (tex && tex->mtl_data) {
                const uint32_t rawStencil =
                    (uint32_t)mtlPixelFormatForGLTex(tex);
                const uint32_t stencilFormat =
                    mglRenderStencilFormatOrFallback(rawStencil);
                if (mglRenderPixelFormatIsInvalid(rawStencil)) {
                    fprintf(stderr,
                            "MGL ERROR: Invalid stencil texture format, falling back to Stencil8\n");
                }
                desc->stencil_format = stencilFormat;
            } else {
                desc->stencil_format = mglRenderInvalidPixelFormat();
            }
        }
    } else {
        uint32_t preferredColor0 = mglRenderInvalidPixelFormat();
        if (commandState && mglPdColorTextureFor(commandState, 0)) {
            preferredColor0 =
                mglPdTextureInfo(mglPdColorTextureFor(commandState, 0))
                    .pixel_format;
        } else if (mglRendererDrawableTexturePort(renderer)) {
            preferredColor0 = mglPdTextureInfo(
                mglRendererDrawableTexturePort(renderer)).pixel_format;
        } else {
            preferredColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        desc->color_format[0] = preferredColor0;

        if (ctx->depth_format.format) {
            desc->depth_format = mglRenderDepthFormatOrFallback(
                ctx->depth_format.mtl_pixel_format);
        }

        if (ctx->stencil_format.format) {
            desc->stencil_format = mglRenderDefaultFBOStencilFormat(
                ctx->stencil_format.mtl_pixel_format);
        }
    }

    /* Derive pipeline attachment formats from the configured C++ pass. */
    const int hasConfiguredRenderPass =
        commandState->renderPassStateOwner != NULL;
    if (hasConfiguredRenderPass) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            void *rpColor = mglPdColorTextureFor(commandState, (size_t)i);
            if (rpColor) {
                desc->color_format[i] =
                    mglPdTextureInfo(rpColor).pixel_format;
            }
        }

        void *rpDepth = mglPdDepthTextureFor(commandState);
        void *rpStencil = mglPdStencilTextureFor(commandState);
        desc->depth_format = mglRenderAttachmentFormatOrInvalid(
            rpDepth ? 1 : 0,
            rpDepth ? (uint32_t)mglPdTextureInfo(rpDepth).pixel_format : 0u);
        desc->stencil_format = mglRenderAttachmentFormatOrInvalid(
            rpStencil ? 1 : 0,
            rpStencil ? (uint32_t)mglPdTextureInfo(rpStencil).pixel_format
                      : 0u);
    }

    const int color0IsIntentionallyDisabled =
        mglRenderColor0IntentionallyDisabled(
            glState->framebuffer ? 1 : 0,
            (uint32_t)mglMetalDrawBufferAt(ctx, 0u)) != 0;

    if (!color0IsIntentionallyDisabled &&
        mglRenderColorFormatNeedsFallback(desc->color_format[0])) {
        uint32_t fallbackColor0 = mglRenderInvalidPixelFormat();
        if (commandState && mglPdColorTextureFor(commandState, 0)) {
            fallbackColor0 =
                mglPdTextureInfo(mglPdColorTextureFor(commandState, 0))
                    .pixel_format;
        } else if (mglRendererDrawableTexturePort(renderer)) {
            fallbackColor0 = mglPdTextureInfo(
                mglRendererDrawableTexturePort(renderer)).pixel_format;
        } else {
            fallbackColor0 = ctx->pixel_format.mtl_pixel_format;
        }
        fallbackColor0 = mglRenderColorFormatOrBGRA(fallbackColor0);
        if (kMglPdVerbosePipelineLogs) {
            fprintf(stderr,
                    "MGL PIPELINE DESC missing color pixel format, fallback pixelFormat=%lu\n",
                    (unsigned long)fallbackColor0);
        }
        desc->color_format[0] = fallbackColor0;
    }

    /* Resolve the pipeline sample count from the C++ render-pass state. */
    uint64_t resolvedSampleCount = 1;
    void *rpColor0 = mglPdColorTextureFor(commandState, 0);
    void *rpDepth = mglPdDepthTextureFor(commandState);
    void *rpStencil = mglPdStencilTextureFor(commandState);
    if (rpColor0 && mglPdTextureInfo(rpColor0).sample_count > 0) {
        resolvedSampleCount = mglPdTextureInfo(rpColor0).sample_count;
    } else if (rpDepth && mglPdTextureInfo(rpDepth).sample_count > 0) {
        resolvedSampleCount = mglPdTextureInfo(rpDepth).sample_count;
    } else if (rpStencil && mglPdTextureInfo(rpStencil).sample_count > 0) {
        resolvedSampleCount = mglPdTextureInfo(rpStencil).sample_count;
    }
    if (resolvedSampleCount == 0) {
        resolvedSampleCount = 1;
    }
    desc->raster_sample_count = (uint32_t)resolvedSampleCount;

    {
        uint32_t packedFormat = 0u;
        if (mglRenderPassUnifyPackedDS(desc->depth_format,
                                       desc->stencil_format, &packedFormat)) {
            desc->depth_format = packedFormat;
            desc->stencil_format = packedFormat;
        }
    }

    desc->alpha_to_coverage_enabled =
        glState->caps.sample_alpha_to_coverage ? 1 : 0;
    desc->alpha_to_one_enabled =
        glState->caps.sample_alpha_to_one ? 1 : 0;

    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        if (mglRenderSkipInvalidColorAttachment(desc->color_format[i])) {
            continue;
        }
        if (mglRenderDrawBufferIsNone(
                (uint32_t)mglMetalDrawBufferAt(ctx, (GLuint)i))) {
            desc->color_write_mask[i] = 0u;
            continue;
        }
        MGLRenderPipelineBlendState blend = {0};
        if (!areas.pipeline_cache_blend_state ||
            !areas.pipeline_cache_blend_state(areas.pipeline_cache_object,
                                              (uint32_t)i, &blend)) {
            fprintf(stderr,
                    "MGL PIPELINE DESC fail: blend state unavailable for attachment %d\n",
                    i);
            return 0;
        }
        desc->color_write_mask[i] = blend.color_write_mask;
        desc->blending_enabled_mask |= mglRenderBlendingEnabledMaskBit(
            glState->caps.blendi[i] ? 1 : 0, i);
        desc->source_rgb_blend_factor[i] = blend.source_rgb_factor;
        desc->destination_rgb_blend_factor[i] = blend.destination_rgb_factor;
        desc->source_alpha_blend_factor[i] = blend.source_alpha_factor;
        desc->destination_alpha_blend_factor[i] = blend.destination_alpha_factor;
        desc->rgb_blend_operation[i] = blend.rgb_operation;
        desc->alpha_blend_operation[i] = blend.alpha_operation;
    }

    if (mglRenderClearColorWriteMasks(glState->caps.rasterizer_discard ? 1 : 0,
                                      tessVertexCapture ? 1 : 0,
                                      cullDistanceCapture ? 1 : 0)) {
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            desc->color_write_mask[i] = 0u;
        }
    }

    if (mglRenderNeedsVertexDescriptor(geometryExpansion ? 1 : 0,
                                       tessCompute ? 1 : 0)) {
        if (!mglRendererGenerateVertexDescriptorState(renderer, desc)) {
            return 0;
        }
    }

    if (kMglPdVerbosePipelineLogs) {
        uint32_t activeColorAttachmentCount = 0;
        for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
            if (desc->color_format[i] != mglRenderInvalidPixelFormat() &&
                desc->color_format[i] != 0u) {
                activeColorAttachmentCount++;
            }
        }
        fprintf(stderr,
                "MGL PIPELINE DESC colorAttachmentCount=%u depthFormat=%u stencilFormat=%u sampleCount=%u\n",
                (unsigned)activeColorAttachmentCount,
                (unsigned)desc->depth_format, (unsigned)desc->stencil_format,
                (unsigned)desc->raster_sample_count);
        fprintf(stderr, "MGL PIPELINE DESC renderTarget[0]=%u\n",
                (unsigned)desc->color_format[0]);
    }

    functions_out->vertex_function = vertexFunction;
    functions_out->fragment_function = fragmentFunction;
    return 1;
}

/* === new command buffer (log 174) ====================================== */

/* MGLRenderer+RenderPass_Private.h: `static const BOOL
 * kMGLDisableSharedEventSync = YES;` - kept so the branch stays verbatim. */
static const int kMglPdDisableSharedEventSync = 1;

/* The two @try blocks of -newCommandBufferLocked travel through the shell's
 * guarded-call twin.  `result` tells the three outcomes apart: 1 = the body
 * fell through, -1 = the body failed on its own (it applied the side effects
 * the original wrote inline) and 0 = an exception was thrown (the original's
 * @catch). */
typedef struct MglPdNewCommandBufferCtx_t {
    void *renderer;
    MGLRendererStateAreas *areas;
    int result;
} MglPdNewCommandBufferCtx;

static int mglPdNewCommandBufferTryBody(void *renderer, void *rawCtx)
{
    MglPdNewCommandBufferCtx *ctx = (MglPdNewCommandBufferCtx *)rawCtx;
    MGLRendererStateAreas *areas = ctx->areas;
    MGLRenderPassManager *manager = areas->render_pass_manager;

    /* AGX DRIVER COMPATIBILITY: Validate command queue health before creating
     * buffer */
    if (!mglRendererBackendGetCommandQueue(areas->backend)) {
        fprintf(stderr, "MGL AGX ERROR: Command queue is NULL - recreating\n");
        mglRendererResetMetalState(renderer);
        if (!mglRendererBackendGetCommandQueue(areas->backend)) {
            fprintf(stderr, "MGL AGX CRITICAL: Cannot recreate command queue\n");
            ctx->result = -1;
            return 0;
        }
    }

    /* CRITICAL FIX: Validate the command queue before dereferencing it to
     * prevent NULL pointer crashes */
    if (!mglRendererBackendGetCommandQueue(areas->backend)) {
        fprintf(stderr,
                "MGL AGX CRITICAL: _commandQueue is NULL - cannot create command buffer\n");
        mglRendererRecordGPUError(renderer);
        ctx->result = -1;
        return 0;
    }

    if (!mglPassManagerInstallNewCommandBufferFromQueue(
            manager, mglRendererBackendGetCommandQueue(areas->backend))) {
        fprintf(stderr,
                "MGL AGX ERROR: Failed to create Metal command buffer - command queue may be in error state\n");
        mglRendererRecordGPUError(renderer);
        /* Force command queue recreation */
        mglRendererResetMetalState(renderer);
        ctx->result = -1;
        return 0;
    }

    areas->batching->currentCommandBufferHasWork = 0;

    /* AGX Driver Validation: Check if the command buffer is immediately
     * invalid */
    MGLRenderCommandBufferState initialState = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            areas->command->currentCommandBufferOwner, &initialState)) {
        fprintf(stderr,
                "MGL AGX CRITICAL: New command buffer owner has no current buffer\n");
        mglRendererRecordGPUError(renderer);
        ctx->result = -1;
        return 0;
    }
    if (initialState.has_error) {
        fprintf(stderr,
                "MGL AGX WARNING: New command buffer has immediate error: %s\n",
                mglRenderCommandBufferErrorDescription(&initialState));
        mglRendererRecordGPUError(renderer);
        /* Don't return false immediately - AGX sometimes creates error-state
         * buffers that recover */
    }

    /* AGX DRIVER COMPATIBILITY: Enhanced validation to prevent rejections */
    if (initialState.status == MGLCommandBufferStatusError) {
        fprintf(stderr,
                "MGL AGX CRITICAL: Command buffer immediately in error state\n");
        mglRendererRecordGPUError(renderer);
        mglPassManagerDiscardCurrentCommandBuffer(manager);
        mglRendererResetMetalState(renderer); /* Force full reset */
        ctx->result = -1;
        return 0;
    }

    /* Additional AGX validation: check for buffer properties that cause
     * rejections */
    memset(&initialState, 0, sizeof(initialState));
    (void)mglRenderCommandBufferOwnerHasState(
        areas->command->currentCommandBufferOwner, &initialState);
    if (initialState.has_error) {
        fprintf(stderr,
                "MGL AGX WARNING: Command buffer has immediate error: %s\n",
                mglRenderCommandBufferErrorDescription(&initialState));
        mglRendererRecordGPUError(renderer);
        mglPassManagerDiscardCurrentCommandBuffer(manager);
        mglRendererResetMetalState(renderer);
        ctx->result = -1;
        return 0;
    }

    /* Validate command queue health */
    if (!mglRendererBackendGetCommandQueue(areas->backend)) {
        fprintf(stderr, "MGL AGX CRITICAL: Command queue became NULL\n");
        mglRendererResetMetalState(renderer);
        ctx->result = -1;
        return 0;
    }

    if (kMglPdVerboseFrameLoopLogs) {
        fprintf(stderr,
                "MGL INFO: Successfully created new Metal command buffer (AGX validated)\n");
    }
    ctx->result = 1;
    return 1;
}

typedef struct MglPdEventWaitCtx_t {
    MGLRendererStateAreas *areas;
    void *event;
    uint32_t sync_name;
    int result;
} MglPdEventWaitCtx;

static int mglPdEventWaitTryBody(void *renderer, void *rawCtx)
{
    MglPdEventWaitCtx *ctx = (MglPdEventWaitCtx *)rawCtx;
    (void)renderer;
    fprintf(stderr, "MGL INFO: Encoding safe event wait: event=%p, syncName=%u\n",
            ctx->event, ctx->sync_name);
    if (mglRenderEncodeWaitForEventForCommandBufferOwner(
            ctx->areas->command->currentCommandBufferOwner, ctx->event,
            ctx->sync_name) != 0) {
        fprintf(stderr,
                "MGL ERROR: Event wait owner facade rejected the request\n");
        ctx->result = -1;
        return 0;
    }
    fprintf(stderr,
            "MGL SUCCESS: Event wait encoded successfully on fresh command buffer\n");
    ctx->result = 1;
    return 1;
}

/* -newCommandBufferLocked (the retired mglRendererNewCommandBufferLockedPort). */
int mglRenderPassNewCommandBufferLocked(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLRenderPassManager *manager = areas.render_pass_manager;
    MGLCommandState *commandState = areas.command;

    /* CRITICAL FIX: Proper encoder cleanup BEFORE creating new command buffer
     * Metal API requires ending encoders before creating new command buffers
     *
     * STEP 0: End any existing render encoder to prevent
     * MTLReleaseAssertionFailure */
    if (mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1) {
        if (kMglPdVerboseFrameLoopLogs) {
            fprintf(stderr,
                    "MGL INFO: Ending existing render encoder before creating new command buffer\n");
        }
        mglRendererEndRenderEncodingLocked(renderer);
    }

    /* STEP 1: Clean up sync tracking list safely.
     * IMPORTANT: Do NOT dereference Sync* entries here. Sync objects are owned
     * by GL sync lifecycle and may already be deleted by glDeleteSync on other
     * paths.  Both this read/clear path and the backend sync append path run on
     * the GL calling thread, so no lock is needed. */
    mglPassManagerClearCurrentCommandBufferSyncListEntries(manager);

    /* A successful C++ submit transaction rotates the owner to the next current
     * command buffer before returning. Consume that exact buffer so the adapter
     * does not immediately allocate and release another one.  Unmarked current
     * buffers still follow the ordinary fresh-rotate path. */
    if (mglPassManagerConsumeTransactionCreatedCurrentCommandBuffer(manager)) {
        areas.batching->currentCommandBufferHasWork = 0;
        return 1;
    }

    /* CRITICAL SAFETY: Validate command queue before creating buffer */
    if (!mglRendererBackendGetCommandQueue(areas.backend)) {
        fprintf(stderr,
                "MGL ERROR: Cannot create command buffer - command queue is NULL\n");
        mglPassManagerDiscardCurrentCommandBuffer(manager);
        return 0;
    }

    /* STEP 1: Create fresh command buffer FIRST with comprehensive AGX driver
     * validation */
    MglPdNewCommandBufferCtx ctx = {renderer, &areas, 0};
    if (!mglPlatformShellGuardedCallCtx(renderer, "new command buffer",
                                        mglPdNewCommandBufferTryBody, &ctx,
                                        NULL)) {
        if (ctx.result != -1) {
            /* @catch (NSException *exception) */
            mglRendererRecordGPUError(renderer);
            mglPassManagerDiscardCurrentCommandBuffer(manager);
            /* AGX DRIVER COMPATIBILITY: Force reset on exception to clear
             * driver state */
            mglRendererResetMetalState(renderer);
        }
        return 0;
    }

    /* STEP 2: Now handle pending event waits on the FRESH command buffer. */
    uint32_t cachedSyncName = 0;
    void *cachedEvent =
        mglPassManagerDetachPendingEventWithSyncName(manager, &cachedSyncName);
    if (cachedEvent) {
        if (!cachedSyncName) {
            fprintf(stderr,
                    "MGL WARNING: dropping pending shared-event wait with no sync name\n");
            return 1;
        }

        if (kMglPdDisableSharedEventSync) {
            fprintf(stderr,
                    "MGL INFO: Shared event wait disabled (debug no-op), skipping wait encode event=%p syncName=%u\n",
                    cachedEvent, cachedSyncName);
            return 1;
        }

        /* SAFELY ENCODE: Event wait functionality on the new command buffer */
        if (kMglPdVerboseFrameLoopLogs) {
            fprintf(stderr,
                    "MGL INFO: Encoding event wait on fresh command buffer\n");
        }

        /* Validate event pointer looks like a valid object address */
        const uintptr_t eventPtr = (uintptr_t)cachedEvent;
        if (eventPtr == 0x10 || eventPtr == 0x30 || eventPtr == 0x1000) {
            fprintf(stderr,
                    "MGL CRITICAL ERROR: Known corrupted event pointer pattern detected: 0x%lx\n",
                    (unsigned long)eventPtr);
            fprintf(stderr,
                    "MGL CRITICAL ERROR: Skipping event wait to prevent crash\n");
            return 0;
        }

        if (eventPtr < 0x1000 || (eventPtr & 0x7) != 0) {
            fprintf(stderr, "MGL ERROR: Suspicious event pointer value: %p\n",
                    cachedEvent);
            fprintf(stderr, "MGL INFO: Skipping event wait for safety\n");
            return 0;
        }

        /* ADDITIONAL SAFETY: Validate command buffer is still valid before
         * encoding */
        if (mglRenderCommandBufferOwnerHasCurrent(
                areas.command->currentCommandBufferOwner) != 1) {
            fprintf(stderr,
                    "MGL ERROR: Command buffer became NULL before event wait encoding\n");
            return 0;
        }

        MglPdEventWaitCtx evt = {&areas, cachedEvent, cachedSyncName, 0};
        (void)mglPlatformShellGuardedCallCtx(renderer, "shared event wait",
                                             mglPdEventWaitTryBody, &evt, NULL);
        if (evt.result == -1) {
            return 0;
        }
        /* @catch: continue without event wait - the system stays stable. */
    }

    return 1;
}

/* === render encoder creation (log 175) ================================= */

/* Twins of the .m clear-value statics. */
static bool mglPdClearValuesFor(const MGLCommandState *commandState,
                                uint32_t attachmentKind, size_t colorIndex,
                                double *clearColorOut, double *clearDepthOut,
                                uint32_t *clearStencilOut)
{
    MGLRenderPassState state = {0};
    if (!mglRenderPassGetPersistentState(commandState, &state)) return false;
    return mglRenderPassPlanClearValues(&state, attachmentKind,
                                        (uint32_t)colorIndex, clearColorOut,
                                        clearDepthOut, clearStencilOut) != 0;
}

static double mglPdClearDepthFor(const MGLCommandState *commandState,
                                 double fallback)
{
    double depth = 0.0;
    if (mglPdClearValuesFor(commandState,
                            MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u, NULL,
                            &depth, NULL)) {
        return depth;
    }
    return fallback;
}

static uint32_t mglPdVisibilityResultTypeFor(
    const MGLCommandState *commandState)
{
    MGLRenderPassState state = {0};
    if (mglRenderPassGetPersistentState(commandState, &state)) {
        return state.visibility_result_type;
    }
    return 0u;
}

/* The @try/@catch of -createRenderEncoderLocked keeps the three-state result
 * convention of log 174: 1 = body fell through, -1 = in-body failure (its
 * side effects already applied), 0 = exception (the original @catch). */
typedef struct MglPdCreateEncoderCtx_t {
    void *renderer;
    MGLRendererStateAreas *areas;
    int result;
} MglPdCreateEncoderCtx;

static int mglPdCreateRenderEncoderTryBody(void *renderer, void *rawCtx)
{
    MglPdCreateEncoderCtx *ctx = (MglPdCreateEncoderCtx *)rawCtx;
    MGLRendererStateAreas *areas = ctx->areas;
    MGLRenderPassManager *manager = areas->render_pass_manager;
    MGLCommandState *commandState = areas->command;

    void *renderEncoder = mglPassManagerCreateRenderEncoder(manager);
    mglPassManagerInstallRenderEncoder(manager, renderEncoder);
    if (mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) != 1) {
        fprintf(stderr,
                "MGL ERROR: Failed to create render encoder - invalid render pass state or command buffer\n");
        fprintf(stderr,
                "MGL DEBUG: Command buffer owner: %p, Render pass state owner: %p\n",
                commandState->currentCommandBufferOwner,
                commandState->renderPassStateOwner);
        mglRendererRecordGPUError(renderer);
        ctx->result = -1;
        return 0;
    }
    /* Enable visibility result mode on the encoder for all draws in this pass
     * when a sample query is active. MTLVisibilityResultModeBoolean writes 1 to
     * the buffer if any samples pass per-fragment tests. */
    if (areas->query_state_owner &&
        mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1) {
        uint32_t visibilityMode = 0;
        uint64_t visibilityOffset = 0;
        if (mglRenderAcquireSampleQuerySlot(areas->query_state_owner,
                                            &visibilityMode,
                                            &visibilityOffset) == 0) {
            mglRenderSetVisibilityResultModeForRenderEncoderOwner(
                commandState->currentRenderEncoderOwner, visibilityMode,
                visibilityOffset);
        }
    }
    mglPassManagerUpdateRenderPassIdentityForContext(manager, areas->ctx);
    /* When trace is disabled, skip the full-struct memset and trace call and
     * clear only the functional flag fields. */
    if (mglTraceLogIsEnabled()) {
        mglTraceFragmentTextureTraceBindings(
            "CLEAR", "new_render_encoder", areas->fragment_trace_bindings,
            TEXTURE_UNITS, areas->ctx ? mglCurrentRenderProgramKey(areas->ctx) : 0u,
            areas->pipeline_cache->pipelineProgramName);
        memset(areas->fragment_trace_bindings, 0,
               sizeof(*areas->fragment_trace_bindings) * TEXTURE_UNITS);
    } else {
        mglClearFragmentTextureTraceFunctionalFlags(
            areas->fragment_trace_bindings, TEXTURE_UNITS);
    }
    if (kMglPdVerboseFrameLoopLogs) {
        fprintf(stderr,
                "MGL INFO: Successfully created Metal render encoder\n");
    }
    mglRendererRecordGPUSuccess(renderer);
    ctx->result = 1;
    return 1;
}

/* -createRenderEncoderLocked:. */
int mglRenderPassCreateRenderEncoderLocked(void *renderer,
                                           uint64_t renderEncoderCall)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLRenderPassManager *manager = areas.render_pass_manager;
    MGLCommandState *commandState = areas.command;

    /* CRITICAL FIX: Validate command buffer state before creating render
     * encoder */
    MGLRenderCommandBufferState commandStateInfo = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            commandState->currentCommandBufferOwner, &commandStateInfo)) {
        fprintf(stderr,
                "MGL ERROR: Cannot create render encoder - command buffer is NULL\n");
        mglRendererRecordGPUError(renderer);
        return 0;
    }

    /* Check if command buffer already has an active encoder (Metal API
     * violation) */
    if (mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1) {
        fprintf(stderr,
                "MGL WARNING: Active render encoder detected - ending it before creating new one\n");
        mglRendererEndRenderEncodingLocked(renderer);
    }

    /* Validate command buffer status. If already committed/completed, rotate to
     * a new buffer. */
    uint32_t bufferStatus = (uint32_t)commandStateInfo.status;
    if (bufferStatus >= MGLCommandBufferStatusCommitted) {
        fprintf(stderr,
                "MGL WARNING: Render encoder requested on finalized command buffer (status: %ld) - creating a fresh command buffer\n",
                (long)bufferStatus);
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: Failed to rotate command buffer before creating render encoder\n");
            mglRendererRecordGPUError(renderer);
            return 0;
        }

        if (!mglRenderCommandBufferOwnerHasState(
                commandState->currentCommandBufferOwner, &commandStateInfo)) {
            fprintf(stderr,
                    "MGL ERROR: newCommandBuffer returned without a current command buffer\n");
            mglRendererRecordGPUError(renderer);
            return 0;
        }

        bufferStatus = (uint32_t)commandStateInfo.status;
        if (bufferStatus >= MGLCommandBufferStatusCommitted) {
            fprintf(stderr,
                    "MGL ERROR: Fresh command buffer is still finalized (status: %ld)\n",
                    (long)bufferStatus);
            mglRendererRecordGPUError(renderer);
            return 0;
        }
    }

    if (kMglPdVerboseFrameLoopLogs) {
        fprintf(stderr,
                "MGL DEBUG: About to create render encoder with descriptor and command buffer\n");
    }
    {
        static uint64_t s_renderPassPreCreateLogCount = 0;
        const uint64_t hit = ++s_renderPassPreCreateLogCount;
        if (mglTraceLogIsEnabled() &&
            (hit <= 128ull || (hit % 512ull) == 0ull)) {
            mglLogRenderPassLifecycle(
                "pre-create", hit, ctx,
                commandState->currentCommandBufferOwner,
                commandState->currentRenderEncoderOwner,
                commandState->renderPassStateOwner, areas.drawable,
                commandState->renderPassFramebuffer,
                commandState->renderPassFramebufferName,
                commandState->renderPassDrawBuffer,
                commandState->renderPassDrawBufferCount);
            if (mglTraceLogIsEnabled()) {
                void *c0 = mglPdColorTextureFor(commandState, 0);
                void *depth = mglPdDepthTextureFor(commandState);
                MGLRenderPassState rpSnapshot = {0};
                (void)mglRenderPassGetPersistentState(commandState, &rpSnapshot);
                mglTraceLog(
                    "RENDERPASS_PRE_CREATE hit=%llu call=%llu program=%u fbo=%u drawBuf=0x%x readBuf=0x%x arrayLen=%lu colorLayered=%d depthLayered=%d stencilLayered=%d "
                    "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
                    "c0=%p fmt=%lu size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s clearDepth=%.6f "
                    "depthState(test=%d write=%d func=0x%x) pending(default=0x%x depth=0x%x)",
                    (unsigned long long)hit,
                    (unsigned long long)renderEncoderCall,
                    (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
                    (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
                    (unsigned)(ctx ? mglPdState(&areas)->draw_buffer : 0u),
                    (unsigned)(ctx ? mglPdState(&areas)->read_buffer : 0u),
                    (unsigned long)rpSnapshot.render_target_array_length,
                    (int)rpSnapshot.color[0].attachment.layered,
                    (int)rpSnapshot.depth.attachment.layered,
                    (int)rpSnapshot.stencil.attachment.layered,
                    (int)(ctx ? mglPdState(&areas)->viewport[0] : 0),
                    (int)(ctx ? mglPdState(&areas)->viewport[1] : 0),
                    (int)(ctx ? mglPdState(&areas)->viewport[2] : 0),
                    (int)(ctx ? mglPdState(&areas)->viewport[3] : 0),
                    (ctx && mglPdState(&areas)->caps.scissor_test) ? 1 : 0,
                    (int)(ctx ? mglPdState(&areas)->var.scissor_box[0] : 0),
                    (int)(ctx ? mglPdState(&areas)->var.scissor_box[1] : 0),
                    (int)(ctx ? mglPdState(&areas)->var.scissor_box[2] : 0),
                    (int)(ctx ? mglPdState(&areas)->var.scissor_box[3] : 0),
                    c0,
                    (unsigned long)(c0 ? mglPdTextureInfo(c0).pixel_format
                                       : mglRenderInvalidPixelFormat()),
                    (unsigned long)(c0 ? mglPdTextureInfo(c0).width : 0),
                    (unsigned long)(c0 ? mglPdTextureInfo(c0).height : 0),
                    mglLoadActionName(mglPdLoadActionFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                        0, MGLLoadActionDontCare)),
                    mglStoreActionName(mglPdStoreActionFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                        0, MGLStoreActionDontCare)),
                    depth,
                    (unsigned long)(depth ? mglPdTextureInfo(depth).pixel_format
                                          : mglRenderInvalidPixelFormat()),
                    (unsigned long)(depth ? mglPdTextureInfo(depth).width : 0),
                    (unsigned long)(depth ? mglPdTextureInfo(depth).height : 0),
                    mglLoadActionName(mglPdLoadActionFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH,
                        0, MGLLoadActionDontCare)),
                    mglStoreActionName(mglPdStoreActionFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH,
                        0, MGLStoreActionDontCare)),
                    mglPdClearDepthFor(commandState, 0.0),
                    (ctx && mglPdState(&areas)->caps.depth_test) ? 1 : 0,
                    (ctx && mglPdState(&areas)->var.depth_writemask) ? 1 : 0,
                    (unsigned)(ctx ? mglPdState(&areas)->var.depth_func : 0u),
                    (unsigned)(ctx ? mglPdState(&areas)->default_fbo_clear_bitmask
                                   : 0u),
                    (unsigned)(ctx && mglPdState(&areas)->framebuffer
                                   ? mglPdState(&areas)
                                         ->framebuffer->depth.clear_bitmask
                                   : 0u));
            }
        }
    }
    /* When a GL sample query (GL_SAMPLES_PASSED / GL_ANY_SAMPLES_PASSED) is
     * active, attach the visibility result buffer to the render-pass owner
     * state so the GPU accumulates a fresh count. */
    void *queryVisibilityBuffer = NULL;
    if (areas.query_state_owner &&
        mglRenderGetQueryVisibilityBuffer(areas.query_state_owner,
                                          &queryVisibilityBuffer) == 0 &&
        queryVisibilityBuffer) {
        const uint32_t visibilityResultType =
            mglPdVisibilityResultTypeFor(commandState);
        mglRenderSetRenderPassStateVisibility(commandState->renderPassStateOwner,
                                             queryVisibilityBuffer,
                                             visibilityResultType);
    }
    MglPdCreateEncoderCtx encCtx = {renderer, &areas, 0};
    if (!mglPlatformShellGuardedCallCtx(renderer, "render encoder creation",
                                        mglPdCreateRenderEncoderTryBody, &encCtx,
                                        NULL)) {
        if (encCtx.result != -1) {
            /* @catch (NSException *exception) */
            fprintf(stderr,
                    "MGL ERROR: Exception creating render encoder - continuing with degraded functionality\n");
            mglRendererRecordGPUError(renderer);
            mglPassManagerClearCurrentRenderEncoder(manager);
        }
        return 0;
    }
    mglRenderSetRenderEncoderOwnerLabel(
        commandState->currentRenderEncoderOwner, "GL Render Encoder");
    {
        static uint64_t s_renderPassCreatedLogCount = 0;
        const uint64_t hit = ++s_renderPassCreatedLogCount;
        if (mglTraceLogIsEnabled() &&
            (hit <= 128ull || (hit % 512ull) == 0ull)) {
            mglLogRenderPassLifecycle(
                "created", hit, ctx, commandState->currentCommandBufferOwner,
                commandState->currentRenderEncoderOwner,
                commandState->renderPassStateOwner, areas.drawable,
                commandState->renderPassFramebuffer,
                commandState->renderPassFramebufferName,
                commandState->renderPassDrawBuffer,
                commandState->renderPassDrawBufferCount);
            if (mglTraceLogIsEnabled()) {
                void *c0 = mglPdColorTextureFor(commandState, 0);
                void *depth = mglPdDepthTextureFor(commandState);
                mglTraceLog(
                    "RENDERPASS_CREATED hit=%llu call=%llu program=%u fbo=%u rpFbo=%u drawBuf=0x%x readBuf=0x%x "
                    "viewport=%d,%d,%d,%d scissor(test=%d box=%d,%d,%d,%d) "
                    "c0=%p fmt=%lu size=%lux%lu la/sa=%s/%s depth=%p fmt=%lu size=%lux%lu la/sa=%s/%s clearDepth=%.6f "
                    "depthState(test=%d write=%d func=0x%x)",
                    (unsigned long long)hit,
                    (unsigned long long)renderEncoderCall,
                    (unsigned)(ctx ? mglCurrentRenderProgramKey(ctx) : 0u),
                    (unsigned)(ctx ? mglRendererSafeFramebufferName(ctx) : 0u),
                    (unsigned)commandState->renderPassFramebufferName,
                    (unsigned)(ctx ? mglPdState(&areas)->draw_buffer : 0u),
                    (unsigned)(ctx ? mglPdState(&areas)->read_buffer : 0u),
                    (int)(ctx ? mglPdState(&areas)->viewport[0] : 0),
                    (int)(ctx ? mglPdState(&areas)->viewport[1] : 0),
                    (int)(ctx ? mglPdState(&areas)->viewport[2] : 0),
                    (int)(ctx ? mglPdState(&areas)->viewport[3] : 0),
                    (ctx && mglPdState(&areas)->caps.scissor_test) ? 1 : 0,
                    (int)(ctx ? mglPdState(&areas)->var.scissor_box[0] : 0),
                    (int)(ctx ? mglPdState(&areas)->var.scissor_box[1] : 0),
                    (int)(ctx ? mglPdState(&areas)->var.scissor_box[2] : 0),
                    (int)(ctx ? mglPdState(&areas)->var.scissor_box[3] : 0),
                    c0,
                    (unsigned long)(c0 ? mglPdTextureInfo(c0).pixel_format
                                       : mglRenderInvalidPixelFormat()),
                    (unsigned long)(c0 ? mglPdTextureInfo(c0).width : 0),
                    (unsigned long)(c0 ? mglPdTextureInfo(c0).height : 0),
                    mglLoadActionName(mglPdLoadActionFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                        0, MGLLoadActionDontCare)),
                    mglStoreActionName(mglPdStoreActionFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                        0, MGLStoreActionDontCare)),
                    depth,
                    (unsigned long)(depth ? mglPdTextureInfo(depth).pixel_format
                                          : mglRenderInvalidPixelFormat()),
                    (unsigned long)(depth ? mglPdTextureInfo(depth).width : 0),
                    (unsigned long)(depth ? mglPdTextureInfo(depth).height : 0),
                    mglLoadActionName(mglPdLoadActionFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH,
                        0, MGLLoadActionDontCare)),
                    mglStoreActionName(mglPdStoreActionFor(
                        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH,
                        0, MGLStoreActionDontCare)),
                    mglPdClearDepthFor(commandState, 0.0),
                    (ctx && mglPdState(&areas)->caps.depth_test) ? 1 : 0,
                    (ctx && mglPdState(&areas)->var.depth_writemask) ? 1 : 0,
                    (unsigned)(ctx ? mglPdState(&areas)->var.depth_func : 0u));
            }
        }
    }
    return 1;
}

/* === flush / writable command buffer (log 176) ========================= */

/* -ensureWritableCommandBufferLocked:. */
int mglRenderPassEnsureWritableCommandBufferLocked(void *renderer,
                                                   const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *commandState = areas.command;

    MGLRenderCommandBufferState bufferState = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            commandState->currentCommandBufferOwner, &bufferState)) {
        if (kMGLDiagnosticStateLogs) {
            mglTraceLog(
                "MGL INFO: %s requested with NULL command buffer, creating one",
                reason ? reason : "operation");
        }
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: Failed to create command buffer for %s\n",
                    reason ? reason : "operation");
            return 0;
        }
        if (!mglRenderCommandBufferOwnerHasState(
                commandState->currentCommandBufferOwner, &bufferState)) {
            fprintf(stderr,
                    "MGL ERROR: Created command buffer owner has no current buffer for %s\n",
                    reason ? reason : "operation");
            return 0;
        }
    }

    const uint32_t status = (uint32_t)bufferState.status;
    if (status >= MGLCommandBufferStatusCommitted) {
        fprintf(stderr,
                "MGL INFO: %s requested on finalized command buffer (status: %ld), rotating\n",
                reason ? reason : "operation", (long)status);
        mglRendererEndRenderEncodingLocked(renderer);
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: Failed to rotate command buffer for %s\n",
                    reason ? reason : "operation");
            return 0;
        }

        memset(&bufferState, 0, sizeof(bufferState));
        if (!mglRenderCommandBufferOwnerHasState(
                commandState->currentCommandBufferOwner, &bufferState) ||
            bufferState.status >= MGLCommandBufferStatusCommitted) {
            fprintf(stderr,
                    "MGL ERROR: Unable to obtain writable command buffer for %s\n",
                    reason ? reason : "operation");
            return 0;
        }
    }

    return 1;
}

typedef struct MglPdCommitCtx_t {
    void *command_buffer;
    int result;
} MglPdCommitCtx;

static int mglPdCommitCommandBufferTryBody(void *renderer, void *rawCtx)
{
    MglPdCommitCtx *ctx = (MglPdCommitCtx *)rawCtx;
    mglRendererCommitCommandBufferWithAGXRecovery(renderer, ctx->command_buffer);
    /* The owner now retains the last submit; flushCommandBuffer waits on that
     * state after releasing METAL_LOCK. */
    ctx->result = 1;
    return 1;
}

/* -flushCommandBufferLocked:.  It calls the public -processGLState: through
 * the existing port: METAL_LOCK() is only a GL-thread assertion, so the
 * *Locked contract is preserved. */
void mglRenderPassFlushCommandBufferLocked(void *renderer, int finish)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLRenderPassManager *manager = areas.render_pass_manager;
    MGLCommandState *commandState = areas.command;

    if (!mglRendererBackendGetDevice(areas.backend) ||
        !mglRendererBackendGetCommandQueue(areas.backend)) {
        fprintf(stderr,
                "MGL ERROR: Metal device or queue is NULL in flushCommandBuffer\n");
        return;
    }

    mglRendererFlushDrawBufferLockedPort(renderer, ctx);

    if (!mglRendererProcessGLStatePort(renderer, 0)) {
        fprintf(stderr,
                "MGL WARNING: processGLState failed in flushCommandBuffer, continuing with cleanup\n");
    }

    /* If processGLStateLocked: left a render encoder active, mark the CB as
     * having work so the commit below is not skipped. */
    if (mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1) {
        areas.batching->currentCommandBufferHasWork = 1;
    }

    mglRendererEndRenderEncodingLocked(renderer);

    /* Skip empty-CB commit when finish!=0: wait on the owner's last submit
     * instead (Metal CBs execute serially on the same queue).  Any path that
     * encodes work (draws/render/blit/compute) into the current CB MUST set
     * currentCommandBufferHasWork before calling flushCommandBuffer:YES, else
     * the skip drops uncommitted work. */
    if (finish && !areas.batching->currentCommandBufferHasWork &&
        mglPassManagerHasLastSubmittedCommandBuffer(manager)) {
        return;
    }
    if (finish && !areas.batching->currentCommandBufferHasWork &&
        !mglPassManagerHasLastSubmittedCommandBuffer(manager)) {
        return;
    }

    if (!mglRenderPassEnsureWritableCommandBufferLocked(renderer,
                                                        "flushCommandBuffer")) {
        fprintf(stderr,
                "MGL ERROR: Unable to obtain writable command buffer in flushCommandBuffer\n");
        return;
    }

    MGLRenderCommandBufferState currentState = {0};
    if (!mglRenderCommandBufferOwnerHasState(
            commandState->currentCommandBufferOwner, &currentState)) {
        fprintf(stderr,
                "MGL WARNING: No current command buffer in flushCommandBuffer\n");
        return;
    }

    const uint32_t currentStatus = (uint32_t)currentState.status;
    if (currentStatus != MGLCommandBufferStatusNotEnqueued) {
        fprintf(stderr,
                "MGL INFO: flushCommandBuffer found finalized buffer (status=%ld), rotating\n",
                (long)currentStatus);
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: Failed to rotate command buffer in flushCommandBuffer\n");
        }
        return;
    }

    const MGLRenderCommandBufferState preCommitState = currentState;
    if (preCommitState.has_error) {
        fprintf(stderr,
                "MGL ERROR: Command buffer has error before commit: %s\n",
                mglRenderCommandBufferErrorDescription(&preCommitState));
        (void)mglPlatformShellGuardedCall(renderer, "command buffer cleanup",
                                          mglRendererCleanupCommandBufferBody);
        return;
    }

    if (!mglRendererValidateMetalObjects(renderer)) {
        fprintf(stderr,
                "MGL WARNING: GPU throttling active - skipping command buffer commit\n");
        (void)mglPlatformShellGuardedCall(renderer, "command buffer cleanup",
                                          mglRendererCleanupCommandBufferBody);
        return;
    }

    void *commandBufferToCommit =
        mglPassManagerDetachCurrentCommandBufferForSubmission(manager);

    MglPdCommitCtx commitCtx = {commandBufferToCommit, 0};
    if (!mglPlatformShellGuardedCallCtx(renderer, "command buffer commit",
                                        mglPdCommitCommandBufferTryBody,
                                        &commitCtx, NULL)) {
        /* @catch (NSException *exception) */
        fprintf(stderr,
                "MGL ERROR: Command buffer commit failed in flushCommandBuffer\n");
        mglRendererRecordGPUError(renderer);
        (void)mglPlatformShellGuardedCall(renderer, "command buffer cleanup",
                                          mglRendererCleanupCommandBufferBody);
    }

    if (!finish) {
        (void)mglRenderPassNewCommandBufferLocked(renderer);
    }
}
