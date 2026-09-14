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
