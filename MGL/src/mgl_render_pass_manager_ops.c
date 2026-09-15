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

#include "mgl_render_pass_sync_ops.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_render_encoder_ops.h"
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
#include "mgl_program_resource.h"   /* mglProgramStageBuiltinMask */
#include "mgl_shader_abi.h"         /* mglAIRPerVertexStrideForResources, mglShaderCompileGLSL */
#include "mgl_metal_ref.h"         /* mglReleaseMetalObjNoNull */
#include "mgl_draw_gs.h"           /* mglDrawGsPassthroughDeclType */
#include "mgl_render_pass_plan.h"    /* mglRenderProcessGLState*, plans */
#include "mgl_buffer_slots.h"        /* kMGL*BufferIndex */
#include "mgl_binding_state_ops.h"  /* mglRendererSyncResourceBindingsForContext */
#include "mgl_swap_diagnostics.h"     /* swap colour copy + diagnostics */
#include <dispatch/dispatch.h>       /* dispatch_async_f */
#include "mgl_trace_log.h"         /* mglTraceLog, kMGLDiagnosticStateLogs */
#include "mgl_gpu_recovery.h"      /* mglRendererRecordGPUError */
#include "mgl_pso_format_class.h"  /* mglRenderDefaultColorPixelFormat */
#include "mgl_render.h"           /* attachment kinds, MS plane adjust */
#include "mgl_texture_compat.h"   /* mglMetalTextureLevelDimension */

#include "mgl_pso_build_ops.h"    /* mglRenderPassSyncPipelineState */
#include "mgl_renderer_ports.h"
#include "mgl_binding_state_ops.h"      /* mglBindingInvalidateLastBoundState */
#include "mgl_trace_strategy.h"         /* mglClearFragmentTraceBindingsForRenderer */
#include "mgl_blit_sampled_copy.h"      /* GLSampled copies refresh */

#include <stdio.h>
#include <stdarg.h>   /* va_list for the GLSL source builder */
#include <stdlib.h>   /* malloc / realloc / free */
#include <string.h>   /* memcpy / strlen */

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
            RETURN_FALSE_ON_FAILURE(mglRenderPassSyncRenderPassStateForContext(renderer, ctx));
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
                    mglRenderPassNewRenderEncoderLockedWithReason(renderer, MGL_ENC_REASON_VAO));
            }

            mglRenderPassUpdateCurrentRenderEncoder(renderer);

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
                    mglRenderPassNewRenderEncoderLockedWithReason(renderer, MGL_ENC_REASON_RS));
            }

            mglRenderPassUpdateCurrentRenderEncoder(renderer);

            mglPdState(&areas)->dirty_bits &= ~DIRTY_RENDER_STATE;
        }

        if (plan.sync_pipeline)
        {
            RETURN_FALSE_ON_FAILURE(mglRenderPassSyncPipelineState(
                renderer, deferredBufferMapForPipelineBuild));
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
    (void)mglRenderPassNewRenderEncoderLockedWithReason(renderer, MGL_ENC_REASON_DRAW);
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
/* Shell forwarder: the drawable is a property on the shell class. */
extern void mglPlatformShellSetDrawable(void *renderer, void *drawable);
extern void mglLogStateSnapshot(const char *tag, GLMContext ctx,
                             void *commandBufferOwner,
                             void *renderEncoderOwner,
                             void *renderPassStateOwner,
                             void *drawable);

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

    if (!mglRenderPassProcessGLStateLocked(renderer, 0)) {
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

/* === AIR passthrough vertex builders (log 177) ========================= */

/* The .m built the source with NSMutableString/appendFormat:; the C twins use
 * this growable buffer instead (vsnprintf into the tail, doubling on demand). */
typedef struct MglPdSource_t {
    char *data;
    size_t len;
    size_t cap;
} MglPdSource;

static void mglPdSourceInit(MglPdSource *src)
{
    src->cap = 4096u;
    src->len = 0u;
    src->data = (char *)malloc(src->cap);
    if (src->data) src->data[0] = '\0';
}

static void mglPdSourceGrow(MglPdSource *src, size_t need)
{
    if (!src->data || need + 1u <= src->cap) return;
    size_t cap = src->cap;
    while (cap < need + 1u) cap *= 2u;
    char *grown = (char *)realloc(src->data, cap);
    if (!grown) return;
    src->data = grown;
    src->cap = cap;
}

/* appendString: */
static void mglPdSourceAppendRaw(MglPdSource *src, const char *text)
{
    if (!src->data || !text) return;
    const size_t n = strlen(text);
    mglPdSourceGrow(src, src->len + n);
    if (!src->data || src->len + n + 1u > src->cap) return;
    memcpy(src->data + src->len, text, n);
    src->len += n;
    src->data[src->len] = '\0';
}

/* appendFormat: */
static void mglPdSourceAppendF(MglPdSource *src, const char *fmt, ...)
{
    if (!src->data) return;
    va_list ap;
    va_start(ap, fmt);
    for (;;) {
        va_list ap2;
        va_copy(ap2, ap);
        const int n = vsnprintf(src->data + src->len, src->cap - src->len, fmt,
                                ap2);
        va_end(ap2);
        if (n < 0) break;
        if ((size_t)n + 1u <= src->cap - src->len) {
            src->len += (size_t)n;
            break;
        }
        mglPdSourceGrow(src, src->len + (size_t)n);
        if (src->len + (size_t)n + 1u > src->cap) break; /* allocation failed */
    }
    va_end(ap);
}

static void mglPdSourceFree(MglPdSource *src)
{
    free(src->data);
    src->data = NULL;
    src->len = 0u;
    src->cap = 0u;
}

/* The .m twin hands the two +1 handles to ARC, which releases them at scope
 * exit; the C twin takes the same +1 and releases it explicitly with the
 * shared helper (mgl_metal_ref.h). */
static bool mglPdLoadAIRMainFunction(const unsigned char *bytes, size_t size,
                                     void **libraryOut, void **functionOut,
                                     char *errorText, size_t errorCap)
{
    if (libraryOut) *libraryOut = NULL;
    if (functionOut) *functionOut = NULL;
    if (!bytes || size == 0u || !libraryOut || !functionOut) {
        if (errorText && errorCap) snprintf(errorText, errorCap, "bad args");
        return false;
    }
    void *libraryHandle = NULL;
    void *functionHandle = NULL;
    if (mglRenderLoadAIRMainFunction(bytes, size, &libraryHandle,
                                     &functionHandle, errorText,
                                     errorCap) != 0 ||
        !libraryHandle || !functionHandle) {
        return false;
    }
    *libraryOut = libraryHandle;
    *functionOut = functionHandle;
    return true;
}

/* Twins of the .m geometry-passthrough type mappers. */
static const char *mglPdGeometryPassthroughColumnSwizzle(unsigned rows)
{
    return mglRenderGLSLColumnSwizzle(rows);
}

static const char *mglPdGeometryPassthroughColumnType(unsigned rows)
{
    return mglRenderGLSLColumnType(rows);
}

static const char *mglPdGeometryPassthroughFloatType(GLenum type)
{
    return mglRenderGLSLIntegerAsFloatType((uint32_t)type);
}

static const char *mglPdGeometryPassthroughSwizzle(GLenum type)
{
    return mglRenderGLSLTypeSwizzle((uint32_t)type);
}

static const char *mglPdGeometryPassthroughType(GLenum type)
{
    return mglRenderGLSLTypeName((uint32_t)type);
}

static unsigned mglPdGeometryPassthroughMatrixCols(GLenum type)
{
    return (unsigned)mglRenderGLSLMatrixCols((uint32_t)type);
}

static unsigned mglPdGeometryPassthroughMatrixRows(GLenum type)
{
    return (unsigned)mglRenderGLSLMatrixRows((uint32_t)type);
}

static bool mglPdGeometryPassthroughNeedsFlat(GLenum type)
{
    return mglRenderGLSLNeedsFlat((uint32_t)type) != 0;
}

static GLenum mglPdPassthroughDeclType(const MGLShaderResourceList *fsInputs,
                                       const MGLShaderResource *output)
{
    uint32_t decl = output->gl_type;
    for (GLuint fi = 0; fsInputs && fsInputs->list && fi < fsInputs->count;
         fi++) {
        const MGLShaderResource *in = &fsInputs->list[fi];
        decl = mglDrawGsPassthroughDeclType(
            decl, in->gl_type,
            output->name && in->name && strcmp(in->name, output->name) == 0
                ? 1
                : 0);
        if (decl != output->gl_type) {
            return (GLenum)decl;
        }
    }
    return (GLenum)decl;
}

/* -ensureAIRGeometryPassthroughFunctionForProgram:outputPrimitive:.  The
 * output-primitive argument was already unused in the .m body. */
int mglRenderPassEnsureAIRGeometryPassthroughFunctionForProgram(
    void *renderer, Program *program, uint32_t outputPrimitive)
{
    (void)outputPrimitive;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;

    if (!program) return 0;
    const uint32_t layerStride = mglPdGeometryPassthroughLayerStride(ctx);
    const uint64_t passthroughKey =
        mglPdGeometryPassthroughCacheKey(program, layerStride);
    void *cachedFunction = NULL;
    if (mglRendererBackendGetPassthroughFunction(
            areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY,
            passthroughKey, &cachedFunction) == 1) {
        return 1;
    }
    (void)mglRendererBackendSetPassthroughFunction(
        areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY, NULL, NULL,
        0u);

    MGLShaderResourceList *outputs =
        &program->shader_resources_list[_GEOMETRY_SHADER][_STAGE_OUTPUT_RES];
    const uint64_t recordStride = mglAIRPerVertexStrideForResources(outputs);
    const uint64_t vec4Stride = recordStride / 16u;
    MglPdSource src;
    mglPdSourceInit(&src);
    mglPdSourceAppendRaw(&src,
                         "#version 460 core\n"
                         "layout(std430, binding = 0) buffer MGLGSOutput {\n"
                         "    vec4 records[];\n"
                         "} mgl_gs_output;\n");
    /* The stage-out record stores every varying as a full vec4 slot, so the
     * reflected gl_type of a GS output is promoted to the record width.  The
     * passthrough VS must declare the interface with the *fragment shader's*
     * input type instead: Metal rejects a pipeline whose vertex output type
     * differs from the fragment input (e.g. record-promoted vec4 vs a declared
     * vec3). */
    const MGLShaderResourceList *fsInputs =
        &program->shader_resources_list[_FRAGMENT_SHADER][_STAGE_INPUT_RES];

    /* Builtins never appear in the reflected output list (all gl_ builtins are
     * filtered during reflection), so ask the frontend for the exact per-stage
     * usage mask instead of scanning the GS source text.  gl_PointSize
     * forwarding matters because the pipeline builder rejects a vertex stage
     * writing point size on a Line/Triangle topology, while Points-topology
     * programs expect the real size. */
    const uint32_t gsBuiltins =
        mglProgramStageBuiltinMask(program, _GEOMETRY_SHADER);
    int hasPointSize = (gsBuiltins & MGL_AIR_BUILTIN_POINT_SIZE) != 0u;
    /* GS-written gl_PrimitiveID is parked at record offset 64 (vec4 slot 4,
     * component x; layout v2 / A04) and ferried to the fragment stage as a flat
     * float varying at the reserved location below. */
    int hasPrimitiveId = (gsBuiltins & MGL_AIR_BUILTIN_PRIMITIVE_ID) != 0u;
    int hasClipDistance = (gsBuiltins & MGL_AIR_BUILTIN_CLIP_DISTANCE) != 0u;
    int result = 0;
    if (hasPrimitiveId) {
        /* Float carrier: the GS kernel stores sitofp(id) and a flat int
         * stage_input that is actually read crashes Apple's AGX compiler (see
         * storeGeometryPrimitiveId). */
        mglPdSourceAppendF(
            &src, "layout(location = %u) flat out float mgl_primitive_id;\n",
            (unsigned)MGL_AIR_PRIMITIVE_ID_LOCATION);
    }
    if (hasClipDistance) {
        mglPdSourceAppendF(&src, "out float gl_ClipDistance[%u];\n",
                           (unsigned)MGL_AIR_PER_VERTEX_CLIP_DISTANCE_COUNT);
    }
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;

        if (output->stream > 0) continue;
        /* gl_PointSize is a built-in: it cannot carry a layout(location)
         * redeclaration.  The kernel parks it in slot 1.x; main() only forwards
         * it when the GS actually declared it, because the pipeline builder
         * rejects a vertex stage that writes point size on a Line/Triangle
         * topology. */
        if (strcmp(output->name, "gl_PointSize") == 0) continue;
        if (getenv("MGL_DUMP_AIR"))
            fprintf(stderr, "MGL PTVS varying: name=%s gl_type=0x%x loc=%u\n",
                    output->name ? output->name : "?",
                    (unsigned)output->gl_type, (unsigned)output->location);
        const GLenum declType = mglPdPassthroughDeclType(fsInputs, output);
        const unsigned matCols = mglPdGeometryPassthroughMatrixCols(declType);
        const unsigned matRows = mglPdGeometryPassthroughMatrixRows(declType);
        if (matCols > 0u) {
            /* Metal rejects matrix stage-out attributes; emit one vector
             * output per column at consecutive locations (GL 4.6 4.4.1). */
            const char *colType = mglPdGeometryPassthroughColumnType(matRows);
            if (!colType || !output->name) {
                fprintf(stderr,
                        "MGL GS ERROR: unsupported passthrough matrix type 0x%x\n",
                        (unsigned)output->gl_type);
                goto done;
            }
            for (unsigned c = 0; c < matCols; c++) {
                mglPdSourceAppendF(
                    &src, "layout(location = %u) out %s %s_c%u;\n",
                    (unsigned)(output->location + c), colType, output->name, c);
            }
            continue;
        }
        /* Integer varyings ride as float carriers (the AIR backend pairs this
         * with an fptosi at the fragment entry; raw int attributes do not
         * survive the GS-expansion pipeline plumbing). */
        const char *type =
            mglPdGeometryPassthroughNeedsFlat(declType)
                ? mglPdGeometryPassthroughFloatType(declType)
                : mglPdGeometryPassthroughType(declType);
        if (!type || !output->name) {
            fprintf(stderr,
                    "MGL GS ERROR: unsupported passthrough varying type 0x%x\n",
                    (unsigned)output->gl_type);
            goto done;
        }
        mglPdSourceAppendF(
            &src, "layout(location = %u) %sout %s %s;\n",
            (unsigned)output->location,
            mglPdGeometryPassthroughNeedsFlat(output->gl_type) ? "flat " : "",
            type, output->name);
    }
    mglPdSourceAppendF(&src,
                       "void main() {\n"
                       "    int mgl_base = gl_VertexID * %lu;\n"
                       "    gl_Position = mgl_gs_output.records[mgl_base];\n",
                       (unsigned long)vec4Stride);
    if (hasPointSize) {
        /* Forward the kernel's point size (slot 1.x).  Only emitted when the GS
         * declared gl_PointSize -- the pipeline builder rejects a vertex stage
         * writing point size on a Line/Triangle topology.  Two-step load: the
         * frontend rejects a member access directly on an SSBO array element. */
        mglPdSourceAppendRaw(
            &src, "    vec4 mgl_point_size = mgl_gs_output.records[mgl_base + 1];\n"
                  "    gl_PointSize = mgl_point_size.x;\n");
    }
    if (hasPrimitiveId) {
        /* Two-step load: the frontend rejects a member access directly on an
         * SSBO array element.  The record already holds the float carrier, so
         * forward it unchanged.  Layout v2: primitive_id @64. */
        const unsigned primSlot =
            (unsigned)(MGL_AIR_PER_VERTEX_PRIMITIVE_ID_OFFSET / 16u);
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_prim_vec = mgl_gs_output.records[mgl_base + %u];\n"
            "    mgl_primitive_id = mgl_prim_vec.x;\n",
            primSlot);
    }
    if (hasClipDistance) {
        /* Clip distances live at byte offset 64 (vec4 slots 4..5). */
        const unsigned clipSlot =
            (unsigned)(MGL_AIR_PER_VERTEX_CLIP_DISTANCE_OFFSET / 16u);
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_clip0 = mgl_gs_output.records[mgl_base + %u];\n"
            "    vec4 mgl_clip1 = mgl_gs_output.records[mgl_base + %u];\n"
            "    gl_ClipDistance[0] = mgl_clip0.x;\n"
            "    gl_ClipDistance[1] = mgl_clip0.y;\n"
            "    gl_ClipDistance[2] = mgl_clip0.z;\n"
            "    gl_ClipDistance[3] = mgl_clip0.w;\n"
            "    gl_ClipDistance[4] = mgl_clip1.x;\n"
            "    gl_ClipDistance[5] = mgl_clip1.y;\n"
            "    gl_ClipDistance[6] = mgl_clip1.z;\n"
            "    gl_ClipDistance[7] = mgl_clip1.w;\n",
            clipSlot, clipSlot + 1u);
    }
    if (getenv("MGL_GS_PROBE")) {
        /* Pixel probe: R = vertex id, G = GPU-read position.y remapped, B =
         * GPU-read varying.r.  Renders the real positions so the geometry stays
         * identifiable. */
        mglPdSourceAppendRaw(
            &src, "    vec4 mgl_probe_pos = mgl_gs_output.records[mgl_base];\n"
                  "    vec4 mgl_probe_col = mgl_gs_output.records[mgl_base + 4];\n"
                  "    gl_Position = mgl_probe_pos;\n"
                  "    gs_fs_color = vec4(float(gl_VertexID) / 6.0,\n"
                  "                        mgl_probe_pos.y * 0.5 + 0.5,\n"
                  "                        abs(mgl_probe_col.r), 1.0);\n"
                  "    return;\n");
    }
    if (getenv("MGL_GS_PROBE_VID")) {
        /* Probe 2: ignore the SSBO entirely; geometry is derived from
         * gl_VertexID alone (six points spread horizontally at mid height).
         * Correct render => vertex ids / draw are sane and the defect is in the
         * SSBO read path. */
        mglPdSourceAppendRaw(
            &src, "    float mgl_vid = float(gl_VertexID);\n"
                  "    gl_Position = vec4(mgl_vid / 3.0 - 1.0, 0.25, 0.0, 1.0);\n"
                  "    gs_fs_color = vec4(mgl_vid / 6.0, 1.0, 0.0, 1.0);\n"
                  "    return;\n");
    }
    if (getenv("MGL_GS_PROBE_WAVE")) {
        /* Probe 3: oscilloscope.  Vertex x is fixed by vid; the polyline y
         * traces records[mgl_base].x as read on the GPU, and color carries
         * .y/.z/.w.  One render reconstructs every slot the passthrough VS
         * actually sees. */
        mglPdSourceAppendRaw(
            &src, "    float mgl_vid = float(gl_VertexID);\n"
                  "    vec4 mgl_p0 = mgl_gs_output.records[mgl_base];\n"
                  "    gl_Position = vec4(mgl_vid / 3.0 - 1.0, mgl_p0.x, 0.0, 1.0);\n"
                  "    gs_fs_color = vec4(mgl_p0.y * 0.5 + 0.5,\n"
                  "                       mgl_p0.z * 0.5 + 0.5,\n"
                  "                       mgl_p0.w * 0.5 + 0.5, 1.0);\n"
                  "    return;\n");
    }
    int fsNeedsLayer = 0;
    int fsNeedsViewport = 0;
    for (GLuint fi = 0; fsInputs && fsInputs->list && fi < fsInputs->count;
         fi++) {
        const MGLShaderResource *in = &fsInputs->list[fi];
        if (!in->name) continue;
        if (strcmp(in->name, "gl_Layer") == 0) fsNeedsLayer = 1;
        if (strcmp(in->name, "gl_ViewportIndex") == 0) fsNeedsViewport = 1;
    }
    {
        const uint32_t fsBuiltins =
            mglProgramStageBuiltinMask(program, _FRAGMENT_SHADER);
        if (!fsNeedsLayer && (fsBuiltins & MGL_AIR_BUILTIN_LAYER) != 0u)
            fsNeedsLayer = 1;
        if (!fsNeedsViewport &&
            (fsBuiltins & MGL_AIR_BUILTIN_VIEWPORT_INDEX) != 0u)
            fsNeedsViewport = 1;
    }
    if (getenv("MGL_PTVS_NO_SPECIALS")) {
        /* Diagnostic: omit the layer/viewport special outputs entirely so the
         * vertex return carries only position + user varyings. */
    } else if ((gsBuiltins &
                (MGL_AIR_BUILTIN_LAYER | MGL_AIR_BUILTIN_VIEWPORT_INDEX)) != 0u ||
               fsNeedsLayer || fsNeedsViewport) {

        /* Layout v2 (A04): layer @52 / viewport @56 share vec4 slot 3 as .y /
         * .z (stream occupies .w). */
        const unsigned layerSlot =
            (unsigned)(MGL_AIR_PER_VERTEX_LAYER_OFFSET / 16u);
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_layer_vp = mgl_gs_output.records[mgl_base + %u];\n"
            "    gl_Layer = floatBitsToInt(mgl_layer_vp.y) * %u;\n"
            "    gl_ViewportIndex = floatBitsToInt(mgl_layer_vp.z);\n",
            layerSlot, (unsigned)layerStride);
    }
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        if (output->stream > 0) continue;
        const GLenum declType = mglPdPassthroughDeclType(fsInputs, output);
        const unsigned matCols = mglPdGeometryPassthroughMatrixCols(declType);
        const unsigned matRows = mglPdGeometryPassthroughMatrixRows(declType);
        if (matCols > 0u) {
            /* Stage-out stores one column per location slot (GL 4.6 4.4.1).
             * Forward each column as its own vector varying. */
            const char *colSwizzle =
                mglPdGeometryPassthroughColumnSwizzle(matRows);
            if (!colSwizzle || !output->name) goto done;
            const unsigned baseSlot =
                (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location);
            for (unsigned c = 0; c < matCols; c++) {
                mglPdSourceAppendF(
                    &src,
                    "    vec4 mgl_slot_%u_%u = mgl_gs_output.records[mgl_base + %u];\n"
                    "    %s_c%u = mgl_slot_%u_%u%s;\n",
                    (unsigned)i, c, baseSlot + c, output->name, c, (unsigned)i,
                    c, colSwizzle);
            }
            continue;
        }
        const char *swizzle = mglPdGeometryPassthroughSwizzle(declType);
        if (!swizzle || !output->name) goto done;
        /* Integer records already hold SIToFP/UIToFP float carriers - forward
         * the float swizzle; do not floatBitsTo*. */
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_slot_%u = mgl_gs_output.records[mgl_base + %u];\n"
            "    %s = mgl_slot_%u%s;\n",
            (unsigned)i,
            (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location),
            output->name, (unsigned)i, swizzle);
    }
    mglPdSourceAppendRaw(&src, "}\n");
    if (getenv("MGL_GS_DIAG")) {
        fprintf(stderr, "MGL GS DIAG passthrough VS source:\n%s\n", src.data);
    }
    {
        unsigned char *bytes = NULL;
        size_t size = 0u;
        char errorText[512] = {0};
        if (mglShaderCompileGLSL(src.data, MGL_STAGE_VERTEX, &bytes, &size,
                                 errorText, sizeof(errorText)) != 0 ||
            !bytes || size == 0u) {
            fprintf(stderr,
                    "MGL GS ERROR: failed to compile AIR passthrough vertex: %s\n",
                    errorText[0] ? errorText : "?");
            mglShaderFree(bytes);
            goto done;
        }
        if (getenv("MGL_DUMP_AIR")) {
            FILE *f = fopen("/tmp/poison_ptvs.air", "wb");
            if (f) {
                fwrite(bytes, 1, size, f);
                fclose(f);
                fprintf(stderr, "MGL DUMP: ptvs.air %zu bytes\n", size);
            }
        }
        void *library = NULL;
        void *function = NULL;
        const bool loaded = mglPdLoadAIRMainFunction(
            bytes, size, &library, &function, errorText, sizeof(errorText));
        mglShaderFree(bytes);
        if (!loaded || !library || !function) {
            fprintf(stderr,
                    "MGL GS ERROR: failed to load AIR passthrough vertex: %s\n",
                    errorText[0] ? errorText : "?");
            goto done;
        }
        result = mglRendererBackendSetPassthroughFunction(
                     areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_GEOMETRY,
                     library, function, passthroughKey) == 0
                     ? 1
                     : 0;
        /* ARC released these at scope exit in the .m twin. */
        mglReleaseMetalObjNoNull(library);
        mglReleaseMetalObjNoNull(function);
    }
done:
    mglPdSourceFree(&src);
    return result;
}

/* -ensureAIRTessEvalPassthroughFunctionForProgram:. */
int mglRenderPassEnsureAIRTessEvalPassthroughFunctionForProgram(void *renderer,
                                                                Program *program)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    if (!program) return 0;
    void *cachedFunction = NULL;
    if (mglRendererBackendGetPassthroughFunction(
            areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION,
            program->pipeline_cache_instance_id, &cachedFunction) == 1) {
        return 1;
    }
    (void)mglRendererBackendSetPassthroughFunction(
        areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION, NULL,
        NULL, 0u);

    MGLShaderResourceList *outputs = &program
                                          ->shader_resources_list
                                              [_TESS_EVALUATION_SHADER]
                                              [_STAGE_OUTPUT_RES];
    const uint64_t recordStride = mglAIRPerVertexStrideForResources(outputs);
    const uint64_t vec4Stride = recordStride / 16u;
    const int hasClipDistance =
        (mglProgramStageBuiltinMask(program, _TESS_EVALUATION_SHADER) &
         MGL_AIR_BUILTIN_CLIP_DISTANCE) != 0u;
    /* Isolines rasterize as lines; Metal rejects a vertex stage that writes
     * point size on a non-point topology. */
    const int writePointSize =
        mglTessWritePointSize((uint32_t)program->tess_gen_point_mode) != 0;
    MglPdSource src;
    mglPdSourceInit(&src);
    mglPdSourceAppendRaw(&src,
                         "#version 460 core\n"
                         "layout(std430, binding = 0) buffer MGLTESOutput {\n"
                         "    vec4 records[];\n"
                         "} mgl_tes_output;\n");
    int result = 0;
    if (hasClipDistance) {
        mglPdSourceAppendF(&src, "out float gl_ClipDistance[%u];\n",
                           (unsigned)MGL_AIR_PER_VERTEX_CLIP_DISTANCE_COUNT);
    }
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        /* Integer varyings are stored as float carriers in the TES record (same
         * ABI as GS expansion); declare float attributes and forward the
         * swizzle - FS converts with fptosi/fptoui. */
        const unsigned matCols = mglPdGeometryPassthroughMatrixCols(
            output->gl_type);
        const unsigned matRows = mglPdGeometryPassthroughMatrixRows(
            output->gl_type);
        if (matCols > 0u) {
            const char *colType = mglPdGeometryPassthroughColumnType(matRows);
            if (!colType || !output->name) {
                fprintf(stderr,
                        "MGL TESS ERROR: unsupported passthrough matrix type 0x%x\n",
                        (unsigned)output->gl_type);
                goto done;
            }
            for (unsigned c = 0; c < matCols; c++) {
                mglPdSourceAppendF(
                    &src, "layout(location = %u) out %s %s_c%u;\n",
                    (unsigned)(output->location + c), colType, output->name, c);
            }
            continue;
        }
        const char *type =
            mglPdGeometryPassthroughNeedsFlat(output->gl_type)
                ? mglPdGeometryPassthroughFloatType(output->gl_type)
                : mglPdGeometryPassthroughType(output->gl_type);
        if (!type || !output->name) {
            fprintf(stderr,
                    "MGL TESS ERROR: unsupported passthrough varying type 0x%x\n",
                    (unsigned)output->gl_type);
            goto done;
        }
        mglPdSourceAppendF(
            &src, "layout(location = %u) %sout %s %s;\n",
            (unsigned)output->location,
            mglPdGeometryPassthroughNeedsFlat(output->gl_type) ? "flat " : "",
            type, output->name);
    }
    mglPdSourceAppendF(&src,
                       "void main() {\n"
                       "    int mgl_base = gl_VertexID * %lu;\n"
                       "    gl_Position = mgl_tes_output.records[mgl_base];\n",
                       (unsigned long)vec4Stride);
    if (writePointSize) {
        mglPdSourceAppendRaw(
            &src, "    vec4 mgl_point_size = mgl_tes_output.records[mgl_base + 1];\n"
                  "    gl_PointSize = mgl_point_size.x;\n");
    }
    if (hasClipDistance) {
        const unsigned clipSlot =
            (unsigned)(MGL_AIR_PER_VERTEX_CLIP_DISTANCE_OFFSET / 16u);
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_clip0 = mgl_tes_output.records[mgl_base + %u];\n"
            "    vec4 mgl_clip1 = mgl_tes_output.records[mgl_base + %u];\n"
            "    gl_ClipDistance[0] = mgl_clip0.x;\n"
            "    gl_ClipDistance[1] = mgl_clip0.y;\n"
            "    gl_ClipDistance[2] = mgl_clip0.z;\n"
            "    gl_ClipDistance[3] = mgl_clip0.w;\n"
            "    gl_ClipDistance[4] = mgl_clip1.x;\n"
            "    gl_ClipDistance[5] = mgl_clip1.y;\n"
            "    gl_ClipDistance[6] = mgl_clip1.z;\n"
            "    gl_ClipDistance[7] = mgl_clip1.w;\n",
            clipSlot, clipSlot + 1u);
    }
    if (program->tess_cull_distance_count > 0u) {
        const int isolines =
            mglTessGenModeIsIsolines((uint32_t)program->tess_gen_mode) != 0;
        if (isolines) {
            /* Both endpoints of an isoline segment share the same v, so the
             * cull condition needs the partner record's distances.  The partner
             * record index is (gl_VertexID ^ 1) -- every patch span holds an
             * even item count. */
            mglPdSourceAppendF(
                &src,
                "    int mgl_partner = (gl_VertexID ^ 1) * %lu;\n"
                "    vec4 mgl_p0 = mgl_tes_output.records[mgl_partner + 1];\n"
                "    vec4 mgl_p1 = mgl_tes_output.records[mgl_partner + 2];\n"
                "    vec4 mgl_p2 = mgl_tes_output.records[mgl_partner + 3];\n",
                (unsigned long)vec4Stride);
        }
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_c0 = mgl_tes_output.records[mgl_base + 1];\n"
            "    vec4 mgl_c1 = mgl_tes_output.records[mgl_base + 2];\n"
            "    vec4 mgl_c2 = mgl_tes_output.records[mgl_base + 3];\n"
            "    bool mgl_culled = false\n"
            "%s"
            "    if (mgl_culled) gl_Position = vec4(2.0, 2.0, 2.0, 1.0);\n",
            isolines ? "        || (mgl_c0.y < 0.0 && mgl_p0.y < 0.0)\n"
                       "        || (mgl_c0.z < 0.0 && mgl_p0.z < 0.0)\n"
                       "        || (mgl_c0.w < 0.0 && mgl_p0.w < 0.0)\n"
                       "        || (mgl_c1.x < 0.0 && mgl_p1.x < 0.0)\n"
                       "        || (mgl_c1.y < 0.0 && mgl_p1.y < 0.0)\n"
                       "        || (mgl_c1.z < 0.0 && mgl_p1.z < 0.0)\n"
                       "        || (mgl_c1.w < 0.0 && mgl_p1.w < 0.0)\n"
                       "        || (mgl_c2.x < 0.0 && mgl_p2.x < 0.0);\n"
                     : "        || mgl_c0.y < 0.0\n"
                       "        || mgl_c0.z < 0.0\n"
                       "        || mgl_c0.w < 0.0\n"
                       "        || mgl_c1.x < 0.0\n"
                       "        || mgl_c1.y < 0.0\n"
                       "        || mgl_c1.z < 0.0\n"
                       "        || mgl_c1.w < 0.0\n"
                       "        || mgl_c2.x < 0.0;\n");
    }
    for (GLuint i = 0; outputs->list && i < outputs->count; i++) {
        MGLShaderResource *output = &outputs->list[i];
        if (output->is_per_patch) continue;
        const unsigned matCols = mglPdGeometryPassthroughMatrixCols(
            output->gl_type);
        const unsigned matRows = mglPdGeometryPassthroughMatrixRows(
            output->gl_type);
        if (matCols > 0u) {
            const char *colSwizzle =
                mglPdGeometryPassthroughColumnSwizzle(matRows);
            if (!colSwizzle || !output->name) goto done;
            const unsigned baseSlot =
                (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location);
            for (unsigned c = 0; c < matCols; c++) {
                mglPdSourceAppendF(
                    &src,
                    "    vec4 mgl_slot_%u_%u = mgl_tes_output.records[mgl_base + %u];\n"
                    "    %s_c%u = mgl_slot_%u_%u%s;\n",
                    (unsigned)i, c, baseSlot + c, output->name, c, (unsigned)i,
                    c, colSwizzle);
            }
            continue;
        }
        const char *swizzle = mglPdGeometryPassthroughSwizzle(output->gl_type);
        if (!swizzle || !output->name) goto done;
        mglPdSourceAppendF(
            &src,
            "    vec4 mgl_slot_%u = mgl_tes_output.records[mgl_base + %u];\n"
            "    %s = mgl_slot_%u%s;\n",
            (unsigned)i,
            (unsigned)(MGL_AIR_PER_VERTEX_STRIDE / 16u + output->location),
            output->name, (unsigned)i, swizzle);
    }
    mglPdSourceAppendRaw(&src, "}\n");
    if (getenv("MGL_GS_DIAG")) {
        fprintf(stderr, "MGL GS DIAG passthrough VS source:\n%s\n", src.data);
    }
    {
        unsigned char *bytes = NULL;
        size_t size = 0u;
        char errorText[512] = {0};
        if (mglShaderCompileGLSL(src.data, MGL_STAGE_VERTEX, &bytes, &size,
                                 errorText, sizeof(errorText)) != 0 ||
            !bytes || size == 0u) {
            fprintf(stderr,
                    "MGL TESS ERROR: failed to compile AIR TES passthrough vertex: %s\nSOURCE:\n%s\n",
                    errorText[0] ? errorText : "?", src.data);
            mglShaderFree(bytes);
            goto done;
        }
        void *library = NULL;
        void *function = NULL;
        const bool loaded = mglPdLoadAIRMainFunction(
            bytes, size, &library, &function, errorText, sizeof(errorText));
        mglShaderFree(bytes);
        if (!loaded || !library || !function) {
            fprintf(stderr,
                    "MGL TESS ERROR: failed to load AIR TES passthrough vertex: %s\n",
                    errorText[0] ? errorText : "?");
            goto done;
        }
        result =
            mglRendererBackendSetPassthroughFunction(
                areas.backend, MGL_RENDERER_BACKEND_PASSTHROUGH_TESS_EVALUATION,
                library, function, program->pipeline_cache_instance_id) == 0
                ? 1
                : 0;
        /* ARC released these at scope exit in the .m twin. */
        mglReleaseMetalObjNoNull(library);
        mglReleaseMetalObjNoNull(function);
    }
done:
    mglPdSourceFree(&src);
    return result;
}

/* === draw-path guard cluster (log 178) ================================= */

/* -invalidateCurrentPipelineStateForReason:. */
void mglRenderPassInvalidateCurrentPipelineState(void *renderer,
                                                 const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    if (areas.pipeline_cache && areas.pipeline_cache->pipelineState) {
        static uint64_t s_pipelineInvalidateCount = 0;
        const uint64_t hit = ++s_pipelineInvalidateCount;
        if (hit <= 16ull || (hit % 512ull) == 0ull) {
            fprintf(stderr,
                    "MGL WARNING: Invalidating current pipeline state after %s hit=%llu\n",
                    reason ? reason : "pipeline failure",
                    (unsigned long long)hit);
        }
    }
    if (areas.pipeline_cache_invalidate) {
        areas.pipeline_cache_invalidate(areas.pipeline_cache_object);
    }
}

/* -ensureCurrentRenderPassMatchesFramebufferForDraw. */
int mglRenderPassEnsureCurrentRenderPassMatchesFramebufferForDraw(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *commandState = areas.command;

    if (!ctx) {
        return 1;
    }

    if (mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) != 1) {
        return 1;
    }

    if (mglRenderPassMatchesCurrentFramebuffer(renderer)) {
        return 1;
    }

    static uint64_t s_fboPassMismatchCount = 0;
    const uint64_t hit = ++s_fboPassMismatchCount;
    if (hit <= 32ull || (hit % 256ull) == 0ull) {
        Framebuffer *fbo = mglPdState(&areas)->framebuffer;
        void *color0 = mglPdColorTextureFor(commandState, 0);
        const GLuint mglDefaultDrawbuffer =
            fbo ? 0u
                : mglDefaultDrawBufferIndexForGL(mglPdState(&areas)->draw_buffer);
        void *expectedDefaultColor0 = NULL;
        if (!fbo) {
            expectedDefaultColor0 =
                mglRenderDefaultDrawBufferIsFront(mglDefaultDrawbuffer)
                    ? mglRendererDrawableTexturePort(renderer)
                    : (mglRenderDefaultDrawBufferIsOffscreen(
                           mglDefaultDrawbuffer, _MAX_DRAW_BUFFERS)
                           ? mglRenderPassDefaultDrawBufferAttachment(
                                 areas.backend, mglDefaultDrawbuffer,
                                 MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR)
                           : NULL);
        }
        const GLuint fboName = fbo ? fbo->name : 0u;
        const GLuint attachment0Name =
            (fbo && (fbo->color_attachment_bitfield & 1u))
                ? fbo->color_attachments[0].texture
                : 0u;
        fprintf(stderr,
                "MGL WARNING: render pass/FBO mismatch before draw hit=%llu fbo=%u drawBuffer=0x%x attachment0=%u passColor0=%p expectedDefaultColor0=%p defaultDrawBuffer=%u; rebuilding encoder\n",
                (unsigned long long)hit, (unsigned)fboName,
                (unsigned)(ctx ? mglPdState(&areas)->draw_buffer : 0u),
                (unsigned)attachment0Name, color0, expectedDefaultColor0,
                (unsigned)mglDefaultDrawbuffer);
        mglLogRenderPassLifecycle(
            fbo ? "fbo-mismatch-before-rebuild"
                : "default-fbo-mismatch-before-rebuild",
            hit, ctx, commandState->currentCommandBufferOwner,
            commandState->currentRenderEncoderOwner,
            commandState->renderPassStateOwner, areas.drawable,
            commandState->renderPassFramebuffer,
            commandState->renderPassFramebufferName,
            commandState->renderPassDrawBuffer,
            commandState->renderPassDrawBufferCount);
    }

    mglRendererEndRenderEncodingLocked(renderer);
    mglMarkRendererDirtyBits(ctx->active_state,
                             DIRTY_FBO | DIRTY_PROGRAM | DIRTY_RENDER_STATE |
                                 DIRTY_VAO);
    return mglRenderPassNewRenderEncoderLockedWithReason(
        renderer, MGL_ENC_REASON_FBO);
}

/* -emergencyResetMetalState. */
typedef struct MglPdEmergencyResetCtx_t {
    int result;
} MglPdEmergencyResetCtx;

static int mglPdEmergencyResetTryBody(void *renderer, void *rawCtx)
{
    MglPdEmergencyResetCtx *ctx = (MglPdEmergencyResetCtx *)rawCtx;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLRenderPassManager *manager = areas.render_pass_manager;

    /* Force cleanup of all Metal objects */
    mglRendererEndRenderEncodingLocked(renderer);

    mglPassManagerDiscardCurrentCommandBuffer(manager);
    mglPassManagerClearCurrentRenderEncoder(manager);
    mglPlatformShellSetDrawable(renderer, NULL);

    /* Re-initialize basic Metal objects */
    if (mglRendererBackendGetDevice(areas.backend) &&
        mglRendererBackendGetCommandQueue(areas.backend)) {
        fprintf(stderr, "MGL CRITICAL: Re-creating Metal command buffer\n");
        (void)mglPassManagerInstallNewCommandBufferFromQueue(
            manager, mglRendererBackendGetCommandQueue(areas.backend));

        if (mglRenderCommandBufferOwnerHasCurrent(
                areas.command->currentCommandBufferOwner) != 1) {
            fprintf(stderr,
                    "MGL CRITICAL: Failed to create new command buffer during recovery\n");
        }
    }
    ctx->result = 1;
    return 1;
}

void mglRenderPassEmergencyResetMetalState(void *renderer)
{
    fprintf(stderr, "MGL CRITICAL: Performing emergency Metal state reset\n");

    MglPdEmergencyResetCtx ctx = {0};
    /* @catch (NSException *exception) only logs and returns. */
    (void)mglPlatformShellGuardedCallCtx(renderer, "emergency metal reset",
                                         mglPdEmergencyResetTryBody, &ctx,
                                         NULL);
}

/* -validateRenderPassAttachmentsAndPipelineFormatsLocked:. */
int mglRenderPassValidateAttachmentsAndPipelineFormats(void *renderer,
                                                       int traceProcess)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *commandState = areas.command;

    /* Guard against invalid render pass state before binding pipeline.  Metal
     * debug validation can abort the process if the encoder/render pass is
     * incompatible. */
    const int hasRenderPassState =
        commandState->renderPassStateOwner != NULL;
    if (!hasRenderPassState) {
        fprintf(stderr,
                "MGL ERROR: processGLState - render pass state owner is nil before pipeline bind\n");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_rpd", ctx,
                                commandState->currentCommandBufferOwner,
                                commandState->currentRenderEncoderOwner,
                                commandState->renderPassStateOwner,
                                areas.drawable);
        }
        return 0;
    }
    int passHasAnyAttachment = 0;
    for (int i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        void *colorAttachment = mglPdColorTextureFor(commandState, (size_t)i);
        if (colorAttachment) {
            passHasAnyAttachment = 1;
            if ((mglPdTextureInfo(colorAttachment).usage &
                 MGLTextureUsageRenderTarget) == 0) {
                fprintf(stderr,
                        "MGL WARNING: processGLState - color attachment %d missing RenderTarget usage (usage=0x%lx); skipping draw\n",
                        i,
                        (unsigned long)mglPdTextureInfo(colorAttachment).usage);
                if (traceProcess) {
                    mglLogStateSnapshot("processGLState.fail.color_usage", ctx,
                                        commandState->currentCommandBufferOwner,
                                        commandState->currentRenderEncoderOwner,
                                        commandState->renderPassStateOwner,
                                        areas.drawable);
                }
                return 0;
            }
        }
    }
    if (mglPdDepthTextureFor(commandState) ||
        mglPdStencilTextureFor(commandState)) {
        passHasAnyAttachment = 1;
    }

    if (!passHasAnyAttachment) {
        fprintf(stderr,
                "MGL WARNING: processGLState - render pass has no attachments, skipping draw to avoid Metal assert\n");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.no_attachments", ctx,
                                commandState->currentCommandBufferOwner,
                                commandState->currentRenderEncoderOwner,
                                commandState->renderPassStateOwner,
                                areas.drawable);
        }
        return 0;
    }

    uint32_t currentColor0Format = mglRenderInvalidPixelFormat();
    uint32_t currentDepthFormat = mglRenderInvalidPixelFormat();
    uint32_t currentStencilFormat = mglRenderInvalidPixelFormat();

    void *rpColor0 = mglPdColorTextureFor(commandState, 0);
    void *rpDepth = mglPdDepthTextureFor(commandState);
    void *rpStencil = mglPdStencilTextureFor(commandState);
    if (rpColor0) {
        currentColor0Format = mglPdTextureInfo(rpColor0).pixel_format;
    }
    if (rpDepth) {
        currentDepthFormat = mglPdTextureInfo(rpDepth).pixel_format;
    }
    if (rpStencil) {
        currentStencilFormat = mglPdTextureInfo(rpStencil).pixel_format;
    }

    /* IMPORTANT: never mutate depth/stencil attachments here to "fit" an
     * existing pipeline.  The active Metal render encoder was already created
     * with a render-pass descriptor, and changing attachments after encoder
     * creation does not make that encoder compatible.  We must instead reject
     * mismatched pipeline/pass combinations and rebuild safely. */
    if (mglRenderPipelinePassColorMismatch(
            (uint32_t)areas.pipeline_cache->pipelineColor0Format,
            currentColor0Format)) {
        static uint64_t s_colorFormatMismatchCount = 0;
        s_colorFormatMismatchCount++;
        if (s_colorFormatMismatchCount <= 16 ||
            (s_colorFormatMismatchCount % 250) == 0) {
            fprintf(stderr,
                    "MGL WARNING: Pipeline/pass color format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild\n",
                    (unsigned long)areas.pipeline_cache->pipelineColor0Format,
                    (unsigned long)currentColor0Format);
        }
        mglRenderPassInvalidateCurrentPipelineState(
            renderer, "pipeline/pass color format mismatch");
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                     DIRTY_RENDER_STATE);
        return 0;
    }

    if (mglRenderPipelinePassAttachmentMismatch(
            (uint32_t)areas.pipeline_cache->pipelineDepthFormat,
            currentDepthFormat)) {
        static uint64_t s_depthFormatMismatchCount = 0;
        s_depthFormatMismatchCount++;
        if (s_depthFormatMismatchCount <= 16 ||
            (s_depthFormatMismatchCount % 250) == 0) {
            fprintf(stderr,
                    "MGL WARNING: Pipeline/pass depth format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild\n",
                    (unsigned long)areas.pipeline_cache->pipelineDepthFormat,
                    (unsigned long)currentDepthFormat);
        }
        mglRenderPassInvalidateCurrentPipelineState(
            renderer, "pipeline/pass depth format mismatch");
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                     DIRTY_RENDER_STATE);
        return 0;
    }

    if (mglRenderPipelinePassAttachmentMismatch(
            (uint32_t)areas.pipeline_cache->pipelineStencilFormat,
            currentStencilFormat)) {
        static uint64_t s_stencilFormatMismatchCount = 0;
        s_stencilFormatMismatchCount++;
        if (s_stencilFormatMismatchCount <= 16 ||
            (s_stencilFormatMismatchCount % 250) == 0) {
            fprintf(stderr,
                    "MGL WARNING: Pipeline/pass stencil format mismatch (pipeline=%lu pass=%lu), forcing pipeline rebuild\n",
                    (unsigned long)areas.pipeline_cache->pipelineStencilFormat,
                    (unsigned long)currentStencilFormat);
        }
        mglRenderPassInvalidateCurrentPipelineState(
            renderer, "pipeline/pass stencil format mismatch");
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                     DIRTY_RENDER_STATE);
        return 0;
    }
    return 1;
}

/* === processGLStateLocked (log 179) ==================================== */

/* Declared in the Objective-C MGLRenderer+Draw_Private.h. */
/* MGLRenderer+Draw_Private.h has this as a `static inline`; C needs its own
 * twin (kMGLDiagnosticStateLogs is 0 in mgl_trace_log.h). */
static bool mglPdShouldTraceCall(uint64_t count)
{
    if (!kMGLDiagnosticStateLogs) {
        return false;
    }
    return (count <= 80ull) || ((count % 500ull) == 0ull);
}
extern void mglLogLoopHeartbeat(const char *tag, uint64_t call,
                                double now_seconds, double *last_time,
                                uint64_t *last_count, double interval);
extern Program *mglResolveProgramFromState(GLMContext ctx);

/* File-scope twins of the .m's function-local statics. */
static uint64_t s_pdProcessGLStateCallCount = 0;
static double s_pdProcessGLStateLastCallTime = 0.0;
static uint64_t s_pdProcessGLStateLastCallCount = 0;
static int s_pdCorruptionRecoveryCount = 0;
static const int kMglPdMaxRecoveryAttempts = 3;
static uint64_t s_pdQuarantineSkipCount = 0;
static uint64_t s_pdRotateFinalizedCount = 0;
static uint64_t s_pdNilEncoderRecoveryCount = 0;
static uint64_t s_pdDrawPipelineLookupCount = 0;
static uint64_t s_pdNilPipelineCount = 0;


/* @try of the corruption-recovery block: 1 = device and queue are usable. */
static int mglPdRecoveryTryBody(void *renderer)
{
    mglRenderPassEmergencyResetMetalState(renderer);
    s_pdCorruptionRecoveryCount++;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    const int deviceOk =
        mglRendererBackendGetDevice(areas.backend) &&
        ((uintptr_t)mglRendererBackendGetDevice(areas.backend) >= 0x1000);
    const int queueOk =
        mglRendererBackendGetCommandQueue(areas.backend) &&
        ((uintptr_t)mglRendererBackendGetCommandQueue(areas.backend) >= 0x1000);
    if (!deviceOk || !queueOk) {
        fprintf(stderr,
                "MGL CRITICAL: Metal recovery failed, aborting operation\n");
        return 0;
    }
    return 1;
}

/* @try of the no-VAO clear path: the catch only logs. */
static int mglPdNoVaoEncoderTryBody(void *renderer)
{
    (void)mglRenderPassNewRenderEncoderLockedWithReason(renderer,
                                                          MGL_ENC_REASON_CLEAR);
    return 1;
}

typedef struct MglPdSetPipelineCtx_t {
    MGLRendererStateAreas *areas;
} MglPdSetPipelineCtx;

/* @try of the set-pipeline block: the body has no failure path, so 0 from the
 * guarded call means "an exception was thrown". */
static int mglPdSetPipelineTryBody(void *renderer, void *rawCtx)
{
    MglPdSetPipelineCtx *ctx = (MglPdSetPipelineCtx *)rawCtx;
    MGLCommandState *commandState = ctx->areas->command;
    /* The areas field is the ADDRESS of the owner slot (mgl_renderer_ports.h);
     * the .m this block came from passed the ivar. */
    void *bindingOwner = ctx->areas->binding_state_owner
                             ? *ctx->areas->binding_state_owner
                             : NULL;
    if (mglRenderBindingSetPipelineIfNeededForOwner(
            bindingOwner,
            commandState->currentRenderEncoderOwner,
            ctx->areas->pipeline_cache
                ? ctx->areas->pipeline_cache->pipelineState
                : NULL) > 0) {
        MGL_PERF_INC(g_mglSetRenderPipelineStateCallsSinceSwap);
    } else {
        MGL_PERF_INC(g_mglSetRenderPipelineStateSkipsSinceSwap);
    }
    return 1;
}

/* -processGLStateLocked:. */
int mglRenderPassProcessGLStateLocked(void *renderer, int draw_command)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLRenderPassManager *manager = areas.render_pass_manager;
    MGLCommandState *commandState = areas.command;
    GLMState *glState = mglPdState(&areas);

    const uint64_t processCall = ++s_pdProcessGLStateCallCount;
    const double processStartSeconds = mglTraceNowSeconds();
    const uint64_t processStartNS = mglTraceClockNS();
    const bool traceProcess = mglPdShouldTraceCall(processCall);
    mglLogLoopHeartbeat("processGLState.loop", processCall, processStartSeconds,
                        &s_pdProcessGLStateLastCallTime,
                        &s_pdProcessGLStateLastCallCount, 0.25);
    if (traceProcess) {
        mglTraceLog("MGL TRACE processGLState.begin call=%llu draw=%d",
                    (unsigned long long)processCall, draw_command ? 1 : 0);
        mglLogStateSnapshot("processGLState.enter", ctx,
                            commandState->currentCommandBufferOwner,
                            commandState->currentRenderEncoderOwner,
                            commandState->renderPassStateOwner, areas.drawable);
    }
    if (!ctx) {
        fprintf(stderr, "MGL ERROR: NULL context detected in processGLState\n");
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.null_ctx", ctx,
                                commandState->currentCommandBufferOwner,
                                commandState->currentRenderEncoderOwner,
                                commandState->renderPassStateOwner,
                                areas.drawable);
        }
        return 0;
    }

    const uintptr_t earlyCtxAddr = (uintptr_t)ctx;
    const int ctxPtrSane = earlyCtxAddr >= 0x1000 ? 1 : 0;
    if (!ctxPtrSane) {
        fprintf(stderr, "MGL ERROR: Invalid context pointer detected: 0x%lx\n",
                (unsigned long)earlyCtxAddr);
        return 0;
    }

    /* Metal corruption recovery is platform materialization - try before plan. */
    int deviceOk = mglRendererBackendGetDevice(areas.backend) &&
                   ((uintptr_t)mglRendererBackendGetDevice(areas.backend) >=
                    0x1000);
    int queueOk = mglRendererBackendGetCommandQueue(areas.backend) &&
                  ((uintptr_t)mglRendererBackendGetCommandQueue(areas.backend) >=
                   0x1000);
    if (!deviceOk || !queueOk) {
        fprintf(stderr,
                "MGL CRITICAL: Metal state corruption detected in processGLState!\n");
        fprintf(stderr, "MGL CRITICAL: device=0x%lx, queue=0x%lx\n",
                (unsigned long)(uintptr_t)mglRendererBackendGetDevice(
                    areas.backend),
                (unsigned long)(uintptr_t)mglRendererBackendGetCommandQueue(
                    areas.backend));
        if (s_pdCorruptionRecoveryCount < kMglPdMaxRecoveryAttempts) {
            fprintf(stderr,
                    "MGL CRITICAL: Attempting Metal state recovery (%d/%d)\n",
                    s_pdCorruptionRecoveryCount + 1, kMglPdMaxRecoveryAttempts);
            if (!mglPlatformShellGuardedCall(renderer, "metal state recovery",
                                             mglPdRecoveryTryBody)) {
                return 0;
            }
            deviceOk = mglRendererBackendGetDevice(areas.backend) &&
                       ((uintptr_t)mglRendererBackendGetDevice(areas.backend) >=
                        0x1000);
            queueOk =
                mglRendererBackendGetCommandQueue(areas.backend) &&
                ((uintptr_t)mglRendererBackendGetCommandQueue(areas.backend) >=
                 0x1000);
            if (!deviceOk || !queueOk) {
                fprintf(stderr,
                        "MGL CRITICAL: Metal recovery failed, aborting operation\n");
                return 0;
            }
        } else {
            fprintf(stderr,
                    "MGL CRITICAL: Maximum recovery attempts exceeded, permanently disabling Metal operations\n");
            return 0;
        }
    }

    int quarantineBlocks = 0;
    if (draw_command) {
        const GLuint blockedProgramKey = mglCurrentRenderProgramKey(ctx);
        if (blockedProgramKey != 0u &&
            areas.gpu_interface_mismatch_blocked_program != 0 &&
            blockedProgramKey == areas.gpu_interface_mismatch_blocked_program) {
            const double now = CFAbsoluteTimeGetCurrent();
            if (now < areas.gpu_interface_mismatch_blocked_until) {
                quarantineBlocks = 1;
                s_pdQuarantineSkipCount++;
                if (s_pdQuarantineSkipCount <= 16 ||
                    (s_pdQuarantineSkipCount % 1000) == 0) {
                    double remaining =
                        areas.gpu_interface_mismatch_blocked_until - now;
                    if (remaining < 0.0) remaining = 0.0;
                    fprintf(stderr,
                            "MGL WARNING: Program %u quarantined due to interface mismatch (%.2fs remaining), skipping draw\n",
                            (unsigned)areas
                                .gpu_interface_mismatch_blocked_program,
                            remaining);
                }
            }
        }
    }

    MGLRenderCommandBufferState processCommandState = {0};
    const int processHasCommand =
        mglRenderGetCommandBufferOwnerState(
            commandState->currentCommandBufferOwner,
            &processCommandState) == 0;
    const int encoderCurrent =
        mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1
            ? 1
            : 0;

    MGLProcessGLStateInputs planIn = {0};
    planIn.has_ctx = 1u;
    planIn.draw_command = draw_command ? 1u : 0u;
    planIn.has_vao = glState->vao != NULL ? 1u : 0u;
    planIn.dirty_state = (glState->dirty_bits & DIRTY_STATE) ? 1u : 0u;
    planIn.ctx_ptr_sane = 1u;
    planIn.device_ok = deviceOk ? 1u : 0u;
    planIn.queue_ok = queueOk ? 1u : 0u;
    planIn.quarantine_blocks_draw = quarantineBlocks ? 1u : 0u;
    planIn.has_command_buffer = processHasCommand ? 1u : 0u;
    planIn.encoder_has_current = encoderCurrent ? 1u : 0u;
    planIn.command_buffer_status =
        processHasCommand ? (uint32_t)processCommandState.status : 0u;

    MGLProcessGLStatePlan plan = {0};
    if (mglRenderProcessGLState(&planIn, &plan) != 0) {
        return 0;
    }

    if (plan.clear_rt_sampled_copy) {
        /* This flag is derived from the current draw's final fragment sampler
         * binding.  Clear it before any early render-state refresh so the
         * previous draw cannot disable culling while DIRTY_VAO/FBO is handled. */
        mglPassManagerSetCurrentDrawUsesRTSampledCopy(manager, 0);
        MGL_FRAME_INC(g_mglProcessDrawCallsSinceSwap);
    }

    if (plan.result == MGL_PGL_RESULT_ABORT) {
        if (draw_command && !planIn.has_vao &&
            plan.process_class == MGL_PROCESS_GL_ABORT) {
            fprintf(stderr, "Error: No VAO defined for ctx\n\n");
        }
        return 0;
    }

    if (plan.non_draw_end_pass_if_fbo_changed) {
        mglRenderPassEndIfFramebufferChangedForNonDraw(renderer,
                                                                   processCall);
    }
    if (plan.no_vao_clear_path) {
        mglRendererEndRenderEncodingLocked(renderer);
        if (!mglRendererValidateMetalObjects(renderer)) {
            fprintf(stderr,
                    "MGL WARNING: GPU throttling active - deferring render encoder creation\n");
            glState->dirty_bits &= ~DIRTY_STATE;
            return 1;
        }
        (void)mglPlatformShellGuardedCall(renderer, "clear-path render encoder",
                                          mglPdNoVaoEncoderTryBody);
        glState->dirty_bits &= ~DIRTY_STATE;
        return 1;
    }
    if (plan.result == MGL_PGL_RESULT_EARLY_OK) {
        return 1;
    }

    if (plan.rotate_finalized_command_buffer) {
        const uint64_t rotateHit = ++s_pdRotateFinalizedCount;
        if (rotateHit <= 16ull || (rotateHit % 500ull) == 0ull) {
            fprintf(stderr,
                    "MGL INFO: processGLState rotating finalized command buffer (status: %ld) hit=%llu\n",
                    (long)planIn.command_buffer_status,
                    (unsigned long long)rotateHit);
        }
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: processGLState failed to create a fresh command buffer\n");
            if (traceProcess) {
                mglLogStateSnapshot(
                    "processGLState.fail.new_cb_rotate", ctx,
                    commandState->currentCommandBufferOwner,
                    commandState->currentRenderEncoderOwner,
                    commandState->renderPassStateOwner, areas.drawable);
            }
            return 0;
        }
    } else if (plan.create_initial_command_buffer) {
        if (kMglPdVerboseFrameLoopLogs) {
            fprintf(stderr,
                    "MGL INFO: processGLState found NULL command buffer, creating one\n");
        }
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: processGLState could not create initial command buffer\n");
            if (traceProcess) {
                mglLogStateSnapshot(
                    "processGLState.fail.new_cb_initial", ctx,
                    commandState->currentCommandBufferOwner,
                    commandState->currentRenderEncoderOwner,
                    commandState->renderPassStateOwner, areas.drawable);
            }
            return 0;
        }
    }

    MGLResourceSyncWork resourceSyncWork = {false, false, false};
    if (plan.process_dirty_domains) {
        if (!mglRenderPassProcessDirtyStateDomains(renderer,
                                                   draw_command ? 1 : 0,
                                                   &resourceSyncWork)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
    }

    /* Phase 2: re-sample encoder/pipeline after dirty-domain materialization. */
    Program *fragmentProgram =
        mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
    MGLProcessGLStateAfterInputs afterIn = {0};
    afterIn.draw_command = draw_command ? 1u : 0u;
    afterIn.encoder_has_current =
        mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1
            ? 1u
            : 0u;
    afterIn.has_pipeline_state =
        (areas.pipeline_cache && areas.pipeline_cache->pipelineState) ? 1u : 0u;
    afterIn.frag_needs_fragcoord =
        fragmentProgram && mglRenderSamplerUnitExplicit(
                               (uint32_t)fragmentProgram->usesFragCoordParams)
            ? 1u
            : 0u;
    afterIn.frag_needs_sample =
        ((fragmentProgram && mglRenderSamplerUnitExplicit(
                                 (uint32_t)fragmentProgram->uses_sample_params)) ||
         mglPlatformShellMSSampleInLoop(renderer))
            ? 1u
            : 0u;
    afterIn.frag_needs_lod_bias =
        fragmentProgram && mglRenderSamplerUnitExplicit(
                               (uint32_t)fragmentProgram->uses_lod_bias)
            ? 1u
            : 0u;
    afterIn.fragment_trace_uses_rt_sampled_copy =
        mglFragmentTextureTraceBindingsUseRTSampledCopy(
            areas.fragment_trace_bindings, TEXTURE_UNITS)
            ? 1u
            : 0u;

    MGLProcessGLStateAfterPlan after = {0};
    if (mglRenderProcessGLStateAfterDirty(&afterIn, &after) != 0) {
        return 0;
    }

    if (after.recover_nil_encoder) {
        const uint64_t nilHit = ++s_pdNilEncoderRecoveryCount;
        if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
            fprintf(stderr,
                    "MGL WARNING: processGLState - current render encoder is nil, attempting recovery hit=%llu\n",
                    (unsigned long long)nilHit);
            mglLogRenderPassLifecycle(
                "nil-encoder-before-recovery", nilHit, ctx,
                commandState->currentCommandBufferOwner,
                commandState->currentRenderEncoderOwner,
                commandState->renderPassStateOwner, areas.drawable,
                commandState->renderPassFramebuffer,
                commandState->renderPassFramebufferName,
                commandState->renderPassDrawBuffer,
                commandState->renderPassDrawBufferCount);
        }
        if (!mglRenderPassNewRenderEncoderLockedWithReason(
                renderer, MGL_ENC_REASON_NIL)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
        if (nilHit <= 16ull || (nilHit % 2048ull) == 0ull) {
            mglLogRenderPassLifecycle(
                "nil-encoder-after-recovery", nilHit, ctx,
                commandState->currentCommandBufferOwner,
                commandState->currentRenderEncoderOwner,
                commandState->renderPassStateOwner, areas.drawable,
                commandState->renderPassFramebuffer,
                commandState->renderPassFramebufferName,
                commandState->renderPassDrawBuffer,
                commandState->renderPassDrawBufferCount);
        }
    }

    if (after.ensure_pass_matches_fbo) {
        if (!mglRenderPassEnsureCurrentRenderPassMatchesFramebufferForDraw(
                renderer)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
        mglRenderPassUpdateCurrentRenderEncoder(renderer);
    }

    if (draw_command && kMglPdVerbosePipelineLogs) {
        s_pdDrawPipelineLookupCount++;
        if (s_pdDrawPipelineLookupCount <= 256ull ||
            (s_pdDrawPipelineLookupCount % 1000ull) == 0ull) {
            Program *lookupProgram = mglResolveProgramFromState(ctx);
            Program *lookupVertexProgram =
                mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
            Program *lookupFragmentProgram =
                mglResolveProgramForStageFromState(ctx, _FRAGMENT_SHADER);
            const GLuint lookupProgramName = mglCurrentRenderProgramKey(ctx);
            Framebuffer *lookupFBO = glState->framebuffer;
            const GLuint lookupFBOName = lookupFBO ? lookupFBO->name : 0;
            fprintf(stderr, "MGL Draw current program key=%u mono=%p vs=%u fs=%u\n",
                    (unsigned)lookupProgramName, (void *)lookupProgram,
                    lookupVertexProgram ? (unsigned)lookupVertexProgram->name
                                        : 0u,
                    lookupFragmentProgram
                        ? (unsigned)lookupFragmentProgram->name
                        : 0u);
            fprintf(stderr,
                    "MGL DRAW pipeline lookup result=%p key=%u vs=%u fs=%u vao=%p fbo=%u\n",
                    areas.pipeline_cache ? areas.pipeline_cache->pipelineState
                                         : NULL,
                    (unsigned)lookupProgramName,
                    lookupVertexProgram ? (unsigned)lookupVertexProgram->name
                                        : 0u,
                    lookupFragmentProgram
                        ? (unsigned)lookupFragmentProgram->name
                        : 0u,
                    glState->vao, (unsigned)lookupFBOName);
        }
    }

    if (after.fail_nil_pipeline) {
        s_pdNilPipelineCount++;
        if (s_pdNilPipelineCount <= 8 || (s_pdNilPipelineCount % 1000) == 0) {
            mglTraceLog(
                "MGL DRAW SKIP: pipelineState is nil, forcing rebuild (occurrence=%llu)",
                (unsigned long long)s_pdNilPipelineCount);
        }
        mglMarkRendererDirtyBits(ctx->active_state,
                                 DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                     DIRTY_RENDER_STATE);
        if (traceProcess) {
            mglLogStateSnapshot("processGLState.fail.nil_pipeline", ctx,
                                commandState->currentCommandBufferOwner,
                                commandState->currentRenderEncoderOwner,
                                commandState->renderPassStateOwner,
                                areas.drawable);
        }
        return 0;
    }

    if (after.validate_attachments) {
        if (!mglRenderPassValidateAttachmentsAndPipelineFormats(
                renderer, traceProcess ? 1 : 0)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
    }

    if (after.set_pipeline) {
        MglPdSetPipelineCtx setCtx = {&areas};
        if (!mglPlatformShellGuardedCallCtx(renderer, "set render pipeline state",
                                            mglPdSetPipelineTryBody, &setCtx,
                                            NULL)) {
            /* @catch (NSException *exception) */
            fprintf(stderr,
                    "MGL ERROR: processGLState - setRenderPipelineState failed\n");
            mglMarkRendererDirtyBits(ctx->active_state,
                                     DIRTY_PROGRAM | DIRTY_VAO | DIRTY_FBO |
                                         DIRTY_RENDER_STATE);
            if (traceProcess) {
                mglLogStateSnapshot("processGLState.fail.set_pipeline", ctx,
                                    commandState->currentCommandBufferOwner,
                                    commandState->currentRenderEncoderOwner,
                                    commandState->renderPassStateOwner,
                                    areas.drawable);
            }
            return 0;
        }
    }

    if (after.sync_resources) {
        if (!mglRendererSyncResourceBindingsForContext(renderer, ctx,
                                                       &resourceSyncWork)) {
            fprintf(stderr, "failure %s:%d\n", __func__, __LINE__);
            return 0;
        }
    }

    if (after.bind_frag_coord_slot) {
        const int useFragCoordParams = afterIn.frag_needs_fragcoord ? 1 : 0;
        const int useSampleParams = afterIn.frag_needs_sample ? 1 : 0;
        uint64_t passHeight = mglPdRenderTargetHeightFor(commandState);
        if (passHeight == 0) {
            for (int i = 0; i < MAX_COLOR_ATTACHMENTS && passHeight == 0; i++) {
                void *color = mglPdColorTextureFor(commandState, (size_t)i);
                passHeight = color ? mglPdTextureInfo(color).height : 0;
            }
            if (passHeight == 0 && mglPdDepthTextureFor(commandState)) {
                passHeight =
                    mglPdTextureInfo(mglPdDepthTextureFor(commandState)).height;
            }
            if (passHeight == 0 && mglPdStencilTextureFor(commandState)) {
                passHeight = mglPdTextureInfo(mglPdStencilTextureFor(commandState))
                                 .height;
            }
        }

        uint32_t numSamples = 1;
        uint32_t sampleBuffers = 0;
        Framebuffer *fbo = glState->framebuffer;
        if (fbo && (fbo->color_attachment_bitfield & 1u)) {
            FBOAttachment *att = &fbo->color_attachments[0];
            Texture *tex = NULL;
            if (mglRenderTargetIsRenderbuffer((uint32_t)att->textarget) &&
                att->buf.rbo) {
                tex = att->buf.rbo->tex;
            } else {
                tex = att->buf.tex;
            }
            if (tex) {
                (void)mglRenderTextureSampleParams(
                    (uint32_t)tex->target, tex->samples, &numSamples,
                    &sampleBuffers);
            }
        } else {
            void *rpColor0 = mglPdColorTextureFor(commandState, 0);
            if (rpColor0) {
                const uint64_t sc = mglPdTextureInfo(rpColor0).sample_count;
                if (sc > 1) {
                    sampleBuffers = 1;
                    numSamples = (uint32_t)sc;
                }
            }
        }
        float fragCoordParams[4] = {0.f, 0.f, 0.f, 0.f};
        mglRenderFillFragCoordSlot(
            useFragCoordParams, useSampleParams, (uint32_t)passHeight,
            mglRenderClipOriginIsLowerLeft(
                (uint32_t)glState->var.clip_origin),
            numSamples, sampleBuffers, mglPlatformShellMSSampleInLoop(renderer),
            (uint32_t)areas.mssample_forced_id, fragCoordParams);
        mglRenderSetRenderBytesForOwner(
            commandState->currentRenderEncoderOwner, fragCoordParams,
            sizeof(fragCoordParams), MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLFragCoordParamsBufferIndex);
        mglBindingInvalidateLastBoundFragmentBufferAtIndex(
            renderer, kMGLFragCoordParamsBufferIndex);
    }

    if (after.bind_lod_bias_slot) {
        const GLfloat biasmax = ctx->active_state->var.max_texture_lod_bias;
        float lodBiasArr[TEXTURE_UNITS];
        for (GLuint unit = 0; unit < TEXTURE_UNITS; unit++) {
            Texture *tex = glState->active_textures[unit];
            Sampler *smp = glState->texture_samplers[unit];

            lodBiasArr[unit] = smp ? smp->params.lod_bias
                                   : (tex ? tex->params.lod_bias : 0.0f);
        }
        mglRenderClampLodBiasArray(lodBiasArr, TEXTURE_UNITS, biasmax);
        mglRenderSetRenderBytesForOwner(
            commandState->currentRenderEncoderOwner, lodBiasArr,
            sizeof(lodBiasArr), MGL_RENDER_BINDING_STAGE_FRAGMENT,
            kMGLLodBiasBufferIndex);
        mglBindingInvalidateLastBoundFragmentBufferAtIndex(
            renderer, kMGLLodBiasBufferIndex);

        mglRenderSetRenderBytesForOwner(
            commandState->currentRenderEncoderOwner, &biasmax, sizeof(biasmax),
            MGL_RENDER_BINDING_STAGE_FRAGMENT, kMGLLodBiasMaxBufferIndex);
        mglBindingInvalidateLastBoundFragmentBufferAtIndex(
            renderer, kMGLLodBiasMaxBufferIndex);
    }

    if (after.maybe_mark_rt_sampled_copy) {
        mglPassManagerSetCurrentDrawUsesRTSampledCopy(manager, 1);
        mglRenderPassUpdateCurrentRenderEncoder(renderer);
    }

    const double processElapsedUs = (mglTraceClockNS() - processStartNS) / 1000.0;
    if (traceProcess) {
        mglTraceLog("MGL TRACE processGLState.end call=%llu draw=%d elapsed=%.1fus",
                    (unsigned long long)processCall, draw_command ? 1 : 0,
                    processElapsedUs);
        mglLogStateSnapshot("processGLState.exit.ok", ctx,
                            commandState->currentCommandBufferOwner,
                            commandState->currentRenderEncoderOwner,
                            commandState->renderPassStateOwner, areas.drawable);
    } else if (processElapsedUs >= 25.0) {
        mglTraceLog("MGL TRACE processGLState.slow call=%llu draw=%d elapsed=%.1fus",
                    (unsigned long long)processCall, draw_command ? 1 : 0,
                    processElapsedUs);
    }
    return 1;
}

/* === mtlSwapBuffersLocked (log 180) ==================================== */

/* Shell forwarders for swap-path state that must be computed on the shell
 * object (the interval and the layer travel in the areas instead). */
extern int mglPlatformShellShouldSkipPresentForUnlockedSwap(void *renderer);
extern MGLSizeValue mglPlatformShellApplyPendingDrawableSize(void *renderer);
extern void *mglPlatformShellDrawablePointer(void *renderer);

/* The .m's file-local constants this TU needs (values copied verbatim from
 * MGLRenderer.m's enum, as mgl_renderer_host.c already does). */
enum {
    MGL_PD_CB_NOT_ENQUEUED = 0u,
    MGL_PD_CB_ERROR = 5u,
};

/* MGLRenderer.m had this as a static inline. */static bool mglPdContextLikelyValid(GLMContext ctx)
{
    return (ctx != NULL) && ((uintptr_t)ctx >= 0x10000u);
}

/* Moved out of MGLRenderer.m (it was that file's static, used only here). */
static void mglPdRecordFrameCommandBufferCompleted(
    void *context, const MGLRenderCommandBufferState *state)
{
    (void)state;
    mglRecordFrameCompleted((uint64_t)(uintptr_t)context);
}

/* File-scope twins of the .m's function-local statics. */
static uint64_t s_pdSwapCallCount = 0;
static double s_pdSwapLastCallTime = 0.0;
static uint64_t s_pdSwapLastCallCount = 0;
static volatile double s_pdMainThreadHeartbeatSeconds = 0.0;
static volatile uint64_t s_pdMainThreadPingCount = 0;
static uint64_t s_pdSwapProcessStateFailCount = 0;
static uint64_t s_pdSwapFinalizedBufferCount = 0;

/* dispatch_async(dispatch_get_main_queue(), ^{ ... }) became dispatch_async_f:
 * the block only touched these two statics, so no context is needed. */
static void mglPdMainThreadPing(void *unused)
{
    (void)unused;
    s_pdMainThreadHeartbeatSeconds = mglTraceNowSeconds();
    s_pdMainThreadPingCount++;
}

typedef struct MglPdPresentCtx_t {
    void *command_state_owner;
    void *drawable;
    int result;
} MglPdPresentCtx;

/* @try of the drawable presentation.  -1 = in-body failure (already logged and
 * the caller just returns), 0 = exception (the caller runs the @catch block). */
static int mglPdPresentTryBody(void *renderer, void *rawCtx)
{
    MglPdPresentCtx *ctx = (MglPdPresentCtx *)rawCtx;
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *commandState = areas.command;
    GLMContext glmCtx = areas.ctx;

    if (!mglRendererDrawableTexturePort(renderer)) {
        fprintf(stderr,
                "MGL ERROR: Drawable texture is NULL, cannot present\n");
        ctx->result = -1;
        return 0;
    }

    void *currentDrawableTexture = mglRendererDrawableTexturePort(renderer);
    MGLRenderTextureInfo currentDrawableInfo =
        mglPdTextureInfo(currentDrawableTexture);
    if (currentDrawableInfo.width == 0 || currentDrawableInfo.height == 0) {
        fprintf(stderr, "MGL ERROR: Drawable has invalid dimensions: %dx%d\n",
                (int)currentDrawableInfo.width,
                (int)currentDrawableInfo.height);
        ctx->result = -1;
        return 0;
    }

    if (kMglPdVerboseFrameLoopLogs) {
        fprintf(stderr,
                "MGL INFO: Presenting drawable with texture: %dx%d, format: %lu\n",
                (int)currentDrawableInfo.width,
                (int)currentDrawableInfo.height,
                (unsigned long)currentDrawableInfo.pixel_format);
    }

    if (mglRenderPresentDrawableForCommandBufferOwner(
            commandState->currentCommandBufferOwner, ctx->drawable, NULL) != 0) {
        fprintf(stderr,
                "MGL ERROR: No command buffer available for drawable presentation\n");
        ctx->result = -1;
        return 0;
    }
    (void)glmCtx;
    ctx->result = 1;
    return 1;
}

typedef struct MglPdCommitSwapCtx_t {
    MGLRendererStateAreas *areas;
    void *command_buffer;
    uint64_t committed_generation;
    uint64_t swap_call;
    int trace_swap;
} MglPdCommitSwapCtx;

/* @try of the frame commit; the catch records a GPU error. */
static int mglPdCommitSwapTryBody(void *renderer, void *rawCtx)
{
    MglPdCommitSwapCtx *ctx = (MglPdCommitSwapCtx *)rawCtx;
    MGLCommandState *commandState = ctx->areas->command;

    if (ctx->trace_swap) {
        char commandBufferLabel[256] = {0};
        if (ctx->command_buffer) {
            (void)mglRenderGetCommandBufferLabel(
                ctx->command_buffer, commandBufferLabel,
                sizeof(commandBufferLabel));
        }
        mglTraceLog(
            "MGL TRACE swap.commit.begin call=%llu cb=%p status=%s label=%s",
            (unsigned long long)ctx->swap_call, ctx->command_buffer,
            mglCommandBufferStatusName(
                ctx->command_buffer
                    ? (uint32_t)mglRenderCommandBufferStatus(ctx->command_buffer)
                    : MGL_PD_CB_ERROR),
            commandBufferLabel[0] ? commandBufferLabel : "(nil)");
    }
    /* Register the frame-completion handler BEFORE commit:
     * commitCommandBufferWithAGXRecovery: commits the CB, and Metal asserts if
     * addCompletedHandler: is called after commit. */
    if (ctx->command_buffer) {
        const int completionResult = mglRenderAddCommandBufferCompletion(
            ctx->command_buffer, mglPdRecordFrameCommandBufferCompleted,
            (void *)(uintptr_t)ctx->committed_generation, NULL);
        if (completionResult != 0) {
            fprintf(stderr,
                    "MGL ERROR: Failed to register C++ frame completion handler\n");
        }
    }
    mglRendererCommitCommandBufferWithAGXRecovery(renderer,
                                                  ctx->command_buffer);
    if (ctx->trace_swap) {
        mglTraceLog("MGL TRACE swap.commit.end call=%llu",
                    (unsigned long long)ctx->swap_call);
    }
    (void)commandState;
    return 1;
}

/* -mtlSwapBuffersLocked:. */
void mglRenderPassMTLSwapBuffersLocked(void *renderer, GLMContext glm_ctx)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLRenderPassManager *manager = areas.render_pass_manager;
    MGLCommandState *commandState = areas.command;

    const uint64_t swapCall = ++s_pdSwapCallCount;
    const double swapStartSeconds = mglTraceNowSeconds();
    const uint64_t swapStartNS = mglTraceClockNS();
    const bool traceSwap = mglPdShouldTraceCall(swapCall);
    mglTraceNoteFrameBoundary();
    MGL_FRAME_STORE(g_mglSwapCallCount, swapCall);
    /* advance the DontCare frame generation. Any color attachment written
     * before this point belongs to the previous frame, so its next write this
     * frame is a "first use" that may skip loading prior contents.  Skips 0 so
     * a zero-initialized texture stamp never matches. */
    mglPassManagerIncrementDontCareFrameGenerationWithWrap(manager);
    MGL_FRAME_STORE(g_mglLastSwapSeconds, swapStartSeconds);
    if (swapCall <= 20ull || (swapCall % 60ull) == 0ull) {
        mglTraceLog(
            "SWAP_RENDERER_ENTRY call=%llu drawArraysSinceSwap=%llu drawElementsSinceSwap=%llu processDrawCallsSinceSwap=%llu",
            (unsigned long long)swapCall,
            (unsigned long long)MGL_FRAME_LOAD(g_mglDrawArraysSinceSwap),
            (unsigned long long)MGL_FRAME_LOAD(g_mglDrawElementsSinceSwap),
            (unsigned long long)MGL_FRAME_LOAD(g_mglProcessDrawCallsSinceSwap));
    }
    mglLogLoopHeartbeat("swap.loop", swapCall, swapStartSeconds,
                        &s_pdSwapLastCallTime, &s_pdSwapLastCallCount, 0.25);

    if (!mglPdContextLikelyValid(glm_ctx)) {
        fprintf(stderr, "MGL CRITICAL: swap.begin invalid glm_ctx=%p\n",
                (void *)glm_ctx);
        return;
    }

    if (areas.ctx != glm_ctx) {
        mglTraceLog("MGL TRACE swap.contextSync old=%p new=%p", (void *)areas.ctx,
                    (void *)glm_ctx);
        mglPlatformShellSetContext(renderer, glm_ctx);
    }
    /* The .m rebinds its `ctx` ivar; C keeps the fresh value in the snapshot and
     * in `activeCtx`. */
    areas.ctx = glm_ctx;
    GLMContext activeCtx = glm_ctx;
    const GLenum drawBuffer = activeCtx->state.draw_buffer;
    const bool shouldPresent =
        mglRenderShouldPresentDrawBuffer((uint32_t)drawBuffer) != 0;
    if (traceSwap) {
        mglTraceLog("MGL TRACE swap.begin call=%llu shouldPresent=%d draw_buffer=0x%x",
                    (unsigned long long)swapCall, shouldPresent ? 1 : 0,
                    (unsigned)drawBuffer);
        mglLogStateSnapshot("swap.enter", activeCtx,
                            commandState->currentCommandBufferOwner,
                            commandState->currentRenderEncoderOwner,
                            commandState->renderPassStateOwner, areas.drawable);
    }

    /* Main-thread responsiveness probe for beachball diagnostics.  Render thread
     * periodically posts a ping to main queue; stale heartbeat means main thread
     * is blocked. */
    if (kMGLDiagnosticStateLogs &&
        (swapCall <= 20ull || (swapCall % 30ull) == 0ull)) {
        dispatch_async_f(dispatch_get_main_queue(), NULL, mglPdMainThreadPing);

        const double hb = s_pdMainThreadHeartbeatSeconds;
        if (hb > 0.0) {
            const double lagMs = (swapStartSeconds - hb) * 1000.0;
            if (lagMs > 500.0) {
                mglTraceLog(
                    "MGL TRACE mainthread.stall suspected lag=%.2fms swapCall=%llu pingCount=%llu",
                    lagMs, (unsigned long long)swapCall,
                    (unsigned long long)s_pdMainThreadPingCount);
                if (traceSwap || (swapCall % 120ull) == 0ull) {
                    mglLogStateSnapshot("mainthread.stall.snapshot", activeCtx,
                                        commandState->currentCommandBufferOwner,
                                        commandState->currentRenderEncoderOwner,
                                        commandState->renderPassStateOwner,
                                        areas.drawable);
                }
            } else if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE mainthread.heartbeat lag=%.2fms swapCall=%llu pingCount=%llu",
                    lagMs, (unsigned long long)swapCall,
                    (unsigned long long)s_pdMainThreadPingCount);
            }
        } else if (traceSwap) {
            mglTraceLog(
                "MGL TRACE mainthread.heartbeat uninitialized swapCall=%llu",
                (unsigned long long)swapCall);
        }
    }

    if (kMGLDiagnosticStateLogs) {
        MGLSwapDrawCounters frameCounters = mglSnapshotSwapDrawCounters();
        mglResetSwapDrawCounters();

        const uint64_t lastDrawArraysCall =
            MGL_FRAME_LOAD(g_mglLastDrawArraysCall);
        const uint64_t lastDrawElementsCall =
            MGL_FRAME_LOAD(g_mglLastDrawElementsCall);
        const double lastDrawArraysSeconds =
            MGL_FRAME_LOAD(g_mglLastDrawArraysSeconds);
        const double lastDrawElementsSeconds =
            MGL_FRAME_LOAD(g_mglLastDrawElementsSeconds);
        const GLuint lastDrawArraysProgram =
            MGL_FRAME_LOAD(g_mglLastDrawArraysProgram);
        const GLuint lastDrawArraysMode =
            MGL_FRAME_LOAD(g_mglLastDrawArraysMode);
        const GLsizei lastDrawArraysCount =
            MGL_FRAME_LOAD(g_mglLastDrawArraysCount);
        const GLuint lastDrawElementsProgram =
            MGL_FRAME_LOAD(g_mglLastDrawElementsProgram);
        const GLuint lastDrawElementsMode =
            MGL_FRAME_LOAD(g_mglLastDrawElementsMode);
        const GLsizei lastDrawElementsCount =
            MGL_FRAME_LOAD(g_mglLastDrawElementsCount);
        const double drawArraysAgeMs =
            (lastDrawArraysSeconds > 0.0)
                ? ((swapStartSeconds - lastDrawArraysSeconds) * 1000.0)
                : -1.0;
        const double drawElementsAgeMs =
            (lastDrawElementsSeconds > 0.0)
                ? ((swapStartSeconds - lastDrawElementsSeconds) * 1000.0)
                : -1.0;
        const int hasFrameWork =
            (frameCounters.draw_arrays > 0 ||
             frameCounters.draw_elements > 0 ||
             frameCounters.draw_arrays_skipped > 0 ||
             frameCounters.draw_elements_skipped > 0 ||
             frameCounters.process_draw_calls > 0);
        if (traceSwap || hasFrameWork || swapCall <= 20ull ||
            (swapCall % 20ull) == 0ull) {
            mglTraceLog(
                "MGL TRACE swap.drawActivity call=%llu processDrawCalls=%llu drawArrays=%llu verts=%llu "
                "drawElements=%llu indices=%llu skipArrays=%llu skipElements=%llu "
                "lastDrawArrays=%llu prog=%u mode=0x%x count=%d age=%.2fms "
                "lastDrawElements=%llu prog=%u mode=0x%x count=%d age=%.2fms",
                (unsigned long long)swapCall,
                (unsigned long long)frameCounters.process_draw_calls,
                (unsigned long long)frameCounters.draw_arrays,
                (unsigned long long)frameCounters.array_vertices,
                (unsigned long long)frameCounters.draw_elements,
                (unsigned long long)frameCounters.element_indices,
                (unsigned long long)frameCounters.draw_arrays_skipped,
                (unsigned long long)frameCounters.draw_elements_skipped,
                (unsigned long long)lastDrawArraysCall,
                (unsigned)lastDrawArraysProgram, (unsigned)lastDrawArraysMode,
                (int)lastDrawArraysCount, drawArraysAgeMs,
                (unsigned long long)lastDrawElementsCall,
                (unsigned)lastDrawElementsProgram, (unsigned)lastDrawElementsMode,
                (int)lastDrawElementsCount, drawElementsAgeMs);
        }
    }

    if (shouldPresent) {
        mglRendererFlushDrawBufferLockedPort(renderer, activeCtx);

        if (!mglRenderPassProcessGLStateLocked(renderer, 0)) {
            s_pdSwapProcessStateFailCount++;
            if (s_pdSwapProcessStateFailCount <= 16 ||
                (s_pdSwapProcessStateFailCount % 500) == 0) {
                fprintf(stderr,
                        "MGL WARNING: mtlSwapBuffers continuing despite processGLState failure (occurrence=%llu)\n",
                        (unsigned long long)s_pdSwapProcessStateFailCount);
            }
        }

        mglRendererEndRenderEncodingLocked(renderer);

        /* Deferred device reset drain.  This is the only safe reset point: the
         * render encoder is closed and the command buffer has not been rebuilt
         * yet, so resetMetalState can swap the command queue / clear caches
         * without racing an active encoder.  The request flag is set by the
         * Metal completion handler (GPURecovery.m) via release-store. */
        /* Completion workers latch recovery requests in the C++ owner on both
         * gates; consume them only at this GL-thread frame boundary. */
        if (areas.gpu_recovery_command_owner &&
            mglRenderCommandRecoveryTakeResetRequest(
                *areas.gpu_recovery_command_owner) == 1) {
            atomic_store_explicit(&areas.core->deviceResetRequested, true,
                                  memory_order_release);
        }
        if (atomic_exchange_explicit(&areas.core->deviceResetRequested, false,
                                     memory_order_acquire)) {
            mglRendererResetMetalState(renderer);
        }

        if (!mglRenderPassEnsureWritableCommandBufferLocked(
                renderer, "mtlSwapBuffers")) {
            fprintf(stderr,
                    "MGL ERROR: Failed to obtain writable command buffer in mtlSwapBuffers\n");
            return;
        }

        const int swapInterval = areas.swap_interval;
        const int skipPresent =
            (swapInterval == 0) &&
            mglPlatformShellShouldSkipPresentForUnlockedSwap(renderer);

        if (mglPlatformShellDrawablePointer(renderer) == NULL) {
            if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.begin call=%llu stage=pre_present",
                    (unsigned long long)swapCall);
            }
            (void)mglPlatformShellApplyPendingDrawableSize(renderer);
            (void)mglRendererNextDrawablePort(renderer);
            if (traceSwap) {
                void *tex = mglRendererDrawableTexturePort(renderer);
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.end call=%llu stage=pre_present drawable=%p tex=%p size=%lux%lu",
                    (unsigned long long)swapCall,
                    mglPlatformShellDrawablePointer(renderer), tex,
                    (unsigned long)(tex ? mglPdTextureInfo(tex).width : 0),
                    (unsigned long)(tex ? mglPdTextureInfo(tex).height : 0));
            }
        }

        if (mglPlatformShellDrawablePointer(renderer) == NULL) {
            fprintf(stderr,
                    "MGL WARNING: Drawable is NULL in mtlSwapBuffers, getting new drawable\n");
            if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.begin call=%llu stage=pre_present_retry",
                    (unsigned long long)swapCall);
            }
            (void)mglPlatformShellApplyPendingDrawableSize(renderer);
            (void)mglRendererNextDrawablePort(renderer);
            if (traceSwap) {
                void *tex = mglRendererDrawableTexturePort(renderer);
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.end call=%llu stage=pre_present_retry drawable=%p tex=%p size=%lux%lu",
                    (unsigned long long)swapCall,
                    mglPlatformShellDrawablePointer(renderer), tex,
                    (unsigned long)(tex ? mglPdTextureInfo(tex).width : 0),
                    (unsigned long)(tex ? mglPdTextureInfo(tex).height : 0));
            }
            if (mglPlatformShellDrawablePointer(renderer) == NULL) {
                fprintf(stderr,
                        "MGL ERROR: Failed to obtain any drawable from Metal layer\n");
                return;
            }
        }

        void *rpColor0 = mglRenderGetRenderPassAttachmentTextureOwner(
            commandState->renderPassStateOwner,
            MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
        void *drawableTexture = mglRendererDrawableTexturePort(renderer);
        if (!skipPresent) {
            mglSwapCopyRenderPassColorToDrawableIfNeeded(
                renderer, rpColor0, drawableTexture, swapCall, traceSwap ? 1 : 0);

            mglSwapScheduleTextureSampleDiagnostics(renderer, rpColor0,
                                                    drawableTexture, swapCall);
        }

        if (areas.layer == NULL) {
            fprintf(stderr,
                    "MGL ERROR: Metal layer is NULL, cannot present drawable\n");
            return;
        }

        MGLRenderCommandBufferState presentCommandState = {0};
        if (!mglRenderCommandBufferOwnerHasState(
                commandState->currentCommandBufferOwner,
                &presentCommandState)) {
            fprintf(stderr,
                    "MGL ERROR: No command buffer available for presentation\n");
            return;
        }

        const uint32_t bufferStatus = (uint32_t)presentCommandState.status;
        if (bufferStatus != MGL_PD_CB_NOT_ENQUEUED) {
            const uint64_t swapFinHit = ++s_pdSwapFinalizedBufferCount;
            if (swapFinHit <= 16ull || (swapFinHit % 500ull) == 0ull) {
                fprintf(stderr,
                        "MGL WARNING: mtlSwapBuffers found finalized command buffer (status: %ld), rotating (hit=%llu)\n",
                        (long)bufferStatus, (unsigned long long)swapFinHit);
            }
            mglRendererEndRenderEncodingLocked(renderer);
            (void)mglRenderPassNewCommandBufferLocked(renderer);
            if (!mglRenderCommandBufferOwnerHasState(
                    commandState->currentCommandBufferOwner,
                    &presentCommandState)) {
                fprintf(stderr,
                        "MGL ERROR: Failed to create new command buffer for presentation\n");
                return;
            }
        }

        MglPdPresentCtx presentCtx = {commandState->currentCommandBufferOwner,
                                      mglPlatformShellDrawablePointer(renderer), 0};
        if (!skipPresent) {
            if (!mglPlatformShellGuardedCallCtx(renderer, "drawable presentation",
                                                mglPdPresentTryBody, &presentCtx,
                                                NULL)) {
                if (presentCtx.result != -1) {
                    /* @catch (NSException *exception) */
                    fprintf(stderr,
                            "MGL ERROR: Critical drawable presentation failure\n");
                    (void)mglPlatformShellGuardedCall(
                        renderer, "command buffer cleanup",
                        mglRendererCleanupCommandBufferBody);
                }
                return;
            }
            if (traceSwap) {
                mglTraceLog("MGL TRACE swap.present call=%llu cbOwner=%p drawable=%p",
                            (unsigned long long)swapCall,
                            commandState->currentCommandBufferOwner,
                            presentCtx.drawable);
            }
        } else if (traceSwap) {
            mglTraceLog(
                "MGL TRACE swap.present.skipped call=%llu reason=unlocked_hidden",
                (unsigned long long)swapCall);
        }

        void *commandBufferToCommit =
            mglPassManagerDetachCurrentCommandBufferForSubmission(manager);
        const uint64_t committedGeneration = mglAdvanceFrameGeneration();
        /* Sweep the bound buffer maps so base/attrib/uniform/SSBO buffers that
         * were encoded this frame keep their pool slots pinned for the committed
         * command buffer (copy-on-write snapshot reuse). */
        BufferMapList *boundLists[3] = {
            &mglPdState(&areas)->vertex_buffer_map_list,
            &mglPdState(&areas)->fragment_buffer_map_list,
            &mglPdState(&areas)->compute_buffer_map_list,
        };
        for (int li = 0; li < 3; ++li) {
            for (GLuint mi = 0; mi < boundLists[li]->count; ++mi) {
                mglNoteBufferEncoded(boundLists[li]->buffers[mi].buf);
            }
        }
        MglPdCommitSwapCtx commitCtx = {&areas, commandBufferToCommit,
                                        committedGeneration, swapCall,
                                        traceSwap ? 1 : 0};
        if (!mglPlatformShellGuardedCallCtx(renderer, "command buffer commit",
                                            mglPdCommitSwapTryBody, &commitCtx,
                                            NULL)) {
            /* @catch (NSException *exception) */
            fprintf(stderr, "MGL ERROR: Failed to commit command buffer\n");
            mglRendererRecordGPUError(renderer);
        }

        if (traceSwap) {
            mglTraceLog(
                "MGL TRACE swap.nextDrawable.begin call=%llu stage=post_commit",
                (unsigned long long)swapCall);
        }
        if (skipPresent) {
            /* Keep the current drawable: nothing was presented, so the surface
             * remains a valid render target for the next frame. */
            if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.reuse call=%llu stage=post_commit",
                    (unsigned long long)swapCall);
            }
        } else if (swapInterval == 0) {
            /* Visible unlocked: defer acquisition off the critical path. */
            mglPlatformShellSetDrawable(renderer, NULL);
            if (traceSwap) {
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.deferred call=%llu stage=post_commit",
                    (unsigned long long)swapCall);
            }
        } else {
            (void)mglRendererNextDrawablePort(renderer);
            if (traceSwap) {
                void *tex = mglRendererDrawableTexturePort(renderer);
                mglTraceLog(
                    "MGL TRACE swap.nextDrawable.end call=%llu stage=post_commit drawable=%p tex=%p size=%lux%lu",
                    (unsigned long long)swapCall,
                    mglPlatformShellDrawablePointer(renderer), tex,
                    (unsigned long)(tex ? mglPdTextureInfo(tex).width : 0),
                    (unsigned long)(tex ? mglPdTextureInfo(tex).height : 0));
            }
            if (mglPlatformShellDrawablePointer(renderer) == NULL) {
                fprintf(stderr,
                        "MGL WARNING: Failed to get next drawable in mtlSwapBuffers\n");
                return;
            }
        }
        if (!mglRenderPassNewCommandBufferLocked(renderer)) {
            fprintf(stderr,
                    "MGL ERROR: Failed to create post-swap command buffer\n");
            return;
        }
        areas.core->defaultDrawableWrittenSinceLastSwap = 0;
        mglMarkRendererDirtyBits(areas.ctx->active_state,
                                 DIRTY_FBO | DIRTY_RENDER_STATE);
        const double swapElapsedUs = (mglTraceClockNS() - swapStartNS) / 1000.0;
        if (traceSwap) {
            mglTraceLog("MGL TRACE swap.end call=%llu elapsed=%.1fus",
                        (unsigned long long)swapCall, swapElapsedUs);
            mglLogStateSnapshot("swap.exit.ok", areas.ctx,
                                commandState->currentCommandBufferOwner,
                                commandState->currentRenderEncoderOwner,
                                commandState->renderPassStateOwner,
                                areas.drawable);
        } else if (swapElapsedUs >= 25000.0) {
            mglTraceLog("MGL TRACE swap.slow call=%llu elapsed=%.1fus",
                        (unsigned long long)swapCall, swapElapsedUs);
        }
    } else if (kMglPdVerboseFrameLoopLogs || traceSwap) {
        fprintf(stderr,
                "MGL INFO: mtlSwapBuffers skipped present because draw_buffer is GL_NONE\n");
    }

    /* Perf summary: snapshot + reset per-frame counters at the swap boundary.
     * Runs on every normal exit path (present + GL_NONE skip).  Early-return
     * error paths intentionally skip this so their counters roll into the next
     * successful frame. */
    if (mglPerfSummaryEnabled()) {
        const double now = mglTraceNowSeconds();
        static _Atomic double s_last_swap_time = 0.0;
        double interval = 0.0;
        const double prev =
            atomic_load_explicit(&s_last_swap_time, memory_order_relaxed);
        if (prev > 0.0) interval = (now - prev) * 1000.0;
        atomic_store_explicit(&s_last_swap_time, now, memory_order_relaxed);
        mglPrintPerfSummary(interval);
        mglResetPerfCounters();
    }
}
