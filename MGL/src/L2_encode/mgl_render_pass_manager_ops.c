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
#include "mgl_render_pass_ops_shared.h"
#include "mgl_state_apply.h"
#include "mgl_rt_sync.h"   /* mglRenderTargetMarkGPUAuthoritative */
#include "mgl_render_encoder_ops.h"
#include "mgl_attachment_binding.h"   /* mglRendererBindFramebufferAttachmentTextures */
#include "mgl_buffer_map.h"           /* map / dirty base buffers */
#include "mgl_texture_bind.h"          /* mglRendererBindMTLTexture */
#include "mgl_draw_buffer.h"          /* Metal draw-buffer mapping */
#include "mgl_sync.h"                 /* MGLMetalAttachmentSubresource */
#include "mgl_renderer_backend.h"     /* default draw buffer attachments */
#include "mgl_renderer_binding_ops.h"  /* mglRendererBindActiveTexturesToMTL */
#include "mgl_stage_encode_drivers.h" /* stage encode bind drivers */
#include "mgl_frame_activity.h"     /* MGL_ENC_REASON_* */
#include "mgl_render_api_query.h"  /* MGLRenderPipelineDescriptorState */
#include "mgl_render_api_lifecycle.h" /* mglRenderLoadAIRMainFunction */
#include "mgl_byte_hash.h"         /* mglHashStepU64 */
#include "mgl_draw_tess.h"         /* mglTessRasterGLMode */
#include "mgl_vertex_layout.h"     /* mglRendererGenerateVertexDescriptorState */
#include "mgl_render_pass_manager.h" /* pass-manager transaction entries */
#include "mgl_render_pass_clear.h"   /* mglRenderPassPlanClearValues */
#include "mgl_program_resource.h"   /* mglProgramStageBuiltinMask */
#include "mgl_shader_abi.h"         /* mglAIRPerVertexStrideForResources, mglShaderCompileGLSL */
#include "mgl_buffer_slots.h"       /* mglProgramStageHasSamplers (shared with the link gate) */
#include "mgl_metal_ref.h"         /* mglReleaseMetalObjNoNull */
#include "mgl_draw_gs.h"           /* mglDrawGsPassthroughDeclType */
#include "mgl_render_pass_plan.h"    /* mglRenderProcessGLState*, plans */
#include "mgl_buffer_slots.h"        /* kMGL*BufferIndex */
#include "mgl_renderer_binding_ops.h"  /* mglRendererSyncResourceBindingsForContext */
#include "mgl_swap_diagnostics.h"     /* swap colour copy + diagnostics */
#include <dispatch/dispatch.h>       /* dispatch_async_f */
#include "mgl_trace_log.h"         /* mglTraceLog, kMGLDiagnosticStateLogs */
#include "mgl_gpu_recovery.h"      /* mglRendererRecordGPUError */
#include "mgl_pso_format_class.h"  /* mglRenderDefaultColorPixelFormat */
#include "mgl_render.h"           /* attachment kinds, MS plane adjust */
#include "mgl_texture_compat.h"   /* mglMetalTextureLevelDimension */

#include "mgl_pso_build_ops.h"    /* mglRenderPassSyncPipelineState */
#include "mgl_renderer_ports.h"
#include "mgl_renderer_binding_ops.h"      /* mglRendererBindingInvalidateLastBoundState */
#include "mgl_trace_strategy.h"         /* mglClearFragmentTraceBindingsForRenderer */

#include <stdio.h>
#include <stdarg.h>   /* va_list for the GLSL source builder */
#include <stdlib.h>   /* malloc / realloc / free */
#include <string.h>   /* memcpy / strlen */
#include <math.h>     /* lrintf */
#include "mgl_readback_policy.h"
#include "mgl_program_backend.h"
#include "mgl_texture_backend.h"
#include "mgl_raster_orientation_state.h"

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
extern void *mglPlatformShellDrawablePointer(void *renderer);
extern MGLSizeValue mglPlatformShellApplyPendingDrawableSize(void *renderer);
extern void mglRendererNextDrawable(void *renderer);
extern void mglRendererPrefetchNextDrawable(void *renderer);
extern int mglPlatformShellGuardedCall(void *renderer, const char *what,
                                       int (*body)(void *));

enum { kMGLLodBiasPackHeaderFloats = 4 };

void mglFillLodBiasPack(GLMContext ctx, Program *prog, int stage,
                               float *pack /* [4 + TEXTURE_UNITS] */)
{
    GLuint slot;
    const GLfloat biasmax =
        (ctx && ctx->active_state)
            ? ctx->active_state->var.max_texture_lod_bias
            : 0.0f;
    GLint vp_w = 1, vp_h = 1;
    GLuint fb_w = 1u, fb_h = 1u;
    memset(pack, 0,
           (size_t)(kMGLLodBiasPackHeaderFloats + TEXTURE_UNITS) *
               sizeof(float));
    pack[0] = biasmax > 0.0f ? biasmax : 14.0f;
    pack[1] = 1.0f;
    pack[2] = 1.0f;
    if (ctx && ctx->active_state) {
        vp_w = ctx->active_state->viewport[2];
        vp_h = ctx->active_state->viewport[3];
        if (vp_w < 1)
            vp_w = 1;
        if (vp_h < 1)
            vp_h = 1;
        if (mglFramebufferPrimaryColorSize(ctx, ctx->active_state->framebuffer,
                                           &fb_w, &fb_h) &&
            fb_w != 0u && fb_h != 0u) {
            /* The Metal viewport is clamped to the render target, which
             * scales Metal dFdx by vp/clamped relative to GL window space. */
            double cx = ctx->active_state->viewport[0];
            double cy = ctx->active_state->viewport[1];
            double cw = vp_w, ch = vp_h;
            mglRenderClampViewport(&cx, &cy, &cw, &ch, fb_w, fb_h);
            pack[1] = (float)(cw / (double)vp_w);
            pack[2] = (float)(ch / (double)vp_h);
        }
    }
    if (!ctx || !prog)
        return;
    for (slot = 0; slot < TEXTURE_UNITS; slot++) {
        GLint unit = prog->sampler_units_by_stage[stage][slot];
        Texture *tex;
        Sampler *smp;
        if (unit < 0)
            unit = (GLint)slot;
        if (unit >= TEXTURE_UNITS)
            continue;
        smp = ctx->active_state->texture_samplers[unit];
        tex = ctx->active_state->active_textures[unit];
        pack[kMGLLodBiasPackHeaderFloats + slot] =
            smp ? smp->params.lod_bias : (tex ? tex->params.lod_bias : 0.0f);
    }
}

void mglFillSamplerWrapTable(GLMContext ctx, Program *prog, int stage,
                                    MGLSamplerWrapSlot *slots)
{
    int i;
    memset(slots, 0, sizeof(MGLSamplerWrapSlot) * MGL_SAMPLER_WRAP_SLOT_COUNT);
    for (i = 0; i < MGL_SAMPLER_WRAP_SLOT_COUNT; i++) {
        GLint unit;
        Sampler *smp;
        Texture *tex;
        TextureParameter *p;
        int c;
        slots[i].wrap_s = (int32_t)GL_REPEAT;
        slots[i].wrap_t = (int32_t)GL_REPEAT;
        slots[i].wrap_r = (int32_t)GL_REPEAT;
        if (!ctx || !prog || stage < 0 || stage >= _MAX_SHADER_TYPES)
            continue;
        unit = prog->sampler_units_by_stage[stage][i];
        if (unit < 0)
            unit = 0;
        if (unit >= TEXTURE_UNITS)
            continue;
        smp = ctx->active_state->texture_samplers[unit];
        tex = ctx->active_state->texture_units[unit].textures[_TEXTURE_2D];
        if (!tex)
            tex = ctx->active_state->texture_units[unit]
                      .textures[_TEXTURE_2D_ARRAY];
        if (!tex)
            tex = ctx->active_state->texture_units[unit].textures[_TEXTURE_3D];
        if (!tex)
            tex = ctx->active_state->active_textures[unit];
        p = smp ? &smp->params : (tex ? &tex->params : NULL);
        if (!p)
            continue;
        slots[i].wrap_s = (int32_t)p->wrap_s;
        slots[i].wrap_t = (int32_t)p->wrap_t;
        slots[i].wrap_r = (int32_t)p->wrap_r;
        for (c = 0; c < 4; c++) {
            if (p->border_color_type == GL_INT)
                slots[i].border[c] = p->border_color_i[c];
            else if (p->border_color_type == GL_UNSIGNED_INT)
                slots[i].border[c] = (int32_t)p->border_color_ui[c];
            else
                slots[i].border[c] = (int32_t)lrintf(p->border_color[c]);
        }
    }
}

/* Local twin of the manager's file-static helper of the same name. */
void mglRenderPassManagerSyncRuntimeOwners(MGLCommandState *state)
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
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs || !cs->currentRenderEncoderOwner ||
        mglRenderEncoderOwnerHasCurrent(cs->currentRenderEncoderOwner) != 1) {
        return;
    }
    (void)mglRenderEndRenderEncoderOwner(cs->currentRenderEncoderOwner);
}

void mglRenderPassManagerClearCurrentRenderEncoder(void *renderer)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
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
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
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
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
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
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
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
void mglRenderPassManagerSyncIdentityView(
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

void mglRenderPassManagerStoreIdentity(
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
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
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
int mglRendererEndRenderEncodingGuardedBody(void *renderer)
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
    mglRendererBindingInvalidateLastBoundState(renderer);

    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    MGLCommandState *cs = areas.command;
    if (!cs ||
        mglRenderEncoderOwnerHasCurrent(cs->currentRenderEncoderOwner) != 1) {
        return;
    }

    /* An active render encoder means work was encoded into the current command
     * buffer, so the flush must not skip the commit. */
    if (areas.batching) {
        mglRendererSetCommandBufferHasWork(renderer, 1);
    }

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
}

/* Declared in MGLRenderer+Draw_Private.h; a real C function in the .m. */
extern Framebuffer *mglRendererGetValidatedFramebuffer(GLMContext ctx,
                                                      const char *where);

/* === dirty state domain processing (P0-1, log 167) ====================== */

/* MGL_STATE() from MGLRenderer_Private.h, in C (same twin as mgl_tess_dispatch.c). */
GLMState *mglPdState(const MGLRendererStateAreas *areas) /* C5-EXEMPT: read helper for the write side */
{
    return areas->ctx ? areas->ctx->active_state : NULL;
}

/* Same question from the read-only view (same two fields). */
GLMState *mglPdStateView(const MGLRendererStateView *view)
{
    return view->ctx ? view->ctx->active_state : NULL;
}

/* Compatibility wrapper: apply lives in mgl_state_apply.c (ARCH-016). */
bool mglRenderPassProcessDirtyStateDomains(void *renderer, int draw_command,
                                           MGLResourceSyncWork *work)
{
    return mglStateApplyDirtyDomains(renderer, draw_command, work);
}

/* -ensureRasterEncoderForDraw (P0-1, log 168).  Replaces
 * mglRendererEnsureRasterEncoderForDrawPort. */
int mglRenderPassEnsureRasterEncoderForDraw(void *renderer)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
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

    const MGLRenderPassFormatCompatInput formatInput = {
        (uint32_t)areas.pipeline_cache->pipelineColor0Format,
        (uint32_t)areas.pipeline_cache->pipelineDepthFormat,
        (uint32_t)areas.pipeline_cache->pipelineStencilFormat,
        rpColor0Format, rpDepthFormat, rpStencilFormat};
    MGLRenderPassFormatCompat formatCompat;
    if (mglRenderPassPlanFormatCompat(&formatInput, &formatCompat) != 0 ||
        formatCompat.color_mismatch || formatCompat.depth_mismatch ||
        formatCompat.stencil_mismatch) {
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
MGLRenderPassIdentityState mglRenderPassIdentitySnapshot(
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
void *mglRenderPassDefaultDrawBufferAttachment(
    MGLRendererBackendHandle *backend, GLuint drawBufferIndex,
    MGLRendererBackendDefaultDrawBufferAttachmentKind kind)
{
    return 
        mglRendererBackendGetDefaultDrawBufferAttachment(
            backend, drawBufferIndex, kind);
}
bool mglRenderPassGetPersistentState(
    const MGLCommandState *commandState,
    MGLRenderPassState *stateOut)
{
    return commandState && stateOut && commandState->renderPassStateOwner &&
           mglRenderGetRenderPassStateOwner(
               commandState->renderPassStateOwner, stateOut) == 0;
}
const MGLRenderPassAttachmentState *
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
void *mglRenderPassTextureFromSnapshot(
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

/* Match entry for a user-framebuffer attachment.  The pass-state side comes
 * from `snapshot`, the framebuffer side from the attachment's Metal
 * subresource; the comparison rule itself is the plan's. */
void mglRenderPassFillUserFboEntry(
    MGLRenderPassAttachmentMatchEntry *entry, void *actual, void *expected,
    int gate, const MGLRenderPassAttachmentState *snapshot,
    const FBOAttachment *attachment)
{
    MGLRenderPassSubresource snapshotSub = {0u, 0u, 0u};
    MGLRenderPassSubresource fboSub = {0u, 0u, 0u};
    if (gate) {
        MGLMetalAttachmentSubresource subresource =
            mglMetalAttachmentSubresourceForAttachment(attachment);
        fboSub = (MGLRenderPassSubresource){subresource.level,
                                            subresource.slice,
                                            subresource.depthPlane};
        if (snapshot) {
            snapshotSub = (MGLRenderPassSubresource){snapshot->level,
                                                     snapshot->slice,
                                                     snapshot->depth_plane};
        }
    }
    mglRenderPassPlanFillSubresourceEntry(entry, actual, expected, gate,
                                          snapshot != NULL, snapshotSub,
                                          fboSub);
}


/* === pipeline descriptor state (log 172) =============================== */

/* pixel_utils.c defines this one; the only declaration lives in the
 * Objective-C MGLRenderer+RenderPass_Private.h. */
extern uint32_t mtlPixelFormatForGLTex(Texture *gl_tex);

/* The discard-stub factory stays in Objective-C (dispatch_once + blocks), so
 * the .m exposes this C-callable bridge. */
extern void *mglRenderPassDiscardStubFragmentFunction(uint32_t valueClass);

void *mglPdColorTextureFor(const MGLCommandState *commandState,
                                 size_t colorIndex)
{
    return mglPdAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorIndex);
}

/* Twin of the .m static: MGLStubFSValueClass is the enum
 * mglRenderMetalPixelFormatValueClass() returns. */
uint32_t mglPdStubFSValueClass(uint32_t fmt)
{
    return mglRenderMetalPixelFormatValueClass(fmt);
}

/* Twins of the .m geometry-passthrough helpers. */
uint32_t mglPdGeometryPassthroughLayerStride(GLMContext ctx)
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

uint64_t mglPdGeometryPassthroughCacheKey(const Program *program,
                                                 uint32_t layerStride)
{
    uint64_t hash = 1469598103934665603ull;
    hash = mglHashStepU64(
        hash, program ? mglProgramBackendView(program)->pipeline_cache_instance_id : 0u);
    hash = mglHashStepU64(hash,
                          program ? mglProgramBackendView(program)->pipeline_cache_generation : 0u);
    hash = mglHashStepU64(hash, layerStride);
    /* v2: layered emulated-MS PTVS adds mgl_emulated_ms_plane (Metal slot 1). */
    return mglHashStepU64(hash, layerStride > 1u ? 2u : 1u);
}

/* === new command buffer (log 174) ====================================== */

/* MGLRenderer+RenderPass_Private.h: `static const BOOL
 * kMGLDisableSharedEventSync = YES;` - kept so the branch stays verbatim. */
const int kMglPdDisableSharedEventSync = 1;

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

int mglPdNewCommandBufferTryBody(void *renderer, void *rawCtx)
{
    MglPdNewCommandBufferCtx *ctx = (MglPdNewCommandBufferCtx *)rawCtx;
    MGLRendererStateAreas *areas = ctx->areas; /* C5-EXEMPT: write side */
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

int mglPdEventWaitTryBody(void *renderer, void *rawCtx)
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
    mglRendererFillStateAreas(renderer, &areas);
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
        mglRendererSetCommandBufferHasWork(renderer, 0);
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
bool mglPdClearValuesFor(const MGLCommandState *commandState,
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

double mglPdClearDepthFor(const MGLCommandState *commandState,
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

uint32_t mglPdVisibilityResultTypeFor(
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

int mglPdCreateRenderEncoderTryBody(void *renderer, void *rawCtx)
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
    mglRendererFillStateAreas(renderer, &areas);
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
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
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

int mglPdCommitCommandBufferTryBody(void *renderer, void *rawCtx)
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
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLRenderPassManager *manager = areas.render_pass_manager;
    MGLCommandState *commandState = areas.command;

    if (!mglRendererBackendGetDevice(areas.backend) ||
        !mglRendererBackendGetCommandQueue(areas.backend)) {
        fprintf(stderr,
                "MGL ERROR: Metal device or queue is NULL in flushCommandBuffer\n");
        return;
    }

    mglRendererFlushDrawBufferLocked(renderer, ctx);

    if (!mglRenderPassProcessGLStateLocked(renderer, 0)) {
        fprintf(stderr,
                "MGL WARNING: processGLState failed in flushCommandBuffer, continuing with cleanup\n");
    }

    /* If processGLStateLocked: left a render encoder active, mark the CB as
     * having work so the commit below is not skipped. */
    if (mglRenderEncoderOwnerHasCurrent(
            commandState->currentRenderEncoderOwner) == 1) {
        mglRendererSetCommandBufferHasWork(renderer, 1);
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

/* === draw-path guard cluster (log 178) ================================= */

/* -invalidateCurrentPipelineStateForReason:. */
