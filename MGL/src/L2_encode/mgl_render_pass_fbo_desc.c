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
#include "mgl_renderer_state_areas.h"
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

#include "mgl_render_pass_ops_shared.h"

/* -mglRenderPassMatchesFramebufferImpl:name: is C now (log 169).  Its three
 * Objective-C targets were already C (mglRendererBindMTLTexture,
 * mglRendererAttachmentTextureFor, mglRendererDrawableTexture), and
 * `_renderPassManager->state` is `areas.command`, so the move was mechanical. */
int mglRenderPassMatchesFramebufferImpl(void *renderer, void *framebuffer,
                                        unsigned int framebuffer_name)
{
    Framebuffer *fbo = (Framebuffer *)framebuffer;
    const GLuint fboName = framebuffer_name;
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLRenderPassState passState = {0};
    bool hasPassState =
        mglRenderPassGetPersistentState(areas.command, &passState);
    if (!ctx || !hasPassState) {
        return 1;
    }
    MGLRenderPassIdentityState identity =
        mglRenderPassIdentitySnapshot(areas.command);
    /* Draw-buffer set must match. FBO object identity does not: Iris binds
     * distinct FBO names that share Metal attachments; requiring pointer/name
     * equality forced a rotate every BindFramebuffer (PERF fboRot). */
    if (identity.draw_buffer != mglPdStateView(&areas)->draw_buffer ||
        identity.draw_buffer_count != (uint32_t)mglMetalDrawBufferCount(ctx)) {
        return 0;
    }
    for (uint32_t i = 0; i < identity.draw_buffer_count; ++i) {
        if (identity.draw_buffers[i] != mglMetalDrawBufferAt(ctx, i)) {
            return 0;
        }
    }
    (void)fboName;

    if (!fbo) {
        GLuint mgl_drawbuffer = mglDefaultDrawBufferIndexForGL(mglPdStateView(&areas)->draw_buffer);
        void * expectedColor0 = NULL;
        void * actualColor0 = mglRenderPassTextureFromSnapshot(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);

        if (mglRenderDefaultDrawBufferIsFront(mgl_drawbuffer)) {
            /* _drawable ? [self mglDrawableTexture] : nil is the same value:
             * -mglDrawableTexture is self.drawable.texture, which is nil when
             * there is no drawable. */
            expectedColor0 = mglRendererDrawableTexture(renderer);
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
            defaultPassNeedsDepth = mglPdStateView(&areas)->caps.depth_test ||
                                    cachedDepth != NULL;
            defaultPassNeedsStencil = mglPdStateView(&areas)->caps.stencil_test ||
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
            mglPdStateView(&areas)->caps.depth_test ? 1 : 0, 0, no_sub, no_sub);
        mglRenderPassFillMatchEntry(
            &entries[2],
            mglRenderPassTextureFromSnapshot(
                &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0),
            expectedStencil,
            (mglPdStateView(&areas)->caps.stencil_test || ctx->stencil_format.format) ? 1
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
        const GLenum drawBuffer = mglMetalDrawBufferAt(ctx, i);
        /* Same choice configureUserFBO makes: GL_NONE keeps Metal slot i
         * occupied.  This path does not look at the attachment textures. */
        MGLRenderPassDrawSlotInput slotInput = {0};
        slotInput.draw_buffer_resolved =
            mglMetalResolveFboDrawAttachmentIndex(ctx, drawBuffer,
                                                  &attachmentIndex) ? 1 : 0;
        slotInput.resolved_index = attachmentIndex;
        slotInput.draw_buffer_is_none =
            mglRenderDrawBufferIsNone((uint32_t)drawBuffer);
        slotInput.slot = i;
        slotInput.max_attachments = MAX_COLOR_ATTACHMENTS;
        slotInput.attached_bitfield = fbo->color_attachment_bitfield;
        MGLRenderPassDrawSlotChoice slotChoice = {0};
        (void)mglRenderPassPlanDrawSlotAttachment(&slotInput, &slotChoice);
        const int drawSlotPresent = slotChoice.present;
        if (drawSlotPresent) {
            attachmentIndex = slotChoice.attachment_index;
        }
        FBOAttachment *attachment = drawSlotPresent ? &fbo->color_attachments[attachmentIndex] : NULL;
        Texture *tex = drawSlotPresent ? mglRendererAttachmentTextureFor(ctx, attachment) : NULL;
        void * expected = NULL;

        if (tex) {
            tex->is_render_target = 1;
            if (!mglTextureMtl(tex)) {
                if (!mglRendererBindMTLTexture(renderer, tex)) {
                    return 0;
                }
            }
            expected = (mglTextureMtl(tex));
        }

        void * actual = mglRenderPassTextureFromSnapshot(
            &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, colorSlot);
        MGLRenderPassAttachmentMatchEntry slotEntry;
        mglRenderPassFillUserFboEntry(
            &slotEntry, actual, expected, attachment != NULL && actual != NULL,
            mglRenderPassAttachmentStateFromSnapshot(
                &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                colorSlot),
            attachment);
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
        if (depthTex && !mglTextureMtl(depthTex)) {
            depthTex->is_render_target = 1;
            if (!mglRendererBindMTLTexture(renderer, depthTex)) {
                return 0;
            }
        }
        expectedDepth = depthTex ? (mglTextureMtl(depthTex)) : NULL;
    }
    void * actualDepth = mglRenderPassTextureFromSnapshot(
        &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    {
        MGLRenderPassAttachmentMatchEntry entry;
        mglRenderPassFillUserFboEntry(
            &entry, actualDepth, expectedDepth,
            fbo->depth.texture != NULL && expectedDepth != NULL,
            mglRenderPassAttachmentStateFromSnapshot(
                &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0),
            &fbo->depth);
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
        if (stencilTex && !mglTextureMtl(stencilTex)) {
            stencilTex->is_render_target = 1;
            if (!mglRendererBindMTLTexture(renderer, stencilTex)) {
                return 0;
            }
        }
        expectedStencil = stencilTex ? (mglTextureMtl(stencilTex)) : NULL;
    }
    void * actualStencil = mglRenderPassTextureFromSnapshot(
        &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0);
    {
        MGLRenderPassAttachmentMatchEntry entry;
        mglRenderPassFillUserFboEntry(
            &entry, actualStencil, expectedStencil,
            fbo->stencil.texture != NULL && expectedStencil != NULL,
            mglRenderPassAttachmentStateFromSnapshot(
                &passState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0),
            &fbo->stencil);
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
bool mglPdGetPersistentAttachmentState(
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

void *mglPdAttachmentTextureFor(const MGLCommandState *commandState,
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

bool mglPdRenderTargetSizeFor(const MGLCommandState *commandState,
                                     uint64_t *widthOut, uint64_t *heightOut)
{
    MGLRenderPassState state = {0};
    if (!mglRenderPassGetPersistentState(commandState, &state)) return false;
    if (widthOut) *widthOut = state.render_target_width;
    if (heightOut) *heightOut = state.render_target_height;
    return true;
}

uint64_t mglPdRenderTargetWidthFor(const MGLCommandState *commandState)
{
    uint64_t width = 0;
    if (mglPdRenderTargetSizeFor(commandState, &width, NULL)) return width;
    return 0;
}

uint64_t mglPdRenderTargetHeightFor(const MGLCommandState *commandState)
{
    uint64_t height = 0;
    if (mglPdRenderTargetSizeFor(commandState, NULL, &height)) return height;
    return 0;
}

void mglPdSetPersistentAttachment(const MGLCommandState *commandState,
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

void mglPdSetPersistentDimensions(const MGLCommandState *commandState,
                                         uint64_t width, uint64_t height)
{
    if (commandState && commandState->renderPassStateOwner) {
        (void)mglRenderSetRenderPassStateDimensions(
            commandState->renderPassStateOwner, width, height);
    }
}

uint64_t mglPdMin(uint64_t a, uint64_t b) { return a < b ? a : b; }

/* -configureUserFBOAttachmentsLocked. */
bool mglRenderPassConfigureUserFBOAttachments(void *renderer)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *commandState = areas.command;
    Framebuffer *fbo = mglPdStateView(&areas)->framebuffer;

    const GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
    for (int i = 0; i < drawBufferCount; i++) {
        GLuint attachmentIndex = 0u;
        const GLuint colorSlot =
            mglMetalColorSlotForDrawBuffer(ctx, (GLuint)i);
        if (colorSlot >= MAX_COLOR_ATTACHMENTS) {
            continue;
        }
        /* GL maps FS location i -> draw_buffers[i].  Metal keeps the same
         * color(i) index, so a GL_NONE hole must still occupy Metal slot i
         * (writeMask=0) or later locations land on the wrong attachment /
         * nowhere.  Prefer COLOR_ATTACHMENTi when present. */
        const GLenum drawBuffer = mglMetalDrawBufferAt(ctx, (GLuint)i);
        MGLRenderPassDrawSlotInput slotInput = {0};
        slotInput.draw_buffer_resolved =
            mglMetalResolveFboDrawAttachmentIndex(ctx, drawBuffer,
                                                  &attachmentIndex) ? 1 : 0;
        slotInput.resolved_index = attachmentIndex;
        slotInput.draw_buffer_is_none =
            mglRenderDrawBufferIsNone((uint32_t)drawBuffer);
        slotInput.slot = (uint32_t)i;
        slotInput.max_attachments = MAX_COLOR_ATTACHMENTS;
        slotInput.attached_bitfield = fbo->color_attachment_bitfield;
        slotInput.require_texture = 1;
        for (uint32_t k = 0; k < MAX_COLOR_ATTACHMENTS; k++) {
            if (fbo->color_attachments[k].texture) {
                slotInput.texture_bitfield |= 1u << k;
            }
        }
        MGLRenderPassDrawSlotChoice slotChoice = {0};
        (void)mglRenderPassPlanDrawSlotAttachment(&slotInput, &slotChoice);
        const int haveAttachment = slotChoice.present;
        if (haveAttachment) {
            attachmentIndex = slotChoice.attachment_index;
        }
        if (haveAttachment) {
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
            if (!mglTextureMtl(tex)) {
                continue;
            }
            mglRenderTargetMarkGPUAuthoritative(tex);

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
                mglApplySRGBStateToRenderTarget(mglTextureMtl(tex), ctx),
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
        if (tex && mglTextureMtl(tex)) {
            mglRenderTargetMarkGPUAuthoritative(tex);
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->depth);
            const int32_t msOffset =
                mglPlatformShellMSSamplePlaneOffset(renderer);
            if (mglRenderMSSamplePlaneAdjust(
                    mglPlatformShellMSSampleInLoop(renderer) ? 1 : 0,
                    (uint32_t)tex->target, msOffset)) {
                subresource.slice += (uint32_t)msOffset;
            }
            mglPdSetPersistentAttachment(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0,
                mglTextureMtl(tex), subresource.level, subresource.slice,
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
        if (tex && mglTextureMtl(tex)) {
            mglRenderTargetMarkGPUAuthoritative(tex);
            MGLMetalAttachmentSubresource subresource =
                mglMetalAttachmentSubresourceForAttachment(&fbo->stencil);
            const int32_t msOffset =
                mglPlatformShellMSSamplePlaneOffset(renderer);
            if (mglRenderMSSamplePlaneAdjust(
                    mglPlatformShellMSSampleInLoop(renderer) ? 1 : 0,
                    (uint32_t)tex->target, msOffset)) {
                subresource.slice += (uint32_t)msOffset;
            }
            mglPdSetPersistentAttachment(
                commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0,
                mglTextureMtl(tex), subresource.level, subresource.slice,
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
void *mglPdDepthTextureFor(const MGLCommandState *commandState)
{
    return mglPdAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0u);
}

void *mglPdStencilTextureFor(const MGLCommandState *commandState)
{
    return mglPdAttachmentTextureFor(
        commandState, MGL_RENDER_RENDER_PASS_ATTACHMENT_STENCIL, 0u);
}

MGLRenderTextureInfo mglPdTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

bool mglPdActionsFor(const MGLCommandState *commandState,
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

uint32_t mglPdLoadActionFor(const MGLCommandState *commandState,
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

uint32_t mglPdStoreActionFor(const MGLCommandState *commandState,
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

void mglPdSetPersistentActions(const MGLCommandState *commandState,
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

void mglPdSetPersistentStoreAction(const MGLCommandState *commandState,
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

MGLRendererBackendHandle *mglPdBackend(GLMContext ctx)
{
    return ctx ? (MGLRendererBackendHandle *)ctx->renderer_backend : NULL;
}

void *mglPdFallbackRenderTarget(GLMContext ctx)
{
    return mglRendererBackendGetFallbackRenderTargetTexture(mglPdBackend(ctx));
}

/* The .m twin returns (__bridge_transfer id): ARC consumes the +1 the creator
 * hands back.  C has no ARC, so the reference is left to the backend's own
 * ownership - the texture is reusable and the context creates at most one. */
void *mglPdCreateTexture(
    const MGLRenderTextureDescriptorState *descriptor)
{
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(descriptor, NULL, &texture) == 0 &&
        texture) {
        return texture;
    }
    return NULL;
}

uint64_t mglPdMax(uint64_t a, uint64_t b) { return a > b ? a : b; }

void *mglPdFallbackRenderTargetForSize(GLMContext ctx, uint64_t width,
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
const int kMglPdVerboseFrameLoopLogs = 0;

/* -finalizeRenderPassDescriptorLocked:traceRenderEncoder:. */
bool mglRenderPassFinalizeRenderPassDescriptor(void *renderer,
                                               uint64_t renderEncoderCall,
                                               int traceRenderEncoder)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
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
            (unsigned)mglPdStateView(&areas)->draw_buffer,
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
        Framebuffer *fbo = mglPdStateView(&areas)->framebuffer;
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
    if (!mglPdStateView(&areas)->framebuffer &&
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
        Framebuffer *fbo = mglPdStateView(&areas)->framebuffer;
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
