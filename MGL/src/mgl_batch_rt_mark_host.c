/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_batch_rt_mark_host.c — batch RT-write-mark host ports (ObjC-zeroing T4).
 *
 * Split from mgl_batch_rt_mark.c so the plan file stays linkable on its own by
 * the plan harnesses (test_batch_issue and friends): these functions need the
 * renderer ports and the trace log, the plan does not.
 *
 * They were the Objective-C category methods
 * -[MGLRenderer markCurrentFramebufferColorAttachmentWrittenAtIndex:] and
 * -[MGLRenderer markCurrentFramebufferDrawAttachmentsWritten]; the ObjC callers
 * now pass the renderer as a handle.
 */

#include "mgl_batch_rt_mark.h"
#include "mgl_types_program.h"   /* Program */
#include "mgl_renderer_ports.h"
#include "mgl_draw_buffer.h"
#include "mgl_trace_strategy.h"
#include "mgl_rt_sync.h"
#include "mgl_texture_compat.h"

#include "mgl_frame_activity.h"  /* MGL_FRAME_* counters */
#include "mgl_trace_log.h"       /* mglTraceNowSeconds */

#include <limits.h>
#include <string.h>

/* ==== host ports (ObjC-zeroing T4) ====
 *
 * Moved from mgl_batch_rt_mark_port.m: the two entry points were Objective-C
 * methods on MGLRenderer, but their bodies only needed C plus two renderer
 * ports (the attachment texture and the render pass state owner).  The ObjC
 * callers now call these functions with the renderer as a handle.
 */

#include "mgl_renderer_ports.h"
#include "mgl_trace_strategy.h"
#include "mgl_draw_buffer.h"
#include "mgl_rt_sync.h"
#include "mgl_texture_compat.h"
#include "mgl_render_api_texture.h"
#include "mgl_render_api_command.h"
#include "mgl_render_api_binding.h"
#include "mgl_texture_backend.h"

/* Defined in MGLRenderer+Draw.m (C linkage); the ObjC private header declares
 * it, so C declares it here. */
extern bool mglRendererProgramHasSampledResourceNamed(Program *program,
                                                      const char *name);

/* Defined in MGLRenderer.m (C linkage); the ObjC private header wraps it in a
 * macro that adds __func__/__LINE__, so C calls the impl directly.  MGL_STATE(ctx) is an ObjC macro over the renderer's
 * activeState ivar; the dual-proxy invariant in MGLRenderer_Private.h says it
 * equals ctx->active_state, which is what this C code uses. */
extern Program *mglResolveProgramFromState(GLMContext ctx);
extern void mglMarkTextureLevelRenderTargetWrittenImpl(Texture *tex,
                                                       GLuint level,
                                                       const char *caller,
                                                       int line);

typedef struct {
    void *renderer;
    GLMContext glm;
    Framebuffer *fbo;
} MGLBatchRtMarkHostCtx;

static int mglBatchRtMarkHostResolveSlot(void *v, uint32_t slot,
                                         uint32_t *att_out)
{
    MGLBatchRtMarkHostCtx *c = (MGLBatchRtMarkHostCtx *)v;
    GLuint att = 0u;
    if (!mglMetalResolveFboDrawAttachmentIndex(
            c->glm, mglMetalDrawBufferAt(c->glm, slot), &att)) {
        return 0;
    }
    if (att_out) {
        *att_out = att;
    }
    return 1;
}

static void mglBatchRtMarkHostMarkAttachment(void *v, uint32_t att)
{
    MGLBatchRtMarkHostCtx *c = (MGLBatchRtMarkHostCtx *)v;
    mglBatchRtMarkColorAttachmentWritten(c->renderer, c->glm, att);
}

static void *mglBatchRtMarkHostAttachmentMtl(void *v, uint32_t att)
{
    MGLBatchRtMarkHostCtx *c = (MGLBatchRtMarkHostCtx *)v;
    Texture *tex =
        mglRendererAttachmentTextureFor(c->glm, &c->fbo->color_attachments[att]);
    return (tex && mglTextureMtl(tex)) ? mglTextureMtl(tex) : NULL;
}

static int mglBatchRtMarkHostRpHas(void *v, void *mtl)
{
    MGLBatchRtMarkHostCtx *c = (MGLBatchRtMarkHostCtx *)v;
    void *owner = mglRendererCommandStateFor(c->renderer)->renderPassStateOwner;
    for (GLuint slot = 0u; slot < MAX_COLOR_ATTACHMENTS; slot++) {
        if (mglRenderGetRenderPassAttachmentTextureOwner(
                owner, MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, slot) == mtl) {
            return 1;
        }
    }
    return 0;
}

void mglBatchRtMarkColorAttachmentWritten(void *renderer, GLMContext ctx,
                                          uint32_t attachment_index)
{
    Framebuffer *fbo = (ctx && ctx->active_state) ? ctx->active_state->framebuffer : NULL;
    if (!fbo ||
        !mgl_batch_rt_attachment_active(fbo->color_attachment_bitfield,
                                        attachment_index,
                                        MAX_COLOR_ATTACHMENTS)) {
        return;
    }
    FBOAttachment *attachment = &fbo->color_attachments[attachment_index];
    Texture *tex = mglRendererAttachmentTextureFor(ctx, attachment);
    mglMarkTextureLevelRenderTargetWrittenImpl(tex, attachment->level,
                                              __func__, __LINE__);
}

void mglBatchRtMarkCurrentFramebufferDrawAttachments(void *renderer,
                                                     GLMContext ctx)
{
    Framebuffer *fbo = (ctx && ctx->active_state) ? ctx->active_state->framebuffer : NULL;
    if (!fbo) {
        return;
    }
    MGLBatchRtMarkHostCtx c = {.renderer = renderer, .glm = ctx, .fbo = fbo};
    MGLBatchRtDrawMarkOps ops = {
        .ctx = &c,
        .max_attachments = MAX_COLOR_ATTACHMENTS,
        .draw_buffer_count = (uint32_t)mglMetalDrawBufferCount(ctx),
        .color_attachment_bitfield = fbo->color_attachment_bitfield,
        .resolve_draw_slot = mglBatchRtMarkHostResolveSlot,
        .mark_attachment = mglBatchRtMarkHostMarkAttachment,
        .has_rp_owner =
            mglRendererCommandStateFor(renderer)->renderPassStateOwner ? 1 : 0,
        .attachment_mtl = mglBatchRtMarkHostAttachmentMtl,
        .rp_has_mtl = mglBatchRtMarkHostRpHas,
    };
    mgl_batch_rt_run_draw_attachments(&ops);

    /* GL 4.6 §17.3.3/§17.3.4: draws write depth only with the depth test on
     * (and DepthMask TRUE), and stencil only with the stencil test on. */
    GLMState *state = ctx->active_state;
    Texture *depth = NULL;
    if (fbo->depth.texture && state->caps.depth_test &&
        state->var.depth_writemask) {
        depth = mglRendererAttachmentTextureFor(ctx, &fbo->depth);
        mglMarkTextureLevelRenderTargetWrittenImpl(depth, fbo->depth.level,
                                                  __func__, __LINE__);
    }
    if (fbo->stencil.texture && state->caps.stencil_test &&
        (state->var.stencil_writemask || state->var.stencil_back_writemask)) {
        Texture *stencil = mglRendererAttachmentTextureFor(ctx, &fbo->stencil);
        if (stencil != depth) {
            mglMarkTextureLevelRenderTargetWrittenImpl(stencil, fbo->stencil.level,
                                                      __func__, __LINE__);
        }
    }
}

/* === Draw-submission records (former -[MGLRenderer recordArrayDrawSubmittedMode:
 * vertexCount:] / -[MGLRenderer recordElementDrawSubmittedMode:indexCount:]) ===
 * The renderer's draw entry points call these once per submitted draw; they keep
 * the frame counters, the last-draw snapshot and the RT-mark in one place. */
void mglBatchRecordArrayDrawSubmitted(void *renderer, GLMContext ctx, GLenum mode,
                                      uint64_t vertex_count)
{
    MGL_FRAME_STORE(g_mglLastDrawArraysSeconds, mglTraceNowSeconds());
    MGL_FRAME_STORE(g_mglLastDrawArraysProgram, mglCurrentRenderProgramKey(ctx));
    MGL_FRAME_STORE(g_mglLastDrawArraysMode, mode);
    MGL_FRAME_STORE(g_mglLastDrawArraysCount,
                    (vertex_count > (uint64_t)INT_MAX) ? INT_MAX : (GLsizei)vertex_count);
    MGL_FRAME_INC(g_mglDrawArraysSinceSwap);
    MGL_FRAME_ADD(g_mglDrawArrayVerticesSinceSwap, vertex_count);
    mglBatchRtMarkCurrentFramebufferDrawAttachments(renderer, ctx);
}

void mglBatchRecordElementDrawSubmitted(void *renderer, GLMContext ctx, GLenum mode,
                                        uint64_t index_count)
{
    MGL_FRAME_STORE(g_mglLastDrawElementsSeconds, mglTraceNowSeconds());
    MGL_FRAME_STORE(g_mglLastDrawElementsProgram, mglCurrentRenderProgramKey(ctx));
    MGL_FRAME_STORE(g_mglLastDrawElementsMode, mode);
    MGL_FRAME_STORE(g_mglLastDrawElementsCount,
                    (index_count > (uint64_t)INT_MAX) ? INT_MAX : (GLsizei)index_count);
    MGL_FRAME_INC(g_mglDrawElementsSinceSwap);
    MGL_FRAME_ADD(g_mglDrawElementIndicesSinceSwap, index_count);
    mglBatchRtMarkCurrentFramebufferDrawAttachments(renderer, ctx);
}
