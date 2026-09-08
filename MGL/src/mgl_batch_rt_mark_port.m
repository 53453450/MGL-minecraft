/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 * A3: RT-write mark ports (Batch cluster). Plans in mgl_batch_rt_mark.c.
 */
#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#include "mgl_render.h"
#include "mgl_batch_rt_mark.h"
#include <string.h>

typedef struct { __unsafe_unretained MGLRenderer *r; GLMContext glm; Framebuffer *fbo; } RtMarkCtx;
static int rtResolveSlot(void *v, uint32_t slot, uint32_t *att_out)
{
    RtMarkCtx *c = v; GLuint att = 0u;
    if (!mglMetalResolveFboDrawAttachmentIndex(c->glm, mglMetalDrawBufferAt(c->glm, slot), &att))
        return 0;
    if (att_out) *att_out = att; return 1;
}
static void rtMarkAtt(void *v, uint32_t att)
{ [((RtMarkCtx *)v)->r markCurrentFramebufferColorAttachmentWrittenAtIndex:att]; }
static void *rtAttMtl(void *v, uint32_t att)
{
    RtMarkCtx *c = v;
    Texture *tex = [c->r framebufferAttachmentTexture:&c->fbo->color_attachments[att]];
    return (tex && tex->mtl_data) ? tex->mtl_data : NULL;
}
static int rtRpHas(void *v, void *mtl)
{
    RtMarkCtx *c = v;
    for (GLuint slot = 0u; slot < MAX_COLOR_ATTACHMENTS; slot++) {
        if (mglRenderGetRenderPassAttachmentTextureOwner(
                c->r->_renderPassManager.state->renderPassStateOwner,
                MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, slot) == mtl)
            return 1;
    }
    return 0;
}

@implementation MGLRenderer (Batch)

- (void)markCurrentFramebufferColorAttachmentWrittenAtIndex:(GLuint)attachmentIndex
{
    Framebuffer *fbo = ctx ? MGL_STATE(ctx)->framebuffer : NULL;
    if (!fbo || !mgl_batch_rt_attachment_active(fbo->color_attachment_bitfield,
                                                attachmentIndex, MAX_COLOR_ATTACHMENTS))
        return;
    FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
    Texture *tex = [self framebufferAttachmentTexture:attachment];
    mglMarkTextureLevelRenderTargetWritten(tex, attachment->level);
    Program *renderingProgram = mglResolveProgramFromState(ctx);
    const int yflip = mgl_batch_rt_yflip_authority(
        renderingProgram && renderingProgram->modules[_VERTEX_SHADER]
                                .mgl_injected_framebuffer_yflip ? 1 : 0,
        renderingProgram ? mglRenderSamplerUnitExplicit(
                               (uint32_t)renderingProgram->modules[_VERTEX_SHADER]
                                   .mgl_injected_framebuffer_yflip) : 0,
        renderingProgram &&
            mglRendererProgramHasSampledResourceNamed(renderingProgram, "InSampler"),
        renderingProgram &&
            mglRendererProgramHasSampledResourceNamed(renderingProgram, "DiffuseSampler"));
    if (tex && yflip) tex->mtl_render_yflip_authority |= 1u;
    if (mgl_batch_rt_should_diag_attachment0(
            attachmentIndex, mglTraceLogIsEnabled() ? 1 : 0,
            mglTextureCanUseGLSampledRenderTargetCopy(tex) ? 1 : 0)) {
        static uint64_t s_guiRTWriteMarkCount = 0;
        uint64_t hit = ++s_guiRTWriteMarkCount;
        if (mgl_batch_rt_should_trace_write_mark(hit))
            [self mglTraceRTSampleCopyWriteMark:tex fbo:fbo attachment:attachment hit:hit];
    }
}

- (void)markCurrentFramebufferDrawAttachmentsWritten
{
    Framebuffer *fbo = ctx ? MGL_STATE(ctx)->framebuffer : NULL;
    if (!fbo) return;
    RtMarkCtx c = {.r = self, .glm = ctx, .fbo = fbo};
    MGLBatchRtDrawMarkOps ops = {
        .ctx = &c, .max_attachments = MAX_COLOR_ATTACHMENTS,
        .draw_buffer_count = (uint32_t)mglMetalDrawBufferCount(ctx),
        .color_attachment_bitfield = fbo->color_attachment_bitfield,
        .resolve_draw_slot = rtResolveSlot, .mark_attachment = rtMarkAtt,
        .has_rp_owner = _renderPassManager.state->renderPassStateOwner ? 1 : 0,
        .attachment_mtl = rtAttMtl, .rp_has_mtl = rtRpHas,
    };
    mgl_batch_rt_run_draw_attachments(&ops);
}

- (void)mglTraceRTSampleCopyWriteMark:(Texture *)tex fbo:(Framebuffer *)fbo
                           attachment:(FBOAttachment *)attachment hit:(uint64_t)hit
{
    if (!tex || !fbo || !attachment || !ctx) return;
    Program *program = mglResolveProgramFromState(ctx);
    Texture *rtColor = NULL, *rtDepth = NULL;
    (void)mglFramebufferLooksLikeGLSampledCopyRenderTarget(ctx, fbo, &rtColor, &rtDepth);
    id colorMTL = tex->mtl_data ? (__bridge id)tex->mtl_data : nil;
    id depthMTL = (rtDepth && rtDepth->mtl_data) ? (__bridge id)rtDepth->mtl_data : nil;
    id rpColor0 = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
    id rpDepth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    MGLRenderTextureInfo colorInfo = {0};
    if (colorMTL) (void)mglRenderGetTextureInfo((__bridge void *)colorMTL, &colorInfo);
    TextureLevel *lvl = mglTextureAttachmentLevel(tex, attachment->level);
    MGLBatchTraceStatePod state; memset(&state, 0, sizeof(state));
    for (int i = 0; i < 4; i++) state.viewport[i] = (int32_t)MGL_STATE(ctx)->viewport[i];
    state.scissor_test = MGL_STATE(ctx)->caps.scissor_test ? 1 : 0;
    for (int i = 0; i < 4; i++) state.scissor[i] = (int32_t)MGL_STATE(ctx)->var.scissor_box[i];
    state.depth_test = MGL_STATE(ctx)->caps.depth_test ? 1 : 0;
    state.depth_write = MGL_STATE(ctx)->var.depth_writemask ? 1 : 0;
    state.depth_func = (uint32_t)MGL_STATE(ctx)->var.depth_func;
    state.blend = MGL_STATE(ctx)->caps.blend ? 1 : 0;
    state.cull = MGL_STATE(ctx)->caps.cull_face ? 1 : 0;
    for (int i = 0; i < 4; i++)
        state.color_mask[i] = MGL_STATE(ctx)->var.color_writemask[0][i] ? 1 : 0;
    MGLBatchTraceRtWriteView v = {
        .hit = hit, .fbo_name = fbo->name,
        .program = program ? program->name : (ctx ? MGL_STATE(ctx)->program_name : 0u),
        .rt_tex = mglTraceTextureName(tex), .rt_label = mglTraceTextureLabel(tex),
        .depth_tex = mglTraceTextureName(rtDepth), .depth_label = mglTraceTextureLabel(rtDepth),
        .level = attachment->level, .ever = lvl ? lvl->ever_written : 0u,
        .full = lvl ? lvl->has_initialized_data : 0u,
        .source = lvl ? lvl->last_init_source : 0u,
        .levels = tex ? tex->num_levels : 0u, .mips = tex ? tex->mipmap_levels : 0u,
        .mipmapped = tex ? tex->mipmapped : 0u, .mtl_color = (__bridge void *)colorMTL,
        .fmt = colorInfo.pixel_format, .width = colorInfo.width, .height = colorInfo.height,
        .rp_color = (__bridge void *)rpColor0, .rp_depth = (__bridge void *)rpDepth,
        .depth_mtl = (__bridge void *)depthMTL,
    };
    mgl_batch_trace_copy_state_to_rt(&v, &state);
    char line[1536];
    if (mgl_batch_trace_format_rt_write_mark(line, sizeof(line), &v) > 0) mglTraceLog("%s", line);
}

@end
