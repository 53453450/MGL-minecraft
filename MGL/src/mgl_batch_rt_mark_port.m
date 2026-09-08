/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * A3: RT-write mark ports split from MGLRenderer+Batch.m (Batch cluster metric).
 * Same (Batch) category; plans in mgl_batch_rt_mark.c.
 */
#import "MGLRenderer_Private.h"
#import "MGLRenderer+Draw_Private.h"
#include "mgl_render.h"
#include "mgl_batch_rt_mark.h"
#include <string.h>

@implementation MGLRenderer (Batch)

- (void)markCurrentFramebufferColorAttachmentWrittenAtIndex:(GLuint)attachmentIndex
{
    Framebuffer *fbo = ctx ? MGL_STATE(ctx)->framebuffer : NULL;
    if (!fbo ||
        !mgl_batch_rt_attachment_active(fbo->color_attachment_bitfield,
                                        attachmentIndex,
                                        MAX_COLOR_ATTACHMENTS)) {
        return;
    }

    FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
    Texture *tex = [self framebufferAttachmentTexture:attachment];
    mglMarkTextureLevelRenderTargetWritten(tex, attachment->level);

    /* A3: Y-flip authority decision in mgl_batch_rt_yflip_authority. */
    Program *renderingProgram = mglResolveProgramFromState(ctx);
    const int yflip = mgl_batch_rt_yflip_authority(
        renderingProgram &&
                renderingProgram->modules[_VERTEX_SHADER]
                    .mgl_injected_framebuffer_yflip
            ? 1
            : 0,
        renderingProgram
            ? mglRenderSamplerUnitExplicit(
                  (uint32_t)renderingProgram->modules[_VERTEX_SHADER]
                      .mgl_injected_framebuffer_yflip)
            : 0,
        renderingProgram &&
                mglRendererProgramHasSampledResourceNamed(renderingProgram,
                                                          "InSampler"),
        renderingProgram &&
                mglRendererProgramHasSampledResourceNamed(renderingProgram,
                                                          "DiffuseSampler"));
    if (tex && yflip) {
        tex->mtl_render_yflip_authority |= 1u;
    }

    if (mgl_batch_rt_should_diag_attachment0(
            attachmentIndex, mglTraceLogIsEnabled() ? 1 : 0,
            mglTextureCanUseGLSampledRenderTargetCopy(tex) ? 1 : 0)) {
        static uint64_t s_guiRTWriteMarkCount = 0;
        uint64_t hit = ++s_guiRTWriteMarkCount;
        if (mgl_batch_rt_should_trace_write_mark(hit)) {
            [self mglTraceRTSampleCopyWriteMark:tex
                                            fbo:fbo
                                     attachment:attachment
                                            hit:hit];
        }
    }
}


- (void)markCurrentFramebufferDrawAttachmentsWritten
{
    Framebuffer *fbo = ctx ? MGL_STATE(ctx)->framebuffer : NULL;
    if (!fbo) {
        return;
    }

    /* Draw-buffer marks first; cross-check RP descriptor for misses (no
     * double-bump via attachmentMarked). See ARCH / GUI Y-flip notes. */
    bool attachmentMarked[MAX_COLOR_ATTACHMENTS] = {false};
    GLsizei drawBufferCount = mglMetalDrawBufferCount(ctx);
    for (GLsizei slot = 0; slot < drawBufferCount; ++slot) {
        GLuint attachmentIndex = 0u;
        if (mglMetalResolveFboDrawAttachmentIndex(ctx,
                                                  mglMetalDrawBufferAt(ctx, (GLuint)slot),
                                                  &attachmentIndex)) {
            [self markCurrentFramebufferColorAttachmentWrittenAtIndex:attachmentIndex];
            if (attachmentIndex < MAX_COLOR_ATTACHMENTS) {
                attachmentMarked[attachmentIndex] = true;
            }
        }
    }

    if (!_renderPassManager.state->renderPassStateOwner) {
        return;
    }

    for (GLuint attachmentIndex = 0u; attachmentIndex < MAX_COLOR_ATTACHMENTS; attachmentIndex++) {
        if (!mgl_batch_rt_should_cross_mark(
                attachmentMarked[attachmentIndex] ? 1 : 0,
                mgl_batch_rt_attachment_active(fbo->color_attachment_bitfield,
                                               attachmentIndex,
                                               MAX_COLOR_ATTACHMENTS))) {
            continue;
        }
        Texture *tex = [self framebufferAttachmentTexture:&fbo->color_attachments[attachmentIndex]];
        id mtlTex = (tex && tex->mtl_data)
            ? (__bridge id)(tex->mtl_data)
            : nil;
        if (!mtlTex) {
            continue;
        }
        for (GLuint colorSlot = 0u; colorSlot < MAX_COLOR_ATTACHMENTS; colorSlot++) {
            if ((__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
                    _renderPassManager.state->renderPassStateOwner,
                    MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR,
                    colorSlot) == mtlTex) {
                [self markCurrentFramebufferColorAttachmentWrittenAtIndex:attachmentIndex];
                break;
            }
- (void)mglTraceRTSampleCopyWriteMark:(Texture *)tex
                                  fbo:(Framebuffer *)fbo
                           attachment:(FBOAttachment *)attachment
                                  hit:(uint64_t)hit
{
    if (!tex || !fbo || !attachment || !ctx) return;
    Program *program = mglResolveProgramFromState(ctx);
    Texture *rtColor = NULL, *rtDepth = NULL;
    (void)mglFramebufferLooksLikeGLSampledCopyRenderTarget(ctx, fbo, &rtColor,
                                                           &rtDepth);
    id colorMTL = tex->mtl_data ? (__bridge id)tex->mtl_data : nil;
    id depthMTL =
        (rtDepth && rtDepth->mtl_data) ? (__bridge id)rtDepth->mtl_data : nil;
    id rpColor0 = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
    id rpDepth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    MGLRenderTextureInfo colorInfo = {0};
    if (colorMTL)
        (void)mglRenderGetTextureInfo((__bridge void *)colorMTL, &colorInfo);
    TextureLevel *lvl = mglTextureAttachmentLevel(tex, attachment->level);
    MGLBatchTraceStatePod state;
    memset(&state, 0, sizeof(state));
    for (int i = 0; i < 4; i++) state.viewport[i] = (int32_t)MGL_STATE(ctx)->viewport[i];
    state.scissor_test = MGL_STATE(ctx)->caps.scissor_test ? 1 : 0;
    for (int i = 0; i < 4; i++)
        state.scissor[i] = (int32_t)MGL_STATE(ctx)->var.scissor_box[i];
    state.depth_test = MGL_STATE(ctx)->caps.depth_test ? 1 : 0;
    state.depth_write = MGL_STATE(ctx)->var.depth_writemask ? 1 : 0;
    state.depth_func = (uint32_t)MGL_STATE(ctx)->var.depth_func;
    state.blend = MGL_STATE(ctx)->caps.blend ? 1 : 0;
    state.cull = MGL_STATE(ctx)->caps.cull_face ? 1 : 0;
    for (int i = 0; i < 4; i++)
        state.color_mask[i] = MGL_STATE(ctx)->var.color_writemask[0][i] ? 1 : 0;
    MGLBatchTraceRtWriteView v = {
        .hit = hit,
        .fbo_name = fbo->name,
        .program = program ? program->name
                           : (ctx ? MGL_STATE(ctx)->program_name : 0u),
        .rt_tex = mglTraceTextureName(tex),
        .rt_label = mglTraceTextureLabel(tex),
        .depth_tex = mglTraceTextureName(rtDepth),
        .depth_label = mglTraceTextureLabel(rtDepth),
        .level = attachment->level,
        .ever = lvl ? lvl->ever_written : 0u,
        .full = lvl ? lvl->has_initialized_data : 0u,
        .source = lvl ? lvl->last_init_source : 0u,
        .levels = tex ? tex->num_levels : 0u,
        .mips = tex ? tex->mipmap_levels : 0u,
        .mipmapped = tex ? tex->mipmapped : 0u,
        .mtl_color = (__bridge void *)colorMTL,
        .fmt = colorInfo.pixel_format,
        .width = colorInfo.width,
        .height = colorInfo.height,
        .rp_color = (__bridge void *)rpColor0,
        .rp_depth = (__bridge void *)rpDepth,
        .depth_mtl = (__bridge void *)depthMTL,
    };
    for (int i = 0; i < 4; i++) v.viewport[i] = state.viewport[i];
    v.scissor_en = state.scissor_test;
    for (int i = 0; i < 4; i++) v.scissor[i] = state.scissor[i];
    v.depth_test = state.depth_test;
    v.depth_write = state.depth_write;
    v.depth_func = state.depth_func;
    v.blend = state.blend;
    v.cull = state.cull;
    for (int i = 0; i < 4; i++) v.color_mask[i] = state.color_mask[i];
    char line[1536];
    if (mgl_batch_trace_format_rt_write_mark(line, sizeof(line), &v) > 0)
        mglTraceLog("%s", line);
}

pth, depthMTL);
}


@end
