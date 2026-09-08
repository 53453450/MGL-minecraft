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
        }
    }
}


- (void)mglTraceRTSampleCopyWriteMark:(Texture *)tex
                                  fbo:(Framebuffer *)fbo
                           attachment:(FBOAttachment *)attachment
                                  hit:(uint64_t)hit
{
    if (!tex || !fbo || !attachment || !ctx) {
        return;
    }
    Program *program = mglResolveProgramFromState(ctx);
    Texture *rtColor = NULL;
    Texture *rtDepth = NULL;
    (void)mglFramebufferLooksLikeGLSampledCopyRenderTarget(ctx, fbo, &rtColor,
                                                           &rtDepth);
    id colorMTL = tex->mtl_data ? (__bridge id)(tex->mtl_data) : nil;
    id depthMTL = (rtDepth && rtDepth->mtl_data)
                      ? (__bridge id)(rtDepth->mtl_data)
                      : nil;
    id rpColor0 = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_COLOR, 0);
    id rpDepth = (__bridge id)mglRenderGetRenderPassAttachmentTextureOwner(
        _renderPassManager.state->renderPassStateOwner,
        MGL_RENDER_RENDER_PASS_ATTACHMENT_DEPTH, 0);
    MGLRenderTextureInfo colorInfo = {0};
    if (colorMTL) {
        (void)mglRenderGetTextureInfo((__bridge void *)colorMTL, &colorInfo);
    }
    mglTraceLog(
        "RT_SAMPLE_COPY_WRITE_MARK hit=%llu fbo=%u program=%u rtTex=%u "
        "label=\"%s\" depthTex=%u depthLabel=\"%s\" viewport=%d,%d,%d,%d "
        "scissor(en=%d box=%d,%d,%d,%d) depth(test=%d write=%d func=0x%x) "
        "blend=%d cull=%d colorMask=%d%d%d%d level=%u "
        "texInit(ever=%u full=%u source=%u) levels=%u mips=%u mipmapped=%u "
        "mtlColor=%p fmt=%lu size=%lux%lu rpColor=%p rpDepth=%p depthMTL=%p",
        (unsigned long long)hit, (unsigned)fbo->name,
        program ? (unsigned)program->name
                : (unsigned)(ctx ? MGL_STATE(ctx)->program_name : 0u),
        (unsigned)mglTraceTextureName(tex), mglTraceTextureLabel(tex),
        (unsigned)mglTraceTextureName(rtDepth), mglTraceTextureLabel(rtDepth),
        (int)MGL_STATE(ctx)->viewport[0], (int)MGL_STATE(ctx)->viewport[1],
        (int)MGL_STATE(ctx)->viewport[2], (int)MGL_STATE(ctx)->viewport[3],
        MGL_STATE(ctx)->caps.scissor_test ? 1 : 0,
        (int)MGL_STATE(ctx)->var.scissor_box[0],
        (int)MGL_STATE(ctx)->var.scissor_box[1],
        (int)MGL_STATE(ctx)->var.scissor_box[2],
        (int)MGL_STATE(ctx)->var.scissor_box[3],
        MGL_STATE(ctx)->caps.depth_test ? 1 : 0,
        MGL_STATE(ctx)->var.depth_writemask ? 1 : 0,
        (unsigned)MGL_STATE(ctx)->var.depth_func,
        MGL_STATE(ctx)->caps.blend ? 1 : 0,
        MGL_STATE(ctx)->caps.cull_face ? 1 : 0,
        MGL_STATE(ctx)->var.color_writemask[0][0] ? 1 : 0,
        MGL_STATE(ctx)->var.color_writemask[0][1] ? 1 : 0,
        MGL_STATE(ctx)->var.color_writemask[0][2] ? 1 : 0,
        MGL_STATE(ctx)->var.color_writemask[0][3] ? 1 : 0,
        (unsigned)attachment->level,
        mglTextureAttachmentLevel(tex, attachment->level)
            ? (unsigned)mglTextureAttachmentLevel(tex, attachment->level)
                  ->ever_written
            : 0u,
        mglTextureAttachmentLevel(tex, attachment->level)
            ? (unsigned)mglTextureAttachmentLevel(tex, attachment->level)
                  ->has_initialized_data
            : 0u,
        mglTextureAttachmentLevel(tex, attachment->level)
            ? (unsigned)mglTextureAttachmentLevel(tex, attachment->level)
                  ->last_init_source
            : 0u,
        tex ? (unsigned)tex->num_levels : 0u,
        tex ? (unsigned)tex->mipmap_levels : 0u,
        tex ? (unsigned)tex->mipmapped : 0u, colorMTL,
        (unsigned long)colorInfo.pixel_format, (unsigned long)colorInfo.width,
        (unsigned long)colorInfo.height, rpColor0, rpDepth, depthMTL);
}


@end
