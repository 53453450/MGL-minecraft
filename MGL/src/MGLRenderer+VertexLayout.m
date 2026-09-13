/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

// MGLRenderer+VertexLayout.m
// Vertex descriptor and blend-state construction extracted from MGLRenderer+RenderPass.m

#import "MGLRenderer_Private.h"
#include "mgl_shader_abi.h"
#include "mgl_air_loader.h"   /* MGLRenderPipelineDescriptorState */
#include "mgl_draw_tess.h"

@implementation MGLRenderer (VertexLayout)



- (BOOL)generateVertexDescriptorState:(MGLRenderPipelineDescriptorState *)state
{
    return mglRenderGenerateVertexDescriptorState(
               ctx,
               state,
               (int)_tessellation.nativeTESActive,
               _tessellation.nativeTESProgram,
               (uint32_t)_tessellation.tcsOutputStride,
               (int)_batching.absoluteVertexBindingOffsets,
               __FUNCTION__) ? YES : NO;
}

- (void) updateBlendStateCache
{
    bool repairedState = false;
    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        uint32_t srcRgb = (uint32_t)MGL_STATE(ctx)->var.blend_src_rgb[i];
        if (mglRenderRepairBlendSrcFactor(&srcRgb)) {
            mglLogRenderStateRepair("blend_src_rgb",
                                    MGL_STATE(ctx)->var.blend_src_rgb[i],
                                    (GLenum)srcRgb);
            MGL_STATE(ctx)->var.blend_src_rgb[i] = (GLenum)srcRgb;
            repairedState = true;
        }
        uint32_t srcAlpha = (uint32_t)MGL_STATE(ctx)->var.blend_src_alpha[i];
        if (mglRenderRepairBlendSrcFactor(&srcAlpha)) {
            mglLogRenderStateRepair("blend_src_alpha",
                                    MGL_STATE(ctx)->var.blend_src_alpha[i],
                                    (GLenum)srcAlpha);
            MGL_STATE(ctx)->var.blend_src_alpha[i] = (GLenum)srcAlpha;
            repairedState = true;
        }
        uint32_t dstRgb = (uint32_t)MGL_STATE(ctx)->var.blend_dst_rgb[i];
        if (mglRenderRepairBlendDstFactor(&dstRgb)) {
            mglLogRenderStateRepair("blend_dst_rgb",
                                    MGL_STATE(ctx)->var.blend_dst_rgb[i],
                                    (GLenum)dstRgb);
            MGL_STATE(ctx)->var.blend_dst_rgb[i] = (GLenum)dstRgb;
            repairedState = true;
        }
        uint32_t dstAlpha = (uint32_t)MGL_STATE(ctx)->var.blend_dst_alpha[i];
        if (mglRenderRepairBlendDstFactor(&dstAlpha)) {
            mglLogRenderStateRepair("blend_dst_alpha",
                                    MGL_STATE(ctx)->var.blend_dst_alpha[i],
                                    (GLenum)dstAlpha);
            MGL_STATE(ctx)->var.blend_dst_alpha[i] = (GLenum)dstAlpha;
            repairedState = true;
        }
        uint32_t eqRgb = (uint32_t)MGL_STATE(ctx)->var.blend_equation_rgb[i];
        if (mglRenderRepairBlendEquation(&eqRgb)) {
            mglLogRenderStateRepair("blend_equation_rgb",
                                    MGL_STATE(ctx)->var.blend_equation_rgb[i],
                                    (GLenum)eqRgb);
            MGL_STATE(ctx)->var.blend_equation_rgb[i] = (GLenum)eqRgb;
            repairedState = true;
        }
        uint32_t eqAlpha = (uint32_t)MGL_STATE(ctx)->var.blend_equation_alpha[i];
        if (mglRenderRepairBlendEquation(&eqAlpha)) {
            mglLogRenderStateRepair("blend_equation_alpha",
                                    MGL_STATE(ctx)->var.blend_equation_alpha[i],
                                    (GLenum)eqAlpha);
            MGL_STATE(ctx)->var.blend_equation_alpha[i] = (GLenum)eqAlpha;
            repairedState = true;
        }

        uint32_t colorMask_i = mglRenderColorWriteMaskFromChannels(
            MGL_STATE(ctx)->caps.use_color_mask[i] ? 1 : 0,
            MGL_STATE(ctx)->var.color_writemask[i][0] ? 1 : 0,
            MGL_STATE(ctx)->var.color_writemask[i][1] ? 1 : 0,
            MGL_STATE(ctx)->var.color_writemask[i][2] ? 1 : 0,
            MGL_STATE(ctx)->var.color_writemask[i][3] ? 1 : 0);
        colorMask_i = mglRenderForceDefaultFBOAlphaWrite(
            i, MGL_STATE(ctx)->framebuffer ? 1 : 0, colorMask_i);
        uint32_t srcRgbF = 0u, srcAlphaF = 0u, dstRgbF = 0u, dstAlphaF = 0u;
        uint32_t rgbOp = 0u, alphaOp = 0u;
        (void)mglRenderBlendFactorFromGL(
            (uint32_t)MGL_STATE(ctx)->var.blend_src_rgb[i], &srcRgbF);
        (void)mglRenderBlendFactorFromGL(
            (uint32_t)MGL_STATE(ctx)->var.blend_src_alpha[i], &srcAlphaF);
        (void)mglRenderBlendFactorFromGL(
            (uint32_t)MGL_STATE(ctx)->var.blend_dst_rgb[i], &dstRgbF);
        (void)mglRenderBlendFactorFromGL(
            (uint32_t)MGL_STATE(ctx)->var.blend_dst_alpha[i], &dstAlphaF);
        (void)mglRenderBlendOperationFromGL(
            (uint32_t)MGL_STATE(ctx)->var.blend_equation_rgb[i], &rgbOp);
        (void)mglRenderBlendOperationFromGL(
            (uint32_t)MGL_STATE(ctx)->var.blend_equation_alpha[i], &alphaOp);
        [_pipelineCache setBlendFactorsForAttachment:(NSUInteger)i
                                        srcRgbFactor:srcRgbF
                                      srcAlphaFactor:srcAlphaF
                                        dstRgbFactor:dstRgbF
                                      dstAlphaFactor:dstAlphaF
                                        rgbOperation:rgbOp
                                      alphaOperation:alphaOp
                                           colorMask:colorMask_i];
    }
    if (repairedState)
        mglMarkStateDirtyBits(ctx->active_state,
                              DIRTY_RENDER_STATE | DIRTY_ALPHA_STATE);
}


@end
