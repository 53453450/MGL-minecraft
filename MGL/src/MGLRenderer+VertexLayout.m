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
    if (!state) {
        return NO;
    }
    state->attrib_count = 0u;
    if (_tessellation.nativeTESActive) {
        MGLTessNativeVertexPlan nativePlan = {0};
        if (!mglTessPlanNativeVertexDescriptor(
                _tessellation.nativeTESProgram,
                (uint32_t)_tessellation.tcsOutputStride, &nativePlan)) {
            NSLog(@"MGL TESS ERROR: unsupported native TES control-point layout");
            return NO;
        }
        for (uint32_t a = 0u; a < nativePlan.n_attribs; a++) {
            const uint32_t attribute = nativePlan.attribs[a].index;
            if (!mglRenderNativeAttribIndexValid(attribute)) {
                continue;
            }
            state->attrib_format[attribute] = nativePlan.attribs[a].format;
            state->attrib_offset[attribute] = nativePlan.attribs[a].offset;
            state->attrib_buffer_index[attribute] = 0u;
            state->attrib_stride[attribute] = nativePlan.stride;
            state->attrib_step_function[attribute] =
                mglRenderNativeAttribStepFunction();
            state->attrib_step_rate[attribute] = 1u;
        }
        state->attrib_count = nativePlan.attrib_count;
        return YES;
    }
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    GLuint activeProgramName = activeProgram ? activeProgram->name : (ctx ? mglCurrentRenderProgramKey(ctx) : 0);
    GLuint maxAttribs;

    if (!vao) {
        NSLog(@"MGL PIPELINE DESC fail: cannot build vertex descriptor without a valid VAO");
        return NO;
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL VERTEX DESC begin program=%u vao=%p enabledMask=0x%x",
              (unsigned)activeProgramName, vao, vao->enabled_attribs);
    }

    maxAttribs = MAX_ATTRIBS;


    NSUInteger layoutStride[31] = {0};
    for (GLuint i = 0; i < maxAttribs; i++)
    {
        if (!mglRendererProgramUsesVertexAttrib(activeProgram, i)) {
            continue;
        }
        BOOL usesCurrentValue = mglRendererVertexAttribUsesCurrentValue(vao, i);
        MGLResolvedVertexAttribBinding resolved = {0};
        bool hasAttribBinding = mglRendererResolveVertexAttribBinding(ctx,
                                                                      vao,
                                                                      i,
                                                                      __FUNCTION__,
                                                                      &resolved);
        if (mglRenderSkipUnboundAttrib(usesCurrentValue ? 1 : 0,
                                       hasAttribBinding ? 1 : 0)) {
            continue;
        }

        {
            Buffer *attribBuffer = hasAttribBinding ? resolved.buffer : NULL;

            if (!usesCurrentValue && !attribBuffer)
            {
                NSLog(@"MGL PIPELINE DESC fail: attrib %u enabled but buffer is invalid", i);
                return NO;
            }

            MGLShaderResource *attrRes =
                mglRendererProgramVertexAttribResource(activeProgram, i);
            GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
            uint32_t format = 0u;
            int needsConversion = 0;
            int effectiveNormalized = 0;
            int conversionKind = 0;
            mglRenderPlanVertexAttribFormat(
                (uint32_t)vao->attrib[i].type, (uint32_t)vao->attrib[i].size,
                vao->attrib[i].integer ? 1 : 0,
                vao->attrib[i].normalized ? 1 : 0,
                mglRendererVertexAttribIsColorInput(activeProgram, i) ? 1 : 0,
                (uint32_t)shaderGlType, &format, &needsConversion,
                &effectiveNormalized, &conversionKind);
            (void)effectiveNormalized;

            if (!mglRenderAttribFormatMapped(format))
            {
                NSLog(@"MGL PIPELINE DESC fail: unable to map attrib %u type/size/normalize to MTL format", i);
                return NO;
            }

            int mapped_buffer_index;

            mapped_buffer_index = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, i, __FUNCTION__);
            if (!mglRenderVertexBufferIndexValid(
                    mapped_buffer_index,
                    (uint32_t)kMGLMaxMetalVertexBufferCount)) {
                NSLog(@"MGL ERROR: Invalid vertex buffer index %d for attribute %d (max valid=%lu)",
                      mapped_buffer_index, i, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return NO;
            }

            uint32_t attribOffset = mglRenderPlanVertexAttribOffset(
                usesCurrentValue ? 1 : 0, needsConversion,
                _batching.absoluteVertexBindingOffsets ? 1 : 0, i,
                kMGLCurrentAttribPoolStride,
                (uint32_t)resolved.relativeoffset,
                (uint32_t)resolved.binding_offset);

            uint32_t stride = mglRenderPlanVertexAttribStride(
                (uint32_t)vao->attrib[i].type, (uint32_t)vao->attrib[i].size,
                vao->attrib[i].integer ? 1 : 0, usesCurrentValue ? 1 : 0,
                conversionKind == MGL_ATTRIB_CONV_INTEGER_SIGN ? 1 : 0,
                (uint32_t)resolved.stride,
                (uint32_t)layoutStride[mapped_buffer_index]);
            layoutStride[mapped_buffer_index] = stride;

            state->attrib_format[i] = (uint32_t)format;
            state->attrib_offset[i] = attribOffset;
            state->attrib_buffer_index[i] = (uint32_t)mapped_buffer_index;
            state->attrib_stride[i] = (uint32_t)stride;
            mglRenderAttribStepFromDivisor(
                usesCurrentValue ? 1 : 0, (uint32_t)resolved.divisor,
                &state->attrib_step_function[i], &state->attrib_step_rate[i]);
            state->attrib_count =
                mglRenderAttribCountAfter(state->attrib_count, i);
        }
    }

    // clear all dirty bits as they have been translated into a vertex descriptor
    vao->dirty_bits = 0;

    return YES;
}

- (void) updateBlendStateCache
{
    bool repairedState = false;
    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        uint32_t srcRgb = (uint32_t)MGL_STATE(ctx)->var.blend_src_rgb[i];
        if (mglRenderApplyBlendRepair(
                mglIsValidGLBlendFactor((GLenum)srcRgb) ? 1 : 0, &srcRgb,
                (uint32_t)GL_ONE)) {
            mglLogRenderStateRepair("blend_src_rgb",
                                    MGL_STATE(ctx)->var.blend_src_rgb[i], GL_ONE);
            MGL_STATE(ctx)->var.blend_src_rgb[i] = (GLenum)srcRgb;
            repairedState = true;
        }
        uint32_t srcAlpha = (uint32_t)MGL_STATE(ctx)->var.blend_src_alpha[i];
        if (mglRenderApplyBlendRepair(
                mglIsValidGLBlendFactor((GLenum)srcAlpha) ? 1 : 0, &srcAlpha,
                (uint32_t)GL_ONE)) {
            mglLogRenderStateRepair("blend_src_alpha",
                                    MGL_STATE(ctx)->var.blend_src_alpha[i], GL_ONE);
            MGL_STATE(ctx)->var.blend_src_alpha[i] = (GLenum)srcAlpha;
            repairedState = true;
        }
        uint32_t dstRgb = (uint32_t)MGL_STATE(ctx)->var.blend_dst_rgb[i];
        if (mglRenderApplyBlendRepair(
                mglIsValidGLBlendFactor((GLenum)dstRgb) ? 1 : 0, &dstRgb,
                (uint32_t)GL_ZERO)) {
            mglLogRenderStateRepair("blend_dst_rgb",
                                    MGL_STATE(ctx)->var.blend_dst_rgb[i], GL_ZERO);
            MGL_STATE(ctx)->var.blend_dst_rgb[i] = (GLenum)dstRgb;
            repairedState = true;
        }
        uint32_t dstAlpha = (uint32_t)MGL_STATE(ctx)->var.blend_dst_alpha[i];
        if (mglRenderApplyBlendRepair(
                mglIsValidGLBlendFactor((GLenum)dstAlpha) ? 1 : 0, &dstAlpha,
                (uint32_t)GL_ZERO)) {
            mglLogRenderStateRepair("blend_dst_alpha",
                                    MGL_STATE(ctx)->var.blend_dst_alpha[i], GL_ZERO);
            MGL_STATE(ctx)->var.blend_dst_alpha[i] = (GLenum)dstAlpha;
            repairedState = true;
        }
        uint32_t eqRgb = (uint32_t)MGL_STATE(ctx)->var.blend_equation_rgb[i];
        if (mglRenderApplyBlendRepair(
                mglIsValidGLBlendEquation((GLenum)eqRgb) ? 1 : 0, &eqRgb,
                (uint32_t)GL_FUNC_ADD)) {
            mglLogRenderStateRepair("blend_equation_rgb",
                                    MGL_STATE(ctx)->var.blend_equation_rgb[i],
                                    GL_FUNC_ADD);
            MGL_STATE(ctx)->var.blend_equation_rgb[i] = (GLenum)eqRgb;
            repairedState = true;
        }
        uint32_t eqAlpha = (uint32_t)MGL_STATE(ctx)->var.blend_equation_alpha[i];
        if (mglRenderApplyBlendRepair(
                mglIsValidGLBlendEquation((GLenum)eqAlpha) ? 1 : 0, &eqAlpha,
                (uint32_t)GL_FUNC_ADD)) {
            mglLogRenderStateRepair("blend_equation_alpha",
                                    MGL_STATE(ctx)->var.blend_equation_alpha[i],
                                    GL_FUNC_ADD);
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
        [_pipelineCache setBlendFactorsForAttachment:(NSUInteger)i
                                        srcRgbFactor:[self blendFactorFromGL:MGL_STATE(ctx)->var.blend_src_rgb[i]]
                                      srcAlphaFactor:[self blendFactorFromGL:MGL_STATE(ctx)->var.blend_src_alpha[i]]
                                        dstRgbFactor:[self blendFactorFromGL:MGL_STATE(ctx)->var.blend_dst_rgb[i]]
                                      dstAlphaFactor:[self blendFactorFromGL:MGL_STATE(ctx)->var.blend_dst_alpha[i]]
                                        rgbOperation:[self blendOperationFromGL: MGL_STATE(ctx)->var.blend_equation_rgb[i]]
                                      alphaOperation:[self blendOperationFromGL: MGL_STATE(ctx)->var.blend_equation_alpha[i]]
                                           colorMask:colorMask_i];
    }
    if (repairedState)
        mglMarkStateDirtyBits(ctx->active_state,
                              DIRTY_RENDER_STATE | DIRTY_ALPHA_STATE);
}

-(bool)bindFramebufferAttachmentTextures
{
    Framebuffer *fbo;

    // MEMORY SAFETY: Validate context and framebuffer
    if (!ctx) {
        NSLog(@"MGL ERROR: NULL context detected in bindFramebufferAttachmentTextures");
        return false;
    }

    // Validate context pointer lower bound only (high addresses are valid on macOS/arm64)
    uintptr_t ctx_addr = (uintptr_t)ctx;
    if (ctx_addr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid context pointer detected in bindFramebufferAttachmentTextures: 0x%lx", ctx_addr);
        return false;
    }

    fbo = MGL_STATE(ctx)->framebuffer;

    // MEMORY SAFETY: Validate framebuffer pointer
    if (!fbo) {
        NSLog(@"MGL ERROR: NULL framebuffer detected in bindFramebufferAttachmentTextures");
        return false;
    }

    // Validate framebuffer pointer lower bound only (high addresses are valid on macOS/arm64)
    uintptr_t fbo_addr = (uintptr_t)fbo;
    if (fbo_addr < 0x1000) {
        NSLog(@"MGL ERROR: Invalid framebuffer pointer detected in bindFramebufferAttachmentTextures: 0x%lx", fbo_addr);
        return false;
    }

    for (int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        if (fbo->color_attachments[i].texture)
        {
            bool isDrawBuffer = true;
            if (fbo->color_attachments[i].textarget == GL_RENDERBUFFER && fbo->color_attachments[i].buf.rbo) {
                isDrawBuffer = fbo->color_attachments[i].buf.rbo->is_draw_buffer;
            }

            if ([self bindFramebufferTexture: &fbo->color_attachments[i] isDrawBuffer:isDrawBuffer] == false)
            {
                DEBUG_PRINT("Failed Framebuffer Attachment\n");
                return false;
            }
        }

        // early out
        if ((fbo->color_attachment_bitfield >> (i+1)) == 0)
            break;
    }

    // depth attachment
    if (fbo->depth.texture)
    {
        if ([self bindFramebufferTexture: &fbo->depth isDrawBuffer: true] == false)
        {
            DEBUG_PRINT("Failed Framebuffer Attachment\n");
            return false;
        }
    }

    // stencil attachment
    if (fbo->stencil.texture)
    {
        if ([self bindFramebufferTexture: &fbo->stencil isDrawBuffer: true] == false)
        {
            DEBUG_PRINT("Failed Framebuffer Attachment\n");
            return false;
        }
    }

    return true;
}

@end
