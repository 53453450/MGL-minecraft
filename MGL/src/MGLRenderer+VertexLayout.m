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
            if (attribute >= 32u) {
                continue;
            }
            state->attrib_format[attribute] = nativePlan.attribs[a].format;
            state->attrib_offset[attribute] = nativePlan.attribs[a].offset;
            state->attrib_buffer_index[attribute] = 0u;
            state->attrib_stride[attribute] = nativePlan.stride;
            state->attrib_step_function[attribute] = 4u;
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
        if (!usesCurrentValue && !hasAttribBinding) {
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

            if (format == 0u)
            {
                NSLog(@"MGL PIPELINE DESC fail: unable to map attrib %u type/size/normalize to MTL format", i);
                return NO;
            }

            int mapped_buffer_index;

            mapped_buffer_index = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, i, __FUNCTION__);
            if (mapped_buffer_index < 0 || mapped_buffer_index >= (int)kMGLMaxMetalVertexBufferCount) {
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
            if (!usesCurrentValue && resolved.divisor)
            {
                state->attrib_step_rate[i] = (uint32_t)resolved.divisor;
                state->attrib_step_function[i] =
                    2u;
            }
            else
            {
                state->attrib_step_rate[i] = 1u;
                state->attrib_step_function[i] =
                    1u;
            }
            if (i + 1u > state->attrib_count) {
                state->attrib_count = i + 1u;
            }
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
        if (!mglIsValidGLBlendFactor(MGL_STATE(ctx)->var.blend_src_rgb[i])) {
            mglLogRenderStateRepair("blend_src_rgb", MGL_STATE(ctx)->var.blend_src_rgb[i], GL_ONE);
            MGL_STATE(ctx)->var.blend_src_rgb[i] = GL_ONE;
            repairedState = true;
        }
        if (!mglIsValidGLBlendFactor(MGL_STATE(ctx)->var.blend_src_alpha[i])) {
            mglLogRenderStateRepair("blend_src_alpha", MGL_STATE(ctx)->var.blend_src_alpha[i], GL_ONE);
            MGL_STATE(ctx)->var.blend_src_alpha[i] = GL_ONE;
            repairedState = true;
        }
        if (!mglIsValidGLBlendFactor(MGL_STATE(ctx)->var.blend_dst_rgb[i])) {
            mglLogRenderStateRepair("blend_dst_rgb", MGL_STATE(ctx)->var.blend_dst_rgb[i], GL_ZERO);
            MGL_STATE(ctx)->var.blend_dst_rgb[i] = GL_ZERO;
            repairedState = true;
        }
        if (!mglIsValidGLBlendFactor(MGL_STATE(ctx)->var.blend_dst_alpha[i])) {
            mglLogRenderStateRepair("blend_dst_alpha", MGL_STATE(ctx)->var.blend_dst_alpha[i], GL_ZERO);
            MGL_STATE(ctx)->var.blend_dst_alpha[i] = GL_ZERO;
            repairedState = true;
        }
        if (!mglIsValidGLBlendEquation(MGL_STATE(ctx)->var.blend_equation_rgb[i])) {
            mglLogRenderStateRepair("blend_equation_rgb", MGL_STATE(ctx)->var.blend_equation_rgb[i], GL_FUNC_ADD);
            MGL_STATE(ctx)->var.blend_equation_rgb[i] = GL_FUNC_ADD;
            repairedState = true;
        }
        if (!mglIsValidGLBlendEquation(MGL_STATE(ctx)->var.blend_equation_alpha[i])) {
            mglLogRenderStateRepair("blend_equation_alpha", MGL_STATE(ctx)->var.blend_equation_alpha[i], GL_FUNC_ADD);
            MGL_STATE(ctx)->var.blend_equation_alpha[i] = GL_FUNC_ADD;
            repairedState = true;
        }

        uint32_t colorMask_i;
        if (!MGL_STATE(ctx)->caps.use_color_mask[i]) {
            colorMask_i = 15u;
        } else {
            colorMask_i = 0u;
            if (MGL_STATE(ctx)->var.color_writemask[i][0]) colorMask_i |= 1u;
            if (MGL_STATE(ctx)->var.color_writemask[i][1]) colorMask_i |= 2u;
            if (MGL_STATE(ctx)->var.color_writemask[i][2]) colorMask_i |= 4u;
            if (MGL_STATE(ctx)->var.color_writemask[i][3]) colorMask_i |= 8u;
        }

        /* Force alpha write when rendering to the default framebuffer (drawable).
         * GL's default framebuffer is conceptually opaque (no alpha channel),
         * but Metal's CAMetalLayer drawable is RGBA8. If the GL app sets
         * glColorMask(R,G,B,0), the alpha channel is never written, leaving
         * the drawable with alpha=0. On macOS, the compositor treats alpha=0
         * as fully transparent, causing the displayed image to appear black.
         * Force alpha write on attachment 0 when rendering to the default
         * framebuffer to ensure the drawable is opaque. */
        if (i == 0 && MGL_STATE(ctx)->framebuffer == NULL) {
            colorMask_i |= 8u;
        }
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
