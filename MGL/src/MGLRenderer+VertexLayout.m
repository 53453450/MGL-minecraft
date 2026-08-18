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

/* GL type -> Metal vertex-format value for TES control-point inputs. */
static uint32_t mglTessControlPointFormat(GLenum type)
{
    return mglRenderTessControlPointFormat((uint64_t)type);
}

@implementation MGLRenderer (VertexLayout)



- (BOOL)generateVertexDescriptorState:(MGLRenderPipelineDescriptorState *)state
{
    if (!state) {
        return NO;
    }
    state->attrib_count = 0u;
    if (_tessellation.nativeTESActive) {

        state->attrib_format[0] = mglDoubleVertexAttribFloatFormat(4u);
        state->attrib_offset[0] = 0u;
        state->attrib_buffer_index[0] = 0u;
        state->attrib_stride[0] = (uint32_t)_tessellation.tcsOutputStride;
        state->attrib_step_function[0] =
            4u;
        state->attrib_step_rate[0] = 1u;
        state->attrib_count = 1u;
        Program *tesProgram = _tessellation.nativeTESProgram;
        MGLShaderResourceList *inputs = tesProgram
            ? &tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER]
                                                  [_STAGE_INPUT_RES]
            : NULL;
        for (GLuint i = 0; inputs && inputs->list && i < inputs->count; i++) {
            MGLShaderResource *input = &inputs->list[i];
            if (input->is_per_patch || input->location >= 30u) continue;
            uint32_t format = mglTessControlPointFormat(input->gl_type);
            if (format == 0u) {
                NSLog(@"MGL TESS ERROR: unsupported control-point varying type "
                      "0x%x for %@", (unsigned)input->gl_type,
                      input->name ? [NSString stringWithUTF8String:input->name]
                                  : @"?");
                return NO;
            }
            NSUInteger attribute = (NSUInteger)input->location + 1u;
            if (attribute >= 32u) continue;
            state->attrib_format[attribute] = (uint32_t)format;
            state->attrib_offset[attribute] =
                (uint32_t)(MGL_AIR_PER_VERTEX_STRIDE +
                           (NSUInteger)input->location * 16u);
            state->attrib_buffer_index[attribute] = 0u;
            state->attrib_stride[attribute] =
                (uint32_t)_tessellation.tcsOutputStride;
            state->attrib_step_function[attribute] =
                4u;
            state->attrib_step_rate[attribute] = 1u;
            if (attribute + 1u > state->attrib_count) {
                state->attrib_count = (uint32_t)(attribute + 1u);
            }
        }
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
    bool attribsEnabledByApp = (vao->enabled_attribs != 0u);
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
        if (!attribsEnabledByApp && !hasAttribBinding) {
            continue;
        }

        {
            uint32_t format;
            Buffer *attribBuffer = hasAttribBinding ? resolved.buffer : NULL;

            if (!usesCurrentValue && !attribBuffer)
            {
                NSLog(@"MGL PIPELINE DESC fail: attrib %u enabled but buffer is invalid", i);
                return NO;
            }

            GLboolean normalized = vao->attrib[i].normalized;
            if (!normalized &&
                vao->attrib[i].type == GL_UNSIGNED_BYTE &&
                vao->attrib[i].size == 4 &&
                mglRendererVertexAttribIsColorInput(activeProgram, i)) {
                normalized = GL_TRUE;
            }

            bool needsConversion = false;
            if (vao->attrib[i].type == GL_DOUBLE) {
                needsConversion = true;
            } else if (vao->attrib[i].integer == 0 &&
                       (vao->attrib[i].type == GL_INT ||
                        vao->attrib[i].type == GL_UNSIGNED_INT)) {
                needsConversion = true;
            } else if (vao->attrib[i].type == GL_FIXED ||
                       vao->attrib[i].type == GL_UNSIGNED_INT_10_10_10_2 ||
                       vao->attrib[i].type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                needsConversion = true;
            } else if (vao->attrib[i].integer == 1) {
                MGLShaderResource *attrRes = mglRendererProgramVertexAttribResource(activeProgram, i);
                GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
                if (mglIntegerAttribNeedsConversion(vao->attrib[i].type,
                                                    shaderGlType,
                                                    vao->attrib[i].size,
                                                    NULL)) {
                    needsConversion = true;
                }
            }

            if (vao->attrib[i].type == GL_DOUBLE) {
                format = mglDoubleVertexAttribFloatFormat(vao->attrib[i].size);
            } else if (vao->attrib[i].integer == 0 &&
                       (vao->attrib[i].type == GL_INT ||
                        vao->attrib[i].type == GL_UNSIGNED_INT)) {
                format = mglDoubleVertexAttribFloatFormat(vao->attrib[i].size);
            } else if (vao->attrib[i].type == GL_FIXED) {
                format = mglDoubleVertexAttribFloatFormat(vao->attrib[i].size);
            } else if (vao->attrib[i].type == GL_UNSIGNED_INT_10_10_10_2) {
                format = mglDoubleVertexAttribFloatFormat(4u);
            } else if (vao->attrib[i].type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                format = mglDoubleVertexAttribFloatFormat(3u);
            } else if (vao->attrib[i].integer == 1) {
                uint32_t convertedFormat = 0u;
                MGLShaderResource *attrRes = mglRendererProgramVertexAttribResource(activeProgram, i);
                GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
                if (mglIntegerAttribNeedsConversion(vao->attrib[i].type,
                                                    shaderGlType,
                                                    vao->attrib[i].size,
                                                    &convertedFormat) &&
                    convertedFormat != 0u) {
                    format = convertedFormat;
                } else {
                    format = glTypeSizeToMtlType(vao->attrib[i].type,
                                                 vao->attrib[i].size,
                                                 normalized);
                }
            } else {
                format = glTypeSizeToMtlType(vao->attrib[i].type,
                                             vao->attrib[i].size,
                                             normalized);
            }

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

            uint32_t attribOffset;
            if (usesCurrentValue) {
                attribOffset = 0u;
            } else if (needsConversion || _batching.absoluteVertexBindingOffsets) {
                attribOffset = (uint32_t)resolved.relativeoffset;
            } else {
                attribOffset = (uint32_t)(resolved.binding_offset + resolved.relativeoffset);
            }

            NSUInteger stride = 0u;
            if (usesCurrentValue) {
                stride = 16u;
            } else if (vao->attrib[i].type == GL_DOUBLE) {
                NSUInteger doubleStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)(vao->attrib[i].size * sizeof(GLdouble));
                stride = mglAlignVertexStrideForMetal(doubleStride);
            } else if (vao->attrib[i].integer == 0 &&
                       (vao->attrib[i].type == GL_INT ||
                        vao->attrib[i].type == GL_UNSIGNED_INT)) {
                NSUInteger intStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)(vao->attrib[i].size * sizeof(GLint));
                stride = mglAlignVertexStrideForMetal(intStride);
            } else if (vao->attrib[i].type == GL_FIXED) {
                NSUInteger fixedStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)(vao->attrib[i].size * sizeof(int32_t));
                stride = mglAlignVertexStrideForMetal(MAX(fixedStride, (NSUInteger)(vao->attrib[i].size * sizeof(GLfloat))));
            } else if (vao->attrib[i].type == GL_UNSIGNED_INT_10_10_10_2) {
                NSUInteger packedStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)sizeof(uint32_t);
                stride = mglAlignVertexStrideForMetal(MAX(packedStride, 4u * sizeof(GLfloat)));
            } else if (vao->attrib[i].type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                NSUInteger packedStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)sizeof(uint32_t);
                stride = mglAlignVertexStrideForMetal(MAX(packedStride, 3u * sizeof(GLfloat)));
            } else if (vao->attrib[i].integer == 1) {
                uint32_t convertedFormat = 0u;
                MGLShaderResource *attrRes = mglRendererProgramVertexAttribResource(activeProgram, i);
                GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
                if (mglIntegerAttribNeedsConversion(vao->attrib[i].type,
                                                    shaderGlType,
                                                    vao->attrib[i].size,
                                                    &convertedFormat) &&
                    convertedFormat != 0u) {
                    stride = mglAlignVertexStrideForMetal(
                        (NSUInteger)vao->attrib[i].size * sizeof(GLint));
                } else if (layoutStride[mapped_buffer_index] == 0) {
                    stride = resolved.stride;
                } else {
                    stride = layoutStride[mapped_buffer_index];
                }
            } else if (layoutStride[mapped_buffer_index] == 0) {
                stride = resolved.stride;
            } else {
                stride = layoutStride[mapped_buffer_index];
            }
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
