// MGLRenderer+VertexLayout.m
// Vertex descriptor and blend-state construction extracted from MGLRenderer+RenderPass.m

#import "MGLRenderer_Private.h"
#include "mgl_shader_abi.h"

#import <objc/message.h>

static MTLVertexFormat mglTessControlPointFormat(GLenum type)
{
    switch (type) {
        case GL_FLOAT: return MTLVertexFormatFloat;
        case GL_FLOAT_VEC2: return MTLVertexFormatFloat2;
        case GL_FLOAT_VEC3: return MTLVertexFormatFloat3;
        case GL_FLOAT_VEC4: return MTLVertexFormatFloat4;
        case GL_INT: return MTLVertexFormatInt;
        case GL_INT_VEC2: return MTLVertexFormatInt2;
        case GL_INT_VEC3: return MTLVertexFormatInt3;
        case GL_INT_VEC4: return MTLVertexFormatInt4;
        case GL_UNSIGNED_INT:
        case GL_BOOL: return MTLVertexFormatUInt;
        case GL_UNSIGNED_INT_VEC2:
        case GL_BOOL_VEC2: return MTLVertexFormatUInt2;
        case GL_UNSIGNED_INT_VEC3:
        case GL_BOOL_VEC3: return MTLVertexFormatUInt3;
        case GL_UNSIGNED_INT_VEC4:
        case GL_BOOL_VEC4: return MTLVertexFormatUInt4;
        default: return MTLVertexFormatInvalid;
    }
}

@implementation MGLRenderer (VertexLayout)

- (MTLVertexDescriptor *)generateVertexDescriptor
{
    MTLVertexDescriptor *vertexDescriptor = [[MTLVertexDescriptor alloc] init];
    if (!vertexDescriptor) {
        NSLog(@"MGL VERTEX ERROR: failed to allocate MTLVertexDescriptor");
        return nil;
    }
    if (_tessellation.nativeTESActive) {
        [vertexDescriptor reset];
        vertexDescriptor.attributes[0].format = MTLVertexFormatFloat4;
        vertexDescriptor.attributes[0].offset = 0u;
        vertexDescriptor.attributes[0].bufferIndex = 0u;
        vertexDescriptor.layouts[0].stride = _tessellation.tcsOutputStride;
        vertexDescriptor.layouts[0].stepFunction =
            MTLVertexStepFunctionPerPatchControlPoint;
        vertexDescriptor.layouts[0].stepRate = 1u;
        Program *tesProgram = _tessellation.nativeTESProgram;
        MGLShaderResourceList *inputs = tesProgram
            ? &tesProgram->shader_resources_list[_TESS_EVALUATION_SHADER]
                                                      [_STAGE_INPUT_RES]
            : NULL;
        for (GLuint i = 0; inputs && inputs->list && i < inputs->count; i++) {
            MGLShaderResource *input = &inputs->list[i];
            if (input->is_per_patch || input->location >= 30u) continue;
            MTLVertexFormat format = mglTessControlPointFormat(input->gl_type);
            if (format == MTLVertexFormatInvalid) {
                NSLog(@"MGL TESS ERROR: unsupported control-point varying type "
                      "0x%x for %@", (unsigned)input->gl_type,
                      input->name ? [NSString stringWithUTF8String:input->name]
                                  : @"?");
                return nil;
            }
            NSUInteger attribute = (NSUInteger)input->location + 1u;
            vertexDescriptor.attributes[attribute].format = format;
            vertexDescriptor.attributes[attribute].offset =
                MGL_AIR_PER_VERTEX_STRIDE +
                (NSUInteger)input->location * 16u;
            vertexDescriptor.attributes[attribute].bufferIndex = 0u;
        }
        return vertexDescriptor;
    }
    VertexArray *vao = mglRendererGetValidatedVAO(ctx, __FUNCTION__);
    Program *activeProgram = mglResolveProgramForStageFromState(ctx, _VERTEX_SHADER);
    GLuint activeProgramName = activeProgram ? activeProgram->name : (ctx ? mglCurrentRenderProgramKey(ctx) : 0);
    GLuint maxAttribs;

    if (!vao) {
        NSLog(@"MGL PIPELINE DESC fail: cannot build vertex descriptor without a valid VAO");
        return nil;
    }

    if (kMGLVerbosePipelineLogs) {
        NSLog(@"MGL VERTEX DESC begin program=%u vao=%p enabledMask=0x%x",
              (unsigned)activeProgramName, vao, vao->enabled_attribs);
    }

    [vertexDescriptor reset]; // ??? debug
    maxAttribs = MAX_ATTRIBS;

    // we can bind a new vertex descriptor without creating a new renderbuffer
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
        // When enabled_attribs tracking is empty but the program uses this attribute,
        // validate the buffer and proceed (Sodium DSA path compatibility).
        if (!attribsEnabledByApp && !hasAttribBinding) {
            continue;
        }

        {
            MTLVertexFormat format;
            Buffer *attribBuffer = hasAttribBinding ? resolved.buffer : NULL;

            if (!usesCurrentValue && !attribBuffer)
            {
                NSLog(@"MGL PIPELINE DESC fail: attrib %u enabled but buffer is invalid", i);
                return NULL;
            }

            GLboolean normalized = vao->attrib[i].normalized;
            if (!normalized &&
                vao->attrib[i].type == GL_UNSIGNED_BYTE &&
                vao->attrib[i].size == 4 &&
                mglRendererVertexAttribIsColorInput(activeProgram, i)) {
                normalized = GL_TRUE;
            }

            /* Determine whether this attrib will be CPU-converted before
             * binding. Converted buffers are reborn starting at the original
             * binding_offset, so the vertex descriptor's attribute offset must
             * NOT include binding_offset for them (only relativeoffset).
             * Non-converted attribs bind the original buffer at offset 0, so
             * their attribute offset must include binding_offset. */
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
                /* These packed/fixed formats have no direct Metal vertex
                 * format (see glTypeSizeToMtlType). Unpack to float on the
                 * CPU in bindVertexBuffersToCurrentRenderEncoder. */
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
                /* Metal's 32-bit integer vertex formats (Int/UInt) cannot feed
                 * float shader inputs; glVertexAttribFormat (non-integer) with
                 * GL_INT/GL_UNSIGNED_INT requires int->float conversion. Use a
                 * float format here and convert the data on the CPU side in
                 * bindVertexBuffersToCurrentRenderEncoder (like GL_DOUBLE). */
                format = mglDoubleVertexAttribFloatFormat(vao->attrib[i].size);
            } else if (vao->attrib[i].type == GL_FIXED) {
                /* GL_FIXED: 16.16 fixed-point, unpacked to float[size] on the
                 * CPU. Output component count matches the GL size. */
                format = mglDoubleVertexAttribFloatFormat(vao->attrib[i].size);
            } else if (vao->attrib[i].type == GL_UNSIGNED_INT_10_10_10_2) {
                /* Packed RGBA -> float4 (CPU unpack). */
                format = MTLVertexFormatFloat4;
            } else if (vao->attrib[i].type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                /* Packed RGB float -> float3 (CPU unpack). */
                format = MTLVertexFormatFloat3;
            } else if (vao->attrib[i].integer == 1) {
                /* glVertexAttribIFormat path: Metal only allows 32-bit Int
                 * formats for int shader inputs and UInt formats for uint
                 * inputs. 8/16-bit signed formats sign-extend to int (and
                 * zero-extend to uint), but unsigned source formats cannot
                 * feed int inputs and signed sources cannot feed uint inputs.
                 * When the source signedness is incompatible with the shader's
                 * declared type, convert the data to the shader's 32-bit
                 * integer type on the CPU side in
                 * bindVertexBuffersToCurrentRenderEncoder. */
                MTLVertexFormat convertedFormat = MTLVertexFormatInvalid;
                MGLShaderResource *attrRes = mglRendererProgramVertexAttribResource(activeProgram, i);
                GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
                if (mglIntegerAttribNeedsConversion(vao->attrib[i].type,
                                                    shaderGlType,
                                                    vao->attrib[i].size,
                                                    &convertedFormat) &&
                    convertedFormat != MTLVertexFormatInvalid) {
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

            if (format == MTLVertexFormatInvalid)
            {
                NSLog(@"MGL PIPELINE DESC fail: unable to map attrib %u type/size/normalize to MTL format", i);
                return nil;
            }

            int mapped_buffer_index;

            mapped_buffer_index = mglRendererResolveVertexAttributeBufferIndex(ctx, vao, i, __FUNCTION__);
            if (mapped_buffer_index < 0 || mapped_buffer_index >= (int)kMGLMaxMetalVertexBufferCount) {
                NSLog(@"MGL ERROR: Invalid vertex buffer index %d for attribute %d (max valid=%lu)",
                      mapped_buffer_index, i, (unsigned long)kMGLMaxMetalVertexBufferIndex);
                return NULL;
            }

            vertexDescriptor.attributes[i].bufferIndex = mapped_buffer_index;
            /* When multiple attributes share a Metal buffer slot (because they
             * use the same VBO/stride/divisor), the per-attribute binding_offset
             * must be folded into the vertex descriptor's attribute offset.
             * The buffer itself is bound at offset 0 in
             * bindVertexBuffersToCurrentRenderEncoder.
             *
             * EXCEPTION: CPU-converted attribs (GL_DOUBLE, GL_INT→float,
             * integer signedness mismatch) produce a fresh buffer that already
             * starts at binding_offset. For those, the attribute offset must be
             * just relativeoffset, otherwise the shader would read past the
             * start of the converted data.
             *
             * EXCEPTION: BindNoFlush batches with per-draw BindVertexBuffer
             * overrides rebind Metal buffers at the absolute
             * VERTEX_BINDING_OFFSET.  Baking the snapshot binding_offset into
             * the descriptor would double-count those overrides. */
            if (usesCurrentValue) {
                vertexDescriptor.attributes[i].offset = 0u;
            } else if (needsConversion || _batching.absoluteVertexBindingOffsets) {
                vertexDescriptor.attributes[i].offset = (NSUInteger)resolved.relativeoffset;
            } else {
                vertexDescriptor.attributes[i].offset = (NSUInteger)(resolved.binding_offset + resolved.relativeoffset);
            }
            vertexDescriptor.attributes[i].format = format;

            if (usesCurrentValue) {
                vertexDescriptor.layouts[mapped_buffer_index].stride = 16u;
            } else if (vao->attrib[i].type == GL_DOUBLE) {
                NSUInteger doubleStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)(vao->attrib[i].size * sizeof(GLdouble));
                vertexDescriptor.layouts[mapped_buffer_index].stride = mglAlignVertexStrideForMetal(doubleStride);
            } else if (vao->attrib[i].integer == 0 &&
                       (vao->attrib[i].type == GL_INT ||
                        vao->attrib[i].type == GL_UNSIGNED_INT)) {
                NSUInteger intStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)(vao->attrib[i].size * sizeof(GLint));
                vertexDescriptor.layouts[mapped_buffer_index].stride = mglAlignVertexStrideForMetal(intStride);
            } else if (vao->attrib[i].type == GL_FIXED) {
                /* GL_FIXED unpacks to float[size]; source element size is
                 * size*4 (each GLfixed is 32-bit). Must match the converted
                 * stride computed in floatVertexBufferForFixedAttrib. */
                NSUInteger fixedStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)(vao->attrib[i].size * sizeof(int32_t));
                vertexDescriptor.layouts[mapped_buffer_index].stride =
                    mglAlignVertexStrideForMetal(MAX(fixedStride, (NSUInteger)(vao->attrib[i].size * sizeof(GLfloat))));
            } else if (vao->attrib[i].type == GL_UNSIGNED_INT_10_10_10_2) {
                /* Packed uint32 -> float4. Must match the converted stride
                 * computed in floatVertexBufferForPacked1010102Attrib. */
                NSUInteger packedStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)sizeof(uint32_t);
                vertexDescriptor.layouts[mapped_buffer_index].stride =
                    mglAlignVertexStrideForMetal(MAX(packedStride, 4u * sizeof(GLfloat)));
            } else if (vao->attrib[i].type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                /* Packed uint32 -> float3. Must match the converted stride
                 * computed in floatVertexBufferForPacked10f11f11fAttrib. */
                NSUInteger packedStride = resolved.stride > 0
                    ? (NSUInteger)resolved.stride
                    : (NSUInteger)sizeof(uint32_t);
                vertexDescriptor.layouts[mapped_buffer_index].stride =
                    mglAlignVertexStrideForMetal(MAX(packedStride, 3u * sizeof(GLfloat)));
            } else if (vao->attrib[i].integer == 1) {
                /* Integer attribs that need CPU conversion (unsigned source
                 * feeding int shader input, or signed source feeding uint
                 * input) are reborn as 32-bit Int/UInt buffers, so the layout
                 * stride must match the converted stride (componentCount * 4).
                 * Directly-compatible integer attribs keep their source stride. */
                MTLVertexFormat convertedFormat = MTLVertexFormatInvalid;
                MGLShaderResource *attrRes = mglRendererProgramVertexAttribResource(activeProgram, i);
                GLuint shaderGlType = attrRes ? attrRes->gl_type : 0u;
                if (mglIntegerAttribNeedsConversion(vao->attrib[i].type,
                                                    shaderGlType,
                                                    vao->attrib[i].size,
                                                    &convertedFormat) &&
                    convertedFormat != MTLVertexFormatInvalid) {
                    NSUInteger convStride = mglAlignVertexStrideForMetal(
                        (NSUInteger)vao->attrib[i].size * sizeof(GLint));
                    vertexDescriptor.layouts[mapped_buffer_index].stride = convStride;
                } else if (vertexDescriptor.layouts[mapped_buffer_index].stride == 0) {
                    vertexDescriptor.layouts[mapped_buffer_index].stride = resolved.stride;
                }
            } else if (vertexDescriptor.layouts[mapped_buffer_index].stride == 0) {
                vertexDescriptor.layouts[mapped_buffer_index].stride = resolved.stride;
            }

            if (!usesCurrentValue && resolved.divisor)
            {
                vertexDescriptor.layouts[mapped_buffer_index].stepRate = resolved.divisor;
                vertexDescriptor.layouts[mapped_buffer_index].stepFunction = MTLVertexStepFunctionPerInstance;
            }
            else
            {
	            vertexDescriptor.layouts[mapped_buffer_index].stepRate = 1;
	            vertexDescriptor.layouts[mapped_buffer_index].stepFunction = MTLVertexStepFunctionPerVertex;
	        }

            static uint64_t s_traceFileVertexDescriptorAttribLogs = 0;
            if (mglProgramNeedsTraceLog(activeProgram) &&
                mglShouldLogTraceFileBindingForProgram(activeProgram, &s_traceFileVertexDescriptorAttribLogs)) {
                MGLShaderResource *resource = mglRendererProgramVertexAttribResource(activeProgram, i);
                mglTraceLog("VATTR_DESC program=%u attrib=%u resource=%s loc=%u metalSlot=%d glBuffer=%u bindingIndex=%u bindingOffset=%lld relOffset=%lld stride=%u size=%u type=0x%x normalized=%u/%u divisor=%u table=%d format=%lu(%s)",
                            (unsigned)activeProgramName,
                            (unsigned)i,
                            resource && resource->name ? resource->name : "(unknown)",
                            resource ? (unsigned)resource->location : 0xffffffffu,
                            mapped_buffer_index,
                            usesCurrentValue ? 0u : (unsigned)attribBuffer->name,
                            usesCurrentValue ? i : (unsigned)resolved.binding_index,
                            usesCurrentValue ? 0ll : (long long)resolved.binding_offset,
                            usesCurrentValue ? 0ll : (long long)resolved.relativeoffset,
                            usesCurrentValue ? 16u : (unsigned)resolved.stride,
                            (unsigned)vao->attrib[i].size,
                            (unsigned)vao->attrib[i].type,
                            (unsigned)vao->attrib[i].normalized,
                            (unsigned)normalized,
                            usesCurrentValue ? 0u : (unsigned)resolved.divisor,
                            usesCurrentValue ? 0 : (resolved.uses_binding_table ? 1 : 0),
                            (unsigned long)format,
                            mglVertexFormatName(format));
            }

	        if (kMGLVerbosePipelineLogs) {
	            NSLog(@"MGL VERTEX DESC attrib=%u enabled=%u glBuffer=%u metalIndex=%d bindingOffset=%lld offset=0x%llx stride=%u size=%u type=0x%x normalized=%u divisor=%u table=%d format=%lu(%s)",
	                  i,
	                  usesCurrentValue ? 0u : 1u,
                      usesCurrentValue ? 0u : attribBuffer->name,
                      mapped_buffer_index,
                      usesCurrentValue ? 0ll : (long long)resolved.binding_offset,
                      usesCurrentValue ? 0ull : (unsigned long long)(uintptr_t)resolved.relativeoffset,
                      usesCurrentValue ? 16u : (unsigned)resolved.stride,
                      (unsigned)vao->attrib[i].size,
                      (unsigned)vao->attrib[i].type,
                      (unsigned)normalized,
                      usesCurrentValue ? 0u : (unsigned)resolved.divisor,
                      usesCurrentValue ? 0 : (resolved.uses_binding_table ? 1 : 0),
                      (unsigned long)format,
                      mglVertexFormatName(format));
            }

            if (vao->attrib[i].type == GL_UNSIGNED_BYTE &&
                vao->attrib[i].size == 4 &&
                vao->attrib[i].normalized == GL_FALSE &&
                !normalized) {
                if (kMGLVerbosePipelineLogs) {
                    NSLog(@"MGL VERTEX DESC note: attrib %u uses UBYTE4 non-normalized (format=%lu)",
                          i, (unsigned long)format);
                }
            }
        }

    }

    // clear all dirty bits as they have been translated into a vertex descriptor
    vao->dirty_bits = 0;

    return vertexDescriptor;
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

        MTLColorWriteMask colorMask_i;
        if (!MGL_STATE(ctx)->caps.use_color_mask[i]) {
            colorMask_i = MTLColorWriteMaskAll;
        } else {
            colorMask_i = MTLColorWriteMaskNone;
            if (MGL_STATE(ctx)->var.color_writemask[i][0]) colorMask_i |= MTLColorWriteMaskRed;
            if (MGL_STATE(ctx)->var.color_writemask[i][1]) colorMask_i |= MTLColorWriteMaskGreen;
            if (MGL_STATE(ctx)->var.color_writemask[i][2]) colorMask_i |= MTLColorWriteMaskBlue;
            if (MGL_STATE(ctx)->var.color_writemask[i][3]) colorMask_i |= MTLColorWriteMaskAlpha;
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
            colorMask_i |= MTLColorWriteMaskAlpha;
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

static inline BOOL MGLBlendFactorIsDualSource(MTLBlendFactor f)
{
    switch (f) {
        case MTLBlendFactorSource1Color:
        case MTLBlendFactorOneMinusSource1Color:
        case MTLBlendFactorSource1Alpha:
        case MTLBlendFactorOneMinusSource1Alpha:
            return YES;
        default:
            return NO;
    }
}

-(void)bindBlendStateToPipelineStateDescriptor:(MTLRenderPipelineDescriptor *)pipelineStateDescriptor
{
    pipelineStateDescriptor.alphaToCoverageEnabled = MGL_STATE(ctx)->caps.sample_alpha_to_coverage ? YES : NO;
    pipelineStateDescriptor.alphaToOneEnabled = MGL_STATE(ctx)->caps.sample_alpha_to_one ? YES : NO;

    BOOL needsDualSource = NO;
    NSUInteger activeColorAttachmentCount = 0;

    for(int i=0; i<MAX_COLOR_ATTACHMENTS; i++)
    {
        if (pipelineStateDescriptor.colorAttachments[i].pixelFormat != MTLPixelFormatInvalid)
        {
            activeColorAttachmentCount++;

            if (mglMetalDrawBufferAt(ctx, (GLuint)i) == GL_NONE) {
                pipelineStateDescriptor.colorAttachments[i].blendingEnabled = NO;
                pipelineStateDescriptor.colorAttachments[i].writeMask = 0;
                continue;
            }

            pipelineStateDescriptor.colorAttachments[i].blendingEnabled =
                MGL_STATE(ctx)->caps.blendi[i] ? true : false;

            pipelineStateDescriptor.colorAttachments[i].sourceRGBBlendFactor = _pipelineCache.state->src_blend_rgb_factor[i];
            pipelineStateDescriptor.colorAttachments[i].destinationRGBBlendFactor = _pipelineCache.state->dst_blend_rgb_factor[i];
            pipelineStateDescriptor.colorAttachments[i].sourceAlphaBlendFactor = _pipelineCache.state->src_blend_alpha_factor[i];
            pipelineStateDescriptor.colorAttachments[i].destinationAlphaBlendFactor = _pipelineCache.state->dst_blend_alpha_factor[i];

            pipelineStateDescriptor.colorAttachments[i].rgbBlendOperation = _pipelineCache.state->rgb_blend_operation[i];
            pipelineStateDescriptor.colorAttachments[i].alphaBlendOperation = _pipelineCache.state->alpha_blend_operation[i];

            pipelineStateDescriptor.colorAttachments[i].writeMask = _pipelineCache.state->color_mask[i];

            if (!needsDualSource &&
                (MGLBlendFactorIsDualSource(_pipelineCache.state->src_blend_rgb_factor[i]) ||
                 MGLBlendFactorIsDualSource(_pipelineCache.state->dst_blend_rgb_factor[i]) ||
                 MGLBlendFactorIsDualSource(_pipelineCache.state->src_blend_alpha_factor[i]) ||
                 MGLBlendFactorIsDualSource(_pipelineCache.state->dst_blend_alpha_factor[i]))) {
                needsDualSource = YES;
            }
        }
    }

    /* dualSourceBlendingEnabled is a pipeline-level property on classic Metal
     * (macOS 10.11+). On Metal 4 (macOS 26+) the property was removed and
     * dual-source blending is enabled implicitly when MTLBlendFactorSource1*
     * factors are used, so set it via runtime introspection only when the
     * setter exists. Metal restricts dual-source blending to a single color
     * attachment; warn but still attempt to enable so Metal reports the error. */
    if (needsDualSource) {
        if (activeColorAttachmentCount > 1) {
            NSLog(@"MGL WARNING: dual-source blending enabled with %lu color "
                  @"attachments; Metal limits dual-source blending to a single "
                  @"color attachment.",
                  (unsigned long)activeColorAttachmentCount);
        }
        SEL setDualSourceSel = @selector(setDualSourceBlendingEnabled:);
        if ([pipelineStateDescriptor respondsToSelector:setDualSourceSel]) {
            ((void(*)(id, SEL, BOOL))objc_msgSend)(pipelineStateDescriptor,
                                                   setDualSourceSel, YES);
        }
    }
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
