/*
 * mgl_vertex_format.m
 * MGL
 *
 * Implementation of the Vertex Format / Pipeline Signature Subsystem.
 * See mgl_vertex_format.h for the API contract.
 *
 * Pure helpers for GL→Metal vertex format translation and pipeline state
 * signature computation.  No renderer state dependency.
 */

#import "mgl_vertex_format.h"

#include "mgl_air_loader.h"   /* MGLRenderCppPipelineDescriptorState */

#include <string.h>

/* === Vertex format mapping (extern) === */

const char *mglVertexFormatName(MTLVertexFormat format)
{
    switch (format) {
        case MTLVertexFormatFloat: return "Float";
        case MTLVertexFormatFloat2: return "Float2";
        case MTLVertexFormatFloat3: return "Float3";
        case MTLVertexFormatFloat4: return "Float4";
        case MTLVertexFormatUChar4: return "UChar4";
        case MTLVertexFormatUChar4Normalized: return "UChar4Normalized";
        case MTLVertexFormatUChar3: return "UChar3";
        case MTLVertexFormatUChar3Normalized: return "UChar3Normalized";
        case MTLVertexFormatUChar2: return "UChar2";
        case MTLVertexFormatUChar2Normalized: return "UChar2Normalized";
        case MTLVertexFormatUChar: return "UChar";
        case MTLVertexFormatUCharNormalized: return "UCharNormalized";
        case MTLVertexFormatShort: return "Short";
        case MTLVertexFormatShort2: return "Short2";
        case MTLVertexFormatShort3: return "Short3";
        case MTLVertexFormatShort4: return "Short4";
        case MTLVertexFormatShortNormalized: return "ShortNormalized";
        case MTLVertexFormatShort2Normalized: return "Short2Normalized";
        case MTLVertexFormatShort3Normalized: return "Short3Normalized";
        case MTLVertexFormatShort4Normalized: return "Short4Normalized";
        case MTLVertexFormatUShort: return "UShort";
        case MTLVertexFormatUShort2: return "UShort2";
        case MTLVertexFormatUShort3: return "UShort3";
        case MTLVertexFormatUShort4: return "UShort4";
        case MTLVertexFormatUShortNormalized: return "UShortNormalized";
        case MTLVertexFormatUShort2Normalized: return "UShort2Normalized";
        case MTLVertexFormatUShort3Normalized: return "UShort3Normalized";
        case MTLVertexFormatUShort4Normalized: return "UShort4Normalized";
        case MTLVertexFormatUInt1010102Normalized: return "UInt1010102Normalized";
        case MTLVertexFormatInt1010102Normalized: return "Int1010102Normalized";
        default: return "Unknown";
    }
}

bool mglIntegerAttribNeedsConversion(GLenum srcType,
                                     GLuint shaderGlType,
                                     GLuint size,
                                     MTLVertexFormat *outFormat)
{
    if (outFormat) {
        *outFormat = MTLVertexFormatInvalid;
    }
    if (size < 1u || size > 4u) {
        return false;
    }

    bool shaderIsInt = (shaderGlType == GL_INT || shaderGlType == GL_INT_VEC2 ||
                        shaderGlType == GL_INT_VEC3 || shaderGlType == GL_INT_VEC4);
    bool shaderIsUint = (shaderGlType == GL_UNSIGNED_INT ||
                         shaderGlType == GL_UNSIGNED_INT_VEC2 ||
                         shaderGlType == GL_UNSIGNED_INT_VEC3 ||
                         shaderGlType == GL_UNSIGNED_INT_VEC4);
    if (!shaderIsInt && !shaderIsUint) {
        return false;
    }

    bool srcUnsigned = (srcType == GL_UNSIGNED_BYTE ||
                        srcType == GL_UNSIGNED_SHORT ||
                        srcType == GL_UNSIGNED_INT);
    bool srcSigned = (srcType == GL_BYTE || srcType == GL_SHORT || srcType == GL_INT);

    bool needConv = (shaderIsInt && srcUnsigned) || (shaderIsUint && srcSigned);
    if (!needConv) {
        return false;
    }

    MTLVertexFormat f = MTLVertexFormatInvalid;
    if (shaderIsInt) {
        switch (size) {
            case 1: f = MTLVertexFormatInt; break;
            case 2: f = MTLVertexFormatInt2; break;
            case 3: f = MTLVertexFormatInt3; break;
            case 4: f = MTLVertexFormatInt4; break;
        }
    } else {
        switch (size) {
            case 1: f = MTLVertexFormatUInt; break;
            case 2: f = MTLVertexFormatUInt2; break;
            case 3: f = MTLVertexFormatUInt3; break;
            case 4: f = MTLVertexFormatUInt4; break;
        }
    }
    if (outFormat) {
        *outFormat = f;
    }
    return f != MTLVertexFormatInvalid;
}

double mglDecodeVertexAttribComponent(const uint8_t *src,
                                      GLenum type,
                                      GLboolean normalized,
                                      NSUInteger component)
{
    if (!src) {
        return 0.0;
    }

    switch (type) {
        case GL_FLOAT: {
            float v = 0.0f;
            memcpy(&v, src + component * sizeof(float), sizeof(v));
            return (double)v;
        }
        case GL_UNSIGNED_BYTE: {
            uint8_t v = 0;
            memcpy(&v, src + component, sizeof(v));
            return normalized ? ((double)v / 255.0) : (double)v;
        }
        case GL_BYTE: {
            int8_t v = 0;
            memcpy(&v, src + component, sizeof(v));
            if (normalized) {
                double d = (double)v / 127.0;
                return d < -1.0 ? -1.0 : d;
            }
            return (double)v;
        }
        case GL_UNSIGNED_SHORT: {
            uint16_t v = 0;
            memcpy(&v, src + component * sizeof(uint16_t), sizeof(v));
            return normalized ? ((double)v / 65535.0) : (double)v;
        }
        case GL_SHORT: {
            int16_t v = 0;
            memcpy(&v, src + component * sizeof(int16_t), sizeof(v));
            if (normalized) {
                double d = (double)v / 32767.0;
                return d < -1.0 ? -1.0 : d;
            }
            return (double)v;
        }
        case GL_UNSIGNED_INT: {
            uint32_t v = 0;
            memcpy(&v, src + component * sizeof(uint32_t), sizeof(v));
            return normalized ? ((double)v / 4294967295.0) : (double)v;
        }
        case GL_INT: {
            int32_t v = 0;
            memcpy(&v, src + component * sizeof(int32_t), sizeof(v));
            if (normalized) {
                double d = (double)v / 2147483647.0;
                return d < -1.0 ? -1.0 : d;
            }
            return (double)v;
        }
        default:
            return 0.0;
    }
}

/* === Pipeline signature === */

uint64_t mglVertexDescriptorSignature(MTLVertexDescriptor *vertexDescriptor)
{
    uint64_t hash = 1469598103934665603ull;
    if (!vertexDescriptor) {
        return hash;
    }

    for (NSUInteger i = 0; i < MAX_ATTRIBS; i++) {
        MTLVertexAttributeDescriptor *attrib = vertexDescriptor.attributes[i];
        if (!attrib) {
            continue;
        }
        hash = mglHashStepU64(hash, (uint64_t)attrib.format);
        hash = mglHashStepU64(hash, (uint64_t)attrib.offset);
        hash = mglHashStepU64(hash, (uint64_t)attrib.bufferIndex);
    }

    /* kMGLMaxMetalVertexBufferCount = 31 (Metal vertex buffer slots 0..30).
     * Referenced as literal to avoid a circular include with MGLRenderer.m,
     * which defines its own static const version.  See mgl_buffer_slots.h. */
    for (NSUInteger i = 0; i < 31u; i++) {
        MTLVertexBufferLayoutDescriptor *layout = vertexDescriptor.layouts[i];
        if (!layout) {
            continue;
        }
        hash = mglHashStepU64(hash, (uint64_t)layout.stride);
        hash = mglHashStepU64(hash, (uint64_t)layout.stepFunction);
        hash = mglHashStepU64(hash, (uint64_t)layout.stepRate);
    }

    return hash;
}

uint64_t mglPipelineDescriptorSignature(MTLRenderPipelineDescriptor *pipelineStateDescriptor)
{
    uint64_t hash = 1469598103934665603ull;
    if (!pipelineStateDescriptor) {
        return hash;
    }

    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.rasterSampleCount);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.rasterizationEnabled);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.alphaToCoverageEnabled);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.alphaToOneEnabled);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.depthAttachmentPixelFormat);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.stencilAttachmentPixelFormat);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.tessellationPartitionMode);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.maxTessellationFactor);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.tessellationFactorScaleEnabled);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.tessellationFactorFormat);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.tessellationControlPointIndexType);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.tessellationFactorStepFunction);
    hash = mglHashStepU64(hash, (uint64_t)pipelineStateDescriptor.tessellationOutputWindingOrder);

    for (NSUInteger i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        MTLRenderPipelineColorAttachmentDescriptor *attachment = pipelineStateDescriptor.colorAttachments[i];
        if (!attachment) {
            continue;
        }
        hash = mglHashStepU64(hash, (uint64_t)attachment.pixelFormat);
        hash = mglHashStepU64(hash, (uint64_t)attachment.blendingEnabled);
        hash = mglHashStepU64(hash, (uint64_t)attachment.sourceRGBBlendFactor);
        hash = mglHashStepU64(hash, (uint64_t)attachment.destinationRGBBlendFactor);
        hash = mglHashStepU64(hash, (uint64_t)attachment.rgbBlendOperation);
        hash = mglHashStepU64(hash, (uint64_t)attachment.sourceAlphaBlendFactor);
        hash = mglHashStepU64(hash, (uint64_t)attachment.destinationAlphaBlendFactor);
        hash = mglHashStepU64(hash, (uint64_t)attachment.alphaBlendOperation);
        hash = mglHashStepU64(hash, (uint64_t)attachment.writeMask);
    }

    return hash;
}

MTLWinding mglMaybeInvertMTLWinding(MTLWinding winding, BOOL invert)
{
    if (!invert) {
        return winding;
    }
    return (winding == MTLWindingClockwise)
        ? MTLWindingCounterClockwise
        : MTLWindingClockwise;
}

/* === P4.2: value-state signatures ===
 *
 * 哈希字段与顺序与 descriptor 版完全一致：mglPipelineDescriptorSignature 先
 * 哈希 pipeline 级字段，再逐 color attachment 哈希；mglVertexDescriptorSignature
 * 先哈希 32 个 attribute，再哈希 31 个 layout。layout 的最终值取「生成顺序里
 * 最后写它的 attribute」（attribute 索引升序即生成顺序），与 ObjC descriptor
 * 的累积写入语义一致。 */

uint64_t mglVertexDescriptorSignatureFromState(
    const MGLRenderCppPipelineDescriptorState *state)
{
    uint64_t hash = 1469598103934665603ull;
    if (!state) {
        return hash;
    }

    uint32_t layoutStride[31] = {0};
    uint32_t layoutStepFunction[31] = {0};  /* PerVertex = 0 */
    uint32_t layoutStepRate[31] = {0};
    for (uint32_t i = 0; i < 31u; i++) {
        layoutStepRate[i] = 1u;
    }

    for (uint32_t i = 0; i < MAX_ATTRIBS; i++) {
        hash = mglHashStepU64(hash, (uint64_t)state->attrib_format[i]);
        hash = mglHashStepU64(hash, (uint64_t)state->attrib_offset[i]);
        hash = mglHashStepU64(hash, (uint64_t)state->attrib_buffer_index[i]);
        uint32_t bi = state->attrib_buffer_index[i] < 31u
            ? state->attrib_buffer_index[i] : 0u;
        /* 只有 format 有效（非 Invalid）的 attribute 才在 ObjC 生成路径里写
         * layout；未写过的 layout 保持默认 (0/PerVertex/1)。 */
        if (state->attrib_format[i] != (uint32_t)MTLVertexFormatInvalid) {
            layoutStride[bi] = state->attrib_stride[i];
            layoutStepFunction[bi] = state->attrib_step_function[i];
            layoutStepRate[bi] = state->attrib_step_rate[i];
        }
    }

    for (uint32_t i = 0; i < 31u; i++) {
        hash = mglHashStepU64(hash, (uint64_t)layoutStride[i]);
        hash = mglHashStepU64(hash, (uint64_t)layoutStepFunction[i]);
        hash = mglHashStepU64(hash, (uint64_t)layoutStepRate[i]);
    }

    return hash;
}

uint64_t mglPipelineDescriptorSignatureFromState(
    const MGLRenderCppPipelineDescriptorState *state)
{
    uint64_t hash = 1469598103934665603ull;
    if (!state) {
        return hash;
    }

    hash = mglHashStepU64(hash, (uint64_t)state->raster_sample_count);
    hash = mglHashStepU64(hash, (uint64_t)state->rasterization_enabled);
    hash = mglHashStepU64(hash, (uint64_t)state->alpha_to_coverage_enabled);
    hash = mglHashStepU64(hash, (uint64_t)state->alpha_to_one_enabled);
    hash = mglHashStepU64(hash, (uint64_t)state->depth_format);
    hash = mglHashStepU64(hash, (uint64_t)state->stencil_format);
    hash = mglHashStepU64(hash, (uint64_t)state->tessellation_partition_mode);
    hash = mglHashStepU64(hash, (uint64_t)state->max_tessellation_factor);
    hash = mglHashStepU64(hash, (uint64_t)state->tessellation_factor_scale_enabled);
    hash = mglHashStepU64(hash, (uint64_t)state->tessellation_factor_format);
    hash = mglHashStepU64(hash, (uint64_t)state->tessellation_control_point_index_type);
    hash = mglHashStepU64(hash, (uint64_t)state->tessellation_factor_step_function);
    hash = mglHashStepU64(hash, (uint64_t)state->tessellation_output_winding_order);

    for (uint32_t i = 0; i < MAX_COLOR_ATTACHMENTS; i++) {
        hash = mglHashStepU64(hash, (uint64_t)state->color_format[i]);
        hash = mglHashStepU64(hash,
            (state->blending_enabled_mask & (1u << i)) ? 1ull : 0ull);
        hash = mglHashStepU64(hash, (uint64_t)state->source_rgb_blend_factor[i]);
        hash = mglHashStepU64(hash, (uint64_t)state->destination_rgb_blend_factor[i]);
        hash = mglHashStepU64(hash, (uint64_t)state->rgb_blend_operation[i]);
        hash = mglHashStepU64(hash, (uint64_t)state->source_alpha_blend_factor[i]);
        hash = mglHashStepU64(hash, (uint64_t)state->destination_alpha_blend_factor[i]);
        hash = mglHashStepU64(hash, (uint64_t)state->alpha_blend_operation[i]);
        hash = mglHashStepU64(hash, (uint64_t)state->color_write_mask[i]);
    }

    return hash;
}
