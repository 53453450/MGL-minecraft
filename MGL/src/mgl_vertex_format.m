/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */



#import "mgl_vertex_format.h"
#import "mgl_render.h"

#include "mgl_air_loader.h"   /* MGLRenderPipelineDescriptorState */

#include <string.h>

/* === Vertex format mapping (extern) === */

const char *mglVertexFormatName(uint32_t format)
{
    return mglRenderVertexFormatName(format);
}

bool mglIntegerAttribNeedsConversion(GLenum srcType,
                                     GLuint shaderGlType,
                                     GLuint size,
                                     void *outFormat)
{
    uint32_t format = mglRenderIntegerAttribConversionFormat(
        (uint64_t)srcType, (uint64_t)shaderGlType, (uint32_t)size);
    if (outFormat) {
        *(uint32_t *)outFormat = format;
    }
    return format != 0u;
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

uint64_t mglVertexDescriptorSignature(const void *vertexDescriptor)
{
    return mglRenderVertexDescriptorSignature(vertexDescriptor);
}

uint64_t mglPipelineDescriptorSignature(const void *pipelineStateDescriptor)
{
    return mglRenderPipelineDescriptorSignature(pipelineStateDescriptor);
}

uint32_t mglMaybeInvertMTLWinding(uint32_t winding, bool invert)
{
    if (!invert) {
        return winding;
    }
    return winding == 0u ? 1u : 0u;
}



uint64_t mglVertexDescriptorSignatureFromState(
    const MGLRenderPipelineDescriptorState *state)
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

        if (state->attrib_format[i] != 0u) {
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
    const MGLRenderPipelineDescriptorState *state)
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
