/* SPDX-License-Identifier: LGPL-3.0-only */
#include "mgl_metal.h"
#include "mgl_render.h"
#include "mgl_render_pixel.h"

extern "C" {
#include "pixel_utils.h"
}
#include "mgl_program_resource.h"
#include "mgl_renderer_backend.h"
#include "mgl_air_loader.h"
#include "mgl_air_tess_abi.h"
#include "mgl_aux_assets.h"
#include "mgl_compute_pipeline_cache.h"
#include "mgl_env_flag.h"
#include "mgl_program_reflection.h"
#include "mgl_types_buffer.h"
#include "mgl_types_texture.h"
#include "mgl_types_program.h"
#include "mgl_types_state.h"
#include "mgl_types_sync.h"
#include "glm_context.h"
#include "mgl_capability.h"
#include "mgl_sync.h"
#include "glm_limits.h"
#include "mgl_shader_abi.h"
#include "mgl_buffer_slots.h"
#include "mgl_buffer_plan.h"
#include "mgl_tess_domain.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <set>
#include <chrono>
#include <tuple>
#include <utility>
#include <vector>

#include <mach/mach.h>
#include <Block.h>
#include <objc/runtime.h>

extern "C" {
#include "pixel_utils.h"
}

#include <cmath>
#include <cstring>
#include <cstdint>

#include "glcorearb.h"

float mglReadbackMissingChannelFloat(int src_channel_idx)
{
    return (src_channel_idx == 3) ? 1.0f : 0.0f;
}

float mglRead16or32SourceFloat(const uint8_t* s, int idx,
                                         int is16u, int is16s, int is16f) {
    if (is16u) {
        uint16_t uv = 0;
        memcpy(&uv, s + (uint64_t)idx * 2u, sizeof(uv));
        return (float)uv / 65535.0f;
    }
    if (is16s) {
        int16_t sv = 0;
        memcpy(&sv, s + (uint64_t)idx * 2u, sizeof(sv));
        return (float)sv / 32767.0f;
    }
    if (is16f) {
        uint16_t hv = 0;
        memcpy(&hv, s + (uint64_t)idx * 2u, sizeof(hv));
        return mglHalfToFloat(hv);
    }
    float fv = 0.0f;
    memcpy(&fv, s + (uint64_t)idx * 4u, sizeof(fv));
    return fv;
}

uint8_t mglExpandUNormBitsTo8(uint32_t value, uint32_t bits) {
    if (bits == 0u) return 0u;
    if (bits >= 8u) return (uint8_t)(value >> (bits - 8u));
    uint32_t maxv = (1u << bits) - 1u;
    return (uint8_t)((value * 255u + (maxv / 2u)) / maxv);
}

uint32_t mglRenderDepth24Stencil8ToFloatBits(const uint8_t* src) {
    uint32_t packed = 0u;
    memcpy(&packed, src, sizeof(uint32_t));
    const float depth = (float)((double)(packed >> 8u) / 16777215.0);
    uint32_t bits = 0u;
    memcpy(&bits, &depth, sizeof(bits));
    return bits;
}

uint32_t mglRenderDepthUint32ToFloatBits(uint32_t raw) {
    const float depth = (float)((double)raw / 4294967295.0);
    uint32_t bits = 0u;
    memcpy(&bits, &depth, sizeof(bits));
    return bits;
}

uint32_t mglRenderDepth24ToFloatBits(const uint8_t* src) {
    const uint32_t v = (uint32_t)src[0] |
                       ((uint32_t)src[1] << 8u) |
                       ((uint32_t)src[2] << 16u);
    const float depth = (float)((double)v / 16777215.0);
    uint32_t bits = 0u;
    memcpy(&bits, &depth, sizeof(bits));
    return bits;
}

uint8_t mglRenderResolveR8SnormSwizzledComponent(uint32_t swizzle,
                                                        uint8_t red) {
    switch (swizzle) {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 0x7fu;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0x00u;
    }
}

uint16_t mglRenderResolveR16UnormSwizzledComponent(uint32_t swizzle,
                                                          uint16_t red) {
    switch (swizzle) {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 65535u;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0u;
    }
}

uint16_t mglRenderResolveR16SnormSwizzledComponent(uint32_t swizzle,
                                                          int16_t red) {
    switch (swizzle) {
        case GL_RED: return static_cast<uint16_t>(red);
        case GL_ALPHA:
        case GL_ONE: return 32767;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0;
    }
}

uint16_t mglRenderResolveR16FloatSwizzledComponent(uint32_t swizzle,
                                                          uint16_t red) {
    switch (swizzle) {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 0x3c00u; /* 1.0 in half float */
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0u;
    }
}

uint32_t mglRenderResolveR32FloatSwizzledComponent(uint32_t swizzle,
                                                          uint32_t red) {
    switch (swizzle) {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 0x3f800000u;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0u;
    }
}

int64_t mglRenderResolveIntegerSwizzledComponent(
    uint32_t swizzle, int64_t red, int64_t green, int64_t blue,
    int64_t alpha, uint32_t components) {
    switch (swizzle) {
        case GL_RED:
            return components >= 1u ? red : 0;
        case GL_GREEN:
            return components >= 2u ? green : 0;
        case GL_BLUE:
            return components >= 3u ? blue : 0;
        case GL_ALPHA:
            return components >= 4u ? alpha : 1;
        case GL_ONE:
            return 1;
        case GL_ZERO:
        default:
            return 0;
    }
}

int64_t mglRenderReadIntegerTexelComponent(
    const uint8_t* texel, uint32_t component, uint32_t component_bytes,
    int is_signed) {
    const uint8_t* p = texel + component * component_bytes;
    if (component_bytes == 1u) {
        return is_signed ? (int64_t)(int8_t)p[0] : (int64_t)p[0];
    }
    if (component_bytes == 2u) {
        if (is_signed) {
            return (int64_t) * (const int16_t*)(const void*)p;
        }
        return (int64_t) * (const uint16_t*)(const void*)p;
    }
    if (is_signed) {
        return (int64_t) * (const int32_t*)(const void*)p;
    }
    return (int64_t) * (const uint32_t*)(const void*)p;
}

void mglRenderWriteIntegerTexelComponent(
    uint8_t* texel, uint32_t component, uint32_t component_bytes,
    int is_signed, int64_t value) {
    uint8_t* p = texel + component * component_bytes;
    if (component_bytes == 1u) {
        if (is_signed) {
            int32_t clamped = (int32_t)value;
            if (clamped > 127) clamped = 127;
            if (clamped < -128) clamped = -128;
            p[0] = (uint8_t)(int8_t)clamped;
        } else {
            uint32_t clamped =
                value < 0 ? 0u :
                value > 255 ? 255u : (uint32_t)value;
            p[0] = (uint8_t)clamped;
        }
        return;
    }
    if (component_bytes == 2u) {
        if (is_signed) {
            int32_t clamped = (int32_t)value;
            if (clamped > 32767) clamped = 32767;
            if (clamped < -32768) clamped = -32768;
            *(int16_t*)(void*)p = (int16_t)clamped;
        } else {
            uint32_t clamped =
                value < 0 ? 0u :
                value > 65535 ? 65535u : (uint32_t)value;
            *(uint16_t*)(void*)p = (uint16_t)clamped;
        }
        return;
    }
    if (is_signed) {
        *(int32_t*)(void*)p = (int32_t)value;
    } else {
        uint64_t clamped =
            value < 0 ? 0ull :
            (uint64_t)value > 0xffffffffull ? 0xffffffffull :
            (uint64_t)value;
        *(uint32_t*)(void*)p = (uint32_t)clamped;
    }
}

extern "C"
int mglRenderPixelFormatIsIntegerColor(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatR8Uint:
        case MTL::PixelFormatR8Sint:
        case MTL::PixelFormatR16Uint:
        case MTL::PixelFormatR16Sint:
        case MTL::PixelFormatR32Uint:
        case MTL::PixelFormatR32Sint:
        case MTL::PixelFormatRG8Uint:
        case MTL::PixelFormatRG8Sint:
        case MTL::PixelFormatRG16Uint:
        case MTL::PixelFormatRG16Sint:
        case MTL::PixelFormatRG32Uint:
        case MTL::PixelFormatRG32Sint:
        case MTL::PixelFormatRGBA8Uint:
        case MTL::PixelFormatRGBA8Sint:
        case MTL::PixelFormatRGBA16Uint:
        case MTL::PixelFormatRGBA16Sint:
        case MTL::PixelFormatRGBA32Uint:
        case MTL::PixelFormatRGBA32Sint:
        case MTL::PixelFormatRGB10A2Uint:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderPixelFormatIsSignedIntegerColor(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatR8Sint:
        case MTL::PixelFormatR16Sint:
        case MTL::PixelFormatR32Sint:
        case MTL::PixelFormatRG8Sint:
        case MTL::PixelFormatRG16Sint:
        case MTL::PixelFormatRG32Sint:
        case MTL::PixelFormatRGBA8Sint:
        case MTL::PixelFormatRGBA16Sint:
        case MTL::PixelFormatRGBA32Sint:
            return 1;
        default:
            return 0;
    }
}

extern "C" int mglRenderRGBExpandParams(uint32_t pixel_format,
                                       uint32_t *src_comp_bytes,
                                       uint32_t *dst_comp_bytes,
                                       uint64_t *alpha_default) {
    uint32_t src = 0u;
    uint32_t dst = 0u;
    uint64_t alpha = 0u;
    switch ((MTL::PixelFormat)pixel_format) {
    case MTL::PixelFormatRGBA16Unorm:
        src = 2u;
        dst = 2u;
        alpha = 65535u; /* 1.0 in unorm16 */
        break;
    case MTL::PixelFormatRGBA16Snorm:
        src = 2u;
        dst = 2u;
        alpha = 32767u; /* 1.0 in snorm16 */
        break;
    case MTL::PixelFormatRGBA16Float:
        src = 2u;
        dst = 2u;
        alpha = 0x3C00u; /* 1.0 in half float */
        break;
    case MTL::PixelFormatRGBA16Sint:
    case MTL::PixelFormatRGBA16Uint:
        src = 2u;
        dst = 2u;
        alpha = 1u;
        break;
    case MTL::PixelFormatRGBA32Float: {
        src = 4u;
        dst = 4u;
        float f = 1.0f;
        memcpy(&alpha, &f, sizeof(f));
        break;
    }
    case MTL::PixelFormatRGBA32Sint:
    case MTL::PixelFormatRGBA32Uint:
        src = 4u;
        dst = 4u;
        alpha = 1u;
        break;
    default:
        return 0;
    }
    if (src_comp_bytes) {
        *src_comp_bytes = src;
    }
    if (dst_comp_bytes) {
        *dst_comp_bytes = dst;
    }
    if (alpha_default) {
        *alpha_default = alpha;
    }
    return 1;
}

uint32_t mglRenderResolveUploadSwizzlePixelFormat(
    uint32_t native, int single_ch, uint32_t single_fmt, int int_multi,
    uint32_t int_fmt, int stencil, uint32_t stencil_fmt, int ds_depth,
    uint32_t ds_fmt) {
    if (single_ch) {
        return single_fmt != 0u ? single_fmt : 70u; /* RGBA8Unorm */
    }
    if (int_multi) {
        return int_fmt != 0u ? int_fmt : native;
    }
    if (stencil) {
        return stencil_fmt;
    }
    if (ds_depth) {
        return ds_fmt != 0u ? ds_fmt : native;
    }
    return native;
}

uint32_t mglRenderIntegerFormatComponentMap(uint32_t format, int map[4]) {
    uint32_t components = 4u;
    int m0 = 0, m1 = 1, m2 = 2, m3 = 3;
    switch (format) {
    case GL_RED_INTEGER:
        components = 1u;
        m0 = 0;
        m1 = -1;
        m2 = -1;
        m3 = -1;
        break;
    case GL_RG_INTEGER:
        components = 2u;
        m0 = 0;
        m1 = 1;
        m2 = -1;
        m3 = -1;
        break;
    case GL_RGB_INTEGER:
        components = 3u;
        m0 = 0;
        m1 = 1;
        m2 = 2;
        m3 = -1;
        break;
    case GL_BGR_INTEGER:
        components = 3u;
        m0 = 2;
        m1 = 1;
        m2 = 0;
        m3 = -1;
        break;
    case GL_RGBA_INTEGER:
        components = 4u;
        m0 = 0;
        m1 = 1;
        m2 = 2;
        m3 = 3;
        break;
    case GL_BGRA_INTEGER:
        components = 4u;
        m0 = 2;
        m1 = 1;
        m2 = 0;
        m3 = 3;
        break;
    case 0x8d95: /* GL_GREEN_INTEGER */
        components = 1u;
        m0 = 1;
        m1 = -1;
        m2 = -1;
        m3 = -1;
        break;
    case 0x8d96: /* GL_BLUE_INTEGER */
        components = 1u;
        m0 = 2;
        m1 = -1;
        m2 = -1;
        m3 = -1;
        break;
    case 0x8d97: /* GL_ALPHA_INTEGER */
        components = 1u;
        m0 = 3;
        m1 = -1;
        m2 = -1;
        m3 = -1;
        break;
    default:
        components = 4u;
        break;
    }
    if (map) {
        map[0] = m0;
        map[1] = m1;
        map[2] = m2;
        map[3] = m3;
    }
    return components;
}

uint32_t mglRenderIntegerTypeComponentBytes(uint32_t type) {
    if (type == GL_BYTE || type == GL_UNSIGNED_BYTE) {
        return 1u;
    }
    if (type == GL_SHORT || type == GL_UNSIGNED_SHORT) {
        return 2u;
    }
    return 4u;
}

int mglRenderDepth32FStencil8NeedsUnpack(uint32_t internalformat,
                                         uint32_t pixel_format,
                                         uint32_t src_bpr, uint32_t width) {
    return internalformat == GL_DEPTH32F_STENCIL8 && pixel_format == 260u &&
                   src_bpr >= width * 5u && src_bpr < width * 8u
               ? 1
               : 0;
}

int mglRenderDirectR32FloatRead(uint32_t pixel_format, uint32_t format,
                                uint32_t type) {
    return pixel_format == 55u /* R32Float */ && format == GL_RED &&
                   type == GL_FLOAT
               ? 1
               : 0;
}

int mglRenderPixelFormatIsDepth32FloatStencil8(uint32_t pixel_format) {
    return pixel_format == 260u /* Depth32Float_Stencil8 */ ? 1 : 0;
}

const char *mglRenderGLSLColumnSwizzle(uint32_t rows) {
    switch (rows) {
    case 2u:
        return ".xy";
    case 3u:
        return ".xyz";
    case 4u:
        return "";
    default:
        return NULL;
    }
}

const char *mglRenderGLSLTypeSwizzle(uint32_t type) {
    switch (type) {
    case GL_FLOAT:
    case GL_INT:
    case GL_UNSIGNED_INT:
        return ".x";
    case GL_FLOAT_VEC2:
    case GL_INT_VEC2:
    case GL_UNSIGNED_INT_VEC2:
        return ".xy";
    case GL_FLOAT_VEC3:
    case GL_INT_VEC3:
    case GL_UNSIGNED_INT_VEC3:
        return ".xyz";
    case GL_FLOAT_VEC4:
    case GL_INT_VEC4:
    case GL_UNSIGNED_INT_VEC4:
        return "";
    default:
        return NULL;
    }
}

const char *mglRenderGLSLIntegerAsFloatType(uint32_t type) {
    switch (type) {
    case GL_INT:
    case GL_UNSIGNED_INT:
        return "float";
    case GL_INT_VEC2:
    case GL_UNSIGNED_INT_VEC2:
        return "vec2";
    case GL_INT_VEC3:
    case GL_UNSIGNED_INT_VEC3:
        return "vec3";
    case GL_INT_VEC4:
    case GL_UNSIGNED_INT_VEC4:
        return "vec4";
    default:
        return NULL;
    }
}

uint32_t mglRenderStructPackSrcStride(uint32_t src_stride, uint32_t elem_stride) {
    return src_stride ? src_stride : elem_stride;
}

int mglRenderStructPackUseBulk(int32_t ai, int64_t buf_size, uint32_t member_size,
                               uint32_t src_stride) {
    return ai == 0 && buf_size >= 0 &&
                   (uint64_t)buf_size >= (uint64_t)member_size * src_stride
               ? 1
               : 0;
}

int mglRenderShouldPackPlainUniformStruct(int spvc_type, int has_members,
                                          uint32_t member_count,
                                          uint64_t required, int sampler_like) {
    return spvc_type == _UNIFORM_CONSTANT_RES && has_members &&
                   member_count > 0u && required > 0u && !sampler_like
               ? 1
               : 0;
}

void mglRenderPackCurrentAttribPool(const uint8_t *values, uint32_t attrib_count,
                                    uint8_t *dst, uint64_t dst_bytes,
                                    uint32_t repeat_count, uint32_t value_bytes) {
    if (!values || !dst || attrib_count == 0u || repeat_count == 0u ||
        value_bytes == 0u) {
        return;
    }
    const uint64_t stride = (uint64_t)repeat_count * (uint64_t)value_bytes;
    if (stride == 0u || attrib_count > dst_bytes / stride) {
        return;
    }
    const uint32_t copy_bytes = value_bytes < 16u ? value_bytes : 16u;
    for (uint32_t a = 0u; a < attrib_count; a++) {
        uint8_t *seg = dst + (uint64_t)a * stride;
        const uint8_t *src = values + (uint64_t)a * 16u;
        for (uint32_t v = 0u; v < repeat_count; v++) {
            memcpy(seg + (uint64_t)v * value_bytes, src, copy_bytes);
            if (copy_bytes < value_bytes) {
                memset(seg + (uint64_t)v * value_bytes + copy_bytes, 0,
                       value_bytes - copy_bytes);
            }
        }
    }
}

extern "C"
int mglRenderExpandUInt8ToUInt16(
    const uint8_t* bytes, uint32_t byte_count, uint16_t** out, uint64_t* out_count) {
    if (!bytes || byte_count == 0u || !out || !out_count) {
        return -1;
    }
    if ((uint64_t)byte_count > (uint64_t)(SIZE_MAX / sizeof(uint16_t))) {
        return -1;
    }
    uint16_t* const dst = (uint16_t*)malloc((size_t)byte_count * sizeof(uint16_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t i = 0u; i < byte_count; i++) {
        dst[i] = (uint16_t)bytes[i];
    }
    *out = dst;
    *out_count = byte_count;
    return 0;
}

extern "C"
int mglRenderExpandTriangleFanArrayIndices(
    uint32_t vertex_count, uint32_t** out_indices, uint64_t* out_count) {
    if (vertex_count < 3u || !out_indices || !out_count) {
        return -1;
    }
    const uint32_t n = vertex_count - 2u;
    const uint64_t need = (uint64_t)n * 3u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t t = 0u; t < n; t++) {
        dst[t*3u+0u] = 0u;
        dst[t*3u+1u] = t + 1u;
        dst[t*3u+2u] = t + 2u;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandTriangleStripArrayIndices(
    uint32_t vertex_count, uint32_t** out_indices, uint64_t* out_count) {
    if (vertex_count < 3u || !out_indices || !out_count) {
        return -1;
    }
    const uint32_t n = vertex_count - 2u;
    const uint64_t need = (uint64_t)n * 3u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t t = 0u; t < n; t++) {
        dst[t*3u+0u] = t + (t & 1u);
        dst[t*3u+1u] = t + ((t & 1u) ? 0u : 1u);
        dst[t*3u+2u] = t + 2u;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandLineLoopArrayIndices(
    uint32_t first_vertex, uint32_t vertex_count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (vertex_count < 2u || !out_indices || !out_count) {
        return -1;
    }
    if ((uint64_t)first_vertex + (uint64_t)vertex_count >
        (uint64_t)UINT32_MAX + 1u) {
        return -1;
    }
    const uint64_t need = (uint64_t)vertex_count + 1u;
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t i = 0u; i < vertex_count; i++) {
        dst[i] = first_vertex + i;
    }
    dst[vertex_count] = first_vertex;
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
int mglRenderExpandQuadArrayLineIndices(
    uint32_t quad_count, uint32_t** out_indices, uint64_t* out_count) {
    if (quad_count == 0u || !out_indices || !out_count) {
        return -1;
    }
    const uint64_t need = (uint64_t)quad_count * 8u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t q = 0u; q < quad_count; q++) {
        const uint32_t b = q * 4u;
        const uint32_t d = q * 8u;
        if (b + 3u > UINT32_MAX) {
            free(dst);
            return -1;
        }
        dst[d+0]=b+0; dst[d+1]=b+1; dst[d+2]=b+1; dst[d+3]=b+2;
        dst[d+4]=b+2; dst[d+5]=b+3; dst[d+6]=b+3; dst[d+7]=b+0;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}





extern "C"
int mglRenderExpandTriangleFanIndices(
    const uint8_t* bytes, uint32_t elem_width, uint32_t count,
    uint32_t** out_indices, uint64_t* out_count) {
    if (!bytes || count < 3u || !out_indices || !out_count) {
        return -1;
    }
    const uint32_t n = count - 2u;
    const uint64_t need = (uint64_t)n * 3u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    #define RDX(i) ((w == 1) ? (uint32_t)bytes[i]         : (w == 2) ? (uint32_t)((const uint16_t*)bytes)[i]         : (uint32_t)((const uint32_t*)bytes)[i])
    const uint32_t c = RDX(0u);
    for (uint32_t t = 0u; t < n; t++) {
        dst[t*3u+0u] = c;
        dst[t*3u+1u] = RDX(t + 1u);
        dst[t*3u+2u] = RDX(t + 2u);
    }
#undef RDX
    *out_indices = dst;
    *out_count = need;
    return 0;
}

extern "C"
float mglRenderFloat11ToFloat(uint32_t val) {
    if (val == 0u) {
        return 0.0f;
    }
    const uint32_t exp = (val >> 6) & 0x1Fu;
    const uint32_t mant = val & 0x3Fu;
    if (exp == 0u) {
        return (float)((double)mant / 64.0) * (1.0 / 16384.0);
    } else if (exp == 31u) {
        return mant ? NAN : INFINITY;
    }
    return ldexpf((float)(1.0 + (double)mant / 64.0), (int)exp - 15);
}

extern "C"
float mglRenderFloat10ToFloat(uint32_t val) {
    if (val == 0u) {
        return 0.0f;
    }
    const uint32_t exp = (val >> 5) & 0x1Fu;
    const uint32_t mant = val & 0x1Fu;
    if (exp == 0u) {
        return (float)((double)mant / 32.0) * (1.0 / 16384.0);
    } else if (exp == 31u) {
        return mant ? NAN : INFINITY;
    }
    return ldexpf((float)(1.0 + (double)mant / 32.0), (int)exp - 15);
}

extern "C"
uint8_t mglRenderFloatToUnorm8(float value) {
    if (!(value > 0.0f)) {
        return 0u;
    }
    if (value >= 1.0f) {
        return 255u;
    }
    return (uint8_t)(value * 255.0f + 0.5f);
}

extern "C"
float mglRenderSnorm16ToFloat(int16_t value) {
    if (value == INT16_MIN) {
        return -1.0f;
    }
    return (float)value / 32767.0f;
}

extern "C"
float mglRenderSnorm8ToFloat(int8_t value) {
    if (value == INT8_MIN) {
        return -1.0f;
    }
    return (float)value / 127.0f;
}

extern "C"
uint32_t mglRenderSingleChannelSwizzleStoragePixelFormat(
    uint32_t internal_format) {
    switch (internal_format) {
        case GL_R8I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Sint);
        case GL_R16I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Sint);
        case GL_R32I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Sint);
        case GL_R8UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint);
        case GL_R16UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Uint);
        case GL_R32UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Uint);
        case GL_R8:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Unorm);
        case GL_R8_SNORM:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Snorm);
        case GL_R16:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Unorm);
        case GL_R16_SNORM:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Snorm);
        case GL_R16F:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Float);
        case GL_R32F:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Float);
        case GL_DEPTH_COMPONENT16:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Unorm);
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Float);
        default:
            return static_cast<uint32_t>(MTL::PixelFormatInvalid);
    }
}

extern "C"
uint32_t mglRenderIntegerMultiChannelSwizzleStoragePixelFormat(
    uint32_t internal_format) {
    switch (internal_format) {
        case GL_RG8I:
        case GL_RGB8I:
        case GL_RGBA8I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Sint);
        case GL_RG16I:
        case GL_RGB16I:
        case GL_RGBA16I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Sint);
        case GL_RG32I:
        case GL_RGB32I:
        case GL_RGBA32I:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Sint);
        case GL_RG8UI:
        case GL_RGB8UI:
        case GL_RGBA8UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint);
        case GL_RG16UI:
        case GL_RGB16UI:
        case GL_RGBA16UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA16Uint);
        case GL_RG32UI:
        case GL_RGB32UI:
        case GL_RGBA32UI:
            return static_cast<uint32_t>(MTL::PixelFormatRGBA32Uint);
        default:
            return static_cast<uint32_t>(MTL::PixelFormatInvalid);
    }
}

extern "C"
int mglRenderTextureUploadNeedsDepthStencilDepthSwizzleBake(
    uint32_t internal_format, int swizzled, uint32_t depth_stencil_mode) {
    if (!swizzled || depth_stencil_mode != GL_DEPTH_COMPONENT) {
        return 0;
    }
    switch (internal_format) {
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderTextureUploadNeedsStencilSwizzleBake(
    uint32_t internal_format, int swizzled, uint32_t depth_stencil_mode) {
    if (!swizzled || depth_stencil_mode != GL_STENCIL_INDEX) {
        return 0;
    }
    switch (internal_format) {
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return 1;
        default:
            return 0;
    }
}

extern "C"
uint32_t mglRenderStencilSwizzleStoragePixelFormat(void) {
    return static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint);
}

extern "C"
int mglRenderTextureUploadNeedsSingleChannelSwizzleBake(
    uint32_t internal_format, int swizzled) {
    if (!swizzled) {
        return 0;
    }
    switch (internal_format) {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_R16UI:
        case GL_R32UI:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderTextureUploadNeedsSingleChannelSwizzle(uint32_t internal_format,
                                                       int swizzled) {
    if (!swizzled) {
        return 0;
    }
    switch (internal_format) {
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
            return 1;
        default:
            return 0;
    }
}

extern "C"
uint32_t mglRenderMTLSwizzleForGLSwizzle(uint32_t gl_swizzle,
                                            uint32_t components) {
    switch (gl_swizzle) {
        case GL_ZERO:
            return (uint32_t)MTL::TextureSwizzleZero;
        case GL_ONE:
            return (uint32_t)MTL::TextureSwizzleOne;
        case GL_RED:
            return components >= 1u
                ? (uint32_t)MTL::TextureSwizzleRed
                : (uint32_t)MTL::TextureSwizzleZero;
        case GL_GREEN:
            return components >= 2u
                ? (uint32_t)MTL::TextureSwizzleGreen
                : (uint32_t)MTL::TextureSwizzleZero;
        case GL_BLUE:
            return components >= 3u
                ? (uint32_t)MTL::TextureSwizzleBlue
                : (uint32_t)MTL::TextureSwizzleZero;
        case GL_ALPHA:
            return components >= 4u
                ? (uint32_t)MTL::TextureSwizzleAlpha
                : (uint32_t)MTL::TextureSwizzleOne;
        default:
            fprintf(stderr,
                    "MGL ERROR: Unknown swizzle value 0x%x in swizzleTexDesc\n",
                    gl_swizzle);
            return (uint32_t)MTL::TextureSwizzleZero;
    }
}

int mglRenderTextureExpandRGBToRGBA(const void* src, void* dst,
                                       size_t texel_count, size_t tex_width,
                                       size_t tex_height,
                                       size_t src_comp_bytes,
                                       size_t dst_comp_bytes,
                                       uint64_t alpha_default) {
    if (!src || !dst || tex_width == 0 || tex_height == 0 ||
        src_comp_bytes == 0 || dst_comp_bytes == 0) {
        return -1;
    }
    const size_t src_pixel = src_comp_bytes * 3;
    const size_t dst_pixel = dst_comp_bytes * 4;
    if (src_pixel == 0 || dst_pixel == 0) {
        return -1;
    }
    uint8_t* out = (uint8_t*)dst;
    for (size_t row = 0; row < tex_height; row++) {
        for (size_t col = 0; col < tex_width; col++) {
            const size_t idx = row * tex_width + col;
            uint8_t* dp = out + (row * tex_width + col) * dst_pixel;
            if (idx >= texel_count) {
                memset(dp, 0, dst_pixel);
                continue;
            }
            const uint8_t* sp = (const uint8_t*)src + idx * src_pixel;
            memcpy(dp, sp, src_pixel);
            memcpy(dp + src_pixel, &alpha_default, dst_comp_bytes);
        }
    }
    return 0;
}

extern "C"
uint32_t mglRenderDoubleVertexAttribFloatFormat(uint32_t size) {
    /* MTLVertexFormat Float/Float2/Float3/Float4 = 28/29/30/31. */
    switch (size) {
        case 1u: return 28u;
        case 2u: return 29u;
        case 3u: return 30u;
        case 4u: return 31u;
        default: return 0u; /* MTLVertexFormatInvalid */
    }
}

extern "C"
uint32_t mglRenderIntegerAttribConversionFormat(
    uint64_t src_type,
    uint64_t shader_gl_type,
    uint32_t size) {
    if (size < 1u || size > 4u) {
        return static_cast<uint32_t>(MTL::VertexFormatInvalid);
    }

    const bool shader_is_int =
        shader_gl_type == GL_INT || shader_gl_type == GL_INT_VEC2 ||
        shader_gl_type == GL_INT_VEC3 || shader_gl_type == GL_INT_VEC4;
    const bool shader_is_uint =
        shader_gl_type == GL_UNSIGNED_INT ||
        shader_gl_type == GL_UNSIGNED_INT_VEC2 ||
        shader_gl_type == GL_UNSIGNED_INT_VEC3 ||
        shader_gl_type == GL_UNSIGNED_INT_VEC4;
    if (!shader_is_int && !shader_is_uint) {
        return static_cast<uint32_t>(MTL::VertexFormatInvalid);
    }

    const bool src_is_unsigned =
        src_type == GL_UNSIGNED_BYTE || src_type == GL_UNSIGNED_SHORT ||
        src_type == GL_UNSIGNED_INT;
    const bool src_is_signed =
        src_type == GL_BYTE || src_type == GL_SHORT || src_type == GL_INT;
    if (!((shader_is_int && src_is_unsigned) ||
          (shader_is_uint && src_is_signed))) {
        return static_cast<uint32_t>(MTL::VertexFormatInvalid);
    }

    if (shader_is_int) {
        switch (size) {
            case 1u: return static_cast<uint32_t>(MTL::VertexFormatInt);
            case 2u: return static_cast<uint32_t>(MTL::VertexFormatInt2);
            case 3u: return static_cast<uint32_t>(MTL::VertexFormatInt3);
            case 4u: return static_cast<uint32_t>(MTL::VertexFormatInt4);
        }
    } else {
        switch (size) {
            case 1u: return static_cast<uint32_t>(MTL::VertexFormatUInt);
            case 2u: return static_cast<uint32_t>(MTL::VertexFormatUInt2);
            case 3u: return static_cast<uint32_t>(MTL::VertexFormatUInt3);
            case 4u: return static_cast<uint32_t>(MTL::VertexFormatUInt4);
        }
    }
    return static_cast<uint32_t>(MTL::VertexFormatInvalid);
}

int mglReadbackFormatChannelMap(uint32_t format, int* slots,
                                          int src_idx[4]) {
    if (!slots || !src_idx) return 0;
    src_idx[0] = src_idx[1] = src_idx[2] = src_idx[3] = 0;
    switch (format) {
        case GL_RGBA: *slots = 4; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; src_idx[3]=3; return 1;
        case GL_BGRA: *slots = 4; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; src_idx[3]=3; return 1;
        case GL_RGB:  *slots = 3; src_idx[0]=0; src_idx[1]=1; src_idx[2]=2; return 1;
        case GL_BGR:  *slots = 3; src_idx[0]=2; src_idx[1]=1; src_idx[2]=0; return 1;
        case GL_RG:   *slots = 2; src_idx[0]=0; src_idx[1]=1; return 1;
        case GL_RED:  *slots = 1; src_idx[0]=0; return 1;
        case GL_GREEN: *slots = 1; src_idx[0]=1; return 1;
        case GL_BLUE:  *slots = 1; src_idx[0]=2; return 1;
        case GL_ALPHA: *slots = 1; src_idx[0]=3; return 1;
        default: return 0;
    }
}

uint32_t mglSizeForType(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
            return sizeof(uint8_t);
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
            return sizeof(uint16_t);
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_24_8:
            return sizeof(uint32_t);
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 8u;
        default:
            return sizeof(uint32_t);
    }
}

uint32_t mglNumComponentsForFormat(uint32_t format) {
    switch (format) {
        case GL_RED:
        case GL_RED_INTEGER:
        case GL_GREEN:
        case GL_BLUE:
        case GL_STENCIL_INDEX:
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_STENCIL:
        case GL_ALPHA:
        case 0x803C: /* GL_ALPHA8 */
        case 0x803E: /* GL_ALPHA16 */
        case 0x8816: /* GL_ALPHA32F_ARB */
        case 0x881C: /* GL_ALPHA16F_ARB */
        case 0x1909: /* GL_LUMINANCE */
        case 0x8040: /* GL_LUMINANCE8 */
        case 0x8048: /* GL_LUMINANCE16 (pixel_utils local define) */
        case 0x8818: /* GL_LUMINANCE32F_ARB */
        case 0x881E: /* GL_LUMINANCE16F_ARB */
        case GL_R8:
        case GL_R8_SNORM:
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
        case GL_R32F:
        case GL_R8I:
        case GL_R8UI:
        case GL_R16I:
        case GL_R16UI:
        case GL_R32I:
        case GL_R32UI:
        case GL_SR8_EXT:
        case 0x8D7E: /* GL_ALPHA8UI_EXT */
        case 0x9014: /* GL_ALPHA8_SNORM */
        case 0x9018: /* GL_ALPHA16_SNORM */
            return 1u;

        case GL_RG:
        case GL_RG_INTEGER:
        case 0x190A: /* GL_LUMINANCE_ALPHA */
        case 0x8819: /* GL_LUMINANCE_ALPHA32F_ARB */
        case 0x881F: /* GL_LUMINANCE_ALPHA16F_ARB */
        case 0x9016: /* GL_LUMINANCE8_ALPHA8_SNORM */
        case 0x901a: /* GL_LUMINANCE16_ALPHA16_SNORM */
        case GL_RG8:
        case GL_RG8_SNORM:
        case GL_RG16:
        case GL_RG16_SNORM:
        case GL_RG16F:
        case GL_RG32F:
        case GL_RG8I:
        case GL_RG8UI:
        case GL_RG16I:
        case GL_RG16UI:
        case GL_RG32I:
        case GL_RG32UI:
        case GL_SRG8_EXT:
            return 2u;

        case 0x8d7b: /* GL_ALPHA8I_EXT */
        case 0x8d81: /* GL_ALPHA32I_EXT */
        case 0x8d87: /* GL_ALPHA16I_EXT */
        case 0x8d8d: /* GL_ALPHA32UI_EXT */
        case 0x8d93: /* GL_ALPHA16UI_EXT */
        case 0x8d72: /* GL_ALPHA32UI_EXT */
            return 1u;

        case GL_RGB:
        case GL_BGR:
        case GL_RGB_INTEGER:
        case GL_BGR_INTEGER:
        case GL_RGB8:
        case GL_RGB8_SNORM:
        case GL_SRGB8:
        case GL_RGB16F:
        case GL_RGB32F:
        case GL_R11F_G11F_B10F:
        case GL_RGB9_E5:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGB565:
            return 3u;

        case 0x8d75: /* alternate GL_RGB8I */
        case 0x8d7a: /* alternate GL_RGB8UI */
        case 0x8d80: /* alternate GL_RGB32UI */
        case 0x8d86: /* alternate GL_RGB16I */
        case 0x8d8c: /* alternate GL_RGB32I */
        case 0x8d92: /* alternate GL_RGB16UI */
            return 3u;

        case GL_RGBA:
        case GL_BGRA:
        case GL_RGBA_INTEGER:
        case GL_BGRA_INTEGER:
        case GL_RGBA8:
        case GL_RGBA8_SNORM:
        case GL_SRGB8_ALPHA8:
        case GL_RGBA16F:
        case GL_RGBA32F:
        case GL_RGBA8I:
        case GL_RGBA8UI:
        case GL_RGBA16I:
        case GL_RGBA16UI:
        case GL_RGBA32I:
        case GL_RGBA32UI:
        case GL_RGB10_A2:
        case GL_RGB10_A2UI:
        case GL_RGB5_A1:
        case GL_RGBA4:
            return 4u;

        case 0x8d78: /* alternate GL_RGBA8UI */
        case 0x8d84: /* alternate GL_RGBA16I */
        case 0x8d8a: /* alternate GL_RGBA32I */
        case 0x8d90: /* alternate GL_RGBA16UI */
            return 4u;

        case 0x8d95: /* GL_GREEN_INTEGER */
        case 0x8d96: /* GL_BLUE_INTEGER */
            return 1u;

        default:
            fprintf(stderr,
                    "MGL WARNING: numComponentsForFormat unknown format 0x%x, "
                    "assuming 4 components\n",
                    format);
            return 4u;
    }
}

int mglPixelTypeIsPacked(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_24_8:
        case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
            return 1;
        default:
            return 0;
    }
}

int mglReadbackRGB10A2TypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
            return 1;
        default:
            return 0;
    }
}

int mglReadbackRG11B10TypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
            return 1;
        default:
            return 0;
    }
}

int mglReadback16or32TypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
        case GL_BYTE:
        case GL_UNSIGNED_SHORT:
        case GL_SHORT:
        case GL_UNSIGNED_INT:
        case GL_INT:
        case GL_FLOAT:
        case GL_HALF_FLOAT:
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            return 1;
        default:
            return 0;
    }
}

int mglWideSrcChannelCount(MTL::PixelFormat pf) {
    switch (pf) {
        case MTL::PixelFormatR32Float:
        case MTL::PixelFormatR16Unorm:
        case MTL::PixelFormatR16Snorm:
        case MTL::PixelFormatR16Float:
            return 1;
        case MTL::PixelFormatRG32Float:
        case MTL::PixelFormatRG16Unorm:
        case MTL::PixelFormatRG16Snorm:
        case MTL::PixelFormatRG16Float:
            return 2;
        case MTL::PixelFormatRGBA32Float:
        case MTL::PixelFormatRGBA16Unorm:
        case MTL::PixelFormatRGBA16Snorm:
        case MTL::PixelFormatRGBA16Float:
            return 4;
        default:
            return 0;
    }
}

int mglReadbackUnorm8ScalarTypeAccepted(uint32_t type) {
    switch (type) {
        case GL_BYTE:
        case GL_SHORT:
        case GL_INT:
        case GL_UNSIGNED_INT:
        case GL_UNSIGNED_SHORT:
        case GL_HALF_FLOAT:
        case GL_FLOAT:
            return 1;
        default:
            return 0;
    }
}

int mglReadbackUnorm8PackedTypeAccepted(uint32_t type) {
    switch (type) {
        case GL_UNSIGNED_BYTE_3_3_2:
        case GL_UNSIGNED_BYTE_2_3_3_REV:
        case GL_UNSIGNED_SHORT_5_6_5:
        case GL_UNSIGNED_SHORT_5_6_5_REV:
        case GL_UNSIGNED_SHORT_4_4_4_4:
        case GL_UNSIGNED_SHORT_4_4_4_4_REV:
        case GL_UNSIGNED_SHORT_5_5_5_1:
        case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        case GL_UNSIGNED_INT_8_8_8_8:
        case GL_UNSIGNED_INT_8_8_8_8_REV:
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_2_10_10_10_REV:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_UNSIGNED_INT_5_9_9_9_REV:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderCopyUnorm8SwizzleTextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    const int source_is_rgba =
        (pf == MTL::PixelFormatRGBA8Unorm ||
         pf == MTL::PixelFormatRGBA8Unorm_sRGB);
    const int source_is_bgra =
        (pf == MTL::PixelFormatBGRA8Unorm ||
         pf == MTL::PixelFormatBGRA8Unorm_sRGB);
    if (!source_is_rgba && !source_is_bgra) {
        return 0;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }
    (void)src_idx;

    uint32_t comp_bytes = mglSizeForType(type);
    uint64_t dst_pixel_bytes = mglPixelTypeIsPacked(type)
        ? (uint64_t)comp_bytes
        : (uint64_t)comp_bytes * (uint64_t)slots;
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    if (format == GL_BGRA) {
        if (dst_pixel_bytes != 4u) return 0;
    } else if (format == GL_RGBA) {
        if (dst_pixel_bytes != 4u &&
            !(type == GL_FLOAT && dst_pixel_bytes == 16u)) {
            return 0;
        }
    } else if (format == GL_BGR || format == GL_RGB) {
        if (type != GL_UNSIGNED_BYTE || dst_pixel_bytes != 3u) return 0;
    } else if (format == GL_RG) {
        if (type != GL_UNSIGNED_BYTE || dst_pixel_bytes != 2u) return 0;
    } else {
        if (type != GL_UNSIGNED_BYTE || dst_pixel_bytes != 1u) return 0;
    }

    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            const uint8_t* s = src_row + (x * 4u);
            uint8_t r = source_is_rgba ? s[0] : s[2];
            uint8_t g = s[1];
            uint8_t b = source_is_rgba ? s[2] : s[0];
            uint8_t a = s[3];
            uint8_t* d = dst_row + (x * dst_pixel_bytes);

            switch (format) {
                case GL_BGRA:
                    d[0] = b;
                    d[1] = g;
                    d[2] = r;
                    d[3] = a;
                    break;
                case GL_RGBA:
                    if (type == GL_FLOAT) {
                        float* fd = reinterpret_cast<float*>(d);
                        fd[0] = (float)r / 255.0f;
                        fd[1] = (float)g / 255.0f;
                        fd[2] = (float)b / 255.0f;
                        fd[3] = (float)a / 255.0f;
                    } else {
                        d[0] = r;
                        d[1] = g;
                        d[2] = b;
                        d[3] = a;
                    }
                    break;
                case GL_BGR:
                    d[0] = b;
                    d[1] = g;
                    d[2] = r;
                    break;
                case GL_RGB:
                    d[0] = r;
                    d[1] = g;
                    d[2] = b;
                    break;
                case GL_RG:
                    d[0] = r;
                    d[1] = g;
                    break;
                case GL_RED:
                    d[0] = r;
                    break;
                case GL_GREEN:
                    d[0] = g;
                    break;
                case GL_BLUE:
                    d[0] = b;
                    break;
                case GL_ALPHA:
                    d[0] = a;
                    break;
                default:
                    return 0;
            }
        }
    }
    return 1;
}

int32_t mglRenderResolveR8IntegerSwizzledComponent(
    uint32_t swizzle, int32_t red, int is_signed) {
    (void)is_signed;
    return (int32_t)mglRenderResolveIntegerSwizzledComponent(
        swizzle, red, 0, 0, 1, 1u);
}

extern "C"
int mglRenderTextureUploadNeedsIntegerMultiChannelSwizzleBake(
    uint32_t internal_format, int swizzled) {
    if (!swizzled) {
        return 0;
    }
    uint32_t components = 0;
    uint32_t component_bytes = 0;
    int is_signed = 0;
    if (mglRenderIntegerFormatLayout(
            internal_format, &components, &component_bytes, &is_signed) != 0 ||
        components <= 1u) {
        return 0;
    }
    return 1;
}

int mglRenderPixelFormatMatchesSwizzleBakeStorage(
    uint32_t internal_format, uint32_t storage_pixel_format) {
    const uint32_t expected =
        mglRenderSingleChannelSwizzleStoragePixelFormat(internal_format);
    if (expected != static_cast<uint32_t>(MTL::PixelFormatInvalid) &&
        expected == storage_pixel_format) {
        return 1;
    }
    if (mglRenderTextureUploadNeedsIntegerMultiChannelSwizzleBake(
            internal_format, 1) != 0) {
        /* Multi-channel integer bake keeps the native storage format. */
        switch (internal_format) {
            case GL_RG8I:
            case GL_RGB8I:
            case GL_RGBA8I:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA8Sint);
            case GL_RG8UI:
            case GL_RGB8UI:
            case GL_RGBA8UI:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint);
            case GL_RG16I:
            case GL_RGB16I:
            case GL_RGBA16I:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA16Sint);
            case GL_RG16UI:
            case GL_RGB16UI:
            case GL_RGBA16UI:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA16Uint);
            case GL_RG32I:
            case GL_RGB32I:
            case GL_RGBA32I:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA32Sint);
            case GL_RG32UI:
            case GL_RGB32UI:
            case GL_RGBA32UI:
                return storage_pixel_format ==
                    static_cast<uint32_t>(MTL::PixelFormatRGBA32Uint);
            default:
                break;
        }
    }
    switch (internal_format) {
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return storage_pixel_format ==
                       static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint) ||
                   storage_pixel_format ==
                       static_cast<uint32_t>(MTL::PixelFormatRGBA32Float);
        default:
            break;
    }
    return 0;
}

extern "C"
int mglRenderTextureSwizzleUsesUploadBake(
    uint32_t internal_format, int swizzled, uint32_t storage_pixel_format) {
    if (!swizzled) {
        return 0;
    }
    if (mglRenderTextureUploadNeedsSingleChannelSwizzleBake(
            internal_format, swizzled) != 0) {
        return mglRenderPixelFormatMatchesSwizzleBakeStorage(
            internal_format, storage_pixel_format);
    }
    if (mglRenderTextureUploadNeedsIntegerMultiChannelSwizzleBake(
            internal_format, swizzled) != 0) {
        return mglRenderPixelFormatMatchesSwizzleBakeStorage(
            internal_format, storage_pixel_format);
    }
    switch (internal_format) {
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
            return storage_pixel_format ==
                       static_cast<uint32_t>(MTL::PixelFormatRGBA8Uint) ||
                   storage_pixel_format ==
                       static_cast<uint32_t>(MTL::PixelFormatRGBA32Float);
        default:
            break;
    }
    return 0;
}

int mglRenderIntegerFormatLayout(
    uint32_t internal_format, uint32_t* out_components,
    uint32_t* out_component_bytes, int* out_signed) {
    if (!out_components || !out_component_bytes || !out_signed) {
        return -1;
    }
    *out_components = 0;
    *out_component_bytes = 0;
    *out_signed = 0;
    switch (internal_format) {
        case GL_R8I:
        case GL_RG8I:
        case GL_RGB8I:
        case GL_RGBA8I:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 1u;
            *out_signed = 1;
            return 0;
        case GL_R8UI:
        case GL_RG8UI:
        case GL_RGB8UI:
        case GL_RGBA8UI:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 1u;
            *out_signed = 0;
            return 0;
        case GL_R16I:
        case GL_RG16I:
        case GL_RGB16I:
        case GL_RGBA16I:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 2u;
            *out_signed = 1;
            return 0;
        case GL_R16UI:
        case GL_RG16UI:
        case GL_RGB16UI:
        case GL_RGBA16UI:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 2u;
            *out_signed = 0;
            return 0;
        case GL_R32I:
        case GL_RG32I:
        case GL_RGB32I:
        case GL_RGBA32I:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 4u;
            *out_signed = 1;
            return 0;
        case GL_R32UI:
        case GL_RG32UI:
        case GL_RGB32UI:
        case GL_RGBA32UI:
            *out_components = mglRenderStoredColorComponents(internal_format);
            *out_component_bytes = 4u;
            *out_signed = 0;
            return 0;
        default:
            return -1;
    }
}

extern "C"
uint8_t* mglRenderCreateIntegerMultiChannelSwizzledUpload(
    uint32_t internal_format,
    uint32_t swizzle_r, uint32_t swizzle_g,
    uint32_t swizzle_b, uint32_t swizzle_a,
    const void* src_data, size_t width, size_t height,
    size_t src_bytes_per_row,
    size_t* out_bytes_per_row, size_t* out_bytes_per_image) {
    if (out_bytes_per_row) *out_bytes_per_row = 0;
    if (out_bytes_per_image) *out_bytes_per_image = 0;
    if (!src_data || width == 0 || height == 0 ||
        !out_bytes_per_row || !out_bytes_per_image) {
        return NULL;
    }
    uint32_t components = 0;
    uint32_t component_bytes = 0;
    int is_signed = 0;
    if (mglRenderIntegerFormatLayout(
            internal_format, &components, &component_bytes, &is_signed) != 0 ||
        components < 2u) {
        return NULL;
    }
    const size_t src_pixel_bytes = (size_t)components * component_bytes;
    const size_t dst_pixel_bytes = 4u * component_bytes;
    const size_t dst_bytes_per_row = width * dst_pixel_bytes;
    const size_t dst_bytes_per_image = dst_bytes_per_row * height;
    if (dst_bytes_per_image == 0 ||
        dst_bytes_per_image > (512u * 1024u * 1024u) ||
        src_bytes_per_row < width * src_pixel_bytes) {
        return NULL;
    }
    uint8_t* dst = (uint8_t*)malloc(dst_bytes_per_image);
    if (!dst) {
        return NULL;
    }
    const uint8_t* src = static_cast<const uint8_t*>(src_data);
    const int64_t default_alpha = 0; /* missing alpha in signed integer texels */
    for (size_t row = 0; row < height; row++) {
        const uint8_t* src_row = src + row * src_bytes_per_row;
        uint8_t* dst_row = dst + row * dst_bytes_per_row;
        for (size_t x = 0; x < width; x++) {
            const uint8_t* in = src_row + x * src_pixel_bytes;
            uint8_t* out = dst_row + x * dst_pixel_bytes;
            const int64_t ch[4] = {
                mglRenderReadIntegerTexelComponent(in, 0u, component_bytes, is_signed),
                components >= 2u
                    ? mglRenderReadIntegerTexelComponent(in, 1u, component_bytes, is_signed)
                    : 0,
                components >= 3u
                    ? mglRenderReadIntegerTexelComponent(in, 2u, component_bytes, is_signed)
                    : 0,
                components >= 4u
                    ? mglRenderReadIntegerTexelComponent(in, 3u, component_bytes, is_signed)
                    : default_alpha,
            };
            const int64_t outv[4] = {
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_r, ch[0], ch[1], ch[2], ch[3], components),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_g, ch[0], ch[1], ch[2], ch[3], components),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_b, ch[0], ch[1], ch[2], ch[3], components),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_a, ch[0], ch[1], ch[2], ch[3], components),
            };
            for (uint32_t c = 0; c < 4u; c++) {
                mglRenderWriteIntegerTexelComponent(
                    out, c, component_bytes, is_signed, outv[c]);
            }
        }
    }
    *out_bytes_per_row = dst_bytes_per_row;
    *out_bytes_per_image = dst_bytes_per_image;
    return dst;
}
