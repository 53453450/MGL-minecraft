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
#include "mgl_render_internal.h"
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

void mglRenderReleaseMetalObject(void* object) {
    if (object) {
        static_cast<NS::Object*>(object)->release();
    }
}

extern "C"
uint8_t* mglRenderCreateChannelExpandedUpload(
    uint32_t internal_format, uint32_t pixel_format, const void* src_data,
    size_t width, size_t height, size_t src_bytes_per_row,
    size_t* out_bytes_per_row, size_t* out_bytes_per_image) {
    if (out_bytes_per_row) *out_bytes_per_row = 0;
    if (out_bytes_per_image) *out_bytes_per_image = 0;
    if (!src_data || width == 0 || height == 0 || src_bytes_per_row == 0 ||
        !out_bytes_per_row || !out_bytes_per_image) {
        return nullptr;
    }

    /* Source and destination parameters (bytes per component / pixel). */
    uint32_t src_comp_u = 0u;
    uint32_t dst_comp_u = 0u;
    uint64_t alpha_default = 0;
    if (!mglRenderRGBExpandParams(pixel_format, &src_comp_u, &dst_comp_u,
                                  &alpha_default)) {
        return nullptr;
    }
    size_t src_comp_bytes = src_comp_u;
    size_t dst_comp_bytes = dst_comp_u;
    size_t src_pixel_bytes = src_comp_bytes * 3u;
    size_t dst_pixel_bytes = dst_comp_bytes * 4u;

    /* Verify source pixel bytes match the internal format. */
    size_t expected_src_bytes =
        sizeForInternalFormat((GLenum)internal_format, 0, 0);
    if (expected_src_bytes > 0 && expected_src_bytes != src_pixel_bytes) {
        /* GL_RGB12: sizeForInternalFormat may differ; stored as 3x16-bit. */
        if (internal_format != GL_RGB12 || expected_src_bytes != 6) {
            return nullptr;
        }
    }

    if (src_bytes_per_row < width * src_pixel_bytes) {
        return nullptr;
    }

    const size_t dst_bytes_per_row = width * dst_pixel_bytes;
    const size_t dst_bytes_per_image = dst_bytes_per_row * height;
    if (dst_bytes_per_image == 0 ||
        dst_bytes_per_image > (512 * 1024 * 1024)) {
        return nullptr;
    }

    uint8_t* dst = (uint8_t*)malloc(dst_bytes_per_image);
    if (!dst) {
        return nullptr;
    }

    for (size_t row = 0; row < height; row++) {
        const uint8_t* src_row =
            (const uint8_t*)src_data + row * src_bytes_per_row;
        uint8_t* dst_row = dst + row * dst_bytes_per_row;
        for (size_t x = 0; x < width; x++) {
            const uint8_t* src_pixel = src_row + x * src_pixel_bytes;
            uint8_t* dst_pixel = dst_row + x * dst_pixel_bytes;
            memcpy(dst_pixel, src_pixel, src_pixel_bytes);
            memcpy(dst_pixel + src_pixel_bytes, &alpha_default,
                   dst_comp_bytes);
        }
    }

    *out_bytes_per_row = dst_bytes_per_row;
    *out_bytes_per_image = dst_bytes_per_image;
    return dst;
}

extern "C"
uint8_t* mglRenderCreateStencilSwizzledUpload(
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
    size_t src_pixel_bytes = 0u;
    switch (internal_format) {
        case GL_DEPTH24_STENCIL8:
            src_pixel_bytes = 4u;
            break;
        case GL_DEPTH32F_STENCIL8:
            src_pixel_bytes = 5u;
            break;
        default:
            return NULL;
    }
    const size_t dst_pixel_bytes = 4u;
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
    for (size_t row = 0; row < height; row++) {
        const uint8_t* src_row = src + row * src_bytes_per_row;
        uint8_t* dst_row = dst + row * dst_bytes_per_row;
        for (size_t x = 0; x < width; x++) {
            const uint8_t* in = src_row + x * src_pixel_bytes;
            uint8_t* out = dst_row + x * dst_pixel_bytes;
            uint8_t stencil = 0u;
            if (internal_format == GL_DEPTH24_STENCIL8) {
                uint32_t packed = 0u;
                memcpy(&packed, in, sizeof(uint32_t));
                stencil = (uint8_t)(packed & 0xffu);
            } else {
                stencil = in[4u];
            }
            const int64_t red = (int64_t)stencil;
            const int64_t outv[4] = {
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_r, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_g, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_b, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_a, red, 0, 0, 1, 1u),
            };
            for (uint32_t c = 0; c < 4u; c++) {
                mglRenderWriteIntegerTexelComponent(
                    out, c, 1u, 0, outv[c]);
            }
        }
    }
    *out_bytes_per_row = dst_bytes_per_row;
    *out_bytes_per_image = dst_bytes_per_image;
    return dst;
}

extern "C"
uint8_t* mglRenderCreateSingleChannelSwizzledUpload(
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
    if (mglRenderTextureUploadNeedsSingleChannelSwizzle(internal_format, 1) == 0 &&
        mglRenderTextureUploadNeedsDepthStencilDepthSwizzleBake(
            internal_format, 1, GL_DEPTH_COMPONENT) == 0) {
        return NULL;
    }
    if (mglRenderTextureUploadNeedsSingleChannelSwizzleBake(internal_format, 1) == 0 &&
        mglRenderTextureUploadNeedsDepthStencilDepthSwizzleBake(
            internal_format, 1, GL_DEPTH_COMPONENT) == 0) {
        return NULL;
    }

    uint32_t dst_component_bytes = 1u;
    int dst_signed = 0;
    uint32_t src_component_bytes = 1u;
    int src_signed = 0;
    switch (internal_format) {
        case GL_R8:
        case GL_R8_SNORM:
            dst_component_bytes = 1u;
            src_component_bytes = 1u;
            break;
        case GL_R8I:
            dst_component_bytes = 1u;
            src_component_bytes = 1u;
            dst_signed = 1;
            src_signed = 1;
            break;
        case GL_R8UI:
            dst_component_bytes = 1u;
            src_component_bytes = 1u;
            break;
        case GL_R16:
        case GL_R16_SNORM:
        case GL_R16F:
            dst_component_bytes = 2u;
            src_component_bytes = 2u;
            break;
        case GL_R16I:
            dst_component_bytes = 2u;
            src_component_bytes = 2u;
            dst_signed = 1;
            src_signed = 1;
            break;
        case GL_R16UI:
            dst_component_bytes = 2u;
            src_component_bytes = 2u;
            break;
        case GL_R32F:
        case GL_R32I:
            dst_component_bytes = 4u;
            src_component_bytes = 4u;
            dst_signed = (internal_format == GL_R32I);
            src_signed = dst_signed;
            break;
        case GL_R32UI:
            dst_component_bytes = 4u;
            src_component_bytes = 4u;
            break;
        case GL_DEPTH_COMPONENT16:
            dst_component_bytes = 2u;
            src_component_bytes = 2u;
            break;
        case GL_DEPTH_COMPONENT24:
            dst_component_bytes = 4u;
            src_component_bytes = 3u;
            break;
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH24_STENCIL8:
            dst_component_bytes = 4u;
            src_component_bytes = 4u;
            break;
        case GL_DEPTH32F_STENCIL8:
            dst_component_bytes = 4u;
            src_component_bytes = 5u;
            break;
        default:
            return NULL;
    }

    const size_t dst_pixel_bytes = 4u * dst_component_bytes;
    const size_t src_pixel_bytes = src_component_bytes;
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
    for (size_t row = 0; row < height; row++) {
        uint8_t* dst_row = dst + row * dst_bytes_per_row;
        const uint8_t* src_row = src + row * src_bytes_per_row;
        for (size_t x = 0; x < width; x++) {
            uint8_t* out = dst_row + x * dst_pixel_bytes;
            if (internal_format == GL_R8) {
                const uint8_t red = src_row[x * src_pixel_bytes];
                out[0] = mglRenderResolveR8SwizzledComponent(swizzle_r, red);
                out[1] = mglRenderResolveR8SwizzledComponent(swizzle_g, red);
                out[2] = mglRenderResolveR8SwizzledComponent(swizzle_b, red);
                out[3] = mglRenderResolveR8SwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_R8_SNORM) {
                const uint8_t red = src_row[x * src_pixel_bytes];
                out[0] = mglRenderResolveR8SnormSwizzledComponent(swizzle_r, red);
                out[1] = mglRenderResolveR8SnormSwizzledComponent(swizzle_g, red);
                out[2] = mglRenderResolveR8SnormSwizzledComponent(swizzle_b, red);
                out[3] = mglRenderResolveR8SnormSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_R16) {
                const uint16_t red =
                    *(const uint16_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint16_t*)(void*)(out + 0) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_r, red);
                *(uint16_t*)(void*)(out + 2) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_g, red);
                *(uint16_t*)(void*)(out + 4) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_b, red);
                *(uint16_t*)(void*)(out + 6) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_R16_SNORM) {
                const int16_t red =
                    *(const int16_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint16_t*)(void*)(out + 0) =
                    mglRenderResolveR16SnormSwizzledComponent(swizzle_r, red);
                *(uint16_t*)(void*)(out + 2) =
                    mglRenderResolveR16SnormSwizzledComponent(swizzle_g, red);
                *(uint16_t*)(void*)(out + 4) =
                    mglRenderResolveR16SnormSwizzledComponent(swizzle_b, red);
                *(uint16_t*)(void*)(out + 6) =
                    mglRenderResolveR16SnormSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_R16F) {
                const uint16_t red =
                    *(const uint16_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint16_t*)(void*)(out + 0) =
                    mglRenderResolveR16FloatSwizzledComponent(swizzle_r, red);
                *(uint16_t*)(void*)(out + 2) =
                    mglRenderResolveR16FloatSwizzledComponent(swizzle_g, red);
                *(uint16_t*)(void*)(out + 4) =
                    mglRenderResolveR16FloatSwizzledComponent(swizzle_b, red);
                *(uint16_t*)(void*)(out + 6) =
                    mglRenderResolveR16FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_R32F ||
                internal_format == GL_DEPTH_COMPONENT32F) {
                const uint32_t red =
                    *(const uint32_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint32_t*)(void*)(out + 0) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_r, red);
                *(uint32_t*)(void*)(out + 4) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_g, red);
                *(uint32_t*)(void*)(out + 8) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_b, red);
                *(uint32_t*)(void*)(out + 12) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_DEPTH_COMPONENT16) {
                const uint16_t red =
                    *(const uint16_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint16_t*)(void*)(out + 0) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_r, red);
                *(uint16_t*)(void*)(out + 2) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_g, red);
                *(uint16_t*)(void*)(out + 4) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_b, red);
                *(uint16_t*)(void*)(out + 6) =
                    mglRenderResolveR16UnormSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_DEPTH_COMPONENT24) {
                const uint32_t red =
                    mglRenderDepth24ToFloatBits(src_row + x * src_pixel_bytes);
                *(uint32_t*)(void*)(out + 0) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_r, red);
                *(uint32_t*)(void*)(out + 4) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_g, red);
                *(uint32_t*)(void*)(out + 8) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_b, red);
                *(uint32_t*)(void*)(out + 12) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_DEPTH_COMPONENT32) {
                const uint32_t red =
                    mglRenderDepthUint32ToFloatBits(
                        *(const uint32_t*)(const void*)(src_row + x * src_pixel_bytes));
                *(uint32_t*)(void*)(out + 0) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_r, red);
                *(uint32_t*)(void*)(out + 4) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_g, red);
                *(uint32_t*)(void*)(out + 8) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_b, red);
                *(uint32_t*)(void*)(out + 12) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_DEPTH24_STENCIL8) {
                const uint32_t red =
                    mglRenderDepth24Stencil8ToFloatBits(src_row + x * src_pixel_bytes);
                *(uint32_t*)(void*)(out + 0) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_r, red);
                *(uint32_t*)(void*)(out + 4) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_g, red);
                *(uint32_t*)(void*)(out + 8) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_b, red);
                *(uint32_t*)(void*)(out + 12) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            if (internal_format == GL_DEPTH32F_STENCIL8) {
                const uint32_t red =
                    *(const uint32_t*)(const void*)(src_row + x * src_pixel_bytes);
                *(uint32_t*)(void*)(out + 0) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_r, red);
                *(uint32_t*)(void*)(out + 4) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_g, red);
                *(uint32_t*)(void*)(out + 8) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_b, red);
                *(uint32_t*)(void*)(out + 12) =
                    mglRenderResolveR32FloatSwizzledComponent(swizzle_a, red);
                continue;
            }
            const int64_t red = mglRenderReadIntegerTexelComponent(
                src_row + x * src_pixel_bytes, 0u, src_component_bytes,
                src_signed);
            const int64_t outv[4] = {
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_r, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_g, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_b, red, 0, 0, 1, 1u),
                mglRenderResolveIntegerSwizzledComponent(
                    swizzle_a, red, 0, 0, 1, 1u),
            };
            for (uint32_t c = 0; c < 4u; c++) {
                mglRenderWriteIntegerTexelComponent(
                    out, c, dst_component_bytes, dst_signed, outv[c]);
            }
        }
    }

    *out_bytes_per_row = dst_bytes_per_row;
    *out_bytes_per_image = dst_bytes_per_image;
    return dst;
}

const char *mglRenderLoadActionName(uint32_t action) {
    switch (static_cast<MTL::LoadAction>(action)) {
        case MTL::LoadActionDontCare: return "DontCare";
        case MTL::LoadActionLoad: return "Load";
        case MTL::LoadActionClear: return "Clear";
        default: return "Unknown";
    }
}

int mglRenderLoadAIRMainFunction(const unsigned char* bytes,
                                    size_t size,
                                    void** library_out,
                                    void** function_out,
                                    char* err,
                                    size_t errcap) {
    if (library_out) *library_out = nullptr;
    if (function_out) *function_out = nullptr;
    if (!bytes || size == 0 || !library_out || !function_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "renderer not initialized");
        return -1;
    }
    return mgl::loadAIRMainFunction(
        renderer.device, bytes, size, library_out, function_out, err, errcap);
}

void mglRenderReleaseSync(GLMContext glm_ctx, Sync* sync) {
    (void)glm_ctx;
    if (!sync) return;
    mgl::releaseBridgedObject(&sync->mtl_command_buffer);
    mgl::releaseBridgedObject(&sync->mtl_event);
}

int mglRenderCreateDepthStencilState(void* depth_stencil_descriptor,
                                        void** depth_stencil_state_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    MTL::DepthStencilDescriptor* descriptor =
        static_cast<MTL::DepthStencilDescriptor*>(depth_stencil_descriptor);
    if (!descriptor || !depth_stencil_state_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::DepthStencilState* state =
        renderer.device->newDepthStencilState(descriptor);
    if (!state) return -1;
    *depth_stencil_state_out = state;
    return 0;
}

int mglRenderCreatePendingEventOwner(MGLPendingEventOwner ** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    mgl::PendingEventOwner* owner = new (std::nothrow) mgl::PendingEventOwner();
    if (!owner) return -1;
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    return 0;
}

void mglRenderDestroyPendingEventOwner(MGLPendingEventOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::PendingEventOwner* owner =
        reinterpret_cast<mgl::PendingEventOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateEvent(void** event_out) {
    if (event_out) *event_out = nullptr;
    if (!event_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::Event* event = renderer.device->newEvent();
    if (!event) return -1;
    *event_out = event;
    return 0;
}

int mglRenderCreateFunction(void* library,
                               const char* name,
                               void* function_constant_values,
                               void** function_out,
                               char* err,
                               size_t errcap) {
    if (function_out) *function_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::Library* source = static_cast<MTL::Library*>(library);
    if (!source || !name || !name[0] || !function_out) return -1;

    NS::String* functionName =
        NS::String::string(name, NS::UTF8StringEncoding);
    MTL::Function* function = nullptr;
    if (function_constant_values) {
        NS::Error* nsError = nullptr;
        function = source->newFunction(
            functionName,
            static_cast<MTL::FunctionConstantValues*>(
                function_constant_values),
            &nsError);
        if (!function) mgl::copyError(nsError, err, errcap);
    } else {
        function = source->newFunction(functionName);
        if (!function && err && errcap) {
            snprintf(err, errcap, "function '%s' not found", name);
        }
    }
    if (!function) return -1;
    *function_out = function;
    return 0;
}

int mglRenderCreateBinaryArchive(void* binary_archive_descriptor,
                                    const char* label,
                                    void** binary_archive_out,
                                    char* err,
                                    size_t errcap) {
    if (binary_archive_out) *binary_archive_out = nullptr;
    if (err && errcap) err[0] = '\0';
    MTL::BinaryArchiveDescriptor* descriptor =
        static_cast<MTL::BinaryArchiveDescriptor*>(binary_archive_descriptor);
    if (!descriptor || !binary_archive_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;

    NS::Error* nsError = nullptr;
    MTL::BinaryArchive* archive =
        renderer.device->newBinaryArchive(descriptor, &nsError);
    if (!archive) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    if (label && label[0]) {
        archive->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    *binary_archive_out = archive;
    return 0;
}

int mglRenderCreateMDIScratchOwner(MGLMDIScratchOwner ** owner_out) {
    if (owner_out) *owner_out = nullptr;
    if (!owner_out) return -1;
    mgl::MDIScratchOwner* owner =
        new (std::nothrow) mgl::MDIScratchOwner();
    if (!owner) return -1;
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    return 0;
}

void mglRenderResetMDIScratchOwner(MGLMDIScratchOwner * owner_handle) {
    mgl::MDIScratchOwner* owner =
        reinterpret_cast<mgl::MDIScratchOwner*>(static_cast<void*>(owner_handle));
    if (!owner) return;
    if (owner->buffer) owner->buffer->release();
    owner->buffer = nullptr;
    owner->capacity = 0;
    owner->offset = 0;
}

void mglRenderDestroyMDIScratchOwner(MGLMDIScratchOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::MDIScratchOwner* owner =
        reinterpret_cast<mgl::MDIScratchOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

namespace mgl {

MTL::Device* wrapDevice(void* objcDevice) {
    MTL::Device* device = static_cast<MTL::Device*>(objcDevice);
    if (device) {
        device->retain();
    }
    return device;
}

} // namespace mgl

int mglRenderInit(void* objc_device) {
    if (!objc_device) {
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (renderer.device) {
        /* A process may own more than one GL context. They must share the
         * same Metal device, but each renderer balances its init/shutdown. */
        if (renderer.device !=
            static_cast<MTL::Device*>(objc_device)) {
            return -1;
        }
        renderer.users++;
        return 0;
    }
    renderer.device = mgl::wrapDevice(objc_device);
    if (!renderer.device) return -1;
    renderer.users = 1;
    return 0;
}

uint32_t mglReadPackedUploadLE(const uint8_t* src, size_t bytes) {
    uint32_t value = 0u;
    if (!src) return 0u;
    if (bytes > sizeof(value)) bytes = sizeof(value);
    for (size_t i = 0; i < bytes; i++) {
        value |= ((uint32_t)src[i]) << (i * 8u);
    }
    return value;
}

uint8_t* mglRenderCreateRGBA8ExpandedUpload(
    const void* src_data, size_t width, size_t height,
    size_t src_bytes_per_row, uint32_t internal_format,
    size_t* out_bytes_per_row, size_t* out_bytes_per_image) {
    if (out_bytes_per_row) *out_bytes_per_row = 0;
    if (out_bytes_per_image) *out_bytes_per_image = 0;
    if (!src_data || width == 0 || height == 0 ||
        src_bytes_per_row == 0 || !out_bytes_per_row || !out_bytes_per_image) {
        return NULL;
    }

    size_t src_pixel_bytes = 0u;
    switch (internal_format) {
        case GL_R3_G3_B2:
            src_pixel_bytes = 1u;
            break;
        case GL_RGBA2:
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB565:
        case GL_RGBA4:
        case GL_RGB5_A1:
            src_pixel_bytes = 2u;
            break;
        case GL_RGB10:
        case GL_RGB12:
            src_pixel_bytes = 4u;
            break;
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB8_SNORM:
        case GL_RGB8I:
        case GL_RGB8UI:
            src_pixel_bytes = 3u;
            break;
        default:
            return NULL;
    }
    if (src_bytes_per_row < width * src_pixel_bytes) {
        return NULL;
    }

    size_t dst_bytes_per_row = width * 4u;
    size_t dst_bytes_per_image = dst_bytes_per_row * height;
    if (dst_bytes_per_image == 0 ||
        dst_bytes_per_image > (512u * 1024u * 1024u)) {
        return NULL;
    }

    uint8_t* dst = (uint8_t*)malloc(dst_bytes_per_image);
    if (!dst) {
        return NULL;
    }

    const uint8_t* src = (const uint8_t*)src_data;
    for (size_t row = 0; row < height; row++) {
        const uint8_t* src_row = src + row * src_bytes_per_row;
        uint8_t* dst_row = dst + row * dst_bytes_per_row;
        for (size_t x = 0; x < width; x++) {
            const uint8_t* src_pixel = src_row + x * src_pixel_bytes;
            uint32_t packed = mglReadPackedUploadLE(src_pixel,
                                                       src_pixel_bytes);
            uint8_t r = 0u, g = 0u, b = 0u, a = 0xffu;
            switch (internal_format) {
                case GL_RGB8:
                case GL_SRGB8:
                case GL_RGB:
                    r = src_pixel[0];
                    g = src_pixel[1];
                    b = src_pixel[2];
                    a = 0xffu;
                    break;
                case GL_RGB8_SNORM:
                    r = src_pixel[0];
                    g = src_pixel[1];
                    b = src_pixel[2];
                    a = 0x7fu; /* 1.0 in snorm */
                    break;
                case GL_RGB8I:
                case GL_RGB8UI:
                    r = src_pixel[0];
                    g = src_pixel[1];
                    b = src_pixel[2];
                    a = 1u; /* 1 in integer */
                    break;
                case GL_R3_G3_B2:
                    r = mglExpandUNormBitsTo8((packed >> 5u) & 0x7u, 3u);
                    g = mglExpandUNormBitsTo8((packed >> 2u) & 0x7u, 3u);
                    b = mglExpandUNormBitsTo8(packed & 0x3u, 2u);
                    break;
                case GL_RGB4:
                case GL_RGB5:
                case GL_RGB565:
                    r = mglExpandUNormBitsTo8((packed >> 11u) & 0x1fu, 5u);
                    g = mglExpandUNormBitsTo8((packed >> 5u) & 0x3fu, 6u);
                    b = mglExpandUNormBitsTo8(packed & 0x1fu, 5u);
                    break;
                case GL_RGB10:
                    r = mglExpandUNormBitsTo8(packed & 0x3ffu, 10u);
                    g = mglExpandUNormBitsTo8((packed >> 10u) & 0x3ffu, 10u);
                    b = mglExpandUNormBitsTo8((packed >> 20u) & 0x3ffu, 10u);
                    break;
                case GL_RGB12:
                    r = mglExpandUNormBitsTo8(packed & 0xfffu, 12u);
                    g = mglExpandUNormBitsTo8((packed >> 12u) & 0xfffu, 12u);
                    b = mglExpandUNormBitsTo8((packed >> 24u) & 0xfffu, 12u);
                    break;
                case GL_RGBA2:
                case GL_RGBA4:
                    r = mglExpandUNormBitsTo8((packed >> 12u) & 0xfu, 4u);
                    g = mglExpandUNormBitsTo8((packed >> 8u) & 0xfu, 4u);
                    b = mglExpandUNormBitsTo8((packed >> 4u) & 0xfu, 4u);
                    a = mglExpandUNormBitsTo8(packed & 0xfu, 4u);
                    break;
                case GL_RGB5_A1:
                    r = mglExpandUNormBitsTo8((packed >> 11u) & 0x1fu, 5u);
                    g = mglExpandUNormBitsTo8((packed >> 6u) & 0x1fu, 5u);
                    b = mglExpandUNormBitsTo8((packed >> 1u) & 0x1fu, 5u);
                    a = (packed & 0x1u) ? 0xffu : 0x00u;
                    break;
                default:
                    break;
            }
            uint8_t* out = dst_row + x * 4u;
            out[0] = r;
            out[1] = g;
            out[2] = b;
            out[3] = a;
        }
    }

    *out_bytes_per_row = dst_bytes_per_row;
    *out_bytes_per_image = dst_bytes_per_image;
    return dst;
}

MGLRenderStencilDescriptorState
mglRenderDescribeStencilDescriptor(const MTL::StencilDescriptor *descriptor) {
    MGLRenderStencilDescriptorState state = {};
    if (!descriptor) return state;
    state.present = 1u;
    state.compare_function = static_cast<uint32_t>(descriptor->stencilCompareFunction());
    state.read_mask = descriptor->readMask();
    state.write_mask = descriptor->writeMask();
    state.stencil_failure_operation =
        static_cast<uint32_t>(descriptor->stencilFailureOperation());
    state.depth_failure_operation =
        static_cast<uint32_t>(descriptor->depthFailureOperation());
    state.depth_stencil_pass_operation =
        static_cast<uint32_t>(descriptor->depthStencilPassOperation());
    return state;
}

MTL::StencilDescriptor* mglRenderBuildStencilDescriptor(
    const MGLRenderStencilDescriptorState& state) {
    if (!state.present) return nullptr;
    MTL::StencilDescriptor* descriptor =
        MTL::StencilDescriptor::alloc()->init();
    if (!descriptor) return nullptr;
    descriptor->setStencilCompareFunction(
        static_cast<MTL::CompareFunction>(state.compare_function));
    descriptor->setReadMask(state.read_mask);
    descriptor->setWriteMask(state.write_mask);
    descriptor->setStencilFailureOperation(
        static_cast<MTL::StencilOperation>(
            state.stencil_failure_operation));
    descriptor->setDepthFailureOperation(
        static_cast<MTL::StencilOperation>(state.depth_failure_operation));
    descriptor->setDepthStencilPassOperation(
        static_cast<MTL::StencilOperation>(
            state.depth_stencil_pass_operation));
    return descriptor;
}

MTL::DepthStencilState* mglRenderCreateDepthStencilFromStateLocked(
    mgl::Renderer& renderer,
    const MGLRenderDepthStencilDescriptorState& state) {
    if (!renderer.device) return nullptr;
    MTL::DepthStencilDescriptor* descriptor =
        MTL::DepthStencilDescriptor::alloc()->init();
    if (!descriptor) return nullptr;
    descriptor->setDepthCompareFunction(
        static_cast<MTL::CompareFunction>(state.depth_compare_function));
    descriptor->setDepthWriteEnabled(state.depth_write_enabled != 0);
    MTL::StencilDescriptor* front =
        mglRenderBuildStencilDescriptor(state.front);
    MTL::StencilDescriptor* back =
        mglRenderBuildStencilDescriptor(state.back);
    if (state.front.present && !front) {
        descriptor->release();
        if (back) back->release();
        return nullptr;
    }
    if (state.back.present && !back) {
        descriptor->release();
        if (front) front->release();
        return nullptr;
    }
    descriptor->setFrontFaceStencil(front);
    descriptor->setBackFaceStencil(back);
    if (front) front->release();
    if (back) back->release();
    MTL::DepthStencilState* result =
        renderer.device->newDepthStencilState(descriptor);
    descriptor->release();
    return result;
}

int mglRenderCreateDepthStencilStateFromState(
    const MGLRenderDepthStencilDescriptorState* descriptor,
    void** depth_stencil_state_out) {
    if (depth_stencil_state_out) *depth_stencil_state_out = nullptr;
    if (!descriptor || !depth_stencil_state_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    MTL::DepthStencilState* state =
        mglRenderCreateDepthStencilFromStateLocked(renderer, *descriptor);
    if (!state) return -1;
    *depth_stencil_state_out = state;
    return 0;
}

int mglRenderCreateAuxFunctions(
    const unsigned char* bytes,
    size_t size,
    uint64_t asset_hash,
    const char* vertex_entry,
    const char* fragment_entry,
    void** vertex_out,
    void** fragment_out,
    char* err,
    size_t errcap) {
    if (vertex_out) *vertex_out = nullptr;
    if (fragment_out) *fragment_out = nullptr;
    if (!vertex_out) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    MTL::Library* library = loadAuxLibraryLocked(
        renderer, bytes, size, asset_hash, err, errcap);
    if (!library) return -1;
    /* vertex_entry may be NULL for fragment-only blobs (e.g. stub FS
     * metallibs compiled at runtime). */
    MTL::Function* vertexFunction = nullptr;
    if (vertex_entry) {
        vertexFunction = newAuxEntryFunction(library, vertex_entry, err, errcap);
        if (!vertexFunction) return -1;
    }
    MTL::Function* fragmentFunction =
        newAuxEntryFunction(library, fragment_entry, err, errcap);
    if (fragment_entry && !fragmentFunction) {
        vertexFunction->release();
        if (err && errcap && !err[0]) {
            snprintf(err, errcap, "aux shader entry function '%s' not found",
                     fragment_entry);
        }
        return -1;
    }
    *vertex_out = vertexFunction;
    if (fragment_out) *fragment_out = fragmentFunction;
    return 0;
}
