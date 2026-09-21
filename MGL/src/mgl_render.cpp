/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

//------------------------------------------------------------------------------------------------

//


//

//------------------------------------------------------------------------------------------------
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "mgl_metal.h"
#include "mgl_render.h"
#include "mgl_render_pixel.h"
extern "C" {
#include "pixel_utils.h"
}
#include "mgl_program_resource.h"  /* mglProgramStageUsesBuiltin */
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

extern "C" void mglMetalCountRelease(int kind);
extern "C" void mglMetalCountCreate(int kind);
extern "C" void mglRecordBufferCowSnapshot(uint64_t bytes);

static_assert(MGLTextureType2D == static_cast<uint32_t>(MTL::TextureType2D));
static_assert(MGLTextureType3D == static_cast<uint32_t>(MTL::TextureType3D));
static_assert(MGLTextureUsageRenderTarget ==
              static_cast<uint32_t>(MTL::TextureUsageRenderTarget));
static_assert(MGLStorageModePrivate ==
              static_cast<uint32_t>(MTL::StorageModePrivate));
static_assert(MGLLoadActionClear == static_cast<uint32_t>(MTL::LoadActionClear));
static_assert(MGLStoreActionMultisampleResolve ==
              static_cast<uint32_t>(MTL::StoreActionMultisampleResolve));
static_assert(MGLCompareFunctionAlways ==
              static_cast<uint32_t>(MTL::CompareFunctionAlways));
static_assert(MGLCommandBufferStatusError ==
              static_cast<uint32_t>(MTL::CommandBufferStatusError));
static_assert(MGLPrimitiveTypeTriangleStrip ==
              static_cast<uint32_t>(MTL::PrimitiveTypeTriangleStrip));
static_assert(MGLWindingCounterClockwise ==
              static_cast<uint32_t>(MTL::WindingCounterClockwise));
static_assert(MGLColorWriteMaskAll ==
              static_cast<uint32_t>(MTL::ColorWriteMaskAll));
static_assert(MGLTessellationControlPointIndexTypeUInt32 ==
              static_cast<uint32_t>(MTL::TessellationControlPointIndexTypeUInt32));
static_assert(MGLBlendFactorOneMinusSource1Alpha ==
              static_cast<uint32_t>(MTL::BlendFactorOneMinusSource1Alpha));
static_assert(MGLBlendOperationMax ==
              static_cast<uint32_t>(MTL::BlendOperationMax));
static_assert(MGLVertexFormatHalf ==
              static_cast<uint32_t>(MTL::VertexFormatHalf));

namespace mgl {



} // namespace mgl

#include "mgl_render_internal.h"


//------------------------------------------------------------------------------------------------

//------------------------------------------------------------------------------------------------
extern "C" {


/* mglRenderPassAttachmentClass / mglRenderPassColorAttachmentIndexValid moved
 * to mgl_render_pass_plan.c (O3.1): they are pure value-state predicates with
 * C linkage (declared in mgl_render.h), and keeping them beside the plan keeps
 * mgl_render_pass_plan.c self-contained so the clear-value harness can link it
 * without pulling in Metal / LLVM. */


void mglRenderShutdown(void) {
    mgl::Renderer& renderer = mgl::renderer();
    {
        std::lock_guard<std::mutex> lock(renderer.mutex);
        if (renderer.users > 1) {
            renderer.users--;
            return;
        }
        renderer.users = 0;
        mgl::releasePipelineCaches(renderer);
        mgl::releaseBindingStates(renderer);
        mgl::releasePackedStructBuffers(renderer);
        mglAirLoaderShutdown();
        if (renderer.device) {
            renderer.device->release();
            renderer.device = nullptr;
        }
    }
}




void mglRenderDeleteMTLObj(GLMContext glm_ctx, void* object) {
    (void)glm_ctx;
    mgl::releaseBridgedObject(&object);
}
























void mglRenderWaitForSync(GLMContext glm_ctx, Sync* sync) {
    (void)glm_ctx;
    if (!sync) return;
    if (sync->mtl_command_buffer) {
        MGLRenderCommandBufferState state = {};
        (void)mglRenderWaitCommandBufferState(
            sync->mtl_command_buffer, &state);
        mgl::releaseBridgedObject(&sync->mtl_command_buffer);
    }
    mgl::releaseBridgedObject(&sync->mtl_event);
}



void mglRenderFlush(GLMContext glm_ctx, bool finish) {
    BackendLeaseScope lease(glm_ctx);
    MGLCommandBufferOwner* command_owner =
        static_cast<MGLCommandBufferOwner*>(rendererOwner(
            glm_ctx, MGL_RENDERER_BACKEND_OWNER_COMMAND_BUFFER));
    if (!command_owner) return;

    Sync boundary = {};
    mglRenderGetSync(glm_ctx, &boundary);
    if (finish) {
        if (boundary.mtl_command_buffer) {
            mglRenderWaitForSync(glm_ctx, &boundary);
        } else {
            MGLRenderCommandBufferState state = {};
            (void)mglRenderWaitCommandBufferOwnerLastSubmitted(
                command_owner, &state);
        }
    } else {
        mglRenderReleaseSync(glm_ctx, &boundary);
    }
}



int mglRenderAttachRuntimeOwners(GLMContext glm_ctx, MGLCommandBufferOwner * command_buffer_owner, MGLRenderEncoderOwner * render_encoder_owner, MGLRenderPassStateOwner * render_pass_state_owner) {
    MGLRendererBackendHandle* backend = rendererBackend(glm_ctx);
    return backend
        ? mglRendererBackendAttachRuntimeOwners(
              backend, command_buffer_owner,
              render_encoder_owner, render_pass_state_owner)
        : -1;
}

void mglRenderDetachRuntimeOwners(GLMContext glm_ctx) {
    if (MGLRendererBackendHandle* backend = rendererBackend(glm_ctx)) {
        (void)mglRendererBackendAttachRuntimeOwners(
            backend, nullptr, nullptr, nullptr);
    }
}























int mglRenderCreateTextureViewRange(
    void* texture,
    uint32_t pixel_format,
    uint32_t texture_type,
    uint64_t level_location,
    uint64_t level_length,
    uint64_t slice_location,
    uint64_t slice_length,
    int use_swizzle,
    uint32_t swizzle_red,
    uint32_t swizzle_green,
    uint32_t swizzle_blue,
    uint32_t swizzle_alpha,
    void** texture_view_out) {
    if (texture_view_out) *texture_view_out = nullptr;
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!source || !texture_view_out || level_length == 0 ||
        slice_length == 0) {
        return -1;
    }
    const NS::Range levels(level_location, level_length);
    const NS::Range slices(slice_location, slice_length);
    MTL::Texture* view = nullptr;
    if (use_swizzle) {
        const MTL::TextureSwizzleChannels swizzle(
            static_cast<MTL::TextureSwizzle>(swizzle_red),
            static_cast<MTL::TextureSwizzle>(swizzle_green),
            static_cast<MTL::TextureSwizzle>(swizzle_blue),
            static_cast<MTL::TextureSwizzle>(swizzle_alpha));
        view = source->newTextureView(
            static_cast<MTL::PixelFormat>(pixel_format),
            static_cast<MTL::TextureType>(texture_type), levels, slices,
            swizzle);
    } else {
        view = source->newTextureView(
            static_cast<MTL::PixelFormat>(pixel_format),
            static_cast<MTL::TextureType>(texture_type), levels, slices);
    }
    if (!view) return -1;
    *texture_view_out = view;
    return 0;
}


int mglRenderTextureReplaceRegion(void* texture,
                                     uint64_t x,
                                     uint64_t y,
                                     uint64_t z,
                                     uint64_t width,
                                     uint64_t height,
                                     uint64_t depth,
                                     uint64_t level,
                                     uint64_t slice,
                                     const void* bytes,
                                     uint64_t bytes_per_row,
                                     uint64_t bytes_per_image,
                                     int use_slice) {
    MTL::Texture* destination = static_cast<MTL::Texture*>(texture);
    if (!destination || !bytes || width == 0 || height == 0 || depth == 0 ||
        bytes_per_row == 0) {
        return -1;
    }
    MTL::Region region = MTL::Region::Make3D(
        static_cast<NS::UInteger>(x), static_cast<NS::UInteger>(y),
        static_cast<NS::UInteger>(z), static_cast<NS::UInteger>(width),
        static_cast<NS::UInteger>(height),
        static_cast<NS::UInteger>(depth));
    if (use_slice) {
        if (bytes_per_image == 0) return -1;
        destination->replaceRegion(
            region, static_cast<NS::UInteger>(level),
            static_cast<NS::UInteger>(slice), bytes,
            static_cast<NS::UInteger>(bytes_per_row),
            static_cast<NS::UInteger>(bytes_per_image));
    } else {
        destination->replaceRegion(
            region, static_cast<NS::UInteger>(level), bytes,
            static_cast<NS::UInteger>(bytes_per_row));
    }
    return 0;
}

int mglRenderTextureGetBytes(void* texture,
                                void* bytes,
                                uint64_t bytes_per_row,
                                uint64_t bytes_per_image,
                                uint64_t x,
                                uint64_t y,
                                uint64_t z,
                                uint64_t width,
                                uint64_t height,
                                uint64_t depth,
                                uint64_t level,
                                uint64_t slice,
                                int use_slice) {
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!source || !bytes || width == 0 || height == 0 || depth == 0 ||
        bytes_per_row == 0) {
        return -1;
    }
    MTL::Region region = MTL::Region::Make3D(
        static_cast<NS::UInteger>(x), static_cast<NS::UInteger>(y),
        static_cast<NS::UInteger>(z), static_cast<NS::UInteger>(width),
        static_cast<NS::UInteger>(height),
        static_cast<NS::UInteger>(depth));
    if (use_slice) {
        if (bytes_per_image == 0) return -1;
        source->getBytes(
            bytes, static_cast<NS::UInteger>(bytes_per_row),
            static_cast<NS::UInteger>(bytes_per_image), region,
            static_cast<NS::UInteger>(level),
            static_cast<NS::UInteger>(slice));
    } else {
        source->getBytes(bytes, static_cast<NS::UInteger>(bytes_per_row),
                         region, static_cast<NS::UInteger>(level));
    }
    return 0;
}


extern "C"
int mglRenderTextureSubUploadPlan(
    uint32_t gl_target,
    uint32_t texture_type,
    uint64_t requested_slice,
    uint64_t xoffset,
    uint64_t yoffset,
    uint64_t zoffset,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    uint64_t source_bytes_per_row,
    uint64_t source_bytes_per_image,
    MGLRenderTextureSubUploadPlan* plan_out) {
    if (!plan_out || width == 0u || height == 0u || depth == 0u ||
        source_bytes_per_row == 0u || source_bytes_per_image == 0u) {
        return -1;
    }

    *plan_out = {};
    plan_out->destination_x = xoffset;
    plan_out->copy_width = width;
    plan_out->copy_height = height;
    plan_out->copy_depth = 1u;
    plan_out->layer_count = 1u;

    if (gl_target == GL_TEXTURE_1D_ARRAY) {
        if (yoffset > std::numeric_limits<uint64_t>::max() - (height - 1u)) {
            *plan_out = {};
            return -1;
        }
        plan_out->destination_base_slice = yoffset;
        plan_out->destination_y = 0u;
        plan_out->destination_z = 0u;
        plan_out->copy_height = 1u;
        plan_out->copy_depth = 1u;
        plan_out->layer_count = height;
        plan_out->source_layer_stride = source_bytes_per_row;
        return 0;
    }

    if (gl_target == GL_TEXTURE_2D_ARRAY ||
        gl_target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        if (zoffset > std::numeric_limits<uint64_t>::max() - (depth - 1u)) {
            *plan_out = {};
            return -1;
        }
        plan_out->destination_base_slice = zoffset;
        plan_out->destination_y = yoffset;
        plan_out->destination_z = 0u;
        plan_out->copy_depth = 1u;
        plan_out->layer_count = depth;
        plan_out->source_layer_stride = source_bytes_per_image;
        return 0;
    }

    switch (static_cast<MTL::TextureType>(texture_type)) {
        case MTL::TextureType3D:
            plan_out->destination_y = yoffset;
            plan_out->destination_z = zoffset;
            plan_out->copy_depth = depth;
            return 0;
        case MTL::TextureTypeCube:
        case MTL::TextureTypeCubeArray:
        case MTL::TextureType2DArray:
        case MTL::TextureType1DArray:
        case MTL::TextureType2DMultisampleArray:
            plan_out->destination_base_slice = requested_slice;
            plan_out->destination_y = yoffset;
            return 0;
        default:
            plan_out->destination_y =
                gl_target == GL_TEXTURE_1D ? 0u : yoffset;
            plan_out->copy_height =
                gl_target == GL_TEXTURE_1D ? 1u : height;
            return 0;
    }
}






extern "C"
int mglRenderGLInternalFormatLooksDepthOrStencil(uint32_t internal_format) {
    switch (internal_format) {
        case GL_DEPTH_COMPONENT:
        case GL_DEPTH_COMPONENT16:
        case GL_DEPTH_COMPONENT24:
        case GL_DEPTH_COMPONENT32:
        case GL_DEPTH_COMPONENT32F:
        case GL_DEPTH_STENCIL:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH32F_STENCIL8:
        case GL_STENCIL_INDEX:
        case GL_STENCIL_INDEX8:
            return 1;
        default:
            return 0;
    }
}


extern "C"
uint64_t mglRenderMetalCompressedBlockHeight(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatBC1_RGBA:
        case MTL::PixelFormatBC1_RGBA_sRGB:
        case MTL::PixelFormatBC2_RGBA:
        case MTL::PixelFormatBC2_RGBA_sRGB:
        case MTL::PixelFormatBC3_RGBA:
        case MTL::PixelFormatBC3_RGBA_sRGB:
        case MTL::PixelFormatBC4_RUnorm:
        case MTL::PixelFormatBC4_RSnorm:
        case MTL::PixelFormatBC5_RGUnorm:
        case MTL::PixelFormatBC5_RGSnorm:
        case MTL::PixelFormatBC6H_RGBFloat:
        case MTL::PixelFormatBC6H_RGBUfloat:
        case MTL::PixelFormatBC7_RGBAUnorm:
        case MTL::PixelFormatBC7_RGBAUnorm_sRGB:
        case MTL::PixelFormatASTC_4x4_sRGB:
        case MTL::PixelFormatASTC_4x4_LDR:
        case MTL::PixelFormatASTC_4x4_HDR:
        case MTL::PixelFormatASTC_5x4_sRGB:
        case MTL::PixelFormatASTC_5x4_LDR:
        case MTL::PixelFormatASTC_5x4_HDR:
            return 4u;
        case MTL::PixelFormatASTC_5x5_sRGB:
        case MTL::PixelFormatASTC_5x5_LDR:
        case MTL::PixelFormatASTC_5x5_HDR:
        case MTL::PixelFormatASTC_6x5_sRGB:
        case MTL::PixelFormatASTC_6x5_LDR:
        case MTL::PixelFormatASTC_6x5_HDR:
        case MTL::PixelFormatASTC_8x5_sRGB:
        case MTL::PixelFormatASTC_8x5_LDR:
        case MTL::PixelFormatASTC_8x5_HDR:
        case MTL::PixelFormatASTC_10x5_sRGB:
        case MTL::PixelFormatASTC_10x5_LDR:
        case MTL::PixelFormatASTC_10x5_HDR:
            return 5u;
        case MTL::PixelFormatASTC_6x6_sRGB:
        case MTL::PixelFormatASTC_6x6_LDR:
        case MTL::PixelFormatASTC_6x6_HDR:
        case MTL::PixelFormatASTC_8x6_sRGB:
        case MTL::PixelFormatASTC_8x6_LDR:
        case MTL::PixelFormatASTC_8x6_HDR:
        case MTL::PixelFormatASTC_10x6_sRGB:
        case MTL::PixelFormatASTC_10x6_LDR:
        case MTL::PixelFormatASTC_10x6_HDR:
            return 6u;
        case MTL::PixelFormatASTC_8x8_sRGB:
        case MTL::PixelFormatASTC_8x8_LDR:
        case MTL::PixelFormatASTC_8x8_HDR:
        case MTL::PixelFormatASTC_10x8_sRGB:
        case MTL::PixelFormatASTC_10x8_LDR:
        case MTL::PixelFormatASTC_10x8_HDR:
            return 8u;
        case MTL::PixelFormatASTC_10x10_sRGB:
        case MTL::PixelFormatASTC_10x10_LDR:
        case MTL::PixelFormatASTC_10x10_HDR:
        case MTL::PixelFormatASTC_12x10_sRGB:
        case MTL::PixelFormatASTC_12x10_LDR:
        case MTL::PixelFormatASTC_12x10_HDR:
            return 10u;
        case MTL::PixelFormatASTC_12x12_sRGB:
        case MTL::PixelFormatASTC_12x12_LDR:
        case MTL::PixelFormatASTC_12x12_HDR:
            return 12u;
        default:
            return 1u;
    }
}

extern "C"
uint64_t mglRenderMetalUploadRowsForPixelFormat(uint32_t pixel_format,
                                                   uint64_t pixel_height) {
    const uint64_t height = pixel_height ? pixel_height : 1u;
    const uint64_t block_height =
        mglRenderMetalCompressedBlockHeight(pixel_format);
    if (block_height <= 1u) {
        return height;
    }
    return (height + block_height - 1u) / block_height;
}




extern "C"
uint32_t mglRenderSRGBPixelFormat(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatRGBA8Unorm:
            return (uint32_t)MTL::PixelFormatRGBA8Unorm_sRGB;
        case MTL::PixelFormatBGRA8Unorm:
            return (uint32_t)MTL::PixelFormatBGRA8Unorm_sRGB;
        default:
            return pixel_format;
    }
}













extern "C"
int mglRenderBuildTextureUploadPlan(
    uint32_t gl_target,
    uint32_t texture_type,
    uint32_t storage_mode,
    uint32_t pixel_format,
    int has_agx_3d_copy_bug,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    uint64_t bytes_per_row,
    uint64_t bytes_per_image,
    uint64_t destination_level,
    uint64_t destination_slice,
    MGLRenderTextureUploadPlan* plan_out) {
    if (!plan_out) return -1;
    *plan_out = {};
    if (width == 0u || bytes_per_row == 0u || bytes_per_image == 0u) {
        return -1;
    }

    const MTL::TextureType type = static_cast<MTL::TextureType>(texture_type);
    const bool is_3d = type == MTL::TextureType3D;
    const bool logical_1d =
        gl_target == GL_TEXTURE_1D || gl_target == GL_TEXTURE_1D_ARRAY;
    const bool logical_1d_array = gl_target == GL_TEXTURE_1D_ARRAY;
    const bool is_array_or_cube =
        type == MTL::TextureTypeCube || type == MTL::TextureTypeCubeArray ||
        type == MTL::TextureType2DArray || type == MTL::TextureType1DArray ||
        type == MTL::TextureType2DMultisampleArray;

    MGLRenderTextureUploadPlan plan = {};
    plan.normalized_height = logical_1d
        ? 1u
        : std::max<uint64_t>(height, 1u);
    plan.normalized_depth = std::max<uint64_t>(depth, 1u);
    plan.copy_depth = is_3d ? plan.normalized_depth : 1u;
    plan.upload_rows = mglRenderMetalUploadRowsForPixelFormat(
        pixel_format, plan.normalized_height);
    if (plan.upload_rows == 0u ||
        bytes_per_row > std::numeric_limits<uint64_t>::max() /
                            plan.upload_rows) {
        return -1;
    }
    plan.expected_bytes_per_image = bytes_per_row * plan.upload_rows;
    if (bytes_per_image < plan.expected_bytes_per_image) return -1;
    plan.normalized_bytes_per_image =
        (is_array_or_cube || !is_3d)
            ? plan.expected_bytes_per_image
            : bytes_per_image;
    plan.destination_slice =
        (is_3d || gl_target == GL_TEXTURE_1D) ? 0u : destination_slice;
    plan.destination_level = destination_level;

    const uint32_t private_mode =
        static_cast<uint32_t>(MTL::StorageModePrivate);
    if (logical_1d && storage_mode != private_mode) {
        plan.route = MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_1D;
        plan.replace_region_dimension =
            (type == MTL::TextureType1D || type == MTL::TextureType1DArray)
                ? 1u
                : 2u;
        plan.replace_use_slice = logical_1d_array ||
                                 type == MTL::TextureType1DArray;
    } else {
        plan.route = static_cast<uint32_t>(mglRenderTextureUploadRoute(
            texture_type, storage_mode, has_agx_3d_copy_bug));
        if (plan.route == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_1D) {
            plan.replace_region_dimension = 1u;
            plan.replace_use_slice = type == MTL::TextureType1DArray;
        } else if (plan.route ==
                   MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_3D) {
            plan.replace_region_dimension = 3u;
            plan.requires_repack =
                plan.normalized_bytes_per_image !=
                plan.expected_bytes_per_image;
        }
    }

    if (plan.route == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REJECT) {
        *plan_out = plan;
        return 0;
    }

    if (plan.route == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_3D) {
        if (plan.requires_repack &&
            plan.copy_depth > std::numeric_limits<uint64_t>::max() /
                                  plan.expected_bytes_per_image) {
            return -1;
        }
    } else {
        if (plan.copy_depth > std::numeric_limits<uint64_t>::max() /
                                  plan.normalized_bytes_per_image) {
            return -1;
        }
        plan.buffer_size =
            plan.normalized_bytes_per_image * plan.copy_depth;
        constexpr uint64_t kMaxTextureUploadStagingBytes =
            512ull * 1024ull * 1024ull;
        if (plan.buffer_size == 0u ||
            plan.buffer_size > kMaxTextureUploadStagingBytes) {
            return -1;
        }
    }

    *plan_out = plan;
    return 0;
}




/* C1: CopyRows + CopyDepthTextureBytesToFloat -> mgl_readback_policy.c */

extern "C"
int mglRenderCopyGLBGRA8RowsToBGRA8CompatibleTextureBytes(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    if (src_bytes_per_row < width * 4u ||
        dst_bytes_per_row < width * 4u) {
        return 0;
    }

    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    bool destinationIsRGBA = (pf == MTL::PixelFormatRGBA8Unorm ||
                              pf == MTL::PixelFormatRGBA8Unorm_sRGB);
    bool destinationIsBGRA = (pf == MTL::PixelFormatBGRA8Unorm ||
                              pf == MTL::PixelFormatBGRA8Unorm_sRGB);
    bool destinationIsRGB9E5 = (pf == MTL::PixelFormatRGB9E5Float);
    bool destinationIsRGB10A2 = (pf == MTL::PixelFormatRGB10A2Unorm ||
                                 pf == MTL::PixelFormatBGR10A2Unorm);
    if (!destinationIsRGBA && !destinationIsBGRA &&
        !destinationIsRGB9E5 && !destinationIsRGB10A2) {
        return 0;
    }

    const uint8_t* srcBytes = static_cast<const uint8_t*>(src);
    uint8_t* dstBytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* srcRow = srcBytes + (y * src_bytes_per_row);
        uint64_t dstY = flip_y ? (height - 1u - y) : y;
        uint8_t* dstRow = dstBytes + (dstY * dst_bytes_per_row);

        for (uint64_t x = 0; x < width; x++) {
            const uint8_t* s = srcRow + (x * 4u);
            uint8_t* d = dstRow + (x * 4u);
            uint8_t b = s[0];
            uint8_t g = s[1];
            uint8_t r = s[2];
            uint8_t a = s[3];

            if (destinationIsBGRA) {
                d[0] = b;
                d[1] = g;
                d[2] = r;
                d[3] = a;
            } else if (destinationIsRGB10A2) {
                /* RGB10A2Unorm: bits [0:9]=R, [10:19]=G, [20:29]=B,
                 * [30:31]=A.  BGR10A2Unorm: bits [0:9]=B, [10:19]=G,
                 * [20:29]=R, [30:31]=A. */
                uint32_t r10 = ((uint32_t)r * 1023u + 127u) / 255u;
                uint32_t g10 = ((uint32_t)g * 1023u + 127u) / 255u;
                uint32_t b10 = ((uint32_t)b * 1023u + 127u) / 255u;
                uint32_t a2 = ((uint32_t)a * 3u + 127u) / 255u;
                uint32_t packed;
                if (pf == MTL::PixelFormatBGR10A2Unorm) {
                    packed = b10 | (g10 << 10) | (r10 << 20) | (a2 << 30);
                } else {
                    packed = r10 | (g10 << 10) | (b10 << 20) | (a2 << 30);
                }
                d[0] = (uint8_t)(packed & 0xFF);
                d[1] = (uint8_t)((packed >> 8) & 0xFF);
                d[2] = (uint8_t)((packed >> 16) & 0xFF);
                d[3] = (uint8_t)((packed >> 24) & 0xFF);
            } else if (destinationIsRGB9E5) {
                /* GL_RGB9_E5 packs three 9-bit mantissas and a 5-bit shared
                 * exponent into a 32-bit word.  Source is BGRA8. */
                uint32_t packed = mglPackRGBToSharedExp(
                    (double)r / 255.0, (double)g / 255.0,
                    (double)b / 255.0);
                d[0] = (uint8_t)(packed & 0xFF);
                d[1] = (uint8_t)((packed >> 8) & 0xFF);
                d[2] = (uint8_t)((packed >> 16) & 0xFF);
                d[3] = (uint8_t)((packed >> 24) & 0xFF);
            } else {
                d[0] = r;
                d[1] = g;
                d[2] = b;
                d[3] = a;
            }
        }
    }

    return 1;
}















/* OpenGL GetTexImage / transfer: missing R/G/B → 0, missing A → 1.
 * Do not replicate the last stored channel (that yields RG→(R,G,G,G)). */










extern "C"
int mglRenderCopyRGB10A2TextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    if (pf != MTL::PixelFormatRGB10A2Unorm ||
        !mglReadbackRGB10A2TypeAccepted(type)) {
        return 0;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }

    const uint64_t src_bpp = 4u;
    uint32_t comp_bytes = mglSizeForType(type);
    uint64_t dst_pixel_bytes = mglPixelTypeIsPacked(type)
        ? (uint64_t)comp_bytes
        : (uint64_t)comp_bytes * (uint64_t)slots;
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            uint32_t packed = 0u;
            memcpy(&packed, src_row + (x * src_bpp), sizeof(packed));
            uint32_t rgb10a2_vals[4] = {
                packed & 1023u,
                (packed >> 10u) & 1023u,
                (packed >> 20u) & 1023u,
                (packed >> 30u) & 3u
            };

            if (type == GL_UNSIGNED_INT_10_10_10_2) {
                uint32_t r10 = rgb10a2_vals[src_idx[0]];
                uint32_t g10 = (slots > 1) ? rgb10a2_vals[src_idx[1]] : 0u;
                uint32_t b10 = (slots > 2) ? rgb10a2_vals[src_idx[2]] : 0u;
                uint32_t a2 = (slots > 3) ? rgb10a2_vals[src_idx[3]] : 0u;
                uint32_t out = (r10 << 22u) | (g10 << 12u) | (b10 << 2u) | a2;
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
                uint32_t r10 = rgb10a2_vals[src_idx[0]];
                uint32_t g10 = (slots > 1) ? rgb10a2_vals[src_idx[1]] : 0u;
                uint32_t b10 = (slots > 2) ? rgb10a2_vals[src_idx[2]] : 0u;
                uint32_t a2 = (slots > 3) ? rgb10a2_vals[src_idx[3]] : 0u;
                uint32_t out = r10 | (g10 << 10u) | (b10 << 20u) | (a2 << 30u);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                float rf = (float)rgb10a2_vals[src_idx[0]] / 1023.0f;
                float gf = (slots > 1)
                    ? (float)rgb10a2_vals[src_idx[1]] / 1023.0f : 0.0f;
                float bf = (slots > 2)
                    ? (float)rgb10a2_vals[src_idx[2]] / 1023.0f : 0.0f;
                uint32_t out = mglPackRGBToSharedExp(rf, gf, bf);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                uint8_t r8 = (uint8_t)((uint64_t)rgb10a2_vals[src_idx[0]] *
                                       255u / 1023u);
                uint8_t g8 = (slots > 1)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[1]] * 255u / 1023u)
                    : 0u;
                uint8_t b8 = (slots > 2)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[2]] * 255u / 1023u)
                    : 0u;
                uint8_t a8 = (slots > 3)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[3]] * 255u / 3u)
                    : 0u;
                uint32_t out = ((uint32_t)r8 << 24u) | ((uint32_t)g8 << 16u) |
                               ((uint32_t)b8 << 8u) | a8;
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                uint8_t r8 = (uint8_t)((uint64_t)rgb10a2_vals[src_idx[0]] *
                                       255u / 1023u);
                uint8_t g8 = (slots > 1)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[1]] * 255u / 1023u)
                    : 0u;
                uint8_t b8 = (slots > 2)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[2]] * 255u / 1023u)
                    : 0u;
                uint8_t a8 = (slots > 3)
                    ? (uint8_t)((uint64_t)rgb10a2_vals[src_idx[3]] * 255u / 3u)
                    : 0u;
                uint32_t out = r8 | ((uint32_t)g8 << 8u) |
                               ((uint32_t)b8 << 16u) | ((uint32_t)a8 << 24u);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else {
                for (int c = 0; c < slots; ++c) {
                    uint32_t raw = rgb10a2_vals[src_idx[c]];
                    float fv = (src_idx[c] == 3)
                        ? (float)raw / 3.0f : (float)raw / 1023.0f;
                    uint8_t* out = dst_row + (x * dst_pixel_bytes) +
                                   (uint64_t)c * (uint64_t)comp_bytes;
                    if (type == GL_UNSIGNED_BYTE) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint8_t iv = (uint8_t)lroundf(cv * 255.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_BYTE) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int8_t iv = (int8_t)lroundf(cv * 127.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_UNSIGNED_SHORT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint16_t iv = (uint16_t)lroundf(cv * 65535.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_SHORT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int16_t iv = (int16_t)lroundf(cv * 32767.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_UNSIGNED_INT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                        uint32_t iv = (uint32_t)llroundf(cv * 4294967295.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_INT) {
                        float cv = fv > 1.0f ? 1.0f : (fv < -1.0f ? -1.0f : fv);
                        int32_t iv = (int32_t)llroundf(cv * 2147483647.0f);
                        memcpy(out, &iv, sizeof(iv));
                    } else if (type == GL_FLOAT) {
                        memcpy(out, &fv, sizeof(fv));
                    } else {
                        uint16_t iv = mglFloatToHalf(fv);
                        memcpy(out, &iv, sizeof(iv));
                    }
                }
            }
        }
    }
    return 1;
}





















/* little-endian packed read + unorm bit expansion (RGBA8 path). */


extern "C" uint32_t mglRenderCollectCopyBackEntries(
    const MGLRenderCopyBackEntry *slots, uint32_t slot_count,
    MGLRenderCopyBackEntry *out, uint32_t out_cap) {
    if (!slots || !out || out_cap == 0u) {
        return 0u;
    }
    uint32_t n = 0u;
    for (uint32_t i = 0u; i < slot_count && n < out_cap; i++) {
        if (slots[i].length == 0u) {
            continue;
        }
        out[n++] = slots[i];
    }
    return n;
}

/* legacy packed GL formats -> RGBA8 (pure data transform). */
/* stage-binding copy-back encode + CPU-prefix sync.
 * Pure validation/encode over the caller-bridged entries; the CB
 * sequencing (detach/commit/wait/AGX recovery) stays in the renderer. */

extern "C"
int mglRenderCopyBackCPUPrefix(
    const MGLRenderCopyBackEntry* entries, uint32_t count,
    uint32_t* failed_index_out) {
    if (failed_index_out) *failed_index_out = count;
    if (!entries && count) return -1;
    for (uint32_t i = 0; i < count; i++) {
        const MGLRenderCopyBackEntry& entry = entries[i];
        if (entry.length == 0 || !entry.destination_buffer) continue;
        Buffer* buffer =
            static_cast<Buffer*>(const_cast<void*>(entry.destination_buffer));
        if (!buffer->data.buffer_data) continue;
        MTL::Buffer* destination =
            static_cast<MTL::Buffer*>(const_cast<void*>(entry.destination));
        if (!destination || !destination->contents() ||
            entry.destination_offset > buffer->data.buffer_size ||
            entry.length >
                buffer->data.buffer_size - entry.destination_offset) {
            if (failed_index_out) *failed_index_out = i;
            return -1;
        }
        buffer->ever_written = GL_TRUE;
        uint8_t* cpu_bytes =
            (uint8_t*)(uintptr_t)buffer->data.buffer_data;
        const uint8_t* metal_bytes =
            (const uint8_t*)destination->contents();
        if (cpu_bytes != metal_bytes) {
            memmove(cpu_bytes + entry.destination_offset,
                    metal_bytes + entry.destination_offset,
                    entry.length);
        }
        buffer->cpu_shadow_pending = GL_FALSE;
    }
    return 0;
}

extern "C"
int mglRenderBuildRuntimeArraySizes(
    const MGLRenderBufferSizeEntry* entries, uint32_t entry_count,
    uint32_t runtime_buffer_index, uint32_t max_slot,
    uint32_t* out_sizes, uint32_t out_capacity) {
    if (!out_sizes || out_capacity < max_slot ||
        (!entries && entry_count != 0)) {
        return -1;
    }
    for (uint32_t i = 0; i < entry_count; i++) {
        const MGLRenderBufferSizeEntry& entry = entries[i];

        if (entry.metal_slot >= max_slot ||
            entry.metal_slot == runtime_buffer_index) {
            continue;
        }
        if (entry.metal_slot >= out_capacity) {
            continue;
        }
        out_sizes[entry.metal_slot] = (uint32_t)entry.visible_size;
    }
    return 0;
}


extern "C" {
GLuint sizeForInternalFormat(GLenum internalformat, GLenum format,
                             GLenum type);
}







/* C1: mglRenderConvertIntegerReadback → mgl_readback_policy.c */


/* GL 4.6 section 11.2.2.2 patch discard predicate.
 * This is evaluated before any tessellation level is clamped to one. */




extern "C"
int mglRenderNativeTESInterfaceSupported(
    void* tes_function, uint64_t tes_metallib_bytes,
    uint32_t tes_gen_point_mode, uint32_t tes_xfb_varying_count,
    uint32_t tes_gen_mode,
    void* tcs_function, uint64_t tcs_metallib_bytes,
    uint32_t tcs_output_vertices) {
    if (!tes_function || tes_metallib_bytes == 0u ||
        tes_gen_point_mode != 0u || tes_xfb_varying_count > 0u) {
        return 0;
    }
    if (tcs_function && (tcs_metallib_bytes == 0u ||
                         tcs_output_vertices == 0u ||
                         tcs_output_vertices > 32u)) {
        return 0;
    }
    if (tes_gen_mode != GL_TRIANGLES && tes_gen_mode != GL_QUADS) {
        return 0;
    }
    MTL::Function* fn = static_cast<MTL::Function*>(tes_function);
    MTL::PatchType expected = tes_gen_mode == GL_QUADS
        ? MTL::PatchTypeQuad : MTL::PatchTypeTriangle;
    if (fn->patchType() != expected) {
        return 0;
    }
    /* The metallib TESS tag now carries 4*controlPointCount + patchKind;
     * a non-zero patchControlPointCount is the real per-patch control
     * point count and must agree with the TCS output vertices.  Zero
     * (legacy encoding) is also tolerated. */
    if (fn->patchControlPointCount() > 0 && tcs_function &&
        tcs_output_vertices != (uint32_t)fn->patchControlPointCount()) {
        return 0;
    }
    return 1;
}

extern "C"
int mglRenderRasterizationIsEmpty(
    int32_t vx, int32_t vy, int32_t vw, int32_t vh,
    uint32_t pass_width, uint32_t pass_height,
    int32_t scissor_enabled,
    int32_t sx, int32_t sy, int32_t sw, int32_t sh) {
    if (vw <= 0 || vh <= 0) {
        return 1;
    }
    if (pass_width == 0 || pass_height == 0) {
        return 0;
    }
    const int64_t fbW = (int64_t)pass_width;
    const int64_t fbH = (int64_t)pass_height;
    const int64_t vx0 = (int64_t)vx;
    const int64_t vy0 = (int64_t)vy;
    const int64_t vx1 = vx0 + (int64_t)vw;
    const int64_t vy1 = vy0 + (int64_t)vh;
    if (vx1 <= 0 || vy1 <= 0 || vx0 >= fbW || vy0 >= fbH) {
        return 1;
    }
    if (scissor_enabled) {
        if (sw <= 0 || sh <= 0) {
            return 1;
        }
        const int64_t sx0 = (int64_t)sx;
        const int64_t sy0 = (int64_t)sy;
        const int64_t sx1 = sx0 + (int64_t)sw;
        const int64_t sy1 = sy0 + (int64_t)sh;
        if (sx1 <= 0 || sy1 <= 0 || sx0 >= fbW || sy0 >= fbH) {
            return 1;
        }
    }
    return 0;
}

/* C1: IntegerReadback Source/Packed/Classify -> mgl_readback_policy.c */
/* C1: GetTexImagePlan -> mgl_readback_policy.c */








/* O3.3: UseInlineFragmentBytes..IsolateCopyLength -> mgl_binding_stage.c */

/* O3.3: IntegerAttribDstIsInt..BindingOffsetInMetal -> mgl_binding_stage.c */



/* O3.3: ImageBindPixelFormat..ImageViewSliceCount -> mgl_binding_texture.c */




uint64_t mglRenderExpectedArrayLayers(uint32_t gl_target, int32_t depth) {
    uint64_t layers = depth > 0 ? (uint64_t)depth : 1u;
    if (gl_target == GL_TEXTURE_CUBE_MAP_ARRAY && layers >= 6u &&
        (layers % 6u) == 0u) {
        layers /= 6u;
    }
    return layers;
}





int mglRenderPrefer1DOverDefault2D(uint32_t expected_type, uint32_t active_target) {
    return expected_type == MGLTextureType2D && active_target == GL_TEXTURE_1D
               ? 1
               : 0;
}

int mglRenderPreferMSOr1DArrayOver2DArray(uint32_t expected_type,
                                          uint32_t active_target) {
    return expected_type == MGLTextureType2DArray &&
                   (active_target == GL_TEXTURE_1D_ARRAY ||
                    active_target == GL_TEXTURE_2D_MULTISAMPLE ||
                    active_target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
               ? 1
               : 0;
}



int mglRenderExpectedTypeIsCube(uint32_t expected_type) {
    return expected_type == MGLTextureTypeCube ? 1 : 0;
}




uint64_t mglRenderFallbackSampledCacheKey(uint32_t texture_type, uint32_t data_kind) {
    return ((uint64_t)texture_type << 8u) | (uint64_t)data_kind;
}

uint32_t mglRenderAGXCompatiblePixelFormat(uint32_t pixel_format, int *converted) {
    int conv = 0;
    switch (pixel_format) {
    case 40u: /* B5G6R5Unorm */
    case 41u: /* A1BGR5Unorm */
    case 43u: /* BGR5A1Unorm */
    case 160u: /* PVRTC_RGB_2BPP */
    case 162u: /* PVRTC_RGB_4BPP */
    case 164u: /* PVRTC_RGBA_2BPP */
    case 166u: /* PVRTC_RGBA_4BPP */
    case 170u: /* EAC_R11Unorm */
    case 174u: /* EAC_RG11Unorm */
    case 178u: /* EAC_RGBA8 */
    case 180u: /* ETC2_RGB8 */
    case 182u: /* ETC2_RGB8A1 */
        conv = 1;
        pixel_format = 70u; /* RGBA8Unorm */
        break;
    default:
        break;
    }
    if (converted) {
        *converted = conv;
    }
    return pixel_format;
}

int mglRenderPromote1DArrayDepthStencil(uint32_t tex_type, uint32_t pixel_format) {
    if (tex_type != MGLTextureType1DArray) {
        return 0;
    }
    switch (pixel_format) {
    case 250u: /* Depth16Unorm */
    case 252u: /* Depth32Float */
    case 253u: /* Stencil8 */
    case 255u: /* Depth24Unorm_Stencil8 */
    case 260u: /* Depth32Float_Stencil8 */
    case 261u: /* X32_Stencil8 */
    case 262u: /* X24_Stencil8 */
        return 1;
    default:
        return 0;
    }
}



int mglRenderPromoteMipmapped1D(uint32_t tex_type) {
    return tex_type == MGLTextureType1D ? 1 : 0;
}

int mglRenderPromoteMipmapped1DArray(uint32_t tex_type) {
    return tex_type == MGLTextureType1DArray ? 1 : 0;
}

int mglRenderCubeFaceSizeValid(uint64_t width, uint64_t height) {
    return width == height ? 1 : 0;
}


uint32_t mglRenderUploadLevelCount(int mipmapped, int tex_mipmapped,
                                   uint32_t effective) {
    return mipmapped && tex_mipmapped ? effective : 1u;
}


int mglRenderEmulateMSAsArray(uint32_t tex_type, uint32_t samples, uint64_t depth,
                              uint32_t *out_type, uint32_t *sample_count,
                              uint64_t *array_len, uint64_t *depth_out) {
    if (tex_type != MGLTextureType2DMultisample &&
        tex_type != MGLTextureType2DMultisampleArray) {
        return 0;
    }
    const uint64_t kMsPlaneStride = 8u;
    uint32_t sc = 1u;
    uint64_t d = 1u;
    uint64_t arr = 1u;
    uint32_t ot = MGLTextureType2DArray;
    uint32_t smp = samples < 2u ? 2u : samples;
    if (tex_type == MGLTextureType2DMultisample) {
        arr = smp < 1u ? 1u : smp;
    } else {
        uint64_t layers = depth < 1u ? 1u : depth;
        arr = layers * kMsPlaneStride;
    }
    if (out_type) {
        *out_type = ot;
    }
    if (sample_count) {
        *sample_count = sc;
    }
    if (array_len) {
        *array_len = arr;
    }
    if (depth_out) {
        *depth_out = d;
    }
    return 1;
}

int mglRenderPreferSharedStorage(int needs_cpu, int is_depth_stencil) {
    return needs_cpu || is_depth_stencil ? 1 : 0;
}


void mglRenderApply1DBackingToDesc(int backed_1d, int backed_1d_array,
                                   uint64_t height, uint32_t *type,
                                   uint64_t *array_len, uint32_t *height_out) {
    if (backed_1d) {
        if (type) {
            *type = MGLTextureType2D;
        }
        if (height_out) {
            *height_out = 1u;
        }
    }
    if (backed_1d_array) {
        if (type) {
            *type = MGLTextureType2DArray;
        }
        if (array_len) {
            *array_len = height < 1u ? 1u : height;
        }
        if (height_out) {
            *height_out = 1u;
        }
    }
}


/* C1: binding slot/sampler/stage/plain-uniform policy -> mgl_binding_policy.c */


/* C1: format-class PSO topology/blend/stencil/viewport -> mgl_pso_format_class.c */












int mglRenderTraceR8RedUByte(uint32_t internalformat, uint32_t format,
                             uint32_t type) {
    return internalformat == GL_R8 && format == GL_RED &&
                   type == GL_UNSIGNED_BYTE
               ? 1
               : 0;
}




uint32_t mglRenderCompletenessCheckFaces(uint32_t target, uint32_t num_faces) {
    return target == GL_TEXTURE_CUBE_MAP_ARRAY ? 1u : num_faces;
}



int mglRenderIs3DReupload(uint32_t target, uint32_t depth) {
    return mglRenderTextureTargetIs3D(target) && depth > 1u ? 1 : 0;
}





uint32_t mglRenderDepthStencilPlaneViewType(uint32_t parent_type) {
    switch (parent_type) {
    case MGLTextureType2DArray:
    case MGLTextureTypeCube:
    case MGLTextureTypeCubeArray:
    case MGLTextureType1DArray:
    case MGLTextureType2DMultisampleArray:
    case MGLTextureType3D:
        return MGLTextureType2D;
    default:
        return parent_type;
    }
}

/* C1: PixelFormatIsPackedDepthStencil -> mgl_pso_format_class.c */


uint32_t mglRenderStencilViewFormat(uint32_t parent_format) {
    return parent_format == 255u /* Depth24Unorm_Stencil8 */
               ? 262u /* X24_Stencil8 */
               : 261u /* X32_Stencil8 */;
}


uint32_t mglRenderRepairedDefaultStencilFormat(uint32_t stencil_format) {
    return stencil_format == 0u /* Invalid */ ||
                   stencil_format == 260u /* Depth32Float_Stencil8 */
               ? 253u /* Stencil8 */
               : stencil_format;
}


int mglRenderPackedD32FNeeds8ByteStride(uint32_t pixel_format,
                                        uint32_t row_bytes, uint32_t width) {
    return mglRenderPixelFormatIsDepth32FloatStencil8(pixel_format) &&
                   row_bytes >= width * 5u && row_bytes < width * 8u
               ? 1
               : 0;
}

/* C1: DepthReadbackPlan -> mgl_readback_policy.c */

/* C1: DefaultDepthPixelFormat -> mgl_pso_format_class.c */




const char *mglRenderGLSLTypeName(uint32_t type) {
    switch (type) {
    case GL_FLOAT:
        return "float";
    case GL_FLOAT_VEC2:
        return "vec2";
    case GL_FLOAT_VEC3:
        return "vec3";
    case GL_FLOAT_VEC4:
        return "vec4";
    case GL_FLOAT_MAT2:
        return "mat2";
    case GL_FLOAT_MAT3:
        return "mat3";
    case GL_FLOAT_MAT4:
        return "mat4";
    case GL_FLOAT_MAT2x3:
        return "mat2x3";
    case GL_FLOAT_MAT2x4:
        return "mat2x4";
    case GL_FLOAT_MAT3x2:
        return "mat3x2";
    case GL_FLOAT_MAT3x4:
        return "mat3x4";
    case GL_FLOAT_MAT4x2:
        return "mat4x2";
    case GL_FLOAT_MAT4x3:
        return "mat4x3";
    case GL_INT:
        return "int";
    case GL_INT_VEC2:
        return "ivec2";
    case GL_INT_VEC3:
        return "ivec3";
    case GL_INT_VEC4:
        return "ivec4";
    case GL_UNSIGNED_INT:
        return "uint";
    case GL_UNSIGNED_INT_VEC2:
        return "uvec2";
    case GL_UNSIGNED_INT_VEC3:
        return "uvec3";
    case GL_UNSIGNED_INT_VEC4:
        return "uvec4";
    default:
        return NULL;
    }
}

uint32_t mglRenderGLSLMatrixCols(uint32_t type) {
    switch (type) {
    case GL_FLOAT_MAT2:
    case GL_FLOAT_MAT2x3:
    case GL_FLOAT_MAT2x4:
        return 2u;
    case GL_FLOAT_MAT3:
    case GL_FLOAT_MAT3x2:
    case GL_FLOAT_MAT3x4:
        return 3u;
    case GL_FLOAT_MAT4:
    case GL_FLOAT_MAT4x2:
    case GL_FLOAT_MAT4x3:
        return 4u;
    default:
        return 0u;
    }
}

uint32_t mglRenderGLSLMatrixRows(uint32_t type) {
    switch (type) {
    case GL_FLOAT_MAT2:
    case GL_FLOAT_MAT3x2:
    case GL_FLOAT_MAT4x2:
        return 2u;
    case GL_FLOAT_MAT3:
    case GL_FLOAT_MAT2x3:
    case GL_FLOAT_MAT4x3:
        return 3u;
    case GL_FLOAT_MAT4:
    case GL_FLOAT_MAT2x4:
    case GL_FLOAT_MAT3x4:
        return 4u;
    default:
        return 0u;
    }
}


const char *mglRenderGLSLColumnType(uint32_t rows) {
    switch (rows) {
    case 1u:
        return "float";
    case 2u:
        return "vec2";
    case 3u:
        return "vec3";
    case 4u:
        return "vec4";
    default:
        return NULL;
    }
}



int mglRenderGLSLNeedsFlat(uint32_t type) {
    switch (type) {
    case GL_INT:
    case GL_INT_VEC2:
    case GL_INT_VEC3:
    case GL_INT_VEC4:
    case GL_UNSIGNED_INT:
    case GL_UNSIGNED_INT_VEC2:
    case GL_UNSIGNED_INT_VEC3:
    case GL_UNSIGNED_INT_VEC4:
        return 1;
    default:
        return 0;
    }
}

/* C1: MSAAArrayLayerStride -> mgl_readback_policy.c (Metal EncodeMultisampleResolve residual here) */


int mglRenderTargetIsRenderbuffer(uint32_t target) {
    return target == GL_RENDERBUFFER ? 1 : 0;
}

int mglRenderMSSamplePlaneAdjust(int in_ms_loop, uint32_t target,
                                 int32_t offset) {
    return in_ms_loop && mglRenderIsMultisampleTextureTarget(target) &&
                   offset > 0
               ? 1
               : 0;
}


int mglRenderClipOriginIsLowerLeft(uint32_t origin) {
    return origin == GL_LOWER_LEFT ? 1 : 0;
}

int mglRenderErrorIsNone(uint32_t error) {
    return error == GL_NO_ERROR ? 1 : 0;
}

uint32_t mglRenderErrorNone(void) {
    return GL_NO_ERROR;
}

uint32_t mglRenderErrorInvalidOperation(void) {
    return GL_INVALID_OPERATION;
}

uint32_t mglRenderErrorInvalidValue(void) {
    return GL_INVALID_VALUE;
}

uint32_t mglRenderErrorOutOfMemory(void) {
    return GL_OUT_OF_MEMORY;
}

uint32_t mglRenderGLBoolean(int value) {
    return value ? (uint32_t)GL_TRUE : (uint32_t)GL_FALSE;
}


int mglRenderStopColorAttachmentScan(uint32_t next_index, uint32_t max,
                                     int next_is_none, int has_next_color) {
    return next_index >= max || (next_is_none && !has_next_color) ? 1 : 0;
}


int mglRenderEmulateTriangleFan(uint32_t mode, int polygon_point) {
    return mode == GL_TRIANGLE_FAN && !polygon_point ? 1 : 0;
}

int mglRenderEmulateLineLoop(uint32_t mode) {
    return mode == GL_LINE_LOOP ? 1 : 0;
}

int mglRenderEmulateQuads(uint32_t mode, int polygon_point) {
    return mode == GL_QUADS && !polygon_point ? 1 : 0;
}


int mglRenderFilterIsNearest(uint32_t filter) {
    return filter == GL_NEAREST ? 1 : 0;
}

uint32_t mglRenderNearestFilter(void) {
    return GL_NEAREST;
}












int mglRenderCPUFormatTypeForInternalFormat(uint32_t internalformat,
                                            uint32_t *out_format,
                                            uint32_t *out_type) {
    uint32_t format = 0u;
    uint32_t type = 0u;
    int known = 1;
    switch (internalformat) {
    case GL_R3_G3_B2:
        format = GL_RGB;
        type = GL_UNSIGNED_BYTE_3_3_2;
        break;
    case GL_RGB4:
    case GL_RGB5:
        format = GL_RGB;
        type = GL_UNSIGNED_SHORT_5_6_5;
        break;
    case GL_RGB5_A1:
        format = GL_RGBA;
        type = GL_UNSIGNED_SHORT_5_5_5_1;
        break;
    case GL_RGBA2:
    case GL_RGBA4:
        format = GL_RGBA;
        type = GL_UNSIGNED_SHORT_4_4_4_4;
        break;
    case GL_RGB12:
        format = GL_RGB;
        type = GL_UNSIGNED_SHORT;
        break;
    case GL_RGB32F:
        format = GL_RGB;
        type = GL_FLOAT;
        break;
    default:
        known = 0;
        break;
    }
    if (out_format) {
        *out_format = format;
    }
    if (out_type) {
        *out_type = type;
    }
    return known;
}



int mglRenderQuadsCountTooSmall(uint32_t mode, int32_t count) {
    return mode == GL_QUADS && count < 4 ? 1 : 0;
}

int mglRenderPolygonPointEmulateMode(uint32_t mode) {
    return mglRenderDrawModeIsTriangles(mode) ||
                   mglRenderDrawModeIsTriangleStrip(mode) ||
                   mglRenderDrawModeNeedsEmulate(mode)
               ? 1
               : 0;
}




















int mglRenderCompareFuncFromGL(uint32_t func, uint32_t *out) {
    uint32_t mapped = MGLCompareFunctionAlways;
    int known = 1;
    switch (func) {
    case GL_NEVER:
        mapped = MGLCompareFunctionNever;
        break;
    case GL_LESS:
        mapped = MGLCompareFunctionLess;
        break;
    case GL_EQUAL:
        mapped = MGLCompareFunctionEqual;
        break;
    case GL_LEQUAL:
        mapped = MGLCompareFunctionLessEqual;
        break;
    case GL_GREATER:
        mapped = MGLCompareFunctionGreater;
        break;
    case GL_NOTEQUAL:
        mapped = MGLCompareFunctionNotEqual;
        break;
    case GL_GEQUAL:
        mapped = MGLCompareFunctionGreaterEqual;
        break;
    case GL_ALWAYS:
        mapped = MGLCompareFunctionAlways;
        break;
    default:
        known = 0;
        break;
    }
    if (out) {
        *out = mapped;
    }
    return known;
}

int mglRenderFrontFaceIsClockwise(uint32_t front_face) {
    return front_face == GL_CW ? 1 : 0;
}

int mglRenderFrontFaceIsCounterClockwise(uint32_t front_face) {
    return front_face == GL_CCW ? 1 : 0;
}

int mglRenderCubeMapFaceSlice(uint32_t textarget, uint32_t *out) {
    uint32_t slice = 0u;
    int known = 1;
    switch (textarget) {
    case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        slice = 0u;
        break;
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        slice = 1u;
        break;
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        slice = 2u;
        break;
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        slice = 3u;
        break;
    case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        slice = 4u;
        break;
    case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
        slice = 5u;
        break;
    default:
        known = 0;
        break;
    }
    if (out) {
        *out = slice;
    }
    return known;
}

int mglRenderAttachmentUsesArrayLayer(uint32_t textarget) {
    return textarget == GL_TEXTURE_1D_ARRAY ||
                   textarget == GL_TEXTURE_2D_ARRAY ||
                   textarget == GL_TEXTURE_CUBE_MAP_ARRAY
               ? 1
               : 0;
}







int mglRenderPackedDepthStencilFormat(uint32_t internalformat) {
    return internalformat == GL_DEPTH32F_STENCIL8 ||
                   internalformat == GL_DEPTH24_STENCIL8
               ? 1
               : 0;
}













int mglRenderRepairBlendSrcFactor(uint32_t *value) {
    if (!value) {
        return 0;
    }
    return mglRenderApplyBlendRepair(mglRenderIsValidGLBlendFactor(*value),
                                     value, GL_ONE);
}

int mglRenderRepairBlendDstFactor(uint32_t *value) {
    if (!value) {
        return 0;
    }
    return mglRenderApplyBlendRepair(mglRenderIsValidGLBlendFactor(*value),
                                     value, GL_ZERO);
}

int mglRenderRepairBlendEquation(uint32_t *value) {
    if (!value) {
        return 0;
    }
    return mglRenderApplyBlendRepair(mglRenderIsValidGLBlendEquation(*value),
                                     value, GL_FUNC_ADD);
}

uint32_t mglRenderRepairDepthFunc(uint32_t func) {
    return mglRenderCompareFuncOrFallback(
        func, mglRenderIsValidGLCompareFunction(func), GL_LESS);
}

uint32_t mglRenderRepairStencilFunc(uint32_t func) {
    return mglRenderCompareFuncOrFallback(
        func, mglRenderIsValidGLCompareFunction(func), GL_ALWAYS);
}






int mglRenderCPUPointerUsable(const void *p) {
    return p && (uintptr_t)p >= 0x1000ull ? 1 : 0;
}







int mglRenderShaderResourceToGLBufferType(int spvc_type) {
    switch (spvc_type) {
    case _UNIFORM_BUFFER_RES:
        return _UNIFORM_BUFFER;
    case _UNIFORM_CONSTANT_RES:
        return _UNIFORM_CONSTANT;
    case _STORAGE_BUFFER_RES:
        return _SHADER_STORAGE_BUFFER;
    case _ATOMIC_COUNTER_RES:
        return _ATOMIC_COUNTER_BUFFER;
    default:
        return -1;
    }
}

int mglRenderUsePlainUniformBuffers(int spvc_type) {
    return spvc_type == _UNIFORM_CONSTANT_RES ? 1 : 0;
}


/* mglRenderBufferPlanEntrySkip removed with MGL_BP_FLAG_SKIP: the
 * SPIRV-era skip rule that produced it never fired (probe, 2026-09-12). */



int mglRenderStructMemberInElementRange(uint32_t member_loc_off,
                                        uint32_t loc_start, uint32_t loc_end) {
    return member_loc_off >= loc_start && member_loc_off < loc_end ? 1 : 0;
}

int mglRenderBindableLocValid(int32_t loc, uint32_t max) {
    return loc >= 0 && (uint32_t)loc < max ? 1 : 0;
}

int mglRenderCPUShadowReadable(const void *cpu, int64_t size) {
    return cpu && size > 0 ? 1 : 0;
}



uint64_t mglRenderClampCopyToStruct(uint64_t dest_off, uint64_t copy_size,
                                    uint64_t struct_size) {
    if (copy_size == 0u || dest_off >= struct_size) {
        return 0u;
    }
    if (dest_off + copy_size > struct_size) {
        return struct_size - dest_off;
    }
    return copy_size;
}

/* mglRenderMappedBufferCountOK moved to mgl_vertex_attrib_plan.c: pure value
 * predicate owned by the vertex-attribute buffer map plan, kept there so that
 * layer and its unit-test harness stay free of Metal/LLVM dependencies. */





int64_t mglRenderMappedUniformSize(int spvc_type, int64_t bound, int64_t buf_size,
                                   int64_t offset, uint64_t reflected) {
    if (spvc_type != _UNIFORM_BUFFER_RES) {
        return bound;
    }
    if (bound <= 0 || reflected == 0u || (uint64_t)bound >= reflected ||
        buf_size <= offset) {
        return bound;
    }
    int64_t remaining = buf_size - offset;
    int64_t want = (int64_t)reflected;
    if (remaining < want) {
        want = remaining;
    }
    return want > bound ? want : bound;
}



int32_t mglRenderPlainUniformBaseLoc(int32_t uniform_location,
                                     uint32_t location) {
    return uniform_location >= 0 ? uniform_location : (int32_t)location;
}

uint32_t mglRenderMemberOffsetInElement(uint32_t offset,
                                        uint32_t elem_byte_start) {
    return offset >= elem_byte_start ? offset - elem_byte_start : offset;
}

int mglRenderMemberOffsetInStruct(uint32_t offset, uint32_t struct_size) {
    return offset < struct_size ? 1 : 0;
}

void mglRenderPlainUniformArrayStrides(const char *name, uint32_t type_bytes,
                                       int32_t array_stride, uint32_t *src_out,
                                       uint32_t *elem_out) {
    uint32_t src = type_bytes;
    uint32_t elem = type_bytes;
    if (name && strchr(name, '.') && array_stride > (int32_t)src &&
        array_stride > 0) {
        elem = (uint32_t)array_stride;
    } else if (elem == 0u) {
        elem = array_stride > 0 ? (uint32_t)array_stride : 0u;
        src = elem ? elem : 4u;
    }
    if (src == 0u) {
        src = 4u;
    }
    if (src_out) {
        *src_out = src;
    }
    if (elem_out) {
        *elem_out = elem;
    }
}

int mglRenderMetalBackingTooSmall(int64_t gl_size, uint64_t metal_length) {
    return gl_size > 0 && metal_length < (uint64_t)gl_size ? 1 : 0;
}

/* O3.3: WritableStorageNeedsGPUAuthoritative -> mgl_binding_stage.c */








extern "C"
uint64_t mglRenderAlignVertexStrideForMetal(uint64_t stride) {
    return (stride + 3u) & ~(uint64_t)3u;
}



extern "C" uint32_t mglRenderGLTypeSizeToVertexFormat(uint32_t type,
                                                      uint32_t size,
                                                      int normalized) {
    switch (type) {
        case GL_UNSIGNED_BYTE:
            if (normalized) {
                switch (size) {
                    case 1u: return MGLVertexFormatUCharNormalized;
                    case 2u: return MGLVertexFormatUChar2Normalized;
                    case 3u: return MGLVertexFormatUChar3Normalized;
                    case 4u: return MGLVertexFormatUChar4Normalized;
                }
            } else {
                switch (size) {
                    case 1u: return MGLVertexFormatUChar;
                    case 2u: return MGLVertexFormatUChar2;
                    case 3u: return MGLVertexFormatUChar3;
                    case 4u: return MGLVertexFormatUChar4;
                }
            }
            break;
        case GL_BYTE:
            if (normalized) {
                switch (size) {
                    case 1u: return MGLVertexFormatCharNormalized;
                    case 2u: return MGLVertexFormatChar2Normalized;
                    case 3u: return MGLVertexFormatChar3Normalized;
                    case 4u: return MGLVertexFormatChar4Normalized;
                }
            } else {
                switch (size) {
                    case 1u: return MGLVertexFormatChar;
                    case 2u: return MGLVertexFormatChar2;
                    case 3u: return MGLVertexFormatChar3;
                    case 4u: return MGLVertexFormatChar4;
                }
            }
            break;
        case GL_UNSIGNED_SHORT:
            if (normalized) {
                switch (size) {
                    case 1u: return MGLVertexFormatUShortNormalized;
                    case 2u: return MGLVertexFormatUShort2Normalized;
                    case 3u: return MGLVertexFormatUShort3Normalized;
                    case 4u: return MGLVertexFormatUShort4Normalized;
                }
            } else {
                switch (size) {
                    case 1u: return MGLVertexFormatUShort;
                    case 2u: return MGLVertexFormatUShort2;
                    case 3u: return MGLVertexFormatUShort3;
                    case 4u: return MGLVertexFormatUShort4;
                }
            }
            break;
        case GL_SHORT:
            if (normalized) {
                switch (size) {
                    case 1u: return MGLVertexFormatShortNormalized;
                    case 2u: return MGLVertexFormatShort2Normalized;
                    case 3u: return MGLVertexFormatShort3Normalized;
                    case 4u: return MGLVertexFormatShort4Normalized;
                }
            } else {
                switch (size) {
                    case 1u: return MGLVertexFormatShort;
                    case 2u: return MGLVertexFormatShort2;
                    case 3u: return MGLVertexFormatShort3;
                    case 4u: return MGLVertexFormatShort4;
                }
            }
            break;
        case GL_HALF_FLOAT:
            switch (size) {
                case 1u: return MGLVertexFormatHalf;
                case 2u: return MGLVertexFormatHalf2;
                case 3u: return MGLVertexFormatHalf3;
                case 4u: return MGLVertexFormatHalf4;
            }
            break;
        case GL_FLOAT:
            switch (size) {
                case 1u: return MGLVertexFormatFloat;
                case 2u: return MGLVertexFormatFloat2;
                case 3u: return MGLVertexFormatFloat3;
                case 4u: return MGLVertexFormatFloat4;
            }
            break;
        case GL_INT:
            switch (size) {
                case 1u: return MGLVertexFormatInt;
                case 2u: return MGLVertexFormatInt2;
                case 3u: return MGLVertexFormatInt3;
                case 4u: return MGLVertexFormatInt4;
            }
            break;
        case GL_UNSIGNED_INT:
            switch (size) {
                case 1u: return MGLVertexFormatUInt;
                case 2u: return MGLVertexFormatUInt2;
                case 3u: return MGLVertexFormatUInt3;
                case 4u: return MGLVertexFormatUInt4;
            }
            break;
        case GL_RGB10:
        case GL_INT_2_10_10_10_REV:
            if (normalized) {
                return MGLVertexFormatInt1010102Normalized;
            }
            break;
        case GL_UNSIGNED_INT_2_10_10_10_REV:
            if (normalized) {
                return MGLVertexFormatUInt1010102Normalized;
            }
            break;
        case GL_UNSIGNED_INT_10_10_10_2:
        case GL_UNSIGNED_INT_10F_11F_11F_REV:
        case GL_FIXED:
            break;
        default:
            break;
    }
    return MGLVertexFormatInvalid;
}




extern "C"
uint64_t mglRenderHashStepU64(uint64_t hash, uint64_t value) {
    return (hash ^ value) * 1099511628211ull;
}


extern "C"
uint32_t mglRenderGLTypeElementByteSize(uint64_t gl_type) {
    switch (gl_type) {
        case GL_FLOAT: case GL_INT: case GL_UNSIGNED_INT: case GL_BOOL:
            return 4u;
        case GL_FLOAT_VEC2: case GL_INT_VEC2: case GL_UNSIGNED_INT_VEC2: case GL_BOOL_VEC2:
            return 8u;
        case GL_FLOAT_VEC3: case GL_INT_VEC3: case GL_UNSIGNED_INT_VEC3: case GL_BOOL_VEC3:
            return 12u;
        case GL_FLOAT_VEC4: case GL_INT_VEC4: case GL_UNSIGNED_INT_VEC4: case GL_BOOL_VEC4:
            return 16u;
        case GL_FLOAT_MAT2:
            return 8u;   /* one column = vec2 */
        case GL_FLOAT_MAT3:
            return 12u;  /* one column = vec3 */
        case GL_FLOAT_MAT4:
            return 16u;  /* one column = vec4 */
        case GL_FLOAT_MAT2x3: return 12u;
        case GL_FLOAT_MAT2x4: return 16u;
        case GL_FLOAT_MAT3x2: return 8u;
        case GL_FLOAT_MAT3x4: return 16u;
        case GL_FLOAT_MAT4x2: return 8u;
        case GL_FLOAT_MAT4x3: return 12u;
        case GL_DOUBLE: return 8u;
        default: return 4u;
    }
}











extern "C"
int
mglRenderExpandQuadArrayIndices(
    uint32_t quad_count, uint32_t** out_indices, uint64_t* out_count) {
    if (quad_count == 0u || !out_indices || !out_count) {
        return -1;
    }
    const uint64_t need = (uint64_t)quad_count * 6u;
    if (need > (uint64_t)(UINT32_MAX / sizeof(uint32_t))) {
        return -1;
    }
    uint32_t* const dst = (uint32_t*)malloc((size_t)need * sizeof(uint32_t));
    if (!dst) {
        return -1;
    }
    for (uint32_t q = 0u; q < quad_count; q++) {
        const uint32_t base = q * 4u;
        const uint32_t d = q * 6u;
        if (base + 3u > UINT32_MAX) {
            free(dst);
            return -1;
        }
        dst[d+0u] = base + 0u;
        dst[d+1u] = base + 1u;
        dst[d+2u] = base + 2u;
        dst[d+3u] = base + 0u;
        dst[d+4u] = base + 2u;
        dst[d+5u] = base + 3u;
    }
    *out_indices = dst;
    *out_count = need;
    return 0;
}





extern "C"
int mglRenderGeometryGatherIndices(
    const uint8_t* bytes,
    uint32_t elem_width,
    uint32_t count,
    int restart_enabled,
    uint32_t restart_index,
    uint32_t input_vertices,
    MGLRenderGeometryGatherResult* out) {
    if (!bytes || count == 0u || input_vertices == 0u || !out) {
        return -1;
    }
    const int has_restart = restart_enabled ? 1 : 0;
    const int w = (elem_width == 1u) ? 1 : (elem_width == 2u ? 2 : 4);
    uint32_t* const gather = (uint32_t*)malloc((size_t)count * sizeof(uint32_t));
    if (!gather) {
        return -1;
    }
    uint32_t gathered = 0u;
    uint32_t primitives = 0u;
    uint32_t max_index = 0u;
    uint32_t in_prim = 0u;
    for (uint32_t i = 0u; i < count; i++) {
        uint32_t index = 0u;
        if (w == 1) {
            index = bytes[i];
        } else if (w == 2) {
            index = ((const uint16_t*)bytes)[i];
        } else {
            index = ((const uint32_t*)bytes)[i];
        }
        if (has_restart && index == restart_index) {
            /* A restart terminates the current primitive.  Any indices since
             * the last complete primitive form an incomplete fragment and
             * must not become the prefix of the next primitive. */
            gathered -= in_prim;
            in_prim = 0u;
            continue;
        }
        gather[gathered++] = index;
        if (index > max_index) {
            max_index = index;
        }
        if (++in_prim == input_vertices) {
            primitives++;
            in_prim = 0u;
        }
    }
    if (gathered == 0u || primitives == 0u) {
        free(gather);
        return -1;
    }
    if (in_prim != 0u) {
        gathered -= in_prim;
    }
    out->gather = gather;
    out->gather_count = gathered;
    out->primitive_count = primitives;
    out->max_index = max_index;
    return 0;
}


extern "C"
int mglRenderThreadgroupSize(
    uint32_t local_x, uint32_t local_y, uint32_t local_z,
    MGLRenderThreadgroupSize* out) {
    if (!out) return -1;
    out->x = local_x ? local_x : 1u;
    out->y = local_y ? local_y : 1u;
    out->z = local_z ? local_z : 1u;
    return 0;
}






extern "C"
int mglRenderPolygonOffsetDecision(
    uint32_t mode, int has_ctx, int produces_polygons,
    uint32_t polygon_mode,
    int cap_point, int cap_line, int cap_fill,
    MGLRenderPolygonOffsetDecision* out) {
    if (!out) return -1;
    const int polygons = (has_ctx && produces_polygons) ? 1 : 0;
    out->triangle_fill_mode =
        (polygons && polygon_mode == GL_LINE) ? 1 : 0;
    /* The repair is the original's else-if AFTER the GL_LINE branch, so a
     * valid GL_LINE mode must not trigger it. */
    out->needs_polygon_mode_repair =
        (polygons && polygon_mode != GL_LINE &&
         polygon_mode != GL_FILL && polygon_mode != GL_POINT)
            ? 1 : 0;
    out->enable_depth_bias = 0;
    if (polygons) {
        switch (polygon_mode) {
            case GL_POINT:
                out->enable_depth_bias = cap_point ? 1 : 0;
                break;
            case GL_LINE:
                out->enable_depth_bias = cap_line ? 1 : 0;
                break;
            case GL_FILL:
            default:
                out->enable_depth_bias = cap_fill ? 1 : 0;
                break;
        }
    }
    (void)mode;
    return 0;
}







/* TES XFB field byte size for a GL type (FLOAT/INT/UINT + vec2/3/4; 0 for
 * unsupported).  Matches the ObjC mglTESXFBFieldByteSize and the packed-write
 * stride contract injected by mglFixMSLTesAsComputeKernel: a zero result
 * means the renderer cannot prove the write stride.  Shared by both gates. */
extern "C"
uint64_t mglRenderTESXFBFieldByteSize(uint64_t gl_type) {
    switch (gl_type) {
        case GL_FLOAT:
        case GL_INT:
        case GL_UNSIGNED_INT:
            return 4u;
        case GL_FLOAT_VEC2:
        case GL_INT_VEC2:
        case GL_UNSIGNED_INT_VEC2:
            return 8u;
        case GL_FLOAT_VEC3:
        case GL_INT_VEC3:
        case GL_UNSIGNED_INT_VEC3:
            return 12u;
        case GL_FLOAT_VEC4:
        case GL_INT_VEC4:
        case GL_UNSIGNED_INT_VEC4:
            return 16u;
        /* XFB packs matrix columns tightly (GL 4.6 §11.1.2.1): mat2 is
         * 4 floats.  Without these sizes, TES XFB stride collapses to 0
         * and gl_in / points capture stay zero (CTS TCTE.gl_in). */
        case GL_FLOAT_MAT2:
            return 16u;
        case GL_FLOAT_MAT2x3:
            return 24u;
        case GL_FLOAT_MAT2x4:
            return 32u;
        case GL_FLOAT_MAT3x2:
            return 24u;
        case GL_FLOAT_MAT3:
            return 36u;
        case GL_FLOAT_MAT3x4:
            return 48u;
        case GL_FLOAT_MAT4x2:
            return 32u;
        case GL_FLOAT_MAT4x3:
            return 48u;
        case GL_FLOAT_MAT4:
            return 64u;
        case GL_DOUBLE:
            return 8u;
        case GL_DOUBLE_VEC2:
            return 16u;
        case GL_DOUBLE_VEC3:
            return 24u;
        case GL_DOUBLE_VEC4:
            return 32u;
        default:
            return 0u;
    }
}

/* Overflow-checked product for tessellation size math; matches the ObjC
 * mglCheckedNSUIntegerProduct ((a != 0 && b > UINT64_MAX / a) rejects).
 * Returns 0 with *result set on success, -1 on bad args / overflow. */
extern "C"
int mglRenderCheckedProduct(uint64_t a, uint64_t b, uint64_t* result) {
    if (!result || (a != 0u && b > UINT64_MAX / a)) {
        return -1;
    }
    *result = a * b;
    return 0;
}

/* 11-bit unsigned float unpack (GL_UNSIGNED_INT_10F_11F_11F_REV CPU decode):
 * 5-bit exponent, 6-bit mantissa, no sign; exponent bias 15.  Denormalized
 * values use 2^(1-15) * mant/64, exp==31 is inf (mant==0) or NaN. */

/* 10-bit unsigned float unpack: 5-bit exponent, 5-bit mantissa, no sign;
 * exponent bias 15.  Denormalized values use 2^(1-15) * mant/32. */

/* Float -> unorm8 with round-to-nearest (0.5 rounds up); matches
 * mglMetalFloatToUnorm8 exactly. */

/* Snorm16 decode: INT16_MIN maps to -1.0 exactly; matches
 * mglMetalSnorm16ToFloat. */

/* Snorm8 decode: INT8_MIN maps to -1.0 exactly; matches
 * mglMetalSnorm8ToFloat. */






extern "C"
uint64_t mglRenderTESXFBVertexStride(const void* program_v) {
    const Program* program = (const Program*)program_v;
    if (!program || program->transform_feedback_varying_count <= 0) {
        return 0u;
    }
    uint64_t stride = 0u;
    for (GLsizei varying = 0;
         varying < program->transform_feedback_varying_count;
         varying++) {
        const char* name = program->transform_feedback_varying_names[varying];
        uint64_t field_bytes = 0u;
        /* Builtins are not in the reflected user-output list; they live at
         * fixed offsets in the TES compute record (pos @0, point size @16). */
        if (name && strcmp(name, "gl_Position") == 0) {
            field_bytes = 16u;
        } else if (name && strcmp(name, "gl_PointSize") == 0) {
            field_bytes = 4u;
        } else {
            const MGLShaderResource* output =
                mglProgramFindStageOutputForXFBName(
                    const_cast<Program*>(program), _TESS_EVALUATION_SHADER,
                    name);
            field_bytes =
                output ? mglRenderTESXFBFieldByteSize(output->gl_type) : 0u;
        }
        if (field_bytes == 0u || stride > UINT64_MAX - field_bytes) {
            return 0u;
        }
        stride += field_bytes;
    }
    return stride;
}





extern "C"
int mglRenderBuildLevelUploadOps(
    const TextureLevel* levels, uint32_t level_count,
    uint32_t texture_type, uint32_t internal_format, uint32_t pixel_format,
    MGLRenderLevelUploadOp* ops, uint32_t ops_capacity,
    uint32_t* op_count_out, uint32_t* short_backing_out, uint32_t* bad_out) {
    if (op_count_out) *op_count_out = 0;
    if (short_backing_out) *short_backing_out = 0;
    if (bad_out) *bad_out = 0;
    if (!levels || !ops || !op_count_out || !short_backing_out || !bad_out ||
        level_count == 0 || ops_capacity < level_count) {
        return -1;
    }
    uint32_t op_count = 0, short_count = 0, bad_count = 0;
    for (uint32_t level = 0; level < level_count; level++) {
        const TextureLevel* l = &levels[level];
        /* mglTextureLevelHasUploadableCPUData, inlined (the compat header is
         * ObjC-typed and cannot be included from this TU). */
        if (!l->complete || !l->data || l->data_size == 0u || l->pitch == 0u) {
            continue;
        }
        switch (l->last_init_source) {
            case kTexImageCopy:
            case kTexImagePBO:
            case kTexSubImageCPU:
            case kTexSubImagePBO:
            case kTexMetalFill:
                break;
            case kTexInitNone:
            case kTexImageNull:
            case kTexRenderTargetWrite:
            default:
                continue;
        }
        if (!(l->has_initialized_data || l->ever_written)) continue;

        MGLRenderLevelUploadPrep prep = {0};
        int prepResult = mglRenderTexturePrepareLevelUpload(
            l, texture_type, internal_format, pixel_format, &prep);
        if (prepResult == -2) {
            MGLRenderLevelUploadOp& op = ops[op_count++];
            op.level = level;
            op.kind = 1u;
            op.width = 0;
            op.height = 0;
            op.bytes_per_row = 0;
            op.bytes_per_image = prep.bytes_per_image;
            op.copy_depth = prep.copy_depth;
            op.available_bytes = prep.available_bytes;
            op.needed_bytes = prep.bytes_per_image * prep.copy_depth;
            op.data = nullptr;
            op.owns_data = 0;
            short_count++;
            continue;
        }
        if (prepResult != 0) {
            bad_count++;
            continue;
        }
        MGLRenderLevelUploadOp& op = ops[op_count++];
        op.level = level;
        op.kind = 0u;
        op.width = MAX((uint32_t)1u, (uint32_t)l->width);
        op.height = MAX((uint32_t)1u, (uint32_t)l->height);
        op.bytes_per_row = prep.bytes_per_row;
        op.bytes_per_image = prep.bytes_per_image;
        op.copy_depth = prep.copy_depth;
        op.available_bytes = prep.available_bytes;
        op.needed_bytes = 0;
        op.data = prep.data;
        op.owns_data = prep.owns_data;
    }
    *op_count_out = op_count;
    *short_backing_out = short_count;
    *bad_out = bad_count;
    return 0;
}

/* per-level CPU upload data preparation. */




extern "C"
uint8_t mglRenderResolveR8SwizzledComponent(uint32_t swizzle, uint8_t red) {
    switch (swizzle) {
        case GL_RED: return red;
        case GL_ALPHA:
        case GL_ONE: return 0xffu;
        case GL_GREEN:
        case GL_BLUE:
        case GL_ZERO:
        default:
            return 0x00u;
    }
}


















extern "C"
uint32_t mglRenderStoredColorComponents(uint32_t internal_format) {
    uint32_t components = mglNumComponentsForFormat(internal_format);
    return components > 0u ? components : 4u;
}




/* RGB->RGBA channel expansion into a caller-provided buffer. */

/* C1: TextureRepackDepthPlanes -> mgl_readback_policy.c */







int mglRenderDescribeDepthStencilDescriptor(
    const void *depth_stencil_descriptor,
    MGLRenderDepthStencilDescriptorState *state_out) {
    if (!state_out) return -1;
    *state_out = {};
    const MTL::DepthStencilDescriptor *descriptor =
        static_cast<const MTL::DepthStencilDescriptor *>(depth_stencil_descriptor);
    if (!descriptor) return -1;
    state_out->depth_compare_function =
        static_cast<uint32_t>(descriptor->depthCompareFunction());
    state_out->depth_write_enabled = descriptor->isDepthWriteEnabled() ? 1u : 0u;
    state_out->front = mglRenderDescribeStencilDescriptor(
        descriptor->frontFaceStencil());
    state_out->back = mglRenderDescribeStencilDescriptor(
        descriptor->backFaceStencil());
    return 0;
}



























/* Prepare: create-or-reuse the pending event and record the GL sync name.
 * Returns a BORROWED event pointer (the owner keeps its reference). */
int mglRenderPendingEventPrepare(MGLPendingEventOwner * owner_handle, GLsizei sync_name, void** event_out) {
    if (event_out) *event_out = nullptr;
    mgl::PendingEventOwner* owner =
        reinterpret_cast<mgl::PendingEventOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !event_out) return -1;
    if (!owner->event) {
        mgl::Renderer& renderer = mgl::renderer();
        std::lock_guard<std::mutex> lock(renderer.mutex);
        if (!renderer.device) return -1;
        MTL::Event* event = renderer.device->newEvent();
        if (!event) return -1;
        owner->event = event;
    }
    owner->sync_name = sync_name;
    *event_out = owner->event;
    return 0;
}

/* Detach: transfer the owner's reference to the caller
 * (the ObjC side bridges it with __bridge_transfer) and clear the slot. */
int mglRenderPendingEventDetach(MGLPendingEventOwner * owner_handle, GLsizei* sync_name_out, void** event_out) {
    if (event_out) *event_out = nullptr;
    if (sync_name_out) *sync_name_out = 0;
    mgl::PendingEventOwner* owner =
        reinterpret_cast<mgl::PendingEventOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !event_out) return -1;
    if (owner->event) {
        *event_out = owner->event;
        owner->event = nullptr;
    }
    if (sync_name_out) *sync_name_out = owner->sync_name;
    owner->sync_name = 0;
    return 0;
}

/* Clear: discard the pending event (owner keeps its allocation). */













int mglRenderSerializeBinaryArchive(void* binary_archive,
                                       void* url,
                                       char* err,
                                       size_t errcap) {
    if (err && errcap) err[0] = '\0';
    MTL::BinaryArchive* archive =
        static_cast<MTL::BinaryArchive*>(binary_archive);
    NS::URL* archiveURL = static_cast<NS::URL*>(url);
    if (!archive || !archiveURL) return -1;
    NS::Error* nsError = nullptr;
    if (!archive->serializeToURL(archiveURL, &nsError)) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    return 0;
}

int mglRenderSetVisibilityResultMode(void* render_encoder,
                                        uint32_t mode,
                                        uint64_t offset) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setVisibilityResultMode(
        static_cast<MTL::VisibilityResultMode>(mode),
        static_cast<NS::UInteger>(offset));
    return 0;
}

int mglRenderSetVisibilityResultModeForRenderEncoderOwner(MGLRenderEncoderOwner * render_encoder_owner, uint32_t mode, uint64_t offset) {
    mgl::RenderEncoderOwner* owner =
        reinterpret_cast<mgl::RenderEncoderOwner*>(static_cast<void*>(render_encoder_owner));
    if (!owner || !owner->encoder || owner->ended) return -1;
    return mglRenderSetVisibilityResultMode(
        owner->encoder, mode, offset);
}

int mglRenderSampleTimestamps(uint64_t* cpu_timestamp_out,
                                 uint64_t* gpu_timestamp_out) {
    if (!cpu_timestamp_out || !gpu_timestamp_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    renderer.device->sampleTimestamps(cpu_timestamp_out, gpu_timestamp_out);
    return 0;
}













void mglRenderInvalidateProgramPipelines(uint64_t program_instance) {
    if (program_instance == 0) return;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    size_t invalidated = 0;
    for (auto it = renderer.computePipelines.begin();
         it != renderer.computePipelines.end();) {
        if (it->first.programInstance == program_instance) {
            if (it->second) it->second->release();
            it = renderer.computePipelines.erase(it);
            invalidated++;
        } else {
            ++it;
        }
    }
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: invalidate compute PSOs "
                "program=%llu count=%zu\n",
                static_cast<unsigned long long>(program_instance), invalidated);
    }
}

namespace {

/* Loads (or returns the cached) MTL::Library for an embedded aux shader asset.
 * Validates the table row (non-empty) and the FNV-1a fingerprint before
 * loading from bytes.  The library is owned by the renderer until shutdown;
 * the returned pointer is borrowed.  Assumes renderer.mutex is held. */


/* Core of mglRenderGetOrCreateAuxComputePipeline: lookup/create against the
 * renderer-lifetime cache.  Assumes renderer.mutex is held. */

/* Core of mglRenderGetOrCreateAuxRenderPipeline: descriptor assembly plus
 * lookup/create against the renderer-lifetime cache.  Assumes renderer.mutex
 * is held. */
int getOrCreateAuxRenderPipelineLocked(mgl::Renderer& renderer,
                                       void* vertex_function,
                                       void* fragment_function,
                                       uint32_t kind,
                                       uint64_t variant,
                                       uint32_t color_format,
                                       uint32_t depth_format,
                                       uint32_t stencil_format,
                                       uint32_t color_write_mask,
                                       int icb_enabled,
                                       uint32_t raster_sample_count,
                                       void** pipeline_out,
                                       char* err,
                                       size_t errcap) {
    mgl::AuxRenderPipelineKey key = {
        kind, variant, color_format, depth_format, stencil_format,
        color_write_mask, raster_sample_count, icb_enabled != 0};
    auto found = renderer.auxRenderPipelines.find(key);
    if (found != renderer.auxRenderPipelines.end()) {
        if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
            fprintf(stderr,
                    "MGL METALCPP: aux render PSO cache hit "
                    "kind=%u variant=%llu\n",
                    kind, static_cast<unsigned long long>(variant));
        }
        found->second->retain();
        *pipeline_out = found->second;
        return 0;
    }
    if (!vertex_function || (!fragment_function &&
                             kind != MGL_RENDER_AUX_RENDER_CLEAR_RECT)) {
        return 1;
    }

    MTL::RenderPipelineDescriptor* descriptor =
        MTL::RenderPipelineDescriptor::alloc()->init();
    if (!descriptor) {
        if (err && errcap) {
            snprintf(err, errcap, "render descriptor allocation failed");
        }
        return -1;
    }
    descriptor->setVertexFunction(
        static_cast<MTL::Function*>(vertex_function));
    descriptor->setFragmentFunction(
        static_cast<MTL::Function*>(fragment_function));
    descriptor->setDepthAttachmentPixelFormat(
        (MTL::PixelFormat)depth_format);
    descriptor->setStencilAttachmentPixelFormat(
        (MTL::PixelFormat)stencil_format);
    descriptor->setRasterSampleCount(raster_sample_count);
    descriptor->setSupportIndirectCommandBuffers(icb_enabled != 0);
    MTL::RenderPipelineColorAttachmentDescriptor* color =
        descriptor->colorAttachments()->object(0);
    color->setPixelFormat((MTL::PixelFormat)color_format);
    color->setWriteMask((MTL::ColorWriteMask)color_write_mask);
    color->setBlendingEnabled(false);

    NS::Error* nsError = nullptr;
    MTL::RenderPipelineState* pipeline =
        renderer.device->newRenderPipelineState(descriptor, &nsError);
    descriptor->release();
    if (!pipeline) {
        mgl::copyError(nsError, err, errcap);
        return -1;
    }
    pipeline->retain();
    renderer.auxRenderPipelines.emplace(key, pipeline);
    if (mgl_env_flag_enabled("MGL_METALCPP_DIAG")) {
        fprintf(stderr,
                "MGL METALCPP: aux render PSO create "
                "kind=%u variant=%llu vs=%p fs=%p\n",
                kind, static_cast<unsigned long long>(variant),
                vertex_function, fragment_function);
    }
    *pipeline_out = pipeline;
    return 0;
}

}  // namespace


int mglRenderGetOrCreateAuxRenderPipeline(
    void* vertex_function,
    void* fragment_function,
    uint32_t kind,
    uint64_t variant,
    uint32_t color_format,
    uint32_t depth_format,
    uint32_t stencil_format,
    uint32_t color_write_mask,
    int icb_enabled,
    uint32_t raster_sample_count,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (!pipeline_out || kind == 0 || raster_sample_count == 0) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return -1;
    }
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap, "Metal-cpp renderer is not initialized");
        return -1;
    }
    return getOrCreateAuxRenderPipelineLocked(
        renderer, vertex_function, fragment_function, kind, variant,
        color_format, depth_format, stencil_format, color_write_mask,
        icb_enabled, raster_sample_count, pipeline_out, err, errcap);
}


int mglRenderGetOrCreateAuxRenderPipelineFromMetallib(
    const unsigned char* bytes,
    size_t size,
    uint64_t asset_hash,
    const char* vertex_entry,
    const char* fragment_entry,
    uint32_t kind,
    uint64_t variant,
    uint32_t color_format,
    uint32_t depth_format,
    uint32_t stencil_format,
    uint32_t color_write_mask,
    int icb_enabled,
    uint32_t raster_sample_count,
    void** pipeline_out,
    char* err,
    size_t errcap) {
    if (pipeline_out) *pipeline_out = nullptr;
    if (!pipeline_out || kind == 0 || raster_sample_count == 0 ||
        !vertex_entry) {
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
    MTL::Function* vertexFunction =
        newAuxEntryFunction(library, vertex_entry, err, errcap);
    if (!vertexFunction) return -1;
    MTL::Function* fragmentFunction =
        newAuxEntryFunction(library, fragment_entry, err, errcap);
    if (fragment_entry && !fragmentFunction) {
        vertexFunction->release();
        if (err && errcap && !err[0]) {
            snprintf(err, errcap, "aux shader entry functions missing");
        }
        return -1;
    }
    int result = getOrCreateAuxRenderPipelineLocked(
        renderer, vertexFunction, fragmentFunction, kind, variant,
        color_format, depth_format, stencil_format, color_write_mask,
        icb_enabled, raster_sample_count, pipeline_out, err, errcap);
    vertexFunction->release();
    if (fragmentFunction) fragmentFunction->release();
    return result;
}








namespace {



} // namespace








































int mglRenderSetComputeBytes(void* compute_encoder,
                                const void* bytes,
                                size_t length,
                                uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder || (!bytes && length != 0)) return -1;
    encoder->setBytes(bytes, static_cast<NS::UInteger>(length), index);
    return 0;
}

int mglRenderSetComputeThreadgroupMemoryLength(void* compute_encoder,
                                                  uint64_t length,
                                                  uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setThreadgroupMemoryLength(static_cast<NS::UInteger>(length),
                                        index);
    return 0;
}


























/* mglRenderClassifyProcessGLState moved to mgl_render_pass_plan.c (O1.1) */

/* Per-sample multisample values are needed when the fragment stage reads a
 * sample-indexed builtin or interpolates at an explicit sample/offset.  The
 * facts come from the exact per-stage builtin mask (see
 * mglProgramStageBuiltinMask); the previous version scanned the shader source
 * for these names.  gl_NumSamples is deliberately excluded: it is a uniform
 * count, not a per-sample value. */
int mglRenderFragmentNeedsPerSampleMSValues(const Program *program) {
    return mglProgramStageUsesBuiltin(program, _FRAGMENT_SHADER,
                                      MGL_AIR_BUILTIN_SAMPLE_ID |
                                          MGL_AIR_BUILTIN_SAMPLE_POSITION |
                                          MGL_AIR_BUILTIN_SAMPLE_MASK |
                                          MGL_AIR_BUILTIN_INTERPOLATE_AT_SAMPLE |
                                          MGL_AIR_BUILTIN_INTERPOLATE_AT_OFFSET |
                                          MGL_AIR_BUILTIN_SAMPLE_INTERPOLATION);
}



void mglRenderFillFragCoordSlot(int use_fragcoord, int use_sample,
                                uint32_t pass_height, int lower_left,
                                uint32_t num_samples, uint32_t sample_buffers,
                                int ms_loop, uint32_t forced_sample_id,
                                float out[4]) {
    if (!out) {
        return;
    }
    uint32_t sb_bits = sample_buffers;
    if (ms_loop) {
        sb_bits = 1u | 0x80000000u | ((forced_sample_id & 0xffu) << 8);
    }
    float ns_as_float = 0.f;
    float sb_as_float = 0.f;
    memcpy(&ns_as_float, &num_samples, sizeof(ns_as_float));
    memcpy(&sb_as_float, &sb_bits, sizeof(sb_as_float));
    if (use_sample && !use_fragcoord) {
        out[0] = 0.f;
        out[1] = 0.f;
        out[2] = ns_as_float;
        out[3] = sb_as_float;
        return;
    }
    out[0] = (float)pass_height;
    out[1] = lower_left ? 1.f : 0.f;
    out[2] = use_sample ? ns_as_float : 0.f;
    out[3] = use_sample ? sb_as_float : 0.f;
}

void mglRenderClampLodBiasArray(float *bias, uint32_t count, float biasmax) {
    if (!bias) {
        return;
    }
    for (uint32_t i = 0u; i < count; i++) {
        float v = bias[i];
        if (biasmax > 0.f) {
            if (v > biasmax) {
                v = biasmax;
            } else if (v < -biasmax) {
                v = -biasmax;
            }
        }
        bias[i] = v;
    }
}



uint64_t mglXfbAdvanceWriteOffset(uint64_t current, uint64_t written) {
    if (written > UINT64_MAX - current) {
        return UINT64_MAX;
    }
    return current + written;
}































/* does the submission own exactly this command buffer?
 * Replaces the ObjC MGLCommandState.detachedCommandBuffer mirror used to
 * guard commit/release of a detached submission. */












int mglRenderAllocateMDIScratch(MGLMDIScratchOwner * owner_handle, uint64_t length, uint64_t alignment, void** buffer_out, uint64_t* offset_out, uint64_t* capacity_out) {
    if (buffer_out) *buffer_out = nullptr;
    if (offset_out) *offset_out = 0;
    if (capacity_out) *capacity_out = 0;
    mgl::MDIScratchOwner* owner =
        reinterpret_cast<mgl::MDIScratchOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !buffer_out || !offset_out || length == 0 ||
        alignment == 0 || (alignment & (alignment - 1u)) != 0u) {
        return -1;
    }

    const uint64_t mask = alignment - 1u;
    if (owner->offset > std::numeric_limits<uint64_t>::max() - mask) {
        return -1;
    }
    uint64_t alignedOffset = (owner->offset + mask) & ~mask;
    if (length > std::numeric_limits<uint64_t>::max() - alignedOffset) {
        return -1;
    }
    uint64_t required = alignedOffset + length;
    if (!owner->buffer || required > owner->capacity) {
        uint64_t nextCapacity = owner->capacity;
        if (nextCapacity == 0) nextCapacity = 64u * 1024u;
        while (nextCapacity < required) {
            if (nextCapacity > std::numeric_limits<uint64_t>::max() / 2u) {
                nextCapacity = required;
                break;
            }
            nextCapacity *= 2u;
        }
        if (nextCapacity > std::numeric_limits<NS::UInteger>::max()) {
            return -1;
        }
        mgl::Renderer& renderer = mgl::renderer();
        std::lock_guard<std::mutex> lock(renderer.mutex);
        if (!renderer.device) return -1;
        MTL::Buffer* next = renderer.device->newBuffer(
            static_cast<NS::UInteger>(nextCapacity),
            MTL::ResourceStorageModeShared);
        if (!next) return -1;
        if (owner->buffer) owner->buffer->release();
        owner->buffer = next;
        owner->capacity = nextCapacity;
        alignedOffset = 0;
        required = length;
    }

    owner->offset = required;
    *buffer_out = owner->buffer;
    *offset_out = alignedOffset;
    if (capacity_out) *capacity_out = owner->capacity;
    return 0;
}

























int mglRenderSetFboMatchCache(MGLRenderPassIdentityOwner * owner_handle, const MGLRenderFboMatchCacheState* cache) {
    mgl::RenderPassIdentityOwner* owner =
        reinterpret_cast<mgl::RenderPassIdentityOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !cache || cache->fbo_name == 0) return -1;
    owner->cache = *cache;
    owner->cache.result = cache->result != 0;
    owner->cache_valid = true;
    return 0;
}








































int mglRenderEncodeTextureUploadLayers(
    void* command_buffer,
    void* source_buffer,
    uint64_t source_offset,
    uint64_t source_bytes_per_row,
    uint64_t source_bytes_per_image,
    uint64_t source_layer_stride,
    uint64_t source_width,
    uint64_t source_height,
    uint64_t source_depth,
    void* destination_texture,
    uint64_t destination_base_slice,
    uint64_t layer_count,
    uint64_t destination_level,
    uint64_t destination_x,
    uint64_t destination_y,
    uint64_t destination_z) {
    MTL::CommandBuffer* command =
        static_cast<MTL::CommandBuffer*>(command_buffer);
    MTL::Buffer* source = static_cast<MTL::Buffer*>(source_buffer);
    MTL::Texture* destination =
        static_cast<MTL::Texture*>(destination_texture);
    if (!command || !source || !destination || source_width == 0 ||
        source_height == 0 || source_depth == 0 ||
        source_bytes_per_row == 0 || source_bytes_per_image == 0 ||
        layer_count == 0 || (layer_count > 1u && source_layer_stride == 0u)) {
        return -1;
    }

    const uint64_t last_layer = layer_count - 1u;
    if ((source_layer_stride != 0u &&
         last_layer > (std::numeric_limits<uint64_t>::max() - source_offset) /
                          source_layer_stride) ||
        last_layer > std::numeric_limits<uint64_t>::max() -
                         destination_base_slice) {
        return -1;
    }

    uint64_t source_layer_span = 0u;
    if (source_depth > std::numeric_limits<uint64_t>::max() /
                           source_bytes_per_image) {
        return -1;
    }
    source_layer_span = source_bytes_per_image * source_depth;
    const uint64_t last_source_offset =
        source_offset + last_layer * source_layer_stride;
    if (last_source_offset > source->length() ||
        source_layer_span > source->length() - last_source_offset) {
        return -1;
    }

    if (destination_level >= destination->mipmapLevelCount()) {
        return -1;
    }
    uint64_t destination_slice_count = destination->arrayLength();
    switch (destination->textureType()) {
        case MTL::TextureTypeCube:
            destination_slice_count = 6u;
            break;
        case MTL::TextureTypeCubeArray:
            if (destination_slice_count >
                std::numeric_limits<uint64_t>::max() / 6u) {
                return -1;
            }
            destination_slice_count *= 6u;
            break;
        default:
            break;
    }
    if (destination_base_slice >= destination_slice_count ||
        layer_count > destination_slice_count - destination_base_slice) {
        return -1;
    }

    const uint64_t mip_width =
        std::max<uint64_t>(1u, destination->width() >> destination_level);
    const uint64_t mip_height =
        std::max<uint64_t>(1u, destination->height() >> destination_level);
    const uint64_t mip_depth =
        std::max<uint64_t>(1u, destination->depth() >> destination_level);
    if (destination_x > mip_width || source_width > mip_width - destination_x ||
        destination_y > mip_height ||
        source_height > mip_height - destination_y ||
        destination_z > mip_depth || source_depth > mip_depth - destination_z) {
        return -1;
    }

    MTL::BlitCommandEncoder* encoder = command->blitCommandEncoder();
    if (!encoder) return -1;
    for (uint64_t layer = 0u; layer < layer_count; ++layer) {
        encoder->copyFromBuffer(
            source,
            static_cast<NS::UInteger>(source_offset +
                                      layer * source_layer_stride),
            static_cast<NS::UInteger>(source_bytes_per_row),
            static_cast<NS::UInteger>(source_bytes_per_image),
            MTL::Size(source_width, source_height, source_depth), destination,
            static_cast<NS::UInteger>(destination_base_slice + layer),
            static_cast<NS::UInteger>(destination_level),
            MTL::Origin(destination_x, destination_y, destination_z));
    }
    encoder->endEncoding();
    return 0;
}


int mglRenderEncodeTextureUpload(void* command_buffer,
                                    void* source_buffer,
                                    uint64_t source_offset,
                                    uint64_t source_bytes_per_row,
                                    uint64_t source_bytes_per_image,
                                    uint64_t source_width,
                                    uint64_t source_height,
                                    uint64_t source_depth,
                                    void* destination_texture,
                                    uint64_t destination_slice,
                                    uint64_t destination_level,
                                    uint64_t destination_x,
                                    uint64_t destination_y,
                                    uint64_t destination_z) {
    return mglRenderEncodeTextureUploadLayers(
        command_buffer, source_buffer, source_offset, source_bytes_per_row,
        source_bytes_per_image, 0u, source_width, source_height, source_depth,
        destination_texture, destination_slice, 1u, destination_level,
        destination_x, destination_y, destination_z);
}


int mglRenderBlitCopyBufferToTexture(void* blit_encoder,
                                        void* source_buffer,
                                        uint64_t source_offset,
                                        uint64_t source_bytes_per_row,
                                        uint64_t source_bytes_per_image,
                                        uint64_t source_width,
                                        uint64_t source_height,
                                        uint64_t source_depth,
                                        void* destination_texture,
                                        uint64_t destination_slice,
                                        uint64_t destination_level,
                                        uint64_t destination_x,
                                        uint64_t destination_y,
                                        uint64_t destination_z) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Buffer* source = static_cast<MTL::Buffer*>(source_buffer);
    MTL::Texture* destination =
        static_cast<MTL::Texture*>(destination_texture);
    if (!encoder || !source || !destination || source_width == 0 ||
        source_height == 0 || source_depth == 0 ||
        source_bytes_per_row == 0 || source_bytes_per_image == 0) {
        return -1;
    }
    encoder->copyFromBuffer(
        source, static_cast<NS::UInteger>(source_offset),
        static_cast<NS::UInteger>(source_bytes_per_row),
        static_cast<NS::UInteger>(source_bytes_per_image),
        MTL::Size(source_width, source_height, source_depth), destination,
        static_cast<NS::UInteger>(destination_slice),
        static_cast<NS::UInteger>(destination_level),
        MTL::Origin(destination_x, destination_y, destination_z));
    return 0;
}



int mglRenderBlitCopyTexture(void* blit_encoder,
                                void* source_texture,
                                uint64_t source_slice,
                                uint64_t source_level,
                                uint64_t source_x,
                                uint64_t source_y,
                                uint64_t source_z,
                                uint64_t width,
                                uint64_t height,
                                uint64_t depth,
                                void* destination_texture,
                                uint64_t destination_slice,
                                uint64_t destination_level,
                                uint64_t destination_x,
                                uint64_t destination_y,
                                uint64_t destination_z) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Texture* source = static_cast<MTL::Texture*>(source_texture);
    MTL::Texture* destination =
        static_cast<MTL::Texture*>(destination_texture);
    if (!encoder || !source || !destination || width == 0 || height == 0 ||
        depth == 0) {
        return -1;
    }
    encoder->copyFromTexture(
        source, static_cast<NS::UInteger>(source_slice),
        static_cast<NS::UInteger>(source_level),
        MTL::Origin(source_x, source_y, source_z),
        MTL::Size(width, height, depth), destination,
        static_cast<NS::UInteger>(destination_slice),
        static_cast<NS::UInteger>(destination_level),
        MTL::Origin(destination_x, destination_y, destination_z));
    return 0;
}

int mglRenderBlitCopyTextureToBuffer(
    void* blit_encoder,
    void* source_texture,
    uint64_t source_slice,
    uint64_t source_level,
    uint64_t source_x,
    uint64_t source_y,
    uint64_t source_z,
    uint64_t width,
    uint64_t height,
    uint64_t depth,
    void* destination_buffer,
    uint64_t destination_offset,
    uint64_t destination_bytes_per_row,
    uint64_t destination_bytes_per_image) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Texture* source = static_cast<MTL::Texture*>(source_texture);
    MTL::Buffer* destination =
        static_cast<MTL::Buffer*>(destination_buffer);
    if (!encoder || !source || !destination || width == 0 || height == 0 ||
        depth == 0 || destination_bytes_per_row == 0 ||
        destination_bytes_per_image == 0) {
        return -1;
    }
    encoder->copyFromTexture(
        source, static_cast<NS::UInteger>(source_slice),
        static_cast<NS::UInteger>(source_level),
        MTL::Origin(source_x, source_y, source_z),
        MTL::Size(width, height, depth), destination,
        static_cast<NS::UInteger>(destination_offset),
        static_cast<NS::UInteger>(destination_bytes_per_row),
        static_cast<NS::UInteger>(destination_bytes_per_image));
    return 0;
}
































namespace {
enum {
    kCmdDrawArrays = 0,
    kCmdDrawElements = 1,
    kCmdDrawArraysInstanced = 2,
    kCmdDrawElementsInstanced = 3,
    kCmdDrawElementsBaseVertex = 4,
    kCmdDrawElementsInstancedBaseVertex = 5,
    kCmdDrawArraysInstancedBaseInstance = 6,
    kCmdDrawElementsInstancedBaseInstance = 7,
    kCmdDrawElementsInstancedBaseVertexBaseInstance = 8,
};
}

int mglRenderReplayBatchDraws(void* render_encoder,
                                 const MGLRenderReplayBatch* batch,
                                 char* err,
                                 size_t errcap) {
    if (err && errcap) err[0] = '\0';
    if (!render_encoder || !batch || !batch->commands ||
        batch->command_count == 0) {
        if (err && errcap) snprintf(err, errcap, "bad args");
        return MGL_RENDER_REPLAY_BATCH_ERROR;
    }
    if (batch->command_count > MGL_RENDER_REPLAY_BATCH_MAX_COMMANDS) {
        return MGL_RENDER_REPLAY_BATCH_NEEDS_OBJC;
    }
    for (uint32_t i = 0; i < batch->command_count; i++) {
        const MGLRenderReplayBatchCommand* cmd = &batch->commands[i];
        if (cmd->count == 0) {
            continue;
        }
        MGLRenderDrawPlan plan = {};
        plan.primitive_type = batch->primitive_type;
        switch (cmd->cmd_type) {

            case kCmdDrawArrays:
            case kCmdDrawArraysInstanced:
            case kCmdDrawArraysInstancedBaseInstance:
                plan.kind = MGL_RENDER_DRAW_ARRAY;
                plan.vertex_start = static_cast<uint64_t>(cmd->first);
                plan.vertex_count = cmd->count;
                plan.instance_count = cmd->instance_count;
                plan.base_instance = cmd->base_instance;
                break;

            case kCmdDrawElements:
            case kCmdDrawElementsInstanced:
            case kCmdDrawElementsBaseVertex:
            case kCmdDrawElementsInstancedBaseVertex:
            case kCmdDrawElementsInstancedBaseInstance:
            case kCmdDrawElementsInstancedBaseVertexBaseInstance:
                if (!cmd->index_buffer ||
                    cmd->index_type == 0xFFFFFFFFu) {
                    if (err && errcap) {
                        snprintf(err, errcap,
                                 "replay command %u: unready index buffer",
                                 i);
                    }
                    return MGL_RENDER_REPLAY_BATCH_NEEDS_OBJC;
                }
                plan.kind = MGL_RENDER_DRAW_INDEXED;
                plan.index_count = cmd->count;
                plan.index_type = cmd->index_type;
                plan.index_buffer = cmd->index_buffer;
                plan.index_buffer_offset = cmd->index_buffer_offset;
                plan.base_vertex = cmd->base_vertex;
                plan.instance_count = cmd->instance_count;
                plan.base_instance = cmd->base_instance;
                break;
            default:
                if (err && errcap) {
                    snprintf(err, errcap,
                             "replay command %u: unknown cmd_type %u",
                             i, cmd->cmd_type);
                }
                return MGL_RENDER_REPLAY_BATCH_NEEDS_OBJC;
        }
        if (mglRenderEncodeDraw(render_encoder, &plan, err, errcap) != 0) {
            if (err && errcap && !err[0]) {
                snprintf(err, errcap, "replay command %u encode failed", i);
            }
            return MGL_RENDER_REPLAY_BATCH_NEEDS_OBJC;
        }
    }
    return MGL_RENDER_REPLAY_BATCH_OK;
}

int mglRenderSetRenderBytes(void* render_encoder,
                               const void* bytes,
                               size_t length,
                               uint32_t stage,
                               uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || (!bytes && length != 0) ||
        stage > MGL_RENDER_BINDING_STAGE_FRAGMENT) {
        return -1;
    }
    if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
        encoder->setVertexBytes(bytes, static_cast<NS::UInteger>(length), index);
    } else {
        encoder->setFragmentBytes(bytes, static_cast<NS::UInteger>(length),
                                  index);
    }
    return 0;
}


int mglRenderSetRenderDepthStencilState(void* render_encoder,
                                           void* depth_stencil_state) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::DepthStencilState* state =
        static_cast<MTL::DepthStencilState*>(depth_stencil_state);
    if (!encoder || !state) return -1;
    encoder->setDepthStencilState(state);
    return 0;
}



int mglRenderSetRenderViewport(void* render_encoder,
                                  double origin_x,
                                  double origin_y,
                                  double width,
                                  double height,
                                  double znear,
                                  double zfar) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setViewport(MTL::Viewport(origin_x, origin_y, width, height,
                                       znear, zfar));
    return 0;
}

int mglRenderSetRenderScissor(void* render_encoder,
                                 uint64_t x,
                                 uint64_t y,
                                 uint64_t width,
                                 uint64_t height) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setScissorRect(MTL::ScissorRect(x, y, width, height));
    return 0;
}

int mglRenderSetDepthClipMode(void* render_encoder, uint32_t mode) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setDepthClipMode(static_cast<MTL::DepthClipMode>(mode));
    return 0;
}

int mglRenderSetStencilReferenceValues(void* render_encoder,
                                          uint32_t front_reference,
                                          uint32_t back_reference) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder) return -1;
    encoder->setStencilReferenceValues(front_reference, back_reference);
    return 0;
}















int mglRenderSetRenderBytesForOwner(MGLRenderEncoderOwner * render_encoder_owner, const void* bytes, size_t length, uint32_t stage, uint32_t index) {
    return mglRenderSetRenderBytes(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        bytes, length, stage, index);
}


int mglRenderSetRenderDepthStencilStateForOwner(MGLRenderEncoderOwner * render_encoder_owner, void* depth_stencil_state) {
    return mglRenderSetRenderDepthStencilState(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        depth_stencil_state);
}



int mglRenderSetRenderViewportForOwner(MGLRenderEncoderOwner * render_encoder_owner, double origin_x, double origin_y, double width, double height, double znear, double zfar) {
    return mglRenderSetRenderViewport(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        origin_x, origin_y, width, height, znear, zfar);
}

int mglRenderSetRenderScissorForOwner(MGLRenderEncoderOwner * render_encoder_owner, uint64_t x, uint64_t y, uint64_t width, uint64_t height) {
    return mglRenderSetRenderScissor(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        x, y, width, height);
}

int mglRenderSetDepthClipModeForOwner(MGLRenderEncoderOwner * render_encoder_owner, uint32_t mode) {
    return mglRenderSetDepthClipMode(
        mglRenderActiveRenderEncoder(render_encoder_owner), mode);
}

int mglRenderSetStencilReferenceValuesForOwner(MGLRenderEncoderOwner * render_encoder_owner, uint32_t front_reference, uint32_t back_reference) {
    return mglRenderSetStencilReferenceValues(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        front_reference, back_reference);
}









int mglRenderUseRenderResource(void* render_encoder,
                                  void* resource,
                                  uint32_t usage,
                                  uint32_t stages) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::Resource* object = static_cast<MTL::Resource*>(resource);
    if (!encoder || !object) return -1;
    encoder->useResource(object, static_cast<MTL::ResourceUsage>(usage),
                         static_cast<MTL::RenderStages>(stages));
    return 0;
}

int mglRenderExecuteIndirectCommands(void* render_encoder,
                                        void* indirect_buffer,
                                        uint64_t location,
                                        uint64_t length) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    MTL::IndirectCommandBuffer* buffer =
        static_cast<MTL::IndirectCommandBuffer*>(indirect_buffer);
    if (!encoder || !buffer || length == 0) return -1;
    encoder->executeCommandsInBuffer(buffer, NS::Range(location, length));
    return 0;
}

int mglRenderReplayBatchDrawsForRenderEncoderOwner(MGLRenderEncoderOwner * render_encoder_owner, const MGLRenderReplayBatch* batch, char* err, size_t errcap) {
    return mglRenderReplayBatchDraws(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        batch, err, errcap);
}

int mglRenderUseRenderResourceForOwner(MGLRenderEncoderOwner * render_encoder_owner, void* resource, uint32_t usage, uint32_t stages) {
    return mglRenderUseRenderResource(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        resource, usage, stages);
}

int mglRenderExecuteIndirectCommandsForOwner(MGLRenderEncoderOwner * render_encoder_owner, void* indirect_buffer, uint64_t location, uint64_t length) {
    return mglRenderExecuteIndirectCommands(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        indirect_buffer, location, length);
}

const char *mglRenderVertexFormatName(uint32_t format) {
    switch (static_cast<MTL::VertexFormat>(format)) {
        case MTL::VertexFormatFloat: return "Float";
        case MTL::VertexFormatFloat2: return "Float2";
        case MTL::VertexFormatFloat3: return "Float3";
        case MTL::VertexFormatFloat4: return "Float4";
        case MTL::VertexFormatUChar4: return "UChar4";
        case MTL::VertexFormatUChar4Normalized: return "UChar4Normalized";
        case MTL::VertexFormatUChar3: return "UChar3";
        case MTL::VertexFormatUChar3Normalized: return "UChar3Normalized";
        case MTL::VertexFormatUChar2: return "UChar2";
        case MTL::VertexFormatUChar2Normalized: return "UChar2Normalized";
        case MTL::VertexFormatUChar: return "UChar";
        case MTL::VertexFormatUCharNormalized: return "UCharNormalized";
        case MTL::VertexFormatShort: return "Short";
        case MTL::VertexFormatShort2: return "Short2";
        case MTL::VertexFormatShort3: return "Short3";
        case MTL::VertexFormatShort4: return "Short4";
        case MTL::VertexFormatShortNormalized: return "ShortNormalized";
        case MTL::VertexFormatShort2Normalized: return "Short2Normalized";
        case MTL::VertexFormatShort3Normalized: return "Short3Normalized";
        case MTL::VertexFormatShort4Normalized: return "Short4Normalized";
        case MTL::VertexFormatUShort: return "UShort";
        case MTL::VertexFormatUShort2: return "UShort2";
        case MTL::VertexFormatUShort3: return "UShort3";
        case MTL::VertexFormatUShort4: return "UShort4";
        case MTL::VertexFormatUShortNormalized: return "UShortNormalized";
        case MTL::VertexFormatUShort2Normalized: return "UShort2Normalized";
        case MTL::VertexFormatUShort3Normalized: return "UShort3Normalized";
        case MTL::VertexFormatUShort4Normalized: return "UShort4Normalized";
        case MTL::VertexFormatUInt1010102Normalized: return "UInt1010102Normalized";
        case MTL::VertexFormatInt1010102Normalized: return "Int1010102Normalized";
        default: return "Unknown";
    }
}

uint64_t mglRenderVertexDescriptorSignature(const void *descriptor) {
    const MTL::VertexDescriptor *vertex =
        static_cast<const MTL::VertexDescriptor *>(descriptor);
    uint64_t hash = 1469598103934665603ull;
    if (!vertex) return hash;
    MTL::VertexAttributeDescriptorArray *attributes = vertex->attributes();
    MTL::VertexBufferLayoutDescriptorArray *layouts = vertex->layouts();
    for (uint32_t i = 0; i < 32u; ++i) {
        MTL::VertexAttributeDescriptor *attrib = attributes ? attributes->object(i) : nullptr;
        if (!attrib) continue;
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(attrib->format()));
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(attrib->offset()));
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(attrib->bufferIndex()));
    }
    for (uint32_t i = 0; i < 31u; ++i) {
        MTL::VertexBufferLayoutDescriptor *layout = layouts ? layouts->object(i) : nullptr;
        if (!layout) continue;
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(layout->stride()));
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(layout->stepFunction()));
        hash = mglRenderHashStepU64(hash, static_cast<uint64_t>(layout->stepRate()));
    }
    return hash;
}





const char *mglRenderStoreActionName(uint32_t action) {
    switch (static_cast<MTL::StoreAction>(action)) {
        case MTL::StoreActionDontCare: return "DontCare";
        case MTL::StoreActionStore: return "Store";
        case MTL::StoreActionMultisampleResolve: return "MSResolve";
        case MTL::StoreActionStoreAndMultisampleResolve: return "Store+MSResolve";
        case MTL::StoreActionUnknown: return "Unknown";
        default: return "Other";
    }
}

} // extern "C"






















