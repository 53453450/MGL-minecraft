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

int mglRenderGetTextureInfo(const void *texture,
                               MGLRenderTextureInfo *info_out) {
    if (info_out) *info_out = {};
    const MTL::Texture *object = static_cast<const MTL::Texture *>(texture);
    if (!object || !info_out) return -1;
    info_out->pixel_format = static_cast<uint32_t>(object->pixelFormat());
    info_out->texture_type = static_cast<uint32_t>(object->textureType());
    info_out->width = object->width();
    info_out->height = object->height();
    info_out->depth = object->depth();
    info_out->mipmap_level_count = object->mipmapLevelCount();
    info_out->array_length = object->arrayLength();
    info_out->usage = static_cast<uint64_t>(object->usage());
    info_out->storage_mode = static_cast<uint32_t>(object->storageMode());
    info_out->sample_count = object->sampleCount();
    return 0;
}

int mglRenderTextureIsFramebufferOnly(const void *texture) {
    const MTL::Texture *object = static_cast<const MTL::Texture *>(texture);
    return object && object->isFramebufferOnly() ? 1 : 0;
}

int mglRenderCreateTextureView(void* texture,
                                  uint32_t pixel_format,
                                  void** texture_view_out) {
    if (texture_view_out) *texture_view_out = nullptr;
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!source || !texture_view_out) return -1;
    MTL::Texture* view = source->newTextureView(
        static_cast<MTL::PixelFormat>(pixel_format));
    if (!view) return -1;
    *texture_view_out = view;
    return 0;
}

int mglRenderSampledTextureViewForBaseLevel(
    Texture *texture_object,
    void *source_texture,
    void **view_out) {
    if (view_out) *view_out = nullptr;
    MTL::Texture *source = static_cast<MTL::Texture *>(source_texture);
    if (!texture_object || !source || !view_out || texture_object->mipmap_levels == 0u) {
        if (view_out) *view_out = source_texture;
        return 0;
    }
    const uint32_t base = texture_object->params.base_level;
    if (base >= texture_object->mipmap_levels || base >= source->mipmapLevelCount()) {
        *view_out = source_texture;
        return 0;
    }
    uint32_t max_level = texture_object->params.max_level == 1000u
        ? texture_object->mipmap_levels - 1u : texture_object->params.max_level;
    if (max_level < base) max_level = base;
    if (max_level >= texture_object->mipmap_levels) max_level = texture_object->mipmap_levels - 1u;
    if (max_level >= source->mipmapLevelCount()) max_level = source->mipmapLevelCount() - 1u;
    const uint64_t level_count = static_cast<uint64_t>(max_level - base + 1u);
    uint64_t slice_count = source->arrayLength();
    const auto type = source->textureType();
    if (type == MTL::TextureTypeCube || type == MTL::TextureTypeCubeArray) {
        slice_count *= 6u;
    }
    const uint32_t components =
        mglRenderStoredColorComponents(texture_object->internalformat);
    uint32_t swizzle_red = mglRenderMTLSwizzleForGLSwizzle(
        texture_object->params.swizzle_r, components);
    uint32_t swizzle_green = mglRenderMTLSwizzleForGLSwizzle(
        texture_object->params.swizzle_g, components);
    uint32_t swizzle_blue = mglRenderMTLSwizzleForGLSwizzle(
        texture_object->params.swizzle_b, components);
    uint32_t swizzle_alpha = mglRenderMTLSwizzleForGLSwizzle(
        texture_object->params.swizzle_a, components);
    /* Formats expanded/baked at upload must not get Metal view swizzle. */
    const bool upload_swizzle_baked =
        texture_object->params.swizzled &&
        !texture_object->is_render_target &&
        mglRenderTextureSwizzleUsesUploadBake(
            texture_object->internalformat, 1,
            static_cast<uint32_t>(source->pixelFormat())) != 0;
    if (upload_swizzle_baked) {
        swizzle_red = static_cast<uint32_t>(MTL::TextureSwizzleRed);
        swizzle_green = static_cast<uint32_t>(MTL::TextureSwizzleGreen);
        swizzle_blue = static_cast<uint32_t>(MTL::TextureSwizzleBlue);
        swizzle_alpha = static_cast<uint32_t>(MTL::TextureSwizzleAlpha);
    }
    const bool identity =
        swizzle_red == static_cast<uint32_t>(MTL::TextureSwizzleRed) &&
        swizzle_green == static_cast<uint32_t>(MTL::TextureSwizzleGreen) &&
        swizzle_blue == static_cast<uint32_t>(MTL::TextureSwizzleBlue) &&
        swizzle_alpha == static_cast<uint32_t>(MTL::TextureSwizzleAlpha);
    if (level_count == 0u ||
        (base == 0u && level_count >= source->mipmapLevelCount() &&
         identity)) {
        *view_out = source_texture;
        return 0;
    }
    if (texture_object->mtl_base_level_view &&
        texture_object->mtl_base_level_view_source == source_texture &&
        texture_object->mtl_base_level_view_base == base &&
        texture_object->mtl_base_level_view_max == max_level &&
        texture_object->mtl_base_level_view_swizzle_r ==
            (GLuint)texture_object->params.swizzle_r &&
        texture_object->mtl_base_level_view_swizzle_g ==
            (GLuint)texture_object->params.swizzle_g &&
        texture_object->mtl_base_level_view_swizzle_b ==
            (GLuint)texture_object->params.swizzle_b &&
        texture_object->mtl_base_level_view_swizzle_a ==
            (GLuint)texture_object->params.swizzle_a) {
        *view_out = texture_object->mtl_base_level_view;
        return 0;
    }

    void *view_handle = nullptr;
    if (mglRenderCreateTextureViewRange(
            source_texture, static_cast<uint32_t>(source->pixelFormat()),
            static_cast<uint32_t>(type), base, level_count, 0u, slice_count,
            identity ? 0 : 1, swizzle_red, swizzle_green, swizzle_blue,
            swizzle_alpha, &view_handle) != 0 || !view_handle) {
        *view_out = source_texture;
        return 0;
    }
    if (texture_object->mtl_base_level_view) {
        static_cast<NS::Object *>(texture_object->mtl_base_level_view)->release();
    }
    static_cast<NS::Object *>(view_handle)->retain();
    texture_object->mtl_base_level_view = view_handle;
    texture_object->mtl_base_level_view_source = source_texture;
    texture_object->mtl_base_level_view_base = base;
    texture_object->mtl_base_level_view_max = max_level;
    texture_object->mtl_base_level_view_swizzle_r =
        (GLuint)texture_object->params.swizzle_r;
    texture_object->mtl_base_level_view_swizzle_g =
        (GLuint)texture_object->params.swizzle_g;
    texture_object->mtl_base_level_view_swizzle_b =
        (GLuint)texture_object->params.swizzle_b;
    texture_object->mtl_base_level_view_swizzle_a =
        (GLuint)texture_object->params.swizzle_a;
    static_cast<NS::Object *>(view_handle)->release();
    *view_out = texture_object->mtl_base_level_view;
    return 0;
}

extern "C"
int mglRenderTextureTargetPlan(
    uint32_t gl_target,
    uint32_t sample_count,
    MGLRenderTextureTargetPlan* plan_out) {
    if (!plan_out) return -1;
    *plan_out = {};
    plan_out->num_faces = 1u;

    switch (gl_target) {
        case GL_TEXTURE_1D:
            plan_out->texture_type = static_cast<uint32_t>(MTL::TextureType2D);
            plan_out->texture_1d_backed_by_2d = 1u;
            return 0;
        case GL_RENDERBUFFER:
            plan_out->texture_type = static_cast<uint32_t>(
                sample_count > 1u ? MTL::TextureType2DMultisample
                                  : MTL::TextureType2D);
            return 0;
        case GL_TEXTURE_1D_ARRAY:
            /* AIR lowers sampler1DArray to texture2d_array, and Metal cannot
             * view a Texture1DArray as Texture2DArray. */
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureType2DArray);
            plan_out->is_array = 1u;
            plan_out->texture_1d_array_backed_by_2d_array = 1u;
            return 0;
        case GL_TEXTURE_2D:
        case GL_TEXTURE_RECTANGLE:
            plan_out->texture_type = static_cast<uint32_t>(MTL::TextureType2D);
            return 0;
        case GL_TEXTURE_2D_ARRAY:
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureType2DArray);
            plan_out->is_array = 1u;
            return 0;
        case GL_TEXTURE_2D_MULTISAMPLE:
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureType2DMultisample);
            return 0;
        case GL_TEXTURE_CUBE_MAP:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_X:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_X:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Y:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Y:
        case GL_TEXTURE_CUBE_MAP_POSITIVE_Z:
        case GL_TEXTURE_CUBE_MAP_NEGATIVE_Z:
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureTypeCube);
            plan_out->num_faces = 6u;
            return 0;
        case GL_TEXTURE_CUBE_MAP_ARRAY:
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureTypeCubeArray);
            /* GL stores cube-array levels in faces[0] with depth=cubes*6.
             * Iterating 6 faces during CPU upload marks the create incomplete
             * (faces 1-5 empty) and leaves DIRTY_TEXTURE_DATA set — later
             * binds can fight imageStore results. */
            plan_out->num_faces = 1u;
            plan_out->is_array = 1u;
            return 0;
        case GL_TEXTURE_3D:
            plan_out->texture_type = static_cast<uint32_t>(MTL::TextureType3D);
            return 0;
        case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
            plan_out->texture_type =
                static_cast<uint32_t>(MTL::TextureType2DMultisampleArray);
            plan_out->is_array = 1u;
            return 0;
        default:
            *plan_out = {};
            return -1;
    }
}

extern "C"
uint32_t mglRenderTextureTypeForShaderResource(
    uint32_t has_resource,
    uint32_t image_dim,
    uint32_t image_arrayed,
    uint32_t image_multisampled) {
    if (!has_resource) return 0u;
    switch (image_dim) {
        case MGL_IMAGE_DIM_1D:
            /* GL 1D/1D-array textures are backed by Metal 2D/2D-array storage
             * (see mglRenderTextureTargetPlan); AIR lowers sampler1DArray the same way. */
            return static_cast<uint32_t>(
                image_arrayed ? MTL::TextureType2DArray : MTL::TextureType2D);
        case MGL_IMAGE_DIM_2D:
            if (image_multisampled) {
                /* Metal cannot shader-write texture2d_ms. Non-RT MS images and
                 * sampler2DMS* are backed as texture2d_array sample planes, and
                 * AIR always declares texture2d_array for MS. Match that type
                 * here so binding does not replace the real texture with a
                 * Multisample fallback (CTS shader_image_load_store WriteMS). */
                (void)image_arrayed;
                return static_cast<uint32_t>(MTL::TextureType2DArray);
            }
            return static_cast<uint32_t>(
                image_arrayed ? MTL::TextureType2DArray : MTL::TextureType2D);
        case MGL_IMAGE_DIM_3D:
            return static_cast<uint32_t>(MTL::TextureType3D);
        case MGL_IMAGE_DIM_CUBE:
            return static_cast<uint32_t>(
                image_arrayed ? MTL::TextureTypeCubeArray
                              : MTL::TextureTypeCube);
        case MGL_IMAGE_DIM_BUFFER:
            /* TEXTURE_BUFFER / samplerBuffer / imageBuffer are packed as
             * texture2d (createMTLTexelBufferTexture + AIR imageBuffer path).
             * Advertising TextureBuffer here makes the binder reject the real
             * texture2d and substitute the 64-texel fallback, so
             * imageLoad != texelFetch (CTS advanced-sync-imageAccess). */
            return static_cast<uint32_t>(MTL::TextureType2D);
        default:
            return 0u;
    }
}

extern "C"
int32_t mglRenderTextureIndexForMetalType(uint32_t texture_type) {
    switch (static_cast<MTL::TextureType>(texture_type)) {
        case MTL::TextureType1D:
            return _TEXTURE_1D;
        case MTL::TextureType1DArray:
            return _TEXTURE_1D_ARRAY;
        case MTL::TextureType2D:
            return _TEXTURE_2D;
        case MTL::TextureType2DMultisample:
            return _TEXTURE_2D_MULTISAMPLE;
        case MTL::TextureType2DArray:
            return _TEXTURE_2D_ARRAY;
        case MTL::TextureType2DMultisampleArray:
            return _TEXTURE_2D_MULTISAMPLE_ARRAY;
        case MTL::TextureType3D:
            return _TEXTURE_3D;
        case MTL::TextureTypeCube:
            return _TEXTURE_CUBE_MAP;
        case MTL::TextureTypeCubeArray:
            return _TEXTURE_CUBE_MAP_ARRAY;
        case MTL::TextureTypeTextureBuffer:
            return _TEXTURE_BUFFER;
        default:
            return -1;
    }
}

extern "C"
uint32_t mglRenderTextureDataKindForPixelFormat(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatR8Sint:
        case MTL::PixelFormatRG8Sint:
        case MTL::PixelFormatRGBA8Sint:
        case MTL::PixelFormatR16Sint:
        case MTL::PixelFormatRG16Sint:
        case MTL::PixelFormatRGBA16Sint:
        case MTL::PixelFormatR32Sint:
        case MTL::PixelFormatRG32Sint:
        case MTL::PixelFormatRGBA32Sint:
            return MGL_RENDER_TEXTURE_DATA_KIND_SINT;

        case MTL::PixelFormatR8Uint:
        case MTL::PixelFormatRG8Uint:
        case MTL::PixelFormatRGBA8Uint:
        case MTL::PixelFormatR16Uint:
        case MTL::PixelFormatRG16Uint:
        case MTL::PixelFormatRGBA16Uint:
        case MTL::PixelFormatR32Uint:
        case MTL::PixelFormatRG32Uint:
        case MTL::PixelFormatRGBA32Uint:
        case MTL::PixelFormatRGB10A2Uint:
            return MGL_RENDER_TEXTURE_DATA_KIND_UINT;

        case MTL::PixelFormatInvalid:
            return MGL_RENDER_TEXTURE_DATA_KIND_UNKNOWN;

        case MTL::PixelFormatDepth16Unorm:
        case MTL::PixelFormatDepth32Float:
        case MTL::PixelFormatDepth24Unorm_Stencil8:
        case MTL::PixelFormatDepth32Float_Stencil8:
            return MGL_RENDER_TEXTURE_DATA_KIND_DEPTH;

        default:
            return MGL_RENDER_TEXTURE_DATA_KIND_FLOAT;
    }
}

extern "C"
const char* mglRenderTextureDataKindName(uint32_t kind) {
    switch (kind) {
        case MGL_RENDER_TEXTURE_DATA_KIND_FLOAT:
            return "float";
        case MGL_RENDER_TEXTURE_DATA_KIND_SINT:
            return "sint";
        case MGL_RENDER_TEXTURE_DATA_KIND_UINT:
            return "uint";
        case MGL_RENDER_TEXTURE_DATA_KIND_DEPTH:
            return "depth";
        default:
            return "unknown";
    }
}

extern "C"
int mglRenderTextureMinFilterUsesMipmaps(uint32_t min_filter) {
    switch (min_filter) {
        case GL_NEAREST_MIPMAP_NEAREST:
        case GL_LINEAR_MIPMAP_NEAREST:
        case GL_NEAREST_MIPMAP_LINEAR:
        case GL_LINEAR_MIPMAP_LINEAR:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderMetalLayerPixelFormatIsSupported(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatBGRA8Unorm:
        case MTL::PixelFormatBGRA8Unorm_sRGB:
            return 1;
        default:
            return 0;
    }
}

extern "C"
uint32_t mglRenderLinearPixelFormat(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatRGBA8Unorm_sRGB:
            return (uint32_t)MTL::PixelFormatRGBA8Unorm;
        case MTL::PixelFormatBGRA8Unorm_sRGB:
            return (uint32_t)MTL::PixelFormatBGRA8Unorm;
        default:
            return pixel_format;
    }
}

extern "C"
uint32_t mglRenderEffectiveMTLPixelFormat(uint32_t pixel_format,
                                            uint32_t srgb_decode_ext) {
    if (srgb_decode_ext == GL_SKIP_DECODE_EXT) {
        return mglRenderLinearPixelFormat(pixel_format);
    }
    return pixel_format;
}

int mglRenderTextureUploadRoute(uint32_t texture_type,
                                   uint32_t storage_mode,
                                   int has_agx_3d_copy_bug) {

    const uint32_t kMTLTextureType1D = 0u;
    const uint32_t kMTLTextureType1DArray = 1u;
    const uint32_t kMTLTextureType3D = 7u;
    const uint32_t kMTLStorageModePrivate = 2u;


    if ((texture_type == kMTLTextureType1D ||
         texture_type == kMTLTextureType1DArray) &&
        storage_mode != kMTLStorageModePrivate) {
        return MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_1D;
    }


    if (texture_type == kMTLTextureType3D && has_agx_3d_copy_bug) {
        if (storage_mode == kMTLStorageModePrivate) {
            return MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REJECT;
        }
        return MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_3D;
    }

    return MGL_RENDER_TEXTURE_UPLOAD_ROUTE_BLIT;
}

extern "C"
void mglRenderCopyTextureBytesToBGRA8(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return;
    }

    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    bool sourceIsRGBA =
        (pf == MTL::PixelFormatRGBA8Unorm ||
         pf == MTL::PixelFormatRGBA8Unorm_sRGB);
    bool sourceIsRGBA32Float = (pf == MTL::PixelFormatRGBA32Float);
    bool sourceIsR8 = (pf == MTL::PixelFormatR8Unorm);
    bool sourceIsRG8 = (pf == MTL::PixelFormatRG8Unorm);
    bool sourceIsR16Unorm = (pf == MTL::PixelFormatR16Unorm);
    bool sourceIsRG16Unorm = (pf == MTL::PixelFormatRG16Unorm);
    bool sourceIsRGBA16Unorm = (pf == MTL::PixelFormatRGBA16Unorm);
    bool sourceIsR16Snorm = (pf == MTL::PixelFormatR16Snorm);
    bool sourceIsRG16Snorm = (pf == MTL::PixelFormatRG16Snorm);
    bool sourceIsRGBA16Snorm = (pf == MTL::PixelFormatRGBA16Snorm);
    bool sourceIsBGR5A1 = (pf == MTL::PixelFormatBGR5A1Unorm);
    bool sourceIsABGR4 = (pf == MTL::PixelFormatABGR4Unorm);
    bool sourceIsRG11B10Float = (pf == MTL::PixelFormatRG11B10Float);
    bool sourceIsR32Float = (pf == MTL::PixelFormatR32Float);
    bool sourceIsRG32Float = (pf == MTL::PixelFormatRG32Float);
    bool sourceIsRG16Float = (pf == MTL::PixelFormatRG16Float);
    bool sourceIsR16Float = (pf == MTL::PixelFormatR16Float);
    bool sourceIsRGBA16Float = (pf == MTL::PixelFormatRGBA16Float);
    bool sourceIsBGR10A2 = (pf == MTL::PixelFormatBGR10A2Unorm);
    bool sourceIsRGB10A2 = (pf == MTL::PixelFormatRGB10A2Unorm);
    bool sourceIsR8Snorm = (pf == MTL::PixelFormatR8Snorm);
    bool sourceIsRG8Snorm = (pf == MTL::PixelFormatRG8Snorm);
    bool sourceIsRGBA8Snorm = (pf == MTL::PixelFormatRGBA8Snorm);
    bool sourceIsR8Uint = (pf == MTL::PixelFormatR8Uint);
    bool sourceIsR8Sint = (pf == MTL::PixelFormatR8Sint);
    bool sourceIsRG8Uint = (pf == MTL::PixelFormatRG8Uint);
    bool sourceIsRG8Sint = (pf == MTL::PixelFormatRG8Sint);
    bool sourceIsRGBA8Uint = (pf == MTL::PixelFormatRGBA8Uint);
    bool sourceIsRGBA8Sint = (pf == MTL::PixelFormatRGBA8Sint);
    bool sourceIsRGB9E5 = (pf == MTL::PixelFormatRGB9E5Float);

    const uint8_t* srcBytes = static_cast<const uint8_t*>(src);
    uint8_t* dstBytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* srcRow = srcBytes + (y * src_bytes_per_row);
        uint64_t dstY = flip_y ? (height - 1u - y) : y;
        uint8_t* dstRow = dstBytes + (dstY * dst_bytes_per_row);

        if (!sourceIsRGBA && !sourceIsRGBA32Float && !sourceIsR8 && !sourceIsRG8 &&
            !sourceIsR16Unorm && !sourceIsRG16Unorm && !sourceIsRGBA16Unorm &&
            !sourceIsR16Snorm && !sourceIsRG16Snorm && !sourceIsRGBA16Snorm &&
            !sourceIsBGR5A1 && !sourceIsABGR4 && !sourceIsRG11B10Float &&
            !sourceIsR32Float && !sourceIsRG32Float && !sourceIsRG16Float &&
            !sourceIsR16Float && !sourceIsRGBA16Float && !sourceIsBGR10A2 &&
            !sourceIsRGB10A2 &&
            !sourceIsR8Snorm && !sourceIsRG8Snorm && !sourceIsRGBA8Snorm &&
            !sourceIsR8Uint && !sourceIsR8Sint && !sourceIsRG8Uint && !sourceIsRG8Sint &&
            !sourceIsRGBA8Uint && !sourceIsRGBA8Sint && !sourceIsRGB9E5) {
            memcpy(dstRow, srcRow, width * 4u);
            continue;
        }

        for (uint64_t x = 0; x < width; x++) {
            uint8_t* d = dstRow + (x * 4u);
            if (sourceIsRGBA32Float) {
                const float* s = reinterpret_cast<const float*>(
                    srcRow + (x * sizeof(float) * 4u));
                d[0] = mglRenderFloatToUnorm8(s[2]);
                d[1] = mglRenderFloatToUnorm8(s[1]);
                d[2] = mglRenderFloatToUnorm8(s[0]);
                d[3] = mglRenderFloatToUnorm8(s[3]);
            } else if (sourceIsRGBA16Float) {
                uint16_t components[4] = {0u, 0u, 0u, 0u};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[2]));
                d[1] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[0]));
                d[3] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[3]));
            } else if (sourceIsRG11B10Float) {
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = mglRenderFloatToUnorm8(
                    mglUnpackUnsignedFloatComponent(packed >> 22u, 5u));
                d[1] = mglRenderFloatToUnorm8(
                    mglUnpackUnsignedFloatComponent(packed >> 11u, 6u));
                d[2] = mglRenderFloatToUnorm8(
                    mglUnpackUnsignedFloatComponent(packed, 6u));
                d[3] = 255u;
            } else if (sourceIsRG32Float) {
                const float* s = reinterpret_cast<const float*>(
                    srcRow + (x * sizeof(float) * 2u));
                d[0] = 0u;
                d[1] = mglRenderFloatToUnorm8(s[1]);
                d[2] = mglRenderFloatToUnorm8(s[0]);
                d[3] = 255u;
            } else if (sourceIsR32Float) {
                float component = 0.0f;
                memcpy(&component, srcRow + x * sizeof(component),
                       sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglRenderFloatToUnorm8(component);
                d[3] = 255u;
            } else if (sourceIsRG16Float) {
                uint16_t components[2] = {0u, 0u};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = 0u;
                d[1] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglHalfToFloat(components[0]));
                d[3] = 255u;
            } else if (sourceIsR16Float) {
                uint16_t component = 0u;
                memcpy(&component, srcRow + x * sizeof(component),
                       sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglRenderFloatToUnorm8(mglHalfToFloat(component));
                d[3] = 255u;
            } else if (sourceIsRGBA16Unorm) {
                uint16_t components[4] = {0u, 0u, 0u, 0u};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = (uint8_t)((components[2] * 255u + 32767u) / 65535u);
                d[1] = (uint8_t)((components[1] * 255u + 32767u) / 65535u);
                d[2] = (uint8_t)((components[0] * 255u + 32767u) / 65535u);
                d[3] = (uint8_t)((components[3] * 255u + 32767u) / 65535u);
            } else if (sourceIsRG16Unorm) {
                uint16_t components[2] = {0u, 0u};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = 0u;
                d[1] = (uint8_t)((components[1] * 255u + 32767u) / 65535u);
                d[2] = (uint8_t)((components[0] * 255u + 32767u) / 65535u);
                d[3] = 255u;
            } else if (sourceIsR16Unorm) {
                uint16_t component = 0u;
                memcpy(&component, srcRow + x * sizeof(component),
                       sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = (uint8_t)((component * 255u + 32767u) / 65535u);
                d[3] = 255u;
            } else if (sourceIsRGBA16Snorm) {
                int16_t components[4] = {0, 0, 0, 0};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[2]));
                d[1] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[0]));
                d[3] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[3]));
            } else if (sourceIsRG16Snorm) {
                int16_t components[2] = {0, 0};
                memcpy(components, srcRow + x * sizeof(components),
                       sizeof(components));
                d[0] = 0u;
                d[1] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(components[0]));
                d[3] = 255u;
            } else if (sourceIsR16Snorm) {
                int16_t component = 0;
                memcpy(&component, srcRow + x * sizeof(component),
                       sizeof(component));
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglRenderFloatToUnorm8(
                    mglRenderSnorm16ToFloat(component));
                d[3] = 255u;
            } else if (sourceIsBGR10A2) {
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)(((packed & 1023u) * 255u) / 1023u);
                d[1] = (uint8_t)((((packed >> 10u) & 1023u) * 255u) / 1023u);
                d[2] = (uint8_t)((((packed >> 20u) & 1023u) * 255u) / 1023u);
                d[3] = (uint8_t)((((packed >> 30u) & 3u) * 255u) / 3u);
            } else if (sourceIsRGB10A2) {
                /* MTLPixelFormatRGB10A2Unorm: R[0:9], G[10:19], B[20:29],
                 * A[30:31] (LSB-first).  BGRA8: d[0]=B, d[1]=G, d[2]=R,
                 * d[3]=A. */
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)((((packed >> 20u) & 1023u) * 255u) / 1023u);
                d[1] = (uint8_t)((((packed >> 10u) & 1023u) * 255u) / 1023u);
                d[2] = (uint8_t)(((packed & 1023u) * 255u) / 1023u);
                d[3] = (uint8_t)((((packed >> 30u) & 3u) * 255u) / 3u);
            } else if (sourceIsR8) {
                d[0] = 0u;
                d[1] = 0u;
                d[2] = srcRow[x];
                d[3] = 255u;
            } else if (sourceIsRG8) {
                const uint8_t* s = srcRow + x * 2u;
                d[0] = 0u;
                d[1] = s[1];
                d[2] = s[0];
                d[3] = 255u;
            } else if (sourceIsBGR5A1) {
                /* MTLPixelFormatBGR5A1Unorm: B[0:4], G[5:9], R[10:14], A[15].
                 * Output BGRA8: d[0]=B, d[1]=G, d[2]=R, d[3]=A. */
                uint16_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)(((packed & 31u) * 255u) / 31u);
                d[1] = (uint8_t)((((packed >> 5u) & 31u) * 255u) / 31u);
                d[2] = (uint8_t)((((packed >> 10u) & 31u) * 255u) / 31u);
                d[3] = ((packed >> 15u) & 1u) ? 255u : 0u;
            } else if (sourceIsABGR4) {
                /* MTLPixelFormatABGR4Unorm: A[0:3], B[4:7], G[8:11], R[12:15].
                 * Output BGRA8: d[0]=B, d[1]=G, d[2]=R, d[3]=A. */
                uint16_t packed = 0u;
                memcpy(&packed, srcRow + x * sizeof(packed), sizeof(packed));
                d[0] = (uint8_t)((((packed >> 4u) & 15u) * 255u) / 15u);
                d[1] = (uint8_t)((((packed >> 8u) & 15u) * 255u) / 15u);
                d[2] = (uint8_t)((((packed >> 12u) & 15u) * 255u) / 15u);
                d[3] = (uint8_t)(((packed & 15u) * 255u) / 15u);
            } else if (sourceIsR8Snorm || sourceIsR8Sint) {
                int8_t s = (int8_t)srcRow[x];
                d[0] = 0u;
                d[1] = 0u;
                d[2] = mglRenderFloatToUnorm8(mglRenderSnorm8ToFloat(s));
                d[3] = 255u;
            } else if (sourceIsRG8Snorm || sourceIsRG8Sint) {
                const int8_t* s = reinterpret_cast<const int8_t*>(
                    srcRow + x * 2u);
                d[0] = 0u;
                d[1] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[0]));
                d[3] = 255u;
            } else if (sourceIsRGBA8Snorm || sourceIsRGBA8Sint) {
                const int8_t* s = reinterpret_cast<const int8_t*>(
                    srcRow + x * 4u);
                d[0] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[2]));
                d[1] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[1]));
                d[2] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[0]));
                d[3] = mglRenderFloatToUnorm8(
                    mglRenderSnorm8ToFloat(s[3]));
            } else if (sourceIsR8Uint) {
                d[0] = 0u;
                d[1] = 0u;
                d[2] = srcRow[x];
                d[3] = 255u;
            } else if (sourceIsRG8Uint) {
                const uint8_t* s = srcRow + x * 2u;
                d[0] = 0u;
                d[1] = s[1];
                d[2] = s[0];
                d[3] = 255u;
            } else if (sourceIsRGBA8Uint) {
                const uint8_t* s = srcRow + x * 4u;
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            } else if (sourceIsRGB9E5) {
                /* MTLPixelFormatRGB9E5Float: 4 bytes/pixel, shared exponent.
                 * Unpack to float R,G,B then convert to BGRA8 UNORM. */
                uint32_t packed = 0u;
                memcpy(&packed, srcRow + x * 4u, sizeof(packed));
                uint32_t exp = (packed >> 27u) & 31u;
                uint32_t mant_r = packed & 511u;
                uint32_t mant_g = (packed >> 9u) & 511u;
                uint32_t mant_b = (packed >> 18u) & 511u;
                float scale = ldexpf(1.0f, (int)exp - 24);
                float rf = (float)mant_r * scale;
                float gf = (float)mant_g * scale;
                float bf = (float)mant_b * scale;
                d[0] = mglRenderFloatToUnorm8(bf);
                d[1] = mglRenderFloatToUnorm8(gf);
                d[2] = mglRenderFloatToUnorm8(rf);
                d[3] = 255u;
            } else {
                const uint8_t* s = srcRow + (x * 4u);
                d[0] = s[2];
                d[1] = s[1];
                d[2] = s[0];
                d[3] = s[3];
            }
        }
    }
}

extern "C"
int mglRenderTextureInternalFormatNeedsRGBA8Expansion(
    uint32_t internal_format, uint32_t pixel_format) {
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    const bool is_rgba8_variant =
        (pf == MTL::PixelFormatRGBA8Unorm ||
         pf == MTL::PixelFormatRGBA8Unorm_sRGB ||
         pf == MTL::PixelFormatRGBA8Snorm ||
         pf == MTL::PixelFormatRGBA8Sint ||
         pf == MTL::PixelFormatRGBA8Uint);
    if (!is_rgba8_variant) {
        return 0;
    }
    switch (internal_format) {
        case GL_RGB4:
        case GL_RGB5:
        case GL_RGB10:
        case GL_RGB12:
        case GL_RGBA2:
        case GL_RGBA4:
        case GL_RGB5_A1:
        case GL_R3_G3_B2:
        case GL_RGB8:
        case GL_SRGB8:
        case GL_RGB8_SNORM:
        case GL_RGB8I:
        case GL_RGB8UI:
        case GL_RGB565:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderTextureNeedsChannelExpansion(uint32_t internal_format,
                                             uint32_t pixel_format) {
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    const bool is_rgba16_variant =
        (pf == MTL::PixelFormatRGBA16Unorm ||
         pf == MTL::PixelFormatRGBA16Snorm ||
         pf == MTL::PixelFormatRGBA16Float ||
         pf == MTL::PixelFormatRGBA16Sint ||
         pf == MTL::PixelFormatRGBA16Uint);
    const bool is_rgba32_variant =
        (pf == MTL::PixelFormatRGBA32Float ||
         pf == MTL::PixelFormatRGBA32Sint ||
         pf == MTL::PixelFormatRGBA32Uint);
    if (!is_rgba16_variant && !is_rgba32_variant) {
        return 0;
    }
    switch (internal_format) {
        case GL_RGB16:
        case GL_RGB16_SNORM:
        case GL_RGB16F:
        case GL_RGB16I:
        case GL_RGB16UI:
        case GL_RGB32F:
        case GL_RGB32I:
        case GL_RGB32UI:
        case GL_RGB12:
            return 1;
        default:
            return 0;
    }
}

int mglRenderIsTextureBufferTarget(uint32_t gl_target) {
    return gl_target == GL_TEXTURE_BUFFER ? 1 : 0;
}

int mglRenderTextureDimsValid(uint32_t gl_target, int32_t width, int32_t height,
                              int32_t depth) {
    if (gl_target == GL_TEXTURE_BUFFER) {
        return 1;
    }
    return width > 0 && height > 0 && width <= 32768 && height <= 32768 &&
                   depth <= 32768
               ? 1
               : 0;
}

int mglRenderTextureBufferNeedsDirty(int is_tbo, int has_buf, uint32_t buf_dirty) {
    return is_tbo && has_buf && buf_dirty != 0u ? 1 : 0;
}

int mglRenderTextureNameIsDefault(uint32_t name) {
    return name == TEX_OBJ_RES_NAME ? 1 : 0;
}

int mglRenderMSTextureUnitIndex(int image_arrayed) {
    return image_arrayed ? _TEXTURE_2D_MULTISAMPLE_ARRAY
                         : _TEXTURE_2D_MULTISAMPLE;
}

int mglRenderIsMultisampleTextureTarget(uint32_t gl_target) {
    return gl_target == GL_TEXTURE_2D_MULTISAMPLE ||
                   gl_target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY
               ? 1
               : 0;
}

int mglRenderRejectDefaultTypedTexture(int typed_is_default, int active_is_real) {
    return typed_is_default && active_is_real ? 1 : 0;
}

int mglRenderImageDimIsBuffer(uint32_t image_dim) {
    return image_dim == MGL_IMAGE_DIM_BUFFER ? 1 : 0;
}

int mglRenderExpectedTypeIsTextureBuffer(uint32_t expected_type) {
    return expected_type == MGLTextureTypeTextureBuffer ? 1 : 0;
}

int mglRenderPlanTexelBuffer2DSize(uint64_t texel_count, uint32_t max_texture_size,
                                   uint32_t *width_out, uint32_t *height_out) {
    const uint32_t kWidth = 4096u;
    uint32_t max2d = max_texture_size;
    if (max2d == 0u || max2d > kWidth) {
        max2d = kWidth;
    }
    if (texel_count == 0u) {
        return 0;
    }
    uint32_t w = texel_count < max2d ? (uint32_t)texel_count : max2d;
    uint32_t h = (uint32_t)((texel_count + w - 1u) / w);
    if (h == 0u || h > max2d) {
        return 0;
    }
    if (width_out) {
        *width_out = w;
    }
    if (height_out) {
        *height_out = h;
    }
    return 1;
}

uint32_t mglRenderFallbackSampledTextureType(uint32_t expected_type) {
    return expected_type ? expected_type : MGLTextureType2D;
}

uint32_t mglRenderFallbackSampledPixelFormat(uint32_t data_kind) {
    switch (data_kind) {
    case 3u: /* MGLTextureDataKindUint */
        return 73u; /* MGLPixelFormatRGBA8Uint */
    case 2u: /* MGLTextureDataKindSint */
        return 74u; /* MGLPixelFormatRGBA8Sint */
    case 4u: /* MGLTextureDataKindDepth */
        return 252u; /* MGLPixelFormatDepth32Float */
    default:
        return 70u; /* MGLPixelFormatRGBA8Unorm */
    }
}

int mglRenderTextureUsageForAccess(uint32_t gl_access, uint32_t *usage_out) {
    switch (gl_access) {
    case GL_READ_ONLY:
    case GL_WRITE_ONLY:
    case GL_READ_WRITE:
        if (usage_out) {
            *usage_out = MGLTextureUsageShaderRead | MGLTextureUsageShaderWrite;
        }
        return 1;
    default:
        return 0;
    }
}

int mglRenderPixelFormatNeedsShaderAtomic(uint32_t pixel_format) {
    return pixel_format == 53u /* R32Uint */ || pixel_format == 54u /* R32Sint */
               ? 1
               : 0;
}

int mglRenderTextureArrayDepthForType(uint32_t tex_type, int is_array,
                                      int ms_emulated, uint64_t width,
                                      uint64_t height, uint64_t depth,
                                      uint64_t *array_out, uint64_t *depth_out) {
    (void)width;
    uint64_t array_len = 1u;
    uint64_t d = 1u;
    if (tex_type == MGLTextureTypeCube) {
        d = 1u;
    } else if (tex_type == MGLTextureTypeCubeArray) {
        array_len = mglRenderExpectedArrayLayers(GL_TEXTURE_CUBE_MAP_ARRAY,
                                                 (int32_t)depth);
        d = 1u;
    } else if (tex_type == MGLTextureType1DArray) {
        array_len = height < 1u ? 1u : height;
        d = 1u;
    } else if (is_array && !ms_emulated) {
        array_len = depth < 1u ? 1u : depth;
        d = 1u;
    } else if (!ms_emulated) {
        array_len = 1u;
        d = depth < 1u ? 1u : depth;
    } else {
        return 0;
    }
    if (array_out) {
        *array_out = array_len;
    }
    if (depth_out) {
        *depth_out = d;
    }
    return 1;
}

uint32_t mglRenderTextureDescHeight(uint32_t tex_type, uint32_t height) {
    return tex_type == MGLTextureType1D || tex_type == MGLTextureType1DArray
               ? 1u
               : height;
}

int mglRenderTextureTargetIsBuffer(uint32_t target) {
    return target == GL_TEXTURE_BUFFER ? 1 : 0;
}

int mglRenderTextureTargetIsArrayOr3D(uint32_t target) {
    return target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_3D ? 1 : 0;
}

int mglRenderTextureTargetIs2D(uint32_t target) {
    return target == GL_TEXTURE_2D ? 1 : 0;
}

int mglRenderTextureTargetIs3D(uint32_t target) {
    return target == GL_TEXTURE_3D ? 1 : 0;
}

int mglRenderImageAccessIsReadOnly(uint32_t access) {
    return access == GL_READ_ONLY ? 1 : 0;
}

int mglRenderImageUnitSliceNeedsFlush(int has_tex, int has_view, int layered,
                                      uint32_t target, uint32_t access) {
    return has_tex && has_view && !layered &&
                   mglRenderTextureTargetIs3D(target) &&
                   !mglRenderImageAccessIsReadOnly(access)
               ? 1
               : 0;
}

uint32_t mglRenderFallbackPixelFormat(uint32_t mapped, uint32_t internalformat) {
    if (mapped != 0u) {
        return mapped;
    }
    if (internalformat == GL_DEPTH24_STENCIL8 ||
        internalformat == GL_DEPTH32F_STENCIL8) {
        return 260u; /* Depth32Float_Stencil8 */
    }
    if (internalformat == GL_DEPTH_COMPONENT ||
        internalformat == GL_DEPTH_COMPONENT16 ||
        internalformat == GL_DEPTH_COMPONENT24 ||
        internalformat == GL_DEPTH_COMPONENT32 ||
        internalformat == GL_DEPTH_COMPONENT32F) {
        return 252u; /* Depth32Float */
    }
    return 70u; /* RGBA8Unorm */
}

int mglRenderTextureTargetIsArray(uint32_t target) {
    return target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_CUBE_MAP_ARRAY
               ? 1
               : 0;
}

uint32_t mglRenderBytesPerPixelForInternalFormat(uint32_t internalformat,
                                                 int *known) {
    uint32_t bpp = 4u;
    int ok = 1;
    switch (internalformat) {
    case GL_RED:
    case GL_R8:
    case GL_R8I:
    case GL_R8UI:
        bpp = 1u;
        break;
    case GL_RG:
    case GL_RG8:
    case GL_RG8I:
    case GL_RG8UI:
    case GL_R16:
    case GL_R16F:
    case GL_R16I:
    case GL_R16UI:
        bpp = 2u;
        break;
    case GL_RGB:
    case GL_RGB8:
    case GL_RGB8I:
    case GL_RGB8UI:
    case GL_SRGB8:
    case GL_R11F_G11F_B10F:
    case GL_RGB9_E5:
        bpp = 3u;
        break;
    case GL_RGBA:
    case GL_RGBA8:
    case GL_RGBA8I:
    case GL_RGBA8UI:
    case GL_RGB10_A2:
    case GL_RGB10_A2UI:
    case GL_SRGB8_ALPHA8:
    case GL_RG16I:
    case GL_RG16UI:
    case GL_R32I:
    case GL_R32UI:
    case GL_R32F:
        bpp = 4u;
        break;
    case GL_RGBA16:
    case GL_RGBA16F:
    case GL_RG32I:
    case GL_RG32UI:
    case GL_RG32F:
    case GL_RGBA16I:
    case GL_RGBA16UI:
        bpp = 8u;
        break;
    case GL_RGB16:
    case GL_RGB16F:
        bpp = 6u;
        break;
    case GL_RGB32F:
    case GL_RGB32I:
    case GL_RGB32UI:
        bpp = 12u;
        break;
    case GL_RGBA32F:
    case GL_RGBA32I:
    case GL_RGBA32UI:
        bpp = 16u;
        break;
    default:
        ok = 0;
        bpp = 4u;
        break;
    }
    if (known) {
        *known = ok;
    }
    return bpp;
}

uint32_t mglRenderMetalPixelFormatBytesPerPixel(uint32_t pixel_format) {
    switch (pixel_format) {
    case 10u: /* MGLPixelFormatR8Unorm */
    case 13u: /* MGLPixelFormatR8Uint */
    case 14u: /* MGLPixelFormatR8Sint */
        return 1u;
    case 30u: /* MGLPixelFormatRG8Unorm */
    case 33u: /* MGLPixelFormatRG8Uint */
    case 34u: /* MGLPixelFormatRG8Sint */
        return 2u;
    default:
        return 4u;
    }
}

uint32_t mglRenderMetalPixelFormatValueClass(uint32_t pixel_format) {
    switch (pixel_format) {
    case 14u:  /* MGLPixelFormatR8Sint */
    case 24u:  /* MGLPixelFormatR16Sint */
    case 34u:  /* MGLPixelFormatRG8Sint */
    case 54u:  /* MGLPixelFormatR32Sint */
    case 64u:  /* MGLPixelFormatRG16Sint */
    case 74u:  /* MGLPixelFormatRGBA8Sint */
    case 104u: /* MGLPixelFormatRG32Sint */
    case 114u: /* MGLPixelFormatRGBA16Sint */
    case 124u: /* MGLPixelFormatRGBA32Sint */
        return 1u; /* int */
    case 13u:  /* MGLPixelFormatR8Uint */
    case 23u:  /* MGLPixelFormatR16Uint */
    case 33u:  /* MGLPixelFormatRG8Uint */
    case 53u:  /* MGLPixelFormatR32Uint */
    case 63u:  /* MGLPixelFormatRG16Uint */
    case 73u:  /* MGLPixelFormatRGBA8Uint */
    case 91u:  /* MGLPixelFormatRGB10A2Uint */
    case 103u: /* MGLPixelFormatRG32Uint */
    case 113u: /* MGLPixelFormatRGBA16Uint */
    case 123u: /* MGLPixelFormatRGBA32Uint */
        return 2u; /* uint */
    default:
        return 0u; /* float */
    }
}

int mglRenderPixelFormatIsDepthOrStencil(uint32_t pixel_format) {
    switch (pixel_format) {
    case 250u: /* Depth16Unorm */
    case 252u: /* Depth32Float */
    case 253u: /* Stencil8 */
    case 255u: /* Depth24Unorm_Stencil8 */
    case 260u: /* Depth32Float_Stencil8 */
        return 1;
    default:
        return 0;
    }
}

int mglRenderSamplerUnitExplicit(uint32_t flag) {
    return flag == GL_TRUE ? 1 : 0;
}

int mglRenderPrefer1DSampler(uint32_t image_dim, int arrayed) {
    return image_dim == MGL_IMAGE_DIM_1D && !arrayed ? 1 : 0;
}

int mglRenderTextureTargetIs1D(uint32_t target) {
    return target == GL_TEXTURE_1D ? 1 : 0;
}

int mglRenderTextureTargetIsMSOr2DArray(uint32_t target) {
    return mglRenderIsMultisampleTextureTarget(target) ||
                   target == GL_TEXTURE_2D_ARRAY
               ? 1
               : 0;
}

void mglRenderMarkTextureLevelWritten(uint8_t *ever_written,
                                      uint8_t *has_initialized,
                                      uint8_t *suspicious_zero) {
    if (ever_written) {
        *ever_written = (uint8_t)GL_TRUE;
    }
    if (has_initialized) {
        *has_initialized = (uint8_t)GL_TRUE;
    }
    if (suspicious_zero) {
        *suspicious_zero = (uint8_t)GL_FALSE;
    }
}

int mglRenderImageAccessWritable(uint32_t access) {
    return access == GL_WRITE_ONLY || access == GL_READ_WRITE ? 1 : 0;
}

uint32_t mglRenderSamplerObjectTarget(void) {
    return GL_TEXTURE_2D;
}

int mglRenderPixelFormatIsUnorm8Color(uint32_t pixel_format) {
    return mglRenderReadbackPixelFormatIsRGBA8(pixel_format) ||
           mglRenderReadbackPixelFormatIsBGRA8(pixel_format);
}

int mglRenderTextureTargetIsCubeMap(uint32_t target) {
    return target == GL_TEXTURE_CUBE_MAP ? 1 : 0;
}

int mglRenderTextureNeedsArrayLengthCheck(uint32_t target) {
    return mglRenderTextureTargetIsArray(target) ||
                   target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY
               ? 1
               : 0;
}

int mglRenderTextureTargetIsLayeredUpload(uint32_t target) {
    return mglRenderAttachmentUsesArrayLayer(target) ||
                   mglRenderTextureTargetIsCubeMap(target) ||
                   mglRenderTextureTargetIs3D(target)
               ? 1
               : 0;
}

int mglRenderTextureTargetIs1DArray(uint32_t target) {
    return target == GL_TEXTURE_1D_ARRAY ? 1 : 0;
}

extern "C"
int mglRenderReadTextureRegionClip(
    int64_t region_x, int64_t region_y,
    int64_t region_w, int64_t region_h,
    int64_t level_width, int64_t level_h,
    MGLRenderReadTextureRegionClip* out) {
    if (!out) return -1;
    const int64_t max_x = region_x + region_w;
    const int64_t max_y = region_y + region_h;
    const int64_t min_x = region_x > 0 ? region_x : 0;
    const int64_t min_y = region_y > 0 ? region_y : 0;
    const int64_t clip_x = max_x < level_width ? max_x : level_width;
    const int64_t clip_y = max_y < level_h ? max_y : level_h;
    const int64_t copy_w = clip_x - min_x;
    const int64_t copy_h = clip_y - min_y;
    out->copy_w = copy_w;
    out->copy_h = copy_h;
    out->dst_x = min_x - region_x;
    out->dst_y = min_y - region_y;
    out->metal_src_x = min_x;
    out->metal_src_y = level_h - clip_y;
    out->empty = (copy_w <= 0 || copy_h <= 0) ? 1 : 0;
    return 0;
}

extern "C"
uint64_t mglRenderMetalTextureLevelDimension(uint64_t base, uint64_t level) {
    const uint64_t one = 1u;
    uint64_t value = base > one ? base : one;
    while (level-- > 0u && value > one) {
        value >>= 1u;
    }
    return value > one ? value : one;
}


int mglRenderCreateDefaultSampler(void** sampler_out) {
    if (sampler_out) *sampler_out = nullptr;
    if (!sampler_out) return -1;
    MTL::SamplerDescriptor* descriptor = MTL::SamplerDescriptor::alloc()->init();
    if (!descriptor) return -1;
    int result = mglRenderCreateSampler(descriptor, sampler_out);
    descriptor->release();
    return result;
}

int mglRenderCreateFilterSampler(uint32_t nearest, void** sampler_out) {
    if (sampler_out) *sampler_out = nullptr;
    if (!sampler_out) return -1;
    MTL::SamplerDescriptor* descriptor = MTL::SamplerDescriptor::alloc()->init();
    if (!descriptor) return -1;
    const MTL::SamplerMinMagFilter filter =
        nearest ? MTL::SamplerMinMagFilterNearest
                : MTL::SamplerMinMagFilterLinear;
    descriptor->setMinFilter(filter);
    descriptor->setMagFilter(filter);
    descriptor->setMipFilter(MTL::SamplerMipFilterNotMipmapped);
    descriptor->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
    descriptor->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
    descriptor->setRAddressMode(MTL::SamplerAddressModeClampToEdge);
    int result = mglRenderCreateSampler(descriptor, sampler_out);
    descriptor->release();
    return result;
}

int mglRenderSetComputeTexture(void* compute_encoder,
                                  void* texture,
                                  uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setTexture(static_cast<MTL::Texture*>(texture), index);
    return 0;
}

int mglRenderSetComputeSampler(void* compute_encoder,
                                  void* sampler,
                                  uint32_t index) {
    MTL::ComputeCommandEncoder* encoder =
        static_cast<MTL::ComputeCommandEncoder*>(compute_encoder);
    if (!encoder) return -1;
    encoder->setSamplerState(static_cast<MTL::SamplerState*>(sampler), index);
    return 0;
}

int mglRenderTextureSampleParams(uint32_t target, int32_t samples,
                                 uint32_t *num_samples,
                                 uint32_t *sample_buffers) {
    if (!num_samples || !sample_buffers) {
        return -1;
    }
    *num_samples = 1u;
    *sample_buffers = 0u;
    if (target == GL_TEXTURE_2D_MULTISAMPLE ||
        target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY ||
        target == GL_RENDERBUFFER) {
        if (samples > 0 || target == GL_TEXTURE_2D_MULTISAMPLE ||
            target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) {
            *sample_buffers = 1u;
            *num_samples = samples > 0 ? (uint32_t)samples : 1u;
        }
    }
    return 0;
}

int mglRenderBlitSynchronizeTexture(void* blit_encoder,
                                       void* texture,
                                       uint64_t slice,
                                       uint64_t level) {
    MTL::BlitCommandEncoder* encoder =
        static_cast<MTL::BlitCommandEncoder*>(blit_encoder);
    MTL::Texture* source = static_cast<MTL::Texture*>(texture);
    if (!encoder || !source) return -1;
    encoder->synchronizeTexture(source, static_cast<NS::UInteger>(slice),
                                static_cast<NS::UInteger>(level));
    return 0;
}

int mglRenderSetRenderTexture(void* render_encoder,
                                 void* texture,
                                 uint32_t stage,
                                 uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT) return -1;
    MTL::Texture* resource = static_cast<MTL::Texture*>(texture);
    if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
        encoder->setVertexTexture(resource, index);
    } else {
        encoder->setFragmentTexture(resource, index);
    }
    return 0;
}

int mglRenderSetRenderSampler(void* render_encoder,
                                 void* sampler,
                                 uint32_t stage,
                                 uint32_t index) {
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!encoder || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT) return -1;
    MTL::SamplerState* resource = static_cast<MTL::SamplerState*>(sampler);
    if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
        encoder->setVertexSamplerState(resource, index);
    } else {
        encoder->setFragmentSamplerState(resource, index);
    }
    return 0;
}

int mglRenderCreateTextureStagingOwner(const void* bytes, uint64_t length, uint64_t resource_options, MGLTextureStagingOwner ** owner_out, void** buffer_out) {
    if (owner_out) *owner_out = nullptr;
    if (buffer_out) *buffer_out = nullptr;
    if (!owner_out || !buffer_out || !bytes || length == 0) return -1;
    void* rawBuffer = nullptr;
    if (mglRenderCreateBufferWithBytes(
            bytes, length, resource_options, "MGL.texture_staging",
            &rawBuffer) != 0 || !rawBuffer) {
        return -1;
    }
    mgl::TextureStagingOwner* owner =
        new (std::nothrow) mgl::TextureStagingOwner();
    if (!owner) {
        static_cast<MTL::Buffer*>(rawBuffer)->release();
        return -1;
    }
    owner->buffer = static_cast<MTL::Buffer*>(rawBuffer);
    *owner_out = reinterpret_cast<decltype(*owner_out)>(owner);
    *buffer_out = owner->buffer;
    return 0;
}

void mglRenderDestroyTextureStagingOwner(MGLTextureStagingOwner ** owner_handle) {
    if (!owner_handle || !*owner_handle) return;
    mgl::TextureStagingOwner* owner =
        reinterpret_cast<mgl::TextureStagingOwner*>(*owner_handle);
    *owner_handle = nullptr;
    delete owner;
}

int mglRenderCreateTextureFromState(
    const MGLRenderTextureDescriptorState* texture_descriptor,
    const char* label,
    void** texture_out) {
    if (texture_out) *texture_out = nullptr;
    if (!texture_descriptor || !texture_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::TextureDescriptor* descriptor =
        mgl::newTextureDescriptor(texture_descriptor);
    if (!descriptor) return -1;
    MTL::Texture* texture = renderer.device->newTexture(descriptor);
    descriptor->release();
    if (!texture) return -1;
    if (label && label[0]) {
        texture->setLabel(
            NS::String::string(label, NS::UTF8StringEncoding));
    }
    *texture_out = texture;
    return 0;
}

int mglRenderCreateBufferTextureFromState(
    void* buffer,
    const MGLRenderTextureDescriptorState* texture_descriptor,
    uint64_t offset,
    uint64_t bytes_per_row,
    void** texture_out) {
    if (texture_out) *texture_out = nullptr;
    MTL::Buffer* source = static_cast<MTL::Buffer*>(buffer);
    if (!source || !texture_descriptor || !texture_out ||
        bytes_per_row == 0) {
        return -1;
    }
    MTL::TextureDescriptor* descriptor =
        mgl::newTextureDescriptor(texture_descriptor);
    if (!descriptor) return -1;
    MTL::Texture* texture = source->newTexture(
        descriptor, static_cast<NS::UInteger>(offset),
        static_cast<NS::UInteger>(bytes_per_row));
    descriptor->release();
    if (!texture) return -1;
    *texture_out = texture;
    return 0;
}

int mglRenderCreateSampler(void* sampler_descriptor,
                              void** sampler_out) {
    if (sampler_out) *sampler_out = nullptr;
    MTL::SamplerDescriptor* descriptor =
        static_cast<MTL::SamplerDescriptor*>(sampler_descriptor);
    if (!descriptor || !sampler_out) return -1;
    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) return -1;
    MTL::SamplerState* sampler = renderer.device->newSamplerState(descriptor);
    if (!sampler) return -1;
    *sampler_out = sampler;
    return 0;
}

int mglRenderCreateSamplerForGL(const TextureParameter* params,
                                   uint32_t target,
                                   void** sampler_out,
                                   char* err,
                                   size_t errcap) {
    if (sampler_out) *sampler_out = nullptr;
    if (err && errcap) err[0] = '\0';
    if (!params || !sampler_out) {
        if (err && errcap) snprintf(err, errcap, "invalid sampler parameters");
        return -1;
    }

    MTL::SamplerMinMagFilter minFilter;
    MTL::SamplerMipFilter mipFilter = MTL::SamplerMipFilterNotMipmapped;
    switch (params->min_filter) {
        case GL_NEAREST:
            minFilter = MTL::SamplerMinMagFilterNearest;
            break;
        case GL_LINEAR:
            minFilter = MTL::SamplerMinMagFilterLinear;
            break;
        case GL_NEAREST_MIPMAP_NEAREST:
            minFilter = MTL::SamplerMinMagFilterNearest;
            mipFilter = MTL::SamplerMipFilterNearest;
            break;
        case GL_LINEAR_MIPMAP_NEAREST:
            minFilter = MTL::SamplerMinMagFilterLinear;
            mipFilter = MTL::SamplerMipFilterNearest;
            break;
        case GL_NEAREST_MIPMAP_LINEAR:
            minFilter = MTL::SamplerMinMagFilterNearest;
            mipFilter = MTL::SamplerMipFilterLinear;
            break;
        case GL_LINEAR_MIPMAP_LINEAR:
            minFilter = MTL::SamplerMinMagFilterLinear;
            mipFilter = MTL::SamplerMipFilterLinear;
            break;
        default:
            if (err && errcap) snprintf(err, errcap,
                                        "invalid GL min filter=0x%x",
                                        params->min_filter);
            return -1;
    }

    MTL::SamplerMinMagFilter magFilter;
    switch (params->mag_filter) {
        case GL_NEAREST:
            magFilter = MTL::SamplerMinMagFilterNearest;
            break;
        case GL_LINEAR:
            magFilter = MTL::SamplerMinMagFilterLinear;
            break;
        default:
            if (err && errcap) snprintf(err, errcap,
                                        "invalid GL mag filter=0x%x",
                                        params->mag_filter);
            return -1;
    }

    auto addressModeForGL = [&](GLenum value,
                                MTL::SamplerAddressMode* out) -> bool {
        if (!out) return false;
        switch (value) {
            case GL_CLAMP_TO_EDGE:
                *out = MTL::SamplerAddressModeClampToEdge;
                return true;
            case GL_CLAMP_TO_BORDER:
                *out = MTL::SamplerAddressModeClampToBorderColor;
                return true;
            case GL_MIRRORED_REPEAT:
                *out = MTL::SamplerAddressModeMirrorRepeat;
                return true;
            case GL_REPEAT:
                *out = MTL::SamplerAddressModeRepeat;
                return true;
            case GL_MIRROR_CLAMP_TO_EDGE:
                *out = MTL::SamplerAddressModeMirrorClampToEdge;
                return true;
            default:
                return false;
        }
    };

    MTL::SamplerAddressMode sAddress;
    MTL::SamplerAddressMode tAddress;
    MTL::SamplerAddressMode rAddress;
    if (!addressModeForGL(params->wrap_s, &sAddress) ||
        !addressModeForGL(params->wrap_t, &tAddress) ||
        !addressModeForGL(params->wrap_r, &rAddress)) {
        if (err && errcap) snprintf(err, errcap,
                                    "invalid GL sampler address mode");
        return -1;
    }

    MTL::SamplerBorderColor borderColor =
        MTL::SamplerBorderColorTransparentBlack;
    const bool hasBorder = params->wrap_s == GL_CLAMP_TO_BORDER ||
                           params->wrap_t == GL_CLAMP_TO_BORDER ||
                           params->wrap_r == GL_CLAMP_TO_BORDER;
    if (hasBorder) {
        const float* color = params->border_color;
        if (color[0] == 0.0f && color[1] == 0.0f &&
            color[2] == 0.0f && color[3] == 1.0f) {
            borderColor = MTL::SamplerBorderColorOpaqueBlack;
        } else if (color[0] == 1.0f && color[1] == 1.0f &&
                   color[2] == 1.0f && color[3] == 1.0f) {
            borderColor = MTL::SamplerBorderColorOpaqueWhite;
        } else if (!(color[0] == 0.0f && color[1] == 0.0f &&
                     color[2] == 0.0f && color[3] == 0.0f)) {
            /* Metal exposes only three named border colors. Match the ObjC
             * fallback for arbitrary GL colors. */
            borderColor = color[3] < 0.5f
                ? MTL::SamplerBorderColorTransparentBlack
                : (color[0] >= 0.5f && color[1] >= 0.5f && color[2] >= 0.5f
                       ? MTL::SamplerBorderColorOpaqueWhite
                       : MTL::SamplerBorderColorOpaqueBlack);
        }
    }

    MTL::CompareFunction compare = MTL::CompareFunctionNever;
    if (params->compare_mode == GL_COMPARE_REF_TO_TEXTURE) {
        switch (params->compare_func) {
            case GL_NEVER: compare = MTL::CompareFunctionNever; break;
            case GL_LESS: compare = MTL::CompareFunctionLess; break;
            case GL_EQUAL: compare = MTL::CompareFunctionEqual; break;
            case GL_LEQUAL: compare = MTL::CompareFunctionLessEqual; break;
            case GL_GREATER: compare = MTL::CompareFunctionGreater; break;
            case GL_NOTEQUAL: compare = MTL::CompareFunctionNotEqual; break;
            case GL_GEQUAL: compare = MTL::CompareFunctionGreaterEqual; break;
            case GL_ALWAYS: compare = MTL::CompareFunctionAlways; break;
            default:
                if (err && errcap) snprintf(err, errcap,
                                            "invalid GL compare function");
                return -1;
        }
    } else if (params->compare_mode != GL_NONE) {
        if (err && errcap) snprintf(err, errcap,
                                    "invalid GL compare mode=0x%x",
                                    params->compare_mode);
        return -1;
    }

    mgl::Renderer& renderer = mgl::renderer();
    std::lock_guard<std::mutex> lock(renderer.mutex);
    if (!renderer.device) {
        if (err && errcap) snprintf(err, errcap,
                                    "renderer is not initialized");
        return -1;
    }
    MTL::SamplerDescriptor* descriptor =
        MTL::SamplerDescriptor::alloc()->init();
    if (!descriptor) {
        if (err && errcap) snprintf(err, errcap,
                                    "sampler descriptor allocation failed");
        return -1;
    }
    descriptor->setMinFilter(minFilter);
    descriptor->setMagFilter(magFilter);
    descriptor->setMipFilter(mipFilter);
    descriptor->setSAddressMode(sAddress);
    descriptor->setTAddressMode(tAddress);
    descriptor->setRAddressMode(rAddress);
    descriptor->setBorderColor(borderColor);
    descriptor->setCompareFunction(compare);
    descriptor->setMaxAnisotropy(
        params->max_anisotropy > 1.0f
            ? std::min<NS::UInteger>(16u,
                                     std::max<NS::UInteger>(
                                         1u, static_cast<NS::UInteger>(
                                                 params->max_anisotropy)))
            : 1u);
    descriptor->setLodMinClamp(params->min_lod < 0.0f ? 0.0f : params->min_lod);
    descriptor->setLodMaxClamp(params->max_lod >= 1000.0f
                                   ? 1e9f : params->max_lod);
    if (target == GL_TEXTURE_RECTANGLE) {
        descriptor->setNormalizedCoordinates(false);
        if (params->wrap_s != GL_CLAMP_TO_EDGE ||
            params->wrap_t != GL_CLAMP_TO_EDGE ||
            params->wrap_r != GL_CLAMP_TO_EDGE) {
            descriptor->setSAddressMode(MTL::SamplerAddressModeClampToEdge);
            descriptor->setTAddressMode(MTL::SamplerAddressModeClampToEdge);
            descriptor->setRAddressMode(MTL::SamplerAddressModeClampToEdge);
        }
    }

    MTL::SamplerState* sampler = renderer.device->newSamplerState(descriptor);
    descriptor->release();
    if (!sampler) {
        if (err && errcap) snprintf(err, errcap,
                                    "Metal sampler creation failed");
        return -1;
    }
    *sampler_out = sampler;
    return 0;
}

int mglRenderBindingGetTextureSlotMask(MGLBindingState * binding_state, uint64_t mask_out[2]) {
    if (mask_out) {
        mask_out[0] = 0;
        mask_out[1] = 0;
    }
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || !mask_out) return -1;
    mask_out[0] = state->textureSlotMask[0];
    mask_out[1] = state->textureSlotMask[1];
    return 0;
}

int mglRenderBindingSetTexture(MGLBindingState * binding_state, void* render_encoder, void* texture, uint32_t stage, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexTextures.size()) {
        return -1;
    }
    if (index < 64u) {
        state->textureSlotMask[0] |= 1ull << index;
    } else {
        state->textureSlotMask[1] |= 1ull << (index - 64u);
    }
    std::vector<MTL::Texture*>& slots =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexTextures : state->fragmentTextures;
    MTL::Texture* newTexture = static_cast<MTL::Texture*>(texture);
    const bool emitted = !state->valid || slots[index] != newTexture;
    const uint32_t setter = stage == MGL_RENDER_BINDING_STAGE_VERTEX
        ? MGL_RENDER_BINDING_VERTEX_TEXTURE
        : MGL_RENDER_BINDING_FRAGMENT_TEXTURE;
    if (emitted) {
        if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
            encoder->setVertexTexture(newTexture, index);
        } else {
            encoder->setFragmentTexture(newTexture, index);
        }
        mgl::BindingState::replaceObject(slots[index], newTexture);
    }
    mgl::recordBindingResult(*state, setter, emitted);
    return emitted ? 1 : 0;
}

int mglRenderBindingSetSampler(MGLBindingState * binding_state, void* render_encoder, void* sampler, uint32_t stage, uint32_t index) {
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    MTL::RenderCommandEncoder* encoder =
        static_cast<MTL::RenderCommandEncoder*>(render_encoder);
    if (!state || !encoder || stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexSamplers.size()) {
        return -1;
    }
    if (index < 64u) {
        state->textureSlotMask[0] |= 1ull << index;
    } else {
        state->textureSlotMask[1] |= 1ull << (index - 64u);
    }
    std::vector<MTL::SamplerState*>& slots =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexSamplers : state->fragmentSamplers;
    MTL::SamplerState* newSampler = static_cast<MTL::SamplerState*>(sampler);
    const bool emitted = !state->valid || slots[index] != newSampler;
    const uint32_t setter = stage == MGL_RENDER_BINDING_STAGE_VERTEX
        ? MGL_RENDER_BINDING_VERTEX_SAMPLER
        : MGL_RENDER_BINDING_FRAGMENT_SAMPLER;
    if (emitted) {
        if (stage == MGL_RENDER_BINDING_STAGE_VERTEX) {
            encoder->setVertexSamplerState(newSampler, index);
        } else {
            encoder->setFragmentSamplerState(newSampler, index);
        }
        mgl::BindingState::replaceObject(slots[index], newSampler);
    }
    mgl::recordBindingResult(*state, setter, emitted);
    return emitted ? 1 : 0;
}

int mglRenderBindingGetTexture(MGLBindingState * binding_state, uint32_t stage, uint32_t index, void** texture_out) {
    if (texture_out) *texture_out = nullptr;
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || !texture_out ||
        stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexTextures.size()) {
        return -1;
    }
    const std::vector<MTL::Texture*>& slots =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexTextures : state->fragmentTextures;
    *texture_out = slots[index];
    return 0;
}

int mglRenderBindingGetSampler(MGLBindingState * binding_state, uint32_t stage, uint32_t index, void** sampler_out) {
    if (sampler_out) *sampler_out = nullptr;
    mgl::BindingState* state = reinterpret_cast<mgl::BindingState*>(static_cast<void*>(binding_state));
    if (!state || !sampler_out ||
        stage > MGL_RENDER_BINDING_STAGE_FRAGMENT ||
        index >= state->vertexSamplers.size()) {
        return -1;
    }
    const std::vector<MTL::SamplerState*>& slots =
        stage == MGL_RENDER_BINDING_STAGE_VERTEX
            ? state->vertexSamplers : state->fragmentSamplers;
    *sampler_out = slots[index];
    return 0;
}

int mglRenderPassUsesColorTextureOwner(MGLRenderPassStateOwner * owner_handle, void* texture, uint32_t* attachment_index_out) {
    auto* owner = reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    if (!owner || !texture) return 0;
    for (uint32_t index = 0; index < MGL_RENDER_MAX_COLOR_ATTACHMENTS; ++index) {
        if (owner->state.color[index].attachment.texture == texture) {
            if (attachment_index_out) *attachment_index_out = index;
            return 1;
        }
    }
    return 0;
}

int mglRenderCopyMatchingTextureSubresourcesForCommandBufferOwner(MGLCommandBufferOwner * command_buffer_owner, void* source_texture, void* destination_texture) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    MTL::Texture* source = static_cast<MTL::Texture*>(source_texture);
    MTL::Texture* destination =
        static_cast<MTL::Texture*>(destination_texture);
    if (!owner || !owner->current || !source || !destination ||
        source->width() != destination->width() ||
        source->height() != destination->height() ||
        source->depth() != destination->depth()) {
        return -1;
    }

    const NS::UInteger slice_count =
        std::min(source->arrayLength(), destination->arrayLength());
    const NS::UInteger level_count = std::min(
        source->mipmapLevelCount(), destination->mipmapLevelCount());
    if (slice_count == 0 || level_count == 0) return -1;

    MTL::BlitCommandEncoder* encoder =
        owner->current->blitCommandEncoder();
    if (!encoder) return -1;
    for (NS::UInteger slice = 0; slice < slice_count; ++slice) {
        for (NS::UInteger level = 0; level < level_count; ++level) {
            const NS::UInteger width =
                std::max<NS::UInteger>(1u, source->width() >> level);
            const NS::UInteger height =
                std::max<NS::UInteger>(1u, source->height() >> level);
            const NS::UInteger depth =
                std::max<NS::UInteger>(1u, source->depth() >> level);
            encoder->copyFromTexture(
                source, slice, level, MTL::Origin(0, 0, 0),
                MTL::Size(width, height, depth), destination, slice, level,
                MTL::Origin(0, 0, 0));
        }
    }
    encoder->endEncoding();
    return 0;
}



extern "C"
int mglRenderMetalPixelFormatIsDepthOrStencil(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatDepth16Unorm:
        case MTL::PixelFormatDepth32Float:
        case MTL::PixelFormatDepth24Unorm_Stencil8:
        case MTL::PixelFormatDepth32Float_Stencil8:
        case MTL::PixelFormatStencil8:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderMetalPixelFormatIsPackedDepthStencil(uint32_t pixel_format) {
    switch (static_cast<MTL::PixelFormat>(pixel_format)) {
        case MTL::PixelFormatDepth24Unorm_Stencil8:
        case MTL::PixelFormatDepth32Float_Stencil8:
            return 1;
        default:
            return 0;
    }
}

extern "C"
int mglRenderTexturePixelFormatCompatibleWithExpectedDataKind(
    uint32_t pixel_format, uint32_t expected_kind) {
    if (expected_kind == MGL_RENDER_TEXTURE_DATA_KIND_UNKNOWN) {
        return 1;
    }
    return mglRenderTextureDataKindForPixelFormat(pixel_format) ==
           expected_kind;
}

extern "C"
int mglRenderTextureUploadNeedsDepthNormalization(uint32_t internal_format,
                                                  uint32_t pixel_format) {
    return (internal_format == GL_DEPTH24_STENCIL8 && pixel_format == 260u) ? 1 : 0;
}

extern "C"
uint8_t *mglRenderCreateDepth24Stencil8NormalizedUpload(
    const void *src_data, size_t width, size_t height,
    size_t src_bytes_per_row, size_t *out_bytes_per_row,
    size_t *out_bytes_per_image) {
    if (out_bytes_per_row) *out_bytes_per_row = 0u;
    if (out_bytes_per_image) *out_bytes_per_image = 0u;
    if (!src_data || width == 0u || height == 0u ||
        src_bytes_per_row < width * 4u || !out_bytes_per_row ||
        !out_bytes_per_image) return nullptr;
    const size_t dst_row = width * 4u;   /* EXPERIMENT: 4-byte texel */
    const size_t dst_image = dst_row * height;
    uint8_t *dst = (uint8_t *)calloc(1u, dst_image);
    if (!dst) return nullptr;
    const uint8_t *src = (const uint8_t *)src_data;
    for (size_t y = 0u; y < height; ++y) {
        const uint8_t *srcRow = src + y * src_bytes_per_row;
        uint8_t *dstRow = dst + y * dst_row;
        for (size_t x = 0u; x < width; ++x) {
            const uint32_t bits = mglRenderDepth24Stencil8ToFloatBits(srcRow + x * 4u);
            memcpy(dstRow + x * 4u, &bits, 4u);
        }
    }
    *out_bytes_per_row = dst_row;
    *out_bytes_per_image = dst_image;
    return dst;
}


int mglRenderEncodeTextureUploadLayersForCommandBufferOwner(MGLCommandBufferOwner * command_buffer_owner, void* source_buffer, uint64_t source_offset, uint64_t source_bytes_per_row, uint64_t source_bytes_per_image, uint64_t source_layer_stride, uint64_t source_width, uint64_t source_height, uint64_t source_depth, void* destination_texture, uint64_t destination_base_slice, uint64_t layer_count, uint64_t destination_level, uint64_t destination_x, uint64_t destination_y, uint64_t destination_z) {
    mgl::CommandBufferOwner* owner =
        reinterpret_cast<mgl::CommandBufferOwner*>(static_cast<void*>(command_buffer_owner));
    if (!owner || !owner->current) return -1;
    return mglRenderEncodeTextureUploadLayers(
        owner->current, source_buffer, source_offset, source_bytes_per_row,
        source_bytes_per_image, source_layer_stride, source_width,
        source_height, source_depth, destination_texture,
        destination_base_slice, layer_count, destination_level,
        destination_x, destination_y, destination_z);
}

extern "C"
int mglRenderTexturePrepareLevelUpload(
    const TextureLevel* level, uint32_t texture_type,
    uint32_t internal_format, uint32_t pixel_format,
    MGLRenderLevelUploadPrep* out) {
    if (out) {
        memset(out, 0, sizeof(*out));
    }
    if (!level || !out) {
        return -1;
    }
    const uint64_t width = level->width;
    const uint64_t height = level->height ? level->height : 1;
    const uint64_t depth = level->depth ? level->depth : 1;
    const uint64_t bytes_per_row = level->pitch;
    const void* src_data = (const void*)(uintptr_t)level->data;
    if (!src_data || width == 0 || height == 0 || bytes_per_row == 0) {
        return -1;
    }
    const uint64_t copy_depth =
        ((MTL::TextureType)texture_type == MTL::TextureType3D) ? depth : 1;
    const uint64_t available_bytes = level->data_size;
    const uint64_t bytes_per_image =
        MIN(available_bytes / copy_depth, bytes_per_row * height);
    out->bytes_per_row = bytes_per_row;
    out->bytes_per_image = bytes_per_image;
    out->copy_depth = copy_depth;
    out->available_bytes = available_bytes;
    if (available_bytes < bytes_per_image * copy_depth) {
        return -2;
    }

    const void* data = src_data;
    uint64_t bpr = bytes_per_row;
    uint64_t bpi = bytes_per_image;
    void* expanded = nullptr;
    if (mglRenderTextureUploadNeedsDepthNormalization(internal_format, pixel_format)) {
        size_t ebpr = 0, ebpi = 0;
        expanded = mglRenderCreateDepth24Stencil8NormalizedUpload(
            src_data, (size_t)width, (size_t)height, (size_t)bytes_per_row,
            &ebpr, &ebpi);
        if (expanded) { data = expanded; bpr = ebpr; bpi = ebpi; }
    } else if (mglRenderTextureInternalFormatNeedsRGBA8Expansion(
            internal_format, pixel_format)) {
        size_t ebpr = 0;
        size_t ebpi = 0;
        expanded = mglRenderCreateRGBA8ExpandedUpload(
            src_data, (size_t)width, (size_t)height,
            (size_t)bytes_per_row, internal_format, &ebpr, &ebpi);
        if (expanded) {
            data = expanded;
            bpr = ebpr;
            bpi = ebpi;
        }
    } else if (mglRenderTextureNeedsChannelExpansion(
                   internal_format, pixel_format)) {
        size_t ebpr = 0;
        size_t ebpi = 0;
        expanded = mglRenderCreateChannelExpandedUpload(
            internal_format, pixel_format, src_data, (size_t)width,
            (size_t)height, (size_t)bytes_per_row, &ebpr, &ebpi);
        if (expanded) {
            data = expanded;
            bpr = ebpr;
            bpi = ebpi;
        }
    }
    out->data = data;
    out->bytes_per_row = bpr;
    out->bytes_per_image = bpi;
    out->owns_data = expanded ? 1 : 0;
    return 0;
}

extern "C"
int mglRenderCopySnorm8TextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    if (pf != MTL::PixelFormatR8Snorm &&
        pf != MTL::PixelFormatRG8Snorm &&
        pf != MTL::PixelFormatRGBA8Snorm) {
        return 0;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }

    uint64_t src_bpp = mglRenderReadbackBytesPerPixel(pixel_format);
    uint32_t comp_bytes = mglSizeForType(type);
    uint64_t dst_pixel_bytes = mglPixelTypeIsPacked(type)
        ? (uint64_t)comp_bytes
        : (uint64_t)comp_bytes * (uint64_t)slots;
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    int src_channels = (int)src_bpp;
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            const int8_t* s = reinterpret_cast<const int8_t*>(
                src_row + (x * src_bpp));
            uint8_t* dp = dst_row + (x * dst_pixel_bytes);
            for (int c = 0; c < slots; ++c) {
                int idx = src_idx[c];
                float fv;
                if (idx >= src_channels) {
                    fv = mglReadbackMissingChannelFloat(idx);
                } else {
                    fv = mglRenderSnorm8ToFloat(s[idx]);
                }
                uint8_t* out = dp + (uint64_t)c * (uint64_t)comp_bytes;
                if (type == GL_BYTE) {
                    int32_t iv = (int32_t)lroundf(fv * 127.0f);
                    if (iv > 127) iv = 127;
                    if (iv < -128) iv = -128;
                    int8_t biv = (int8_t)iv;
                    memcpy(out, &biv, sizeof(biv));
                } else if (type == GL_UNSIGNED_BYTE) {
                    float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                    uint8_t iv = (uint8_t)lroundf(cv * 255.0f);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_FLOAT) {
                    memcpy(out, &fv, sizeof(fv));
                } else if (type == GL_HALF_FLOAT) {
                    uint16_t iv = mglFloatToHalf(fv);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_SHORT) {
                    int32_t iv = (int32_t)lroundf(fv * 32767.0f);
                    if (iv > 32767) iv = 32767;
                    if (iv < -32768) iv = -32768;
                    int16_t siv = (int16_t)iv;
                    memcpy(out, &siv, sizeof(siv));
                } else if (type == GL_UNSIGNED_SHORT) {
                    float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                    uint16_t iv = (uint16_t)lroundf(cv * 65535.0f);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_INT) {
                    int64_t iv = (int64_t)llroundf(fv * 2147483647.0f);
                    if (iv > 2147483647LL) iv = 2147483647LL;
                    if (iv < -2147483648LL) iv = -2147483648LL;
                    int32_t iiv = (int32_t)iv;
                    memcpy(out, &iiv, sizeof(iiv));
                } else if (type == GL_UNSIGNED_INT) {
                    float cv = fv > 1.0f ? 1.0f : (fv < 0.0f ? 0.0f : fv);
                    uint32_t iv = (uint32_t)llroundf(cv * 4294967295.0f);
                    memcpy(out, &iv, sizeof(iv));
                }
            }
        }
    }
    return 1;
}

extern "C"
int mglRenderCopyRG11B10TextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    if (pf != MTL::PixelFormatRG11B10Float ||
        !mglReadbackRG11B10TypeAccepted(type)) {
        return 0;
    }

    const uint64_t src_bpp = 4u;
    uint32_t comp_bytes = mglSizeForType(type);
    if (type == GL_UNSIGNED_INT_10F_11F_11F_REV && format == GL_RGB) {
        if (dst_bytes_per_row < width * src_bpp) {
            return 0;
        }
        const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
        uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
        for (uint64_t y = 0; y < height; y++) {
            const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
            uint64_t dst_y = flip_y ? (height - 1u - y) : y;
            uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
            memcpy(dst_row, src_row, width * src_bpp);
        }
        return 1;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }

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
            float float_vals[4] = {
                mglUnpackUnsignedFloatComponent(packed, 6u),
                mglUnpackUnsignedFloatComponent(packed >> 11u, 6u),
                mglUnpackUnsignedFloatComponent(packed >> 22u, 5u),
                1.0f
            };

            if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                float r = float_vals[src_idx[0]];
                float g = (slots > 1) ? float_vals[src_idx[1]] : 0.0f;
                float b = (slots > 2) ? float_vals[src_idx[2]] : 0.0f;
                uint32_t out = (mglFloatToFloat11(r) & 0x7ffu) |
                               ((mglFloatToFloat11(g) & 0x7ffu) << 11u) |
                               ((mglFloatToFloat10(b) & 0x3ffu) << 22u);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                float r = float_vals[src_idx[0]];
                float g = (slots > 1) ? float_vals[src_idx[1]] : 0.0f;
                float b = (slots > 2) ? float_vals[src_idx[2]] : 0.0f;
                uint32_t out = mglPackRGBToSharedExp(r, g, b);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                uint8_t r8 = mglRenderFloatToUnorm8(float_vals[src_idx[0]]);
                uint8_t g8 = (slots > 1)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[1]]) : 0u;
                uint8_t b8 = (slots > 2)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[2]]) : 0u;
                uint8_t a8 = (slots > 3)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[3]]) : 0u;
                uint32_t out = ((uint32_t)r8 << 24u) | ((uint32_t)g8 << 16u) |
                               ((uint32_t)b8 << 8u) | a8;
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                uint8_t r8 = mglRenderFloatToUnorm8(float_vals[src_idx[0]]);
                uint8_t g8 = (slots > 1)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[1]]) : 0u;
                uint8_t b8 = (slots > 2)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[2]]) : 0u;
                uint8_t a8 = (slots > 3)
                    ? mglRenderFloatToUnorm8(float_vals[src_idx[3]]) : 0u;
                uint32_t out = r8 | ((uint32_t)g8 << 8u) |
                               ((uint32_t)b8 << 16u) | ((uint32_t)a8 << 24u);
                memcpy(dst_row + (x * dst_pixel_bytes), &out, sizeof(out));
            } else {
                for (int c = 0; c < slots; ++c) {
                    float fv = float_vals[src_idx[c]];
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

extern "C"
int mglRenderCopy16or32TextureBytesToGL(
    const void* src, uint64_t src_bytes_per_row,
    void* dst, uint64_t dst_bytes_per_row,
    uint64_t width, uint64_t height,
    uint32_t pixel_format, uint32_t format, uint32_t type, int flip_y) {
    if (!src || !dst || width == 0u || height == 0u) {
        return 0;
    }
    const MTL::PixelFormat pf = static_cast<MTL::PixelFormat>(pixel_format);
    const int is16u =
        (pf == MTL::PixelFormatR16Unorm ||
         pf == MTL::PixelFormatRG16Unorm ||
         pf == MTL::PixelFormatRGBA16Unorm);
    const int is16s =
        (pf == MTL::PixelFormatR16Snorm ||
         pf == MTL::PixelFormatRG16Snorm ||
         pf == MTL::PixelFormatRGBA16Snorm);
    const int is16f =
        (pf == MTL::PixelFormatR16Float ||
         pf == MTL::PixelFormatRG16Float ||
         pf == MTL::PixelFormatRGBA16Float);
    const int is32f =
        (pf == MTL::PixelFormatR32Float ||
         pf == MTL::PixelFormatRG32Float ||
         pf == MTL::PixelFormatRGBA32Float);
    if (!(is16u || is16s || is16f || is32f) ||
        !mglReadback16or32TypeAccepted(type)) {
        return 0;
    }

    const uint64_t src_bpp = mglRenderReadbackBytesPerPixel(pixel_format);
    int src_channels = mglWideSrcChannelCount(pf);
    if (src_bpp == 0u || src_channels == 0) {
        return 0;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }

    uint32_t comp_bytes = mglSizeForType(type);
    uint64_t dst_pixel_bytes = mglPixelTypeIsPacked(type)
        ? (uint64_t)comp_bytes
        : (uint64_t)comp_bytes * (uint64_t)slots;
    if (dst_pixel_bytes == 0u || dst_bytes_per_row < width * dst_pixel_bytes) {
        return 0;
    }

    const int output_is_packed = mglPixelTypeIsPacked(type);
    const uint8_t* src_bytes = static_cast<const uint8_t*>(src);
    uint8_t* dst_bytes = static_cast<uint8_t*>(dst);
    for (uint64_t y = 0; y < height; y++) {
        const uint8_t* src_row = src_bytes + (y * src_bytes_per_row);
        uint64_t dst_y = flip_y ? (height - 1u - y) : y;
        uint8_t* dst_row = dst_bytes + (dst_y * dst_bytes_per_row);
        for (uint64_t x = 0; x < width; x++) {
            const uint8_t* s = src_row + (x * src_bpp);
            uint8_t* dp = dst_row + (x * dst_pixel_bytes);

            if (output_is_packed) {
                float fvals[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int c = 0; c < slots; ++c) {
                    int idx = src_idx[c];
                    if (idx >= src_channels) {
                        fvals[c] = mglReadbackMissingChannelFloat(idx);
                    } else {
                        fvals[c] = mglRead16or32SourceFloat(
                            s, idx, is16u, is16s, is16f);
                    }
                }
                if (slots < 4) {
                    const int needs_alpha =
                        (type == GL_UNSIGNED_SHORT_4_4_4_4 ||
                         type == GL_UNSIGNED_SHORT_4_4_4_4_REV ||
                         type == GL_UNSIGNED_SHORT_5_5_5_1 ||
                         type == GL_UNSIGNED_SHORT_1_5_5_5_REV ||
                         type == GL_UNSIGNED_INT_8_8_8_8 ||
                         type == GL_UNSIGNED_INT_8_8_8_8_REV ||
                         type == GL_UNSIGNED_INT_10_10_10_2 ||
                         type == GL_UNSIGNED_INT_2_10_10_10_REV);
                    if (needs_alpha) fvals[3] = 1.0f;
                }

                if (type == GL_UNSIGNED_BYTE_3_3_2) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    dp[0] = (uint8_t)(((uint32_t)lroundf(r * 7.0f) << 5) |
                                      ((uint32_t)lroundf(g * 7.0f) << 2) |
                                      (uint32_t)lroundf(b * 3.0f));
                } else if (type == GL_UNSIGNED_BYTE_2_3_3_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    dp[0] = (uint8_t)((uint32_t)lroundf(r * 7.0f) |
                                      ((uint32_t)lroundf(g * 7.0f) << 3) |
                                      ((uint32_t)lroundf(b * 3.0f) << 6));
                } else if (type == GL_UNSIGNED_SHORT_5_6_5) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    uint16_t packed = (uint16_t)(((uint32_t)lroundf(r * 31.0f) << 11) |
                                                 ((uint32_t)lroundf(g * 63.0f) << 5) |
                                                 (uint32_t)lroundf(b * 31.0f));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_5_6_5_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    uint16_t packed = (uint16_t)((uint32_t)lroundf(r * 31.0f) |
                                                 ((uint32_t)lroundf(g * 63.0f) << 5) |
                                                 ((uint32_t)lroundf(b * 31.0f) << 11));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_4_4_4_4) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint16_t packed = (uint16_t)(((uint32_t)lroundf(r * 15.0f) << 12) |
                                                 ((uint32_t)lroundf(g * 15.0f) << 8) |
                                                 ((uint32_t)lroundf(b * 15.0f) << 4) |
                                                 (uint32_t)lroundf(a * 15.0f));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_4_4_4_4_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint16_t packed = (uint16_t)((uint32_t)lroundf(r * 15.0f) |
                                                 ((uint32_t)lroundf(g * 15.0f) << 4) |
                                                 ((uint32_t)lroundf(b * 15.0f) << 8) |
                                                 ((uint32_t)lroundf(a * 15.0f) << 12));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_5_5_5_1) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint16_t packed = (uint16_t)(((uint32_t)lroundf(r * 31.0f) << 11) |
                                                 ((uint32_t)lroundf(g * 31.0f) << 6) |
                                                 ((uint32_t)lroundf(b * 31.0f) << 1) |
                                                 (a >= 0.5f ? 1u : 0u));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint16_t packed = (uint16_t)((uint32_t)lroundf(r * 31.0f) |
                                                 ((uint32_t)lroundf(g * 31.0f) << 5) |
                                                 ((uint32_t)lroundf(b * 31.0f) << 10) |
                                                 ((a >= 0.5f ? 1u : 0u) << 15));
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint32_t packed = ((uint32_t)lroundf(r * 255.0f) << 24) |
                                      ((uint32_t)lroundf(g * 255.0f) << 16) |
                                      ((uint32_t)lroundf(b * 255.0f) << 8) |
                                      (uint32_t)lroundf(a * 255.0f);
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint32_t packed = (uint32_t)lroundf(r * 255.0f) |
                                      ((uint32_t)lroundf(g * 255.0f) << 8) |
                                      ((uint32_t)lroundf(b * 255.0f) << 16) |
                                      ((uint32_t)lroundf(a * 255.0f) << 24);
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_10_10_10_2) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint32_t packed = ((uint32_t)lroundf(r * 1023.0f) << 22) |
                                      ((uint32_t)lroundf(g * 1023.0f) << 12) |
                                      ((uint32_t)lroundf(b * 1023.0f) << 2) |
                                      (uint32_t)lroundf(a * 3.0f);
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
                    float r = fvals[0] > 1.0f ? 1.0f : (fvals[0] < 0.0f ? 0.0f : fvals[0]);
                    float g = (slots > 1) ? (fvals[1] > 1.0f ? 1.0f : (fvals[1] < 0.0f ? 0.0f : fvals[1])) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] > 1.0f ? 1.0f : (fvals[2] < 0.0f ? 0.0f : fvals[2])) : 0.0f;
                    float a = (slots > 3) ? (fvals[3] > 1.0f ? 1.0f : (fvals[3] < 0.0f ? 0.0f : fvals[3])) : 1.0f;
                    uint32_t packed = (uint32_t)lroundf(r * 1023.0f) |
                                      ((uint32_t)lroundf(g * 1023.0f) << 10) |
                                      ((uint32_t)lroundf(b * 1023.0f) << 20) |
                                      ((uint32_t)lroundf(a * 3.0f) << 30);
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                    float r = fvals[0] < 0.0f ? 0.0f : fvals[0];
                    float g = (slots > 1) ? (fvals[1] < 0.0f ? 0.0f : fvals[1]) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] < 0.0f ? 0.0f : fvals[2]) : 0.0f;
                    uint32_t packed = mglFloatToFloat11(r) |
                                      (mglFloatToFloat11(g) << 11) |
                                      (mglFloatToFloat10(b) << 22);
                    memcpy(dp, &packed, sizeof(packed));
                } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                    float r = fvals[0] < 0.0f ? 0.0f : fvals[0];
                    float g = (slots > 1) ? (fvals[1] < 0.0f ? 0.0f : fvals[1]) : 0.0f;
                    float b = (slots > 2) ? (fvals[2] < 0.0f ? 0.0f : fvals[2]) : 0.0f;
                    uint32_t packed = mglPackRGBToSharedExp(r, g, b);
                    memcpy(dp, &packed, sizeof(packed));
                }
                continue;
            }

            for (int c = 0; c < slots; ++c) {
                int idx = src_idx[c];
                float fv;
                if (idx >= src_channels) {
                    fv = mglReadbackMissingChannelFloat(idx);
                } else {
                    fv = mglRead16or32SourceFloat(
                        s, idx, is16u, is16s, is16f);
                }
                uint8_t* out = dp + (uint64_t)c * (uint64_t)comp_bytes;
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
    return 1;
}

extern "C"
int mglRenderCopyUnorm8ScalarTextureBytesToGL(
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
    if ((!source_is_rgba && !source_is_bgra) ||
        !mglReadbackUnorm8ScalarTypeAccepted(type)) {
        return 0;
    }

    int slots = 0;
    int src_idx[4] = {0, 0, 0, 0};
    if (!mglReadbackFormatChannelMap(format, &slots, src_idx)) {
        return 0;
    }

    uint32_t comp_bytes = mglSizeForType(type);
    uint64_t dst_pixel_bytes = (uint64_t)comp_bytes * (uint64_t)slots;
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
            const uint8_t* s = src_row + (x * 4u);
            const unsigned cv[4] = {
                source_is_rgba ? s[0] : s[2],
                s[1],
                source_is_rgba ? s[2] : s[0],
                s[3]
            };
            uint8_t* dp = dst_row + (x * dst_pixel_bytes);
            for (int c = 0; c < slots; ++c) {
                unsigned v = cv[src_idx[c]];
                uint8_t* out = dp + (uint64_t)c * (uint64_t)comp_bytes;
                if (type == GL_BYTE) {
                    float fv = (float)v / 255.0f;
                    int32_t iv = (int32_t)lroundf(fv * 127.0f);
                    if (iv > 127) iv = 127;
                    if (iv < -128) iv = -128;
                    int8_t biv = (int8_t)iv;
                    memcpy(out, &biv, sizeof(biv));
                } else if (type == GL_UNSIGNED_SHORT) {
                    uint16_t iv = (uint16_t)((uint32_t)v * 257u);
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_SHORT) {
                    int32_t scaled = (int32_t)((uint32_t)v * 32767u / 255u);
                    if (scaled > 32767) scaled = 32767;
                    int16_t iv = (int16_t)scaled;
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_UNSIGNED_INT) {
                    uint32_t iv = (uint32_t)v * 16843009u;
                    memcpy(out, &iv, sizeof(iv));
                } else if (type == GL_INT) {
                    int32_t scaled =
                        (int32_t)((uint64_t)v * 2147483647ULL / 255u);
                    if (scaled > 2147483647) scaled = 2147483647;
                    memcpy(out, &scaled, sizeof(scaled));
                } else if (type == GL_FLOAT) {
                    float fv = (float)v / 255.0f;
                    memcpy(out, &fv, sizeof(fv));
                } else {
                    uint16_t iv = mglFloatToHalf((float)v / 255.0f);
                    memcpy(out, &iv, sizeof(iv));
                }
            }
        }
    }
    return 1;
}

extern "C"
int mglRenderCopyUnorm8PackedTextureBytesToGL(
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
    if ((!source_is_rgba && !source_is_bgra) ||
        !mglReadbackUnorm8PackedTypeAccepted(type)) {
        return 0;
    }

    uint64_t dst_pixel_bytes = (uint64_t)mglSizeForType(type);
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
            const uint8_t* s = src_row + (x * 4u);
            uint32_t r = source_is_rgba ? s[0] : s[2];
            uint32_t g = s[1];
            uint32_t b = source_is_rgba ? s[2] : s[0];
            uint32_t a = s[3];
            uint32_t rr = r, gg = g, bb = b, aa = a;
            if (format == GL_BGRA || format == GL_BGR) {
                uint32_t tmp = rr;
                rr = bb;
                bb = tmp;
            }
            uint8_t* d = dst_row + (x * dst_pixel_bytes);
            if (type == GL_UNSIGNED_BYTE_3_3_2) {
                d[0] = (uint8_t)(((rr >> 5u) << 5u) | ((gg >> 5u) << 2u) | (bb >> 6u));
            } else if (type == GL_UNSIGNED_BYTE_2_3_3_REV) {
                d[0] = (uint8_t)((rr >> 5u) | ((gg >> 5u) << 3u) | ((bb >> 6u) << 6u));
            } else if (type == GL_UNSIGNED_SHORT_5_6_5) {
                uint16_t packed = (uint16_t)(((rr >> 3u) << 11u) | ((gg >> 2u) << 5u) | (bb >> 3u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_SHORT_5_6_5_REV) {
                uint16_t packed = (uint16_t)((rr >> 3u) | ((gg >> 2u) << 5u) | ((bb >> 3u) << 11u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_SHORT_4_4_4_4) {
                uint16_t packed = (uint16_t)(((rr >> 4u) << 12u) | ((gg >> 4u) << 8u) |
                                             ((bb >> 4u) << 4u) | (aa >> 4u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_SHORT_4_4_4_4_REV) {
                uint16_t packed = (uint16_t)((rr >> 4u) | ((gg >> 4u) << 4u) |
                                             ((bb >> 4u) << 8u) | ((aa >> 4u) << 12u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_SHORT_5_5_5_1) {
                uint16_t packed = (uint16_t)(((rr >> 3u) << 11u) | ((gg >> 3u) << 6u) |
                                             ((bb >> 3u) << 1u) | (aa >= 128u ? 1u : 0u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_SHORT_1_5_5_5_REV) {
                uint16_t packed = (uint16_t)((rr >> 3u) | ((gg >> 3u) << 5u) |
                                             ((bb >> 3u) << 10u) |
                                             ((aa >= 128u ? 1u : 0u) << 15u));
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8) {
                uint32_t packed = ((uint32_t)rr << 24u) | ((uint32_t)gg << 16u) |
                                  ((uint32_t)bb << 8u) | aa;
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_8_8_8_8_REV) {
                uint32_t packed = rr | ((uint32_t)gg << 8u) |
                                  ((uint32_t)bb << 16u) | ((uint32_t)aa << 24u);
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_10_10_10_2) {
                uint32_t r10 = rr * 1023u / 255u;
                uint32_t g10 = gg * 1023u / 255u;
                uint32_t b10 = bb * 1023u / 255u;
                uint32_t a2 = aa * 3u / 255u;
                uint32_t packed = (r10 << 22u) | (g10 << 12u) | (b10 << 2u) | a2;
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
                uint32_t r10 = rr * 1023u / 255u;
                uint32_t g10 = gg * 1023u / 255u;
                uint32_t b10 = bb * 1023u / 255u;
                uint32_t a2 = aa * 3u / 255u;
                uint32_t packed = r10 | (g10 << 10u) | (b10 << 20u) | (a2 << 30u);
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
                uint32_t packed = mglPackUnsignedFloatFromUNorm8(rr, 6u) |
                                  (mglPackUnsignedFloatFromUNorm8(gg, 6u) << 11u) |
                                  (mglPackUnsignedFloatFromUNorm8(bb, 5u) << 22u);
                memcpy(d, &packed, sizeof(packed));
            } else if (type == GL_UNSIGNED_INT_5_9_9_9_REV) {
                uint32_t packed = mglPackRGBToSharedExp(
                    (double)rr / 255.0, (double)gg / 255.0, (double)bb / 255.0);
                memcpy(d, &packed, sizeof(packed));
            }
        }
    }
    return 1;
}

int mglRenderBindingSetTextureForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, void* texture, uint32_t stage, uint32_t index) {
    return mglRenderBindingSetTexture(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        texture, stage, index);
}

int mglRenderBindingSetSamplerForOwner(MGLBindingState * binding_state, MGLRenderEncoderOwner * render_encoder_owner, void* sampler, uint32_t stage, uint32_t index) {
    return mglRenderBindingSetSampler(
        binding_state, mglRenderActiveRenderEncoder(render_encoder_owner),
        sampler, stage, index);
}

int mglRenderSetRenderTextureForOwner(MGLRenderEncoderOwner * render_encoder_owner, void* texture, uint32_t stage, uint32_t index) {
    return mglRenderSetRenderTexture(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        texture, stage, index);
}

int mglRenderSetRenderSamplerForOwner(MGLRenderEncoderOwner * render_encoder_owner, void* sampler, uint32_t stage, uint32_t index) {
    return mglRenderSetRenderSampler(
        mglRenderActiveRenderEncoder(render_encoder_owner),
        sampler, stage, index);
}

static MGLRenderTextureDescriptorState
mglRenderReadTextureDescriptor(MTL::TextureDescriptor* descriptor) {
    MGLRenderTextureDescriptorState state = {};
    if (!descriptor) return state;
    state.texture_type = static_cast<uint32_t>(descriptor->textureType());
    state.pixel_format = static_cast<uint32_t>(descriptor->pixelFormat());
    state.width = descriptor->width();
    state.height = descriptor->height();
    state.depth = descriptor->depth();
    state.mipmap_level_count = descriptor->mipmapLevelCount();
    state.sample_count = descriptor->sampleCount();
    state.array_length = descriptor->arrayLength();
    state.resource_options = descriptor->resourceOptions();
    state.usage = descriptor->usage();
    state.cpu_cache_mode = static_cast<uint32_t>(descriptor->cpuCacheMode());
    state.storage_mode = static_cast<uint32_t>(descriptor->storageMode());
    state.hazard_tracking_mode =
        static_cast<uint32_t>(descriptor->hazardTrackingMode());
    state.compression_type =
        static_cast<uint32_t>(descriptor->compressionType());
    state.placement_sparse_page_size =
        static_cast<uint32_t>(descriptor->placementSparsePageSize());
    state.allow_gpu_optimized_contents =
        descriptor->allowGPUOptimizedContents() ? 1u : 0u;
    MTL::TextureSwizzleChannels swizzle = descriptor->swizzle();
    state.swizzle_red = static_cast<uint32_t>(swizzle.red);
    state.swizzle_green = static_cast<uint32_t>(swizzle.green);
    state.swizzle_blue = static_cast<uint32_t>(swizzle.blue);
    state.swizzle_alpha = static_cast<uint32_t>(swizzle.alpha);
    state.has_swizzle = 1u;
    return state;
}

int mglRenderCreateTextureFromDescriptor(
    void* descriptor_handle,
    const char* label,
    void** texture_out) {
    auto* descriptor =
        reinterpret_cast<MTL::TextureDescriptor*>(descriptor_handle);
    MGLRenderTextureDescriptorState state =
        mglRenderReadTextureDescriptor(descriptor);
    return mglRenderCreateTextureFromState(&state, label, texture_out);
}

int mglRenderCreateBufferTextureFromDescriptor(
    void* buffer,
    void* descriptor_handle,
    uint64_t offset,
    uint64_t bytes_per_row,
    void** texture_out) {
    MGLRenderTextureDescriptorState state =
        mglRenderReadTextureDescriptor(
            reinterpret_cast<MTL::TextureDescriptor*>(descriptor_handle));
    return mglRenderCreateBufferTextureFromState(
        buffer, &state, offset, bytes_per_row, texture_out);
}

void* mglRenderGetRenderPassAttachmentTextureOwner(MGLRenderPassStateOwner * owner_handle, uint32_t attachment_kind, uint32_t color_index) {
    auto* owner = reinterpret_cast<mgl::RenderPassStateOwner*>(static_cast<void*>(owner_handle));
    auto* attachment = mglRenderAttachmentForOwner(
        owner, attachment_kind, color_index);
    return attachment ? attachment->texture : nullptr;
}
