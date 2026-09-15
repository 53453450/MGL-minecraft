/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_sampled_fallback.c — the sampled-texture fallback chain moved out of
 * MGLRenderer+Texture.m (P0-1, log 132).
 *
 * Mechanical translation: `self`/`_backend`/`_device` arrive through the state
 * areas, the file statics mglTextureCreateTexture / …CreateBuffer /
 * …CreateBufferTexture / …BufferContents / …ReplaceRegion are repeated as the
 * C twins mglSf*, NSLog becomes fprintf on the same sink, and the one
 * @try/@catch (the texture-buffer texture creation) becomes the existing shell
 * guarded call (mgl_gpu_recovery.h), which is the C-side home of that frame.
 *
 * OWNERSHIP (log 128 rule): the creates return +1; the backend cache retains
 * through Set/Put, so the twin drops its creation reference before returning
 * the borrowed pointer — the ARC local's end-of-scope release.  Failure paths
 * drop it as well.
 */

#include <stdio.h>
#include <string.h>

#include <CoreFoundation/CoreFoundation.h>

#include "mgl_sampled_fallback.h"
#include "mgl_renderer_ports.h"    /* state areas */
#include "mgl_renderer_backend.h"  /* fallback resource / sampled-texture caches */
#include "mgl_render.h"            /* texture/buffer creation, region replace */
#include "mgl_texture_compat.h"    /* mglTextureDataKindName */
#include "mgl_region_value.h"      /* mglTextureRegion1D / 2D */
#include "mgl_gpu_recovery.h"      /* mglPlatformShellGuardedCallCtx (@try/@catch) */

/* The texture-usage / storage enums of MGLRenderer+Texture.m (log 132). */
enum {
    MGL_TEXTURE_RESOURCE_STORAGE_SHARED = 0u,
    MGL_TEXTURE_USAGE_SHADER_READ = 1u,
};

/* The Objective-C header's kMGLEnableSampledTextureFallback = YES. */
#define kMGLEnableSampledTextureFallback 1

/* === C twins of the +Texture.m file statics ============================== */

/* +1 texture, or NULL. */
static void *mglSfCreateTexture(const MGLRenderTextureDescriptorState *desc)
{
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(desc, NULL, &texture) == 0 && texture) {
        return texture;
    }
    return NULL;
}

/* +1 buffer, or NULL. */
static void *mglSfCreateBuffer(size_t length, uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBuffer((uint64_t)length, options, NULL, &buffer) == 0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

static void *mglSfBufferContents(void *buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents(buffer, &contents, &length) == 0
               ? contents
               : NULL;
}

/* +1 texture, or NULL. */
static void *mglSfCreateBufferTexture(void *buffer,
                                      const MGLRenderTextureDescriptorState *desc,
                                      size_t offset, size_t bytes_per_row)
{
    void *texture = NULL;
    if (mglRenderCreateBufferTextureFromState(buffer, desc, offset, bytes_per_row,
                                              &texture) == 0 &&
        texture) {
        return texture;
    }
    return NULL;
}

static void mglSfReplaceRegion(void *texture, MGLRegionValue region,
                               size_t level, size_t slice, const void *bytes,
                               size_t bytes_per_row, size_t bytes_per_image,
                               int use_slice)
{
    if (mglRenderTextureReplaceRegion(
            texture, region.origin.x, region.origin.y, region.origin.z,
            region.size.width, region.size.height, region.size.depth, level,
            slice, bytes, bytes_per_row, bytes_per_image,
            use_slice ? 1 : 0) != 0) {
        fprintf(stderr, "MGL ERROR: texture region replace failed\n");
    }
}

/* The @try/@catch around the texture-buffer texture creation. */
typedef struct {
    void *storage;
    const MGLRenderTextureDescriptorState *desc;
    size_t texel_count;
    size_t bytes_per_texel;
    void *texture_out;
} MglSfGuardedCtx;

static int mglSfCreateBufferTextureGuarded(void *renderer, void *ctx_raw)
{
    (void)renderer;
    MglSfGuardedCtx *ctx = (MglSfGuardedCtx *)ctx_raw;
    ctx->texture_out = mglSfCreateBufferTexture(
        ctx->storage, ctx->desc, 0, ctx->texel_count * ctx->bytes_per_texel);
    return ctx->texture_out != NULL ? 1 : 0;
}

/* -fallbackSampledTexture */
void *mglSampledFallbackTexture(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);

    void *cached = mglRendererBackendGetFallbackResource(
        areas.backend, MGL_RENDERER_BACKEND_FALLBACK_SAMPLED_TEXTURE);
    if (cached || !kMGLEnableSampledTextureFallback) {
        return cached;
    }

    MGLRenderTextureDescriptorState desc = {
        .texture_type = MGLTextureType2D,
        .pixel_format = mglRenderFallbackSampledPixelFormat(
            (uint32_t)MGLTextureDataKindFloat),
        .width = 1u,
        .height = 1u,
        .depth = 1u,
        .mipmap_level_count = 1u,
        .sample_count = 1u,
        .array_length = 1u,
        .usage = MGL_TEXTURE_USAGE_SHADER_READ,
    };

    void *texture = mglSfCreateTexture(&desc);
    if (texture) {
        uint32_t pixel = 0xff000000u;
        mglSfReplaceRegion(texture, mglTextureRegion2D(0, 0, 1, 1), 0, 0, &pixel,
                           sizeof(pixel), 0, 0);
        if (mglRendererBackendSetFallbackResource(
                areas.backend, MGL_RENDERER_BACKEND_FALLBACK_SAMPLED_TEXTURE,
                texture) != 0) {
            CFRelease((CFTypeRef)texture);
            return NULL;
        }
        fprintf(stderr,
                "MGL INFO: Created 1x1 fallback sampled texture for missing "
                "shader resources\n");
    } else {
        fprintf(stderr, "MGL ERROR: Failed to create fallback sampled texture\n");
        return NULL;
    }

    /* The backend cache keeps it (the ARC local released its +1 here). */
    CFRelease((CFTypeRef)texture);
    return texture;
}

/* -fallbackCubeSampledTexture */
void *mglSampledFallbackCubeTexture(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);

    void *cached = mglRendererBackendGetFallbackResource(
        areas.backend, MGL_RENDERER_BACKEND_FALLBACK_CUBE_SAMPLED_TEXTURE);
    if (cached || !kMGLEnableSampledTextureFallback) {
        return cached;
    }

    MGLRenderTextureDescriptorState desc = {
        .texture_type = MGLTextureTypeCube,
        .pixel_format = mglRenderFallbackSampledPixelFormat(
            (uint32_t)MGLTextureDataKindFloat),
        .width = 1u,
        .height = 1u,
        .depth = 1u,
        .array_length = 1u,
        .mipmap_level_count = 1u,
        .sample_count = 1u,
        .usage = MGL_TEXTURE_USAGE_SHADER_READ,
    };

    void *texture = mglSfCreateTexture(&desc);
    if (texture) {
        uint32_t pixel = 0xff000000u;
        for (size_t face = 0; face < 6; face++) {
            mglSfReplaceRegion(texture, mglTextureRegion2D(0, 0, 1, 1), 0, face,
                               &pixel, sizeof(pixel), sizeof(pixel), 1);
        }
        if (mglRendererBackendSetFallbackResource(
                areas.backend, MGL_RENDERER_BACKEND_FALLBACK_CUBE_SAMPLED_TEXTURE,
                texture) != 0) {
            CFRelease((CFTypeRef)texture);
            return NULL;
        }
        fprintf(stderr,
                "MGL INFO: Created 1x1 fallback cube sampled texture for missing "
                "shader resources\n");
    } else {
        fprintf(stderr,
                "MGL ERROR: Failed to create fallback cube sampled texture\n");
        return NULL;
    }

    CFRelease((CFTypeRef)texture);
    return texture;
}

/* -fallbackTextureBufferSampledTexture */
void *mglSampledFallbackTextureBuffer(void *renderer)
{
    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);

    void *cached_texture = mglRendererBackendGetFallbackResource(
        areas.backend, MGL_RENDERER_BACKEND_FALLBACK_SINT_TEXTURE_BUFFER);
    if (cached_texture || !kMGLEnableSampledTextureFallback) {
        return cached_texture;
    }

    const size_t k_texel_count = 64u;
    const size_t k_bytes_per_texel = 4u;

    void *storage = mglRendererBackendGetFallbackResource(
        areas.backend, MGL_RENDERER_BACKEND_FALLBACK_TEXTURE_BUFFER_STORAGE);
    if (!storage) {
        storage = mglSfCreateBuffer(k_texel_count * k_bytes_per_texel,
                                    MGL_TEXTURE_RESOURCE_STORAGE_SHARED);
        if (storage && mglSfBufferContents(storage)) {
            memset(mglSfBufferContents(storage), 0,
                   k_texel_count * k_bytes_per_texel);
        }
        if (storage &&
            mglRendererBackendSetFallbackResource(
                areas.backend,
                MGL_RENDERER_BACKEND_FALLBACK_TEXTURE_BUFFER_STORAGE,
                storage) != 0) {
            CFRelease((CFTypeRef)storage);
            storage = NULL;
        }
    }

    if (!storage) {
        fprintf(stderr,
                "MGL ERROR: Failed to create fallback texture-buffer backing "
                "storage\n");
        return NULL;
    }

    MGLRenderTextureDescriptorState desc = {
        .texture_type = MGLTextureTypeTextureBuffer,
        .pixel_format = mglRenderFallbackSampledPixelFormat(
            (uint32_t)MGLTextureDataKindSint),
        .width = k_texel_count,
        .height = 1u,
        .depth = 1u,
        .array_length = 1u,
        .mipmap_level_count = 1u,
        .sample_count = 1u,
        .usage = MGL_TEXTURE_USAGE_SHADER_READ,
    };

    MglSfGuardedCtx guarded = {
        .storage = storage,
        .desc = &desc,
        .texel_count = k_texel_count,
        .bytes_per_texel = k_bytes_per_texel,
        .texture_out = NULL,
    };
    if (!mglPlatformShellGuardedCallCtx(renderer,
                                        "fallback texture-buffer texture creation",
                                        mglSfCreateBufferTextureGuarded, &guarded,
                                        NULL)) {
        fprintf(stderr,
                "MGL ERROR: Failed to create fallback texture-buffer texture: "
                "(caught exception)\n");
        guarded.texture_out = NULL;
    }
    cached_texture = guarded.texture_out;

    if (cached_texture &&
        mglRendererBackendSetFallbackResource(
            areas.backend, MGL_RENDERER_BACKEND_FALLBACK_SINT_TEXTURE_BUFFER,
            cached_texture) != 0) {
        cached_texture = NULL;
    }
    if (guarded.texture_out) {
        /* The cache holds its own reference (or the failure path drops it). */
        CFRelease((CFTypeRef)guarded.texture_out);
    }
    if (cached_texture) {
        fprintf(stderr,
                "MGL INFO: Created fallback signed integer texture buffer for "
                "missing/invalid texel-buffer resources\n");
    }

    return cached_texture;
}

/* -fallbackSampledTextureForExpectedType: */
void *mglSampledFallbackTextureForType(void *renderer, uint32_t expected_type)
{
    if (mglRenderExpectedTypeIsCube(expected_type)) {
        return mglSampledFallbackCubeTexture(renderer);
    }
    if (mglRenderExpectedTypeIsTextureBuffer(expected_type)) {
        return mglSampledFallbackTextureBuffer(renderer);
    }
    return mglSampledFallbackTexture(renderer);
}

/* -fallbackSampledTextureForExpectedType:dataKind: */
void *mglSampledFallbackTextureForExpectedType(void *renderer,
                                               uint32_t expected_type,
                                               uint32_t data_kind)
{
    if (!kMGLEnableSampledTextureFallback) {
        return NULL;
    }

    MGLRendererStateAreas areas;
    mglRendererFillStateAreas(renderer, &areas);

    uint32_t texture_type = mglRenderFallbackSampledTextureType(expected_type);
    if (mglRenderExpectedTypeIsTextureBuffer(texture_type)) {
        return mglSampledFallbackTextureBuffer(renderer);
    }

    uint32_t pixel_format =
        mglRenderFallbackSampledPixelFormat((uint32_t)data_kind);

    size_t key_value = (size_t)mglRenderFallbackSampledCacheKey(texture_type,
                                                                data_kind);
    void *cached_texture = NULL;
    int cache_result = mglRendererBackendGetFallbackSampledTexture(
        areas.backend, key_value, &cached_texture);
    if (cache_result == 1) {
        return cached_texture;
    }
    if (cache_result < 0) {
        return NULL;
    }

    MGLRenderTextureDescriptorState desc = {
        .texture_type = texture_type,
        .pixel_format = pixel_format,
        .width = 1u,
        .height = 1u,
        .depth = 1u,
        .array_length = 1u,
        .mipmap_level_count = 1u,
        .sample_count = 1u,
        .usage = MGL_TEXTURE_USAGE_SHADER_READ,
    };
    if (texture_type == MGLTextureType2DMultisample ||
        texture_type == MGLTextureType2DMultisampleArray) {
        desc.sample_count = 2u;
    }

    void *texture = mglSfCreateTexture(&desc);
    if (!texture) {
        fprintf(stderr,
                "MGL ERROR: Failed to create %s fallback sampled texture "
                "type=%lu format=%lu\n",
                mglTextureDataKindName(data_kind), (unsigned long)texture_type,
                (unsigned long)pixel_format);
        return NULL;
    }

    uint32_t pixel = data_kind == MGLTextureDataKindDepth ? 0u : 0xff000000u;
    MGLRegionValue region = (texture_type == MGLTextureType1D ||
                             texture_type == MGLTextureType1DArray)
                                ? mglTextureRegion1D(0, 1)
                                : mglTextureRegion2D(0, 0, 1, 1);
    if (texture_type == MGLTextureTypeCube ||
        texture_type == MGLTextureTypeCubeArray) {
        size_t slice_count = 6u;
        for (size_t slice = 0; slice < slice_count; slice++) {
            mglSfReplaceRegion(texture, mglTextureRegion2D(0, 0, 1, 1), 0, slice,
                               &pixel, sizeof(pixel), sizeof(pixel), 1);
        }
    } else if (texture_type == MGLTextureType1DArray ||
               texture_type == MGLTextureType2DArray) {
        mglSfReplaceRegion(texture, region, 0, 0, &pixel, sizeof(pixel),
                           sizeof(pixel), 1);
    } else {
        mglSfReplaceRegion(texture, region, 0, 0, &pixel, sizeof(pixel), 0, 0);
    }

    if (mglRendererBackendPutFallbackSampledTexture(areas.backend, key_value,
                                                    texture) != 0) {
        CFRelease((CFTypeRef)texture);
        return NULL;
    }
    fprintf(stderr, "MGL INFO: Created %s fallback sampled texture type=%lu "
                    "format=%lu\n",
            mglTextureDataKindName(data_kind), (unsigned long)texture_type,
            (unsigned long)pixel_format);

    CFRelease((CFTypeRef)texture);
    return texture;
}
