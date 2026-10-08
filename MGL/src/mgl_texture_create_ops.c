/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_create_ops.c - C homes of three MGLRenderer(Texture) blocks that
 * had no self sends (P0-1, log 182): the GL completeness check, the packed
 * depth/stencil stencil-plane upload and the texel-buffer texture builder.
 */

#include "mgl_texture_create_ops.h"

#include "mgl_byte_hash.h"        /* mglTraceHashBytes / mglTraceFormatBytes */
#include "mgl_gpu_recovery.h"     /* mglRendererRecordGPUSuccess */
#include "mgl_metal_ref.h"        /* mglReleaseMetalObjNoNull */
#include "mgl_pixel_format.h"     /* mglTextureBytesPerPixelForFormat */
#include "mgl_pso_format_class.h" /* mglRenderPixelFormatIsPackedDepthStencil */
#include "mgl_renderer_ports.h"
#include "mgl_texture_compat.h"   /* mglTextureNeedsChannelExpansion */
#include "mgl_trace_log.h"
#include "mgl_region_value.h"

#include "mgl_render.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The .m's file-local constants this TU needs (values copied verbatim). */
enum {
    MGL_PD_TEXTURE_USAGE_SHADER_READ = 1u,
    MGL_PD_TEXTURE_USAGE_SHADER_WRITE = 2u,
    /* Matches MTLTextureUsageShaderAtomic (macOS 14+ / Metal 3.1). */
    MGL_PD_TEXTURE_USAGE_SHADER_ATOMIC = 0x20u,
};

/* pixel_utils.c defines this; the only declaration is in the Objective-C
 * MGLRenderer+RenderPass_Private.h. */
extern uint32_t mtlPixelFormatForGLTex(Texture *gl_tex);

/* Twin of the .m's mglTextureReplaceRegion, exposed to the other C hosts
 * (log 183).  The .m raised NSException on failure so the caller's @try/@catch
 * could report it; this twin reports the failure instead and every caller maps
 * it to the same "upload failed" path. */
int mglTextureReplaceRegionValue(void *texture, MGLRegionValue region,
                                     uint64_t level, uint64_t slice,
                                     const void *bytes, uint64_t bytesPerRow,
                                     uint64_t bytesPerImage, int useSlice)
{
    return mglRenderTextureReplaceRegion(
               texture, region.origin.x, region.origin.y, region.origin.z,
               region.size.width, region.size.height, region.size.depth, level,
               slice, bytes, bytesPerRow, bytesPerImage,
               useSlice ? 1 : 0) == 0
               ? 1
               : 0;
}

static void *mglPdTextureBufferContents(void *buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents(buffer, &contents, &length) == 0
               ? contents
               : NULL;
}

/* -checkTextureCompleteness:texType:numFaces:effectiveMipmapLevels:
 *  storageMipmapped: */
int mglTextureCheckCompleteness(void *tex, uint32_t tex_type,
                               unsigned num_faces,
                               unsigned int *outEffectiveMipmapLevels,
                               int *outStorageMipmapped)
{
    Texture *texture = (Texture *)tex;
    (void)tex_type; /* unused: completeness does not depend on Metal type */

    unsigned int effective_mipmap_levels = texture->mipmap_levels;
    int storageMipmapped = 0;

    const uint64_t completeness_check_faces = mglRenderCompletenessCheckFaces(
        (uint32_t)texture->target, (uint32_t)num_faces);

    /* Texture storage is independent from GL_TEXTURE_MAX_LEVEL.  Minecraft uses
     * BASE/MAX_LEVEL to express temporary GpuTextureView mip windows; if those
     * sampler parameters shrink the Metal texture allocation, later full-atlas
     * sampling loses the higher mip levels and distant terrain reads
     * empty/incorrect data.  Apply BASE/MAX only to completeness checks and
     * sampled Metal views, not to the underlying storage level count. */

    /* For CUBE_MAP_ARRAY, glTexImage3D stores all layer data in faces[0] with
     * depth = 6 * num_cubes.  Faces 1-5 are never populated by
     * createTextureLevel, so only check face 0 for completeness.  The upload
     * code also reads from face 0 and distributes slices to Metal array
     * layers. */

    storageMipmapped = (texture->mipmap_levels > 1u) &&
                       (texture->num_levels > 1u || texture->is_render_target);

    if (texture->num_levels > 1) {
        /* mipmapped texture */
        if (effective_mipmap_levels == 0) {
            effective_mipmap_levels = texture->num_levels;
        }

        /* Cap Metal storage to populated GL levels for both sampled and RT
         * textures. Skipping this for is_render_target left capacity-sized
         * chains (e.g. mipmap_levels=11 with num_levels=2) uninitialized above
         * the upload window; sampled-copy only Y-flips num_levels, so a wrong
         * MAX_LEVEL/view would sample empty high mips (MC blocks atlas). */
        if (texture->num_levels > 0u &&
            texture->num_levels < effective_mipmap_levels) {
            static uint64_t s_mipmap_count_mismatch_logs = 0;
            if (++s_mipmap_count_mismatch_logs <= 8 ||
                (s_mipmap_count_mismatch_logs % 2048) == 0) {
                fprintf(stderr,
                        "MGL TEXTURE MIP COMPAT: tex=%u target=0x%x size=%ux%u num_levels=%u mipmap_levels=%u effective=%u base=%u max=%u immutable=%u isRT=%u; capping Metal mip count to uploaded levels hit=%llu\n",
                        texture->name, texture->target, texture->width,
                        texture->height, texture->num_levels,
                        texture->mipmap_levels, effective_mipmap_levels,
                        texture->params.base_level, texture->params.max_level,
                        texture->immutable_storage,
                        texture->is_render_target,
                        (unsigned long long)s_mipmap_count_mismatch_logs);
            }
            effective_mipmap_levels = texture->num_levels;
        }

        /* GL texture completeness only requires levels in
         * [base_level, min(max_level, mipmap_levels-1)] to be complete.  Levels
         * below base_level may be uninitialised and must NOT cause the texture
         * to be rejected.  Minecraft 1.21.11 sets base_level>0 on mipmap
         * texture views (GlCommandEncoder.java). */
        const uint32_t check_start = texture->params.base_level;
        uint32_t check_end = (texture->params.max_level == 1000u)
                                 ? (texture->mipmap_levels > 0u
                                        ? texture->mipmap_levels - 1u
                                        : 0u)
                                 : texture->params.max_level;
        if (check_end >= texture->mipmap_levels) {
            check_end =
                (texture->mipmap_levels > 0u) ? texture->mipmap_levels - 1u : 0u;
        }
        /* Storage only needs the levels it allocates; whether the chain is
         * complete enough to sample is decided per draw (§8.17). */
        if (check_end >= effective_mipmap_levels) {
            check_end = effective_mipmap_levels - 1u;
        }
        if (check_end < check_start) check_end = check_start;

        for (int face = 0; face < (int)completeness_check_faces; face++) {
            for (uint32_t i = check_start; i <= check_end; i++) {
                /* incomplete texture */
                if (texture->faces[face].levels[i].complete == false) {
                    static uint64_t s_incomplete_mip_logs = 0;
                    if (++s_incomplete_mip_logs <= 32 ||
                        (s_incomplete_mip_logs % 512) == 0) {
                        fprintf(stderr,
                                "MGL TEXTURE INCOMPLETE: tex=%u target=0x%x face=%d level=%u incomplete num_levels=%u mipmap_levels=%u effective=%u base=%u max=%u check=[%u,%u] hit=%llu\n",
                                texture->name, texture->target, face, i,
                                texture->num_levels, texture->mipmap_levels,
                                effective_mipmap_levels,
                                texture->params.base_level,
                                texture->params.max_level, check_start,
                                check_end,
                                (unsigned long long)s_incomplete_mip_logs);
                    }
                    return 0;
                }
            }
        }

        texture->mipmapped = true;
    } else if (texture->num_levels == 1) {
        if (!storageMipmapped) {
            effective_mipmap_levels = 1;
        }
        /* single level texture / incomplete texture */
        for (int face = 0; face < (int)completeness_check_faces; face++) {
            if (texture->faces[face].levels[0].complete == false) {
                static uint64_t s_incomplete_base_logs = 0;
                if (++s_incomplete_base_logs <= 32 ||
                    (s_incomplete_base_logs % 512) == 0) {
                    fprintf(stderr,
                            "MGL TEXTURE INCOMPLETE: tex=%u target=0x%x face=%d base incomplete size=%ux%u hit=%llu\n",
                            texture->name, texture->target, face, texture->width,
                            texture->height,
                            (unsigned long long)s_incomplete_base_logs);
                }
                return 0;
            }
        }
    } else {
        fprintf(stderr,
                "MGL TEXTURE ERROR: texture %u has no complete levels for Metal creation target=0x%x\n",
                texture->name, texture->target);
        return 0;
    }

    texture->complete = true;

    if (outEffectiveMipmapLevels) {
        *outEffectiveMipmapLevels = effective_mipmap_levels;
    }
    if (outStorageMipmapped) *outStorageMipmapped = storageMipmapped;
    return 1;
}

/* -createMTLTexelBufferTexture:. */
typedef struct MglPdTexelBufferCtx_t {
    MGLRenderTextureDescriptorState descriptor;
    void *metal_buffer;
    uint64_t offset;
    uint64_t bytes_per_row;
    void *created;
} MglPdTexelBufferCtx;

/* @try of MTLBuffer→TextureBuffer; the catch logs and the caller returns nil. */
static int mglPdTexelBufferTryBody(void *renderer, void *rawCtx)
{
    MglPdTexelBufferCtx *ctx = (MglPdTexelBufferCtx *)rawCtx;
    (void)renderer;
    if (mglRenderCreateBufferTextureFromState(
            ctx->metal_buffer, &ctx->descriptor, ctx->offset,
            ctx->bytes_per_row, &ctx->created) != 0) {
        ctx->created = NULL;
        return 0;
    }
    return ctx->created != NULL ? 1 : 0;
}

void *mglTextureCreateMTLTexelBufferTexture(void *renderer, void *tex)
{
    Texture *texture = (Texture *)tex;

    Buffer *sourceBuffer = texture->texture_buffer;
    if (!sourceBuffer || texture->texture_buffer_size <= 0) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: tex=%u has no attached buffer/size buffer=%p size=%lld\n",
                texture->name, (void *)sourceBuffer,
                (long long)texture->texture_buffer_size);
        return NULL;
    }

    if (texture->texture_buffer_offset < 0 ||
        texture->texture_buffer_offset > sourceBuffer->size ||
        texture->texture_buffer_size >
            sourceBuffer->size - texture->texture_buffer_offset) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: invalid range tex=%u buffer=%u off=%lld size=%lld bufferSize=%lld\n",
                texture->name, sourceBuffer->name,
                (long long)texture->texture_buffer_offset,
                (long long)texture->texture_buffer_size,
                (long long)sourceBuffer->size);
        return NULL;
    }

    const uint64_t bytesPerTexel =
        mglTextureBytesPerPixelForFormat(texture->internalformat);
    if (bytesPerTexel == 0) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: unsupported internal format 0x%x tex=%u buffer=%u\n",
                texture->internalformat, texture->name, sourceBuffer->name);
        return NULL;
    }

    const uint64_t texelCount =
        (uint64_t)texture->texture_buffer_size / bytesPerTexel;
    if (texelCount == 0) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: zero texel count tex=%u buffer=%u size=%lld bpt=%lu\n",
                texture->name, sourceBuffer->name,
                (long long)texture->texture_buffer_size,
                (unsigned long)bytesPerTexel);
        return NULL;
    }

    /* GL_RGBA8 is normalized (float-sampleable).  Forcing RGBA8Uint here made
     * samplerBuffer + layout(binding) CTS reject the real texture as
     * actualKind=uint vs expectedKind=float and substitute a 1x1 fallback. */
    const uint32_t bufferPixelFormat = mtlPixelFormatForGLTex(texture);
    if (mglRenderColorFormatNeedsFallback(bufferPixelFormat)) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: invalid Metal format for tex=%u internal=0x%x\n",
                texture->name, texture->internalformat);
        return NULL;
    }

    if (!mglRendererProcessBuffer(renderer, sourceBuffer)) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: failed to process source buffer tex=%u buffer=%u\n",
                texture->name, sourceBuffer->name);
        return NULL;
    }

    if (!sourceBuffer->data.mtl_data) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: no Metal buffer for tex=%u buffer=%u\n",
                texture->name, sourceBuffer->name);
        return NULL;
    }

    /* Metal texture_buffer shares storage with the attached MTLBuffer.
     * RGB formats that Metal lacks need a temporary expanded RGBA buffer. */
    void *metalBuffer = sourceBuffer->data.mtl_data;
    uint64_t metalOffset = (uint64_t)texture->texture_buffer_offset;
    uint64_t bytesPerRow = texelCount * bytesPerTexel;
    void *expandedData = NULL;
    void *expandedMetalBuffer = NULL;
    const int needsExpand = mglTextureNeedsChannelExpansion(
        texture->internalformat, bufferPixelFormat);

    if (needsExpand) {
        const uint8_t *sourceBytes = NULL;
        if (sourceBuffer->data.buffer_data) {
            sourceBytes =
                ((const uint8_t *)(uintptr_t)sourceBuffer->data.buffer_data) +
                (size_t)texture->texture_buffer_offset;
        } else {
            void *contents =
                mglPdTextureBufferContents(sourceBuffer->data.mtl_data);
            if (contents) {
                sourceBytes = ((const uint8_t *)contents) +
                              (size_t)texture->texture_buffer_offset;
            }
        }
        if (!sourceBytes) {
            fprintf(stderr,
                    "MGL TEXBUFFER ERROR: no readable backing for RGB expand "
                    "tex=%u buffer=%u\n",
                    texture->name, sourceBuffer->name);
            return NULL;
        }
        uint32_t srcCompU = 0u, dstCompU = 0u;
        uint64_t alphaDefault = 0;
        if (!mglRenderRGBExpandParams(bufferPixelFormat, &srcCompU, &dstCompU,
                                      &alphaDefault)) {
            fprintf(stderr,
                    "MGL TEXBUFFER ERROR: RGB expand params missing tex=%u "
                    "format=%lu\n",
                    texture->name, (unsigned long)bufferPixelFormat);
            return NULL;
        }
        const uint64_t dstPixelBytes = (uint64_t)dstCompU * 4u;
        const uint64_t expandedBytes = texelCount * dstPixelBytes;
        expandedData = calloc(1u, (size_t)expandedBytes);
        if (!expandedData ||
            mglRenderTextureExpandRGBToRGBA(
                sourceBytes, expandedData, texelCount, texelCount, 1u,
                (uint64_t)srcCompU, (uint64_t)dstCompU, alphaDefault) != 0) {
            fprintf(stderr,
                    "MGL TEXBUFFER ERROR: channel expansion failed tex=%u "
                    "buffer=%u\n",
                    texture->name, sourceBuffer->name);
            free(expandedData);
            return NULL;
        }
        /* Shared staging buffer so TextureBuffer can view the expanded texels. */
        enum { MGL_PD_STORAGE_SHARED = 0u };
        if (mglRenderCreateBuffer(expandedBytes, MGL_PD_STORAGE_SHARED, NULL,
                                  &expandedMetalBuffer) != 0 ||
            !expandedMetalBuffer) {
            fprintf(stderr,
                    "MGL TEXBUFFER ERROR: failed creating expand buffer tex=%u "
                    "bytes=%llu\n",
                    texture->name, (unsigned long long)expandedBytes);
            free(expandedData);
            return NULL;
        }
        void *dstContents = mglPdTextureBufferContents(expandedMetalBuffer);
        if (!dstContents) {
            fprintf(stderr,
                    "MGL TEXBUFFER ERROR: expand buffer not CPU-mappable "
                    "tex=%u\n",
                    texture->name);
            mglReleaseMetalObjNoNull(expandedMetalBuffer);
            free(expandedData);
            return NULL;
        }
        memcpy(dstContents, expandedData, (size_t)expandedBytes);
        metalBuffer = expandedMetalBuffer;
        metalOffset = 0u;
        bytesPerRow = expandedBytes;
        free(expandedData);
        expandedData = NULL;
    }

    uint64_t bufferUsage =
        MGL_PD_TEXTURE_USAGE_SHADER_READ | MGL_PD_TEXTURE_USAGE_SHADER_WRITE;
    if (mglRenderPixelFormatNeedsShaderAtomic(bufferPixelFormat)) {
        bufferUsage |= MGL_PD_TEXTURE_USAGE_SHADER_ATOMIC;
    }
    MGLRenderTextureDescriptorState bufferDesc = {
        .texture_type = MGLTextureTypeTextureBuffer,
        .pixel_format = bufferPixelFormat,
        .width = texelCount,
        .height = 1u,
        .depth = 1u,
        .mipmap_level_count = 1u,
        .sample_count = 1u,
        .array_length = 1u,
        .usage = bufferUsage,
    };

    MglPdTexelBufferCtx tryCtx = {bufferDesc, metalBuffer, metalOffset,
                                  bytesPerRow, NULL};
    void *bufferTexture = NULL;
    if (mglPlatformShellGuardedCallCtx(renderer, "texel buffer texture creation",
                                       mglPdTexelBufferTryBody, &tryCtx,
                                       NULL)) {
        bufferTexture = tryCtx.created;
    } else {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: failed creating TextureBuffer tex=%u "
                "buffer=%u texels=%lu\n",
                texture->name, sourceBuffer->name, (unsigned long)texelCount);
        if (tryCtx.created) mglReleaseMetalObjNoNull(tryCtx.created);
        if (expandedMetalBuffer) mglReleaseMetalObjNoNull(expandedMetalBuffer);
        return NULL;
    }

    /* TextureBuffer retains the MTLBuffer; drop the staging +1. */
    if (expandedMetalBuffer) mglReleaseMetalObjNoNull(expandedMetalBuffer);

    if (!bufferTexture) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: Metal TextureBuffer nil tex=%u buffer=%u "
                "format=%lu texels=%lu\n",
                texture->name, sourceBuffer->name,
                (unsigned long)bufferPixelFormat, (unsigned long)texelCount);
        return NULL;
    }

    texture->dirty_bits = 0;
    sourceBuffer->data.dirty_bits = 0;

    {
        static uint64_t s_texBufferCreateLogs = 0;
        const uint64_t hit = ++s_texBufferCreateLogs;
        if (hit <= 2ull || (hit % 4096ull) == 0ull) {
            fprintf(stderr,
                    "MGL TEXBUFFER CREATE tex=%u buffer=%u internal=0x%x "
                    "mtlFormat=%lu texels=%lu rowBytes=%lu bytes=%lld "
                    "offset=%lld as=texture_buffer expand=%d\n",
                    texture->name, sourceBuffer->name, texture->internalformat,
                    (unsigned long)bufferPixelFormat, (unsigned long)texelCount,
                    (unsigned long)bytesPerRow,
                    (long long)texture->texture_buffer_size,
                    (long long)texture->texture_buffer_offset, needsExpand);
        }
    }

    mglRendererRecordGPUSuccess(renderer);
    return bufferTexture;
}
