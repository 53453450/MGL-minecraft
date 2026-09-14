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

static const uint64_t kMglPdDepthStencilUploadRowAlignment = 256u;

static uint64_t mglPdMinU64(uint64_t a, uint64_t b) { return a < b ? a : b; }

/* Twin of the .m's mglTextureInfo. */
static MGLRenderTextureInfo mglPdTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) (void)mglRenderGetTextureInfo(texture, &info);
    return info;
}

/* Twin of the .m's mglTextureCreateTexture (returns the +1 handle). */
static void *mglPdTextureCreateTexture(
    const MGLRenderTextureDescriptorState *descriptor)
{
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(descriptor, NULL, &texture) == 0 &&
        texture) {
        return texture;
    }
    return NULL;
}

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

/* Twin of the .m's mglTextureGetBytes (same no-raise convention). */
static int mglPdTextureGetBytes(void *texture, void *bytes,
                                uint64_t bytesPerRow, uint64_t bytesPerImage,
                                MGLRegionValue region, uint64_t level,
                                uint64_t slice, int useSlice)
{
    return mglRenderTextureGetBytes(
               texture, bytes, bytesPerRow, bytesPerImage, region.origin.x,
               region.origin.y, region.origin.z, region.size.width,
               region.size.height, region.size.depth, level, slice,
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

static uint64_t mglPdDepthStencilAlignedBytesPerRow(uint64_t logicalBytesPerRow)
{
    if (logicalBytesPerRow == 0) {
        return 0;
    }
    return ((logicalBytesPerRow + kMglPdDepthStencilUploadRowAlignment - 1u) /
            kMglPdDepthStencilUploadRowAlignment) *
           kMglPdDepthStencilUploadRowAlignment;
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

/* -uploadPackedDepthStencilStencilPlane:texName:bytes:width:height:
 *  bytesPerRow:level:slice:xorigin:yorigin: */
int mglTextureUploadPackedDepthStencilStencilPlane(
    void *texture, unsigned int texName, const void *packedBytes, uint64_t width,
    uint64_t height, uint64_t bytesPerRow, uint64_t level, uint64_t slice,
    uint64_t xorigin, uint64_t yorigin)
{
    if (!texture || !packedBytes || width == 0 || height == 0) {
        return 0;
    }
    const uint32_t parentFormat =
        (uint32_t)mglPdTextureInfo(texture).pixel_format;
    if (!mglRenderPixelFormatIsPackedDepthStencil(parentFormat)) {
        return 0;
    }

    void *metalUpload = NULL;
    const void *srcBytes = packedBytes;
    uint64_t srcBytesPerRow = bytesPerRow;
    if (bytesPerRow >= width * 8u) {
        /* Already Metal packed layout. */
    } else if (mglRenderPackedD32FNeeds8ByteStride(
                   parentFormat, (uint32_t)bytesPerRow, (uint32_t)width)) {
        srcBytesPerRow = width * 8u;
        const uint64_t repackBytes = srcBytesPerRow * height;
        metalUpload = calloc(1u, repackBytes);
        if (!metalUpload) return 0;
        const uint8_t *srcBase = (const uint8_t *)packedBytes;
        uint8_t *dstBase = (uint8_t *)metalUpload;
        for (uint64_t y = 0; y < height; ++y) {
            const uint8_t *srcRow = srcBase + y * bytesPerRow;
            uint8_t *dstRow = dstBase + y * srcBytesPerRow;
            for (uint64_t x = 0; x < width; ++x) {
                memcpy(dstRow + x * 8u, srcRow + x * 5u, 4u);
                dstRow[x * 8u + 4u] = srcRow[x * 5u + 4u];
            }
        }
        srcBytes = metalUpload;
    } else {
        return 0;
    }

    const uint64_t logicalStencilBytesPerRow = width;
    const uint64_t stencilBytesPerRow =
        mglPdDepthStencilAlignedBytesPerRow(logicalStencilBytesPerRow);
    if (stencilBytesPerRow == 0) {
        free(metalUpload);
        return 0;
    }
    const uint64_t stencilBytesPerImage = stencilBytesPerRow * height;
    uint8_t *stencilBytes = (uint8_t *)calloc(1u, stencilBytesPerImage);
    if (!stencilBytes) {
        free(metalUpload);
        return 0;
    }

    const uint8_t *srcBase = (const uint8_t *)srcBytes;
    for (uint64_t y = 0; y < height; ++y) {
        const uint8_t *srcRow = srcBase + y * srcBytesPerRow;
        uint8_t *dstRow = stencilBytes + y * stencilBytesPerRow;
        for (uint64_t x = 0; x < width; ++x) {
            dstRow[x] = srcRow[x * 8u + 4u];
        }
    }
    free(metalUpload);

    void *stencilViewRaw = NULL;
    const uint32_t viewType = mglRenderDepthStencilPlaneViewType(
        (uint32_t)mglPdTextureInfo(texture).texture_type);
    int uploaded = 0;
    const uint32_t stencilViewFormat = mglRenderStencilViewFormat(parentFormat);
    if (mglRenderCreateTextureViewRange(
            texture, stencilViewFormat, viewType, level, 1u, slice, 1u, 0, 0, 0,
            0, 0, &stencilViewRaw) == 0 &&
        stencilViewRaw) {
        /* The .m wrapped the upload in @try/@catch and only logged the failure;
         * the C twin reports it instead (see mglPdTextureReplaceRegion). */
        uploaded = mglTextureReplaceRegionValue(
            stencilViewRaw, mglTextureRegion2D(xorigin, yorigin, width, height),
            0u, 0u, stencilBytes, stencilBytesPerRow, stencilBytesPerImage, 0);
        mglReleaseMetalObjNoNull(stencilViewRaw);
    }
    free(stencilBytes);
    if (!uploaded) {
        fprintf(stderr,
                "MGL WARNING: depth/stencil stencil-plane blit upload failed tex=%u slice=%lu\n",
                (unsigned)texName, (unsigned long)slice);
    }
    return uploaded;
}

/* -createMTLTexelBufferTexture:. */
typedef struct MglPdTexelBufferCtx_t {
    MGLRenderTextureDescriptorState descriptor;
    void *upload_bytes;
    uint64_t bytes_per_row;
    uint64_t tex_width;
    uint64_t tex_height;
    void *created;
    int result;
} MglPdTexelBufferCtx;

/* @try of the create+upload; the catch logs and the caller returns nil. */
static int mglPdTexelBufferTryBody(void *renderer, void *rawCtx)
{
    MglPdTexelBufferCtx *ctx = (MglPdTexelBufferCtx *)rawCtx;
    (void)renderer;
    ctx->created = mglPdTextureCreateTexture(&ctx->descriptor);
    if (ctx->created) {
        if (!mglTextureReplaceRegionValue(
                ctx->created,
                mglTextureRegion2D(0, 0, ctx->tex_width, ctx->tex_height), 0, 0,
                ctx->upload_bytes, ctx->bytes_per_row, 0, 0)) {
            return 0;
        }
    }
    ctx->result = 1;
    return 1;
}

void *mglTextureCreateMTLTexelBufferTexture(void *renderer, void *tex)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    Texture *texture = (Texture *)tex;
    GLMState *glState = ctx ? (areas.core && areas.core->activeState
                                   ? areas.core->activeState
                                   : ctx->active_state)
                            : NULL;

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

    const uint8_t *sourceBytes = NULL;
    if (sourceBuffer->data.buffer_data) {
        sourceBytes = ((const uint8_t *)(uintptr_t)sourceBuffer->data.buffer_data) +
                      (size_t)texture->texture_buffer_offset;
    } else if (sourceBuffer->data.mtl_data) {
        void *contents = mglPdTextureBufferContents(sourceBuffer->data.mtl_data);
        if (contents) {
            sourceBytes = ((const uint8_t *)contents) +
                          (size_t)texture->texture_buffer_offset;
        }
    }

    if (!sourceBytes) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: no readable backing for tex=%u buffer=%u cpu=%p mtl=%p\n",
                texture->name, sourceBuffer->name,
                (void *)(uintptr_t)sourceBuffer->data.buffer_data,
                sourceBuffer->data.mtl_data);
        return NULL;
    }

    /* The AIR backend emits Minecraft's CloudFaces texel buffer as a
     * texture2d<int>. Keep GL lookup semantics as GL_TEXTURE_BUFFER, but create
     * a Metal 2D backing so the generated MSL argument type matches.  A texel
     * buffer can be much wider than Metal's max 2D texture width, so pack it
     * into rows instead of creating texelCount x 1.
     *
     * The AIR backend lowers GL texture buffers to 2D Metal textures and emits
     * spvTexelBufferCoord(tc) using its MSL texel_buffer_texture_width option.
     * Keep this packing width in lockstep with program.c. */
    uint32_t packedW = 0u;
    uint32_t packedH = 0u;
    if (!mglRenderPlanTexelBuffer2DSize(
            texelCount,
            glState ? glState->var.max_texture_size : 4096u, &packedW,
            &packedH)) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: texel buffer too large for 2D fallback tex=%u buffer=%u texels=%lu max=%u\n",
                texture->name, sourceBuffer->name, (unsigned long)texelCount,
                glState ? glState->var.max_texture_size : 4096u);
        return NULL;
    }
    const uint64_t texWidth = packedW;
    const uint64_t texHeight = packedH;

    uint64_t bytesPerRow = texWidth * bytesPerTexel;
    uint64_t packedBytes = bytesPerRow * texHeight;
    void *packedData = NULL;
    const uint8_t *uploadBytes = sourceBytes;

    /* Channel expansion for 3-channel RGB -> 4-channel RGBA Metal formats.
     * GL_RGB32* (12 bytes/texel) maps to Metal RGBA32* (16 bytes/texel).  Expand
     * each texel by inserting a default alpha before uploading. */
    void *expandedData = NULL;
    if (mglTextureNeedsChannelExpansion(texture->internalformat,
                                        bufferPixelFormat)) {
        uint32_t srcCompU = 0u, dstCompU = 0u;
        uint64_t alphaDefault = 0;
        if (mglRenderRGBExpandParams(bufferPixelFormat, &srcCompU, &dstCompU,
                                     &alphaDefault)) {
            const uint64_t srcCompBytes = srcCompU;
            const uint64_t dstCompBytes = dstCompU;
            const uint64_t dstPixelBytes = dstCompBytes * 4;
            const uint64_t expandedBytesPerRow = texWidth * dstPixelBytes;
            const uint64_t expandedPackedBytes = expandedBytesPerRow * texHeight;
            expandedData = calloc(1u, expandedPackedBytes);
            if (expandedData) {
                if (mglRenderTextureExpandRGBToRGBA(
                        sourceBytes, expandedData, texelCount, texWidth,
                        texHeight, srcCompBytes, dstCompBytes,
                        alphaDefault) != 0) {
                    fprintf(stderr,
                            "MGL TEXBUFFER ERROR: channel expansion failed tex=%u buffer=%u\n",
                            texture->name, sourceBuffer->name);
                    free(expandedData);
                    return NULL;
                }
                uploadBytes = (const uint8_t *)expandedData;
                bytesPerRow = expandedBytesPerRow;
                packedBytes = expandedPackedBytes;
            }
        }
    }

    if (texHeight > 1 && !expandedData) {
        packedData = calloc(1u, packedBytes);
        if (!packedData) {
            fprintf(stderr,
                    "MGL TEXBUFFER ERROR: failed allocating packed data tex=%u buffer=%u bytes=%lu\n",
                    texture->name, sourceBuffer->name,
                    (unsigned long)packedBytes);
            return NULL;
        }

        memcpy(packedData, sourceBytes, (size_t)texture->texture_buffer_size);
        uploadBytes = (const uint8_t *)packedData;
    }

    const uint64_t sourceHash =
        mglTraceHashBytes(sourceBytes, (size_t)texture->texture_buffer_size);
    const uint64_t uploadHash = mglTraceHashBytes(uploadBytes, packedBytes);
    char sourceHead[64];
    char uploadHead[64];
    sourceHead[0] = '\0';
    uploadHead[0] = '\0';
    mglTraceFormatBytes(sourceBytes,
                        (size_t)mglPdMinU64(
                            (uint64_t)texture->texture_buffer_size, 64u),
                        sourceHead, sizeof(sourceHead));
    mglTraceFormatBytes(uploadBytes, (size_t)mglPdMinU64(packedBytes, 64u),
                        uploadHead, sizeof(uploadHead));

    uint64_t bufferUsage =
        MGL_PD_TEXTURE_USAGE_SHADER_READ | MGL_PD_TEXTURE_USAGE_SHADER_WRITE;
    /* imageAtomic* on iimageBuffer needs ShaderAtomic (R32I/R32UI). */
    if (mglRenderPixelFormatNeedsShaderAtomic(bufferPixelFormat)) {
        bufferUsage |= MGL_PD_TEXTURE_USAGE_SHADER_ATOMIC;
    }
    MGLRenderTextureDescriptorState bufferDesc = {
        .texture_type = MGLTextureType2D,
        .pixel_format = bufferPixelFormat,
        .width = texWidth,
        .height = texHeight,
        .depth = 1u,
        .mipmap_level_count = 1u,
        .sample_count = 1u,
        .array_length = 1u,
        /* imageStore requires ShaderWrite; sampling still needs ShaderRead. */
        .usage = bufferUsage,
    };

    MglPdTexelBufferCtx tryCtx = {bufferDesc, (void *)uploadBytes, bytesPerRow,
                                  texWidth,     texHeight,        NULL,
                                  0};
    void *bufferTexture = NULL;
    if (mglPlatformShellGuardedCallCtx(renderer, "texel buffer texture creation",
                                       mglPdTexelBufferTryBody, &tryCtx,
                                       NULL)) {
        bufferTexture = tryCtx.created;
    } else {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: failed creating/uploading tex=%u buffer=%u\n",
                texture->name, sourceBuffer->name);
        if (tryCtx.created) mglReleaseMetalObjNoNull(tryCtx.created);
        free(expandedData);
        free(packedData);
        return NULL;
    }

    if (!bufferTexture) {
        fprintf(stderr,
                "MGL TEXBUFFER ERROR: Metal texture creation returned nil tex=%u buffer=%u format=%lu texels=%lu\n",
                texture->name, sourceBuffer->name,
                (unsigned long)bufferPixelFormat, (unsigned long)texelCount);
        free(expandedData);
        free(packedData);
        return NULL;
    }

    texture->dirty_bits = 0;
    sourceBuffer->data.dirty_bits = 0;

    void *readbackData = calloc(1u, packedBytes);
    uint64_t readbackHash = 0ull;
    char readbackHead[64];
    readbackHead[0] = '\0';
    if (readbackData) {
        if (mglPdTextureGetBytes(bufferTexture, readbackData, bytesPerRow, 0,
                                 mglTextureRegion2D(0, 0, texWidth, texHeight),
                                 0, 0, 0)) {
            readbackHash = mglTraceHashBytes(readbackData, packedBytes);
            mglTraceFormatBytes(readbackData,
                                (size_t)mglPdMinU64(packedBytes, 64u),
                                readbackHead, sizeof(readbackHead));
        }
    }

    {
        static uint64_t s_texBufferCreateLogs = 0;
        const uint64_t hit = ++s_texBufferCreateLogs;
        if (hit <= 2ull || (hit % 4096ull) == 0ull) {
            fprintf(stderr,
                    "MGL TEXBUFFER CREATE tex=%u buffer=%u internal=0x%x mtlFormat=%lu texels=%lu packed=%lux%lu rowBytes=%lu bytes=%lld offset=%lld as=texture2d sourceHash=0x%016llx uploadHash=0x%016llx readbackHash=0x%016llx sourceHead=%s uploadHead=%s readbackHead=%s\n",
                    texture->name, sourceBuffer->name, texture->internalformat,
                    (unsigned long)bufferPixelFormat, (unsigned long)texelCount,
                    (unsigned long)texWidth, (unsigned long)texHeight,
                    (unsigned long)bytesPerRow,
                    (long long)texture->texture_buffer_size,
                    (long long)texture->texture_buffer_offset,
                    (unsigned long long)sourceHash,
                    (unsigned long long)uploadHash,
                    (unsigned long long)readbackHash, sourceHead, uploadHead,
                    readbackHead);
        }
    }

    free(readbackData);
    free(expandedData);
    free(packedData);
    mglRendererRecordGPUSuccess(renderer);
    return bufferTexture;
}
