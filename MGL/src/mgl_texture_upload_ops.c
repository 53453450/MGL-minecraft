/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_upload_ops.c - C homes of -uploadTextureSliceViaBlit: and
 * -copyTextureUploadWithDedicatedCommandBuffer: (P0-1, log 183).
 */

#include "mgl_texture_compat.h"
#include "mgl_texture_upload_ops.h"

#include "mgl_capability.h"        /* MGLCapabilityHasBug */
#include "mgl_gpu_recovery.h"      /* mglRendererShouldSkipGPUOperations */
#include "mgl_pso_format_class.h"  /* mglRenderPixelFormatIsPackedDepthStencil */
#include "mgl_readback_policy.h"   /* mglRenderTextureRepackDepthPlanes */
#include "mgl_renderer_backend.h"
#include "mgl_renderer_ports.h"
#include "mgl_texture_create_ops.h" /* mglTextureReplaceRegionValue */
#include "mgl_render_pass_manager_ops.h" /* ensure-writable-command-buffer */
#include "pixel_utils.h"               /* MGLPixelFormatR8Unorm */
#include "mgl_thread_affinity.h"        /* MGL_ASSERT_GL_THREAD */
#include "mgl_trace_log.h"

#include "mgl_render.h"

#include <dispatch/dispatch.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The .m's file-local constants this TU needs (values copied verbatim). */
enum {
    MGL_PD_TEXTURE_STORAGE_PRIVATE = 2u,
    MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED = 0u,
};
/* MGLRenderer+Texture_Private.h: static const BOOL / NSTimeInterval. */
static const int kMglPdSynchronizeTextureUploads = 0;
static const double kMglPdTextureUploadWaitTimeoutSeconds = 0.25;
static const int kMglPdUseDedicatedTextureUploadCommandBuffer = 0;

/* Twin of the .m's mglTextureInfo. */
static MGLRenderTextureInfo mglPdTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) (void)mglRenderGetTextureInfo(texture, &info);
    return info;
}

/* The Objective-C header's mglMarkTextureLevelMetalFilled is a static inline
 * there; this is the C twin (same shape as mgl_blit_drivers.c's). */
extern void mglMarkGLSampledCopyLevelDirty(Texture *tex, GLuint level);
extern bool mglRendererBindMTLTexture(void *renderer, Texture *tex);

static void mglUpMarkTextureLevelMetalFilled(Texture *tex, GLuint level,
                                             size_t upload_size)
{
    TextureLevel *tex_level = mglTextureAttachmentLevel(tex, level);
    if (!tex_level) {
        return;
    }
    mglRenderMarkTextureLevelWritten(&tex_level->ever_written,
                                     &tex_level->has_initialized_data,
                                     &tex_level->suspicious_zero_upload);
    tex_level->last_init_source = kTexMetalFill;
    tex_level->last_upload_size = upload_size;
    tex_level->last_src_ptr = NULL;
    tex_level->last_src_hash = 0ull;
    if (tex->is_render_target) {
        tex->mtl_render_target_write_version++;
        mglMarkGLSampledCopyLevelDirty(tex, level);
    }
}

/* mglUpMax() is an Objective-C header macro; use the C maximum inline. */
#define mglUpMax(a, b) ((a) > (b) ? (a) : (b))

/* Twin of the .m's mglDepthStencilAlignedBytesPerRow (the row alignment
 * constant lives in the Objective-C header). */
#define kMglUpDepthStencilUploadRowAlignment 256u

static uint64_t mglUpDepthStencilAlignedBytesPerRow(uint64_t logicalBytesPerRow)
{
    if (logicalBytesPerRow == 0) {
        return 0;
    }
    return ((logicalBytesPerRow + kMglUpDepthStencilUploadRowAlignment - 1u) /
            kMglUpDepthStencilUploadRowAlignment) *
           kMglUpDepthStencilUploadRowAlignment;
}

/* Twin of the .m's mglCreateDepthStencilMetalUpload (same unpack loop). */
static void *mglUpCreateDepthStencilMetalUpload(
    Texture *tex, uint32_t pixelFormat, const uint8_t *src, uint64_t width,
    uint64_t height, uint64_t srcBytesPerRow, uint64_t *outBytesPerRow,
    uint64_t *outBytesPerImage)
{
    if (outBytesPerRow) *outBytesPerRow = 0;
    if (outBytesPerImage) *outBytesPerImage = 0;
    if (!tex || !src || width == 0 || height == 0 || srcBytesPerRow == 0 ||
        !mglRenderDepth32FStencil8NeedsUnpack((uint32_t)tex->internalformat,
                                              (uint32_t)pixelFormat,
                                              (uint32_t)srcBytesPerRow,
                                              (uint32_t)width)) {
        return NULL;
    }
    uint64_t logicalBytesPerRow = width * 8u;
    uint64_t dstBytesPerRow =
        mglUpDepthStencilAlignedBytesPerRow(logicalBytesPerRow);
    if (dstBytesPerRow == 0) {
        return NULL;
    }
    uint64_t dstBytesPerImage = dstBytesPerRow * height;
    uint8_t *dst = calloc(1u, (size_t)dstBytesPerImage);
    if (!dst) return NULL;
    for (uint64_t y = 0; y < height; ++y) {
        const uint8_t *srcRow = src + y * srcBytesPerRow;
        uint8_t *dstRow = dst + y * dstBytesPerRow;
        for (uint64_t x = 0; x < width; ++x) {
            memcpy(dstRow + x * 8u, srcRow + x * 5u, 4u);
            dstRow[x * 8u + 4u] = srcRow[x * 5u + 4u];
        }
    }
    if (outBytesPerRow) *outBytesPerRow = dstBytesPerRow;
    if (outBytesPerImage) *outBytesPerImage = dstBytesPerImage;
    return dst;
}

/* mglUpMin() is an Objective-C header macro; C has no same-named inline. */
#define mglUpMin(a, b) ((a) < (b) ? (a) : (b))
#define mglUpMax(a, b) ((a) > (b) ? (a) : (b))

/* Twin of the .m's mglTextureCreateBufferWithBytes (returns the +1 handle). */
static void *mglUpCreateBufferWithBytes(const void *bytes, uint64_t length,
                                        uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBufferWithBytes(bytes, (size_t)length, options, NULL,
                                       &buffer) == 0 && buffer) {
        return buffer;
    }
    return NULL;
}


typedef struct MglPdUploadCompletionCtx_t {
    void *renderer;
    const char *reason;
    int *upload_error;
    dispatch_semaphore_t semaphore;
} MglPdUploadCompletionCtx;

static void mglPdUploadCompletionDestroy(void *context) { free(context); }

static void mglPdUploadCompletionBody(
    void *context, const MGLRenderCommandBufferState *uploadState)
{
    MglPdUploadCompletionCtx *ctx = (MglPdUploadCompletionCtx *)context;
    if (!ctx) return;
    if (uploadState && uploadState->has_error) {
        if (ctx->upload_error) *ctx->upload_error = 1;
        fprintf(stderr,
                "MGL ERROR: dedicated upload command buffer failed (%s): %s\n",
                ctx->reason ? ctx->reason : "texture_upload",
                mglRenderCommandBufferErrorDescription(uploadState));
        mglRendererRecordGPUError(ctx->renderer);
    }
    if (ctx->semaphore) {
        dispatch_semaphore_signal(ctx->semaphore);
    }
}

/* -copyTextureUploadWithDedicatedCommandBuffer:... */
int mglTextureCopyUploadWithDedicatedCommandBuffer(
    void *renderer, void *sourceBuffer, uint64_t sourceOffset,
    uint64_t sourceBytesPerRow, uint64_t sourceBytesPerImage,
    uint64_t sourceLayerStride, uint64_t layerCount, MGLSizeValue sourceSize,
    void *texture, uint64_t destinationSlice, uint64_t destinationLevel,
    MGLOriginValue destinationOrigin, const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *commandState = areas.command;

    MGL_ASSERT_GL_THREAD();
    if (!sourceBuffer || !texture ||
        !mglRendererBackendGetCommandQueue(areas.backend) || layerCount == 0u ||
        sourceBytesPerRow == 0u || sourceBytesPerImage == 0u ||
        sourceSize.width == 0u || sourceSize.height == 0u ||
        sourceSize.depth == 0u ||
        (layerCount > 1u && sourceLayerStride == 0u)) {
        fprintf(stderr,
                "MGL ERROR: dedicated texture upload prerequisites missing (source=%p texture=%p queue=%p)\n",
                sourceBuffer, texture,
                mglRendererBackendGetCommandQueue(areas.backend));
        return 0;
    }

    if (!kMglPdUseDedicatedTextureUploadCommandBuffer) {
        /* Texture uploads are GL commands and must stay ordered with draws in
         * the same context.  Committing a standalone upload command buffer here
         * can leapfrog an open render command buffer, so encode the blit into
         * the current command buffer after closing the active render encoder. */
        mglRendererEndRenderEncodingLocked(renderer);

        if (!mglRenderPassEnsureWritableCommandBufferLocked(
                renderer, reason ? reason : "texture_upload")) {
            fprintf(stderr,
                    "MGL ERROR: failed to obtain current command buffer for %s\n",
                    reason ? reason : "texture_upload");
            return 0;
        }

        if (mglRenderEncodeTextureUploadLayersForCommandBufferOwner(
                commandState->currentCommandBufferOwner, sourceBuffer,
                sourceOffset, sourceBytesPerRow, sourceBytesPerImage,
                sourceLayerStride, sourceSize.width, sourceSize.height,
                sourceSize.depth, texture, destinationSlice, layerCount,
                destinationLevel, destinationOrigin.x, destinationOrigin.y,
                destinationOrigin.z) != 0) {
            fprintf(stderr, "MGL ERROR: C++ ordered upload encode failed (%s)\n",
                    reason ? reason : "texture_upload");
            mglRendererRecordGPUError(renderer);
            return 0;
        }

        return 1;
    }

    void *uploadCB = NULL;
    if (mglRenderCreateCommandBuffer(
            mglRendererBackendGetCommandQueue(areas.backend), &uploadCB) != 0 ||
        !uploadCB) {
        fprintf(stderr,
                "MGL ERROR: failed to create dedicated upload command buffer for %s\n",
                reason ? reason : "texture_upload");
        mglRendererRecordGPUError(renderer);
        return 0;
    }

    if (reason) {
        char label[256];
        snprintf(label, sizeof(label), "MGL.%s", reason);
        (void)mglRenderSetCommandBufferLabel(uploadCB, label);
    } else {
        (void)mglRenderSetCommandBufferLabel(uploadCB, "MGL.texture_upload");
    }

    if (mglRenderEncodeTextureUploadLayers(
            uploadCB, sourceBuffer, sourceOffset, sourceBytesPerRow,
            sourceBytesPerImage, sourceLayerStride, sourceSize.width,
            sourceSize.height, sourceSize.depth, texture, destinationSlice,
            layerCount, destinationLevel, destinationOrigin.x,
            destinationOrigin.y, destinationOrigin.z) != 0) {
        fprintf(stderr, "MGL ERROR: C++ dedicated upload encode failed (%s)\n",
                reason ? reason : "texture_upload");
        mglRendererRecordGPUError(renderer);
        return 0;
    }

    dispatch_semaphore_t completionSemaphore =
        kMglPdSynchronizeTextureUploads ? dispatch_semaphore_create(0) : NULL;
    int uploadError = 0;
    MglPdUploadCompletionCtx *completionCtx =
        (MglPdUploadCompletionCtx *)calloc(1u, sizeof(*completionCtx));
    if (completionCtx) {
        completionCtx->renderer = renderer;
        completionCtx->reason = reason;
        completionCtx->upload_error = &uploadError;
        completionCtx->semaphore = completionSemaphore;
    }
    (void)mglRenderAddCommandBufferCompletion(uploadCB, mglPdUploadCompletionBody,
                                              completionCtx,
                                              mglPdUploadCompletionDestroy);

    if (mglRenderCommitCommandBuffer(uploadCB) != 0) {
        fprintf(stderr,
                "MGL ERROR: Metal-cpp texture command-buffer commit failed\n");
    }

    if (!kMglPdSynchronizeTextureUploads) {
        /* Keep uploads ordered on the same queue but avoid stalling the render
         * thread. */
        return 1;
    }

    const dispatch_time_t deadline = dispatch_time(
        DISPATCH_TIME_NOW,
        (int64_t)(kMglPdTextureUploadWaitTimeoutSeconds * NSEC_PER_SEC));
    if (dispatch_semaphore_wait(completionSemaphore, deadline) != 0) {
        fprintf(stderr,
                "MGL WARNING: dedicated upload wait timed out (%s), continuing asynchronously\n",
                reason ? reason : "texture_upload");
        return 1;
    }

    return !uploadError;
}

/* -uploadTextureSliceViaBlit:... */
int mglTextureUploadSliceViaBlit(void *renderer, void *texture,
                                 unsigned int texName, GLenum texTarget,
                                 const void *bytes, uint64_t bytesPerRow,
                                 uint64_t bytesPerImage, uint64_t width,
                                 uint64_t height, uint64_t depth, uint64_t level,
                                 uint64_t slice)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    if (!texture || !bytes || bytesPerRow == 0 || bytesPerImage == 0 ||
        width == 0) {
        return 0;
    }

    if (mglRendererShouldSkipGPUOperations(renderer)) {
        fprintf(stderr,
                "MGL AGX: Skipping texture upload during recovery\n");
        return 0;
    }

    const uint32_t textureType = mglPdTextureInfo(texture).texture_type;
    MGLRenderTextureUploadPlan uploadPlan = {0};
    if (mglRenderBuildTextureUploadPlan(
            (uint32_t)texTarget, textureType,
            (uint32_t)mglPdTextureInfo(texture).usage,
            (uint32_t)mglPdTextureInfo(texture).pixel_format,
            MGLCapabilityHasBug(areas.core ? &areas.core->capability : NULL,
                                MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB)
                ? 1
                : 0,
            width, height, depth, bytesPerRow, bytesPerImage, level, slice,
            &uploadPlan) != 0) {
        fprintf(stderr,
                "MGL WARNING: Rejecting invalid texture upload plan (tex=%u target=0x%x level=%lu slice=%lu)\n",
                (unsigned)texName, (unsigned)texTarget, (unsigned long)level,
                (unsigned long)slice);
        return 0;
    }

    if (mglTraceLogIsEnabled() &&
        mglRenderPixelFormatIsPackedDepthStencil(
            (uint32_t)mglPdTextureInfo(texture).pixel_format) &&
        (mglRenderTextureTargetIsArrayOr3D((uint32_t)texTarget)) &&
        bytesPerRow >= 16) {
        const uint8_t *probe = (const uint8_t *)bytes;
        mglTraceLog(
            "TEXTURE_UPLOAD_DS tex=%u target=0x%x fmt=%lu slice=%lu level=%lu size=%lux%lu bpr=%lu bpi=%lu first=%02x %02x %02x %02x %02x %02x %02x %02x next=%02x %02x %02x %02x %02x %02x %02x %02x",
            (unsigned)texName, (unsigned)texTarget,
            (unsigned long)mglPdTextureInfo(texture).pixel_format,
            (unsigned long)slice, (unsigned long)level, (unsigned long)width,
            (unsigned long)height, (unsigned long)bytesPerRow,
            (unsigned long)bytesPerImage, probe[0], probe[1], probe[2],
            probe[3], probe[4], probe[5], probe[6], probe[7], probe[8],
            probe[9], probe[10], probe[11], probe[12], probe[13], probe[14],
            probe[15]);
    }

    if (textureType == MGLTextureTypeCube ||
        textureType == MGLTextureTypeCubeArray) {
        static uint64_t s_cubeUploadLogs = 0;
        const uint64_t hit = ++s_cubeUploadLogs;
        if (hit <= 4ull || (hit % 2048ull) == 0ull) {
            fprintf(stderr,
                    "MGL CUBE UPLOAD tex=%u glTarget=0x%x face=%lu slice=%lu level=%lu origin=(0,0,0) size=%lux%lux%lu bpr=%lu bpi=%lu ptr=%p\n",
                    texName, texTarget, (unsigned long)slice,
                    (unsigned long)slice, (unsigned long)level,
                    (unsigned long)width,
                    (unsigned long)uploadPlan.normalized_height,
                    (unsigned long)uploadPlan.copy_depth,
                    (unsigned long)bytesPerRow,
                    (unsigned long)uploadPlan.normalized_bytes_per_image,
                    (const void *)bytes);
        }
    }

    const uint32_t uploadRoute = uploadPlan.route;

    /* Shared packed depth/stencil textures can be updated directly for the
     * depth plane.  AGX requires a separate X32_Stencil8 view upload for the
     * stencil plane, using a 2D view over the selected array slice. */
    if (mglRenderPixelFormatIsPackedDepthStencil(
            (uint32_t)mglPdTextureInfo(texture).pixel_format) &&
        mglPdTextureInfo(texture).storage_mode !=
            MGL_PD_TEXTURE_STORAGE_PRIVATE) {
        int uploaded = mglTextureReplaceRegionValue(
            texture,
            mglTextureRegion2D(0, 0, width, uploadPlan.normalized_height),
            level, slice, bytes, bytesPerRow,
            uploadPlan.normalized_bytes_per_image, 1);
        if (!uploaded) {
            fprintf(stderr,
                    "MGL WARNING: depth/stencil replaceRegion upload failed tex=%u\n",
                    (unsigned)texName);
        }
        if (uploaded && bytesPerRow >= width * 5u) {
            uploaded = mglTextureUploadPackedDepthStencilStencilPlane(
                texture, texName, bytes, width, uploadPlan.normalized_height,
                bytesPerRow, level, slice, 0u, 0u);
        }
        return uploaded;
    }

    /* Shared 2D array uploads via replaceRegion: the blit path can leave array
     * slices unpopulated on some AGX drivers when uploading CPU data during
     * initial texture creation.  Shared storage is safe here because bind
     * happens before the first draw that samples this texture. */
    if (textureType == MGLTextureType2DArray &&
        mglPdTextureInfo(texture).storage_mode !=
            MGL_PD_TEXTURE_STORAGE_PRIVATE) {
        if (mglTextureReplaceRegionValue(
                texture,
                mglTextureRegion2D(0, 0, width, uploadPlan.normalized_height),
                uploadPlan.destination_level, uploadPlan.destination_slice,
                bytes, bytesPerRow, uploadPlan.normalized_bytes_per_image,
                1)) {
            return 1;
        }
        fprintf(stderr,
                "MGL WARNING: 2D array replaceRegion upload failed (tex=%u level=%lu slice=%lu)\n",
                (unsigned)texName, (unsigned long)level, (unsigned long)slice);
        return 0;
    }

    /* 1D texture upload via replaceRegion branch:
     * - 1D textures are a low-frequency update path; replaceRegion is safe in
     *   this scenario;
     * - Before entering this function, the caller has already flushed CPU-side
     *   deferred draws via mglFlushPendingDrawsBeforeTextureWrite, avoiding
     *   ordering races between the upload and uncommitted render command
     *   buffers;
     * - Only available for shared storage; Private storage (e.g. MSAA) must
     *   fall back to the blit path. */
    if (uploadRoute == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_1D) {
        const MGLRegionValue region =
            uploadPlan.replace_region_dimension == 1u
                ? mglTextureRegion1D(0, width)
                : mglTextureRegion2D(0, 0, width,
                                     uploadPlan.normalized_height);
        if (mglTextureReplaceRegionValue(
                texture, region, uploadPlan.destination_level,
                uploadPlan.destination_slice, bytes, bytesPerRow,
                uploadPlan.normalized_bytes_per_image,
                uploadPlan.replace_use_slice != 0u)) {
            if (mglTraceLogIsEnabled() &&
                mglPdTextureInfo(texture).pixel_format ==
                    MGLPixelFormatR8Unorm &&
                width > 0) {
                const uint8_t *first = (const uint8_t *)bytes;
                mglTraceLog(
                    "TEXTURE_UPLOAD_1D_REPLACE tex=%u target=0x%x mtlType=%lu size=%lux%lu bpr=%lu bpi=%lu first=%u",
                    (unsigned)texName, (unsigned)texTarget,
                    (unsigned long)textureType, (unsigned long)width,
                    (unsigned long)uploadPlan.normalized_height,
                    (unsigned long)bytesPerRow,
                    (unsigned long)uploadPlan.normalized_bytes_per_image,
                    first ? first[0] : 0u);
            }
            return 1;
        }
        fprintf(stderr,
                "MGL WARNING: 1D texture replaceRegion upload failed, falling back to blit (tex=%u level=%lu slice=%lu)\n",
                (unsigned)texName, (unsigned long)level, (unsigned long)slice);
    }

    /* 3D texture upload via replaceRegion branch:
     * - 3D uses replaceRegion to work around the AGX driver's
     *   copyFromBuffer:toTexture: slice OOB assertion (triggered even when
     *   destinationSlice=0); driver bug tracked via
     *   MGLCapabilityHasBug(MGL_BUG_3D_COPY_FROM_BUFFER_SLICE_OOB).
     * - Metal requires bytesPerImage for 3D replaceRegion uploads, so padded
     *   depth planes are repacked and uploaded with the tight image stride.
     * - Only shared storage supports replaceRegion.  Do not fall back to the
     *   known-bad copyFromBuffer path while the AGX bug marker is active. */
    if (uploadRoute == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REPLACE_3D) {
        const void *replaceBytes = bytes;
        void *tightlyPackedBytes = NULL;
        if (uploadPlan.requires_repack) {
            tightlyPackedBytes = mglRenderTextureRepackDepthPlanes(
                bytes, uploadPlan.normalized_bytes_per_image,
                uploadPlan.expected_bytes_per_image, uploadPlan.copy_depth);
            if (!tightlyPackedBytes) {
                return 0;
            }
            replaceBytes = tightlyPackedBytes;
        }

        const MGLRegionValue region = mglTextureRegion3D(
            0, 0, 0, width, uploadPlan.normalized_height,
            uploadPlan.copy_depth);
        if (mglTextureReplaceRegionValue(
                texture, region, uploadPlan.destination_level,
                uploadPlan.destination_slice, replaceBytes, bytesPerRow,
                uploadPlan.expected_bytes_per_image, 1)) {
            free(tightlyPackedBytes);
            return 1;
        }
        free(tightlyPackedBytes);
        fprintf(stderr,
                "MGL WARNING: 3D texture replaceRegion upload failed (tex=%u level=%lu)\n",
                (unsigned)texName, (unsigned long)level);
        return 0;
    }

    /* 3D + Private while the AGX copyFromBuffer workaround is required:
     * rejected by the C++ route (blit is known-bad and replaceRegion does not
     * support private storage). */
    if (uploadRoute == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REJECT) {
        fprintf(stderr,
                "MGL WARNING: Rejecting private 3D upload while AGX copyFromBuffer workaround is required (tex=%u level=%lu)\n",
                (unsigned)texName, (unsigned long)level);
        return 0;
    }

    /* 2D / 2DArray / Cube texture upload via blit path (dedicated CB +
     * completion handler):
     * - replaceRegion must not be used: when the texture is being sampled by an
     *   in-flight command buffer, replaceRegion's CPU direct writes are not
     *   subject to GPU-side ordering constraints, causing data races with
     *   in-flight sampling draws (this previously caused Minecraft GUI item
     *   rendering corruption);
     * - The blit path is required to guarantee GPU-side ordering. */
    void *stagingOwner = NULL;
    void *borrowedStagingBuffer = NULL;
    void *uploadBuffer = NULL;
    if (mglRenderCreateTextureStagingOwner(
            bytes, uploadPlan.buffer_size,
            MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED, &stagingOwner,
            &borrowedStagingBuffer) == 0 &&
        stagingOwner && borrowedStagingBuffer) {
        uploadBuffer = borrowedStagingBuffer;
    }
    if (!uploadBuffer) {
        mglRenderDestroyTextureStagingOwner(&stagingOwner);
        fprintf(stderr,
                "MGL WARNING: Failed to allocate upload buffer for texture blit\n");
        return 0;
    }

    const int uploaded = mglTextureCopyUploadWithDedicatedCommandBuffer(
        renderer, uploadBuffer, 0, bytesPerRow,
        uploadPlan.normalized_bytes_per_image, 0, 1,
        mglTextureSize(width, uploadPlan.normalized_height,
                       uploadPlan.copy_depth),
        texture, uploadPlan.destination_slice, uploadPlan.destination_level,
        mglTextureOrigin(0, 0, 0), "texture_upload_blit");
    /* The encoded blit retains its source resource until command-buffer
     * completion; release the C++ staging owner as soon as encoding ends. */
    mglRenderDestroyTextureStagingOwner(&stagingOwner);
    if (!uploaded) {
        fprintf(stderr,
                "MGL WARNING: Dedicated texture upload failed (level=%lu slice=%lu)\n",
                (unsigned long)level, (unsigned long)slice);
        return 0;
    }
    if (mglRenderPixelFormatIsPackedDepthStencil(
            (uint32_t)mglPdTextureInfo(texture).pixel_format) &&
        bytesPerRow >= width * 5u) {
        (void)mglTextureUploadPackedDepthStencilStencilPlane(
            texture, texName, bytes, width, uploadPlan.normalized_height,
            bytesPerRow, level, slice, 0u, 0u);
    }
    return 1;
}


/* === The CPU-upload tree (log 195) ========================================
 * -uploadFullCPUTextureDataIntoTexture:, -encodeTextureBytesUpload:,
 * -reUploadExistingCPUTextureData:, -reUploadExistingCPUTextureDataArrayLevel:,
 * -fillSmallRGBA8TextureWithGradient:tex: and
 * -fillTextureWithSafeInitialContents:tex:pixelFormat: moved from
 * MGLRenderer+Texture.m.  The last two carried the tree's only @try blocks;
 * they keep their catch logic through the shell's guarded call (rule 58 (b)).
 */

/* ctx of the small-RGBA8 gradient fill's guarded body. */
typedef struct MglUpFillCtx_t {
    void *texture;
    Texture *tex;
} MglUpFillCtx;

/* ALTERNATIVE 1 of the safe fill: the MTLBuffer-to-texture copy, which the .m
 * wrapped in its own @try so a failing copy falls through to the gradient. */
typedef struct MglUpSafeFillCtx_t {
    void *texture;
    Texture *tex;
    void *device;
    const void *proper_data;
    MGLRegionValue proper_region;
    uint64_t proper_bytes_per_row;
    uint64_t fill_size;
    uint64_t data_size;
} MglUpSafeFillCtx;

static int mglUpSafeFillBody(void *renderer, void *rawCtx)
{
    MglUpSafeFillCtx *fill = (MglUpSafeFillCtx *)rawCtx;
    void *texture = fill->texture;
    Texture *tex = fill->tex;
    void *device = fill->device;
    const void *properData = fill->proper_data;
    MGLRegionValue properRegion = fill->proper_region;
    uint64_t properBytesPerRow = fill->proper_bytes_per_row;
    uint64_t fillSize = fill->fill_size;
    uint64_t dataSize = fill->data_size;
    (void)device;
    (void)renderer;

                            // ALTERNATIVE 1: Try MTLBuffer-to-texture copy approach

                            if (properData && dataSize > 0) {

                                fprintf(stderr, "MGL INFO: Attempting buffer-based texture fill\n");


                                // Create a temporary MTLBuffer with the texture data

                                void *tempBuffer =
                                    mglUpCreateBufferWithBytes(
                                        properData, fillSize,
                                        MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED);


                                if (tempBuffer) {

                                    fprintf(stderr, "MGL INFO: Created temporary MTLBuffer for texture data\n");


                                    if (mglRendererShouldSkipGPUOperations(renderer)) {

                                        fprintf(stderr, "MGL AGX: Skipping texture fill during recovery - texture will be empty\n");

                                    } else {

                                        int uploaded = mglTextureCopyUploadWithDedicatedCommandBuffer(
            renderer, tempBuffer, 0, properBytesPerRow, fillSize, 0, 1, mglTextureSize(properRegion.size.width, properRegion.size.height, 1), texture, 0, 0, mglTextureOrigin(0, 0, 0), "texture_fill_initialization");

                                        if (uploaded) {

                                            fprintf(stderr, "MGL SUCCESS: Texture data copied using dedicated upload command buffer\n");

                                            mglUpMarkTextureLevelMetalFilled(tex, 0, fillSize);

                                        } else {

                                            fprintf(stderr, "MGL WARNING: Dedicated texture fill upload failed - texture may remain uninitialized\n");

                                        }

                                    }


                                    // Clean up the temporary buffer

                                    tempBuffer = NULL;

                                }

                            }
    return 1;
}

static int mglUpFillBody(void *renderer, void *rawCtx)
{
    MglUpFillCtx *fillCtx = (MglUpFillCtx *)rawCtx;
    void *texture = fillCtx->texture;
    Texture *tex = fillCtx->tex;
    (void)renderer;


                                    // Create a simple pattern that's not magenta

                                    uint64_t pixelCount = mglPdTextureInfo(texture).width * mglPdTextureInfo(texture).height;

                                    uint32_t *simpleData = calloc(pixelCount, sizeof(uint32_t));


                                    if (simpleData) {

                                        // Create a simple gradient pattern instead of magenta

                                        for (uint64_t y = 0; y < mglPdTextureInfo(texture).height; y++) {

                                            for (uint64_t x = 0; x < mglPdTextureInfo(texture).width; x++) {

                                                uint64_t index = y * mglPdTextureInfo(texture).width + x;


                                                // Create a simple gradient from blue to green

                                                uint8_t r = (uint8_t)(x * 255 / mglPdTextureInfo(texture).width);

                                                uint8_t g = (uint8_t)(y * 255 / mglPdTextureInfo(texture).height);

                                                uint8_t b = 128;

                                                uint8_t a = 255;


                                                simpleData[index] = (a << 24) | (b << 16) | (g << 8) | r;

                                            }

                                        }


                                        // Try direct replaceRegion for simple cases

                                        MGLRegionValue simpleRegion = mglTextureRegion2D(0, 0, mglPdTextureInfo(texture).width, mglPdTextureInfo(texture).height);

                                        (void)mglTextureReplaceRegionValue(
                                            texture, simpleRegion, 0, 0,
                                            simpleData,
                                            mglPdTextureInfo(texture).width * sizeof(uint32_t),
                                            mglPdTextureInfo(texture).width * mglPdTextureInfo(texture).height * sizeof(uint32_t),
                                            1);


                                        fprintf(stderr, "MGL SUCCESS: Simple direct color fill completed\n");

                                        mglUpMarkTextureLevelMetalFilled(tex, 0, pixelCount * sizeof(uint32_t));

                                        free(simpleData);

                                    }
    return 1;
}

/* -uploadFullCPUTextureDataIntoTexture: */
int mglTextureUploadFullCPUData(void *renderer, Texture *tex, void *texture,
                                const char *reason)
{
    if (!tex || !texture || !tex->faces[0].levels) {
        return false;
    }
    if (!mglRenderTextureTargetIs2D((uint32_t)tex->target) ||
        mglPdTextureInfo(texture).texture_type != MGLTextureType2D) {
        return false;
    }

    int numFaces = 1;
    GLuint levelCount = mglUpMin((GLuint)mglPdTextureInfo(texture).mipmap_level_count,
                            tex->num_levels ? tex->num_levels : 1u);
    if (levelCount == 0u ||
        !mglTextureHasUploadableCPUData(tex, numFaces, levelCount)) {
        return false;
    }


    MGLRenderLevelUploadOp uploadOps[levelCount ? levelCount : 1u];
    uint32_t opCount = 0;
    uint32_t shortCount = 0;
    uint32_t badCount = 0;
    if (mglRenderBuildLevelUploadOps(
            tex->faces[0].levels, levelCount,
            (uint32_t)mglPdTextureInfo(texture).texture_type,
            (uint32_t)tex->internalformat,
            (uint32_t)mglPdTextureInfo(texture).pixel_format,
            uploadOps, levelCount,
            &opCount, &shortCount, &badCount) != 0) {
        return false;
    }

    bool uploadedAny = false;
    bool failedAny = (shortCount + badCount) > 0;
    for (uint32_t i = 0; i < opCount; i++) {
        MGLRenderLevelUploadOp *op = &uploadOps[i];
        if (op->kind == 1u) {
            static uint64_t s_shortBackingLogs = 0;
            uint64_t hit = ++s_shortBackingLogs;
            if (kMGLDiagnosticStateLogs &&
                (hit <= 32ull || (hit % 512ull) == 0ull)) {
                mglTraceLog("MGL TEXTURE CPU-REFRESH skip short backing tex=%u level=%u face=0 have=%llu need=%llu reason=%s hit=%llu",
                              (unsigned)tex->name,
                              (unsigned)op->level,
                              (unsigned long long)op->available_bytes,
                              (unsigned long long)op->needed_bytes,
                              reason ? reason : "(null)",
                              (unsigned long long)hit);
            }
            continue;
        }

        bool uploaded = mglTextureUploadSliceViaBlit(
            renderer, texture, tex->name, tex->target, op->data, (uint64_t)op->bytes_per_row, (uint64_t)op->bytes_per_image, (uint64_t)op->width, (uint64_t)op->height, (uint64_t)op->copy_depth, op->level, 0);
        if (op->owns_data) {
            free((void *)op->data);
        }
        if (uploaded) {
            uploadedAny = true;
            /* CPU refreshed Metal mip while a Y-flip sampled copy may still
             * hold the previous RT contents for that level. */
            mglMarkGLSampledCopyLevelDirty(tex, op->level);
        } else {
            failedAny = true;
        }
    }

    static uint64_t s_refreshLogs = 0;
    uint64_t hit = ++s_refreshLogs;
    if (kMGLDiagnosticStateLogs &&
        (uploadedAny || hit <= 32ull || (hit % 512ull) == 0ull)) {
        mglTraceLog("MGL TEXTURE CPU-REFRESH tex=%u mtl=%p uploaded=%d failed=%d dirty=0x%x levels=%u reason=%s hit=%llu",
                      (unsigned)tex->name,
                      texture,
                      uploadedAny ? 1 : 0,
                      failedAny ? 1 : 0,
                      (unsigned)tex->dirty_bits,
                      (unsigned)levelCount,
                      reason ? reason : "(null)",
                      (unsigned long long)hit);
    }

    if (uploadedAny && !failedAny) {
        tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
        mglRendererRecordGPUSuccess(renderer);
        return true;
    }

    return false;
}

/* -encodeTextureBytesUpload:... */
int mglTextureEncodeBytesUpload(void *renderer, Texture *tex, void *buffer,
                                uint64_t sourceOffset, uint64_t sourceBytesPerRow,
                                uint64_t sourceBytesPerImage, uint64_t width,
                                uint64_t height, uint64_t depth, uint64_t slice,
                                uint64_t level, uint64_t xoffset,
                                uint64_t yoffset, uint64_t zoffset,
                                const char *reason)
{
    MGL_ASSERT_GL_THREAD();
    if (!tex || !buffer || sourceBytesPerRow == 0 || width == 0 || height == 0) {
        return false;
    }

    if (tex->mtl_data == NULL) {
        mglRendererBindMTLTexture(renderer, tex);
        if (tex->mtl_data == NULL) {
            return false;
        }
    }

    void *texture = tex->mtl_data;
    if (!texture) {
        return false;
    }

    uint32_t textureType = mglPdTextureInfo(texture).texture_type;
    MGLRenderTextureSubUploadPlan uploadPlan = {0};
    if (mglRenderTextureSubUploadPlan(
            (uint32_t)tex->target, (uint32_t)textureType, (uint64_t)slice,
            (uint64_t)xoffset, (uint64_t)yoffset, (uint64_t)zoffset,
            (uint64_t)width, (uint64_t)height, (uint64_t)depth,
            (uint64_t)sourceBytesPerRow, (uint64_t)sourceBytesPerImage,
            &uploadPlan) != 0) {
        return false;
    }
    uint64_t destinationSlice = (uint64_t)uploadPlan.destination_base_slice;
    MGLOriginValue destinationOrigin = mglTextureOrigin(
        (uint64_t)uploadPlan.destination_x,
        (uint64_t)uploadPlan.destination_y,
        (uint64_t)uploadPlan.destination_z);
    uint64_t copyHeight = (uint64_t)uploadPlan.copy_height;
    uint64_t copyDepth = (uint64_t)uploadPlan.copy_depth;
    uint64_t layerCount = (uint64_t)uploadPlan.layer_count;
    uint64_t sourceLayerStride =
        (uint64_t)uploadPlan.source_layer_stride;
    if (copyHeight > UINT64_MAX / sourceBytesPerRow) {
        return false;
    }
    uint64_t expectedBytesPerImage = sourceBytesPerRow * copyHeight;
    uint64_t copyBytesPerImage = sourceBytesPerImage;
    if (textureType == MGLTextureTypeCube ||
        textureType == MGLTextureTypeCubeArray ||
        textureType == MGLTextureType2DArray ||
        textureType == MGLTextureType1DArray ||
        textureType == MGLTextureType2DMultisampleArray) {
        copyBytesPerImage = expectedBytesPerImage;
    } else if (textureType == MGLTextureType3D) {
        if (copyBytesPerImage < expectedBytesPerImage) {
            copyBytesPerImage = expectedBytesPerImage;
        }
    } else {
        copyBytesPerImage = expectedBytesPerImage;
    }

    uint64_t maxDestinationSlices = mglPdTextureInfo(texture).array_length;
    if (textureType == MGLTextureTypeCube) {
        maxDestinationSlices = 6UL;
    } else if (textureType == MGLTextureTypeCubeArray) {
        maxDestinationSlices = mglPdTextureInfo(texture).array_length * 6UL;
    }

    if (level >= mglPdTextureInfo(texture).mipmap_level_count ||
        destinationSlice >= maxDestinationSlices ||
        layerCount > maxDestinationSlices - destinationSlice ||
        destinationOrigin.x > mglPdTextureInfo(texture).width ||
        destinationOrigin.y > mglPdTextureInfo(texture).height ||
        destinationOrigin.z > mglPdTextureInfo(texture).depth ||
        width > mglPdTextureInfo(texture).width - destinationOrigin.x ||
        copyHeight > mglPdTextureInfo(texture).height - destinationOrigin.y ||
        copyDepth > mglPdTextureInfo(texture).depth - destinationOrigin.z) {
        fprintf(stderr, "MGL ERROR: texture sub upload out of bounds tex=%u level=%lu slice=%lu origin=(%lu,%lu,%lu) size=%lux%lux%lu texture=%lux%lux%lu\n",
              tex->name,
              (unsigned long)level,
              (unsigned long)destinationSlice,
              (unsigned long)destinationOrigin.x,
              (unsigned long)destinationOrigin.y,
              (unsigned long)destinationOrigin.z,
              (unsigned long)width,
              (unsigned long)copyHeight,
              (unsigned long)copyDepth,
              (unsigned long)mglPdTextureInfo(texture).width,
              (unsigned long)mglPdTextureInfo(texture).height,
              (unsigned long)mglPdTextureInfo(texture).depth);
        return false;
    }

    return mglTextureCopyUploadWithDedicatedCommandBuffer(
            renderer, buffer, sourceOffset, sourceBytesPerRow, copyBytesPerImage, sourceLayerStride, layerCount, mglTextureSize(
                                                       (uint64_t)uploadPlan.copy_width,
                                                       copyHeight, copyDepth), texture, destinationSlice, level, destinationOrigin, reason ? reason : "texture_sub_upload");
}

/* ALTERNATIVE 1 of the safe fill: the MTLBuffer-to-texture copy, which the .m
 * wrapped in its own @try so a failing copy falls through to the gradient. */



/* -reUploadExistingCPUTextureDataArrayLevel:... */
void mglTextureReUploadArrayLevel(void *renderer, Texture *tex, void *texture,
                                  uint32_t pixelFormat, int face, int level,
                                  int texture1DArrayBackedBy2DArray,
                                  uint32_t tex_type)
{
    uint64_t lvlWidth  = tex->faces[face].levels[level].width;
    uint64_t lvlHeight = tex->faces[face].levels[level].height;
    uint64_t lvlPitch  = tex->faces[face].levels[level].pitch;


                /* Array texture re-upload: loop over array layers and upload

                 * each slice independently.  Mirrors the DIRTY_TEXTURE_DATA

                 * array path (12861-13087).  The old code only uploaded

                 * slice 0 and passed the entire array's data_size as

                 * bytesPerImage with depth=num_layers, causing a crash in

                 * uploadTextureSliceViaBlit's newBufferWithBytes. */

                GLuint num_layers = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray)

                    ? tex->faces[face].levels[level].height

                    : tex->faces[face].levels[level].depth;

                if (num_layers == 0) return;


                int arraySliceIs1D = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray);

                uint64_t uploadSliceHeight = arraySliceIs1D ? 1UL : mglUpMax((uint64_t)lvlHeight, 1UL);

                uint64_t baseBytesPerRow = lvlPitch;

                uint64_t uploadSliceRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);

                if (uploadSliceRows == 0 || baseBytesPerRow > (UINT64_MAX / uploadSliceRows)) {

                    fprintf(stderr, "MGL WARNING: Re-upload array invalid row layout tex=%d face=%d level=%d bpr=%lu rows=%lu\n",

                          tex->name,

                          face,

                          level,

                          (unsigned long)baseBytesPerRow,

                          (unsigned long)uploadSliceRows);

                    return;

                }

                uint64_t logicalBytesPerImage = baseBytesPerRow * uploadSliceRows;

                uint64_t backingBytes = tex->faces[face].levels[level].data_size;

                /* data_size is page-rounded; do not treat the slack as layer
                 * stride or reads land in the wrong slice. */

                uint64_t requiredArrayBytes = 0;

                uint64_t safeLayerCount = mglUpMax((uint64_t)num_layers, 1UL);

                if (logicalBytesPerImage == 0 ||

                    logicalBytesPerImage > (UINT64_MAX / safeLayerCount) ||

                    backingBytes < (requiredArrayBytes = logicalBytesPerImage * safeLayerCount)) {

                    fprintf(stderr, "MGL WARNING: Re-upload array backing too small tex=%d face=%d level=%d backing=%lu layerBytes=%lu layers=%u\n",

                          tex->name, face, level,

                          (unsigned long)backingBytes,

                          (unsigned long)logicalBytesPerImage,

                          num_layers);

                    return;

                }


                for (GLuint layer = 0; layer < num_layers; layer++)

                {

                    size_t offset = logicalBytesPerImage * layer;

                    const void *layerSrcData = (const uint8_t *)tex->faces[face].levels[level].data + offset;

                    void *expandedUploadData = NULL;
                    void *swizzledUploadData = NULL;

                    uint64_t effectiveBytesPerRow = baseBytesPerRow;

                    uint64_t effectiveBytesPerImage = logicalBytesPerImage;

                    if (mglTextureUploadNeedsSwizzleBake(tex)) {
                        uint64_t swzBPR = 0;
                        uint64_t swzBPI = 0;
                        swizzledUploadData = mglCreateSwizzledUpload(
                            tex, (const uint8_t *)layerSrcData, lvlWidth,
                            uploadSliceHeight, baseBytesPerRow, &swzBPR,
                            &swzBPI);
                        if (swizzledUploadData) {
                            layerSrcData = swizzledUploadData;
                            effectiveBytesPerRow = swzBPR;
                            effectiveBytesPerImage = swzBPI;
                        }
                    }

                    if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                    mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {

                        uint64_t expandedBPR = 0, expandedBPI = 0;

                        expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,

                                                                          (const uint8_t *)layerSrcData,

                                                                          lvlWidth,

                                                                          uploadSliceHeight,

                                                                          baseBytesPerRow,

                                                                          &expandedBPR,

                                                                          &expandedBPI);

                        if (expandedUploadData) {

                            layerSrcData = expandedUploadData;

                            effectiveBytesPerRow = expandedBPR;

                            effectiveBytesPerImage = expandedBPI;

                        }

                    } else if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                           mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {

                        uint64_t expandedBPR = 0, expandedBPI = 0;

                        expandedUploadData = mglCreateChannelExpandedUpload(tex,

                                                                             pixelFormat,

                                                                             (const uint8_t *)layerSrcData,

                                                                             lvlWidth,

                                                                             uploadSliceHeight,

                                                                             baseBytesPerRow,

                                                                             &expandedBPR,

                                                                             &expandedBPI);

                        if (expandedUploadData) {

                            layerSrcData = expandedUploadData;

                            effectiveBytesPerRow = expandedBPR;

                            effectiveBytesPerImage = expandedBPI;

                        }

                    }

                    uint64_t dsBytesPerRow = 0;
                    uint64_t dsBytesPerImage = 0;
                    void *dsUploadData = mglUpCreateDepthStencilMetalUpload(
                        tex, pixelFormat, (const uint8_t *)layerSrcData,
                        lvlWidth, uploadSliceHeight, effectiveBytesPerRow,
                        &dsBytesPerRow, &dsBytesPerImage);
                    if (dsUploadData) {
                        free(expandedUploadData);
                        expandedUploadData = dsUploadData;
                        layerSrcData = dsUploadData;
                        effectiveBytesPerRow = dsBytesPerRow;
                        effectiveBytesPerImage = dsBytesPerImage;
                    }


                    uint64_t alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);

                    uint64_t alignedBytesPerRow = effectiveBytesPerRow;

                    if (alignedBytesPerRow % alignment != 0) {

                        alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;

                    }


                    uintptr_t addr = (uintptr_t)layerSrcData;

                    if (addr % alignment != 0 || alignedBytesPerRow != effectiveBytesPerRow) {

                        uint64_t alignedUploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);

                        if (alignedUploadRows == 0 || alignedBytesPerRow > (UINT64_MAX / alignedUploadRows)) {

                            fprintf(stderr, "MGL WARNING: Re-upload array rejecting aligned row layout bpr=%lu rows=%lu tex=%d face=%d level=%d layer=%u\n",

                                  (unsigned long)alignedBytesPerRow,

                                  (unsigned long)alignedUploadRows,

                                  tex->name,

                                  face,

                                  level,

                                  layer);

                            free(expandedUploadData);

                            continue;

                        }

                        uint64_t alignedSize = alignedBytesPerRow * alignedUploadRows;

                        if (alignedSize > 0 && alignedSize <= (512 * 1024 * 1024)) {

                            void *alignedData = aligned_alloc(alignment, alignedSize);

                            if (alignedData) {

                                memset(alignedData, 0, alignedSize);

                                for (uint64_t row = 0; row < alignedUploadRows; row++) {

                                    uint64_t copySize = mglUpMin(effectiveBytesPerRow, alignedBytesPerRow);

                                    memcpy((uint8_t *)alignedData + row * alignedBytesPerRow,

                                           (const uint8_t *)layerSrcData + row * effectiveBytesPerRow, copySize);

                                }

                                mglTextureUploadSliceViaBlit(
            renderer, texture, tex->name, tex->target, alignedData, alignedBytesPerRow, alignedSize, lvlWidth, lvlHeight, 1, level, layer);

                                free(alignedData);

                            }

                        }

                    } else {

                        mglTextureUploadSliceViaBlit(
            renderer, texture, tex->name, tex->target, layerSrcData, effectiveBytesPerRow, effectiveBytesPerImage, lvlWidth, lvlHeight, 1, level, layer);

                    }

                    free(swizzledUploadData);
                    free(expandedUploadData);

                }
}

/* -fillSmallRGBA8TextureWithGradient:tex: */
void mglTextureFillSmallGradient(void *renderer, void *texture, Texture *tex)
{
                            if (mglRenderIsSmallRGBA8(
                                    (uint32_t)mglPdTextureInfo(texture).width,
                                    (uint32_t)mglPdTextureInfo(texture).height,
                                    (uint32_t)tex->internalformat)) {

                                fprintf(stderr, "MGL INFO: Attempting simple direct color fill for small RGBA8 texture\n");/* The .m ran this fill inside @try/@catch; the C port keeps the same catch
     * logic through the shell's guarded call (rule 58 (b)). */
    {
        char fillFailure[256] = {0};
        MglUpFillCtx fillCtx = { texture, tex };
        if (!mglPlatformShellGuardedCallCtxReason(renderer, "small RGBA8 fill",
                                                  mglUpFillBody, &fillCtx,
                                                  fillFailure,
                                                  sizeof(fillFailure))) {
            fprintf(stderr,
                    "MGL WARNING: Simple direct fill also failed: %s\n",
                    fillFailure[0] ? fillFailure : "(null)");
        }
    }

                            } else {

                                fprintf(stderr, "MGL INFO: Skipping complex texture - would use deferred initialization\n");

                            }
}

/* -reUploadExistingCPUTextureData:... */
void mglTextureReUploadExisting(void *renderer, Texture *tex, void *texture,
                                uint32_t pixelFormat, uint32_t num_faces,
                                uint32_t upload_level_count, int is_array,
                                int texture1DBackedBy2D,
                                int texture1DArrayBackedBy2DArray,
                                uint32_t tex_type)
{
    fprintf(stderr, "MGL INFO: Re-uploading existing CPU texture data (tex=%d, dims=%lux%lu)\n",

          tex->name, (unsigned long)mglPdTextureInfo(texture).width, (unsigned long)mglPdTextureInfo(texture).height);


    for (int face = 0; face < num_faces; face++) {

        for (int level = 0; level < (int)upload_level_count; level++) {

            TextureLevel *uploadLevel = &tex->faces[face].levels[level];

            if (!mglTextureLevelHasUploadableCPUData(uploadLevel)) {

                continue;

            }


            uint64_t lvlWidth  = tex->faces[face].levels[level].width;

            uint64_t lvlHeight = tex->faces[face].levels[level].height;

            uint64_t lvlDepth  = tex->faces[face].levels[level].depth;

            uint64_t lvlPitch  = tex->faces[face].levels[level].pitch;

            if (lvlPitch == 0 || lvlWidth == 0) continue;


            if (is_array)

            {
                mglTextureReUploadArrayLevel(renderer, tex, texture, pixelFormat, face, level,
                                                                         texture1DArrayBackedBy2DArray, tex_type);
            }

            else

            {

            /* Non-array re-upload (2D, 3D, 1D, cube).

             * For 3D textures, bytesPerImage must be a single 2D slice

             * (bytesPerRow * height), 0T the full volume data_size.

             * uploadTextureSliceViaBlit computes bufferSize =

             * safeBytesPerImage * copyDepth, so passing the full volume

             * as bytesPerImage AND depth would double-count and cause

             * newBufferWithBytes to read past the source buffer. */

            uint64_t bytesPerRow = lvlPitch;

            uint64_t fullDataSize = tex->faces[face].levels[level].data_size;

            if (fullDataSize == 0) fullDataSize = bytesPerRow * mglUpMax((uint64_t)lvlHeight, 1UL);


            int is3DReupload = mglRenderIs3DReupload(
                                    (uint32_t)tex->target, (uint32_t)lvlDepth) != 0;

            uint64_t singleSliceBPI = bytesPerRow * mglUpMax((uint64_t)lvlHeight, 1UL);

            uint64_t bytesPerImage = is3DReupload ? singleSliceBPI : fullDataSize;

            uint64_t uploadDepth = is3DReupload ? lvlDepth : (lvlDepth > 1 ? lvlDepth : 1);


            const void *srcData = (const void *)tex->faces[face].levels[level].data;

            void *expandedUploadData = NULL;

            /* Channel expansion for 2D/non-3D only.  3D expansion would

             * require per-slice handling (see DIRTY_TEXTURE_DATA 3D path). */

            if (!is3DReupload) {

                if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                    mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {

                    uint64_t expandedBytesPerRow = 0;

                    uint64_t expandedBytesPerImage = 0;

                    expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,

                                                                      (const uint8_t *)srcData,

                                                                      lvlWidth,

                                                                      mglUpMax((uint64_t)lvlHeight, 1UL),

                                                                      bytesPerRow,

                                                                      &expandedBytesPerRow,

                                                                      &expandedBytesPerImage);

                    if (expandedUploadData) {

                        srcData = expandedUploadData;

                        bytesPerRow = expandedBytesPerRow;

                        bytesPerImage = expandedBytesPerImage;

                    }

                } else if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                           mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {

                    uint64_t expandedBytesPerRow = 0;

                    uint64_t expandedBytesPerImage = 0;

                    expandedUploadData = mglCreateChannelExpandedUpload(tex,

                                                                         pixelFormat,

                                                                         (const uint8_t *)srcData,

                                                                         lvlWidth,

                                                                         mglUpMax((uint64_t)lvlHeight, 1UL),

                                                                         bytesPerRow,

                                                                         &expandedBytesPerRow,

                                                                         &expandedBytesPerImage);

                    if (expandedUploadData) {

                        srcData = expandedUploadData;

                        bytesPerRow = expandedBytesPerRow;

                        bytesPerImage = expandedBytesPerImage;

                    }

                }

            }

            /* Combined depth/stencil CPU shadows use a packed layout
             * (DEPTH32F_STENCIL8 = 5 bytes/texel) while the Metal texture
             * expects 8 bytes/texel with stencil at byte 4; repack here so
             * the non-array refresh path matches the dirty/array paths. */
            uint64_t dsBytesPerRow = 0;
            uint64_t dsBytesPerImage = 0;
            void *dsUploadData = mglUpCreateDepthStencilMetalUpload(
                tex, pixelFormat, (const uint8_t *)srcData,
                lvlWidth, mglUpMax((uint64_t)lvlHeight, 1UL),
                bytesPerRow, &dsBytesPerRow, &dsBytesPerImage);
            if (dsUploadData) {
                free(expandedUploadData);
                expandedUploadData = dsUploadData;
                srcData = dsUploadData;
                bytesPerRow = dsBytesPerRow;
                bytesPerImage = dsBytesPerImage;
            }

            uint64_t alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);

            uint64_t alignedBytesPerRow = bytesPerRow;

            if (alignedBytesPerRow % alignment != 0) {

                alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;

            }


            uintptr_t addr = (uintptr_t)srcData;

            if (addr % alignment != 0 || alignedBytesPerRow != bytesPerRow) {

                uint64_t rowCount = mglUpMax((uint64_t)lvlHeight, 1UL);

                uint64_t alignedSliceBPI = alignedBytesPerRow * rowCount;

                uint64_t alignedSize = alignedSliceBPI * uploadDepth;

                if (alignedSize > 0 && alignedSize <= (512 * 1024 * 1024)) {

                    void *alignedData = aligned_alloc(alignment, alignedSize);

                    if (alignedData) {

                        memset(alignedData, 0, alignedSize);

                        for (uint64_t z = 0; z < uploadDepth; z++) {

                            for (uint64_t row = 0; row < rowCount; row++) {

                                uint64_t copySize = mglUpMin(bytesPerRow, alignedBytesPerRow);

                                memcpy((uint8_t *)alignedData + z * alignedSliceBPI + row * alignedBytesPerRow,

                                       (const uint8_t *)srcData + z * singleSliceBPI + row * bytesPerRow, copySize);

                            }

                        }

                        mglTextureUploadSliceViaBlit(
            renderer, texture, tex->name, tex->target, alignedData, alignedBytesPerRow, alignedSliceBPI, lvlWidth, lvlHeight, uploadDepth, level, face);

                        free(alignedData);

                    }

                }

            } else {

                mglTextureUploadSliceViaBlit(
            renderer, texture, tex->name, tex->target, srcData, bytesPerRow, bytesPerImage, lvlWidth, lvlHeight, uploadDepth, level, face);

            }

            free(expandedUploadData);

            } /* end else (non-array) */

        }

    }
}

/* -fillTextureWithSafeInitialContents:tex:pixelFormat: */
void mglTextureFillSafeInitialContents(void *renderer, void *texture,
                                       Texture *tex, uint32_t pixelFormat)
{


    if (mglPdTextureInfo(texture).width == 0 || mglPdTextureInfo(texture).height == 0 || mglPdTextureInfo(texture).width > 16384 || mglPdTextureInfo(texture).height > 16384) {

        fprintf(stderr, "MGL WARNING: Skipping texture fill due to invalid dimensions: %lux%lu\n", (unsigned long)mglPdTextureInfo(texture).width, (unsigned long)mglPdTextureInfo(texture).height);

    } else {

        // Determine pixel format size to create appropriate black data

        uint64_t bytesPerPixel = (uint64_t)mglRenderMetalPixelFormatBytesPerPixel(
            mglPdTextureInfo(texture).pixel_format);

        // Calculate dynamic alignment for Metal textures based on pixel format

        uint64_t bytesPerRow = mglPdTextureInfo(texture).width * bytesPerPixel;

        uint64_t alignment = mglRendererOptimalAlignmentForPixelFormat(mglPdTextureInfo(texture).pixel_format);

        if (bytesPerRow % alignment != 0) {

            bytesPerRow = ((bytesPerRow + alignment - 1) / alignment) * alignment;

        }


        uint64_t dataSize = bytesPerRow * mglPdTextureInfo(texture).height;


        // Validate that dataSize is reasonable (not too large)

        if (dataSize > 64 * 1024 * 1024) { // 64MB limit per texture level

            fprintf(stderr, "MGL WARNING: Skipping texture fill due to excessive size: %lu bytes\n", (unsigned long)dataSize);

        } else {

            // Allocate initialization data for texture clear.

            // aligned_alloc has been unreliable in this environment; calloc is safer here.

            (void)alignment;

            void *blackData = calloc(dataSize, 1);

            if (blackData) {

                // CRITICAL SECURITY FIX: Comprehensive validation to prevent Metal driver crashes

                // calloc already zero-initializes


                // Multi-layer validation for all parameters

                if (!blackData) {

                    fprintf(stderr, "MGL SECURITY ERROR: blackData is NULL after memset - CORRUPTION DETECTED\n");

                    return;
                }

                if (bytesPerRow == 0) {

                    fprintf(stderr, "MGL SECURITY ERROR: Invalid bytesPerRow (0) for texture fill\n");

                    free(blackData);

                    return;
                }

                if (dataSize == 0) {

                    fprintf(stderr, "MGL SECURITY ERROR: Invalid dataSize (0) for texture fill\n");

                    free(blackData);

                    return;
                }

                if (!texture) {

                    fprintf(stderr, "MGL SECURITY ERROR: Metal texture is NULL\n");

                    free(blackData);

                    return;
                }

                if (mglPdTextureInfo(texture).width == 0 || mglPdTextureInfo(texture).height == 0) {

                    fprintf(stderr, "MGL SECURITY ERROR: Invalid texture dimensions %lux%lu\n", (unsigned long)mglPdTextureInfo(texture).width, (unsigned long)mglPdTextureInfo(texture).height);

                    free(blackData);

                    return;
                }


                // Additional validation: verify blackData contains expected zeros (anti-corruption check)

                uint8_t *bytes = (uint8_t *)blackData;

                bool dataCorrupted = false;

                for (uint64_t i = 0; i < mglUpMin(dataSize, 1024); i++) { // Check first 1KB only for performance

                    if (bytes[i] != 0) {

                        dataCorrupted = true;

                        break;

                    }

                }

                if (dataCorrupted) {

                    fprintf(stderr, "MGL SECURITY ERROR: blackData corruption detected - memory safety issue\n");

                    free(blackData);

                    return;
                }


                fprintf(stderr, "MGL INFO: All validations passed for texture fill (size=%lu, bytesPerRow=%lu)\n", (unsigned long)dataSize, (unsigned long)bytesPerRow);


                // ULTRA-DEFENSIVE: Final validation immediately before Metal API call

                // This prevents race conditions and memory corruption between validation and use

                if (!blackData) {

                    fprintf(stderr, "MGL CRITICAL ERROR: blackData became NULL before Metal call - RACE CONDITION DETECTED\n");

                    free(blackData);

                    return;
                }

                if (!texture) {

                    fprintf(stderr, "MGL CRITICAL ERROR: Metal texture became NULL before Metal call - RACE CONDITION DETECTED\n");

                    free(blackData);

                    return;
                }

                if (bytesPerRow == 0 || dataSize == 0) {

                    fprintf(stderr, "MGL CRITICAL ERROR: Parameters became invalid before Metal call - RACE CONDITION DETECTED\n");

                    free(blackData);

                    return;
                }


                // Additional verification: Check if Metal texture is still valid

                if (mglPdTextureInfo(texture).width == 0 || mglPdTextureInfo(texture).height == 0) {

                    fprintf(stderr, "MGL CRITICAL ERROR: Metal texture dimensions became invalid before Metal call\n");

                    free(blackData);

                    return;
                }


                // Final integrity check: Verify blackData still contains expected zeros

                uint8_t *finalCheck = (uint8_t *)blackData;

                bool finalCorruption = false;

                for (uint64_t i = 0; i < mglUpMin(dataSize, 256); i++) { // Check first 256 bytes

                    if (finalCheck[i] != 0) {

                        finalCorruption = true;

                        break;

                    }

                }

                if (finalCorruption) {

                    fprintf(stderr, "MGL CRITICAL ERROR: Memory corruption detected immediately before Metal call\n");

                    free(blackData);

                    return;
                }


                fprintf(stderr, "MGL INFO: FIXING: Implementing proper texture filling for Apple Metal compatibility\n");


                // PROPER FIX: Use Apple Metal-compatible texture filling approach

                // The issue was using incorrect bytesPerRow and region parameters

                fprintf(stderr, "MGL INFO: Implementing Metal-compliant texture fill operations\n");


                // Use Metal's standard pattern for texture filling.

                uint64_t pixelSize = bytesPerPixel;

                uint64_t properBytesPerRow = mglPdTextureInfo(texture).width * pixelSize;


                // Ensure proper alignment for Apple Metal driver

                if (properBytesPerRow % 64 != 0) {

                    properBytesPerRow = ((properBytesPerRow + 63) / 64) * 64;

                }


                // Fill the entire level. A previous 1x1 safety fill left large textures

                // mostly uninitialized while their Metal backing existed.

                MGLRegionValue properRegion = mglTextureRegion2D(0, 0, mglPdTextureInfo(texture).width, mglPdTextureInfo(texture).height);


                // Create properly aligned texture data buffer

                uint64_t fillSize = properBytesPerRow * properRegion.size.height;

                uint8_t *properData = (uint8_t *)calloc(fillSize, 1);


                if (properData) {

                    // Initialize with safe texture data (transparent black with alpha = 0)

                    for (uint64_t y = 0; y < properRegion.size.height; y++) {

                        uint8_t *row = properData + (y * properBytesPerRow);

                        for (uint64_t x = 0; x < properRegion.size.width; x++) {

                            uint8_t *pixel = row + (x * pixelSize);

                            pixel[0] = 0;  // R

                            if (pixelSize > 1) pixel[1] = 0;  // G

                            if (pixelSize > 2) pixel[2] = 0;  // B

                            if (pixelSize > 3) pixel[3] = 0; // A = transparent for uninitialized color data

                        }

                    }


                    /* The .m wrapped both fill attempts in @try/@catch; the C
                     * port runs each attempt through the shell's guarded call and
                     * keeps the same catch logic (rule 58 (b)). */
                    char safeFillFailure[256] = {0};
                    MGLRendererStateAreas safeAreas;
                    mglRendererStateAreasPort(renderer, &safeAreas);
                    /* The outer @try's prologue logs (they used to sit inside
                     * that block; the C port runs the attempt through the
                     * shell's guard instead, so they are emitted first). */
                    fprintf(stderr, "MGL INFO: Performing Metal-compliant texture fill:\n");
                    fprintf(stderr, "  - Region: %dx%d\n",
                            (int)properRegion.size.width,
                            (int)properRegion.size.height);
                    fprintf(stderr, "  - bytesPerRow: %lu\n",
                            (unsigned long)properBytesPerRow);
                    fprintf(stderr, "  - dataSize: %lu\n",
                            (unsigned long)fillSize);
                    // ALTERNATIVE APPROACH: Safe texture filling without replaceRegion
                    fprintf(stderr,
                            "MGL INFO: Using alternative texture filling methods (AGX-safe)\n");

                    MglUpSafeFillCtx safeFillCtx = {
                        texture, tex,
                        mglRendererBackendGetDevice(safeAreas.backend),
                        properData, properRegion, properBytesPerRow, fillSize,
                        dataSize };
                    if (!mglPlatformShellGuardedCallCtxReason(
                            renderer, "safe texture fill", mglUpSafeFillBody,
                            &safeFillCtx, safeFillFailure,
                            sizeof(safeFillFailure))) {
                        fprintf(stderr,
                                "MGL WARNING: Buffer-based texture fill failed - trying alternative\n");

                        // ALTERNATIVE 2: Simple direct color filling for basic cases
                        // (the .m's outer @try covered this call as well, so it
                        // is guarded too; the name/reason split of the original
                        // log collapses into the reason text the shell hands back)
                        char altFailure[256] = {0};
                        MglUpFillCtx altCtx = { texture, tex };
                        if (!mglPlatformShellGuardedCallCtxReason(
                                renderer, "safe texture fill alternative",
                                mglUpFillBody, &altCtx, altFailure,
                                sizeof(altFailure))) {
                            fprintf(stderr,
                                    "MGL ERROR: Metal texture fill failed - investigating root cause\n");
                            fprintf(stderr, "MGL ERROR: Exception: %s\n",
                                    altFailure[0] ? altFailure : "(null)");
                            fprintf(stderr,
                                    "MGL INFO: This indicates our parameters are still incompatible with AGX driver\n");
                        }
                    }


                    free(properData);

                } else {

                    fprintf(stderr, "MGL ERROR: Failed to allocate properly aligned texture data\n");

                }

                free(blackData);

            } else {

                fprintf(stderr, "MGL ERROR: Failed to allocate aligned memory for texture fill (%lu bytes)\n", (unsigned long)dataSize);

            }

        }

    }
}
