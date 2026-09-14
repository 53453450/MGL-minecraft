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
        mglRendererEndRenderEncodingPort(renderer);

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
