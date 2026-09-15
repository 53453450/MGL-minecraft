/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_readback_ops.c - C homes of MGLRenderer(Texture)'s readback
 * family (P0-1, log 181): the staging helper plus the colour, depth and
 * integer readbacks.  The .m methods keep their call sites (they are called
 * from mtlGetTexImage: and the readPixels paths in the same file).
 */

#include "mgl_texture_readback_ops.h"

#include "error.h"                 /* mglDispatchError */
#include "mgl_blit_color_state.h"  /* mglBlitResolvedReadbackTexture */
#include "mgl_metal_ref.h"         /* mglReleaseMetalObjNoNull */
#include "mgl_readback.h"          /* mglMetalReadback* */
#include "mgl_readback_policy.h"   /* depth/integer readback plans */
#include "mgl_renderer_backend.h"  /* mglRendererBackendGetDevice */
#include "mgl_render_pass_manager.h"
#include "mgl_renderer_ports.h"
#include "mgl_sync.h"
#include "mgl_texture_compat.h"
#include "mgl_trace_log.h"

#include "mgl_render.h"
#include "mgl_render_pass_manager_ops.h" /* mglRenderPass* C entries */
#include "mgl_thread_affinity.h"        /* MGL_ASSERT_GL_THREAD */
#include "mgl_texture_bind.h"           /* mglRendererBindMTLTexture */
#include "mgl_draw_buffer.h"            /* mglDefaultDrawBufferIndexForGL */
#include "mgl_texture_readback_clear.h" /* pending-clear helpers */
#include "mgl_pso_format_class.h"       /* mglRenderColorAttachmentBitSet */
#include "mgl_frame_activity.h"
#include "error.h"
#include "mgl_gpu_recovery.h"

#include <stdio.h>
#include <string.h>

/* The .m's file-local constants this TU needs (values copied verbatim). */
enum {
    MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED = 0u,
    MGL_PD_TEXTURE_STORAGE_PRIVATE = 2u,
};

extern void mglPlatformShellSetContext(void *renderer, GLMContext glm_ctx);
extern MGLSizeValue mglPlatformShellApplyPendingDrawableSize(void *renderer);

/* MGLRenderer.m defines this; the only declaration is in the Objective-C
 * MGLRenderer+Texture_Private.h. */
extern void mglMetalCopyRows(const uint8_t *src, uint64_t src_bytes_per_row,
                             uint8_t *dst, uint64_t dst_bytes_per_row,
                             uint64_t copy_bytes_per_row, uint64_t row_count,
                             signed char flip_y);

/* Twin of the .m's mglTextureInfo (and of the same helper in
 * mgl_render_pass_manager_ops.c). */
static MGLRenderTextureInfo mglPdTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) (void)mglRenderGetTextureInfo(texture, &info);
    return info;
}

static uint64_t mglPdMaxU64(uint64_t a, uint64_t b) { return a > b ? a : b; }
static uint64_t mglPdMinU64(uint64_t a, uint64_t b) { return a < b ? a : b; }
static int64_t mglPdMaxI64(int64_t a, int64_t b) { return a > b ? a : b; }
static int64_t mglPdMinI64(int64_t a, int64_t b) { return a < b ? a : b; }

/* The .m used these statics; they never took ownership of the source, only of
 * what the resolve helpers handed back. */
static void *mglPdResolveOwned(void *renderer, void *sourceTexture,
                               uint64_t sourceLevel, uint64_t sourceSlice,
                               uint64_t sourceDepthPlane, const char *reason)
{
    return mglBlitResolvedReadbackTexture(renderer, sourceTexture, sourceLevel,
                                          sourceSlice, sourceDepthPlane, reason);
}

static void *mglPdTextureCreateBuffer(uint64_t length, uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBuffer(length, options, NULL, &buffer) == 0 && buffer) {
        return buffer;
    }
    return NULL;
}

/* Twin of the .m's mglTextureGetBytes (reports failure instead of raising). */
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

static void mglPdTextureCopyTextureToBuffer(
    void *encoder, void *source, uint64_t sourceSlice, uint64_t sourceLevel,
    MGLOriginValue sourceOrigin, MGLSizeValue sourceSize, void *destination,
    uint64_t destinationOffset, uint64_t bytesPerRow, uint64_t bytesPerImage)
{
    (void)mglRenderBlitCopyTextureToBuffer(
        encoder, source, sourceSlice, sourceLevel, sourceOrigin.x, sourceOrigin.y,
        sourceOrigin.z, sourceSize.width, sourceSize.height, sourceSize.depth,
        destination, destinationOffset, bytesPerRow, bytesPerImage);
}

static void mglPdTextureEndBlitEncoder(void *encoder)
{
    if (!encoder) return;
    (void)mglRenderEndBlitEncoder(encoder);
}

typedef struct MglPdStageCopyCtx_t {
    void *encoder;
    void *source;
    uint64_t source_slice;
    uint64_t source_level;
    MGLOriginValue copy_origin;
    MGLSizeValue copy_size;
    void *destination;
    uint64_t bytes_per_row;
    uint64_t bytes_per_image;
    int encoder_ended;
} MglPdStageCopyCtx;

static int mglPdStageCopyTryBody(void *renderer, void *rawCtx)
{
    MglPdStageCopyCtx *ctx = (MglPdStageCopyCtx *)rawCtx;
    (void)renderer;
    mglPdTextureCopyTextureToBuffer(
        ctx->encoder, ctx->source, ctx->source_slice, ctx->source_level,
        ctx->copy_origin, ctx->copy_size, ctx->destination, 0u,
        ctx->bytes_per_row, ctx->bytes_per_image);
    mglPdTextureEndBlitEncoder(ctx->encoder);
    ctx->encoder_ended = 1;
    return 1;
}

/* The nested @try that ends the encoder from the catch block. */
static int mglPdStageEndEncoderBody(void *renderer, void *rawCtx)
{
    MglPdStageCopyCtx *ctx = (MglPdStageCopyCtx *)rawCtx;
    (void)renderer;
    mglPdTextureEndBlitEncoder(ctx->encoder);
    ctx->encoder_ended = 1;
    return 1;
}

/* -readbackStageAndWaitTexture:... */
void *mglTextureReadbackStageAndWait(void *renderer, void *sourceTexture,
                                     uint64_t sourceLevel, uint64_t sourceSlice,
                                     uint64_t sourceDepthPlane,
                                     MGLOriginValue copyOrigin,
                                     MGLSizeValue copySize,
                                     uint64_t stagingBytesPerRow,
                                     uint64_t stagingSize, const char *reason,
                                     const char *logKind, int *outSuccess)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLRenderPassManager *manager = areas.render_pass_manager;
    MGLCommandState *commandState = areas.command;

    if (outSuccess) {
        *outSuccess = 1;
    }

    void *readBuffer = mglPdTextureCreateBuffer(
        stagingSize, MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED);
    void *blitEncoder =
        readBuffer ? mglRenderCreateBlitEncoderBorrowed(
                         commandState->currentCommandBufferOwner)
                   : NULL;
    if (!readBuffer || !blitEncoder) {
        fprintf(stderr,
                "MGL WARNING: readPixels failed to create %s resources for %s\n",
                logKind ? logKind : "readback", reason ? reason : "unknown");
        mglDispatchError(ctx, __func__, (GLenum)mglRenderErrorOutOfMemory());
        if (readBuffer) mglReleaseMetalObjNoNull(readBuffer);
        return NULL;
    }

    /* @try of the texture copy: on exception the .m ended the encoder inside
     * a nested @try, reported the failure and returned nil (outSuccess keeps
     * the YES written at entry). */
    MglPdStageCopyCtx copyCtx = {blitEncoder, sourceTexture, sourceSlice,
                                 sourceLevel, copyOrigin, copySize, readBuffer,
                                 stagingBytesPerRow, stagingSize};
    if (!mglPlatformShellGuardedCallCtx(renderer, "readback texture copy",
                                        mglPdStageCopyTryBody, &copyCtx, NULL)) {
        if (!copyCtx.encoder_ended) {
            (void)mglPlatformShellGuardedCallCtx(
                renderer, "readback blit encoder end",
                mglPdStageEndEncoderBody, &copyCtx, NULL);
        }
        fprintf(stderr,
                "MGL WARNING: readPixels %s texture copy failed for %s\n",
                logKind ? logKind : "readback", reason ? reason : "unknown");
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        mglReleaseMetalObjNoNull(readBuffer);
        return NULL;
    }

    void *readbackCommandBuffer =
        mglPassManagerDetachCurrentCommandBufferForSubmission(manager);
    MGLRenderCommandBufferTransaction readbackTransaction = {0};
    const int readbackTransactionResult =
        mglPassManagerCommitCommandBufferTransaction(
            manager, readbackCommandBuffer,
            areas.gpu_recovery_command_owner
                ? *areas.gpu_recovery_command_owner
                : NULL,
            1, &readbackTransaction);
    if (readbackTransactionResult != 0 || readbackTransaction.has_error) {
        fprintf(stderr,
                "MGL WARNING: readPixels %s owner transaction failed for %s\n",
                logKind ? logKind : "readback", reason ? reason : "unknown");
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        if (outSuccess) {
            *outSuccess = 0;
        }
    }

    if (readbackTransactionResult == 0 &&
        readbackTransaction.completion.has_error) {
        fprintf(stderr,
                "MGL WARNING: readPixels %s command buffer failed for %s: %s; returning zeroed data\n",
                logKind ? logKind : "readback", reason ? reason : "unknown",
                mglRenderCommandBufferErrorDescription(
                    &readbackTransaction.completion));
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        if (outSuccess) {
            *outSuccess = 0;
        }
    }

    mglPassManagerReleaseDetachedCommandBufferIfOwned(manager,
                                                      readbackCommandBuffer);
    (void)mglRenderPassNewCommandBufferLocked(renderer);
    return readBuffer;
}

/* -mglReadColorTextureAsBGRA8:... */
int mglTextureReadColorAsBGRA8(void *renderer, void *sourceTexture,
                               uint64_t sourceLevel, uint64_t sourceSlice,
                               uint64_t sourceDepthPlane, void *pixelBytes,
                               uint64_t bytesPerRow, uint64_t bytesPerImage,
                               MGLRegionValue region, const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLCommandState *commandState = areas.command;
    int result = 0;
    void *ownedResolved = NULL;

    uint64_t readSize = bytesPerImage;
    if (readSize == 0u && bytesPerRow > 0u) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0u) {
        return 0;
    }

    if (!sourceTexture || region.size.width == 0u || region.size.height == 0u) {
        return sourceTexture != NULL;
    }

    if (mglRenderTextureIsFramebufferOnly(sourceTexture)) {
        static uint64_t s_framebufferOnlyReadCount = 0;
        const uint64_t hit = ++s_framebufferOnlyReadCount;
        if (hit <= 16ull || (hit % 256ull) == 0ull) {
            fprintf(stderr,
                    "MGL WARNING: readPixels cannot read framebufferOnly texture for %s hit=%llu\n",
                    reason ? reason : "unknown", (unsigned long long)hit);
        }
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return 0;
    }

    if (!mglMetalReadbackFormatIsBGRA8Compatible(
            mglPdTextureInfo(sourceTexture).pixel_format)) {
        static uint64_t s_unsupportedReadFormatCount = 0;
        const uint64_t hit = ++s_unsupportedReadFormatCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            fprintf(stderr,
                    "MGL WARNING: readPixels unsupported Metal color readback format=%lu for %s hit=%llu; returning zero data\n",
                    (unsigned long)mglPdTextureInfo(sourceTexture).pixel_format,
                    reason ? reason : "unknown", (unsigned long long)hit);
        }
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return 0;
    }

    if (mglPdTextureInfo(sourceTexture).sample_count > 1u) {
        ownedResolved = mglPdResolveOwned(renderer, sourceTexture, sourceLevel,
                                          sourceSlice, sourceDepthPlane, reason);
        sourceTexture = ownedResolved;
        if (!sourceTexture) {
            goto done;
        }
        sourceLevel = 0u;
        sourceSlice = 0u;
        sourceDepthPlane = 0u;
    }

    if (bytesPerRow < region.size.width * 4u) {
        fprintf(stderr,
                "MGL WARNING: readPixels destination row too small row=%lu width=%lu for %s\n",
                (unsigned long)bytesPerRow, (unsigned long)region.size.width,
                reason ? reason : "unknown");
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        goto done;
    }

    uint64_t levelWidth = mglPdTextureInfo(sourceTexture).width;
    uint64_t levelHeight = mglPdTextureInfo(sourceTexture).height;
    if (sourceLevel > 0u) {
        if (sourceLevel >=
            mglPdTextureInfo(sourceTexture).mipmap_level_count) {
            fprintf(stderr,
                    "MGL WARNING: readPixels invalid mip level=%lu mipLevels=%lu for %s\n",
                    (unsigned long)sourceLevel,
                    (unsigned long)mglPdTextureInfo(sourceTexture)
                        .mipmap_level_count,
                    reason ? reason : "unknown");
            mglDispatchError(ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            goto done;
        }
        levelWidth = mglPdMaxU64(
            1u, mglPdTextureInfo(sourceTexture).width >> sourceLevel);
        levelHeight = mglPdMaxU64(
            1u, mglPdTextureInfo(sourceTexture).height >> sourceLevel);
    }

    MGLRenderReadTextureRegionClip clip = {0};
    mglRenderReadTextureRegionClip(
        (int64_t)region.origin.x, (int64_t)region.origin.y,
        (int64_t)region.size.width, (int64_t)region.size.height,
        (int64_t)levelWidth, (int64_t)levelHeight, &clip);
    const int64_t copyW = (int64_t)clip.copy_w;
    const int64_t copyH = (int64_t)clip.copy_h;
    const int64_t dstX = (int64_t)clip.dst_x;
    const int64_t dstY = (int64_t)clip.dst_y;
    const int64_t metalSrcX = (int64_t)clip.metal_src_x;
    const int64_t metalSrcY = (int64_t)clip.metal_src_y;
    if (clip.empty) {
        result = 1;
        goto done;
    }

    const uint64_t stagingBytesPerPixel = mglMetalReadbackBytesPerPixel(
        mglPdTextureInfo(sourceTexture).pixel_format);
    const uint64_t stagingBytesPerRow =
        (uint64_t)copyW * stagingBytesPerPixel;
    const uint64_t stagingSize = stagingBytesPerRow * (uint64_t)copyH;
    const uint64_t outputBytesPerRow = (uint64_t)copyW * 4u;
    if (stagingSize == 0u) {
        result = 1;
        goto done;
    }

    const uint64_t dstOffset =
        ((uint64_t)dstY * bytesPerRow) + ((uint64_t)dstX * 4u);
    if (dstOffset >= readSize || outputBytesPerRow > bytesPerRow ||
        ((uint64_t)copyH - 1u) * bytesPerRow + outputBytesPerRow >
            readSize - dstOffset) {
        fprintf(stderr,
                "MGL WARNING: readPixels clipped copy exceeds destination storage for %s\n",
                reason ? reason : "unknown");
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        goto done;
    }

    if (!mglRenderPassEnsureWritableCommandBufferLocked(
            renderer, "mglReadColorTextureAsBGRA8")) {
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        goto done;
    }

    int readbackSuccess = 1;
    void *readBuffer = mglTextureReadbackStageAndWait(
        renderer, sourceTexture, sourceLevel, sourceSlice, sourceDepthPlane,
        mglTextureOrigin((uint64_t)metalSrcX, (uint64_t)metalSrcY,
                         sourceDepthPlane),
        mglTextureSize((uint64_t)copyW, (uint64_t)copyH, 1u),
        stagingBytesPerRow, stagingSize, reason, "readback", &readbackSuccess);
    if (!readBuffer) {
        goto done;
    }

    if (readbackSuccess) {
        uint8_t *dst = ((uint8_t *)pixelBytes) + dstOffset;
        mglMetalCopyTextureBytesToBGRA8(
            (const uint8_t *)mglPdTextureBufferContents(readBuffer),
            stagingBytesPerRow, dst, bytesPerRow, (uint64_t)copyW,
            (uint64_t)copyH, mglPdTextureInfo(sourceTexture).pixel_format, 1);
    }
    result = readbackSuccess;
    /* The staging buffer's +1 is ours now (the .m let ARC autorelease it). */
    mglReleaseMetalObjNoNull(readBuffer);

done:
    if (ownedResolved) mglReleaseMetalObjNoNull(ownedResolved);
    (void)commandState;
    return result;
}

/* -mglReadDepthTextureAsFloat:... */
int mglTextureReadDepthAsFloat(void *renderer, void *sourceTexture,
                               uint64_t sourceLevel, uint64_t sourceSlice,
                               uint64_t sourceDepthPlane, void *pixelBytes,
                               uint64_t bytesPerRow, uint64_t bytesPerImage,
                               MGLRegionValue region, const char *reason)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    int result = 0;
    void *ownedResolved = NULL;

    uint64_t readSize = bytesPerImage;
    if (readSize == 0u && bytesPerRow > 0u) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0u) {
        return 0;
    }

    if (!sourceTexture || region.size.width == 0u || region.size.height == 0u) {
        return sourceTexture != NULL;
    }

    int isDepth16 = 0;
    int isPackedD32FS8 = 0;
    if (!mglRenderDepthReadbackPlan(
            (uint32_t)mglPdTextureInfo(sourceTexture).pixel_format, &isDepth16,
            &isPackedD32FS8)) {
        static uint64_t s_unsupportedDepthReadFormatCount = 0;
        const uint64_t hit = ++s_unsupportedDepthReadFormatCount;
        if (hit <= 32ull || (hit % 256ull) == 0ull) {
            fprintf(stderr,
                    "MGL WARNING: readPixels unsupported Metal depth readback format=%lu for %s hit=%llu\n",
                    (unsigned long)mglPdTextureInfo(sourceTexture).pixel_format,
                    reason ? reason : "unknown", (unsigned long long)hit);
        }
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return 0;
    }
    int sourceIsDepthStencil = isPackedD32FS8 != 0;
    int sourceIsDepth16 = isDepth16 != 0;

    if (mglPdTextureInfo(sourceTexture).sample_count > 1u) {
        ownedResolved = mglPdResolveOwned(renderer, sourceTexture, sourceLevel,
                                          sourceSlice, sourceDepthPlane, reason);
        sourceTexture = ownedResolved;
        if (!sourceTexture) {
            goto done;
        }
        sourceLevel = 0u;
        sourceSlice = 0u;
        sourceDepthPlane = 0u;
    }

    if (sourceIsDepthStencil) {
        void *resolved = mglBlitDepthFloatTextureForReadback(
            renderer, sourceTexture, reason);
        if (ownedResolved) mglReleaseMetalObjNoNull(ownedResolved);
        ownedResolved = resolved;
        sourceTexture = ownedResolved;
        if (!sourceTexture) {
            goto done;
        }
        sourceLevel = 0u;
        sourceSlice = 0u;
        sourceDepthPlane = 0u;
        sourceIsDepthStencil = 0;
        sourceIsDepth16 = 0;
    }

    if (bytesPerRow < region.size.width * sizeof(float)) {
        fprintf(stderr,
                "MGL WARNING: readPixels depth destination row too small row=%lu width=%lu for %s\n",
                (unsigned long)bytesPerRow, (unsigned long)region.size.width,
                reason ? reason : "unknown");
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        goto done;
    }

    uint64_t levelWidth = mglPdTextureInfo(sourceTexture).width;
    uint64_t levelHeight = mglPdTextureInfo(sourceTexture).height;
    if (sourceLevel > 0u) {
        if (sourceLevel >=
            mglPdTextureInfo(sourceTexture).mipmap_level_count) {
            fprintf(stderr,
                    "MGL WARNING: readPixels invalid depth mip level=%lu mipLevels=%lu for %s\n",
                    (unsigned long)sourceLevel,
                    (unsigned long)mglPdTextureInfo(sourceTexture)
                        .mipmap_level_count,
                    reason ? reason : "unknown");
            mglDispatchError(ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            goto done;
        }
        levelWidth = mglPdMaxU64(
            1u, mglPdTextureInfo(sourceTexture).width >> sourceLevel);
        levelHeight = mglPdMaxU64(
            1u, mglPdTextureInfo(sourceTexture).height >> sourceLevel);
    }

    MGLRenderReadTextureRegionClip clip = {0};
    mglRenderReadTextureRegionClip(
        (int64_t)region.origin.x, (int64_t)region.origin.y,
        (int64_t)region.size.width, (int64_t)region.size.height,
        (int64_t)levelWidth, (int64_t)levelHeight, &clip);
    const int64_t copyW = (int64_t)clip.copy_w;
    const int64_t copyH = (int64_t)clip.copy_h;
    const int64_t dstX = (int64_t)clip.dst_x;
    const int64_t dstY = (int64_t)clip.dst_y;
    const int64_t metalSrcX = (int64_t)clip.metal_src_x;
    const int64_t metalSrcY = (int64_t)clip.metal_src_y;
    if (clip.empty) {
        result = 1;
        goto done;
    }

    const uint64_t sourceDepthBytes = sourceIsDepthStencil
                                          ? sizeof(float)
                                          : (sourceIsDepth16 ? sizeof(uint16_t)
                                                             : sizeof(float));
    uint64_t stagingBytesPerRow = (uint64_t)copyW * sourceDepthBytes;
    /* Metal requires destinationBytesPerRow to be a multiple of 4 bytes on
     * macOS. Depth16Unorm (2 bytes/pixel) can produce a non-aligned row for
     * narrow reads, causing the blit to return zeros/garbage. */
    stagingBytesPerRow = (stagingBytesPerRow + 3u) & ~3u;
    const uint64_t stagingSize = stagingBytesPerRow * (uint64_t)copyH;
    if (stagingSize == 0u) {
        result = 1;
        goto done;
    }

    const uint64_t dstOffset =
        ((uint64_t)dstY * bytesPerRow) + ((uint64_t)dstX * sizeof(float));
    const uint64_t destinationCopyBytesPerRow = (uint64_t)copyW * sizeof(float);
    if (dstOffset >= readSize ||
        destinationCopyBytesPerRow > bytesPerRow ||
        ((uint64_t)copyH - 1u) * bytesPerRow + destinationCopyBytesPerRow >
            readSize - dstOffset) {
        fprintf(stderr,
                "MGL WARNING: readPixels clipped depth copy exceeds destination storage for %s\n",
                reason ? reason : "unknown");
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        goto done;
    }

    if (!mglRenderPassEnsureWritableCommandBufferLocked(
            renderer, "mglReadDepthTextureAsFloat")) {
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        goto done;
    }

    int readbackSuccess = 1;
    void *readBuffer = mglTextureReadbackStageAndWait(
        renderer, sourceTexture, sourceLevel, sourceSlice, sourceDepthPlane,
        mglTextureOrigin((uint64_t)metalSrcX, (uint64_t)metalSrcY,
                         sourceDepthPlane),
        mglTextureSize((uint64_t)copyW, (uint64_t)copyH, 1u),
        stagingBytesPerRow, stagingSize, reason, "depth readback",
        &readbackSuccess);
    if (!readBuffer) {
        goto done;
    }

    if (readbackSuccess) {
        uint8_t *dst = ((uint8_t *)pixelBytes) + dstOffset;
        if (sourceIsDepthStencil || sourceIsDepth16) {
            /* Depth16 / unpacked depth-float -> GL float. */
            mglRenderCopyDepthTextureBytesToFloat(
                mglPdTextureBufferContents(readBuffer), stagingBytesPerRow, dst,
                bytesPerRow, (uint64_t)copyW, (uint64_t)copyH,
                sourceDepthBytes, sourceIsDepth16 ? 1 : 0, 1);
        } else {
            mglMetalCopyRows(
                (const uint8_t *)mglPdTextureBufferContents(readBuffer),
                stagingBytesPerRow, dst, bytesPerRow, stagingBytesPerRow,
                (uint64_t)copyH, 1);
        }
    }
    result = readbackSuccess;
    mglReleaseMetalObjNoNull(readBuffer);

done:
    if (ownedResolved) mglReleaseMetalObjNoNull(ownedResolved);
    return result;
}

/* -mglReadIntegerTextureAsRGBA32:... */
int mglTextureReadIntegerAsRGBA32(void *renderer, void *sourceTexture,
                                  void *pixelBytes, uint64_t bytesPerRow,
                                  uint64_t bytesPerImage, MGLRegionValue region,
                                  uint64_t outputComponents,
                                  uint64_t outputComponentBytes,
                                  const int *componentMap, GLenum packedType,
                                  uint64_t mipmapLevel, uint64_t mtlSlice,
                                  int isRenderTarget)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGLRenderPassManager *manager = areas.render_pass_manager;
    MGLCommandState *commandState = areas.command;
    int result = 0;
    void *ownedResolved = NULL;

    MGLRenderIntegerReadbackSource src = {0};
    mglRenderIntegerReadbackSourceClassify(
        (uint32_t)mglPdTextureInfo(sourceTexture).pixel_format, &src);
    if (!src.recognized) {
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return 0;
    }
    const uint64_t componentCount = (uint64_t)src.component_count;
    const uint64_t sourceComponentBytes = (uint64_t)src.component_bytes;
    const int sourceSigned = src.source_signed != 0;
    const int sourceRGB10A2Uint = src.source_rgb10a2_uint != 0;

    if (mglPdTextureInfo(sourceTexture).sample_count > 1u) {
        ownedResolved = mglPdResolveOwned(renderer, sourceTexture, mipmapLevel,
                                          mtlSlice, 0u,
                                          "integer FBO readback");
        sourceTexture = ownedResolved;
        if (!sourceTexture) {
            goto done;
        }
        mipmapLevel = 0u;
        mtlSlice = 0u;
    }

    MGLRenderIntegerPackedType packed = {0};
    mglRenderIntegerReadbackPackedTypeClassify((uint32_t)packedType, &packed);
    const int isPackedType = packed.is_packed != 0;
    const uint32_t packedBitWidths[4] = {
        packed.bit_widths[0], packed.bit_widths[1], packed.bit_widths[2],
        packed.bit_widths[3]};
    const uint32_t packedShifts[4] = {
        packed.shifts[0], packed.shifts[1], packed.shifts[2], packed.shifts[3]};
    const uint64_t packedOutputBytes = (uint64_t)packed.output_bytes;
    if (packed.output_components > 0u) {
        outputComponents = (uint64_t)packed.output_components;
    }

    const uint64_t dstPixelBytes =
        isPackedType ? packedOutputBytes
                     : (outputComponentBytes * outputComponents);
    const uint64_t readSize =
        bytesPerImage ? bytesPerImage : bytesPerRow * region.size.height;
    if (!pixelBytes || bytesPerRow < region.size.width * dstPixelBytes) {
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        goto done;
    }

    (void)readSize;
    if (region.size.width == 0u || region.size.height == 0u) {
        result = 1;
        goto done;
    }

    const int64_t minX = mglPdMaxI64(0, (int64_t)region.origin.x);
    const int64_t minY = mglPdMaxI64(0, (int64_t)region.origin.y);
    const int64_t maxX =
        mglPdMinI64((int64_t)mglPdTextureInfo(sourceTexture).width,
                    (int64_t)region.origin.x + (int64_t)region.size.width);
    const int64_t maxY =
        mglPdMinI64((int64_t)mglPdTextureInfo(sourceTexture).height,
                    (int64_t)region.origin.y + (int64_t)region.size.height);
    const int64_t copyW = maxX - minX;
    const int64_t copyH = maxY - minY;
    if (copyW <= 0 || copyH <= 0) {
        result = 1;
        goto done;
    }

    const uint64_t srcPixelBytes =
        sourceRGB10A2Uint ? 4u : componentCount * sourceComponentBytes;
    const uint64_t srcBytesPerRow = (uint64_t)copyW * srcPixelBytes;
    const uint64_t stagingSize = srcBytesPerRow * (uint64_t)copyH;
    if (!mglRenderPassEnsureWritableCommandBufferLocked(
            renderer, "mglReadIntegerTextureAsRGBA32")) {
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        goto done;
    }

    void *readBuffer = mglPdTextureCreateBuffer(
        stagingSize, MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED);
    void *blit = readBuffer
                     ? mglRenderCreateBlitEncoderBorrowed(
                           commandState->currentCommandBufferOwner)
                     : NULL;
    if (!readBuffer || !blit) {
        mglDispatchError(ctx, __func__,
                         (GLenum)mglRenderErrorOutOfMemory());
        if (readBuffer) mglReleaseMetalObjNoNull(readBuffer);
        goto done;
    }

    /* Calculate the texture height at the specified mipmap level. */
    uint64_t levelHeight = mglPdTextureInfo(sourceTexture).height;
    if (mipmapLevel > 0u) {
        levelHeight = mglPdMaxU64(
            1u, mglPdTextureInfo(sourceTexture).height >> mipmapLevel);
    }
    /* Render-target textures are stored top-to-bottom in Metal (Metal y=0 = GL
     * y=levelHeight-1), so the blit source origin must be Y-flipped.
     * Non-render-target textures (e.g. storage images written via imageStore)
     * store data in GL order (Metal y=0 = GL y=0), so no source Y-flip is
     * needed.  Using the flipped origin for storage images would read the wrong
     * rows and corrupt the readback. */
    const uint64_t blitSrcY =
        isRenderTarget ? (levelHeight - (uint64_t)maxY) : (uint64_t)minY;
    mglPdTextureCopyTextureToBuffer(
        blit, sourceTexture, mtlSlice, mipmapLevel,
        mglTextureOrigin((uint64_t)minX, blitSrcY, 0u),
        mglTextureSize((uint64_t)copyW, (uint64_t)copyH, 1u), readBuffer, 0u,
        srcBytesPerRow, stagingSize);
    mglPdTextureEndBlitEncoder(blit);
    void *integerReadbackCommandBuffer =
        mglPassManagerDetachCurrentCommandBufferForSubmission(manager);
    MGLRenderCommandBufferTransaction integerReadbackTransaction = {0};
    const int integerReadbackResult = mglPassManagerCommitCommandBufferTransaction(
        manager, integerReadbackCommandBuffer,
        areas.gpu_recovery_command_owner ? *areas.gpu_recovery_command_owner
                                         : NULL,
        1, &integerReadbackTransaction);
    if (integerReadbackResult != 0 || integerReadbackTransaction.has_error) {
        fprintf(stderr,
                "MGL ERROR: integer texture readback owner transaction failed\n");
        mglPassManagerReleaseDetachedCommandBufferIfOwned(
            manager, integerReadbackCommandBuffer);
        mglReleaseMetalObjNoNull(readBuffer);
        goto done;
    }
    mglPassManagerReleaseDetachedCommandBufferIfOwned(
        manager, integerReadbackCommandBuffer);

    const uint64_t dstX = (uint64_t)(minX - (int64_t)region.origin.x);
    const uint64_t dstY = (uint64_t)(minY - (int64_t)region.origin.y);

    MGLRenderIntegerReadbackConvertParams convert = {
        .src = (const uint8_t *)mglPdTextureBufferContents(readBuffer),
        .src_bytes_per_row = srcBytesPerRow,
        .source_component_count = (uint32_t)componentCount,
        .source_component_bytes = (uint32_t)sourceComponentBytes,
        .source_signed = sourceSigned ? 1 : 0,
        .source_rgb10a2_uint = sourceRGB10A2Uint ? 1 : 0,
        .copy_w = (uint32_t)copyW,
        .copy_h = (uint32_t)copyH,
        .dst = (uint8_t *)pixelBytes,
        .dst_bytes_per_row = bytesPerRow,
        .dst_pixel_bytes = dstPixelBytes,
        .dst_x = dstX,
        .dst_y = dstY,
        .output_components = (uint32_t)outputComponents,
        .component_map = componentMap,
        .output_component_bytes = (uint32_t)outputComponentBytes,
        .packed_type = (uint32_t)packedType,
        .is_packed_type = isPackedType ? 1 : 0,
        .packed_bit_widths = packedBitWidths,
        .packed_shifts = packedShifts,
        .packed_output_bytes = (uint32_t)packedOutputBytes,
        /* Integer RTs retain GL row order after CPU upload + FragCoord remap;
         * flipping here Y-mirrors glGetTexImage. */
        .flip_y = 0,
    };
    if (mglRenderConvertIntegerReadback(&convert) != 0) {
        (void)mglRenderPassNewCommandBufferLocked(renderer);
        mglReleaseMetalObjNoNull(readBuffer);
        goto done;
    }

    (void)mglRenderPassNewCommandBufferLocked(renderer);
    result = 1;
    mglReleaseMetalObjNoNull(readBuffer);

done:
    if (ownedResolved) mglReleaseMetalObjNoNull(ownedResolved);
    return result;
}

/* === readPixels family (log 185) ======================================= */

/* -mtlReadDepthPixels:pixelBytes:bytesPerRow:bytesPerImage:fromRegion: */
void mglTextureReadDepthPixels(void *renderer, GLMContext glm_ctx,
                               void *pixelBytes, uint64_t bytesPerRow,
                               uint64_t bytesPerImage, MGLRegionValue region)
{
    MGL_ASSERT_GL_THREAD();
    mglPlatformShellSetContext(renderer, glm_ctx);
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    uint64_t readSize = bytesPerImage;
    if (readSize == 0u && bytesPerRow > 0u) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0u) {
        return;
    }

    if (glm_ctx->active_state->readbuffer) {
        Framebuffer *fbo = glm_ctx->active_state->readbuffer;
        FBOAttachment *attachment = fbo ? &fbo->depth : NULL;
        Texture *readTextureObject =
            mglRendererAttachmentTextureFor(glm_ctx, attachment);
        if (!readTextureObject) {
            fprintf(stderr,
                    "MGL WARNING: readPixels FBO has no depth attachment fbo=%u\n",
                    fbo ? (unsigned)fbo->name : 0u);
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return;
        }

        readTextureObject->is_render_target = 1;
        if (!mglRendererBindMTLTexture(renderer, readTextureObject) ||
            !readTextureObject->mtl_data) {
            fprintf(stderr,
                    "MGL WARNING: readPixels could not bind FBO depth texture fbo=%u tex=%u\n",
                    fbo ? (unsigned)fbo->name : 0u,
                    (unsigned)readTextureObject->name);
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return;
        }

        void *texture = readTextureObject->mtl_data;
        const MGLMetalAttachmentSubresource subresource =
            mglMetalAttachmentSubresourceForAttachment(attachment);

        mglRendererEndRenderEncodingPort(renderer);
        if (!mglRenderPassEnsureWritableCommandBufferLocked(
                renderer, "mtlReadDepthPixels.fbo")) {
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return;
        }
        mglTextureApplyPendingFBODepthClearForReadback(renderer, fbo, attachment,
                                                       readTextureObject,
                                                       texture);
        (void)mglTextureReadDepthAsFloat(
            renderer, texture, subresource.level, subresource.slice,
            subresource.depthPlane, pixelBytes, bytesPerRow, bytesPerImage,
            region, "FBO depth readback");
        return;
    }

    const GLuint drawBufferIndex =
        mglDefaultDrawBufferIndexForGL(glm_ctx->active_state->read_buffer);
    void *texture = NULL;
    if (drawBufferIndex < _MAX_DRAW_BUFFERS) {
        texture = mglRendererBackendGetDefaultDrawBufferAttachment(
            areas.backend, drawBufferIndex,
            MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_DEPTH);
    }

    if (!texture) {
        fprintf(stderr,
                "MGL WARNING: readPixels default framebuffer has no depth texture slot=%u\n",
                (unsigned)drawBufferIndex);
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    mglRendererEndRenderEncodingPort(renderer);
    if (!mglRenderPassEnsureWritableCommandBufferLocked(
            renderer, "mtlReadDepthPixels.default")) {
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }
    mglTextureApplyPendingDefaultDepthClear(renderer, texture);
    (void)mglTextureReadDepthAsFloat(renderer, texture, 0u, 0u, 0u, pixelBytes,
                                     bytesPerRow, bytesPerImage, region,
                                     "default framebuffer depth readback");
}

/* -mtlReadIntegerPixels:pixelBytes:bytesPerRow:bytesPerImage:fromRegion:
 *  format:type: */
void mglTextureReadIntegerPixels(void *renderer, GLMContext glm_ctx,
                                 void *pixelBytes, uint64_t bytesPerRow,
                                 uint64_t bytesPerImage, MGLRegionValue region,
                                 GLenum format, GLenum type)
{
    mglPlatformShellSetContext(renderer, glm_ctx);
    Framebuffer *fbo = glm_ctx ? glm_ctx->active_state->readbuffer : NULL;
    const GLenum readBuffer = glm_ctx ? glm_ctx->active_state->read_buffer
                                      : (GLenum)mglRenderEmptyDrawBuffer();
    uint32_t att = 0u;
    if (!fbo ||
        !mglRenderDrawBufferIsColorAttachment(
            (uint32_t)readBuffer, (uint32_t)MAX_COLOR_ATTACHMENTS, &att)) {
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    FBOAttachment *attachment = &fbo->color_attachments[att];
    Texture *textureObj = mglRendererAttachmentTextureFor(glm_ctx, attachment);
    if (!textureObj || !mglRendererBindMTLTexture(renderer, textureObj) ||
        !textureObj->mtl_data) {
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    void *texture = textureObj->mtl_data;
    const MGLMetalAttachmentSubresource subresource =
        mglMetalAttachmentSubresourceForAttachment(attachment);

    mglRendererEndRenderEncodingPort(renderer);
    if (!mglRenderPassEnsureWritableCommandBufferLocked(
            renderer, "mtlReadIntegerPixels.fbo")) {
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }
    mglTextureApplyPendingFBOColorClearForReadback(renderer, fbo, attachment,
                                                   textureObj, texture,
                                                   readBuffer);

    /* Determine output component count and component mapping.
     * componentMap[c] = source component index for output component c, or -1. */
    int componentMap[4] = {0, 1, 2, 3};
    const uint64_t outputComponents =
        (uint64_t)mglRenderIntegerFormatComponentMap((uint32_t)format,
                                                     componentMap);
    const uint64_t outputComponentBytes =
        (uint64_t)mglRenderIntegerTypeComponentBytes((uint32_t)type);

    (void)mglTextureReadIntegerAsRGBA32(
        renderer, texture, pixelBytes, bytesPerRow, bytesPerImage, region,
        outputComponents, outputComponentBytes, componentMap, type,
        subresource.level, subresource.slice, 1);
}

/* -mtlReadDrawable:pixelBytes:bytesPerRow:bytesPerImage:fromRegion: */
void mglTextureReadDrawable(void *renderer, GLMContext glm_ctx,
                            void *pixelBytes, uint64_t bytesPerRow,
                            uint64_t bytesPerImage, MGLRegionValue region)
{
    mglPlatformShellSetContext(renderer, glm_ctx);
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    uint64_t readSize = bytesPerImage;
    if (readSize == 0 && bytesPerRow > 0) {
        readSize = bytesPerRow * region.size.height;
    }
    if (!pixelBytes || readSize == 0) {
        return;
    }

    if (glm_ctx->active_state->readbuffer) {
        Framebuffer *fbo = glm_ctx->active_state->readbuffer;
        const GLenum readBuffer = glm_ctx->active_state->read_buffer;
        if (!fbo ||
            !mglRenderFBOReadBufferValid(
                (uint32_t)readBuffer,
                (uint32_t)glm_ctx->active_state->max_color_attachments,
                (uint32_t)MAX_COLOR_ATTACHMENTS)) {
            static uint64_t s_invalidReadFBOCount = 0;
            const uint64_t hit = ++s_invalidReadFBOCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                fprintf(stderr,
                        "MGL WARNING: readPixels invalid FBO read buffer=0x%x maxColor=%u hit=%llu; returning zero data\n",
                        (unsigned)readBuffer,
                        (unsigned)glm_ctx->active_state->max_color_attachments,
                        (unsigned long long)hit);
            }
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return;
        }

        uint32_t attachmentIndex = 0u;
        if (!mglRenderDrawBufferIsColorAttachment(
                (uint32_t)readBuffer, (uint32_t)MAX_COLOR_ATTACHMENTS,
                &attachmentIndex) ||
            !mglRenderColorAttachmentBitSet(
                (uint32_t)fbo->color_attachment_bitfield, attachmentIndex)) {
            static uint64_t s_missingReadAttachmentCount = 0;
            const uint64_t hit = ++s_missingReadAttachmentCount;
            if (hit <= 32ull || (hit % 256ull) == 0ull) {
                fprintf(stderr,
                        "MGL WARNING: readPixels FBO read attachment 0x%x is not attached fbo=%u hit=%llu; returning zero data\n",
                        (unsigned)readBuffer, (unsigned)fbo->name,
                        (unsigned long long)hit);
            }
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return;
        }

        FBOAttachment *attachment = &fbo->color_attachments[attachmentIndex];
        Texture *readTextureObject =
            mglRendererAttachmentTextureFor(glm_ctx, attachment);
        if (!readTextureObject) {
            fprintf(stderr,
                    "MGL WARNING: readPixels FBO attachment has no texture fbo=%u attachment=0x%x\n",
                    (unsigned)fbo->name, (unsigned)readBuffer);
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return;
        }

        readTextureObject->is_render_target = 1;
        if (!mglRendererBindMTLTexture(renderer, readTextureObject) ||
            !readTextureObject->mtl_data) {
            fprintf(stderr,
                    "MGL WARNING: readPixels could not bind FBO read texture fbo=%u attachment=0x%x tex=%u\n",
                    (unsigned)fbo->name, (unsigned)readBuffer,
                    (unsigned)readTextureObject->name);
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return;
        }

        void *texture = readTextureObject->mtl_data;
        const MGLMetalAttachmentSubresource subresource =
            mglMetalAttachmentSubresourceForAttachment(attachment);
        mglRendererEndRenderEncodingPort(renderer);
        if (!mglRenderPassEnsureWritableCommandBufferLocked(
                renderer, "mtlReadDrawable.fbo")) {
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            return;
        }
        mglTextureApplyPendingFBOColorClearForReadback(
            renderer, fbo, attachment, readTextureObject, texture, readBuffer);
        (void)mglTextureReadColorAsBGRA8(
            renderer, texture, subresource.level, subresource.slice,
            subresource.depthPlane, pixelBytes, bytesPerRow, bytesPerImage,
            region, "FBO color readback");
        return;
    }

    void *texture = NULL;

    uint32_t mappedDraw = 0u;
    if (!mglRenderDefaultReadBufferIndex(
            (uint32_t)glm_ctx->active_state->read_buffer, &mappedDraw)) {
        fprintf(stderr,
                "MGL WARNING: readPixels unsupported default read buffer=0x%x; returning zero data\n",
                (unsigned)glm_ctx->active_state->read_buffer);
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }
    const GLuint mgl_drawbuffer = (GLuint)(int)mappedDraw;

    if (mglRenderDefaultDrawBufferIsFront(mgl_drawbuffer)) {
        if (!areas.drawable) {
            (void)mglPlatformShellApplyPendingDrawableSize(renderer);
            (void)mglRendererNextDrawablePort(renderer);
        }
        texture = areas.drawable ? mglRendererDrawableTexturePort(renderer) : NULL;
    } else if (mglRenderDefaultDrawBufferIsOffscreen(mgl_drawbuffer,
                                                    _MAX_DRAW_BUFFERS)) {
        texture = mglRendererBackendGetDefaultDrawBufferAttachment(
            areas.backend, mgl_drawbuffer,
            MGL_RENDERER_BACKEND_DEFAULT_DRAW_BUFFER_COLOR);
    }

    if (!texture) {
        fprintf(stderr,
                "MGL WARNING: readPixels default drawbuffer slot=%u has no texture; returning zero data\n",
                (unsigned)mgl_drawbuffer);
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    mglRendererEndRenderEncodingPort(renderer);
    if (!mglRenderPassEnsureWritableCommandBufferLocked(
            renderer, "mtlReadDrawable.default")) {
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }
    if (mglRenderDefaultDrawBufferIsFront(mgl_drawbuffer)) {
        mglTextureApplyPendingDefaultColorClear(renderer, texture);
    }
    (void)mglTextureReadColorAsBGRA8(renderer, texture, 0u, 0u, 0u, pixelBytes,
                                     bytesPerRow, bytesPerImage, region,
                                     "default framebuffer readback");
}

/* -mtlGetTexImage:tex:pixelBytes:bytesPerRow:bytesPerImage:fromRegion:format:
 *  type:mipmapLevel:slice: */
typedef struct MglPdPreReadbackCtx_t {
    void *renderer;
    void *command_buffer;
} MglPdPreReadbackCtx;

/* @try of the pre-readback flush: the catch only logs a warning. */
static int mglPdPreReadbackTryBody(void *renderer, void *rawCtx)
{
    MglPdPreReadbackCtx *ctx = (MglPdPreReadbackCtx *)rawCtx;
    mglRendererCommitCommandBufferWithAGXRecovery(renderer, ctx->command_buffer);
    (void)mglRenderWaitCommandBuffer(ctx->command_buffer);
    return 1;
}

void mglTextureGetTexImage(void *renderer, GLMContext glm_ctx, Texture *tex,
                           void *pixelBytes, uint64_t bytesPerRow,
                           uint64_t bytesPerImage, MGLRegionValue region,
                           GLenum format, GLenum type, uint64_t level,
                           uint64_t slice)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *commandState = areas.command;
    void *texture = NULL;

    mglPlatformShellSetContext(renderer, glm_ctx);

    if (!tex) {
        fprintf(stderr, "MGL ERROR: mtlGetTexImage called with NULL texture\n");
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    if (!pixelBytes) {
        fprintf(stderr,
                "MGL WARNING: mtlGetTexImage called with NULL destination for texture %u\n",
                tex->name);
        return;
    }

    if (!tex->mtl_data && !mglRendererBindMTLTexture(renderer, tex)) {
        fprintf(stderr,
                "MGL ERROR: mtlGetTexImage failed to bind texture %u\n",
                tex->name);
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    texture = tex->mtl_data;
    if (!texture) {
        fprintf(stderr,
                "MGL ERROR: mtlGetTexImage texture %u has no Metal texture\n",
                tex->name);
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    if (mglRenderTextureIsFramebufferOnly(texture)) {
        fprintf(stderr,
                "MGL ERROR: Cannot read from framebuffer only texture %u\n\n",
                tex->name);
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    if (!mglRendererSynchronizeRenderPassForTextureReadbackPort(
            renderer, texture, "mtlGetTexImage")) {
        mglDispatchError(glm_ctx, __func__,
                         (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    /* Ensure any pending texture upload blit commands are committed before
     * reading back. Without this, getBytes may return stale/zero data because
     * the blit encoding the upload is still in the uncommitted command buffer. */
    mglRendererEndRenderEncodingPort(renderer);
    if (mglRenderCommandBufferOwnerHasCurrent(
            commandState->currentCommandBufferOwner) == 1) {
        void *pendingCB = mglPassManagerDetachCurrentCommandBufferForSubmission(
            areas.render_pass_manager);
        MglPdPreReadbackCtx preCtx = {renderer, pendingCB};
        (void)mglPlatformShellGuardedCallCtx(renderer, "pre-readback flush",
                                             mglPdPreReadbackTryBody, &preCtx,
                                             NULL);
        MGLRenderCommandBufferState pendingState = {0};
        (void)mglRenderGetCommandBufferState(pendingCB, &pendingState);
        if (pendingState.has_error) {
            fprintf(stderr,
                    "MGL WARNING: mtlGetTexImage pre-readback command buffer error: %s\n",
                    mglRenderCommandBufferErrorDescription(&pendingState));
        }
        (void)mglRenderPassNewCommandBufferLocked(renderer);
    }

    MGLRegionValue readRegion = region;
    uint64_t readSlice = slice;
    /* TextureType3D uses origin.z for depth planes; arrayLength is always 1.
     * Callers pass the depth index via `slice` (see mglGetTexImage's layer
     * loop). Remap so blit uses slice=0 and origin.z = depth plane. */
    if (mglPdTextureInfo(texture).texture_type == MGLTextureType3D) {
        readRegion.origin.z = slice;
        if (readRegion.size.depth < 1u) {
            readRegion.size.depth = 1u;
        }
        readSlice = 0u;
    }
    /* Single-sample pass-rendered RTs are stored top-row-first in Metal (NDC
     * y=-1 at high row addresses) and need a CPU Y-flip for GL's bottom-up
     * readPixels.  Multisample RTs already land in GL row order after resolve -
     * flipping them re-inverts DSA MSAA float/unorm getTexImage (3a8cb5c). */
    const int flipRenderTargetRows =
        tex->is_render_target && tex->samples <= 1u;

    /* Integer texture readback path: when the source texture is an integer
     * format and the output format is GL_*_INTEGER, use the dedicated integer
     * readback function that handles packed types and component mapping. */
    MGLRenderIntegerReadbackClassify classify = {0};
    mglRenderIntegerReadbackClassify(
        (uint32_t)mglPdTextureInfo(texture).pixel_format, (uint32_t)format,
        (uint32_t)type, &classify);

    if (classify.source_is_integer_texture && classify.output_is_integer_format) {
        /* Pass the original (non-Y-flipped) region.  The integer readback does
         * its own Y-flip on the blit source origin AND Y-flips the output rows,
         * so passing a pre-Y-flipped readRegion here would double-flip. */
        (void)mglTextureReadIntegerAsRGBA32(
            renderer, texture, pixelBytes, bytesPerRow, bytesPerImage,
            readRegion, (uint64_t)classify.output_components,
            (uint64_t)classify.output_component_bytes, classify.component_map,
            type, level, readSlice, tex->is_render_target ? 1 : 0);
        return;
    }

    const uint64_t dstPixelBytes = (uint64_t)sizeForFormatType(format, type);
    const int directR32FloatRead =
        mglRenderDirectR32FloatRead((uint32_t)mglPdTextureInfo(texture).pixel_format,
                                    (uint32_t)format, (uint32_t)type) != 0;
    int useBGRA8Conversion =
        (dstPixelBytes > 0u && readRegion.size.depth == 1u &&
         !directR32FloatRead &&
         mglMetalReadbackFormatIsBGRA8Compatible(
             mglPdTextureInfo(texture).pixel_format));

    /* MGL_TEXTURE_STORAGE_PRIVATE textures cannot be read directly with
     * getBytes: use a blit-to-buffer path to convert GPU-private tiled memory
     * to linear CPU memory. */
    if (mglPdTextureInfo(texture).storage_mode ==
        MGL_PD_TEXTURE_STORAGE_PRIVATE) {
        MGLRenderGetTexImagePlan plan = {0};
        mglRenderGetTexImagePlan(
            (uint32_t)mglPdTextureInfo(texture).pixel_format, (uint32_t)format,
            (uint32_t)type, (uint32_t)readRegion.size.width,
            (uint32_t)readRegion.size.height, (uint32_t)readRegion.size.depth,
            (uint32_t)dstPixelBytes,
            (uint32_t)mglMetalReadbackBytesPerPixel(
                mglPdTextureInfo(texture).pixel_format),
            mglMetalReadbackFormatIsBGRA8Compatible(
                mglPdTextureInfo(texture).pixel_format)
                ? 1
                : 0,
            (uint32_t)bytesPerRow, (uint32_t)bytesPerImage, 1, &plan);
        useBGRA8Conversion = plan.use_bgra8_conversion;
        const uint64_t rowBytes = (uint64_t)plan.row_bytes;
        const uint64_t imageBytes = (uint64_t)plan.image_bytes;
        const uint64_t totalBytes = (uint64_t)plan.total_bytes;

        void *stagingBuffer = mglPdTextureCreateBuffer(
            totalBytes, MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED);
        if (!stagingBuffer) {
            fprintf(stderr,
                    "MGL ERROR: mtlGetTexImage failed to allocate staging buffer for texture %u\n",
                    tex->name);
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorOutOfMemory());
            return;
        }

        void *blitCB = NULL;
        if (mglRenderCreateCommandBuffer(
                mglRendererBackendGetCommandQueue(areas.backend), &blitCB) != 0 ||
            !blitCB) {
            fprintf(stderr,
                    "MGL ERROR: mtlGetTexImage failed to create blit command buffer for texture %u\n",
                    tex->name);
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            mglReleaseMetalObjNoNull(stagingBuffer);
            return;
        }

        void *blitEncoder = NULL;
        if (mglRenderCreateBlitEncoder(blitCB, &blitEncoder) != 0 ||
            !blitEncoder) {
            fprintf(stderr,
                    "MGL ERROR: mtlGetTexImage failed to create blit encoder for texture %u\n",
                    tex->name);
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            mglReleaseMetalObjNoNull(stagingBuffer);
            return;
        }

        mglPdTextureCopyTextureToBuffer(blitEncoder, texture, readSlice, level,
                                        readRegion.origin, readRegion.size,
                                        stagingBuffer, 0, rowBytes, imageBytes);

        mglPdTextureEndBlitEncoder(blitEncoder);
        if (mglRenderCommitCommandBuffer(blitCB) != 0) {
            fprintf(stderr,
                    "MGL ERROR: Metal-cpp texture command-buffer commit failed\n");
        }
        (void)mglRenderWaitCommandBuffer(blitCB);

        MGLRenderCommandBufferState blitState = {0};
        (void)mglRenderGetCommandBufferState(blitCB, &blitState);
        if (blitState.has_error) {
            fprintf(stderr,
                    "MGL ERROR: mtlGetTexImage blit failed for texture %u: %s\n",
                    tex->name,
                    mglRenderCommandBufferErrorDescription(&blitState));
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            mglReleaseMetalObjNoNull(stagingBuffer);
            return;
        }

        if (useBGRA8Conversion) {
            if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL(
                    (const uint8_t *)mglPdTextureBufferContents(stagingBuffer),
                    rowBytes, (uint8_t *)pixelBytes, bytesPerRow,
                    readRegion.size.width, readRegion.size.height,
                    mglPdTextureInfo(texture).pixel_format, format, type,
                    flipRenderTargetRows)) {
                fprintf(stderr,
                        "MGL ERROR: mtlGetTexImage unsupported BGRA8 conversion texture=%u format=0x%x type=0x%x\n",
                        tex->name, (unsigned)format, (unsigned)type);
                mglDispatchError(glm_ctx, __func__,
                                 (GLenum)mglRenderErrorInvalidOperation());
            }
        } else if (flipRenderTargetRows && readRegion.size.depth == 1u) {
            mglMetalCopyRows(
                (const uint8_t *)mglPdTextureBufferContents(stagingBuffer),
                rowBytes, (uint8_t *)pixelBytes, bytesPerRow, rowBytes,
                readRegion.size.height, 1);
        } else {
            memcpy(pixelBytes, mglPdTextureBufferContents(stagingBuffer),
                   totalBytes);
        }
        if (mglTraceLogIsEnabled() &&
            mglRenderTraceR8RedUByte((uint32_t)tex->internalformat,
                                     (uint32_t)format, (uint32_t)type) &&
            readRegion.size.width > 0 && readRegion.size.height > 0) {
            const uint8_t *rb = (const uint8_t *)pixelBytes;
            mglTraceLog(
                "GET_TEX_IMAGE_R8 tex=%u target=0x%x isRT=%d fmt=%lu rowBytes=%lu dstBPR=%lu size=%lux%lu first=%u,%u,%u,%u,%u,%u,%u,%u",
                (unsigned)tex->name, (unsigned)tex->target,
                tex->is_render_target ? 1 : 0,
                (unsigned long)mglPdTextureInfo(texture).pixel_format,
                (unsigned long)rowBytes, (unsigned long)bytesPerRow,
                (unsigned long)readRegion.size.width,
                (unsigned long)readRegion.size.height, rb[0],
                rb[mglPdMinU64(1u, totalBytes - 1)],
                rb[mglPdMinU64(2u, totalBytes - 1)],
                rb[mglPdMinU64(3u, totalBytes - 1)],
                rb[mglPdMinU64(4u, totalBytes - 1)],
                rb[mglPdMinU64(5u, totalBytes - 1)],
                rb[mglPdMinU64(6u, totalBytes - 1)],
                rb[mglPdMinU64(7u, totalBytes - 1)]);
        }
        mglReleaseMetalObjNoNull(stagingBuffer);
        return;
    }

    /* The .m wrapped this block in @try/@catch; the only calls that could throw
     * are getBytes and the row copy, and their C twins report failure instead
     * (see mglPdTextureGetBytes), so each failure maps to the same log +
     * dispatch error the catch produced. */
    if (useBGRA8Conversion ||
        (flipRenderTargetRows && readRegion.size.depth == 1u)) {
        MGLRenderGetTexImagePlan plan = {0};
        mglRenderGetTexImagePlan(
            (uint32_t)mglPdTextureInfo(texture).pixel_format, (uint32_t)format,
            (uint32_t)type, (uint32_t)readRegion.size.width,
            (uint32_t)readRegion.size.height, (uint32_t)readRegion.size.depth,
            (uint32_t)dstPixelBytes,
            (uint32_t)mglMetalReadbackBytesPerPixel(
                mglPdTextureInfo(texture).pixel_format),
            mglMetalReadbackFormatIsBGRA8Compatible(
                mglPdTextureInfo(texture).pixel_format)
                ? 1
                : 0,
            (uint32_t)bytesPerRow, (uint32_t)bytesPerImage, 0, &plan);
        const uint64_t rowBytes = (uint64_t)plan.row_bytes;
        const uint64_t totalBytes = (uint64_t)plan.image_bytes;
        void *readback = calloc(1u, totalBytes ? totalBytes : 1u);
        if (!readback) {
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorOutOfMemory());
            return;
        }
        if (!mglPdTextureGetBytes(texture, readback, rowBytes, bytesPerImage,
                                  readRegion, level, readSlice, 1)) {
            fprintf(stderr,
                    "MGL ERROR: mtlGetTexImage texture read failed for texture %u\n",
                    tex->name);
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
            free(readback);
            return;
        }
        if (useBGRA8Conversion) {
            if (!mglMetalCopyBGRA8CompatibleTextureBytesToGL(
                    (const uint8_t *)readback, rowBytes, (uint8_t *)pixelBytes,
                    bytesPerRow, readRegion.size.width, readRegion.size.height,
                    mglPdTextureInfo(texture).pixel_format, format, type,
                    flipRenderTargetRows)) {
                fprintf(stderr,
                        "MGL ERROR: mtlGetTexImage unsupported BGRA8 conversion texture=%u format=0x%x type=0x%x\n",
                        tex->name, (unsigned)format, (unsigned)type);
                mglDispatchError(glm_ctx, __func__,
                                 (GLenum)mglRenderErrorInvalidOperation());
            }
        } else {
            mglMetalCopyRows((const uint8_t *)readback, rowBytes,
                             (uint8_t *)pixelBytes, bytesPerRow, rowBytes,
                             readRegion.size.height, 1);
        }
        free(readback);
    } else {
        if (!mglPdTextureGetBytes(texture, pixelBytes, bytesPerRow,
                                  bytesPerImage, readRegion, level, readSlice,
                                  1)) {
            fprintf(stderr,
                    "MGL ERROR: mtlGetTexImage texture read failed for texture %u\n",
                    tex->name);
            mglDispatchError(glm_ctx, __func__,
                             (GLenum)mglRenderErrorInvalidOperation());
        }
    }
}
