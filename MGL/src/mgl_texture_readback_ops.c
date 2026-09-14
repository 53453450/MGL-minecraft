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
#include "mgl_gpu_recovery.h"

#include <stdio.h>
#include <string.h>

/* The .m's file-local constants this TU needs (values copied verbatim). */
enum {
    MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED = 0u,
};

extern void mglPlatformShellSetContext(void *renderer, GLMContext glm_ctx);

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
