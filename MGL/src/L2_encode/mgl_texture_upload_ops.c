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

#include "mgl_metal_ref.h"       /* MGLMetalKindTexture, mglMetalCountCreate */
#include "mgl_pixel_format.h"       /* mglTextureBytesPerPixelForFormat */
#include "mgl_texture_mip_ops.h"     /* mglTextureLogMipDiagnostics */
#include "mgl_texture_compat.h"
#include "mgl_texture_upload_ops.h"
#include "mgl_texture_plan.h"

#include "mgl_capability.h"        /* MGLCapabilityHasBug */
#include "mgl_gpu_recovery.h"      /* mglRendererShouldSkipGPUOperations */
#include "mgl_pso_format_class.h"  /* mglRenderPixelFormatIsPackedDepthStencil */
#include "mgl_readback_policy.h"   /* mglRenderTextureRepackDepthPlanes */
#include "mgl_renderer_backend.h"
#include "mgl_ports_batch_flush.h"
#include "mgl_renderer_state_areas.h"
#include "mgl_texture_create_ops.h" /* mglTextureReplaceRegionValue */
#include "mgl_render_pass_manager_ops.h" /* ensure-writable-command-buffer */
#include "pixel_utils.h"               /* MGLPixelFormatR8Unorm */
#include "mgl_thread_affinity.h"        /* MGL_ASSERT_GL_THREAD */
#include "mgl_trace_log.h"

#define MGL_TEX_UPLOAD_DIAG(FMT, ...)                                          \
    do {                                                                       \
        if (mglTraceLogIsEnabled()) {                                          \
            fprintf(stderr, FMT, ##__VA_ARGS__);                               \
        }                                                                      \
    } while (0)

#include "mgl_render_api_command.h"

#include "mgl_render_api_texture.h"

#include "mgl_render_api_buffer.h"

#include "mgl_render_api_pixel.h"

#include "mgl_render_api_query.h"

#include <dispatch/dispatch.h>
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mgl_texture_backend.h"
#include "mgl_textures_internal.h"
#include "mgl_buffer_backend.h"

enum { MGL_UPLOAD_SCRATCH_SLOTS = 2 };

static uint8_t *s_upload_expand_scratch[MGL_UPLOAD_SCRATCH_SLOTS];
static size_t s_upload_expand_capacity[MGL_UPLOAD_SCRATCH_SLOTS];
static _Atomic uint8_t s_upload_expand_busy[MGL_UPLOAD_SCRATCH_SLOTS];

static uint8_t *s_upload_align_scratch[MGL_UPLOAD_SCRATCH_SLOTS];
static size_t s_upload_align_capacity[MGL_UPLOAD_SCRATCH_SLOTS];
static _Atomic uint8_t s_upload_align_busy[MGL_UPLOAD_SCRATCH_SLOTS];

static int mglUploadScratchTrySlot(_Atomic uint8_t *busy_flags, uint32_t *slot_out)
{
    for (uint32_t i = 0; i < MGL_UPLOAD_SCRATCH_SLOTS; i++) {
        uint8_t expected = 0;
        if (atomic_compare_exchange_strong(&busy_flags[i], &expected, 1)) {
            *slot_out = i;
            return 1;
        }
    }
    return 0;
}

static void mglUploadScratchReleaseSlot(_Atomic uint8_t *busy_flags, uint32_t slot)
{
    if (slot < MGL_UPLOAD_SCRATCH_SLOTS) {
        atomic_store_explicit(&busy_flags[slot], 0, memory_order_release);
    }
}

static uint8_t *mglUploadScratchGrow(uint8_t **storage, size_t *capacity,
                                     size_t need)
{
    if (*storage && *capacity >= need) {
        return *storage;
    }
    free(*storage);
    *storage = (uint8_t *)malloc(need);
    *capacity = *storage ? need : 0;
    return *storage;
}

uint8_t *mglUploadExpandScratchAcquire(size_t size)
{
    if (size == 0 || size > (512u * 1024u * 1024u)) {
        return NULL;
    }
    uint32_t slot = 0;
    if (mglUploadScratchTrySlot(s_upload_expand_busy, &slot)) {
        if (mglUploadScratchGrow(&s_upload_expand_scratch[slot],
                                 &s_upload_expand_capacity[slot], size)) {
            return s_upload_expand_scratch[slot];
        }
        mglUploadScratchReleaseSlot(s_upload_expand_busy, slot);
    }
    return (uint8_t *)malloc(size);
}

void mglUploadExpandScratchRelease(uint8_t *ptr)
{
    if (!ptr) {
        return;
    }
    for (uint32_t i = 0; i < MGL_UPLOAD_SCRATCH_SLOTS; i++) {
        if (s_upload_expand_scratch[i] && ptr >= s_upload_expand_scratch[i] &&
            ptr < s_upload_expand_scratch[i] + s_upload_expand_capacity[i]) {
            mglUploadScratchReleaseSlot(s_upload_expand_busy, i);
            return;
        }
    }
    free(ptr);
}

void *mglUploadAlignScratchAcquire(size_t alignment, size_t size)
{
    if (alignment == 0 || size == 0 || size > (512u * 1024u * 1024u)) {
        return NULL;
    }
    size_t total = size + alignment - 1u;
    uint32_t slot = 0;
    if (mglUploadScratchTrySlot(s_upload_align_busy, &slot)) {
        if (mglUploadScratchGrow(&s_upload_align_scratch[slot],
                                 &s_upload_align_capacity[slot], total)) {
            uintptr_t base = (uintptr_t)s_upload_align_scratch[slot];
            uintptr_t aligned =
                (base + alignment - 1u) & ~(uintptr_t)(alignment - 1u);
            return (void *)aligned;
        }
        mglUploadScratchReleaseSlot(s_upload_align_busy, slot);
    }
    void *heap = aligned_alloc(alignment, size);
    return heap;
}

void mglUploadAlignScratchRelease(void *ptr)
{
    if (!ptr) {
        return;
    }
    for (uint32_t i = 0; i < MGL_UPLOAD_SCRATCH_SLOTS; i++) {
        if (!s_upload_align_scratch[i]) {
            continue;
        }
        uintptr_t base = (uintptr_t)s_upload_align_scratch[i];
        uintptr_t end = base + s_upload_align_capacity[i];
        uintptr_t p = (uintptr_t)ptr;
        if (p >= base && p < end) {
            mglUploadScratchReleaseSlot(s_upload_align_busy, i);
            return;
        }
    }
    free(ptr);
}

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
extern int mglRendererShouldSkipGPUOperations(void *renderer);
extern const char *mglCommandBufferStatusName(uint32_t status);
extern bool mglRendererBindMTLTexture(void *renderer, Texture *tex);
extern Texture *findTexture(GLMContext ctx, GLuint texture);

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
        mglTextureBumpWriteVersion(tex);
        mglTextureViewFamilyWritten(tex);
    }
}


/* mglUpMax() is an Objective-C header macro; use the C maximum inline. */
#define mglUpMax(a, b) ((a) > (b) ? (a) : (b))


/* mtlPixelFormatForGLTex lives in the Objective-C private header. */
extern uint32_t mtlPixelFormatForGLTex(Texture *tex);

/* The renderer's capability snapshot through the state areas (the .m read its
 * own ivar; the C caller has the renderer handle). */
static const MGLCapability *mglUpCapability(void *renderer)
{
    static MGLCapability fallback;
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    return areas.core ? &areas.core->capability : &fallback;
}

/* mglUpMin() is an Objective-C header macro; C has no same-named inline. */
#define mglUpMin(a, b) ((a) < (b) ? (a) : (b))
#define mglUpMax(a, b) ((a) > (b) ? (a) : (b))

/* Twins of the .m's blit helpers (the trace path reads a sample back). */
static void mglUpEndBlitEncoder(void *encoder)
{
    if (!encoder) return;
    (void)mglRenderEndBlitEncoder(encoder);
}

static void mglUpCommitCommandBuffer(void *commandBuffer)
{
    if (!commandBuffer) return;
    if (mglRenderCommitCommandBuffer(commandBuffer) != 0) {
        fprintf(stderr,
                "MGL ERROR: Metal-cpp texture command-buffer commit failed\n");
    }
}

static void mglUpWaitCommandBuffer(void *commandBuffer)
{
    if (!commandBuffer) return;
    if (mglRenderWaitCommandBuffer(commandBuffer) != 0) {
        fprintf(stderr,
                "MGL ERROR: Metal-cpp texture command-buffer wait failed\n");
    }
}

static void mglUpCopyTextureToBuffer(
    void *encoder, void *source, uint64_t sourceSlice, uint64_t sourceLevel,
    MGLOriginValue sourceOrigin, MGLSizeValue sourceSize, void *destination,
    uint64_t destinationOffset, uint64_t bytesPerRow, uint64_t bytesPerImage)
{
    (void)mglRenderBlitCopyTextureToBuffer(
        encoder, source, sourceSlice, sourceLevel, sourceOrigin.x,
        sourceOrigin.y, sourceOrigin.z, sourceSize.width, sourceSize.height,
        sourceSize.depth, destination, destinationOffset, bytesPerRow,
        bytesPerImage);
}

/* Twins of the .m's mglTextureCreateBuffer / mglTextureCreateCommandBuffer /
 * mglTextureCreateBlitEncoder (the +1 handles the .m returned as id). */
static void *mglUpCreateBuffer(uint64_t length, uint64_t options)
{
    void *buffer = NULL;
    if (mglRenderCreateBuffer((size_t)length, options, NULL, &buffer) == 0 &&
        buffer) {
        return buffer;
    }
    return NULL;
}

static void *mglUpCreateCommandBuffer(void *queue)
{
    if (!queue) return NULL;
    void *commandBuffer = NULL;
    if (mglRenderCreateCommandBuffer(queue, &commandBuffer) == 0 &&
        commandBuffer) {
        return commandBuffer;
    }
    return NULL;
}

static void *mglUpCreateBlitEncoder(void *commandBuffer)
{
    if (!commandBuffer) return NULL;
    void *encoder = NULL;
    if (mglRenderCreateBlitEncoder(commandBuffer, &encoder) == 0 && encoder) {
        return encoder;
    }
    return NULL;
}

/* Twin of the .m's mglTextureBufferContents. */
static void *mglUpTextureBufferContents(void *buffer)
{
    void *contents = NULL;
    uint64_t length = 0u;
    return buffer && mglRenderGetBufferContents(buffer, &contents, &length) == 0
               ? contents
               : NULL;
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
    MGLOriginValue destinationOrigin, int flipY, const char *reason,
    MGLTextureStagingOwner **staging_owner_io)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
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
                destinationOrigin.z, flipY) != 0) {
            fprintf(stderr, "MGL ERROR: C++ ordered upload encode failed (%s)\n",
                    reason ? reason : "texture_upload");
            mglRendererRecordGPUError(renderer);
            return 0;
        }
        if (areas.batching) {
            mglRendererSetCommandBufferHasWork(renderer, 1);
        }
        /* The blit rides in the frame's command buffer: the pool's generation
         * guard keeps these bytes out of circulation until that buffer
         * completes, so returning the owner now is safe. */
        if (staging_owner_io) {
            mglRenderDestroyTextureStagingOwner(staging_owner_io);
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
            destinationOrigin.y, destinationOrigin.z, flipY) != 0) {
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

    /* This buffer is committed without being waited on, and the blit keeps
     * reading the staging bytes until it completes — hand the owner to that
     * completion instead of returning it now. */
    if (staging_owner_io && *staging_owner_io) {
        (void)mglRenderDestroyTextureStagingOwnerAfterCommandBuffer(
            staging_owner_io, uploadCB);
    }

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
        MGL_TEX_UPLOAD_DIAG("MGL WARNING: dedicated upload wait timed out (%s), continuing asynchronously\n",
                reason ? reason : "texture_upload");
        return 1;
    }

    return !uploadError;
}

/* Metal forbids CPU access to packed depth/stencil textures and whole-texel
 * buffer copies into them, so the canonical CPU layout (GL_DEPTH32F_STENCIL8:
 * 5-byte float + u8; others: 24_8 uint) is split into a depth plane and a
 * stencil plane and each is blitted with its aspect. */
static int mglUpUploadPackedDepthStencil(void *renderer, void *texture,
                                         GLenum internalformat,
                                         const void *bytes,
                                         uint64_t bytesPerRow, uint64_t width,
                                         uint64_t height, uint64_t level,
                                         uint64_t slice, uint64_t x,
                                         uint64_t y, int flipY)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    const uint64_t texelBytes =
        internalformat == GL_DEPTH32F_STENCIL8 ? 5u : 4u;
    if (!bytes || width == 0 || height == 0 ||
        bytesPerRow < width * texelBytes) {
        return 0;
    }
    /* Depth24Unorm_Stencil8 takes the depth plane as 24-bit unorm in a
     * 32-bit word; Depth32Float_Stencil8 takes float32. */
    const int unormDepth =
        !mglRenderPixelFormatIsDepth32FloatStencil8(
            (uint32_t)mglPdTextureInfo(texture).pixel_format);
    const uint64_t depthPlaneBytes = width * height * 4u;
    uint8_t *planes = malloc((size_t)(depthPlaneBytes + width * height));
    if (!planes) return 0;
    for (uint64_t row = 0; row < height; ++row) {
        const uint8_t *src = (const uint8_t *)bytes + row * bytesPerRow;
        for (uint64_t col = 0; col < width; ++col, src += texelBytes) {
            uint32_t depthWord;
            uint8_t stencil;
            if (texelBytes == 4u) {
                uint32_t packed;
                memcpy(&packed, src, 4u);
                stencil = (uint8_t)(packed & 0xffu);
                if (unormDepth) {
                    depthWord = packed >> 8;
                } else {
                    float depth = (float)(packed >> 8) / 16777215.0f;
                    memcpy(&depthWord, &depth, 4u);
                }
            } else {
                float depth;
                memcpy(&depth, src, 4u);
                stencil = src[4];
                if (unormDepth) {
                    if (!(depth > 0.0f)) depth = 0.0f;
                    if (depth > 1.0f) depth = 1.0f;
                    depthWord = (uint32_t)llround((double)depth * 16777215.0);
                } else {
                    memcpy(&depthWord, &depth, 4u);
                }
            }
            const uint64_t texel = row * width + col;
            memcpy(planes + texel * 4u, &depthWord, 4u);
            planes[depthPlaneBytes + texel] = stencil;
        }
    }

    MGLTextureStagingOwner *stagingOwner = NULL;
    void *stagingBuffer = NULL;
    int uploaded = 0;
    if (mglRenderCreateTextureStagingOwner(
            planes, depthPlaneBytes + width * height,
            MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED, &stagingOwner,
            &stagingBuffer) == 0 &&
        stagingBuffer) {
        mglRendererEndRenderEncodingLocked(renderer);
        uploaded =
            mglRenderPassEnsureWritableCommandBufferLocked(
                renderer, "packed_depth_stencil_upload") &&
            mglRenderEncodePackedDepthStencilUploadForCommandBufferOwner(
                areas.command->currentCommandBufferOwner, stagingBuffer, width,
                height, texture, slice, level, x, y, flipY) == 0;
        if (uploaded && areas.batching) {
            mglRendererSetCommandBufferHasWork(renderer, 1);
        }
    }
    mglRenderDestroyTextureStagingOwner(&stagingOwner);
    free(planes);
    if (!uploaded) {
        MGL_TEX_UPLOAD_DIAG("MGL WARNING: packed depth/stencil upload failed level=%lu slice=%lu\n",
                (unsigned long)level, (unsigned long)slice);
    }
    return uploaded;
}

/* -uploadTextureSliceViaBlit:... */
int mglTextureUploadSliceViaBlit(void *renderer, void *texture,
                                 unsigned int texName, GLenum texTarget,
                                 const void *bytes, uint64_t bytesPerRow,
                                 uint64_t bytesPerImage, uint64_t width,
                                 uint64_t height, uint64_t depth, uint64_t level,
                                 uint64_t slice, int flipY)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);

    if (!texture || !bytes || bytesPerRow == 0 || bytesPerImage == 0 ||
        width == 0) {
        return 0;
    }

    if (mglRendererShouldSkipGPUOperations(renderer)) {
        MGL_TEX_UPLOAD_DIAG("MGL AGX: Skipping texture upload during recovery\n");
        return 0;
    }

    const uint32_t textureType = mglPdTextureInfo(texture).texture_type;
    MGLRenderTextureUploadPlan uploadPlan = {0};
    /* Note: this used to pass the AGX "copyFromBuffer slice OOB" bug marker,
     * which diverted every GL_TEXTURE_3D upload to the replaceRegion route (and
     * rejected Private-storage 3D uploads outright).  An independent Metal
     * reproduction on 2026-09-20 showed copyFromBuffer:->texture: handles both
     * Shared and Private 3D destinations correctly, so the marker is gone and
     * 3D uploads use the normal blit route.  See
     * docs/AGX_COPY3D_DRIVER_BUG_RECHECK_2026-09-20.md. */
    if (mglRenderBuildTextureUploadPlan(
            (uint32_t)texTarget, textureType,
            mglPdTextureInfo(texture).storage_mode,
            (uint32_t)mglPdTextureInfo(texture).pixel_format,
            0,
            width, height, depth, bytesPerRow, bytesPerImage, level, slice,
            &uploadPlan) != 0) {
        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Rejecting invalid texture upload plan (tex=%u target=0x%x level=%lu slice=%lu)\n",
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

    if (mglRenderPixelFormatIsPackedDepthStencil(
            (uint32_t)mglPdTextureInfo(texture).pixel_format)) {
        const Texture *tex = findTexture(areas.ctx, texName);
        const GLenum internalformat =
            tex ? tex->internalformat
                : (bytesPerRow >= width * 5u ? GL_DEPTH32F_STENCIL8
                                             : GL_DEPTH24_STENCIL8);
        return mglUpUploadPackedDepthStencil(
            renderer, texture, internalformat, bytes, bytesPerRow, width,
            uploadPlan.normalized_height, uploadPlan.destination_level,
            uploadPlan.destination_slice, 0u, 0u, flipY);
    }

    /* Shared 2D array uploads via replaceRegion: the blit path can leave array
     * slices unpopulated on some AGX drivers when uploading CPU data during
     * initial texture creation.  Shared storage is safe here because bind
     * happens before the first draw that samples this texture.
     * PERF-020: once a mip level was written or sampled, prefer the blit path
     * below so in-flight draws cannot race replaceRegion CPU writes. */
    const Texture *tex2dArray = findTexture(areas.ctx, texName);
    const uint32_t dstLevel = uploadPlan.destination_level;
    const GLboolean arrayLevelIdle =
        !tex2dArray || dstLevel >= tex2dArray->num_levels ||
        !tex2dArray->faces[0].levels[dstLevel].ever_written;
    if (textureType == MGLTextureType2DArray && !flipY && arrayLevelIdle &&
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
        MGL_TEX_UPLOAD_DIAG("MGL WARNING: 2D array replaceRegion upload failed (tex=%u level=%lu slice=%lu)\n",
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
        MGL_TEX_UPLOAD_DIAG("MGL WARNING: 1D texture replaceRegion upload failed, falling back to blit (tex=%u level=%lu slice=%lu)\n",
                (unsigned)texName, (unsigned long)level, (unsigned long)slice);
    }

    /* 3D texture upload via replaceRegion branch:
     * - The C++ route only selects this for GL_TEXTURE_1D/1D_ARRAY storage
     *   that maps to a 3D Metal texture, and for that case replaceRegion is
     *   the routable option.  The former AGX "copyFromBuffer slice OOB"
     *   diversion of *real* 3D textures is gone: an independent Metal
     *   reproduction (2026-09-20) showed copyFromBuffer:->texture: is correct
     *   for both Shared and Private 3D destinations.
     * - Metal requires bytesPerImage for 3D replaceRegion uploads, so padded
     *   depth planes are repacked and uploaded with the tight image stride.
     * - Only shared storage supports replaceRegion. */
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
        MGL_TEX_UPLOAD_DIAG("MGL WARNING: 3D texture replaceRegion upload failed (tex=%u level=%lu)\n",
                (unsigned)texName, (unsigned long)level);
        return 0;
    }

    /* Defensive: the C++ route no longer rejects Private 3D uploads now that
     * the AGX copyFromBuffer marker is gone (copyFromBuffer is correct for
     * Private 3D destinations).  Kept so an unexpected REJECT cannot fall
     * through into the blit path silently. */
    if (uploadRoute == MGL_RENDER_TEXTURE_UPLOAD_ROUTE_REJECT) {
        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Rejecting texture upload (route=REJECT tex=%u level=%lu)\n",
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
        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Failed to allocate upload buffer for texture blit\n");
        return 0;
    }

    const int uploaded = mglTextureCopyUploadWithDedicatedCommandBuffer(
        renderer, uploadBuffer, 0, bytesPerRow,
        uploadPlan.normalized_bytes_per_image, 0, 1,
        mglTextureSize(width, uploadPlan.normalized_height,
                       uploadPlan.copy_depth),
        texture, uploadPlan.destination_slice, uploadPlan.destination_level,
        mglTextureOrigin(0, 0, 0), flipY, "texture_upload_blit",
        &stagingOwner);
    /* Consumed by the upload path above (deferred to the upload command
     * buffer's completion when it travelled in its own buffer); still a no-op
     * when an early return left it with us. */
    mglRenderDestroyTextureStagingOwner(&stagingOwner);
    if (!uploaded) {
        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Dedicated texture upload failed (level=%lu slice=%lu)\n",
                (unsigned long)level, (unsigned long)slice);
        return 0;
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

                                MGL_TEX_UPLOAD_DIAG("MGL INFO: Attempting buffer-based texture fill\n");


                                // Create a temporary MTLBuffer with the texture data

                                MGLTextureStagingOwner *stagingOwner = NULL;
                                void *tempBuffer = NULL;
                                const int staged = mglRenderCreateTextureStagingOwner(
                                        properData, fillSize,
                                        MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED,
                                        &stagingOwner, &tempBuffer) == 0 &&
                                    stagingOwner != NULL && tempBuffer != NULL;


                                if (staged) {

                                    MGL_TEX_UPLOAD_DIAG("MGL INFO: Created temporary MTLBuffer for texture data\n");


                                    if (mglRendererShouldSkipGPUOperations(renderer)) {

                                        MGL_TEX_UPLOAD_DIAG("MGL AGX: Skipping texture fill during recovery - texture will be empty\n");

                                    } else {

                                        int uploaded = mglTextureCopyUploadWithDedicatedCommandBuffer(
            renderer, tempBuffer, 0, properBytesPerRow, fillSize, 0, 1, mglTextureSize(properRegion.size.width, properRegion.size.height, 1), texture, 0, 0, mglTextureOrigin(0, 0, 0), 0, "texture_fill_initialization", NULL);

                                        if (uploaded) {

                                            MGL_TEX_UPLOAD_DIAG("MGL SUCCESS: Texture data copied using dedicated upload command buffer\n");

                                            mglUpMarkTextureLevelMetalFilled(tex, 0, fillSize);

                                        } else {

                                            MGL_TEX_UPLOAD_DIAG("MGL WARNING: Dedicated texture fill upload failed - texture may remain uninitialized\n");

                                        }

                                    }


                                    // Clean up the temporary buffer: the
                                    // encoded upload retains its source until
                                    // command-buffer completion, so the
                                    // staging owner can be released now (this
                                    // used to leak one MTLBuffer per fill).

                                    mglRenderDestroyTextureStagingOwner(&stagingOwner);
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


                                        MGL_TEX_UPLOAD_DIAG("MGL SUCCESS: Simple direct color fill completed\n");

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
            renderer, texture, tex->name, tex->target, op->data, (uint64_t)op->bytes_per_row, (uint64_t)op->bytes_per_image, (uint64_t)op->width, (uint64_t)op->height, (uint64_t)op->copy_depth, op->level, 0, 0);
        if (op->owns_data) {
            mglUploadExpandScratchRelease((uint8_t *)op->data);
        }
        if (uploaded) {
            uploadedAny = true;
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

    if (mglTextureMtl(tex) == NULL) {
        mglRendererBindMTLTexture(renderer, tex);
        if (mglTextureMtl(tex) == NULL) {
            return false;
        }
    }

    void *texture = mglTextureMtl(tex);
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

    const int flipY = 0;
    return mglTextureCopyUploadWithDedicatedCommandBuffer(
            renderer, buffer, sourceOffset, sourceBytesPerRow, copyBytesPerImage, sourceLayerStride, layerCount, mglTextureSize(
                                                       (uint64_t)uploadPlan.copy_width,
                                                       copyHeight, copyDepth), texture, destinationSlice, level, destinationOrigin, flipY, reason ? reason : "texture_sub_upload", NULL);
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

                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: Re-upload array invalid row layout tex=%d face=%d level=%d bpr=%lu rows=%lu\n",

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

                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: Re-upload array backing too small tex=%d face=%d level=%d backing=%lu layerBytes=%lu layers=%u\n",

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



                    uint64_t alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);

                    uint64_t alignedBytesPerRow = effectiveBytesPerRow;

                    if (alignedBytesPerRow % alignment != 0) {

                        alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;

                    }


                    uintptr_t addr = (uintptr_t)layerSrcData;

                    if (addr % alignment != 0 || alignedBytesPerRow != effectiveBytesPerRow) {

                        uint64_t alignedUploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);

                        if (alignedUploadRows == 0 || alignedBytesPerRow > (UINT64_MAX / alignedUploadRows)) {

                            if (mglTraceLogIsEnabled()) {
                                MGL_TEX_UPLOAD_DIAG("MGL WARNING: Re-upload array rejecting aligned row layout bpr=%lu rows=%lu tex=%d face=%d level=%d layer=%u\n",
                                      (unsigned long)alignedBytesPerRow,
                                      (unsigned long)alignedUploadRows,
                                      tex->name, face, level, layer);
                            }

                            mglUploadExpandScratchRelease(expandedUploadData);

                            continue;

                        }

                        uint64_t alignedSize = alignedBytesPerRow * alignedUploadRows;

                        if (alignedSize > 0 && alignedSize <= (512 * 1024 * 1024)) {

                            void *alignedData = mglUploadAlignScratchAcquire(alignment, alignedSize);

                            if (alignedData) {

                                memset(alignedData, 0, alignedSize);

                                for (uint64_t row = 0; row < alignedUploadRows; row++) {

                                    uint64_t copySize = mglUpMin(effectiveBytesPerRow, alignedBytesPerRow);

                                    memcpy((uint8_t *)alignedData + row * alignedBytesPerRow,

                                           (const uint8_t *)layerSrcData + row * effectiveBytesPerRow, copySize);

                                }

                                mglTextureUploadSliceViaBlit(
            renderer, texture, tex->name, tex->target, alignedData, alignedBytesPerRow, alignedSize, lvlWidth, lvlHeight, 1, level, layer, 0);

                                mglUploadAlignScratchRelease(alignedData);

                            }

                        }

                    } else {

                        mglTextureUploadSliceViaBlit(
            renderer, texture, tex->name, tex->target, layerSrcData, effectiveBytesPerRow, effectiveBytesPerImage, lvlWidth, lvlHeight, 1, level, layer, 0);

                    }

                    free(swizzledUploadData);
                    mglUploadExpandScratchRelease(expandedUploadData);

                }
}

/* -fillSmallRGBA8TextureWithGradient:tex: */
void mglTextureFillSmallGradient(void *renderer, void *texture, Texture *tex)
{
                            if (mglRenderIsSmallRGBA8(
                                    (uint32_t)mglPdTextureInfo(texture).width,
                                    (uint32_t)mglPdTextureInfo(texture).height,
                                    (uint32_t)tex->internalformat)) {

                                MGL_TEX_UPLOAD_DIAG("MGL INFO: Attempting simple direct color fill for small RGBA8 texture\n");/* The .m ran this fill inside @try/@catch; the C port keeps the same catch
     * logic through the shell's guarded call (rule 58 (b)). */
    {
        char fillFailure[256] = {0};
        MglUpFillCtx fillCtx = { texture, tex };
        if (!mglPlatformShellGuardedCallCtxReason(renderer, "small RGBA8 fill",
                                                  mglUpFillBody, &fillCtx,
                                                  fillFailure,
                                                  sizeof(fillFailure))) {
            MGL_TEX_UPLOAD_DIAG("MGL WARNING: Simple direct fill also failed: %s\n",
                    fillFailure[0] ? fillFailure : "(null)");
        }
    }

                            } else {

                                MGL_TEX_UPLOAD_DIAG("MGL INFO: Skipping complex texture - would use deferred initialization\n");

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
    MGL_TEX_UPLOAD_DIAG("MGL INFO: Re-uploading existing CPU texture data (tex=%d, dims=%lux%lu)\n",

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

            /* Block-compressed levels store pitch as bytes per row-of-blocks.
             * Padding that pitch to the linear-texture alignment (64) and
             * iterating lvlHeight pixel rows over-reads the CPU shadow and
             * hands Metal an illegal bytesPerRow (CTS 4x4 BPTC is pitch 16).
             * Keep the tight block stride and only realign the pointer. */
            const uint64_t blockH =
                mglRenderMetalCompressedBlockHeight(pixelFormat);
            const int is_block_compressed = (blockH > 1u);
            uint64_t alignment =
                mglRendererOptimalAlignmentForPixelFormat(pixelFormat);
            uint64_t alignedBytesPerRow = bytesPerRow;
            if (!is_block_compressed && alignedBytesPerRow % alignment != 0) {
                alignedBytesPerRow =
                    ((alignedBytesPerRow + alignment - 1) / alignment) *
                    alignment;
            }

            uintptr_t addr = (uintptr_t)srcData;
            if (addr % alignment != 0 || alignedBytesPerRow != bytesPerRow) {
                uint64_t rowCount =
                    is_block_compressed
                        ? mglMetalUploadRowsForPixelFormat(
                              pixelFormat, mglUpMax((uint64_t)lvlHeight, 1UL))
                        : mglUpMax((uint64_t)lvlHeight, 1UL);
                if (rowCount == 0u) {
                    rowCount = 1u;
                }
                uint64_t alignedSliceBPI = alignedBytesPerRow * rowCount;
                uint64_t alignedSize = alignedSliceBPI * uploadDepth;
                uint64_t srcRowStride =
                    is_block_compressed ? bytesPerRow : bytesPerRow;
                uint64_t srcSliceStride =
                    is_block_compressed
                        ? (bytesPerRow * rowCount)
                        : singleSliceBPI;

                if (alignedSize > 0 && alignedSize <= (512 * 1024 * 1024)) {
                    void *alignedData = mglUploadAlignScratchAcquire(alignment, alignedSize);
                    if (alignedData) {
                        memset(alignedData, 0, alignedSize);
                        for (uint64_t z = 0; z < uploadDepth; z++) {
                            for (uint64_t row = 0; row < rowCount; row++) {
                                uint64_t copySize =
                                    mglUpMin(bytesPerRow, alignedBytesPerRow);
                                memcpy((uint8_t *)alignedData +
                                           z * alignedSliceBPI +
                                           row * alignedBytesPerRow,
                                       (const uint8_t *)srcData +
                                           z * srcSliceStride +
                                           row * srcRowStride,
                                       copySize);
                            }
                        }
                        mglTextureUploadSliceViaBlit(
                            renderer, texture, tex->name, tex->target,
                            alignedData, alignedBytesPerRow, alignedSliceBPI,
                            lvlWidth, lvlHeight, uploadDepth, level, face, 0);
                        mglUploadAlignScratchRelease(alignedData);
                    }
                }
            } else {
                mglTextureUploadSliceViaBlit(
                    renderer, texture, tex->name, tex->target, srcData,
                    bytesPerRow, bytesPerImage, lvlWidth, lvlHeight,
                    uploadDepth, level, face, 0);
            }

            mglUploadExpandScratchRelease(expandedUploadData);

            } /* end else (non-array) */

        }

    }
}

/* -fillTextureWithSafeInitialContents:tex:pixelFormat: */
void mglTextureFillSafeInitialContents(void *renderer, void *texture,
                                       Texture *tex, uint32_t pixelFormat)
{


    if (mglPdTextureInfo(texture).width == 0 || mglPdTextureInfo(texture).height == 0 || mglPdTextureInfo(texture).width > 16384 || mglPdTextureInfo(texture).height > 16384) {

        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Skipping texture fill due to invalid dimensions: %lux%lu\n", (unsigned long)mglPdTextureInfo(texture).width, (unsigned long)mglPdTextureInfo(texture).height);

    } else {

        // Determine pixel format size to create appropriate black data

        uint64_t bytesPerPixel = (uint64_t)mglRenderMetalPixelFormatBytesPerPixel(
            mglPdTextureInfo(texture).pixel_format);

        if (bytesPerPixel == 0) {
            /* Block-compressed or unrecognised Metal format: it has no linear
             * per-pixel row stride, so this fill cannot be sized.  Substituting
             * a default of 4 used to hand Metal a bytesPerRow that is invalid
             * for block formats (and half the real row for
             * Depth32Float_Stencil8), so skip the fill instead of guessing. */
            MGL_TEX_UPLOAD_DIAG("MGL WARNING: Skipping texture fill: no bytes-per-pixel for Metal pixel format %lu\n",
                    (unsigned long)mglPdTextureInfo(texture).pixel_format);
            return;
        }

        // Calculate dynamic alignment for Metal textures based on pixel format

        uint64_t bytesPerRow = mglPdTextureInfo(texture).width * bytesPerPixel;

        uint64_t alignment = mglRendererOptimalAlignmentForPixelFormat(mglPdTextureInfo(texture).pixel_format);

        if (bytesPerRow % alignment != 0) {

            bytesPerRow = ((bytesPerRow + alignment - 1) / alignment) * alignment;

        }


        uint64_t dataSize = bytesPerRow * mglPdTextureInfo(texture).height;


        // Validate that dataSize is reasonable (not too large)

        if (dataSize > 64 * 1024 * 1024) { // 64MB limit per texture level

            MGL_TEX_UPLOAD_DIAG("MGL WARNING: Skipping texture fill due to excessive size: %lu bytes\n", (unsigned long)dataSize);

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


                MGL_TEX_UPLOAD_DIAG("MGL INFO: All validations passed for texture fill (size=%lu, bytesPerRow=%lu)\n", (unsigned long)dataSize, (unsigned long)bytesPerRow);


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


                MGL_TEX_UPLOAD_DIAG("MGL INFO: FIXING: Implementing proper texture filling for Apple Metal compatibility\n");


                // PROPER FIX: Use Apple Metal-compatible texture filling approach

                // The issue was using incorrect bytesPerRow and region parameters

                MGL_TEX_UPLOAD_DIAG("MGL INFO: Implementing Metal-compliant texture fill operations\n");


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
                    MGLRendererStateView safeAreas;
                    mglRendererFillStateView(renderer, &safeAreas);
                    /* The outer @try's prologue logs (they used to sit inside
                     * that block; the C port runs the attempt through the
                     * shell's guard instead, so they are emitted first). */
                    MGL_TEX_UPLOAD_DIAG("MGL INFO: Performing Metal-compliant texture fill:\n");
                    fprintf(stderr, "  - Region: %dx%d\n",
                            (int)properRegion.size.width,
                            (int)properRegion.size.height);
                    fprintf(stderr, "  - bytesPerRow: %lu\n",
                            (unsigned long)properBytesPerRow);
                    fprintf(stderr, "  - dataSize: %lu\n",
                            (unsigned long)fillSize);
                    // ALTERNATIVE APPROACH: Safe texture filling without replaceRegion
                    MGL_TEX_UPLOAD_DIAG("MGL INFO: Using alternative texture filling methods (AGX-safe)\n");

                    MglUpSafeFillCtx safeFillCtx = {
                        texture, tex,
                        mglRendererBackendGetDevice(safeAreas.backend),
                        properData, properRegion, properBytesPerRow, fillSize,
                        dataSize };
                    if (!mglPlatformShellGuardedCallCtxReason(
                            renderer, "safe texture fill", mglUpSafeFillBody,
                            &safeFillCtx, safeFillFailure,
                            sizeof(safeFillFailure))) {
                        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Buffer-based texture fill failed - trying alternative\n");

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
                            MGL_TEX_UPLOAD_DIAG("MGL INFO: This indicates our parameters are still incompatible with AGX driver\n");
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


/* === The dirty-CPU-data upload tree (log 196) =============================
 * -uploadDirtyCPUTextureData:…, -uploadDirtyCPUTextureData3DLevel:… and
 * -uploadDirtyCPUTextureDataNon3DLevel:… moved from MGLRenderer+Texture.m.
 * Signatures come from the port that wrapped the dispatcher and from the two
 * callees' own selector lists (rule 63 (c)).
 */

/* ctx of the per-slice blit each level loader runs inside an @try. */
typedef struct MglUpSliceBlitCtx_t {
    void *renderer;
    void *texture;
    GLuint tex_name;
    GLenum tex_target;
    const void *data;
    uint64_t bytes_per_row;
    uint64_t bytes_per_image;
    uint64_t width;
    uint64_t height;
    uint64_t depth;
    uint64_t level;
    uint64_t slice;
    int flip_y;
    int *uploaded_out;
} MglUpSliceBlitCtx;

static int mglUpSliceBlitBody(void *renderer, void *rawCtx)
{
    MglUpSliceBlitCtx *ctx = (MglUpSliceBlitCtx *)rawCtx;
    *ctx->uploaded_out = mglTextureUploadSliceViaBlit(
        renderer, ctx->texture, ctx->tex_name, ctx->tex_target, ctx->data,
        ctx->bytes_per_row, ctx->bytes_per_image, ctx->width, ctx->height,
        ctx->depth, ctx->level, ctx->slice, ctx->flip_y);
    return 1;
}

/* -uploadDirtyCPUTextureData3DLevel:… */
int mglTextureUploadDirty3DLevel(void *renderer, Texture *tex, void *texture,
                                 uint32_t pixelFormat, int face, int level,
                                 uint64_t width, uint64_t height, uint64_t depth,
                                 int *outSkipped)
{
    uint64_t bytesPerRow;
    uint64_t bytesPerImage;

                bytesPerRow = tex->faces[face].levels[level].pitch;
                if (bytesPerRow == 0) {
                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: Invalid 3D bytesPerRow (0), skipping upload (tex=%d face=%d level=%d)\n", tex->name, face, level);
                    if (outSkipped) *outSkipped = 1;
                    return 1;
                }

                uint64_t uploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, mglUpMax((uint64_t)height, 1UL));
                if (uploadRows == 0 || bytesPerRow > (UINT64_MAX / uploadRows)) {
                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: Invalid 3D bytesPerImage overflow (tex=%d face=%d level=%d rows=%lu bpr=%lu)\n",
                          tex->name,
                          face,
                          level,
                          (unsigned long)uploadRows,
                          (unsigned long)bytesPerRow);
                    if (outSkipped) *outSkipped = 1;
                    return 1;
                }
                bytesPerImage = bytesPerRow * uploadRows;

                if (tex->faces[face].levels[level].data && bytesPerRow > 0 && bytesPerImage > 0) {
                    void *srcData = (void *)tex->faces[face].levels[level].data;
                    uintptr_t addr = (uintptr_t)srcData;

                    uint8_t *swizzled3DUploadData = NULL;
                    if (level == 0 && face == 0 &&
                        mglTextureUploadNeedsSwizzleBake(tex)) {
                        uint64_t texDepth = mglUpMax((uint64_t)depth, 1UL);
                        uint64_t texHeight = mglUpMax((uint64_t)height, 1UL);
                        uint64_t swzBPR = 0;
                        uint64_t swzBPI = 0;
                        uint8_t *firstSlice =
                            mglCreateSwizzledUpload(
                                tex, (const uint8_t *)srcData, width, texHeight,
                                bytesPerRow, &swzBPR, &swzBPI);
                        if (firstSlice) {
                            uint64_t totalSize = swzBPI * texDepth;
                            if (totalSize > 0 &&
                                totalSize <= (512 * 1024 * 1024)) {
                                swizzled3DUploadData =
                                    (uint8_t *)malloc(totalSize);
                                if (swizzled3DUploadData) {
                                    memcpy(swizzled3DUploadData, firstSlice,
                                           swzBPI);
                                    for (uint64_t z = 1; z < texDepth; z++) {
                                        const uint8_t *sliceSrc =
                                            (const uint8_t *)srcData +
                                            z * bytesPerImage;
                                        uint8_t *sliceDst =
                                            swizzled3DUploadData + z * swzBPI;
                                        uint8_t *sliceSwz =
                                            mglCreateSwizzledUpload(
                                                tex, sliceSrc, width, texHeight,
                                                bytesPerRow, &swzBPR, &swzBPI);
                                        if (sliceSwz) {
                                            memcpy(sliceDst, sliceSwz, swzBPI);
                                            free(sliceSwz);
                                        } else {
                                            memset(sliceDst, 0, swzBPI);
                                        }
                                    }
                                    srcData = swizzled3DUploadData;
                                    bytesPerRow = swzBPR;
                                    bytesPerImage = swzBPI;
                                    addr = (uintptr_t)srcData;
                                }
                            }
                            free(firstSlice);
                        }
                    }

                    uint8_t *expanded3DUploadData = NULL;
                    if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                    mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
                        uint64_t expandedBytesPerRow = 0;
                        uint64_t expandedBytesPerImagePerSlice = 0;
                        uint64_t texDepth = mglUpMax((uint64_t)depth, 1UL);
                        uint64_t texHeight = mglUpMax((uint64_t)height, 1UL);

                        uint8_t *firstSlice = mglCreateRGBA8ExpandedUpload(tex,
                                                                           (const uint8_t *)srcData,
                                                                           width,
                                                                           texHeight,
                                                                           bytesPerRow,
                                                                           &expandedBytesPerRow,
                                                                           &expandedBytesPerImagePerSlice);
                        if (firstSlice) {
                            uint64_t totalExpandedSize = expandedBytesPerImagePerSlice * texDepth;
                            if (totalExpandedSize > 0 && totalExpandedSize <= (512 * 1024 * 1024)) {
                                expanded3DUploadData = (uint8_t *)malloc(totalExpandedSize);
                                if (expanded3DUploadData) {
                                    memcpy(expanded3DUploadData, firstSlice, expandedBytesPerImagePerSlice);
                                    for (uint64_t z = 1; z < texDepth; z++) {
                                        const uint8_t *sliceSrc = (const uint8_t *)srcData + z * bytesPerImage;
                                        uint8_t *sliceDst = expanded3DUploadData + z * expandedBytesPerImagePerSlice;
                                        uint64_t dummyRow = 0, dummyImage = 0;
                                        uint8_t *sliceExpanded = mglCreateRGBA8ExpandedUpload(tex,
                                                                                             sliceSrc,
                                                                                             width,
                                                                                             texHeight,
                                                                                             bytesPerRow,
                                                                                             &dummyRow,
                                                                                             &dummyImage);
                                        if (sliceExpanded) {
                                            memcpy(sliceDst, sliceExpanded, expandedBytesPerImagePerSlice);
                                            free(sliceExpanded);
                                        } else {
                                            memset(sliceDst, 0, expandedBytesPerImagePerSlice);
                                        }
                                    }
                                    srcData = expanded3DUploadData;
                                    bytesPerRow = expandedBytesPerRow;
                                    bytesPerImage = expandedBytesPerImagePerSlice;
                                    addr = (uintptr_t)srcData;
                                }
                            }
                            free(firstSlice);
                        }
                    } else if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                           mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
                        uint64_t expandedBytesPerRow = 0;
                        uint64_t expandedBytesPerImagePerSlice = 0;
                        uint64_t texDepth = mglUpMax((uint64_t)depth, 1UL);
                        uint64_t texHeight = mglUpMax((uint64_t)height, 1UL);

                        uint8_t *firstSlice = mglCreateChannelExpandedUpload(tex,
                                                                              pixelFormat,
                                                                              (const uint8_t *)srcData,
                                                                              width,
                                                                              texHeight,
                                                                              bytesPerRow,
                                                                              &expandedBytesPerRow,
                                                                              &expandedBytesPerImagePerSlice);
                        if (firstSlice) {
                            uint64_t totalExpandedSize = expandedBytesPerImagePerSlice * texDepth;
                            if (totalExpandedSize > 0 && totalExpandedSize <= (512 * 1024 * 1024)) {
                                expanded3DUploadData = (uint8_t *)malloc(totalExpandedSize);
                                if (expanded3DUploadData) {
                                    memcpy(expanded3DUploadData, firstSlice, expandedBytesPerImagePerSlice);
                                    for (uint64_t z = 1; z < texDepth; z++) {
                                        const uint8_t *sliceSrc = (const uint8_t *)srcData + z * bytesPerImage;
                                        uint8_t *sliceDst = expanded3DUploadData + z * expandedBytesPerImagePerSlice;
                                        uint64_t dummyRow = 0, dummyImage = 0;
                                        uint8_t *sliceExpanded = mglCreateChannelExpandedUpload(tex,
                                                                                                 pixelFormat,
                                                                                                 sliceSrc,
                                                                                                 width,
                                                                                                 texHeight,
                                                                                                 bytesPerRow,
                                                                                                 &dummyRow,
                                                                                                 &dummyImage);
                                        if (sliceExpanded) {
                                            memcpy(sliceDst, sliceExpanded, expandedBytesPerImagePerSlice);
                                            free(sliceExpanded);
                                        } else {
                                            memset(sliceDst, 0, expandedBytesPerImagePerSlice);
                                        }
                                    }
                                    srcData = expanded3DUploadData;
                                    bytesPerRow = expandedBytesPerRow;
                                    bytesPerImage = expandedBytesPerImagePerSlice;
                                    addr = (uintptr_t)srcData;
                                }
                            }
                            free(firstSlice);
                        }
                    }

                    uint64_t alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);
                    uint64_t alignedBytesPerRow = bytesPerRow;
                    if (alignedBytesPerRow % alignment != 0) {
                        alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                    }

                    uint64_t addrAlignment = MGLCapabilityTextureAlignment(mglUpCapability(renderer));
                    if (addr % addrAlignment != 0 || alignedBytesPerRow != bytesPerRow) {
                        uint64_t alignedUploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, mglUpMax((uint64_t)height, 1UL));
                        if (alignedUploadRows == 0 || alignedBytesPerRow > (UINT64_MAX / alignedUploadRows)) {
                            MGL_TEX_UPLOAD_DIAG("MGL WARNING: Rejecting aligned 3D upload row overflow (tex=%d level=%d rows=%lu bpr=%lu)\n",
                                  tex->name,
                                  level,
                                  (unsigned long)alignedUploadRows,
                                  (unsigned long)alignedBytesPerRow);
                            if (outSkipped) *outSkipped = 1;
                            return 1;
                        }
                        uint64_t alignedBytesPerImage = alignedBytesPerRow * alignedUploadRows;
                        uint64_t alignedDepth = mglUpMax((uint64_t)depth, 1UL);
                        if (alignedBytesPerImage > (UINT64_MAX / alignedDepth)) {
                            MGL_TEX_UPLOAD_DIAG("MGL WARNING: Rejecting aligned 3D upload size overflow (tex=%d level=%d bpi=%lu depth=%lu)\n",
                                  tex->name,
                                  level,
                                  (unsigned long)alignedBytesPerImage,
                                  (unsigned long)alignedDepth);
                            if (outSkipped) *outSkipped = 1;
                            return 1;
                        }
                        uint64_t alignedSize = alignedBytesPerImage * alignedDepth;
                        if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                            MGL_TEX_UPLOAD_DIAG("MGL WARNING: Rejecting aligned 3D upload staging size=%lu (tex=%d level=%d)\n",
                                  (unsigned long)alignedSize, tex->name, level);
                            if (outSkipped) *outSkipped = 1;
                            return 1;
                        }
                        void *alignedData = mglUploadAlignScratchAcquire(alignment, alignedSize);

                        if (alignedData) {
                            memset(alignedData, 0, alignedSize);
                            uint64_t srcRowSize = bytesPerRow;
                            uint64_t dstRowSize = alignedBytesPerRow;
                            uint64_t texUploadRows = alignedUploadRows;
                            uint64_t texDepth = mglUpMax((uint64_t)depth, 1UL);
                            uint8_t *srcPtr = (uint8_t *)srcData;
                            uint8_t *dstPtr = (uint8_t *)alignedData;

                            for (uint64_t z = 0; z < texDepth; z++) {
                                for (uint64_t row = 0; row < texUploadRows; row++) {
                                    uint64_t copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                    uint64_t dstOffset = z * alignedBytesPerImage + row * dstRowSize;
                                    uint64_t srcOffset = z * bytesPerImage + row * srcRowSize;
                                    memcpy(dstPtr + dstOffset, srcPtr + srcOffset, copySize);
                                    if (dstRowSize > copySize) {
                                        memset(dstPtr + dstOffset + copySize, 0, dstRowSize - copySize);
                                    }
                                }
                            }

                            if (!alignedData) {
                                fprintf(stderr, "MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash\n", level);
                                if (outSkipped) *outSkipped = 1;
                                return 1;
                            }
                            if (alignedBytesPerRow == 0) {
                                fprintf(stderr, "MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash\n", level);
                                if (outSkipped) *outSkipped = 1;
                                return 1;
                            }
                            { /* the .m wrapped this blit in @try/@catch; the guarded call keeps the catch logic (rule 58 (b)) */

                                int uploaded = 0;

                                MglUpSliceBlitCtx blitCtx = { renderer, texture, tex->name, tex->target, alignedData, alignedBytesPerRow, alignedBytesPerImage, width, height, depth, level, 0, 0, &uploaded };

                                char blitFailure[256] = {0};

                                if (!mglPlatformShellGuardedCallCtxReason(

                                        renderer, "3D aligned blit upload", mglUpSliceBlitBody, &blitCtx,

                                        blitFailure, sizeof(blitFailure))) {

                                    fprintf(stderr,

                                            "MGL ERROR: Failed to upload aligned 3D texture data (level %d, face %d): %s\n",

                                            level, face, blitFailure[0] ? blitFailure : "(null)");

                                } else if (!uploaded) {

                                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: 3D aligned blit upload failed (level %d, face %d)\n", level, face);

                                }

                            }
                            mglUploadAlignScratchRelease(alignedData);
                        } else {
                            fprintf(stderr, "MGL ERROR: Failed to allocate aligned memory for 3D texture upload\n");
                        }
                    } else {
                        if (!srcData) {
                            fprintf(stderr, "MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash\n", level);
                            if (outSkipped) *outSkipped = 1;
                            return 1;
                        }
                        if (bytesPerRow == 0) {
                            fprintf(stderr, "MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash\n", level);
                            if (outSkipped) *outSkipped = 1;
                            return 1;
                        }
                        if (bytesPerImage == 0) {
                            fprintf(stderr, "MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d) - SKIPPING to prevent crash\n", level);
                            if (outSkipped) *outSkipped = 1;
                            return 1;
                        }
                        { /* the .m wrapped this blit in @try/@catch; the guarded call keeps the catch logic (rule 58 (b)) */

                            int uploaded = 0;

                            MglUpSliceBlitCtx blitCtx = { renderer, texture, tex->name, tex->target, srcData, bytesPerRow, bytesPerImage, width, height, depth, level, 0, 0, &uploaded };

                            char blitFailure[256] = {0};

                            if (!mglPlatformShellGuardedCallCtxReason(

                                    renderer, "3D direct blit upload", mglUpSliceBlitBody, &blitCtx,

                                    blitFailure, sizeof(blitFailure))) {

                                fprintf(stderr,

                                        "MGL ERROR: Failed to upload 3D texture data (level %d, face %d): %s\n",

                                        level, face, blitFailure[0] ? blitFailure : "(null)");

                            } else if (!uploaded) {

                                MGL_TEX_UPLOAD_DIAG("MGL WARNING: 3D direct blit upload failed (level %d, face %d)\n", level, face);

                            }

                        }
                    }
                    mglUploadExpandScratchRelease(expanded3DUploadData);
                    free(swizzled3DUploadData);
                } else {
                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: Skipping 3D texture upload due to invalid data or parameters\n");
                }

    return 1;
}

/* -uploadDirtyCPUTextureDataNon3DLevel:… */
int mglTextureUploadDirtyNon3DLevel(void *renderer, Texture *tex, void *texture,
                                    uint32_t pixelFormat, int face, int level,
                                    uint64_t width, uint64_t height,
                                    uint64_t depth, int is_array,
                                    int texture1DArrayBackedBy2DArray,
                                    uint32_t tex_type, int *outSkipped)
{
    uint64_t bytesPerRow;
    uint64_t bytesPerImage;
    bool hasExplicitDataSize = false;
    MGLRegionValue region;

                bytesPerRow = tex->faces[face].levels[level].pitch;
                if (bytesPerRow == 0) {
                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: Invalid bytesPerRow (0), skipping upload (tex=%d face=%d level=%d)\n", tex->name, face, level);
                    if (outSkipped) *outSkipped = 1;
                    return 1;
                }

                bytesPerImage = tex->faces[face].levels[level].data_size;
                hasExplicitDataSize = (bytesPerImage > 0);
                if (bytesPerImage == 0) {
                    uint64_t fallbackHeight = (height > 0) ? (uint64_t)height : 1;
                    bytesPerImage = bytesPerRow * fallbackHeight;
                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: data_size was 0, using fallback bytesPerImage=%lu (tex=%d face=%d level=%d)\n",
                          (unsigned long)bytesPerImage, tex->name, face, level);
                }
                if (bytesPerImage == 0) {
                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: Invalid bytesPerImage (0), skipping upload (tex=%d face=%d level=%d)\n", tex->name, face, level);
                    if (outSkipped) *outSkipped = 1;
                    return 1;
                }

                if (is_array)
                {
                    GLuint num_layers;
                    size_t offset;
                    GLubyte *tex_data;
                    int arraySliceIs1D;
                    uint64_t uploadSliceHeight;
                    uint64_t backingBytes;
                    uint64_t logicalBytesPerImage;

                    num_layers = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray)
                        ? tex->faces[face].levels[level].height
                        : tex->faces[face].levels[level].depth;
                    if (num_layers == 0) {
                        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Array texture has 0 layers, skipping upload (tex=%d face=%d level=%d)\n", tex->name, face, level);
                        if (outSkipped) *outSkipped = 1;
                        return 1;
                    }

                    arraySliceIs1D = (tex_type == MGLTextureType1DArray || texture1DArrayBackedBy2DArray);
                    uploadSliceHeight = arraySliceIs1D ? 1UL : mglUpMax((uint64_t)height, 1UL);
                    backingBytes = bytesPerImage;
                    uint64_t uploadSliceRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);
                    if (uploadSliceRows == 0 || bytesPerRow > (UINT64_MAX / uploadSliceRows)) {
                        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Array texture invalid row layout tex=%d face=%d level=%d bpr=%lu rows=%lu\n",
                              tex->name,
                              face,
                              level,
                              (unsigned long)bytesPerRow,
                              (unsigned long)uploadSliceRows);
                        if (outSkipped) *outSkipped = 1;
                        return 1;
                    }
                    logicalBytesPerImage = bytesPerRow * uploadSliceRows;
                    /* data_size is page-rounded; do not treat the slack as
                     * layer stride or reads land in the wrong slice. */
                    uint64_t requiredArrayBytes = 0;
                    uint64_t safeLayerCount = mglUpMax((uint64_t)num_layers, 1UL);
                    if (logicalBytesPerImage == 0 ||
                        logicalBytesPerImage > (UINT64_MAX / safeLayerCount) ||
                        backingBytes < (requiredArrayBytes = logicalBytesPerImage * safeLayerCount)) {
                        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Array texture backing too small for logical slices tex=%d face=%d level=%d backing=%lu layerBytes=%lu layers=%u\n",
                              tex->name,
                              face,
                              level,
                              (unsigned long)backingBytes,
                              (unsigned long)logicalBytesPerImage,
                              num_layers);
                        if (outSkipped) *outSkipped = 1;
                        return 1;
                    }
                    bytesPerImage = logicalBytesPerImage;

                    if (!arraySliceIs1D)
                        region = mglTextureRegion2D(0,0,width,height);
                    else if (height >= 1)
                        region = mglTextureRegion2D(0,0,width,1);
                    else {
                        fprintf(stderr, "MGL TEXTURE ERROR: invalid array texture height=%lu for tex=%u face=%d level=%d\n",
                              (unsigned long)height,
                              tex->name,
                              face,
                              level);
                        return 0;
                    }

                    for(int layer=0; layer<num_layers; layer++)
                    {
                        offset = bytesPerImage * layer;

                        tex_data = (GLubyte *)tex->faces[face].levels[level].data;
                        tex_data += offset;

                        if (tex_data && bytesPerRow > 0 && bytesPerImage > 0) {
                            void *srcData = (void *)tex_data;
                            void *expandedUploadData = NULL;
                            void *swizzledUploadData = NULL;
                            uintptr_t addr = (uintptr_t)srcData;

                            uint64_t effectiveBytesPerRow = bytesPerRow;
                            uint64_t effectiveBytesPerImage = bytesPerImage;
                            if (mglTextureUploadNeedsSwizzleBake(tex)) {
                                uint64_t swzBPR = 0;
                                uint64_t swzBPI = 0;
                                swizzledUploadData =
                                    mglCreateSwizzledUpload(
                                        tex, (const uint8_t *)srcData, width,
                                        uploadSliceHeight, bytesPerRow, &swzBPR,
                                        &swzBPI);
                                if (swizzledUploadData) {
                                    srcData = swizzledUploadData;
                                    effectiveBytesPerRow = swzBPR;
                                    effectiveBytesPerImage = swzBPI;
                                    addr = (uintptr_t)srcData;
                                }
                            }

                            if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                    mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
                                uint64_t expandedBytesPerRow = 0;
                                uint64_t expandedBytesPerImage = 0;
                                expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,
                                                                                   (const uint8_t *)srcData,
                                                                                   width,
                                                                                   uploadSliceHeight,
                                                                                   bytesPerRow,
                                                                                   &expandedBytesPerRow,
                                                                                   &expandedBytesPerImage);
                                if (expandedUploadData) {
                                    srcData = expandedUploadData;
                                    effectiveBytesPerRow = expandedBytesPerRow;
                                    effectiveBytesPerImage = expandedBytesPerImage;
                                    addr = (uintptr_t)srcData;
                                }
                            } else if (!mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                           mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
                                uint64_t expandedBytesPerRow = 0;
                                uint64_t expandedBytesPerImage = 0;
                                expandedUploadData = mglCreateChannelExpandedUpload(tex,
                                                                                     pixelFormat,
                                                                                     (const uint8_t *)srcData,
                                                                                     width,
                                                                                     uploadSliceHeight,
                                                                                     bytesPerRow,
                                                                                     &expandedBytesPerRow,
                                                                                     &expandedBytesPerImage);
                                if (expandedUploadData) {
                                    srcData = expandedUploadData;
                                    effectiveBytesPerRow = expandedBytesPerRow;
                                    effectiveBytesPerImage = expandedBytesPerImage;
                                    addr = (uintptr_t)srcData;
                                }
                            }

                            uint64_t alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);
                            uint64_t alignedBytesPerRow = effectiveBytesPerRow;
                            if (alignedBytesPerRow % alignment != 0) {
                                alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                            }

                            if (addr % alignment != 0 || alignedBytesPerRow != effectiveBytesPerRow) {
                                uint64_t alignedUploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, uploadSliceHeight);
                                if (alignedUploadRows == 0 || alignedBytesPerRow > (UINT64_MAX / alignedUploadRows)) {
                                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: Rejecting aligned array upload row layout bpr=%lu rows=%lu (tex=%d level=%d layer=%d)\n",
                                          (unsigned long)alignedBytesPerRow,
                                          (unsigned long)alignedUploadRows,
                                          tex->name,
                                          level,
                                          layer);
                                    free(swizzledUploadData);
                                    mglUploadExpandScratchRelease(expandedUploadData);
                                    continue;
                                }
                                uint64_t alignedBytesPerImage = alignedBytesPerRow * alignedUploadRows;
                                uint64_t alignedSize = alignedBytesPerImage;
                                if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: Rejecting aligned array upload staging size=%lu (tex=%d level=%d layer=%d)\n",
                                          (unsigned long)alignedSize, tex->name, level, layer);
                                    free(swizzledUploadData);
                                    mglUploadExpandScratchRelease(expandedUploadData);
                                    continue;
                                }
                                void *alignedData = mglUploadAlignScratchAcquire(alignment, alignedSize);

                                if (alignedData) {
                                    memset(alignedData, 0, alignedSize);
                                    uint64_t srcRowSize = effectiveBytesPerRow;
                                    uint64_t dstRowSize = alignedBytesPerRow;
                                    uint8_t *srcPtr = (uint8_t *)srcData;
                                    uint8_t *dstPtr = (uint8_t *)alignedData;

                                    for (uint64_t row = 0; row < alignedUploadRows; row++) {
                                        uint64_t copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                        memcpy(dstPtr + (row * dstRowSize), srcPtr + (row * srcRowSize), copySize);
                                        if (dstRowSize > copySize) {
                                            memset(dstPtr + (row * dstRowSize) + copySize, 0, dstRowSize - copySize);
                                        }
                                    }

                                    if (!alignedData) {
                                        fprintf(stderr, "MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash\n", level, layer);
                                        continue;
                                    }
                                    if (alignedBytesPerRow == 0) {
                                        fprintf(stderr, "MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash\n", level, layer);
                                        continue;
                                    }
                                    if (bytesPerImage == 0) {
                                        fprintf(stderr, "MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash\n", level, layer);
                                        continue;
                                    }
                                    /* the .m wrapped this blit in @try/@catch; the
                                     * guarded call keeps the catch logic (rule 58 (b)) */
                                    {
                                        int uploaded = 0;
                                        MglUpSliceBlitCtx blitCtx = { renderer, texture,
                                            tex->name, tex->target, alignedData,
                                            alignedBytesPerRow, alignedBytesPerImage,
                                            width, uploadSliceHeight, 1, level, layer,
                                            0, &uploaded };
                                        char blitFailure[256] = {0};
                                        if (!mglPlatformShellGuardedCallCtxReason(
                                                renderer, "array texture blit upload",
                                                mglUpSliceBlitBody, &blitCtx,
                                                blitFailure, sizeof(blitFailure))) {
                                            fprintf(stderr,
                                                    "MGL ERROR: Failed to upload aligned array texture data (level %d, layer %d): %s\n",
                                                    level, layer,
                                                    blitFailure[0] ? blitFailure : "(null)");
                                        } else if (hasExplicitDataSize) {
                                            if (!uploaded) {
                                                MGL_TEX_UPLOAD_DIAG("MGL WARNING: Array texture blit upload failed (level %d, layer %d)\n", level, layer);
                                            }
                                        } else {
                                            MGL_TEX_UPLOAD_DIAG("MGL INFO: Skipping array upload with synthesized data size (level %d, layer %d)\n", level, layer);
                                        }
                                    }
                                    mglUploadAlignScratchRelease(alignedData);
                                } else {
                                    fprintf(stderr, "MGL ERROR: Failed to allocate aligned memory for array texture upload (level %d, layer %d)\n", level, layer);
                                }
                            } else {
                                if (!srcData) {
                                    fprintf(stderr, "MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash\n", level, layer);
                                    free(swizzledUploadData);
                                    mglUploadExpandScratchRelease(expandedUploadData);
                                    continue;
                                }
                                if (effectiveBytesPerRow == 0) {
                                    fprintf(stderr, "MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash\n", level, layer);
                                    free(swizzledUploadData);
                                    mglUploadExpandScratchRelease(expandedUploadData);
                                    continue;
                                }
                                if (effectiveBytesPerImage == 0) {
                                    fprintf(stderr, "MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, layer %d) - SKIPPING to prevent crash\n", level, layer);
                                    free(swizzledUploadData);
                                    mglUploadExpandScratchRelease(expandedUploadData);
                                    continue;
                                }
                                if (hasExplicitDataSize) {
                                    int uploaded = mglTextureUploadSliceViaBlit(
            renderer, texture, tex->name, tex->target, srcData, effectiveBytesPerRow, effectiveBytesPerImage, width, uploadSliceHeight, 1, level, layer, 0);
                                    if (!uploaded) {
                                        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Array texture direct blit upload failed (level %d, layer %d)\n", level, layer);
                                    }
                                } else {
                                    MGL_TEX_UPLOAD_DIAG("MGL INFO: Skipping array upload with synthesized data size (level %d, layer %d)\n", level, layer);
                                }
                            }
                            free(swizzledUploadData);
                            mglUploadExpandScratchRelease(expandedUploadData);
                        } else {
                            MGL_TEX_UPLOAD_DIAG("MGL WARNING: Skipping array texture upload due to invalid data or parameters\n");
                        }
                    }
                }
                else
                {
                    DEBUG_PRINT("tex void *data update %d\n", tex->name);

                    if (tex->faces[face].levels[level].data && bytesPerRow > 0 && bytesPerImage > 0) {
                        void *srcData = (void *)tex->faces[face].levels[level].data;
                        void *swizzledUploadData = NULL;
                        void *expandedUploadData = NULL;
                        uintptr_t addr = (uintptr_t)srcData;
                        if (level == 0 && face == 0 && mglTextureUploadNeedsSwizzleBake(tex)) {
                            uint64_t swizzledBytesPerRow = 0;
                            uint64_t swizzledBytesPerImage = 0;
                            swizzledUploadData = mglCreateSwizzledUpload(tex,
                                                                                      (const uint8_t *)srcData,
                                                                                      width,
                                                                                      mglUpMax((uint64_t)height, 1UL),
                                                                                      bytesPerRow,
                                                                                      &swizzledBytesPerRow,
                                                                                      &swizzledBytesPerImage);
                            if (swizzledUploadData) {
                                srcData = swizzledUploadData;
                                bytesPerRow = swizzledBytesPerRow;
                                bytesPerImage = swizzledBytesPerImage;
                                addr = (uintptr_t)srcData;
                                if (mglTraceLogIsEnabled()) {
                                    const uint8_t *swz = (const uint8_t *)swizzledUploadData;
                                    mglTraceLog("TEXTURE_SWIZZLE_UPLOAD_R8 tex=%u target=0x%x swzR=0x%x size=%lux%lu bpr=%lu first=%u",
                                                (unsigned)tex->name,
                                                (unsigned)tex->target,
                                                (unsigned)tex->params.swizzle_r,
                                                (unsigned long)width,
                                                (unsigned long)mglUpMax((uint64_t)height, 1UL),
                                                (unsigned long)bytesPerRow,
                                                swz[0]);
                                }
                            }
                        }
                        if (!swizzledUploadData &&
                            !mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                            mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, pixelFormat)) {
                            uint64_t expandedBytesPerRow = 0;
                            uint64_t expandedBytesPerImage = 0;
                            expandedUploadData = mglCreateRGBA8ExpandedUpload(tex,
                                                                               (const uint8_t *)srcData,
                                                                               width,
                                                                               mglUpMax((uint64_t)height, 1UL),
                                                                               bytesPerRow,
                                                                               &expandedBytesPerRow,
                                                                               &expandedBytesPerImage);
                            if (expandedUploadData) {
                                srcData = expandedUploadData;
                                bytesPerRow = expandedBytesPerRow;
                                bytesPerImage = expandedBytesPerImage;
                                addr = (uintptr_t)srcData;
                            }
                        } else if (!swizzledUploadData &&
                                   !mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) &&
                                   mglTextureNeedsChannelExpansion(tex->internalformat, pixelFormat)) {
                            uint64_t expandedBytesPerRow = 0;
                            uint64_t expandedBytesPerImage = 0;
                            expandedUploadData = mglCreateChannelExpandedUpload(tex,
                                                                                 pixelFormat,
                                                                                 (const uint8_t *)srcData,
                                                                                 width,
                                                                                 mglUpMax((uint64_t)height, 1UL),
                                                                                 bytesPerRow,
                                                                                 &expandedBytesPerRow,
                                                                                 &expandedBytesPerImage);
                            if (expandedUploadData) {
                                srcData = expandedUploadData;
                                bytesPerRow = expandedBytesPerRow;
                                bytesPerImage = expandedBytesPerImage;
                                addr = (uintptr_t)srcData;
                            }
                        }

                        /* Block-compressed pitch is bytes/row-of-blocks. Do not
                         * pad it to the linear 64-byte alignment — Metal rejects
                         * or mis-reads BC6H/BC7 uploads with a padded stride
                         * (4x4 BPTC is pitch 16). */
                        const uint64_t blockH2D =
                            mglRenderMetalCompressedBlockHeight(pixelFormat);
                        const int is_block_compressed_2d = (blockH2D > 1u);
                        uint64_t alignment = mglRendererOptimalAlignmentForPixelFormat(pixelFormat);
                        uint64_t alignedBytesPerRow = bytesPerRow;
                        if (!is_block_compressed_2d &&
                            alignedBytesPerRow % alignment != 0) {
                            alignedBytesPerRow = ((alignedBytesPerRow + alignment - 1) / alignment) * alignment;
                        }

                        if (addr % alignment != 0 || alignedBytesPerRow != bytesPerRow) {
                            uint64_t texHeight = mglUpMax((uint64_t)height, 1UL);
                            uint64_t uploadRows = mglMetalUploadRowsForPixelFormat(pixelFormat, texHeight);
                            if (uploadRows == 0 || alignedBytesPerRow > (UINT64_MAX / uploadRows)) {
                                MGL_TEX_UPLOAD_DIAG("MGL WARNING: Rejecting aligned 2D upload row layout bpr=%lu rows=%lu (tex=%d level=%d face=%d)\n",
                                      (unsigned long)alignedBytesPerRow,
                                      (unsigned long)uploadRows,
                                      tex->name,
                                      level,
                                      face);
                                free(swizzledUploadData);
                                mglUploadExpandScratchRelease(expandedUploadData);
                                if (outSkipped) *outSkipped = 1;
                                return 1;
                            }
                            uint64_t alignedBytesPerImage = alignedBytesPerRow * uploadRows;
                            uint64_t alignedSize = alignedBytesPerImage;
                            if (alignedSize == 0 || alignedSize > (512 * 1024 * 1024)) {
                                MGL_TEX_UPLOAD_DIAG("MGL WARNING: Rejecting aligned 2D upload staging size=%lu (tex=%d level=%d face=%d)\n",
                                      (unsigned long)alignedSize, tex->name, level, face);
                                free(swizzledUploadData);
                                mglUploadExpandScratchRelease(expandedUploadData);
                                if (outSkipped) *outSkipped = 1;
                                return 1;
                            }
                            void *alignedData = mglUploadAlignScratchAcquire(alignment, alignedSize);

                            if (alignedData) {
                                memset(alignedData, 0, alignedSize);
                                uint64_t srcRowSize = bytesPerRow;
                                uint64_t dstRowSize = alignedBytesPerRow;
                                uint8_t *srcPtr = (uint8_t *)srcData;
                                uint8_t *dstPtr = (uint8_t *)alignedData;

                                for (uint64_t row = 0; row < uploadRows; row++) {
                                    uint64_t copySize = (srcRowSize < dstRowSize) ? srcRowSize : dstRowSize;
                                    memcpy(dstPtr + (row * dstRowSize), srcPtr + (row * srcRowSize), copySize);
                                    if (dstRowSize > copySize) {
                                        memset(dstPtr + (row * dstRowSize) + copySize, 0, dstRowSize - copySize);
                                    }
                                }

                                if (!alignedData) {
                                    fprintf(stderr, "MGL SECURITY ERROR: NULL alignedData passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash\n", level, face);
                                    mglUploadAlignScratchRelease(alignedData);
                                    if (outSkipped) *outSkipped = 1;
                                    return 1;
                                }
                                if (alignedBytesPerRow == 0) {
                                    fprintf(stderr, "MGL SECURITY ERROR: Invalid alignedBytesPerRow (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash\n", level, face);
                                    mglUploadAlignScratchRelease(alignedData);
                                    if (outSkipped) *outSkipped = 1;
                                    return 1;
                                }
                                if (bytesPerImage == 0) {
                                    fprintf(stderr, "MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash\n", level, face);
                                    mglUploadAlignScratchRelease(alignedData);
                                    if (outSkipped) *outSkipped = 1;
                                    return 1;
                                }
                                if (hasExplicitDataSize) {
                                    int uploaded = mglTextureUploadSliceViaBlit(
            renderer, texture, tex->name, tex->target, alignedData, alignedBytesPerRow, alignedBytesPerImage, width, height, 1, level, face, 0);
                                    if (!uploaded) {
                                        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Aligned 2D blit upload failed (level %d, face %d)\n", level, face);
                                    }
                                } else {
                                    MGL_TEX_UPLOAD_DIAG("MGL INFO: Skipping 2D upload with synthesized data size (level %d, face %d)\n", level, face);
                                }
                                mglUploadAlignScratchRelease(alignedData);
                            } else {
                                fprintf(stderr, "MGL ERROR: Failed to allocate aligned memory for 2D texture upload (level %d, face %d)\n", level, face);
                            }
                        } else {
                            if (!srcData) {
                                fprintf(stderr, "MGL SECURITY ERROR: NULL srcData passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash\n", level, face);
                                if (outSkipped) *outSkipped = 1;
                                return 1;
                            }
                            if (bytesPerRow == 0) {
                                fprintf(stderr, "MGL SECURITY ERROR: Invalid bytesPerRow (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash\n", level, face);
                                if (outSkipped) *outSkipped = 1;
                                return 1;
                            }
                            if (bytesPerImage == 0) {
                                fprintf(stderr, "MGL SECURITY ERROR: Invalid bytesPerImage (0) passed to Metal replaceRegion (level %d, face %d) - SKIPPING to prevent crash\n", level, face);
                                if (outSkipped) *outSkipped = 1;
                                return 1;
                            }
                            if (hasExplicitDataSize) {
                                int uploaded = mglTextureUploadSliceViaBlit(
            renderer, texture, tex->name, tex->target, srcData, bytesPerRow, bytesPerImage, width, height, 1, level, face, 0);
                                if (!uploaded) {
                                    MGL_TEX_UPLOAD_DIAG("MGL WARNING: 2D direct blit upload failed (level %d, face %d)\n", level, face);
                                }
                            } else {
                                MGL_TEX_UPLOAD_DIAG("MGL INFO: Skipping 2D upload with synthesized data size (level %d, face %d)\n", level, face);
                            }
                        }
                        free(swizzledUploadData);
                        mglUploadExpandScratchRelease(expandedUploadData);
                    } else {
                        MGL_TEX_UPLOAD_DIAG("MGL WARNING: Skipping 2D texture upload due to invalid data or parameters\n");
                    }
                }

    return 1;
}

/* -uploadDirtyCPUTextureData:… */
int mglTextureUploadDirty(void *renderer, Texture *tex, void *texture,
                          uint32_t pixelFormat, uint32_t num_faces,
                          uint32_t upload_level_count, int is_array,
                          int texture1DBackedBy2D,
                          int texture1DArrayBackedBy2DArray, uint32_t tex_type,
                          int *outAllLevelsUploaded)
{
    MGL_ASSERT_GL_THREAD();

    if (kMGLDiagnosticStateLogs) {
        mglTraceLog("MGL DEBUG: DIRTY_TEXTURE_DATA detected - attempting texture filling");
        mglTraceLog("MGL DEBUG: Texture details: target=0x%x, internalformat=0x%x, levels=%d effectiveLevels=%u",
                      tex->target, tex->internalformat, tex->num_levels, upload_level_count);
    }

    MGLRegionValue region;
    uint64_t width, height, depth;
    int anyLevelSkipped = 0;

    for(int face=0; face<num_faces; face++)
    {
        for (int level=0; level<upload_level_count; level++)
        {
            TextureLevel *uploadLevel = &tex->faces[face].levels[level];
            if (!mglTextureLevelHasUploadableCPUData(uploadLevel)) {
                /* TexImage NULL, GenerateMipmap, or never written: nothing to
                 * blit from CPU. That is not an upload failure; retrying on
                 * every bind closes the render encoder (cube gather CTS).
                 * Rendered levels live only on the GPU and must be
                 * reported so a recreated texture repairs them. */
                if (uploadLevel &&
                    uploadLevel->last_init_source == kTexRenderTargetWrite)
                    anyLevelSkipped = 1;
                continue;
            }

            width = tex->faces[face].levels[level].width;
            height = tex->faces[face].levels[level].height;
            depth = tex->faces[face].levels[level].depth;

            if (texture1DBackedBy2D)
                region = mglTextureRegion2D(0,0,width,1);
            else if (depth > 1)
                region = mglTextureRegion3D(0,0,0,width,height,depth);
            else if (height > 1)
                region = mglTextureRegion2D(0,0,width,height);
            else
                region = mglTextureRegion1D(0,width);

            uint64_t bytesPerRow;
            uint64_t bytesPerImage;
            bool hasExplicitDataSize = false;

            int levelSkipped = 0;

            if (tex_type == MGLTextureType3D)
            {
                if (!mglTextureUploadDirty3DLevel(
                        renderer, tex, texture, pixelFormat, face, level, width,
                        height, depth, &levelSkipped)) {
                    return 0;
                }
            }
            else
            {
                if (!mglTextureUploadDirtyNon3DLevel(
                        renderer, tex, texture, pixelFormat, face, level, width,
                        height, depth, is_array, texture1DArrayBackedBy2DArray,
                        tex_type, &levelSkipped)) {
                    return 0;
                }
            }

            if (levelSkipped)
                anyLevelSkipped = 1;
        }
    }

    if (outAllLevelsUploaded)
        *outAllLevelsUploaded = !anyLevelSkipped;

    return 1;
}


/* The @try of the texture create in the create path (rule 58 (b)): the guarded
 * body publishes the +1 handle through the ctx. */
typedef struct MglUpCreateTextureCtx_t {
    const MGLRenderTextureDescriptorState *descriptor;
    void *out_texture;
    /* Sparse textures come out of a placement heap; the caller keeps it. */
    void *out_heap;
} MglUpCreateTextureCtx;

static void *mglUpCreateTexture(
    const MGLRenderTextureDescriptorState *descriptor, void **heap_out)
{
    void *texture = NULL;
    if (heap_out)
        *heap_out = NULL;
    if (descriptor && descriptor->placement_sparse_page_size != 0) {
        if (mglRenderCreateSparseTextureFromState(descriptor, NULL, heap_out,
                                                  &texture) == 0 &&
            texture) {
            return texture;
        }
        return NULL;
    }
    if (mglRenderCreateTextureFromState(descriptor, NULL, &texture) == 0 &&
        texture) {
        return texture;
    }
    return NULL;
}

static int mglUpCreateTextureBody(void *renderer, void *rawCtx)
{
    MglUpCreateTextureCtx *ctx = (MglUpCreateTextureCtx *)rawCtx;
    ctx->out_texture = mglUpCreateTexture(ctx->descriptor, &ctx->out_heap);
    (void)renderer;
    return 1;
}

/* -createMTLTextureFromGLTexture: (log 197).  Every callee it had is already C,
 * which is why the move is a pure translation. */
void *mglTextureCreateFromGLTexture(void *renderer, Texture *tex)
{
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    GLMContext ctx = areas.ctx;
    MGL_ASSERT_GL_THREAD();
    mglMetalCountCreate(MGLMetalKindTexture);
    // PROPER FIX: Enhanced pre-creation validation to prevent AGX driver issues
    void *device = mglRendererBackendGetDevice(areas.backend);
    void *commandQueue = mglRendererBackendGetCommandQueue(areas.backend);
    if (!device || !commandQueue) {
        fprintf(stderr, "MGL ERROR: Metal device or command queue not available for texture creation\n");
        return NULL;
    }

    // Check if we're in a recovery state that would make texture creation futile
    if (mglRendererShouldSkipGPUOperations(renderer)) {
        MGL_TEX_UPLOAD_DIAG("MGL AGX: GPU operations temporarily suspended during recovery\n");
        return NULL;
    }

    // Validate texture dimensions to prevent Metal assertion failures.
    // Texture buffers (GL_TEXTURE_BUFFER) can have very large widths (millions of texels)
    // since they map to MGLTextureTypeTextureBuffer which uses GPU address space.
    if (!mglRenderTextureDimsValid(tex->target, tex->width, tex->height,
                                   tex->depth)) {
            fprintf(stderr, "MGL ERROR: Invalid texture dimensions %dx%dx%d - rejecting\n",
                  tex ? tex->width : 0, tex ? tex->height : 0, tex ? tex->depth : 0);
            tex->dirty_bits = 0;
            return NULL;
        }

    if (mglRenderIsTextureBufferTarget(tex->target)) {
        return mglTextureCreateMTLTexelBufferTexture(renderer, tex);
    }

    uint64_t width, height, depth;

    MGLRenderTextureDescriptorState tex_desc = {0};
    uint32_t tex_type;
    uint32_t pixelFormat;
    uint num_faces;
    GLuint effective_mipmap_levels;
    GLuint upload_level_count;
    int storageMipmapped;
    int mipmapped;
    int is_array;
    int texture1DBackedBy2D;
    int texture1DArrayBackedBy2DArray;

    effective_mipmap_levels = 0;
    upload_level_count = 0;
    storageMipmapped = 0;

    MGLRenderTextureTargetPlan targetPlan = {0};
    if (mglRenderTextureTargetPlan(
            (uint32_t)tex->target,
            (uint32_t)tex->samples,
            &targetPlan) != 0) {
        fprintf(stderr, "MGL TEXTURE ERROR: unsupported texture target 0x%x for Metal texture creation tex=%u\n",
              tex->target,
              tex->name);
        return NULL;
    }
    tex_type = (uint32_t)targetPlan.texture_type;
    num_faces = (uint)targetPlan.num_faces;
    is_array = targetPlan.is_array != 0u;
    texture1DBackedBy2D = targetPlan.texture_1d_backed_by_2d != 0u;
    texture1DArrayBackedBy2DArray =
        targetPlan.texture_1d_array_backed_by_2d_array != 0u;

    int effectiveMipmapped = 0;
    if (!mglTextureCheckCompleteness(tex, tex_type, num_faces,
                                     &effective_mipmap_levels,
                                     &effectiveMipmapped)) {
        return NULL;
    }
    storageMipmapped = effectiveMipmapped ? 1 : 0;

    // Pixel format for the GL internal format, with the swizzle storage
    // substitution; the AGX substitution is the plan's.
    pixelFormat = mtlPixelFormatForGLTex(tex);
    int expandsSingleChannelSwizzle = mglTextureUploadNeedsSingleChannelSwizzle(tex);
    pixelFormat = mglRenderResolveUploadSwizzlePixelFormat(
        pixelFormat, expandsSingleChannelSwizzle ? 1 : 0,
        mglRenderSingleChannelSwizzleStoragePixelFormat(
            (uint32_t)tex->internalformat),
        mglTextureUploadNeedsIntegerMultiChannelSwizzleBake(tex) ? 1 : 0,
        mglRenderIntegerMultiChannelSwizzleStoragePixelFormat(
            (uint32_t)tex->internalformat),
        mglTextureUploadNeedsStencilSwizzleBake(tex) ? 1 : 0,
        mglRenderStencilSwizzleStoragePixelFormat(),
        mglTextureUploadNeedsDepthStencilDepthSwizzleBake(tex) ? 1 : 0,
        mglRenderSingleChannelSwizzleStoragePixelFormat(
            (uint32_t)tex->internalformat));


    width = tex->width;
    height = tex->height;
    depth = tex->depth;

    MGLTexturePlanInput planIn;
    memset(&planIn, 0, sizeof(planIn));
    planIn.target = targetPlan;
    planIn.width = width;
    planIn.height = height;
    planIn.depth = depth;
    planIn.access = (uint32_t)tex->access;
    planIn.is_render_target = tex->is_render_target ? 1 : 0;
    planIn.effective_mipmap_levels = effective_mipmap_levels;
    planIn.effective_mipmapped = effectiveMipmapped;
    planIn.tex_mipmapped = tex->mipmapped ? 1 : 0;
    planIn.pixel_format = pixelFormat;
    planIn.format_is_depth_or_stencil =
        mglMetalPixelFormatIsDepthOrStencil(pixelFormat) ? 1 : 0;
    planIn.first_uploadable_level = mglTextureFirstUploadableLevel(
        tex, (int)num_faces, mglUpMax(effective_mipmap_levels, (GLuint)1));
    planIn.dirty_texture_data = (tex->dirty_bits & DIRTY_TEXTURE_DATA) ? 1 : 0;
    planIn.conservative_cpu_cache =
        MGLCapabilityUseConservativeCPUCache(mglUpCapability(renderer)) ? 1 : 0;
    MGLTexturePlan plan;
    (void)mglTexturePlanBuild(&planIn, &plan);
    if (plan.clear_tex_mipmapped) {
        tex->mipmapped = false;
    }

    tex_desc = plan.desc;
    tex_type = plan.upload_texture_type;
    pixelFormat = plan.pixel_format;
    effective_mipmap_levels = plan.effective_mipmap_levels;
    upload_level_count = plan.upload_level_count;
    storageMipmapped = plan.storage_mipmapped;
    mipmapped = storageMipmapped;
    is_array = plan.is_array;
    texture1DBackedBy2D = plan.texture_1d_backed_by_2d;
    texture1DArrayBackedBy2DArray = plan.texture_1d_array_backed_by_2d_array;
    const bool hasUploadableCPUData = plan.has_uploadable_cpu_data != 0;

    // Normalize depth/array semantics per Metal texture type.
    if ((tex_type == MGLTextureTypeCube ||
         tex_type == MGLTextureTypeCubeArray) &&
        !mglRenderCubeFaceSizeValid((uint64_t)width, (uint64_t)height)) {
            fprintf(stderr, "MGL ERROR: invalid cube texture size %lux%lu for tex=%u glTarget=0x%x\n",
                  (unsigned long)width, (unsigned long)height, tex->name, tex->target);
    }
    if (tex_type == MGLTextureTypeCubeArray) {
        uint64_t cubeCount = (uint64_t)depth;
        if (cubeCount > 1u && (cubeCount % 6u) != 0u) {
            MGL_TEX_UPLOAD_DIAG("MGL WARNING: cube-array depth=%lu is not a multiple of 6, treating as cube count\n",
                  (unsigned long)cubeCount);
        }
    }
    if (plan.status != MGL_TEXTURE_PLAN_OK) {
            fprintf(stderr, "MGL TEXTURE ERROR: invalid texture access 0x%x for tex=%u\n",
                  tex->access,
                  tex->name);
            return NULL;
    }

    if (tex_desc.texture_type == MGLTextureTypeCube || tex_desc.texture_type == MGLTextureTypeCubeArray) {
        fprintf(stderr, "MGL CUBE DESC tex=%u glTarget=0x%x type=%lu width=%lu height=%lu depth=%lu arrayLength=%lu pixelFormat=%lu usage=%lu storage=%lu mipmapped=%d\n",
              tex->name,
              tex->target,
              (unsigned long)tex_desc.texture_type,
              (unsigned long)tex_desc.width,
              (unsigned long)tex_desc.height,
              (unsigned long)tex_desc.depth,
              (unsigned long)tex_desc.array_length,
              (unsigned long)tex_desc.pixel_format,
              (unsigned long)tex_desc.usage,
              (unsigned long)tex_desc.storage_mode,
              (int)mipmapped);
    }

    void *texture;

    // CRITICAL FIX: Safe texture creation with proper validation
    /* The .m created the texture inside @try so a Metal throw ran its catch;
     * the guarded call does the same and hands the reason back (rule 58 (b)).
     * Note the texture identity travels through the ctx: the guarded body must
     * publish it even though the caller owns the result. */
    /* A sparse texture asks the descriptor for its page size; everything else
     * in the create path then picks the sparse primitive on its own. */
    /* MTLSparsePageSize64 == 102 (not the texel count).  Non-zero also
     * gates the sparse create path in mglUpCreateTexture. */
    if (tex->params.sparse)
        tex_desc.placement_sparse_page_size = 102; /* SparsePageSize64 */
    MglUpCreateTextureCtx createCtx = { &tex_desc, NULL, NULL };
    char createFailure[256] = {0};
    if (!mglPlatformShellGuardedCallCtxReason(
            renderer, "texture creation", mglUpCreateTextureBody, &createCtx,
            createFailure, sizeof(createFailure))) {
        fprintf(stderr, "MGL ERROR: Exception creating texture: %s\n",
                createFailure[0] ? createFailure : "(null)");
        mglRendererRecordGPUError(renderer);
        return NULL;
    }
    texture = createCtx.out_texture;
    /* The heap outlives this call: page commitment needs it and the texture's
     * destroy path releases it. */
    if (texture && createCtx.out_heap) {
        tex->sparse_heap = createCtx.out_heap;
        tex->sparse_heap_next_tile = 0;
        if (tex->sparse_commit_count > 0u &&
            mglSparseReplayCommitments(tex, texture) != 0) {
            fprintf(stderr,
                    "MGL ERROR: sparse backing recreation could not restore commitments tex=%u\n",
                    (unsigned)tex->name);
            mglSafeReleaseMetalObj(&texture);
            mglSafeReleaseMetalObj(&tex->sparse_heap);
            return NULL;
        }
    }

    // CRITICAL FIX: Validate texture creation result instead of asserting
    if (!texture) {
        fprintf(stderr, "MGL ERROR: Failed to create Metal texture with descriptor\n");
        return NULL;
    }

    int cpuUploadRequired =
        ((tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) && hasUploadableCPUData;
    int cpuUploadVerified = !cpuUploadRequired;
    int allLevelsUploaded = 1;

    if (cpuUploadRequired)
    {
        if (!mglTextureUploadDirty(
                renderer, tex, texture, pixelFormat,
                num_faces, upload_level_count, is_array, texture1DBackedBy2D,
                texture1DArrayBackedBy2DArray, tex_type, &allLevelsUploaded)) {
            return NULL;
        }
    }
    else
    {
        if (hasUploadableCPUData) {
            mglTextureReUploadExisting(renderer, tex, texture,
                                        pixelFormat, num_faces, upload_level_count,
                                        is_array, texture1DBackedBy2D,
                                        texture1DArrayBackedBy2DArray, tex_type);
        } else if (tex->is_render_target || mglMetalPixelFormatIsDepthOrStencil(pixelFormat)) {
            static uint64_t s_skipRenderTargetFillLogs = 0;
            uint64_t hit = ++s_skipRenderTargetFillLogs;
            if (hit <= 8ull || (hit % 2048ull) == 0ull) {
                fprintf(stderr, "MGL TEXTURE SKIP implicit fill tex=%u renderTarget=%u format=%lu sourceSafe=0 hit=%llu\n",
                      (unsigned)tex->name,
                      (unsigned)tex->is_render_target,
                      (unsigned long)pixelFormat,
                      (unsigned long long)hit);
            }
        } else {
            mglTextureFillSafeInitialContents(renderer, texture, tex,
                                          pixelFormat);
        }
    }

    if (cpuUploadRequired && mglRenderTextureTargetIs2D((uint32_t)tex->target) &&
        mglPdTextureInfo(texture).texture_type == MGLTextureType2D &&
        !mglTextureUploadNeedsSwizzleBake(tex)) {
        int fullCPUUploadVerified = mglTextureUploadFullCPUData(
            renderer, tex, texture,
            "createMTLTexture.cpuData");
        cpuUploadVerified = allLevelsUploaded && fullCPUUploadVerified;
    } else if (cpuUploadRequired) {
        /*
         * Non-2D uploads still use the legacy creation path above. The current GUI
         * atlas failure is 2D; avoid changing array/cube semantics in this pass.
         * If any mip level was skipped (invalid layout, NULL data, etc.) keep
         * DIRTY_TEXTURE_DATA set so the level gets retried on next bind.
         */
        cpuUploadVerified = allLevelsUploaded;
    }

    if (cpuUploadRequired && !cpuUploadVerified) {
        static uint64_t s_createTextureCPUUploadIncompleteLogs = 0;
        uint64_t hit = ++s_createTextureCPUUploadIncompleteLogs;
        if (hit <= 8ull || (hit % 2048ull) == 0ull) {
            TextureLevel *level0 = mglTraceTextureBaseLevel(tex);
            fprintf(stderr, "MGL TEXTURE CREATE CPU-UPLOAD INCOMPLETE tex=%u target=0x%x dirtyBefore=0x%x level0=%ux%u source=%u upload=%lu hit=%llu\n",
                  (unsigned)tex->name,
                  (unsigned)tex->target,
                  (unsigned)tex->dirty_bits,
                  level0 ? (unsigned)level0->width : 0u,
                  level0 ? (unsigned)level0->height : 0u,
                  level0 ? (unsigned)level0->last_init_source : 0u,
                  (unsigned long)(level0 ? level0->last_upload_size : 0u),
                  (unsigned long long)hit);
        }
        tex->dirty_bits &= ~(DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_ACCESS);
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    } else {
        tex->dirty_bits = 0;
    }

    mglTextureLogMipDiagnostics(renderer, tex,
                                texture,
                                effective_mipmap_levels);

    mglRendererRecordGPUSuccess(renderer);

    return texture;
}


/* === The texSubImage trio (log 198) =======================================
 * -mtlTexSubImage:… (the METAL_LOCK wrapper), -mtlTexSubImageLocked:… and
 * -mtlTexSubImageBytes:… moved from MGLRenderer+Texture.m.  Their only callers
 * are this file's C entries, so no port moves.
 */

/* -mtlTexSubImageLocked:… */
int mglTextureSubImage(void *renderer, GLMContext glm_ctx, Texture *tex,
                       Buffer *buf, uint64_t src_offset, uint64_t src_pitch,
                       uint64_t src_image_size, uint64_t src_size, uint32_t slice,
                       uint32_t level, uint64_t width, uint64_t height,
                       uint64_t depth, uint64_t xoffset, uint64_t yoffset,
                       uint64_t zoffset)
{
    if (!renderer) {
        return 0;
    }
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    GLMContext ctx = areas.ctx;
    void *device = mglRendererBackendGetDevice(areas.backend);
    (void)ctx;
    (void)device;
    if (!tex || !buf) {
        fprintf(stderr, "MGL ERROR: mtlTexSubImage called with null tex/buf (tex=%p buf=%p)\n", tex, buf);
        return 0;
    }

    if (src_pitch == 0 || width == 0 || height == 0) {
        fprintf(stderr, "MGL ERROR: mtlTexSubImage invalid dimensions/pitch tex=%u width=%zu height=%zu src_pitch=%zu\n",
              tex->name, width, height, src_pitch);
        return 0;
    }

    // we can deal with a null buffer but we need a texture
    if (mglBufferMtl(buf) == NULL)
    {
        mglRendererBindMTLBuffer(renderer, buf);
        if (mglBufferMtl(buf) == NULL) {
            return 0;
        }
    }

    void *buffer = mglBufferMtl(buf);
    if (!buffer) {
        fprintf(stderr, "MGL ERROR: mtlTexSubImage missing Metal buffer object tex=%u\n", tex->name);
        return 0;
    }


    if (mglTextureMtl(tex)) {
        void *dstTexture = mglTextureMtl(tex);
        uint32_t dstPixelFormat = mglPdTextureInfo(dstTexture).pixel_format;
        int needsChannelExpand = mglTextureNeedsChannelExpansion(tex->internalformat, dstPixelFormat);
        int needsRGBA8Expand = 0;
        if (!needsChannelExpand) {
            needsRGBA8Expand = mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat, dstPixelFormat);
        }
        if (needsChannelExpand || needsRGBA8Expand) {
            uint32_t rgbDst = 0u;
            uint64_t dstBytesPerPixel = 4u;
            if (needsChannelExpand &&
                mglRenderRGBExpandParams(dstPixelFormat, NULL, &rgbDst,
                                         NULL)) {
                dstBytesPerPixel = (uint64_t)rgbDst * 4u;
            } else if (needsChannelExpand) {
                dstBytesPerPixel = 16u;
            }
            uint64_t cpuBytesPerPixel = (tex->faces[0].levels && level < tex->num_levels &&
                                           tex->faces[0].levels[level].width > 0u &&
                                           tex->faces[0].levels[level].pitch > 0u)
                ? (uint64_t)(tex->faces[0].levels[level].pitch / tex->faces[0].levels[level].width)
                : mglTextureBytesPerPixelForFormat(tex->internalformat);
            if (cpuBytesPerPixel == 0u) {
                cpuBytesPerPixel = (uint64_t)sizeForInternalFormat(tex->internalformat, 0, 0);
            }
            if (cpuBytesPerPixel > 0u && cpuBytesPerPixel != dstBytesPerPixel) {
                uint64_t copyHeight = mglUpMax((uint64_t)height, 1UL);
                uint64_t copyDepth = mglUpMax((uint64_t)depth, 1UL);
                uint64_t dstRowBytes = (uint64_t)width * dstBytesPerPixel;
                uint64_t dstImageBytes = dstRowBytes * copyHeight;
                size_t sourceImagePitch = src_image_size;
                size_t minimumImagePitch = src_pitch * copyHeight;
                if (sourceImagePitch < minimumImagePitch) {
                    sourceImagePitch = minimumImagePitch;
                }
                size_t packedBytes = dstImageBytes * copyDepth;
                if (packedBytes != 0u && packedBytes <= (512u * 1024u * 1024u)) {
                    const uint8_t *sourceBase = (const uint8_t *)mglUpTextureBufferContents(buffer);
                    /* NSMutableData -> calloc/free (the tree's twin); the buffer
                     * travels as one pointer instead of a wrapper object. */
                    void *packedUpload = calloc(1u, (size_t)packedBytes);
                    if (packedUpload && sourceBase) {
                        uint8_t *packedBytesPtr = (uint8_t *)packedUpload;
                        bool expandOK = true;
                        for (uint64_t z = 0; z < copyDepth && expandOK; z++) {
                            size_t sliceBaseOff = src_offset + (size_t)z * sourceImagePitch;
                            size_t lastRowOff = sliceBaseOff + (size_t)(copyHeight - 1u) * src_pitch;
                            size_t rowBytesCpu = (uint64_t)width * cpuBytesPerPixel;
                            if (lastRowOff > src_size || rowBytesCpu > src_size - lastRowOff) {
                                expandOK = false;
                                break;
                            }
                            const uint8_t *sliceSrc = sourceBase + sliceBaseOff;
                            uint64_t expandedBPR = 0, expandedBPI = 0;
                            uint8_t *expanded = NULL;
                            if (needsRGBA8Expand) {
                                expanded = mglCreateRGBA8ExpandedUpload(tex,
                                                                        sliceSrc,
                                                                        width,
                                                                        copyHeight,
                                                                        src_pitch,
                                                                        &expandedBPR,
                                                                        &expandedBPI);
                            } else {
                                expanded = mglCreateChannelExpandedUpload(tex,
                                                                           dstPixelFormat,
                                                                           sliceSrc,
                                                                           width,
                                                                           copyHeight,
                                                                           src_pitch,
                                                                           &expandedBPR,
                                                                           &expandedBPI);
                            }
                            if (!expanded) {
                                expandOK = false;
                                break;
                            }
                            memcpy(packedBytesPtr + (z * dstImageBytes), expanded, expandedBPI);
                            mglUploadExpandScratchRelease(expanded);
                        }
                        if (expandOK) {
                            MGLTextureStagingOwner *stagingOwner = NULL;
                            void *uploadBuffer = NULL;
                            if (mglRenderCreateTextureStagingOwner(
                                    packedUpload, packedBytes,
                                    MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED,
                                    &stagingOwner, &uploadBuffer) == 0 &&
                                stagingOwner && uploadBuffer) {
                                bool uploaded = mglTextureEncodeBytesUpload(
                                    renderer, tex, uploadBuffer, 0, dstRowBytes,
                                    dstImageBytes, width, height, depth, slice,
                                    level, xoffset, yoffset, zoffset,
                                    "mtlTexSubImage");
                                /* The encoded upload retains its source until
                                 * command-buffer completion; return the
                                 * staging buffer instead of leaking the +1
                                 * reference this path used to drop. */
                                mglRenderDestroyTextureStagingOwner(&stagingOwner);
                                if (!uploaded) {
                                    fprintf(stderr, "MGL ERROR: mtlTexSubImage expanded PBO upload failed (tex=%u slice=%u level=%u)\n",
                                          tex->name, slice, level);
                                }
                                free(packedUpload);
                                return 0;
                            }
                        }
                    }
                    /* the .m's NSMutableData was released by ARC on every path;
                     * the calloc twin must free the fall-through as well. */
                    free(packedUpload);
                }
            }
        }
    }

    bool uploaded = mglTextureEncodeBytesUpload(
                                renderer, tex, buffer, src_offset, src_pitch, src_image_size, width, height, depth, slice, level, xoffset, yoffset, zoffset, "mtlTexSubImage");
    if (!uploaded) {
        fprintf(stderr, "MGL ERROR: mtlTexSubImage dedicated upload failed (tex=%u slice=%u level=%u)\n",
              tex->name, slice, level);
    }
}

/* -mtlTexSubImageBytes:… */
int mglTextureSubImageBytes(void *renderer, GLMContext glm_ctx,
                            Texture *tex, const void *bytes, uint64_t bytes_size,
                            uint64_t src_offset, uint64_t src_pitch,
                            uint64_t src_image_size, uint32_t slice, uint32_t level,
                            uint64_t width, uint64_t height, uint64_t depth,
                            uint64_t xoffset, uint64_t yoffset, uint64_t zoffset)
{
    if (!renderer) {
        return 0;
    }
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    GLMContext ctx = areas.ctx;
    void *device = mglRendererBackendGetDevice(areas.backend);
    (void)ctx;
    (void)device;
    (void)glm_ctx;
    if (!tex || !bytes || src_pitch == 0 || width == 0 || height == 0) {
        return false;
    }
    if (src_offset > bytes_size || level >= tex->num_levels) {
        return false;
    }

    uint64_t bytesPerPixel = mglTextureBytesPerPixelForFormat(tex->internalformat);
    if (bytesPerPixel == 0u &&
        tex->faces[0].levels &&
        tex->faces[0].levels[level].width > 0u) {
        TextureLevel *levelInfo = &tex->faces[0].levels[level];
        if (levelInfo->pitch > 0u &&
            (levelInfo->pitch % levelInfo->width) == 0u) {
            bytesPerPixel = (uint64_t)(levelInfo->pitch / levelInfo->width);
        }
    }
    if (bytesPerPixel == 0u) {
        return false;
    }

    uint64_t copyHeight = mglUpMax((uint64_t)height, 1UL);
    uint64_t copyDepth = mglUpMax((uint64_t)depth, 1UL);
    uint64_t rowBytes = (uint64_t)width * bytesPerPixel;
    if (rowBytes == 0u || rowBytes > src_pitch) {
        return false;
    }

    if (!mglTextureMtl(tex)) {
        return false;
    }

    /* Channel expansion: GL_RGB32* (12 bytes/pixel) -> Metal RGBA32* (16 bytes/pixel).
     * The CPU backing stores 3 channels per pixel, but the Metal texture expects
     * 4 channels. We must expand each pixel by inserting a default alpha before
     * uploading, otherwise the data layout mismatches and pixels shift. */
    void *dstTexture = mglTextureMtl(tex);
    uint32_t dstPixelFormat = mglPdTextureInfo(dstTexture).pixel_format;
    int needsChannelExpand = mglTextureNeedsChannelExpansion(tex->internalformat,
                                                              dstPixelFormat);
    uint64_t dstBytesPerPixel = bytesPerPixel;
    if (needsChannelExpand) {
        uint32_t rgbDst = 0u;
        if (mglRenderRGBExpandParams(dstPixelFormat, NULL, &rgbDst, NULL)) {
            dstBytesPerPixel = (uint64_t)rgbDst * 4u;
        } else {
            needsChannelExpand = 0;
        }
    }

    /* RGBA8 expansion: Metal has no RGB8 pixel format, so GL_RGB8-family
     * internal formats (3 bytes/pixel in the CPU backing store) are backed
     * by Metal RGBA8 variants (4 bytes/pixel).  Without per-pixel channel
     * expansion the 3-byte source is uploaded directly into a 4-byte Metal
     * texture, shifting pixels and producing vertical stripes.  Every other
     * upload path (createMTLTextureFromGLTexture, refreshMetalTextureCPUData,
     * mtlCopyImageSubData) expands via mglCreateRGBA8ExpandedUpload; the
     * direct mtlTexSubImageBytes path must do the same. */
    int needsRGBA8Expand = 0;
    if (!needsChannelExpand) {
        needsRGBA8Expand = mglTextureInternalFormatNeedsRGBA8Expansion(tex->internalformat,
                                                                        dstPixelFormat);
        if (needsRGBA8Expand) {
            dstBytesPerPixel = 4;
        }
    }

    size_t sourceImagePitch = src_image_size;
    size_t minimumImagePitch = src_pitch * copyHeight;
    if (sourceImagePitch < minimumImagePitch) {
        sourceImagePitch = minimumImagePitch;
    }

    uint64_t dstRowBytes = (uint64_t)width * dstBytesPerPixel;
    uint64_t dstImageBytes = dstRowBytes * copyHeight;
    size_t packedBytes = dstImageBytes * copyDepth;
    if (packedBytes == 0u || packedBytes > (512u * 1024u * 1024u)) {
        return false;
    }

    /* NSMutableData -> calloc/free (the tree's twin), keeping the .m's
     * "allocated for the whole method" lifetime: every early return frees it. */
    void *packedUpload = calloc(1u, (size_t)packedBytes);
    if (!packedUpload) {
        return false;
    }

    const uint8_t *sourceBase = (const uint8_t *)bytes;
    uint8_t *packedBytesPtr = (uint8_t *)packedUpload;

    if (needsChannelExpand) {
        uint32_t srcCompU = 0u, dstCompU = 0u;
        uint64_t alphaDefault = 0;
        if (!mglRenderRGBExpandParams(dstPixelFormat, &srcCompU, &dstCompU,
                                      &alphaDefault)) {
            free(packedUpload);
            return false;
        }
        uint64_t srcCompBytes = srcCompU;
        uint64_t dstCompBytes = dstCompU;
        uint64_t srcPixelBytes = srcCompBytes * 3;  /* 3 channels in source */
        uint64_t dstPixelBytes = dstCompBytes * 4;  /* 4 channels in destination */

        for (uint64_t z = 0; z < copyDepth; z++) {
            for (uint64_t y = 0; y < copyHeight; y++) {
                size_t srcRowOffset = src_offset + ((size_t)z * sourceImagePitch) + ((size_t)y * src_pitch);
                if (srcRowOffset > bytes_size || rowBytes > bytes_size - srcRowOffset) {
                    free(packedUpload);
                    return false;
                }
                const uint8_t *srcRow = sourceBase + srcRowOffset;
                uint8_t *dstRow = packedBytesPtr + (z * dstImageBytes) + (y * dstRowBytes);
                for (uint64_t x = 0; x < width; x++) {
                    const uint8_t *srcPixel = srcRow + x * srcPixelBytes;
                    uint8_t *dstPixel = dstRow + x * dstPixelBytes;
                    /* Copy 3 channels (R, G, B) */
                    memcpy(dstPixel, srcPixel, srcPixelBytes);
                    /* Set alpha channel to default value */
                    memcpy(dstPixel + srcPixelBytes, &alphaDefault, dstCompBytes);
                }
            }
        }
    } else if (needsRGBA8Expand) {

        for (uint64_t z = 0; z < copyDepth; z++) {
            size_t sliceBaseOff = src_offset + (size_t)z * sourceImagePitch;
            size_t lastRowOff = sliceBaseOff + (size_t)(copyHeight - 1u) * src_pitch;
            if (lastRowOff > bytes_size || rowBytes > bytes_size - lastRowOff) {
                free(packedUpload);
                return false;
            }
            const uint8_t *sliceSrc = sourceBase + sliceBaseOff;
            uint64_t expandedBPR = 0, expandedBPI = 0;
            uint8_t *expanded = mglCreateRGBA8ExpandedUpload(tex,
                                                              sliceSrc,
                                                              width,
                                                              copyHeight,
                                                              src_pitch,
                                                              &expandedBPR,
                                                              &expandedBPI);
            if (!expanded) {
                free(packedUpload);
                return false;
            }
            memcpy(packedBytesPtr + (z * dstImageBytes), expanded, expandedBPI);
            mglUploadExpandScratchRelease(expanded);
        }
    } else {
        /* No channel expansion needed - direct copy */
        for (uint64_t z = 0; z < copyDepth; z++) {
            for (uint64_t y = 0; y < copyHeight; y++) {
                size_t srcRowOffset = src_offset + ((size_t)z * sourceImagePitch) + ((size_t)y * src_pitch);
                if (srcRowOffset > bytes_size || rowBytes > bytes_size - srcRowOffset) {
                    static uint64_t s_subUploadRangeFailLogs = 0;
                    uint64_t hit = ++s_subUploadRangeFailLogs;
                    if (hit <= 32ull || (hit % 512ull) == 0ull) {
                        fprintf(stderr, "MGL TEXSUBIMAGE BYTES range fail tex=%u level=%u off=%zu rowBytes=%lu pitch=%zu image=%zu size=%zu z=%lu y=%lu hit=%llu\n",
                              (unsigned)tex->name,
                              (unsigned)level,
                              srcRowOffset,
                              (unsigned long)rowBytes,
                              src_pitch,
                              sourceImagePitch,
                              bytes_size,
                              (unsigned long)z,
                              (unsigned long)y,
                              (unsigned long long)hit);
                    }
                    free(packedUpload);
                    return false;
                }
                memcpy(packedBytesPtr + (z * dstImageBytes) + (y * dstRowBytes),
                       sourceBase + srcRowOffset,
                       rowBytes);
            }
        }
    }

    uint64_t metalSlice = slice;
    if (mglRenderTextureTargetIsArray((uint32_t)tex->target)) {
        metalSlice = zoffset;
    }

    bool uploaded = false;
    if (mglRenderPixelFormatIsPackedDepthStencil((uint32_t)dstPixelFormat)) {
        const int flipY = 0;
        uploaded = true;
        for (uint64_t z = 0; uploaded && z < copyDepth; ++z) {
            uploaded = mglUpUploadPackedDepthStencil(
                renderer, dstTexture, tex->internalformat,
                packedBytesPtr + z * dstImageBytes, dstRowBytes, width,
                copyHeight, level, metalSlice + z, xoffset, yoffset, flipY);
        }
    } else {
        MGLTextureStagingOwner *stagingOwner = NULL;
        void *uploadBuffer = NULL;
        if (mglRenderCreateTextureStagingOwner(
                packedBytesPtr, dstImageBytes * copyDepth,
                MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED, &stagingOwner,
                &uploadBuffer) == 0 &&
            stagingOwner && uploadBuffer) {
            /* The encoded upload retains its source until command-buffer
             * completion; return the staging buffer to the pool right after
             * encoding instead of leaking the +1 reference per upload. */
            uploaded = mglTextureEncodeBytesUpload(
                renderer, tex, uploadBuffer, 0, dstRowBytes, dstImageBytes,
                width, height, depth, slice, level, xoffset, yoffset, zoffset,
                "mtlTexSubImageBytes");
            mglRenderDestroyTextureStagingOwner(&stagingOwner);
        }
    }
    if (uploaded && mglRenderTargetHasGPUContent(
                        tex->is_render_target ? 1 : 0,
                        mglTextureWriteVersion(tex))) {
        /* Direct CPU→Metal refresh of an FBO-attached texture must invalidate
         * the Y-flip sampled copy.  textures.c also releases the copy, but
         * bumping write_version keeps any concurrent/lazy refresh coherent
         * with the post-upload Metal contents (KHR-GL46.texture_barrier).
         * Rebuild immediately so every MRT attachment has a fresh Y-flip
         * copy before the first feedback draw (color1+ previously rebuilt
         * only as a side-effect of color0's sample-gate repair). */
        mglUpMarkTextureLevelMetalFilled(tex, level, packedBytes);
        void *source = mglTextureMtl(tex);
        if (source) {
        }
    }
    free(packedUpload);
    return uploaded;
}

/* -mtlTexSubImage:… (METAL_LOCK is the thread-affinity assertion only, so the
 * C dispatcher links straight to the Locked entry). */
int mglTextureSubImageDispatch(void *renderer, GLMContext glm_ctx, Texture *tex,
                               Buffer *buf, uint64_t src_offset,
                               uint64_t src_pitch, uint64_t src_image_size,
                               uint64_t src_size, uint32_t slice, uint32_t level,
                               uint64_t width, uint64_t height, uint64_t depth,
                               uint64_t xoffset, uint64_t yoffset,
                               uint64_t zoffset)
{
    return mglTextureSubImage(renderer, glm_ctx, tex, buf, src_offset,
                              src_pitch, src_image_size, src_size, slice, level,
                              width, height, depth, xoffset, yoffset, zoffset);
}

/* -traceSampledTextureReadback:glTex:level:program:binding:stage:reason:hit:
 * (log 199).  The .m took two NSStrings; the C entry takes the C strings the
 * port already carried and prints them directly (its cold path used to pass the
 * NSString pointer to a %s conversion). */
void mglTextureTraceSampledReadback(void *renderer, void *texture, Texture *glTex,
                                    TextureLevel *level0, uint32_t program,
                                    uint32_t binding, const char *stage,
                                    const char *reason, uint64_t hit)
{
    if (!renderer) {
        return;
    }
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    void *device = mglRendererBackendGetDevice(areas.backend);
    void *commandQueue = mglRendererBackendGetCommandQueue(areas.backend);
    if (!texture || !device || !commandQueue) {
        return;
    }

    MGLRenderTextureInfo textureInfo = {0};
    if (mglRenderGetTextureInfo(texture,
                                   &textureInfo) != 0) {
        return;
    }
    uint32_t fmt = textureInfo.pixel_format;
    int fourByteColor =
        mglRenderPixelFormatIsUnorm8Color(fmt) != 0;
    if (!fourByteColor) {
        mglTraceLog("MGL TRACE sampled.readback skip program=%u binding=%u glTex=%u reason=%s fmt=%lu type=%lu size=%lux%lu hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              reason,
              (unsigned long)fmt,
              (unsigned long)textureInfo.texture_type,
              (unsigned long)textureInfo.width,
              (unsigned long)textureInfo.height,
              (unsigned long long)hit);
        return;
    }

    uint64_t texWidth = (uint64_t)textureInfo.width;
    uint64_t texHeight = (uint64_t)textureInfo.height;
    if (texWidth == 0 || texHeight == 0) {
        return;
    }

    uint64_t sampleWidth = mglUpMin(texWidth, 8u);
    uint64_t sampleHeight = mglUpMin(texHeight, 8u);
    uint64_t bytesPerPixel = 4u;
    uint64_t bytesPerRow = sampleWidth * bytesPerPixel;
    uint64_t byteCount = bytesPerRow * sampleHeight;
    if (byteCount == 0) {
        return;
    }

    void *readback = mglUpCreateBuffer(byteCount, MGL_PD_TEXTURE_RESOURCE_STORAGE_SHARED);
    void *cb = mglUpCreateCommandBuffer(commandQueue);
    void *blit = mglUpCreateBlitEncoder(cb);
    if (!readback || !cb || !blit) {
        mglTraceLog("MGL TRACE sampled.readback setup-fail program=%u binding=%u glTex=%u reason=%s readback=%p cb=%p blit=%p hit=%llu",
              (unsigned)program,
              (unsigned)binding,
              glTex ? (unsigned)glTex->name : 0u,
              reason,
              readback,
              cb,
              blit,
              (unsigned long long)hit);
        return;
    }

    mglUpCopyTextureToBuffer(
        blit, texture, 0, 0, mglTextureOrigin(0, 0, 0),
        mglTextureSize(sampleWidth, sampleHeight, 1), readback, 0,
        bytesPerRow, byteCount);
    mglUpEndBlitEncoder(blit);
    mglUpCommitCommandBuffer(cb);
    mglUpWaitCommandBuffer(cb);

    const uint8_t *p = (const uint8_t *)mglUpTextureBufferContents(readback);
    uint64_t byteSum = 0;
    uint64_t nonZeroBytes = 0;
    uint32_t firstPixel = 0;
    uint32_t pixelXor = 0;
    uint32_t minPixel = UINT32_MAX;
    uint32_t maxPixel = 0;
    uint64_t pixelCount = byteCount / sizeof(uint32_t);

    if (p) {
        for (uint64_t i = 0; i < byteCount; i++) {
            byteSum += (uint64_t)p[i];
            if (p[i] != 0) {
                nonZeroBytes++;
            }
        }
        if (byteCount >= sizeof(firstPixel)) {
            memcpy(&firstPixel, p, sizeof(firstPixel));
        }
        for (uint64_t i = 0; i < pixelCount; i++) {
            uint32_t pixel = 0;
            memcpy(&pixel, p + (i * sizeof(pixel)), sizeof(pixel));
            pixelXor ^= pixel;
            if (pixel < minPixel) {
                minPixel = pixel;
            }
            if (pixel > maxPixel) {
                maxPixel = pixel;
            }
        }
    }

    MGLRenderCommandBufferState sampledState = {0};
    (void)mglRenderGetCommandBufferState(
        cb, &sampledState);
    /* NSString stringWithFormat: -> a stack buffer (only used when the
     * command buffer reports an error). */
    char sampledError[256] = {0};
    if (sampledState.has_error) {
        snprintf(sampledError, sizeof(sampledError), "%s (domain=%s code=%lld)",
                 sampledState.error_description, sampledState.error_domain,
                 (long long)sampledState.error_code);
    }
    mglTraceLog("MGL TRACE sampled.readback stage=%s program=%u binding=%u glTex=%u reason=%s hit=%llu "
          "mtl=%p fmt=%lu type=%lu size=%lux%lu sample=%lux%lu status=%s error=%s "
          "nonZero=%lu/%lu sum=%llu first=0x%08x min=0x%08x max=0x%08x xor=0x%08x "
          "level(init ever=%u full=%u zero=%u source=%u upload=%lu src=%p hash=0x%016llx)",
          stage,
          (unsigned)program,
          (unsigned)binding,
          glTex ? (unsigned)glTex->name : 0u,
          reason,
          (unsigned long long)hit,
          texture,
          (unsigned long)fmt,
          (unsigned long)textureInfo.texture_type,
          (unsigned long)texWidth,
          (unsigned long)texHeight,
          (unsigned long)sampleWidth,
          (unsigned long)sampleHeight,
          mglCommandBufferStatusName(
              (uint32_t)sampledState.status),
          sampledError,
          (unsigned long)nonZeroBytes,
          (unsigned long)byteCount,
          (unsigned long long)byteSum,
          firstPixel,
          minPixel == UINT32_MAX ? 0u : minPixel,
          maxPixel,
          pixelXor,
          level0 ? (unsigned)level0->ever_written : 0u,
          level0 ? (unsigned)level0->has_initialized_data : 0u,
          level0 ? (unsigned)level0->suspicious_zero_upload : 0u,
          level0 ? (unsigned)level0->last_init_source : 0u,
          (unsigned long)(level0 ? level0->last_upload_size : 0u),
          level0 ? (void *)level0->last_src_ptr : NULL,
          (unsigned long long)(level0 ? level0->last_src_hash : 0ull));
}

/* @try of the fallback create: the guarded body publishes the +1 handle. */
typedef struct MglUpFallbackCtx_t {
    Texture *tex;
    void *out_texture;
} MglUpFallbackCtx;

static int mglUpFallbackBody(void *renderer, void *rawCtx);

/* -createFallbackMTLTexture: (log 199). */
void *mglTextureCreateFallback(void *renderer, Texture *tex)
{
    if (!renderer) {
        return NULL;
    }
    // Validate texture parameters before creating Metal texture to prevent Metal assertion failures
    if (!tex || tex->width <= 0 || tex->height <= 0 || tex->width > 32768 || tex->height > 32768) {
        MGL_TEX_UPLOAD_DIAG("MGL AGX: Skipping fallback texture creation - invalid dimensions %dx%d\n",
              tex ? tex->width : 0, tex ? tex->height : 0);
        return NULL;
    }

    MGL_TEX_UPLOAD_DIAG("MGL AGX: Creating emergency fallback texture (size: %dx%dx%d)\n", tex->width, tex->height, tex->depth);

    char fallbackFailure[256] = {0};
    MglUpFallbackCtx fallbackCtx = { tex, NULL };
    if (!mglPlatformShellGuardedCallCtxReason(
            renderer, "fallback texture creation", mglUpFallbackBody,
            &fallbackCtx, fallbackFailure, sizeof(fallbackFailure))) {
        MGL_TEX_UPLOAD_DIAG("MGL AGX: Even fallback texture creation failed: %s\n",
                fallbackFailure[0] ? fallbackFailure : "(null)");
        return NULL;
    }
    return fallbackCtx.out_texture;
}

static int mglUpFallbackBody(void *renderer, void *rawCtx)
{
    MglUpFallbackCtx *fallback = (MglUpFallbackCtx *)rawCtx;
    Texture *tex = fallback->tex;
    MGLRendererStateView areas;
    mglRendererFillStateView(renderer, &areas);
    void *device = mglRendererBackendGetDevice(areas.backend);
    (void)device;
    {
        uint32_t fallbackFormat = mglRenderFallbackPixelFormat(
            mtlPixelFormatForGLTex(tex), (uint32_t)tex->internalformat);

        int isDepthOrStencilFormat =
            mglRenderPixelFormatIsDepthOrStencil(fallbackFormat) != 0;

        MGLRenderTextureDescriptorState fallbackDesc = {
            .texture_type = MGLTextureType2D,
            .pixel_format = fallbackFormat,
            .width = mglUpMax(tex->width, 1), .height = mglUpMax(tex->height, 1),
            .depth = 1u, .mipmap_level_count = 1u,
            .sample_count = 1u, .array_length = 1u,
            .usage = MGL_TEXTURE_USAGE_SHADER_READ,
        };
        if (tex->is_render_target || isDepthOrStencilFormat) {
            fallbackDesc.usage |= MGL_TEXTURE_USAGE_RENDER_TARGET;
        }

        void *fallbackTexture =
            mglUpCreateTexture(&fallbackDesc, NULL);

        if (fallbackTexture) {
            // Fill with simple gradient pattern using a simple approach
            uint64_t width = mglPdTextureInfo(fallbackTexture).width;
            uint64_t height = mglPdTextureInfo(fallbackTexture).height;

            if (!isDepthOrStencilFormat && width <= 512 && height <= 512) {
                uint32_t *gradientData = calloc(width * height, sizeof(uint32_t));
                if (gradientData) {
                    // Create simple red-blue gradient
                    for (uint64_t y = 0; y < height; y++) {
                        for (uint64_t x = 0; x < width; x++) {
                            uint64_t index = y * width + x;
                            uint8_t r = (uint8_t)((x * 255) / width);
                            uint8_t g = 128;
                            uint8_t b = (uint8_t)((y * 255) / height);
                            uint8_t a = 255;
                            gradientData[index] = ((uint32_t)a << 24) | ((uint32_t)b << 16) | ((uint32_t)g << 8) | (uint32_t)r;
                        }
                    }

                    MGLRegionValue region = mglTextureRegion2D(0, 0, width, height);
                    (void)mglTextureReplaceRegionValue(
                        fallbackTexture, region, 0, 0, gradientData,
                        width * sizeof(uint32_t), 0, 0);

                    free(gradientData);
                    MGL_TEX_UPLOAD_DIAG("MGL AGX: Fallback color texture created with gradient pattern\n");
                }
            }
        }

        fallback->out_texture = fallbackTexture;
        return 1;
    }
}
