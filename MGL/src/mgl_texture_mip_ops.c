/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_mip_ops.c - the mipmap / image-unit / texbuffer leaves of
 * MGLRenderer+Texture.m (P0-1, log 194), moved verbatim.  Two Objective-C
 * idioms needed the tree's established C twins:
 *   * the @try/@catch around the blit and getBytes calls become the shell's
 *     guarded call with an exception-reason buffer (rule 58 (b)); the mipmap
 *     path's synthetic -[NSException raise:] becomes a failure string that
 *     reproduces the same "name: reason" text;
 *   * NSMutableData -> calloc/free, NSMutableString + appendFormat: -> the
 *     MglTxText growable buffer (same shape as mgl_render_pass_manager_ops.c's
 *     MglPdSource).
 */

#include "mgl_texture_mip_ops.h"

#include "mgl_frame_activity.h"
#include "mgl_gpu_recovery.h"          /* mglPlatformShellGuardedCallCtxReason */
#include "mgl_readback_policy.h"
#include "mgl_region_value.h"
#include "mgl_render_pass_manager_ops.h"
#include "mgl_renderer_ports.h"
#include "mgl_renderer_backend.h"
#include "mgl_texture_bind.h"          /* mglRendererBindMTLTexture */
#include "mgl_texture_compat.h"     /* mglTextureLevelHasUploadableCPUData */
#include "mgl_trace_log.h"
#include "mgl_types_state.h"

#include "mgl_render.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Declared in Objective-C headers a .c file cannot include. */
extern GLuint mglRendererSafeFramebufferName(GLMContext ctx);
extern uint32_t sizeForInternalFormat(GLenum internalformat, GLsizei width,
                                      GLsizei height);
extern uint64_t mglTextureBytesPerPixelForFormat(GLenum internalformat);
extern int mglCheckErrorsC(GLMContext ctx, const char *function, int line);
extern void mglPlatformShellSetContext(void *renderer, GLMContext glm_ctx);
extern int mglRenderPassProcessGLState(void *renderer, int draw_command);
extern void mglRenderPassFlushCommandBuffer(void *renderer, int finish);
extern int mglRenderPassNewCommandBufferLocked(void *renderer);

/* The .m's mglTextureCreateCurrentBlitEncoder / ...EndBlitEncoder /
 * mglTextureCreateTexture / mglTextureGetBytes, in C (same shapes as the
 * mgl_texture_create_ops.c twins). */
/* The .m raised NSException here; the C twin hands the same text to the caller
 * through mglTxGetBytes' failure path below. */
static char g_mglTxGetBytesError[128];

static const char *mglTxTakeGetBytesError(void)
{
    return g_mglTxGetBytesError;
}

static void mglTxRaiseGetBytesError(uint64_t level, uint64_t slice)
{
    snprintf(g_mglTxGetBytesError, sizeof(g_mglTxGetBytesError),
             "MGLTextureGetBytesError: C++ texture getBytes failed (level=%lu slice=%lu)",
             (unsigned long)level, (unsigned long)slice);
}

static void *mglTxCreateCurrentBlitEncoder(void *commandBufferOwner)
{
    return mglRenderCreateBlitEncoderBorrowed(commandBufferOwner);
}

static void mglTxEndBlitEncoder(void *encoder)
{
    if (!encoder) return;
    (void)mglRenderEndBlitEncoder(encoder);
}

static void *mglTxCreateTexture(
    const MGLRenderTextureDescriptorState *descriptor)
{
    void *texture = NULL;
    if (mglRenderCreateTextureFromState(descriptor, NULL, &texture) == 0 &&
        texture) {
        return texture;
    }
    return NULL;
}

static void mglTxGetBytes(void *texture, void *bytes, uint64_t bytesPerRow,
                          uint64_t bytesPerImage, MGLRegionValue region,
                          uint64_t level, uint64_t slice, int useSlice)
{
    if (mglRenderTextureGetBytes(
            texture, bytes, bytesPerRow, bytesPerImage, region.origin.x,
            region.origin.y, region.origin.z, region.size.width,
            region.size.height, region.size.depth, level, slice,
            useSlice ? 1 : 0) != 0) {
        /* The .m raised here so its caller's @try could report it; the C
         * twin reports through the same guarded-call reason buffer. */
        mglTxRaiseGetBytesError(level, slice);
    }
}

/* The .m's mglTextureInfo(), in C. */
static MGLRenderTextureInfo mglTxTextureInfo(void *texture)
{
    MGLRenderTextureInfo info = {0};
    if (texture) {
        (void)mglRenderGetTextureInfo(texture, &info);
    }
    return info;
}

/* MGL_TEXTURE_USAGE_* live in an Objective-C header; the values are the
 * Metal-cpp option bits the descriptor builder consumes. */
#define kMglTxUsageShaderRead 1u
#define kMglTxUsageShaderWrite 2u
#define kMglTxUsagePixelFormatView 16u

/* NSMutableString + appendString:/appendFormat: twin. */
typedef struct MglTxText_t {
    char *data;
    size_t len;
    size_t cap;
} MglTxText;

static void mglTxTextInit(MglTxText *text)
{
    text->cap = 256u;
    text->len = 0u;
    text->data = (char *)malloc(text->cap);
    if (text->data) text->data[0] = '\0';
}

static void mglTxTextAppend(MglTxText *text, const char *raw)
{
    if (!text->data || !raw) return;
    const size_t n = strlen(raw);
    if (text->len + n + 1u > text->cap) {
        size_t cap = text->cap;
        while (cap < text->len + n + 1u) cap *= 2u;
        char *grown = (char *)realloc(text->data, cap);
        if (!grown) return;
        text->data = grown;
        text->cap = cap;
    }
    memcpy(text->data + text->len, raw, n);
    text->len += n;
    text->data[text->len] = '\0';
}

static void mglTxTextAppendF(MglTxText *text, const char *fmt, ...)
{
    if (!text->data) return;
    va_list ap;
    va_start(ap, fmt);
    char stack[256];
    va_list ap2;
    va_copy(ap2, ap);
    const int need = vsnprintf(stack, sizeof(stack), fmt, ap2);
    va_end(ap2);
    if (need > 0) {
        if ((size_t)need < sizeof(stack)) {
            mglTxTextAppend(text, stack);
        } else {
            char *heap = (char *)malloc((size_t)need + 1u);
            if (heap) {
                (void)vsnprintf(heap, (size_t)need + 1u, fmt, ap);
                mglTxTextAppend(text, heap);
                free(heap);
            }
        }
    }
    va_end(ap);
}

/* @try of the mipmap blit-encoder end. */
typedef struct MglTxEndBlitCtx_t {
    void *encoder;
} MglTxEndBlitCtx;

static int mglTxEndBlitBody(void *renderer, void *rawCtx)
{
    MglTxEndBlitCtx *ctx = (MglTxEndBlitCtx *)rawCtx;
    mglTxEndBlitEncoder(ctx->encoder);
    (void)renderer;
    return 1;
}

/* @try of the texbuffer getBytes read. */
typedef struct MglTxGetBytesCtx_t {
    void *texture;
    void *out_bytes;
    uint64_t bytes_per_row;
    uint32_t width;
    uint32_t height;
    char *failure_out;
    size_t failure_capacity;
} MglTxGetBytesCtx;

static int mglTxGetBytesBody(void *renderer, void *rawCtx)
{
    MglTxGetBytesCtx *ctx = (MglTxGetBytesCtx *)rawCtx;
    mglTxGetBytes(ctx->texture, ctx->out_bytes, ctx->bytes_per_row, 0,
                       mglRegion2D(0, 0, ctx->width, ctx->height), 0, 0,
                       0);
    (void)renderer;
    return 1;
}

/* -mtlGenerateMipmaps:forTexture: */
void mglTextureGenerateMipmaps(void *renderer, GLMContext glm_ctx,
                               struct Texture_t *tex)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *commandState = areas.command;
    mglPlatformShellSetContext(renderer, glm_ctx);

    if (!tex) {
        fprintf(stderr, "MGL ERROR: mtlGenerateMipmaps called with NULL texture\n");
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    RETURN_ON_FAILURE(mglRenderPassProcessGLState(renderer, 0));

    // end encoding on current render encoder
    mglRendererEndRenderEncodingLocked(renderer);

    RETURN_ON_FAILURE(mglRenderPassEnsureWritableCommandBufferLocked(
        renderer, "mtlGenerateMipmaps"));

    // no failure path..?
    RETURN_ON_FAILURE(mglRendererBindMTLTexture(renderer, tex));

    void *texture;

    texture = tex->mtl_data;
    if (!texture) {
        fprintf(stderr, "MGL ERROR: mtlGenerateMipmaps texture %u has no Metal texture after bind\n", tex->name);
        mglDispatchError(glm_ctx, __FUNCTION__, (GLenum)mglRenderErrorInvalidOperation());
        return;
    }

    if (mglTxTextureInfo(texture).mipmap_level_count <= 1u) {
        return;
    }

    // start blit encoder
    void *blitCommandEncoder;
    blitCommandEncoder = mglTxCreateCurrentBlitEncoder(
        commandState->currentCommandBufferOwner);
    if (!blitCommandEncoder) {
        fprintf(stderr, "MGL ERROR: Failed to create blit encoder for mipmap generation\n");
        return;
    }

    /* The .m raised a synthetic NSException on a failed generate and ended the
     * blit encoder inside the same @try; C records the failure text and runs
     * the same catch logic, replaying both log lines verbatim (rule 58 (b)).
     * The encoder end is guarded because Metal can throw from it. */
    char failure[256] = {0};
    MglTxEndBlitCtx endCtx = { blitCommandEncoder };
    if (mglRenderBlitGenerateMipmaps(blitCommandEncoder, texture) != 0) {
        snprintf(failure, sizeof(failure),
                 "MGLGenerateMipmapsError: C++ mipmap generation failed for texture %u",
                 tex->name);
    } else if (!mglPlatformShellGuardedCallCtxReason(
                   renderer, "mipmap blit encoder end", mglTxEndBlitBody,
                   &endCtx, failure, sizeof(failure))) {
        /* thrown by the end call itself; `failure` holds the reason */
    }
    if (failure[0] != '\0') {
        fprintf(stderr,
                "MGL ERROR: generateMipmapsForTexture failed for texture %u: %s\n",
                tex->name, failure);
        char endFailure[256] = {0};
        if (!mglPlatformShellGuardedCallCtxReason(
                renderer, "mipmap blit encoder end cleanup", mglTxEndBlitBody,
                &endCtx, endFailure, sizeof(endFailure))) {
            fprintf(stderr,
                    "MGL WARNING: failed to end mipmap blit encoder after exception: %s\n",
                    endFailure[0] ? endFailure : "(null)");
        }
        mglDispatchError(glm_ctx, __FUNCTION__,
                         (GLenum)mglRenderErrorInvalidOperation());
    }
}

/* -flushImageUnitSlice:unit: */
void mglTextureFlushImageUnitSlice(void *renderer, GLMContext glm_ctx,
                                   uint32_t unit)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *commandState = areas.command;
    if (!glm_ctx || unit >= glm_ctx->active_state->var.max_image_units ||
        unit >= TEXTURE_UNITS) {
        return;
    }
    ImageUnit *iu = &glm_ctx->active_state->image_units[unit];
    if (!mglRenderImageUnitSliceNeedsFlush(
            iu->tex ? 1 : 0, iu->mtl_image_view ? 1 : 0, iu->layered ? 1 : 0,
            iu->tex ? (uint32_t)iu->tex->target : 0u,
            (uint32_t)iu->access)) {
        return;
    }
    if (!mglRendererBindMTLTexture(renderer, iu->tex) || !iu->tex->mtl_data) {
        return;
    }
    void *dst3d = iu->tex->mtl_data;
    void *staging = iu->mtl_image_view;
    MGLRenderTextureInfo info = mglTxTextureInfo(dst3d);
    if (info.texture_type != MGLTextureType3D || info.width == 0u) {
        return;
    }
    const uint64_t level = (uint64_t)iu->level;
    const uint64_t layer = (uint64_t)iu->layer;
    if (level >= info.mipmap_level_count || layer >= info.depth) {
        return;
    }

    mglRendererEndRenderEncodingLocked(renderer);
    if (!commandState->currentCommandBufferOwner &&
        !mglRenderPassNewCommandBufferLocked(renderer)) {
        return;
    }
    void *blit = mglRenderCreateBlitEncoderBorrowed(
        commandState->currentCommandBufferOwner);
    if (!blit) {
        return;
    }
    (void)mglRenderBlitCopyTexture(
        blit, staging, 0u, 0u, 0u, 0u, 0u,
        info.width, info.height, 1u,
        dst3d, 0u, level, 0u, 0u, layer);
    (void)mglRenderEndBlitEncoder(blit);
    mglRenderPassFlushCommandBuffer(renderer, 0);
}

/* -prepareImageUnitSlice:unit: */
void mglTexturePrepareImageUnitSlice(void *renderer, GLMContext glm_ctx,
                                     uint32_t unit)
{
    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);
    MGLCommandState *commandState = areas.command;
    if (!glm_ctx || unit >= glm_ctx->active_state->var.max_image_units ||
        unit >= TEXTURE_UNITS) {
        return;
    }
    ImageUnit *iu = &glm_ctx->active_state->image_units[unit];
    if (!iu->tex || iu->layered ||
        !mglRenderTextureTargetIs3D((uint32_t)iu->tex->target)) {
        return;
    }
    if (!mglRendererBindMTLTexture(renderer, iu->tex) || !iu->tex->mtl_data) {
        return;
    }
    void *src3d = iu->tex->mtl_data;
    MGLRenderTextureInfo info = mglTxTextureInfo(src3d);
    if (info.texture_type != MGLTextureType3D || info.width == 0u) {
        return;
    }
    const uint64_t level = (uint64_t)iu->level;
    const uint64_t layer = (uint64_t)iu->layer;
    if (level >= info.mipmap_level_count || layer >= info.depth) {
        return;
    }

    if (iu->mtl_image_view) {
        mglTextureFlushImageUnitSlice(renderer, glm_ctx, unit);
        mglRenderReleaseMetalObject(iu->mtl_image_view);
        iu->mtl_image_view = NULL;
    }

    MGLRenderTextureDescriptorState desc = {
        .texture_type = MGLTextureType2D,
        .pixel_format = info.pixel_format,
        .width = info.width,
        .height = info.height,
        .depth = 1u,
        .mipmap_level_count = 1u,
        .sample_count = 1u,
        .array_length = 1u,
        .usage = kMglTxUsageShaderRead | kMglTxUsageShaderWrite |
                 kMglTxUsagePixelFormatView,
        .storage_mode = info.storage_mode,
    };
    void *staging = mglTxCreateTexture(&desc);
    if (!staging) {
        return;
    }

    mglRendererEndRenderEncodingLocked(renderer);
    if (!commandState->currentCommandBufferOwner &&
        !mglRenderPassNewCommandBufferLocked(renderer)) {
        return;
    }
    void *blit = mglRenderCreateBlitEncoderBorrowed(
        commandState->currentCommandBufferOwner);
    if (!blit) {
        return;
    }
    (void)mglRenderBlitCopyTexture(
        blit, src3d, 0u, level, 0u, 0u, layer,
        info.width, info.height, 1u,
        staging, 0u, 0u, 0u, 0u, 0u);
    (void)mglRenderEndBlitEncoder(blit);
    mglRenderPassFlushCommandBuffer(renderer, 0);

    iu->mtl_image_view = staging;
}

/* -syncTextureBufferFromImage:tex: */
void mglTextureSyncBufferFromImage(void *renderer, GLMContext glm_ctx,
                                  struct Texture_t *tex)
{
    if (!glm_ctx || !tex ||
        !mglRenderTextureTargetIsBuffer((uint32_t)tex->target) ||
        !tex->mtl_data || !tex->texture_buffer || tex->texture_buffer_size <= 0) {
        return;
    }

    Buffer *sourceBuffer = tex->texture_buffer;
    void *texture = tex->mtl_data;
    MGLRenderTextureInfo info = mglTxTextureInfo(texture);
    if (info.width == 0u || info.height == 0u) {
        return;
    }

    uint64_t bytesPerTexel = mglTextureBytesPerPixelForFormat(tex->internalformat);
    if (bytesPerTexel == 0u) {
        bytesPerTexel = (uint64_t)sizeForInternalFormat(tex->internalformat, 0, 0);
    }
    if (bytesPerTexel == 0u) {
        return;
    }

    uint64_t bytesPerRow = (uint64_t)info.width * bytesPerTexel;
    uint64_t packedBytes = bytesPerRow * (uint64_t)info.height;
    if (packedBytes == 0u ||
        (size_t)tex->texture_buffer_size > packedBytes) {
        return;
    }

    /* NSMutableData -> calloc/free (the tree's established twin). */
    void *packedData = calloc(1u, (size_t)packedBytes);
    if (!packedData) {
        return;
    }

    char failure[256] = {0};
    MglTxGetBytesCtx getCtx = { texture, packedData, (uint64_t)bytesPerRow,
                                info.width, info.height, failure,
                                sizeof(failure) };
    g_mglTxGetBytesError[0] = '\0';
    if (!mglPlatformShellGuardedCallCtxReason(renderer, "texbuffer getBytes",
                                              mglTxGetBytesBody, &getCtx,
                                              failure, sizeof(failure))) {
        fprintf(stderr,
                "MGL TEXBUFFER SYNC ERROR: getBytes failed tex=%u buffer=%u: %s\n",
                tex->name, sourceBuffer->name,
                failure[0] ? failure : "(null)");
        free(packedData);
        return;
    }

    mglRendererBufferSubData(glm_ctx, sourceBuffer,
                             (size_t)tex->texture_buffer_offset,
                             (size_t)tex->texture_buffer_size, packedData);
    free(packedData);
}

/* -logMTLTextureMipDiagnostics:metal:effectiveMipLevels: */
void mglTextureLogMipDiagnostics(void *renderer, struct Texture_t *tex,
                                 void *texture, uint32_t effective_mip_levels)
{
    (void)renderer;
    static uint64_t s_mipDiagLogs = 0;
    uint64_t diagHit = ++s_mipDiagLogs;
    if (kMGLDiagnosticStateLogs &&
        (diagHit <= 128ull || (diagHit % 512ull) == 0ull)) {
        uint64_t mtlMipCount = mglTxTextureInfo(texture).mipmap_level_count;
        uint32_t mtlFmt = mglTxTextureInfo(texture).pixel_format;
        uint32_t mtlStorage = mglTxTextureInfo(texture).storage_mode;
        uint64_t uploadedLevels = 0;
        uint64_t skippedLevels = 0;
        uint64_t skippedSourceNone = 0;
        uint64_t skippedNoData = 0;
        MglTxText levelSummary;
        mglTxTextInit(&levelSummary);
        uint64_t levelsToSummarize = tex->num_levels < 16u ? (uint64_t)tex->num_levels : 16u;
        for (uint64_t lvl = 0; lvl < levelsToSummarize; lvl++) {
            TextureLevel *tl = (tex->faces[0].levels && lvl < tex->num_levels)
                ? &tex->faces[0].levels[lvl] : NULL;
            if (!tl) { mglTxTextAppend(&levelSummary, "-"); continue; }
            bool uploadable = mglTextureLevelHasUploadableCPUData(tl);
            if (uploadable) uploadedLevels++; else skippedLevels++;
            if (!uploadable) {
                if (tl->last_init_source == kTexImageNull || tl->last_init_source == kTexInitNone)
                    skippedSourceNone++;
                if (!tl->has_initialized_data && !tl->ever_written)
                    skippedNoData++;
            }
            mglTxTextAppendF(&levelSummary, "[%u:s%u:w%u:e%u:i%u]",
                             (unsigned)lvl, (unsigned)tl->last_init_source,
                             (unsigned)tl->width, (unsigned)tl->ever_written,
                             (unsigned)tl->has_initialized_data);
        }
        mglTraceLog("MGL TEX_MIP_DIAG tex=%u target=0x%x dims=%ux%u internal=0x%x "
                      "numLevels=%u mipmapLevels=%u effectiveMipLevels=%u mtlMipCount=%lu "
                      "mtlFmt=%lu mtlStorage=%ld mipmapped=%d baseLevel=%u maxLevel=%u "
                      "uploadedLevels=%lu skippedLevels=%lu skippedSourceNone=%lu skippedNoData=%lu "
                      "levels=%s hit=%llu",
                      (unsigned)tex->name, (unsigned)tex->target,
                      (unsigned)tex->width, (unsigned)tex->height,
                      (unsigned)tex->internalformat,
                      (unsigned)tex->num_levels, (unsigned)tex->mipmap_levels,
                      (unsigned)effective_mip_levels, (unsigned long)mtlMipCount,
                      (unsigned long)mtlFmt, (long)mtlStorage, (int)(tex->mipmapped ? 1 : 0),
                      (unsigned)tex->params.base_level, (unsigned)tex->params.max_level,
                      (unsigned long)uploadedLevels, (unsigned long)skippedLevels,
                      (unsigned long)skippedSourceNone, (unsigned long)skippedNoData,
                      levelSummary.data ? levelSummary.data : "",
                      (unsigned long long)diagHit);
    }
}
