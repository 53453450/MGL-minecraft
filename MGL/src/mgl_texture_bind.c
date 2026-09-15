/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_bind.c — -[MGLRenderer bindMTLTextureLocked:] moved out of
 * MGLRenderer+Binding.m (P0-1, log 100).
 *
 * The method was 339 lines of struct-field arithmetic with exactly four
 * Objective-C steps in it; those are ports now:
 *   -createMTLTextureFromGLTexture:      mglRendererCreateMTLTextureFromGLTexturePort
 *   -createFallbackMTLTexture:           mglRendererCreateFallbackMTLTexturePort
 *   -uploadFullCPUTextureDataIntoTexture:metal:reason:
 *                                        mglRendererUploadFullCPUTextureDataPort
 *   -uploadDirtyCPUTextureData:...       mglRendererUploadDirtyCPUTextureDataPort
 * Everything else (dirty bits, render-target usage/mip reconciliation, the GPU
 * preservation blit, the sampler cascade, the AGX fallback circuit breaker and
 * the MIP_DIAG state signature) reads C storage directly.
 *
 * Ownership notes, because ARC is not available here:
 *   - both creation ports return +1 (the shell bridges the method result), so
 *     the value stored in tex->mtl_data / tex->params.mtl_data owns exactly one
 *     reference and mglSafeReleaseMetalObj balances it;
 *   - the old texture kept alive across the preservation blit is a plain
 *     CFRetain / CFRelease pair, NOT mglSafeReleaseMetalObj: it was already
 *     counted when it was created, and counting a second release would skew
 *     mglMetalGetCreated/Released.
 */

#include <stddef.h>              /* size_t, for mgl_types_buffer.h (no stddef of its own) */
#include <stdio.h>

#include "glm_context.h"         /* GLMContext, before the type headers that need it */
#include "mgl_texture_upload_ops.h"
#include "mgl_texture_bind.h"
#include "mgl_renderer_ports.h"  /* state areas + creation/upload ports */
#include "mgl_render.h"          /* mglRender* texture predicates and queries */
#include "mgl_render_pass_manager_ops.h" /* mglRendererEndRenderEncodingLocked */
#include "mgl_texture_compat.h"  /* sampled-copy release, swizzle predicate */
#include "mgl_texture_sampler.h" /* mglTextureCreateSamplerForTexParam */
#include "mgl_metal_ref.h"       /* mglSafeReleaseMetalObj */
#include "mgl_env_flag.h"        /* mgl_env_flag_enabled */
#include "mgl_state_log.h"       /* mglMipDiag* */
#include "mgl_trace_log.h"       /* mglTraceLog / kMGLDiagnosticStateLogs */

#include <CoreFoundation/CoreFoundation.h>  /* CFRetain / CFRelease for the aliases */
#include <time.h>                /* clock_gettime for the fallback circuit breaker */

/* Wall clock in UNIX seconds: the C twin of [[NSDate date]
 * timeIntervalSince1970], which gates the fallback-texture circuit breaker. */
static double mglTextureBindNowSeconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

/* +1 default sampler, the twin of the file-local mglBindingCreateDefaultSampler
 * of MGLRenderer+Binding.m (and of mgl_texture_sampler.c's private helper). */
static void *mglTextureBindCreateDefaultSampler(void)
{
    void *sampler = NULL;
    if (mglRenderCreateDefaultSampler(&sampler) == 0 && sampler) {
        return sampler;
    }
    return NULL;
}

/* Temporary +1 alias of a Metal object that is already accounted for: no
 * MGLMetalKind create/release counters, unlike mglSafeReleaseMetalObj. */
static void *mglTextureBindRetainAlias(void *object)
{
    return object ? (void *)CFRetain((CFTypeRef)object) : NULL;
}

static void mglTextureBindReleaseAlias(void *object)
{
    if (object) {
        CFRelease((CFTypeRef)object);
    }
}

/* Minimum of the Metal level count and the populated GL level count, kept as an
 * explicit comparison (utils.h's MIN is unparenthesized, so `(GLuint)MIN(a, b)`
 * casts the comparison result rather than the chosen value). */
static uint32_t mglTextureBindUploadLevelCount(uint64_t metal_levels,
                                               uint32_t gl_levels)
{
    const uint64_t wanted = gl_levels ? (uint64_t)gl_levels : 1u;
    return (uint32_t)(metal_levels < wanted ? metal_levels : wanted);
}

bool mglRendererBindMTLTexture(void *renderer, Texture *tex)
{
    if (!renderer || !tex) {
        return false;
    }

    MGLRendererStateAreas areas;
    mglRendererStateAreasPort(renderer, &areas);

    if (mglRenderTextureBufferNeedsDirty(
            mglRenderIsTextureBufferTarget(tex->target),
            tex->texture_buffer ? 1 : 0,
            tex->texture_buffer ? tex->texture_buffer->data.dirty_bits : 0u)) {
        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
    }

    if (tex->mtl_data &&
        mglRenderTextureNeedsArrayLengthCheck((uint32_t)tex->target)) {
        MGLRenderTextureInfo existingInfo = {0};
        if (mglRenderGetTextureInfo(tex->mtl_data, &existingInfo) == 0) {
            /* Metal cube-array arrayLength is cube count; GL depth is usually
             * face count (cubes * 6). Comparing raw depth forced a rebuild that
             * wiped imageStore results on the next bind. */
            uint64_t expectedLayers = mglRenderExpectedArrayLayers(
                tex->target, tex->depth);
            if (existingInfo.array_length < expectedLayers ||
                existingInfo.width != (uint64_t)tex->width ||
                existingInfo.height != (uint64_t)tex->height) {
                tex->dirty_bits |= DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_DATA;
            }
        }
    }

    /* If this texture is now used as a render target but was previously created
     * without render-target usage, force a recreate with proper usage flags.
     * When the old texture already has GPU-written data (e.g. from imageStore
     * in a compute shader), preserve it via a GPU-to-GPU blit instead of
     * re-uploading potentially stale CPU data. */
    if (tex->mtl_data && tex->is_render_target) {
        void *existingTexture = tex->mtl_data;
        MGLRenderTextureInfo existingInfo = {0};
        bool hasExistingInfo = existingTexture &&
            mglRenderGetTextureInfo(existingTexture, &existingInfo) == 0;
        if (existingTexture && !hasExistingInfo) {
            fprintf(stderr,
                    "MGL ERROR: Failed to query texture %u metadata before render-target transition\n",
                    tex->name);
            return false;
        }
        const uint64_t requiredRenderTargetUsage = (1ull << 2) | (1ull << 0);
        /* Prefer populated num_levels over allocation capacity (mipmap_levels).
         * Capacity-only sizing recreated MC atlases as 11-level Metal textures
         * when only 2 GL levels were uploaded. */
        uint64_t requiredMipLevels =
            (mglRenderTargetIsRenderbuffer((uint32_t)tex->target) || tex->samples > 1u)
                ? 1u
                : (tex->num_levels > 1u
                       ? (uint64_t)tex->num_levels
                       : ((tex->mipmap_levels > 1u) ? (uint64_t)tex->mipmap_levels : 1u));
        bool usageMismatch = hasExistingInfo &&
            ((existingInfo.usage & requiredRenderTargetUsage) != requiredRenderTargetUsage);
        /* Grow when RT needs more mips; also shrink when populated num_levels
         * is known and smaller than an over-allocated Metal chain (atlas case). */
        bool mipCountMismatch = hasExistingInfo &&
            (requiredMipLevels > existingInfo.mipmap_level_count ||
             (tex->num_levels > 1u &&
              requiredMipLevels < existingInfo.mipmap_level_count));
        if (existingTexture && (usageMismatch || mipCountMismatch)) {
            fprintf(stderr,
                    "MGL WARNING: Recreating texture %u for render-target use (old usage=0x%lx oldMips=%lu requiredMips=%lu)\n",
                    tex->name,
                    (unsigned long)existingInfo.usage,
                    (unsigned long)existingInfo.mipmap_level_count,
                    (unsigned long)requiredMipLevels);

            /* Keep the old texture alive so its GPU data can be blitted to the
             * new one after tex->mtl_data is released. */
            void *oldTexture = mglTextureBindRetainAlias(existingTexture);

            mglSafeReleaseMetalObj((void **)&tex->mtl_data);
            mglTextureReleaseGLSampledCopy(tex);

            /* Create a new texture with correct usage.  Don't set
             * DIRTY_TEXTURE_DATA so that createMTLTextureFromGLTexture
             * skips CPU data upload — we'll blit GPU data instead. */
            void *newTexture =
                mglTextureCreateFromGLTexture(renderer, tex);
            MGLRenderTextureInfo newInfo = {0};
            bool dimensionsMatch = newTexture &&
                mglRenderGetTextureInfo(newTexture, &newInfo) == 0 &&
                newInfo.width == existingInfo.width &&
                newInfo.height == existingInfo.height &&
                newInfo.depth == existingInfo.depth;
            if (oldTexture && dimensionsMatch) {
                /* The creation port's +1 moves into the texture slot. */
                tex->mtl_data = newTexture;
                const void *metalTexture = newTexture;
                newTexture = NULL;
                const bool packedDepthStencil =
                    mglRenderPackedDepthStencilFormat(
                        (uint32_t)tex->internalformat) != 0;
                if (packedDepthStencil) {
                    const bool isArray =
                        mglRenderTextureTargetIsLayeredUpload(
                            (uint32_t)tex->target) != 0;
                    int allLevelsUploaded = 1;
                    const uint32_t levelCount = mglTextureBindUploadLevelCount(
                        newInfo.mipmap_level_count,
                        tex->num_levels ? tex->num_levels : 1u);
                    if (mglTextureUploadDirty(
                            renderer, tex, (void *)metalTexture,
                            (uint32_t)newInfo.pixel_format, 1, levelCount,
                            isArray ? 1 : 0,
                            mglRenderTextureTargetIs1D((uint32_t)tex->target) ? 1 : 0,
                            mglRenderTextureTargetIs1DArray((uint32_t)tex->target) ? 1 : 0,
                            (uint32_t)newInfo.texture_type,
                            &allLevelsUploaded) && allLevelsUploaded) {
                        tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
                    } else {
                        tex->dirty_bits |= DIRTY_TEXTURE_DATA;
                    }
                } else {
                    /* Blit GPU data from old texture to new texture to preserve
                     * any writes (e.g. imageStore) that occurred before the
                     * is_render_target transition. */
                    mglRendererEndRenderEncodingLocked(renderer);
                    if (mglRenderPassEnsureWritableCommandBufferLocked(
                            renderer, "is_render_target_blit")) {
                        void *owner = areas.command
                                          ? areas.command->currentCommandBufferOwner
                                          : NULL;
                        if (mglRenderCopyMatchingTextureSubresourcesForCommandBufferOwner(
                                owner, oldTexture, (void *)metalTexture) != 0) {
                            fprintf(stderr,
                                    "MGL ERROR: Metal-cpp render-target preservation blit failed texture=%u\n",
                                    tex->name);
                            tex->dirty_bits |= DIRTY_TEXTURE_DATA;
                            mglTextureBindReleaseAlias(oldTexture);
                            return false;
                        }
                    }
                    tex->dirty_bits = 0;
                }
            } else {
                /* Fallback: use the old CPU-data-upload path */
                tex->dirty_bits |= DIRTY_TEXTURE_DATA;
            }
            if (newTexture) {
                /* Not stored: the port's +1 is ours to release. */
                mglSafeReleaseMetalObj(&newTexture);
            }
            mglTextureBindReleaseAlias(oldTexture);
        }
    }

    if (tex->dirty_bits)
    {
        /* LEVEL/ACCESS require a new Metal texture object. DATA-only dirty
         * should upload in place; destroying an existing texture here wipes
         * GPU imageStore results when cube-array CPU uploads stay incomplete. */
        const bool storageShapeChanged =
            (tex->dirty_bits & (DIRTY_TEXTURE_LEVEL | DIRTY_TEXTURE_ACCESS)) != 0;
        bool textureNeedsRebuild = storageShapeChanged;
        bool samplerNeedsRebuild =
            storageShapeChanged ||
            ((tex->dirty_bits & (DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_PARAM)) != 0);

        if (tex->mtl_data &&
            !storageShapeChanged &&
            (tex->dirty_bits & DIRTY_TEXTURE_DATA) != 0) {
            void *existingTexture = tex->mtl_data;
            bool uploadedDirty = false;
            if (existingTexture && !tex->metal_data_authoritative) {
                if (mglRenderTextureTargetIs2D((uint32_t)tex->target)) {
                    MGLRenderTextureInfo metalInfo = {0};
                    if (mglRenderGetTextureInfo(existingTexture, &metalInfo) == 0 &&
                        metalInfo.texture_type == MGLTextureType2D &&
                        !mglTextureUploadNeedsSwizzleBake(tex)) {
                        uploadedDirty =
                            mglTextureUploadFullCPUData(
                                renderer, tex, existingTexture,
                                "bindMTLTexture.dirtyData") != 0;
                    }
                }
                if (!uploadedDirty) {
                    MGLRenderTextureInfo metalInfo = {0};
                    if (mglRenderGetTextureInfo(existingTexture, &metalInfo) == 0) {
                        const bool isArray =
                            mglRenderTextureTargetIsLayeredUpload(
                                (uint32_t)tex->target) != 0;
                        const bool texture1DBackedBy2D =
                            mglRenderTextureTargetIs1D((uint32_t)tex->target) != 0;
                        const bool texture1DArrayBackedBy2DArray =
                            mglRenderTextureTargetIs1DArray((uint32_t)tex->target) != 0;
                        int allLevelsUploaded = 1;
                        const uint32_t levelCount = mglTextureBindUploadLevelCount(
                            metalInfo.mipmap_level_count,
                            tex->num_levels ? tex->num_levels : 1u);
                        uploadedDirty =
                            mglTextureUploadDirty(
                                renderer, tex, existingTexture,
                                (uint32_t)metalInfo.pixel_format, 1, levelCount,
                                isArray ? 1 : 0,
                                texture1DBackedBy2D ? 1 : 0,
                                texture1DArrayBackedBy2DArray ? 1 : 0,
                                (uint32_t)metalInfo.texture_type,
                                &allLevelsUploaded) != 0 &&
                            allLevelsUploaded != 0;
                    }
                }
            }
            if (uploadedDirty || tex->metal_data_authoritative) {
                /* Successful CPU upload, or GPU is source of truth after
                 * imageStore / MemoryBarrier — drop DATA dirty without
                 * releasing mtl_data. */
                tex->dirty_bits &= ~DIRTY_TEXTURE_DATA;
            }
            samplerNeedsRebuild =
                storageShapeChanged ||
                ((tex->dirty_bits & (DIRTY_TEXTURE_DATA | DIRTY_TEXTURE_PARAM)) != 0);
        }

        /* Texture parameter changes only affect the Metal sampler object. Do
         * not throw away texture storage for wrap/filter/lod updates; doing so
         * can turn Minecraft's frequent sampler changes into render-pass and
         * upload storms. */
        if (tex->mtl_data)
        {
            if (textureNeedsRebuild) {
                mglSafeReleaseMetalObj((void **)&tex->mtl_data);
                mglTextureReleaseGLSampledCopy(tex);
            }
        }

        if (samplerNeedsRebuild && tex->params.mtl_data)
        {
            mglSafeReleaseMetalObj((void **)&tex->params.mtl_data);
        }
    }

    if (tex->mtl_data == NULL)
    {
        tex->mtl_data = mglTextureCreateFromGLTexture(renderer, tex);

        /* AGX-SAFE: Handle NULL texture gracefully when in GPU recovery mode */
        if (!tex->mtl_data) {
            /* Circuit breaker: limit fallback texture creations to prevent
             * infinite loops. */
            static int s_fallbackTextureCount = 0;
            static double s_fallbackTextureWindowStart = 0;
            const double now = mglTextureBindNowSeconds();
            if (now - s_fallbackTextureWindowStart > 5.0) {
                s_fallbackTextureCount = 0;
                s_fallbackTextureWindowStart = now;
            }
            if (s_fallbackTextureCount >= 4096) {
                fprintf(stderr,
                        "MGL AGX: Fallback texture limit reached (%d in %.1fs), suppressing further fallbacks\n",
                        s_fallbackTextureCount, now - s_fallbackTextureWindowStart);
                tex->mtl_data = NULL;
                tex->dirty_bits = 0;
            } else {
                s_fallbackTextureCount++;
                fprintf(stderr,
                        "MGL AGX: Primary texture creation returned NULL, attempting fallback texture creation (%d/4096)\n",
                        s_fallbackTextureCount);
                /* Create a simple fallback texture to prevent crashes */
                tex->mtl_data =
                    mglRendererCreateFallbackMTLTexturePort(renderer, tex);

                if (tex->mtl_data) {
                    fprintf(stderr, "MGL SUCCESS: Fallback texture created successfully\n");
                    tex->dirty_bits = 0;
                } else {
                    fprintf(stderr,
                            "MGL ERROR: Even fallback texture creation failed - this texture will remain NULL\n");
                }
            }
        } else {
            if (kMGLDiagnosticStateLogs) {
                mglTraceLog("MGL SUCCESS: Primary texture created successfully");
            }
        }
    }

    if (tex->params.mtl_data == NULL)
    {
        tex->params.mtl_data =
            mglTextureCreateSamplerForTexParam(&tex->params, tex->target);
        /* Sampler creation should not fail even in recovery mode */
        if (!tex->params.mtl_data) {
            fprintf(stderr, "MGL WARNING: Sampler creation failed, using default\n");
            tex->params.mtl_data = mglTextureBindCreateDefaultSampler();
        }
        if ((tex->name == 21u || tex->name == 27u) &&
            mgl_env_flag_enabled("MGL_TRACE_TEXTURE_NAMES")) {
            mglTraceLog("SAMPLER_CREATE tex=%u minFilter=0x%x magFilter=0x%x mipFilter=%d minLod=%.3f maxLod=%.3f base=%u max=%u mips=%u",
                        (unsigned)tex->name,
                        (unsigned)tex->params.min_filter,
                        (unsigned)tex->params.mag_filter,
                        (tex->params.min_filter >= 0x2700) ? 1 : 0,
                        (double)tex->params.min_lod,
                        (double)tex->params.max_lod,
                        (unsigned)tex->params.base_level,
                        (unsigned)tex->params.max_level,
                        (unsigned)tex->mipmap_levels);
        }
    }

    if (tex->params.mtl_data) {
        tex->dirty_bits &= ~DIRTY_TEXTURE_PARAM;
    }

    if (mglMipDiagEnabled()) {
        void *mtlTex = tex->mtl_data;
        MGLRenderTextureInfo textureInfo = {0};
        bool hasTextureInfo = mtlTex &&
            mglRenderGetTextureInfo(mtlTex, &textureInfo) == 0;
        uint64_t signature = 1469598103934665603ULL;
        signature = mglMipDiagMixState(signature, (uint64_t)(uintptr_t)tex->mtl_data);
        signature = mglMipDiagMixState(
            signature, hasTextureInfo ? textureInfo.mipmap_level_count : 0u);
        signature = mglMipDiagMixState(signature, tex->num_levels);
        signature = mglMipDiagMixState(signature, tex->mipmap_levels);
        signature = mglMipDiagMixState(signature, tex->params.base_level);
        signature = mglMipDiagMixState(signature, tex->params.max_level);
        signature = mglMipDiagMixState(signature, tex->mipmapped ? 1u : 0u);
        signature = mglMipDiagMixState(signature, tex->genmipmaps ? 1u : 0u);

        /* Direct-mapped by name; a collision only costs an extra line. */
        static uint64_t s_textureState[128];
        if (mglMipDiagStateChanged(&s_textureState[tex->name & 127u], signature)) {
            fprintf(stderr,
                    "MGL MIP_DIAG texture glTex=%u target=0x%x size=%ux%u "
                    "glLevels=%u mipmapLevels=%u mtlLevels=%lu base=%u max=%u "
                    "mipmapped=%d genmipmaps=%d renderTarget=%d mtlTex=%p\n",
                    (unsigned)tex->name,
                    (unsigned)tex->target,
                    (unsigned)tex->width,
                    (unsigned)tex->height,
                    (unsigned)tex->num_levels,
                    (unsigned)tex->mipmap_levels,
                    (unsigned long)(hasTextureInfo ? textureInfo.mipmap_level_count : 0u),
                    (unsigned)tex->params.base_level,
                    (unsigned)tex->params.max_level,
                    tex->mipmapped ? 1 : 0,
                    tex->genmipmaps ? 1 : 0,
                    tex->is_render_target ? 1 : 0,
                    mtlTex);
        }
    }

    return true;
}
