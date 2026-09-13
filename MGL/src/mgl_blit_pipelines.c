/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_blit_pipelines.c — the blit/clear pipeline, sampler and depth-state
 * cache, moved out of Objective-C (P0-1, first slice of MGLRenderer+Blit.m).
 *
 * The bodies were already C; the ObjC parts were the method shells, the
 * NSError/NSString error descriptions (now char buffers on stderr) and the
 * `id` handles (now borrowed void *).  Ownership follows the renderer-lifetime
 * caches exactly as before: the +1 from the metal-cpp creation calls is handed
 * to the cache (which retains) and the borrow is what callers receive.
 */

#include "mgl_blit_pipelines.h"
#include "mgl_renderer_ports.h"
#include "mgl_draw_issue.h"          /* mglDrawHostDevice */
#include "mgl_render.h"
#include "mgl_metal_ref.h"           /* mglReleaseMetalObjNoNull */
#include "mgl_batch_path.h"          /* mgl_batch_icb_support_indirect_command_buffers */
#include "mgl_aux_assets.h"
#include "mgl_texture_compat.h"
#include "mgl_types_program.h"

#include <stdio.h>
#include <string.h>

/* === asset helpers (were file-local functions in MGLRenderer+Blit.m) === */

static void mglBlitAuxError(char *errbuf, size_t errcap, const char *fmt,
                            const char *a, const char *b)
{
    if (!errbuf || errcap == 0) {
        return;
    }
    snprintf(errbuf, errcap, fmt, a ? a : "", b ? b : "");
}

static void *mglBlitLookupAuxRenderPipeline(
    uint32_t kind, uint64_t variant,
    uint32_t colorFormat, uint32_t depthFormat,
    uint32_t stencilFormat, uint32_t colorWriteMask,
    uint32_t rasterSampleCount)
{
    void *pipeline = NULL;
    int icbEnabled = mgl_batch_icb_support_indirect_command_buffers();
    if (mglRenderGetOrCreateAuxRenderPipeline(
            NULL, NULL, kind, variant, (uint32_t)colorFormat,
            (uint32_t)depthFormat, (uint32_t)stencilFormat,
            (uint32_t)colorWriteMask, icbEnabled, rasterSampleCount,
            &pipeline, NULL, 0) == 0 && pipeline) {
        /* The renderer-lifetime cache owns it; hand back the borrow. */
        mglReleaseMetalObjNoNull(pipeline);
        return pipeline;
    }
    return NULL;
}

static void *mglBlitCreateAuxRenderPipelineFromAsset(
    const char *assetName, const char *vsEntry, const char *fsEntry,
    uint32_t kind, uint64_t variant,
    uint32_t colorFormat, uint32_t depthFormat,
    uint32_t stencilFormat, uint32_t colorWriteMask,
    uint32_t rasterSampleCount, char *errbuf, size_t errcap)
{
    const MGLAuxShaderAsset *asset = mglAuxShaderAssetFind(assetName);
    if (!asset || !asset->data || asset->size == 0) {
        mglBlitAuxError(errbuf, errcap, "aux shader asset '%s' missing%s",
                        assetName, "");
        return NULL;
    }
    void *pipeline = NULL;
    char message[512] = {0};
    int icbEnabled = mgl_batch_icb_support_indirect_command_buffers();
    if (mglRenderGetOrCreateAuxRenderPipelineFromMetallib(
            asset->data, asset->size, asset->hash,
            vsEntry, fsEntry, kind, variant,
            (uint32_t)colorFormat, (uint32_t)depthFormat,
            (uint32_t)stencilFormat, (uint32_t)colorWriteMask, icbEnabled,
            rasterSampleCount, &pipeline, message, sizeof(message)) == 0 &&
        pipeline) {
        mglReleaseMetalObjNoNull(pipeline);
        return pipeline;
    }
    mglBlitAuxError(errbuf, errcap, "%s%s",
                    message[0] ? message
                               : "Metal-cpp auxiliary render pipeline creation failed",
                    "");
    return NULL;
}

static void *mglBlitLookupAuxComputePipeline(uint32_t kind, uint64_t variant)
{
    void *pipeline = NULL;
    if (mglRenderGetOrCreateAuxComputePipeline(
            NULL, kind, variant, &pipeline, NULL, 0) == 0 && pipeline) {
        mglReleaseMetalObjNoNull(pipeline);
        return pipeline;
    }
    return NULL;
}

static void *mglBlitCreateAuxComputePipelineFromAsset(
    const char *assetName, const char *entryName,
    uint32_t kind, uint64_t variant, char *errbuf, size_t errcap)
{
    const MGLAuxShaderAsset *asset = mglAuxShaderAssetFind(assetName);
    if (!asset || !asset->data || asset->size == 0) {
        mglBlitAuxError(errbuf, errcap, "aux shader asset '%s' missing%s",
                        assetName, "");
        return NULL;
    }
    void *pipeline = NULL;
    char message[512] = {0};
    if (mglRenderGetOrCreateAuxComputePipelineFromMetallib(
            asset->data, asset->size, asset->hash, entryName,
            kind, variant, &pipeline, message, sizeof(message)) == 0 &&
        pipeline) {
        mglReleaseMetalObjNoNull(pipeline);
        return pipeline;
    }
    mglBlitAuxError(errbuf, errcap, "%s%s",
                    message[0] ? message
                               : "Metal-cpp auxiliary compute pipeline creation failed",
                    "");
    return NULL;
}

/* Returns +1 (the caller gives it to the backend blit cache). */
static void *mglBlitCreateSampler(void *device, uint32_t nearest)
{
    (void)device;
    void *sampler = NULL;
    if (mglRenderCreateFilterSampler(nearest, &sampler) == 0 && sampler) {
        return sampler;
    }
    return NULL;
}

/* Returns +1 (the caller gives it to the backend blit cache). */
static void *mglBlitCreateDepthStencilState(
    void *device, const MGLRenderDepthStencilDescriptorState *descriptor)
{
    (void)device;
    void *state = NULL;
    if (mglRenderCreateDepthStencilStateFromState(descriptor, &state) == 0 &&
        state) {
        return state;
    }
    return NULL;
}

/* === the entry points (were MGLRenderer methods) === */

void *mglBlitScaledSamplerForFilter(void *renderer, uint32_t filter)
{
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    int wantsNearest = mglRenderFilterIsNearest(filter) != 0;
    MGLRendererBackendBlitCacheKind cacheKind = wantsNearest
        ? MGL_RENDERER_BACKEND_BLIT_CACHE_NEAREST_SAMPLER
        : MGL_RENDERER_BACKEND_BLIT_CACHE_LINEAR_SAMPLER;
    void *cached = mglRendererBackendGetBlitCachedObject(areas.backend, cacheKind);
    if (cached) {
        return cached;
    }

    void *sampler = mglBlitCreateSampler(mglDrawHostDevice(renderer),
                                         wantsNearest ? 1u : 0u);
    if (!sampler) {
        fprintf(stderr, "MGL ERROR: failed to create scaled blit sampler filter=0x%x\n",
                filter);
        return NULL;
    }

    if (mglRendererBackendSetBlitCachedObject(areas.backend, cacheKind,
                                              sampler) != 0) {
        mglReleaseMetalObjNoNull(sampler);
        return NULL;
    }
    mglReleaseMetalObjNoNull(sampler);   /* the cache retains it now */
    return mglRendererBackendGetBlitCachedObject(areas.backend, cacheKind);
}

void *mglBlitScaledPipelineForPixelFormat(void *renderer, uint32_t pixelFormat)
{
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    pixelFormat = mglRenderColorFormatOrBGRA(pixelFormat);

    uint64_t variant = (uint64_t)pixelFormat;
    void *cached = mglBlitLookupAuxRenderPipeline(
        MGL_RENDER_AUX_RENDER_SCALED_BLIT, variant,
        pixelFormat, mglRenderInvalidPixelFormat(), mglRenderInvalidPixelFormat(),
        MGLColorWriteMaskAll, 1u);
    if (cached) return cached;

    char error[512] = {0};
    void *pipeline = mglBlitCreateAuxRenderPipelineFromAsset(
        "scaled_blit", "mgl_scaled_blit_vs", "mgl_scaled_blit_fs",
        MGL_RENDER_AUX_RENDER_SCALED_BLIT, variant,
        pixelFormat, mglRenderInvalidPixelFormat(), mglRenderInvalidPixelFormat(),
        MGLColorWriteMaskAll, 1u, error, sizeof(error));
    if (!pipeline) {
        fprintf(stderr,
                "MGL ERROR: scaled blit asset pipeline create failed pixelFormat=%lu error=%s\n",
                (unsigned long)pixelFormat, error[0] ? error : "(none)");
        if (areas.ctx) mglDispatchError(areas.ctx, __func__, (GLenum)mglRenderErrorInvalidOperation());
        return NULL;
    }
    fprintf(stderr, "MGL INFO: created scaled blit pipeline pixelFormat=%lu (Metal-cpp asset)\n",
            (unsigned long)pixelFormat);
    return pipeline;
}

void *mglBlitScaledDepthPipelineForPixelFormat(void *renderer, uint32_t pixelFormat)
{
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    if (mglRenderPixelFormatIsInvalid(pixelFormat)) {
        return NULL;
    }

    uint32_t stencilFormat = mglRenderDepthBlitStencilFormat(pixelFormat);
    uint64_t variant = ((uint64_t)pixelFormat << 1) |
                       (!mglRenderPixelFormatIsInvalid(stencilFormat) ? 1u : 0u);
    void *cached = mglBlitLookupAuxRenderPipeline(
        MGL_RENDER_AUX_RENDER_SCALED_DEPTH_BLIT, variant,
        mglRenderInvalidPixelFormat(), pixelFormat, stencilFormat,
        MGLColorWriteMaskNone, 1u);
    if (cached) return cached;

    char error[512] = {0};
    void *pipeline = mglBlitCreateAuxRenderPipelineFromAsset(
        "scaled_depth_blit", "mgl_scaled_depth_blit_vs",
        "mgl_scaled_depth_blit_fs",
        MGL_RENDER_AUX_RENDER_SCALED_DEPTH_BLIT, variant,
        mglRenderInvalidPixelFormat(), pixelFormat, stencilFormat,
        MGLColorWriteMaskNone, 1u, error, sizeof(error));
    if (!pipeline) {
        fprintf(stderr,
                "MGL ERROR: scaled depth asset pipeline create failed depthPixelFormat=%lu error=%s\n",
                (unsigned long)pixelFormat, error[0] ? error : "(none)");
        if (areas.ctx) mglDispatchError(areas.ctx, __func__, (GLenum)mglRenderErrorInvalidOperation());
        return NULL;
    }
    fprintf(stderr,
            "MGL INFO: created scaled depth blit pipeline depthPixelFormat=%lu (Metal-cpp asset)\n",
            (unsigned long)pixelFormat);
    return pipeline;
}

void *mglBlitScaledComputePipelineForPixelFormat(void *renderer, uint32_t pixelFormat)
{
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    pixelFormat = mglRenderColorFormatOrBGRA(pixelFormat);

    MGLTextureDataKind dataKind = mglTextureDataKindForPixelFormat(pixelFormat);
    const char *entryName = "mgl_scaled_blit_cs";
    if (dataKind == MGLTextureDataKindUint) {
        entryName = "mgl_scaled_blit_cs_uint";
    } else if (dataKind == MGLTextureDataKindSint) {
        entryName = "mgl_scaled_blit_cs_int";
    } else if (dataKind == MGLTextureDataKindDepth) {
        return NULL;
    }
    /* Encode data kind so uint/int/float caches do not collide. */
    uint64_t variant =
        ((uint64_t)(uint32_t)dataKind << 32) | (uint64_t)pixelFormat;

    void *cached = mglBlitLookupAuxComputePipeline(
        MGL_RENDER_AUX_COMPUTE_SCALED_BLIT, variant);
    if (cached) return cached;

    char error[512] = {0};
    void *pipeline = mglBlitCreateAuxComputePipelineFromAsset(
        "scaled_blit_cs", entryName, MGL_RENDER_AUX_COMPUTE_SCALED_BLIT,
        variant, error, sizeof(error));
    if (!pipeline) {
        fprintf(stderr,
                "MGL ERROR: scaled blit asset compute pipeline create failed pixelFormat=%lu kind=%s entry=%s error=%s\n",
                (unsigned long)pixelFormat, mglTextureDataKindName(dataKind), entryName,
                error[0] ? error : "(none)");
        if (areas.ctx) mglDispatchError(areas.ctx, __func__, (GLenum)mglRenderErrorInvalidOperation());
        return NULL;
    }
    fprintf(stderr,
            "MGL INFO: created scaled blit compute pipeline pixelFormat=%lu kind=%s entry=%s (Metal-cpp asset)\n",
            (unsigned long)pixelFormat, mglTextureDataKindName(dataKind), entryName);
    return pipeline;
}

void *mglBlitMsaaIntegerResolvePipeline(void *renderer, int signedInteger)
{
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    const char *entryName = signedInteger
        ? "mgl_msaa_resolve_int" : "mgl_msaa_resolve_uint";
    void *cached = mglBlitLookupAuxComputePipeline(
        MGL_RENDER_AUX_COMPUTE_MSAA_INTEGER_RESOLVE,
        signedInteger ? 1u : 0u);
    if (cached) return cached;

    char error[512] = {0};
    void *pipeline = mglBlitCreateAuxComputePipelineFromAsset(
        "msaa_integer_resolve", entryName,
        MGL_RENDER_AUX_COMPUTE_MSAA_INTEGER_RESOLVE,
        signedInteger ? 1u : 0u, error, sizeof(error));
    if (!pipeline) {
        fprintf(stderr,
                "MGL ERROR: MSAA integer resolve asset pipeline create failed signed=%d error=%s\n",
                signedInteger ? 1 : 0, error[0] ? error : "(none)");
        if (areas.ctx) mglDispatchError(areas.ctx, __func__, (GLenum)mglRenderErrorInvalidOperation());
        return NULL;
    }
    return pipeline;
}

void *mglBlitClearRectPipeline(void *renderer, uint32_t colorFormat,
                              uint32_t depthFormat, int writesColor,
                              int writesDepth)
{
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    if (!mglRenderClearRectPipelineReady(writesColor ? 1 : 0, colorFormat,
                                         writesDepth ? 1 : 0, depthFormat)) {
        return NULL;
    }

    uint64_t variant = (uint64_t)(uint32_t)colorFormat |
                       ((uint64_t)(uint32_t)depthFormat << 16) |
                       ((uint64_t)(writesColor ? 1u : 0u) << 32) |
                       ((uint64_t)(writesDepth ? 1u : 0u) << 33);
    void *cached = mglBlitLookupAuxRenderPipeline(
        MGL_RENDER_AUX_RENDER_CLEAR_RECT, variant,
        colorFormat, depthFormat, mglRenderInvalidPixelFormat(),
        writesColor ? MGLColorWriteMaskAll : MGLColorWriteMaskNone, 1u);
    if (cached) return cached;

    char error[512] = {0};
    void *pipeline = mglBlitCreateAuxRenderPipelineFromAsset(
        "clear_rect", "mgl_clear_rect_vs",
        writesColor ? "mgl_clear_rect_fs" : NULL,
        MGL_RENDER_AUX_RENDER_CLEAR_RECT, variant,
        colorFormat, depthFormat, mglRenderInvalidPixelFormat(),
        writesColor ? MGLColorWriteMaskAll : MGLColorWriteMaskNone,
        1u, error, sizeof(error));
    if (!pipeline) {
        fprintf(stderr,
                "MGL ERROR: scissored clear asset pipeline create failed color=%lu depth=%lu writesColor=%d writesDepth=%d error=%s\n",
                (unsigned long)colorFormat, (unsigned long)depthFormat,
                writesColor ? 1 : 0, writesDepth ? 1 : 0,
                error[0] ? error : "(none)");
        if (areas.ctx) mglDispatchError(areas.ctx, __func__, (GLenum)mglRenderErrorInvalidOperation());
        return NULL;
    }
    fprintf(stderr, "MGL INFO: created scissored clear pipeline (Metal-cpp asset)\n");
    return pipeline;
}

void *mglBlitClearRectDepthState(void *renderer)
{
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    void *cached = mglRendererBackendGetBlitCachedObject(
        areas.backend, MGL_RENDERER_BACKEND_BLIT_CACHE_CLEAR_DEPTH_STATE);
    if (cached) {
        return cached;
    }

    MGLRenderDepthStencilDescriptorState desc = {0};
    desc.depth_compare_function = MGLCompareFunctionAlways;
    desc.depth_write_enabled = 1u;
    void *depthState = mglBlitCreateDepthStencilState(mglDrawHostDevice(renderer), &desc);
    if (!depthState ||
        mglRendererBackendSetBlitCachedObject(
            areas.backend, MGL_RENDERER_BACKEND_BLIT_CACHE_CLEAR_DEPTH_STATE,
            depthState) != 0) {
        if (depthState) {
            mglReleaseMetalObjNoNull(depthState);
        }
        return NULL;
    }
    mglReleaseMetalObjNoNull(depthState);   /* the cache retains it now */
    return mglRendererBackendGetBlitCachedObject(
        areas.backend, MGL_RENDERER_BACKEND_BLIT_CACHE_CLEAR_DEPTH_STATE);
}
