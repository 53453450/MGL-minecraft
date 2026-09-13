/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_sampler.c — sampler creation, formerly
 * -[MGLRenderer createMTLSamplerForTexParam:target:],
 * -[MGLRenderer fallbackSamplerState] and the file-local
 * mglTextureCreateSampler helper of MGLRenderer+Texture.m.
 */

#include "mgl_texture_sampler.h"
#include "mgl_renderer_ports.h"
#include "mgl_draw_issue.h"        /* mglDrawHostDevice */
#include "mgl_render.h"
#include "mgl_metal_ref.h"

#include <stdio.h>

/* Returns +1. */
static void *mglTextureCreateDefaultSampler(void *device)
{
    (void)device;
    void *sampler = NULL;
    if (mglRenderCreateDefaultSampler(&sampler) == 0 && sampler) {
        return sampler;
    }
    return NULL;
}

void *mglTextureCreateSamplerForTexParam(const TextureParameter *tex_param,
                                         uint32_t target)
{
    mglMetalCountCreate(MGLMetalKindSampler);
    void *sampler = NULL;
    char error[256] = {0};
    if (mglRenderCreateSamplerForGL(tex_param, target, &sampler, error,
                                    sizeof(error)) == 0 &&
        sampler) {
        return sampler;
    }
    fprintf(stderr, "MGL SAMPLER ERROR: Metal-cpp sampler creation failed: %s\n",
            error[0] ? error : "unknown");
    return NULL;
}

void *mglTextureFallbackSamplerState(void *renderer)
{
    MGLRendererStateAreas areas; mglRendererStateAreasPort(renderer, &areas);
    void *cached = mglRendererBackendGetFallbackResource(
        areas.backend, MGL_RENDERER_BACKEND_FALLBACK_SAMPLER);
    if (cached) {
        return cached;
    }

    void *sampler = mglTextureCreateDefaultSampler(mglDrawHostDevice(renderer));
    if (!sampler) {
        fprintf(stderr, "MGL ERROR: Failed to create fallback sampler state\n");
        return NULL;
    }
    if (mglRendererBackendSetFallbackResource(
            areas.backend, MGL_RENDERER_BACKEND_FALLBACK_SAMPLER,
            sampler) != 0) {
        mglReleaseMetalObjNoNull(sampler);
        return NULL;
    }
    mglReleaseMetalObjNoNull(sampler);   /* the backend cache retains it now */
    return mglRendererBackendGetFallbackResource(
        areas.backend, MGL_RENDERER_BACKEND_FALLBACK_SAMPLER);
}
