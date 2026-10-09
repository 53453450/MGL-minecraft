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
#include "glm_context.h"

#include <stdio.h>
#include <string.h>

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

/* Metal always filters cube faces seamlessly.  With GL seamless disabled
 * (the default), approximate non-seamless edges by dropping bilinear so
 * adjacent-face colors do not bleed — needed when CTS leaves cross-type
 * cube views at the default LINEAR mag filter. */
static void mglTextureAdjustCubeSamplerForSeamless(
    TextureParameter *params, uint32_t target)
{
    if (!params ||
        (target != GL_TEXTURE_CUBE_MAP &&
         target != GL_TEXTURE_CUBE_MAP_ARRAY)) {
        return;
    }
    GLMContext ctx = MGLgetCurrentContext();
    if (ctx && ctx->active_state &&
        ctx->active_state->caps.texture_cube_map_seamless) {
        return;
    }
    params->mag_filter = GL_NEAREST;
    switch (params->min_filter) {
        case GL_LINEAR:
            params->min_filter = GL_NEAREST;
            break;
        case GL_LINEAR_MIPMAP_NEAREST:
            params->min_filter = GL_NEAREST_MIPMAP_NEAREST;
            break;
        case GL_LINEAR_MIPMAP_LINEAR:
            params->min_filter = GL_NEAREST_MIPMAP_LINEAR;
            break;
        default:
            break;
    }
}

void *mglTextureCreateSamplerForTexParam(const TextureParameter *tex_param,
                                         uint32_t target)
{
    mglMetalCountCreate(MGLMetalKindSampler);
    void *sampler = NULL;
    char error[256] = {0};
    TextureParameter local;
    const TextureParameter *use = tex_param;
    if (tex_param &&
        (target == GL_TEXTURE_CUBE_MAP ||
         target == GL_TEXTURE_CUBE_MAP_ARRAY)) {
        memcpy(&local, tex_param, sizeof(local));
        mglTextureAdjustCubeSamplerForSeamless(&local, target);
        use = &local;
    }
    if (mglRenderCreateSamplerForGL(use, target, &sampler, error,
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
    MGLRendererStateAreas areas; mglRendererFillStateAreas(renderer, &areas);
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
