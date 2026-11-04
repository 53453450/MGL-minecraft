/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_texture_backend.c -- release of the backend state of a Texture.
 */

#include "mgl_texture_backend.h"
#include "mgl_render_api_lifecycle.h" /* mglRenderReleaseMetalObject */
#include "mgl_render_api_texture.h"

void *mglRenderTextureHandle(const Texture *texture)
{
    return mglTextureMtl(texture);
}

void *mglRenderTextureSamplerHandle(const Texture *texture)
{
    return mglTextureSamplerState(texture);
}

void *mglRenderTextureViewParentHandle(const Texture *texture)
{
    return mglTextureViewParent(texture);
}

void mglRenderTextureForgetHandle(Texture *texture)
{
    mglTextureForgetMtlData(texture);
}

void mglRenderTextureClearSamplerHandle(Texture *texture)
{
    if (texture) {
        mglTextureBackendEnsure(texture)->sampler_state = NULL;
    }
}

void mglRenderTextureClearViewParentHandle(Texture *texture)
{
    if (texture) {
        mglTextureBackendEnsure(texture)->view_parent = NULL;
    }
}

void mglRenderTextureDropBaseLevelView(GLMContext ctx, Texture *texture)
{
    mglTextureBackendDropBaseLevelView(ctx, texture);
}

void mglRenderTextureReleaseBackend(GLMContext ctx, Texture *texture)
{
    mglTextureBackendRelease(ctx, texture);
}

int mglRenderTextureStorageIsFallback(const Texture *texture)
{
    return mglTextureStorageIsFallback(texture);
}

uint32_t mglRenderTextureWriteVersion(const Texture *texture)
{
    return mglTextureWriteVersion(texture);
}

void mglTextureBackendDropBaseLevelView(GLMContext ctx, Texture *texture)
{
    if (!texture || !texture->backend ||
        !texture->backend->base_level_view.view) {
        return;
    }
    if (ctx) {
        mglRendererDeleteMetalObject(ctx, texture->backend->base_level_view.view);
    }
    memset(&texture->backend->base_level_view, 0,
           sizeof(texture->backend->base_level_view));
}

void mglTextureBackendRelease(GLMContext ctx, Texture *texture)
{
    if (!texture || !texture->backend) {
        return;
    }
    mglTextureBackendDropBaseLevelView(ctx, texture);
    free(texture->backend);
    texture->backend = NULL;
}

void mglImageUnitSetView(ImageUnit *unit, void *view)
{
    if (!unit) {
        return;
    }
    if (unit->backend && unit->backend->view) {
        mglRenderReleaseMetalObject(unit->backend->view);
        unit->backend->view = NULL;
    }
    if (!view) {
        return;
    }
    if (!unit->backend) {
        unit->backend = (struct MGLImageUnitBackend *)calloc(
            1, sizeof(*unit->backend));
        if (!unit->backend) {
            mglRenderReleaseMetalObject(view);
            return;
        }
    }
    unit->backend->view = view;
}

void mglImageUnitBackendRelease(ImageUnit *unit)
{
    if (!unit || !unit->backend) {
        return;
    }
    mglImageUnitSetView(unit, NULL);
    free(unit->backend);
    unit->backend = NULL;
}
