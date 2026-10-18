/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_texture_backend.c -- release of the backend state of a Texture.
 */

#include "mgl_texture_backend.h"

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
