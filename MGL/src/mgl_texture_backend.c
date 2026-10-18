/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_texture_backend.c -- release of the backend state of a Texture.
 */

#include "mgl_texture_backend.h"
#include "mgl_render_api_lifecycle.h" /* mglRenderReleaseMetalObject */

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
