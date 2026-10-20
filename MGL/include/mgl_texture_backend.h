/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_texture_backend.h -- access to the backend state of a Texture.
 *
 * Texture::backend (MGLTextureBackend, see mgl_backend_objects.h) is allocated
 * the first time the backend has something to keep and freed with the texture.
 * Readers use the view, which never returns NULL; writers use the ensure
 * accessor.  If that allocation fails the writer gets a scratch sink, so the
 * caller stays straight-line and the texture simply caches nothing.
 */

#ifndef MGL_TEXTURE_BACKEND_H
#define MGL_TEXTURE_BACKEND_H

#include <stdlib.h>
#include <string.h>

#include "glm_context.h"
#include "mgl_backend_objects.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Release the cached base-level view (through the renderer's deferred
 * deletion when `ctx` is set) and forget it. */
void mglTextureBackendDropBaseLevelView(GLMContext ctx, Texture *texture);

/* Drop everything the backend keeps for `texture` and free the struct. */
void mglTextureBackendRelease(GLMContext ctx, Texture *texture);

/* The Metal view kept for an image unit, or NULL. */
static inline void *mglImageUnitView(const ImageUnit *unit);
/* Keep `view` (a reference the caller hands over) for the image unit and
 * release the previous one; NULL just releases. */
void mglImageUnitSetView(ImageUnit *unit, void *view);
/* Release the unit's view and free its backend struct. */
void mglImageUnitBackendRelease(ImageUnit *unit);

#ifdef __cplusplus
}
#endif

static inline void *mglImageUnitView(const ImageUnit *unit)
{
    return unit && unit->backend ? unit->backend->view : NULL;
}

/* The Metal texture kept for `texture`, or NULL. */
static inline void *mglTextureMtl(const Texture *texture)
{
    return texture && texture->backend ? texture->backend->mtl_data : NULL;
}

/* The texture's own Metal sampler state, or NULL. */
static inline void *mglTextureSamplerState(const Texture *texture)
{
    return texture && texture->backend ? texture->backend->sampler_state : NULL;
}

/* The owner Metal texture a texture view was last built from, or NULL. */
static inline void *mglTextureViewParent(const Texture *texture)
{
    return texture && texture->backend ? texture->backend->view_parent : NULL;
}

static inline struct MGLTextureBackend *
mglTextureBackendEnsure(Texture *texture)
{
    static struct MGLTextureBackend sink;
    if (!texture) {
        memset(&sink, 0, sizeof(sink));
        return &sink;
    }
    if (!texture->backend) {
        texture->backend = (struct MGLTextureBackend *)calloc(
            1, sizeof(struct MGLTextureBackend));
        if (!texture->backend) {
            memset(&sink, 0, sizeof(sink));
            return &sink;
        }
    }
    return texture->backend;
}

static inline int mglTextureStorageIsFallback(const Texture *texture)
{
    return (texture && texture->backend && texture->backend->storage_is_fallback)
               ? 1 : 0;
}

static inline void mglTextureSetStorageIsFallback(Texture *texture, int value)
{
    mglTextureBackendEnsure(texture)->storage_is_fallback = value ? 1u : 0u;
}

#endif /* MGL_TEXTURE_BACKEND_H */
