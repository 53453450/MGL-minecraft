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

#ifdef __cplusplus
}
#endif

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

#endif /* MGL_TEXTURE_BACKEND_H */
