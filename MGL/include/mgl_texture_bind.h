/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_texture_bind.h — per-texture Metal materialization, formerly
 * -[MGLRenderer bindMTLTextureLocked:] (P0-1).
 *
 * The body was already C-shaped: the texture record and every Metal object it
 * touches live in C storage, and the decisions (render-target usage / mip
 * transition, dirty-bit bookkeeping, the fallback circuit breaker) read plain
 * struct fields.  The four steps that still need Objective-C — Metal texture
 * creation, the two CPU-data uploads and the render-target preservation blit —
 * come back through the ports declared in mgl_renderer_ports.h.
 */

#ifndef MGL_TEXTURE_BIND_H
#define MGL_TEXTURE_BIND_H

#include "mgl_types_texture.h"   /* Texture */

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Materialize (or refresh) the Metal texture and sampler of `tex`, then push
 * whatever is dirty.  False when the texture cannot be made usable.
 *
 * The caller holds the renderer lock: METAL_LOCK() is only a GL-thread assert,
 * so -[MGLRenderer bindMTLTexture:] keeps it and calls straight in. */
bool mglRendererBindMTLTexture(void *renderer, Texture *tex);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TEXTURE_BIND_H */
