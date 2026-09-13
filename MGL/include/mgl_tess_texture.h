/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_tess_texture.h — the tessellation texture pre-bind, formerly
 * -[MGLRenderer ensureTessTextureMetalData:count:ctx:] (P0-1, log 123).
 */

#ifndef MGL_TESS_TEXTURE_H
#define MGL_TESS_TEXTURE_H

#include "glm_context.h"
#include "mgl_draw_tess.h"   /* MGLTessTextureBind */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Materialize the Metal texture of every bind that does not have one yet.
 * Returns 1 (the method returned YES when there was nothing to do, too). */
int mglTessEnsureTextureMetalData(void *renderer,
                                  const MGLTessTextureBind *binds,
                                  uint32_t count, GLMContext draw_ctx);

#ifdef __cplusplus
}
#endif

#endif /* MGL_TESS_TEXTURE_H */
