/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by its respective copyright holder.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_blit_sampled_copy.h — the GL-sampled render-target copy machinery of
 * MGLRenderer+Blit.m (P0-1).  Bodies were already C; only the method shells and
 * the ObjC handles are gone.
 */

#ifndef MGL_BLIT_SAMPLED_COPY_H
#define MGL_BLIT_SAMPLED_COPY_H

#include "glm_context.h"
#include "mgl_types_texture.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* True when `tex` (a 2D float/int render target) may keep a GL-sampled copy of
 * `source` for FBO feedback. */
int mglBlitTextureCanUseGLSampledRenderTargetCopy(Texture *tex, void *source);

#ifdef __cplusplus
}
#endif

#endif /* MGL_BLIT_SAMPLED_COPY_H */
