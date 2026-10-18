/*
 * SPDX-License-Identifier: Apache-2.0 AND LGPL-3.0-only
 *
 * This file contains material from the Apache-2.0-licensed MGL baseline.
 * Copyrightable modifications made after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c are licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE-APACHE-2.0, LICENSE, and LICENSING.md.
 */

/*
 * mgl_rt_sync.h
 * MGL
 *
 * Render target authority: which side holds the content of a texture that is
 * (or was) rendered into.
 */

#ifndef MGL_RT_SYNC_H
#define MGL_RT_SYNC_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The first write to a render target makes the GPU the authority for its
 * content: later readbacks and copies read the Metal texture rather than the
 * CPU level it was created from. */
static inline void mglRenderTargetMarkGPUAuthoritative(Texture *tex)
{
    if (tex && tex->is_render_target &&
        tex->mtl_render_target_write_version == 0u) {
        tex->mtl_render_target_write_version = 1u;
    }
}

/* Whether the GPU, not the CPU level, holds the content of `tex`. */
static inline int mglTextureHasGPUContent(const Texture *tex)
{
    return tex && tex->is_render_target &&
           tex->mtl_render_target_write_version != 0u;
}

#ifdef __cplusplus
}
#endif

#endif /* MGL_RT_SYNC_H */
