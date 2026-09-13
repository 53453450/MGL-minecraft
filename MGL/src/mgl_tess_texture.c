/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_tess_texture.c — the texture half of the tessellation pre-bind moved out
 * of MGLRenderer+Tessellation.m (P0-1, log 123).  One message send
 * (bindMTLTexture:) became the C entry, and MGL_STATE(drawCtx) its twin.
 */

#include "mgl_tess_texture.h"
#include "mgl_texture_bind.h"   /* mglRendererBindMTLTexture */

static GLMState *mglTessTextureState(GLMContext draw_ctx)
{
    return draw_ctx ? draw_ctx->active_state : NULL;
}

int mglTessEnsureTextureMetalData(void *renderer,
                                  const MGLTessTextureBind *binds,
                                  uint32_t count, GLMContext draw_ctx)
{
    if (!binds || !draw_ctx || !draw_ctx->active_state) {
        return 1;
    }
    GLMState *state = mglTessTextureState(draw_ctx);
    for (uint32_t i = 0; i < count; i++) {
        const GLuint unit = binds[i].gl_unit;
        Texture *ptr = mglTessTextureBindIsStorage(binds[i].kind)
            ? state->image_units[unit].tex
            : state->active_textures[unit];
        if (ptr && !ptr->mtl_data) {
            (void)mglRendererBindMTLTexture(renderer, ptr);
        }
    }
    return 1;
}
