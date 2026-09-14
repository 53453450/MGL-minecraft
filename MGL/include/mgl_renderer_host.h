/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * This file was added after baseline commit
 * 79d38f666336141d962109a864a6744bf66e438c and is licensed under
 * LGPL-3.0-only by their respective copyright holders.
 * See LICENSE and LICENSING.md.
 */

/*
 * mgl_renderer_host.h — home for the exported C symbols that used to live in
 * MGLRenderer.m (P0-1, log 160).
 *
 * These are the file's non-static functions; their names must not change
 * (mgl_renderer_backend.h / MGLRenderer+Draw_Private.h declare them).
 */

#ifndef MGL_RENDERER_HOST_H
#define MGL_RENDERER_HOST_H

#include <stdint.h>

#include "glm_context.h"        /* GLMContext */
#include "mgl_types_texture.h"  /* Texture */

#ifdef __cplusplus
extern "C" {
#endif

/* BOOL returns are signed char on macOS (rule 26). */
signed char mglRendererTextureLooksRecoverableSampled2D(GLMContext glctx,
                                                       Texture *tex,
                                                       uint32_t expected_type,
                                                       uint32_t expected_kind);
signed char mglRendererGLSampledCopyLooksUsable(Texture *tex,
                                                uint32_t expected_type,
                                                uint32_t expected_kind,
                                                int allow_previous_write_version,
                                                void **copy_out,
                                                signed char *used_previous_out);
void mglLogDrawWithoutSwapWatchdog(const char *kind, uint64_t draw_call,
                                   GLMContext ctx, void *command_buffer_owner,
                                   void *render_encoder_owner,
                                   void *render_pass_state_owner);
Texture *mglFindFramebufferColorTexturePairedWithDepth(GLMContext glctx,
                                                       Texture *depth_texture,
                                                       GLuint *fbo_name_out);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_HOST_H */
