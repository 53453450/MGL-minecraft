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

/* The three BOOL-returning symbols (mglRendererTextureLooksRecoverableSampled2D,
 * mglRendererGLSampledCopyLooksUsable, mglRendererTextureLooksLikeSampledColor2D)
 * keep their ObjC-header declarations; the C TUs that need them restate them
 * locally as `signed char` (rule 26), so they are not declared here to avoid
 * clashing with MGLRenderer+Draw_Private.h when the .m includes this header. */



void mglMarkTextureLevelRenderTargetWrittenImpl(Texture *tex, GLuint level,
                                                const char *caller, int line);
/* mglTraceReplayCommandVertexAttribSamples keeps its mgl_trace_strategy.h
 * declaration; mglFindFramebufferColorTexturePairedWithDepth keeps its
 * MGLRenderer+Draw_Private.h one. */

int mglRendererEnsureNewCommandBuffer(void *renderer);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_HOST_H */
