/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * mgl_debug_fbo_stats.h -- diagnostic FBO sampling (MGL_DEBUG_FBO_STATS).
 */

#ifndef MGL_DEBUG_FBO_STATS_H
#define MGL_DEBUG_FBO_STATS_H

#include <stdint.h>

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

void mglDebugFboStatsSetFrame(uint64_t swap_call);
void mglDebugFboStatsOnDrawFramebufferChange(GLMContext ctx, Framebuffer *old_fbo);
void mglDebugFboStatsAtSwap(GLMContext ctx, GLuint width, GLuint height);

#ifdef __cplusplus
}
#endif

#endif /* MGL_DEBUG_FBO_STATS_H */
