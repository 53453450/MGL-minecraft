/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Deferred draw-batch flush orchestration (C++ home).
 */

#ifndef MGL_RENDERER_BATCH_H
#define MGL_RENDERER_BATCH_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

void mglRenderFlushDrawBuffer(GLMContext context);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_BATCH_H */
