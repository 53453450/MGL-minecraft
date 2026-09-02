/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Pipeline/PSO sync orchestration (C++ home).
 */

#ifndef MGL_RENDERER_PIPELINE_H
#define MGL_RENDERER_PIPELINE_H

#include <stdbool.h>

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

int mglRenderSyncPipeline(GLMContext context, int deferred_buffer_map);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_PIPELINE_H */
