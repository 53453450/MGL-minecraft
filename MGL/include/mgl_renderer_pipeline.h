/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Pipeline/PSO sync orchestration (C++ home).
 * Cache key construction: mgl_pipeline_cache_key.{h,c}
 * GPU recovery breakers: mgl_pipeline_recovery.{h,c}
 */

#ifndef MGL_RENDERER_PIPELINE_H
#define MGL_RENDERER_PIPELINE_H

#include <stdbool.h>

#include "glm_context.h"
#include "mgl_pipeline_cache_key.h"
#include "mgl_pipeline_recovery.h"

#ifdef __cplusplus
extern "C" {
#endif

int mglRenderSyncPipeline(GLMContext context, int deferred_buffer_map);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_PIPELINE_H */
