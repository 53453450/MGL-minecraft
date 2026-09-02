/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * C/C++ renderer state-sync facade.  GL→Metal synchronization orchestration
 * lives here (or in mgl_renderer_sync.cpp); ObjC retains only platform-shell
 * callbacks during migration.
 */

#ifndef MGL_RENDERER_SYNC_H
#define MGL_RENDERER_SYNC_H

#include <stdbool.h>
#include <stdint.h>

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Work already performed by processDirtyStateDomains within one
 * processGLState invocation; syncResourceBindings skips these steps. */
typedef struct MGLResourceSyncWork_t {
    bool mappedBuffers;
    bool updatedBaseLists;
    bool boundActiveTextures;
} MGLResourceSyncWork;

/* Primary GL→Metal state sync entry.  Returns 1 on success, 0 on failure.
 * During migration this may delegate to the ObjC renderer; the long-term
 * implementation lives entirely in C++. */
int mglRenderProcessGLState(GLMContext context, int draw_command);

/* Domain-scoped sync helper for incremental migration from ObjC. */
int mglRenderProcessDirtyStateDomains(GLMContext context,
                                      unsigned int domain_mask,
                                      int draw_command);

/* ObjC renderer bridge — implemented in MGLRenderer+RenderPass.m until Phase 2
 * migration completes. */
bool mglRendererObjCProcessGLState(GLMContext context, bool draw_command);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_SYNC_H */
