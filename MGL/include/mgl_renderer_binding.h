/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Resource binding orchestration (C++ home). Phase 3+: migrates buffer/texture
 * bind resolution from ObjC categories.
 */

#ifndef MGL_RENDERER_BINDING_H
#define MGL_RENDERER_BINDING_H

#include <stdbool.h>

#include "glm_context.h"
#include "mgl_renderer_sync.h"

#ifdef __cplusplus
extern "C" {
#endif

int mglRenderSyncResourceBindings(GLMContext context,
                                  const MGLResourceSyncWork *already_done);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_BINDING_H */
