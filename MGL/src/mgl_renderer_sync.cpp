/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * GL→Metal state synchronization orchestration (C++ home).
 * Phase 2+: migrates processGLState / processDirtyStateDomains from ObjC.
 */

#include "mgl_renderer_sync.h"

#include "mgl_types_state.h"

extern "C" bool mglRendererObjCProcessGLState(GLMContext context, bool draw_command);

extern "C" int mglRenderProcessGLState(GLMContext context, int draw_command)
{
    if (!context) {
        return 0;
    }
    return mglRendererObjCProcessGLState(context, draw_command != 0) ? 1 : 0;
}

extern "C" int mglRenderProcessDirtyStateDomains(GLMContext context,
                                                 unsigned int domain_mask,
                                                 int draw_command)
{
    (void)domain_mask;
    return mglRenderProcessGLState(context, draw_command);
}
