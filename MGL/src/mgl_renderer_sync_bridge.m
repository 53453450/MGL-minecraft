/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * ObjC processGLState / pipeline sync bridge — entry from C++ sync facade.
 */

#import "MGLRenderer_Private.h"
#import "MGLRenderer+RenderPass_Private.h"

bool mglRendererObjCSyncPipeline(GLMContext context, int deferred_buffer_map)
{
    MGLRenderer *renderer = mglRendererForContext(context);
    if (!renderer) {
        return false;
    }
    return [renderer syncPipelineStateWithDeferredBufferMap:deferred_buffer_map != 0];
}

bool mglRendererObjCProcessGLState(GLMContext context, bool draw_command)
{
    MGLRenderer *renderer = mglRendererForContext(context);
    if (!renderer) {
        return false;
    }
    return [renderer processGLState:draw_command];
}
