/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_pipeline.h"

extern "C" bool mglRendererObjCSyncPipeline(GLMContext context,
                                            int deferred_buffer_map);

extern "C" int mglRenderSyncPipeline(GLMContext context, int deferred_buffer_map)
{
    if (!context) {
        return 0;
    }
    return mglRendererObjCSyncPipeline(context, deferred_buffer_map) ? 1 : 0;
}
