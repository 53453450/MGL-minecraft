/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_binding.h"

extern "C" bool mglRendererObjCSyncResourceBindings(
    GLMContext context, const MGLResourceSyncWork *already_done);

extern "C" int mglRenderSyncResourceBindings(
    GLMContext context, const MGLResourceSyncWork *already_done)
{
    if (!context) {
        return 0;
    }
    return mglRendererObjCSyncResourceBindings(context, already_done) ? 1 : 0;
}
