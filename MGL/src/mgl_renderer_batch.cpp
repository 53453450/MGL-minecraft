/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_batch.h"

extern "C" void mglRendererObjCFlushDrawBuffer(GLMContext context);

extern "C" void mglRenderFlushDrawBuffer(GLMContext context)
{
    if (!context) {
        return;
    }
    mglRendererObjCFlushDrawBuffer(context);
}
