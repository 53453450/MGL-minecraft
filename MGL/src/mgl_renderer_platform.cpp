/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_platform.h"

extern "C" void mglRendererObjCSwapBuffers(GLMContext context);
extern "C" void mglRendererObjCClearBuffer(
    GLMContext context, unsigned int type, unsigned int mask);

extern "C" void mglRenderSwapBuffers(GLMContext context)
{
    if (!context) {
        return;
    }
    mglRendererObjCSwapBuffers(context);
}

extern "C" void mglRenderClearBuffer(
    GLMContext context, uint32_t type, uint32_t mask)
{
    if (!context) {
        return;
    }
    mglRendererObjCClearBuffer(context, type, mask);
}
