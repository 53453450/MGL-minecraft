/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_texture.h"

extern "C" void mglRendererObjCBindTexture(GLMContext context,
                                           Texture *texture);
extern "C" void mglRendererObjCGenerateMipmaps(GLMContext context,
                                               Texture *texture);

extern "C" void mglRenderBindTexture(GLMContext context, Texture *texture)
{
    if (!context) {
        return;
    }
    mglRendererObjCBindTexture(context, texture);
}

extern "C" void mglRenderGenerateMipmaps(GLMContext context, Texture *texture)
{
    if (!context) {
        return;
    }
    mglRendererObjCGenerateMipmaps(context, texture);
}
