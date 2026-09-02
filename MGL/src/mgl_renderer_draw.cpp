/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_draw.h"

extern "C" bool mglRendererObjCDrawArrays(GLMContext context, GLenum mode,
                                          GLint first, GLsizei count);
extern "C" bool mglRendererObjCDrawElements(GLMContext context, GLenum mode,
                                            GLsizei count, GLenum type,
                                            const void *indices);

extern "C" int mglRenderDrawArrays(GLMContext context, GLenum mode, GLint first,
                                   GLsizei count)
{
    if (!context) {
        return 0;
    }
    return mglRendererObjCDrawArrays(context, mode, first, count) ? 1 : 0;
}

extern "C" int mglRenderDrawElements(GLMContext context, GLenum mode,
                                     GLsizei count, GLenum type,
                                     const void *indices)
{
    if (!context) {
        return 0;
    }
    return mglRendererObjCDrawElements(context, mode, count, type, indices)
               ? 1
               : 0;
}
