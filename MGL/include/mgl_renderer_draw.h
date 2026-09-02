/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Draw dispatch orchestration (C++ home).
 */

#ifndef MGL_RENDERER_DRAW_H
#define MGL_RENDERER_DRAW_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

int mglRenderDrawArrays(GLMContext context, GLenum mode, GLint first,
                        GLsizei count);
int mglRenderDrawElements(GLMContext context, GLenum mode, GLsizei count,
                          GLenum type, const void *indices);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_DRAW_H */
