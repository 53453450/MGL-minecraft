/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Platform-shell orchestration (swap/clear). C++ home for drawable lifecycle.
 */

#ifndef MGL_RENDERER_PLATFORM_H
#define MGL_RENDERER_PLATFORM_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

void mglRenderSwapBuffers(GLMContext context);
void mglRenderClearBuffer(GLMContext context, uint32_t type, uint32_t mask);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_PLATFORM_H */
