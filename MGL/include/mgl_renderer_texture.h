/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Non-drawable texture/compute/blit orchestration (C++ home).
 */

#ifndef MGL_RENDERER_TEXTURE_H
#define MGL_RENDERER_TEXTURE_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

void mglRenderBindTexture(GLMContext context, Texture *texture);
void mglRenderGenerateMipmaps(GLMContext context, Texture *texture);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_TEXTURE_H */
