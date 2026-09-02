/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Compute dispatch orchestration (C++ home).
 */

#ifndef MGL_RENDERER_COMPUTE_H
#define MGL_RENDERER_COMPUTE_H

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

void mglRenderDispatchCompute(
    GLMContext context, uint32_t groups_x, uint32_t groups_y, uint32_t groups_z);
void mglRenderDispatchComputeIndirect(GLMContext context, intptr_t indirect);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_COMPUTE_H */
