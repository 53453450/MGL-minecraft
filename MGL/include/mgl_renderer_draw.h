/*
 * SPDX-License-Identifier: LGPL-3.0-only
 *
 * Draw dispatch orchestration (C++ home).
 */

#ifndef MGL_RENDERER_DRAW_H
#define MGL_RENDERER_DRAW_H

#include <stddef.h>
#include <stdint.h>

#include "glm_context.h"

#ifdef __cplusplus
extern "C" {
#endif

void mglRenderDrawArrays(GLMContext context, uint32_t mode, int32_t first,
                         int32_t count);
void mglRenderDrawElements(GLMContext context, uint32_t mode, int32_t count,
                           uint32_t type, const void *indices);
void mglRenderDrawRangeElements(GLMContext context, uint32_t mode,
                                uint32_t start, uint32_t end, int32_t count,
                                uint32_t type, const void *indices);
void mglRenderDrawArraysInstanced(GLMContext context, uint32_t mode,
                                  int32_t first, int32_t count,
                                  int32_t instance_count);
void mglRenderDrawElementsInstanced(GLMContext context, uint32_t mode,
                                    int32_t count, uint32_t type,
                                    const void *indices,
                                    int32_t instance_count);
void mglRenderDrawElementsBaseVertex(GLMContext context, uint32_t mode,
                                     int32_t count, uint32_t type,
                                     const void *indices, int32_t base_vertex);
void mglRenderDrawRangeElementsBaseVertex(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex);
void mglRenderDrawElementsInstancedBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex);
void mglRenderDrawArraysIndirect(GLMContext context, uint32_t mode,
                                 const void *indirect);
void mglRenderDrawElementsIndirect(GLMContext context, uint32_t mode,
                                   uint32_t type, const void *indirect);
void mglRenderDrawArraysInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count, uint32_t base_instance);
void mglRenderDrawElementsInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, uint32_t base_instance);
void mglRenderDrawElementsInstancedBaseVertexBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex,
    uint32_t base_instance);

void mglRenderMultiDrawArrays(GLMContext context, uint32_t mode,
                              const int32_t *firsts, const int32_t *counts,
                              int32_t draw_count);
void mglRenderMultiDrawElements(GLMContext context, uint32_t mode,
                                const int32_t *counts, uint32_t type,
                                const void *const *indices, int32_t draw_count);
void mglRenderMultiDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, const int32_t *counts, uint32_t type,
    const void *const *indices, int32_t draw_count,
    const int32_t *base_vertices);
void mglRenderMultiDrawArraysIndirect(GLMContext context, uint32_t mode,
                                      const void *indirect, int32_t draw_count,
                                      int32_t stride);
void mglRenderMultiDrawElementsIndirect(GLMContext context, uint32_t mode,
                                        uint32_t type, const void *indirect,
                                        int32_t draw_count, int32_t stride);

#ifdef __cplusplus
}
#endif

#endif /* MGL_RENDERER_DRAW_H */
