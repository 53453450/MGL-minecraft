/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_draw.h"

extern "C" void mglRendererObjCDrawArrays(GLMContext, uint32_t, int32_t,
                                          int32_t);
extern "C" void mglRendererObjCDrawElements(GLMContext, uint32_t, int32_t,
                                          uint32_t, const void *);
extern "C" void mglRendererObjCDrawRangeElements(GLMContext, uint32_t, uint32_t,
                                                 uint32_t, int32_t, uint32_t,
                                                 const void *);
extern "C" void mglRendererObjCDrawArraysInstanced(GLMContext, uint32_t,
                                                   int32_t, int32_t, int32_t);
extern "C" void mglRendererObjCDrawElementsInstanced(GLMContext, uint32_t,
                                                     int32_t, uint32_t,
                                                     const void *, int32_t);
extern "C" void mglRendererObjCDrawElementsBaseVertex(GLMContext, uint32_t,
                                                      int32_t, uint32_t,
                                                      const void *, int32_t);
extern "C" void mglRendererObjCDrawRangeElementsBaseVertex(
    GLMContext, uint32_t, uint32_t, uint32_t, int32_t, uint32_t, const void *,
    int32_t);
extern "C" void mglRendererObjCDrawElementsInstancedBaseVertex(
    GLMContext, uint32_t, int32_t, uint32_t, const void *, int32_t, int32_t);
extern "C" void mglRendererObjCDrawArraysIndirect(GLMContext, uint32_t,
                                                  const void *);
extern "C" void mglRendererObjCDrawElementsIndirect(GLMContext, uint32_t,
                                                    uint32_t, const void *);
extern "C" void mglRendererObjCDrawArraysInstancedBaseInstance(
    GLMContext, uint32_t, int32_t, int32_t, int32_t, uint32_t);
extern "C" void mglRendererObjCDrawElementsInstancedBaseInstance(
    GLMContext, uint32_t, int32_t, uint32_t, const void *, int32_t, uint32_t);
extern "C" void mglRendererObjCDrawElementsInstancedBaseVertexBaseInstance(
    GLMContext, uint32_t, int32_t, uint32_t, const void *, int32_t, int32_t,
    uint32_t);
extern "C" void mglRendererObjCMultiDrawArrays(GLMContext, uint32_t,
                                               const int32_t *, const int32_t *,
                                               int32_t);
extern "C" void mglRendererObjCMultiDrawElements(GLMContext, uint32_t,
                                                 const int32_t *, uint32_t,
                                                 const void *const *, int32_t);
extern "C" void mglRendererObjCMultiDrawElementsBaseVertex(
    GLMContext, uint32_t, const int32_t *, uint32_t, const void *const *,
    int32_t, const int32_t *);
extern "C" void mglRendererObjCMultiDrawArraysIndirect(GLMContext, uint32_t,
                                                       const void *, int32_t,
                                                       int32_t);
extern "C" void mglRendererObjCMultiDrawElementsIndirect(GLMContext, uint32_t,
                                                         uint32_t,
                                                         const void *, int32_t,
                                                         int32_t);

extern "C" void mglRendererDrawArrays(GLMContext context, uint32_t mode,
                                      int32_t first, int32_t count)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawArrays(context, mode, first, count);
}

extern "C" void mglRendererDrawElements(GLMContext context, uint32_t mode,
                                        int32_t count, uint32_t type,
                                        const void *indices)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawElements(context, mode, count, type, indices);
}

extern "C" void mglRendererDrawRangeElements(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawRangeElements(context, mode, start, end, count, type,
                                     indices);
}

extern "C" void mglRendererDrawArraysInstanced(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawArraysInstanced(context, mode, first, count,
                                       instance_count);
}

extern "C" void mglRendererDrawElementsInstanced(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawElementsInstanced(context, mode, count, type, indices,
                                         instance_count);
}

extern "C" void mglRendererDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawElementsBaseVertex(context, mode, count, type, indices,
                                          base_vertex);
}

extern "C" void mglRendererDrawRangeElementsBaseVertex(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawRangeElementsBaseVertex(context, mode, start, end, count,
                                               type, indices, base_vertex);
}

extern "C" void mglRendererDrawElementsInstancedBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawElementsInstancedBaseVertex(
        context, mode, count, type, indices, instance_count, base_vertex);
}

extern "C" void mglRendererDrawArraysIndirect(GLMContext context, uint32_t mode,
                                              const void *indirect)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawArraysIndirect(context, mode, indirect);
}

extern "C" void mglRendererDrawElementsIndirect(GLMContext context,
                                                uint32_t mode, uint32_t type,
                                                const void *indirect)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawElementsIndirect(context, mode, type, indirect);
}

extern "C" void mglRendererDrawArraysInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count, uint32_t base_instance)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawArraysInstancedBaseInstance(
        context, mode, first, count, instance_count, base_instance);
}

extern "C" void mglRendererDrawElementsInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, uint32_t base_instance)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawElementsInstancedBaseInstance(
        context, mode, count, type, indices, instance_count, base_instance);
}

extern "C" void mglRendererDrawElementsInstancedBaseVertexBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex,
    uint32_t base_instance)
{
    if (!context) {
        return;
    }
    mglRendererObjCDrawElementsInstancedBaseVertexBaseInstance(
        context, mode, count, type, indices, instance_count, base_vertex,
        base_instance);
}

extern "C" void mglRendererMultiDrawArrays(
    GLMContext context, uint32_t mode, const int32_t *firsts,
    const int32_t *counts, int32_t draw_count)
{
    if (!context) {
        return;
    }
    mglRendererObjCMultiDrawArrays(context, mode, firsts, counts, draw_count);
}

extern "C" void mglRendererMultiDrawElements(
    GLMContext context, uint32_t mode, const int32_t *counts, uint32_t type,
    const void *const *indices, int32_t draw_count)
{
    if (!context) {
        return;
    }
    mglRendererObjCMultiDrawElements(context, mode, counts, type, indices,
                                     draw_count);
}

extern "C" void mglRendererMultiDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, const int32_t *counts, uint32_t type,
    const void *const *indices, int32_t draw_count,
    const int32_t *base_vertices)
{
    if (!context) {
        return;
    }
    mglRendererObjCMultiDrawElementsBaseVertex(context, mode, counts, type,
                                               indices, draw_count,
                                               base_vertices);
}

extern "C" void mglRendererMultiDrawArraysIndirect(
    GLMContext context, uint32_t mode, const void *indirect, int32_t draw_count,
    int32_t stride)
{
    if (!context) {
        return;
    }
    mglRendererObjCMultiDrawArraysIndirect(context, mode, indirect, draw_count,
                                           stride);
}

extern "C" void mglRendererMultiDrawElementsIndirect(
    GLMContext context, uint32_t mode, uint32_t type, const void *indirect,
    int32_t draw_count, int32_t stride)
{
    if (!context) {
        return;
    }
    mglRendererObjCMultiDrawElementsIndirect(context, mode, type, indirect,
                                             draw_count, stride);
}

extern "C" void mglRenderDrawArrays(GLMContext context, uint32_t mode,
                                    int32_t first, int32_t count)
{
    mglRendererDrawArrays(context, mode, first, count);
}

extern "C" void mglRenderDrawElements(GLMContext context, uint32_t mode,
                                      int32_t count, uint32_t type,
                                      const void *indices)
{
    mglRendererDrawElements(context, mode, count, type, indices);
}

extern "C" void mglRenderDrawRangeElements(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices)
{
    mglRendererDrawRangeElements(context, mode, start, end, count, type,
                                 indices);
}

extern "C" void mglRenderDrawArraysInstanced(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count)
{
    mglRendererDrawArraysInstanced(context, mode, first, count, instance_count);
}

extern "C" void mglRenderDrawElementsInstanced(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count)
{
    mglRendererDrawElementsInstanced(context, mode, count, type, indices,
                                     instance_count);
}

extern "C" void mglRenderDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex)
{
    mglRendererDrawElementsBaseVertex(context, mode, count, type, indices,
                                      base_vertex);
}

extern "C" void mglRenderDrawRangeElementsBaseVertex(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex)
{
    mglRendererDrawRangeElementsBaseVertex(context, mode, start, end, count,
                                           type, indices, base_vertex);
}

extern "C" void mglRenderDrawElementsInstancedBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex)
{
    mglRendererDrawElementsInstancedBaseVertex(
        context, mode, count, type, indices, instance_count, base_vertex);
}

extern "C" void mglRenderDrawArraysIndirect(GLMContext context, uint32_t mode,
                                            const void *indirect)
{
    mglRendererDrawArraysIndirect(context, mode, indirect);
}

extern "C" void mglRenderDrawElementsIndirect(GLMContext context,
                                              uint32_t mode, uint32_t type,
                                              const void *indirect)
{
    mglRendererDrawElementsIndirect(context, mode, type, indirect);
}

extern "C" void mglRenderDrawArraysInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count, uint32_t base_instance)
{
    mglRendererDrawArraysInstancedBaseInstance(context, mode, first, count,
                                               instance_count, base_instance);
}

extern "C" void mglRenderDrawElementsInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, uint32_t base_instance)
{
    mglRendererDrawElementsInstancedBaseInstance(
        context, mode, count, type, indices, instance_count, base_instance);
}

extern "C" void mglRenderDrawElementsInstancedBaseVertexBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex,
    uint32_t base_instance)
{
    mglRendererDrawElementsInstancedBaseVertexBaseInstance(
        context, mode, count, type, indices, instance_count, base_vertex,
        base_instance);
}

extern "C" void mglRenderMultiDrawArrays(
    GLMContext context, uint32_t mode, const int32_t *firsts,
    const int32_t *counts, int32_t draw_count)
{
    mglRendererMultiDrawArrays(context, mode, firsts, counts, draw_count);
}

extern "C" void mglRenderMultiDrawElements(
    GLMContext context, uint32_t mode, const int32_t *counts, uint32_t type,
    const void *const *indices, int32_t draw_count)
{
    mglRendererMultiDrawElements(context, mode, counts, type, indices,
                                 draw_count);
}

extern "C" void mglRenderMultiDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, const int32_t *counts, uint32_t type,
    const void *const *indices, int32_t draw_count,
    const int32_t *base_vertices)
{
    mglRendererMultiDrawElementsBaseVertex(context, mode, counts, type, indices,
                                           draw_count, base_vertices);
}

extern "C" void mglRenderMultiDrawArraysIndirect(
    GLMContext context, uint32_t mode, const void *indirect, int32_t draw_count,
    int32_t stride)
{
    mglRendererMultiDrawArraysIndirect(context, mode, indirect, draw_count,
                                       stride);
}

extern "C" void mglRenderMultiDrawElementsIndirect(
    GLMContext context, uint32_t mode, uint32_t type, const void *indirect,
    int32_t draw_count, int32_t stride)
{
    mglRendererMultiDrawElementsIndirect(context, mode, type, indirect,
                                         draw_count, stride);
}
