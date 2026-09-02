/*
 * SPDX-License-Identifier: LGPL-3.0-only
 */

#include "mgl_renderer_draw.h"

#include "mgl_types_state.h"

extern "C" void mglRendererDrawArrays(GLMContext context, uint32_t mode,
                                      int32_t first, int32_t count);
extern "C" void mglRendererDrawElements(GLMContext context, uint32_t mode,
                                        int32_t count, uint32_t type,
                                        const void *indices);
extern "C" void mglRendererDrawRangeElements(GLMContext context, uint32_t mode,
                                             uint32_t start, uint32_t end,
                                             int32_t count, uint32_t type,
                                             const void *indices);
extern "C" void mglRendererDrawArraysInstanced(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count);
extern "C" void mglRendererDrawElementsInstanced(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count);
extern "C" void mglRendererDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex);
extern "C" void mglRendererDrawRangeElementsBaseVertex(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex);
extern "C" void mglRendererDrawElementsInstancedBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex);
extern "C" void mglRendererDrawArraysIndirect(GLMContext context, uint32_t mode,
                                              const void *indirect);
extern "C" void mglRendererDrawElementsIndirect(GLMContext context,
                                                uint32_t mode, uint32_t type,
                                                const void *indirect);
extern "C" void mglRendererDrawArraysInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count, uint32_t base_instance);
extern "C" void mglRendererDrawElementsInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, uint32_t base_instance);
extern "C" void mglRendererDrawElementsInstancedBaseVertexBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex,
    uint32_t base_instance);

extern "C" bool mglRendererObjCDrawArrays(GLMContext context, GLenum mode,
                                          GLint first, GLsizei count);
extern "C" bool mglRendererObjCDrawElements(GLMContext context, GLenum mode,
                                            GLsizei count, GLenum type,
                                            const void *indices);

extern "C" void mglRenderDrawArrays(GLMContext context, uint32_t mode,
                                    int32_t first, int32_t count)
{
    if (!context) {
        return;
    }
    mglRendererDrawArrays(context, mode, first, count);
}

extern "C" void mglRenderDrawElements(GLMContext context, uint32_t mode,
                                      int32_t count, uint32_t type,
                                      const void *indices)
{
    if (!context) {
        return;
    }
    mglRendererDrawElements(context, mode, count, type, indices);
}

extern "C" void mglRenderDrawRangeElements(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices)
{
    if (!context) {
        return;
    }
    mglRendererDrawRangeElements(context, mode, start, end, count, type,
                                 indices);
}

extern "C" void mglRenderDrawArraysInstanced(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count)
{
    if (!context) {
        return;
    }
    mglRendererDrawArraysInstanced(context, mode, first, count, instance_count);
}

extern "C" void mglRenderDrawElementsInstanced(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count)
{
    if (!context) {
        return;
    }
    mglRendererDrawElementsInstanced(context, mode, count, type, indices,
                                     instance_count);
}

extern "C" void mglRenderDrawElementsBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t base_vertex)
{
    if (!context) {
        return;
    }
    mglRendererDrawElementsBaseVertex(context, mode, count, type, indices,
                                      base_vertex);
}

extern "C" void mglRenderDrawRangeElementsBaseVertex(
    GLMContext context, uint32_t mode, uint32_t start, uint32_t end,
    int32_t count, uint32_t type, const void *indices, int32_t base_vertex)
{
    if (!context) {
        return;
    }
    mglRendererDrawRangeElementsBaseVertex(context, mode, start, end, count,
                                           type, indices, base_vertex);
}

extern "C" void mglRenderDrawElementsInstancedBaseVertex(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex)
{
    if (!context) {
        return;
    }
    mglRendererDrawElementsInstancedBaseVertex(
        context, mode, count, type, indices, instance_count, base_vertex);
}

extern "C" void mglRenderDrawArraysIndirect(GLMContext context, uint32_t mode,
                                            const void *indirect)
{
    if (!context) {
        return;
    }
    mglRendererDrawArraysIndirect(context, mode, indirect);
}

extern "C" void mglRenderDrawElementsIndirect(GLMContext context,
                                              uint32_t mode, uint32_t type,
                                              const void *indirect)
{
    if (!context) {
        return;
    }
    mglRendererDrawElementsIndirect(context, mode, type, indirect);
}

extern "C" void mglRenderDrawArraysInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t first, int32_t count,
    int32_t instance_count, uint32_t base_instance)
{
    if (!context) {
        return;
    }
    mglRendererDrawArraysInstancedBaseInstance(context, mode, first, count,
                                               instance_count, base_instance);
}

extern "C" void mglRenderDrawElementsInstancedBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, uint32_t base_instance)
{
    if (!context) {
        return;
    }
    mglRendererDrawElementsInstancedBaseInstance(
        context, mode, count, type, indices, instance_count, base_instance);
}

extern "C" void mglRenderDrawElementsInstancedBaseVertexBaseInstance(
    GLMContext context, uint32_t mode, int32_t count, uint32_t type,
    const void *indices, int32_t instance_count, int32_t base_vertex,
    uint32_t base_instance)
{
    if (!context) {
        return;
    }
    mglRendererDrawElementsInstancedBaseVertexBaseInstance(
        context, mode, count, type, indices, instance_count, base_vertex,
        base_instance);
}

extern "C" bool mglRendererObjCDrawArrays(GLMContext context, GLenum mode,
                                          GLint first, GLsizei count)
{
    if (!context) {
        return false;
    }
    mglRendererDrawArrays(context, (uint32_t)mode, (int32_t)first,
                          (int32_t)count);
    return context->active_state &&
           context->active_state->error == GL_NO_ERROR;
}

extern "C" bool mglRendererObjCDrawElements(GLMContext context, GLenum mode,
                                            GLsizei count, GLenum type,
                                            const void *indices)
{
    if (!context) {
        return false;
    }
    mglRendererDrawElements(context, (uint32_t)mode, (int32_t)count,
                            (uint32_t)type, indices);
    return context->active_state &&
           context->active_state->error == GL_NO_ERROR;
}
